"""GPU benchmark execution; all CUDA entry points are reached only by explicit CLI runs."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import importlib
import math
from pathlib import Path
import shutil
import time
import traceback
from typing import Any

from cambacica.benchmark.artifacts import (
    git_provenance,
    implementation_sha256,
    new_run_id,
    publish_artifact_bundle,
    require_storage_layout,
    runtime_identity,
    write_artifact_bundle,
)
from cambacica.benchmark.config import BenchmarkConfig
from cambacica.benchmark.metrics import summarize_step_times, tokens_per_second
from cambacica.benchmark.safety import (
    BenchmarkLock,
    GpuSafetyError,
    GpuSafetyMonitor,
    apply_allocator_limit,
    numeric_stability_check,
    preflight_gpu,
)
from cambacica.model.config import get_architecture

ADAMW_SETTINGS = {
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "betas": [0.9, 0.95],
    "weight_decay": 0.1,
    "epsilon": 1e-8,
}


def _classify_failure(error: BaseException) -> str:
    name = type(error).__name__.lower()
    message = str(error).lower()
    if "non-finite" in message or "nonfinite" in message or "numerical" in message:
        return "numerical_failure"
    if "outofmemory" in name or ("out of memory" in message and "cuda" in message):
        return "cuda_oom"
    if (
        "no available kernel" in message
        or "no viable backend" in message
        or (
            "backend" in message
            and any(
                term in message
                for term in ("not available", "unsupported", "not supported")
            )
        )
    ):
        return "incompatible_attention_backend"
    if isinstance(error, GpuSafetyError):
        return "gpu_safety_policy"
    return "benchmark_error"


def _torch_and_nvml() -> tuple[Any, Any]:
    try:
        torch_module = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is optional; install the C3 extra with `pip install -e '.[c3]'`"
        ) from exc
    try:
        nvml = importlib.import_module("pynvml")
    except ImportError as exc:
        raise RuntimeError(
            "NVML process monitoring is required; install the C3 extra with `pip install -e '.[c3]'`"
        ) from exc
    return torch_module, nvml


def _attention_profiler_result(
    torch_module: Any,
    model: Any,
    input_ids: Any,
    precision: str,
) -> dict[str, object]:
    observed: set[str] = set()
    error_message = None
    try:
        activities = [torch_module.profiler.ProfilerActivity.CPU]
        activities.append(torch_module.profiler.ProfilerActivity.CUDA)
        with torch_module.profiler.profile(activities=activities) as profiler:
            with torch_module.inference_mode(), _autocast(torch_module, precision):
                model(input_ids)
            torch_module.cuda.synchronize(0)
        for item in profiler.key_averages():
            key = item.key.lower()
            if "flash_attention" in key:
                observed.add("flash")
            elif "efficient_attention" in key:
                observed.add("efficient")
            elif "cudnn_attention" in key:
                observed.add("cudnn")
            elif "attention_math" in key or "scaled_dot_product_attention_math" in key:
                observed.add("math")
    except Exception as exc:  # profiler availability differs across supported builds
        error_message = f"{type(exc).__name__}: {exc}"
    if not observed:
        actual = "unconfirmed"
    elif len(observed) == 1:
        actual = next(iter(observed))
    else:
        actual = ",".join(sorted(observed))
    return {"requested": model.attention_backend, "observed": actual, "profiler_error": error_message}


def _autocast(torch_module: Any, precision: str):
    if precision == "bf16":
        return torch_module.autocast("cuda", dtype=torch_module.bfloat16)
    if precision == "fp16":
        return torch_module.autocast("cuda", dtype=torch_module.float16)
    return nullcontext()


def _optimizer_scaler(torch_module: Any, precision: str):
    return torch_module.amp.GradScaler("cuda", enabled=precision == "fp16")


def _synthetic_tokens(torch_module: Any, config: BenchmarkConfig):
    generator = torch_module.Generator(device="cpu")
    generator.manual_seed(config.seed)
    tokens = torch_module.randint(
        low=0,
        high=24_576,
        size=(config.microbatch_size, config.context_length + 1),
        dtype=torch_module.long,
        generator=generator,
        device="cpu",
    )
    digest = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
    return tokens, digest


def _finite_step(
    torch_module: Any,
    model: Any,
    optimizer: Any,
    scaler: Any,
    input_ids: Any,
    labels: Any,
    precision: str,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    with _autocast(torch_module, precision):
        logits, loss = model(input_ids, labels=labels)
    if tuple(logits.shape) != (
        input_ids.shape[0],
        input_ids.shape[1],
        model.spec.vocab_size,
    ):
        raise RuntimeError(f"unexpected logits shape: {tuple(logits.shape)}")
    if precision == "fp16":
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()

    loss_finite = torch_module.isfinite(loss.detach()).all()
    gradient_checks = []
    for parameter in model.parameters():
        if parameter.grad is not None:
            gradient_checks.append(torch_module.isfinite(parameter.grad).all())
    if not gradient_checks:
        raise GpuSafetyError("no gradients were produced by the causal loss")
    gradients_finite = torch_module.stack(gradient_checks).all()
    values = torch_module.stack(
        (loss.detach().float(), loss_finite.float(), gradients_finite.float())
    ).cpu().tolist()
    loss_value = float(values[0])
    numeric_stability_check(bool(values[1]), bool(values[2]))
    if precision == "fp16":
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return loss_value


def _run_steps(
    torch_module: Any,
    *,
    model: Any,
    optimizer: Any,
    scaler: Any,
    input_ids: Any,
    labels: Any,
    precision: str,
    steps: int,
    monitor: GpuSafetyMonitor,
    diagnostics: dict[str, object],
    timed: bool,
) -> tuple[list[float], list[Any], float]:
    losses: list[float] = []
    events: list[Any] = []
    if timed:
        torch_module.cuda.synchronize(0)
        wall_start = time.perf_counter()
    else:
        wall_start = 0.0
    for _ in range(steps):
        start_event = end_event = None
        if timed:
            start_event = torch_module.cuda.Event(enable_timing=True)
            end_event = torch_module.cuda.Event(enable_timing=True)
            start_event.record()
        loss_value = _finite_step(
            torch_module,
            model,
            optimizer,
            scaler,
            input_ids,
            labels,
            precision,
        )
        if timed:
            end_event.record()
            events.append((start_event, end_event))
        losses.append(loss_value)
        sample = monitor.sample()
        diagnostics["last_gpu_memory_sample"] = sample
        diagnostics["peak_process_memory_mib"] = monitor.peak_process_memory_mib
    if not timed:
        return losses, events, 0.0
    torch_module.cuda.synchronize(0)
    wall_seconds = time.perf_counter() - wall_start
    return losses, events, wall_seconds


def _execute_benchmark(
    config: BenchmarkConfig,
    *,
    torch_module: Any,
    nvml: Any,
    diagnostics: dict[str, object],
) -> dict[str, object]:
    snapshot, handle = preflight_gpu(torch_module, nvml, config)
    diagnostics["startup_gpu"] = snapshot.to_dict()
    diagnostics["visible_cuda_devices"] = int(torch_module.cuda.device_count())
    if config.precision == "bf16" and not torch_module.cuda.is_bf16_supported():
        raise GpuSafetyError("BF16 is not supported by the visible physical GPU 0")
    torch_module.backends.cudnn.benchmark = False
    if hasattr(torch_module.backends.cuda.matmul, "allow_tf32"):
        torch_module.backends.cuda.matmul.allow_tf32 = False
    allocator_fraction = apply_allocator_limit(
        torch_module, config, snapshot.total_memory_mib
    )
    diagnostics["allocator_fraction"] = allocator_fraction
    diagnostics["allocator_backend"] = "native"
    # Force context creation only after identity and the allocator ceiling pass.
    torch_module.empty(1, device="cuda:0")
    torch_module.cuda.synchronize(0)
    monitor = GpuSafetyMonitor(nvml, handle, config)
    diagnostics["last_gpu_memory_sample"] = monitor.sample()

    from cambacica.model.transformer import CambacicaCausalLM, count_instantiated_parameters

    spec = get_architecture(config.architecture)
    activation_dtype = {
        "bf16": torch_module.bfloat16,
        "fp16": torch_module.float16,
        "fp32": None,
    }[config.precision]
    model = CambacicaCausalLM(
        spec,
        tie_weights=config.tie_weights,
        seed=config.seed,
        attention_backend=config.attention_backend,
        activation_dtype=activation_dtype,
        device="cpu",
    ).to("cuda:0")
    actual_parameters = count_instantiated_parameters(model)
    expected_parameters = spec.parameter_count(tie_weights=config.tie_weights)
    if actual_parameters != expected_parameters:
        raise AssertionError(
            f"instantiated parameter count {actual_parameters} differs from "
            f"analytical count {expected_parameters}"
        )
    if config.tie_weights and model.lm_head.weight is not model.token_embedding.weight:
        raise AssertionError("tied output projection does not share the embedding weight")
    model.train()
    optimizer = torch_module.optim.AdamW(
        model.parameters(),
        lr=ADAMW_SETTINGS["learning_rate"],
        betas=tuple(ADAMW_SETTINGS["betas"]),
        weight_decay=ADAMW_SETTINGS["weight_decay"],
        eps=ADAMW_SETTINGS["epsilon"],
    )
    scaler = _optimizer_scaler(torch_module, config.precision)
    token_ids, input_hash = _synthetic_tokens(torch_module, config)
    input_ids = token_ids[:, :-1].to("cuda:0")
    labels = token_ids[:, 1:].to("cuda:0")
    del token_ids
    diagnostics["synthetic_token_ids_sha256"] = input_hash
    diagnostics["parameter_count"] = actual_parameters
    diagnostics["optimizer"] = ADAMW_SETTINGS
    diagnostics["initial_process_memory"] = monitor.sample()

    attention_backend = _attention_profiler_result(
        torch_module, model, input_ids, config.precision
    )
    diagnostics["attention_backend"] = attention_backend
    if (
        config.attention_backend != "auto"
        and attention_backend["observed"] not in (config.attention_backend,)
    ):
        raise GpuSafetyError(
            f"requested {config.attention_backend} attention backend was not confirmed "
            f"(observed {attention_backend['observed']})"
        )
    del attention_backend
    torch_module.cuda.empty_cache()
    torch_module.cuda.reset_peak_memory_stats(0)
    monitor.sample()

    warmup_losses, _, _ = _run_steps(
        torch_module,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        input_ids=input_ids,
        labels=labels,
        precision=config.precision,
        steps=config.warmup_steps,
        monitor=monitor,
        diagnostics=diagnostics,
        timed=False,
    )
    timed_losses, events, wall_seconds = _run_steps(
        torch_module,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        input_ids=input_ids,
        labels=labels,
        precision=config.precision,
        steps=config.timed_steps,
        monitor=monitor,
        diagnostics=diagnostics,
        timed=True,
    )
    event_times_ms = [float(start.elapsed_time(end)) for start, end in events]
    if any(not math.isfinite(value) for value in event_times_ms):
        raise GpuSafetyError("non-finite CUDA event timing detected")
    allocated_mib = torch_module.cuda.max_memory_allocated(0) / (1024 * 1024)
    reserved_mib = torch_module.cuda.max_memory_reserved(0) / (1024 * 1024)
    final_memory = monitor.sample()
    diagnostics["last_gpu_memory_sample"] = final_memory
    diagnostics["peak_process_memory_mib"] = monitor.peak_process_memory_mib
    summary = summarize_step_times(event_times_ms)
    training_tokens = config.tokens_per_step * config.timed_steps
    return {
        "architecture": spec.to_dict(),
        "parameter_count": actual_parameters,
        "weight_tying": config.tie_weights,
        "context_length": config.context_length,
        "microbatch_size": config.microbatch_size,
        "tokens_per_step": config.tokens_per_step,
        "warmup_steps": config.warmup_steps,
        "timed_steps": config.timed_steps,
        "optimizer": ADAMW_SETTINGS,
        "precision": config.precision,
        "activation_checkpointing": "none",
        "attention_backend": diagnostics["attention_backend"],
        "step_time_gpu_event_ms": summary,
        "wall_time_seconds": wall_seconds,
        "wall_mean_step_time_ms": wall_seconds * 1000 / config.timed_steps,
        "training_tokens_per_second": tokens_per_second(
            config.tokens_per_step, config.timed_steps, wall_seconds
        ),
        "training_tokens_timed": training_tokens,
        "peak_cuda_allocated_mib": allocated_mib,
        "peak_cuda_reserved_mib": reserved_mib,
        "peak_process_memory_mib_nvml": monitor.peak_process_memory_mib,
        "gpu_memory_available_at_startup_mib": snapshot.free_memory_mib,
        "gpu_memory_used_at_startup_mib": snapshot.used_memory_mib,
        "gpu_uuid": snapshot.uuid,
        "gpu_physical_index": snapshot.physical_index,
        "gpu_name": snapshot.name,
        "gpu_pci_bus_id": snapshot.pci_bus_id,
        "gpu_mapping_method": snapshot.mapping_method,
        "gpu_driver_version": snapshot.driver_version,
        "gpu_processes_at_startup": list(snapshot.processes),
        "cuda_runtime_version": torch_module.version.cuda,
        "pytorch_version": torch_module.__version__,
        "loss": {
            "warmup_mean": sum(warmup_losses) / len(warmup_losses)
            if warmup_losses
            else None,
            "timed_mean": sum(timed_losses) / len(timed_losses),
            "timed_first": timed_losses[0],
            "timed_last": timed_losses[-1],
            "finite": True,
        },
        "gradients_finite": True,
        "numerical_status": "finite",
        "synthetic_token_ids_sha256": input_hash,
        "memory_policy": config.to_dict()["memory_policy"],
        "gpu_memory_last_sample": final_memory,
        "gpu_memory_peak_sampled_process_mib": monitor.peak_process_memory_mib,
    }


def _write_and_publish(
    *,
    stage: Path,
    output_root: Path,
    config: BenchmarkConfig,
    run_id: str,
    configuration: dict[str, object],
    result: dict[str, object],
    implementation_digest: str,
    publish: bool,
) -> tuple[dict[str, str], Path | None]:
    hashes = write_artifact_bundle(
        stage,
        run_id=run_id,
        configuration=configuration,
        result=result,
        implementation_digest=implementation_digest,
    )
    if not publish:
        return hashes, None
    destination = publish_artifact_bundle(
        stage,
        output_root=output_root,
        architecture=config.architecture,
        context_length=config.context_length,
        run_id=run_id,
    )
    return hashes, destination


def run_benchmark(config: BenchmarkConfig) -> dict[str, object]:
    """Run one benchmark and preserve a checksummed artifact on every failure."""

    config.validate()
    run_id = new_run_id(config.seed)
    config_dict = config.to_dict()
    stage = config.scratch_root / run_id
    stage.mkdir(parents=True, exist_ok=False)
    provenance = git_provenance()
    implementation_digest = implementation_sha256()
    diagnostics: dict[str, object] = {}
    result: dict[str, object] = {
        "schema_version": "c3-benchmark-result-v1",
        "run_id": run_id,
        "status": "failed",
        "failure_kind": None,
        "error": None,
        "architecture_id": config.architecture,
        "parameter_count": None,
        "performance": None,
        "diagnostics": diagnostics,
        "git": provenance,
        "implementation_sha256": implementation_digest,
        "runtime": {"python_version": __import__("platform").python_version()},
    }
    storage_valid = False
    lock: BenchmarkLock | None = None
    try:
        require_storage_layout(config.scratch_root, config.output_root)
        storage_valid = True
        try:
            lock = BenchmarkLock(config.lock_path).acquire()
        except Exception:
            raise
        torch_module, nvml = _torch_and_nvml()
        result["runtime"] = runtime_identity(torch_module)
        performance = _execute_benchmark(
            config,
            torch_module=torch_module,
            nvml=nvml,
            diagnostics=diagnostics,
        )
        result.update(
            {
                "status": "complete",
                "performance": performance,
                "parameter_count": performance["parameter_count"],
                "failure_kind": None,
                "error": None,
            }
        )
    except (Exception, KeyboardInterrupt) as exc:
        result["status"] = "failed"
        result["failure_kind"] = _classify_failure(exc)
        if isinstance(exc, GpuSafetyError):
            diagnostics.update(exc.diagnostics)
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
        }
    finally:
        if lock is not None:
            lock.release()

    try:
        hashes, destination = _write_and_publish(
            stage=stage,
            output_root=config.output_root,
            config=config,
            run_id=run_id,
            configuration=config_dict,
            result=result,
            implementation_digest=implementation_digest,
            publish=storage_valid,
        )
        result["artifact_checksums"] = hashes
        result["artifact_path"] = str(destination or stage)
        if destination is not None:
            shutil.rmtree(stage, ignore_errors=True)
    except Exception as exc:
        result["status"] = "failed" if result["status"] != "complete" else "complete_unpublished"
        result["publication_error"] = f"{type(exc).__name__}: {exc}"
        result["artifact_path"] = str(stage)
        if stage.exists():
            try:
                write_artifact_bundle(
                    stage,
                    run_id=run_id,
                    configuration=config_dict,
                    result=result,
                    implementation_digest=implementation_digest,
                )
            except Exception as artifact_exc:
                result["artifact_write_error"] = (
                    f"{type(artifact_exc).__name__}: {artifact_exc}"
                )
    if result["status"] not in {"complete"}:
        result["exit_code"] = 1
    else:
        result["exit_code"] = 0
    return result
