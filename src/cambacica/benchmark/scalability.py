"""C3.6 synthetic scalability matrix and deterministic restart machinery.

The module separates CPU-only planning and artifact validation from the explicit
GPU execution entry points. Importing it reads the approved JSON baseline but
does not import or initialize PyTorch or CUDA.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import io
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cambacica.benchmark import runner as c3_runner
from cambacica.benchmark.artifacts import (
    canonical_json,
    git_provenance,
    implementation_sha256,
    require_storage_layout,
    runtime_identity,
    sha256_file,
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
from cambacica.model.baseline import (
    DEFAULT_BASELINE_PATH,
    load_training_baseline,
    validate_training_baseline,
)
from cambacica.model.config import get_architecture

BASELINE = validate_training_baseline(load_training_baseline())
BASELINE_SHA256 = hashlib.sha256(DEFAULT_BASELINE_PATH.read_bytes()).hexdigest()
BASELINE_ID = str(BASELINE["baseline_id"])
BASELINE_APPROVED = BASELINE["scientist_approved"]
BASELINE_FROZEN = BASELINE["frozen_inputs"]
BASELINE_PROPOSAL = BASELINE["scalability_validation_proposal"]
OPTIMIZER_FIXTURE = BASELINE["measured_baseline_choices"]["optimizer_fixture"]
STREAM_SEED = int(BASELINE["measured_baseline_choices"]["seed"])
CONTEXT_LENGTH = 4_096
WARMUP_UPDATES = 5
MEASURED_UPDATES = 20
REPEATS = 1
STREAM_SAMPLE_COUNT = (WARMUP_UPDATES + MEASURED_UPDATES) * 4 * 4
SCRATCH_ROOT = Path("/tmp/cambacica-base-180m/c3.6-scalability")
OUTPUT_ROOT = Path("/mnt/data/cambacica-base-180m/c3-scalability/c3.6")
LOCK_PATH = Path("/tmp/cambacica-base-180m/c3-benchmark.lock")
GPU_MEMORY_BUDGET_MIB = 40_960
PROCESS_PEAK_TARGET_MIB = 36_864
TORCH_ALLOCATOR_LIMIT_MIB = 35_840
MINIMUM_FREE_MEMORY_MIB = 4_096
CELL_IDS = tuple(
    f"cell-m{microbatch}-a{accumulation}"
    for microbatch in (1, 2, 4)
    for accumulation in (1, 2, 4)
)
_SUITE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ScalabilityValidationError(ValueError):
    """A C3.6 configuration or artifact does not satisfy its fixed contract."""


class CheckpointIntegrityError(RuntimeError):
    """A checkpoint checksum, schema, or pinned identity failed validation."""


@dataclass(frozen=True)
class CellConfig:
    """One approved C3.6 matrix cell; training choices remain fixed to C3.4."""

    cell_id: str
    microbatch_size: int
    gradient_accumulation_steps: int
    context_length: int = CONTEXT_LENGTH
    architecture_id: str = "C"
    tied_weights: bool = True
    precision: str = "bf16"
    seed: int = STREAM_SEED
    warmup_updates: int = WARMUP_UPDATES
    measured_updates: int = MEASURED_UPDATES
    repeats: int = REPEATS
    attention_backend: str = "auto"
    activation_checkpointing: str = "none"
    optimizer_fixture: str = "C3.4 synthetic AdamW fixture"

    @property
    def tokens_per_update(self) -> int:
        return (
            self.microbatch_size
            * self.gradient_accumulation_steps
            * self.context_length
        )

    def validate(self) -> CellConfig:
        if (
            self.cell_id
            != f"cell-m{self.microbatch_size}-a{self.gradient_accumulation_steps}"
        ):
            raise ScalabilityValidationError(
                "cell_id does not match its batch settings"
            )
        if (self.microbatch_size, self.gradient_accumulation_steps) not in {
            (microbatch, accumulation)
            for microbatch in (1, 2, 4)
            for accumulation in (1, 2, 4)
        }:
            raise ScalabilityValidationError("cell is outside the approved 3x3 matrix")
        if self.context_length != CONTEXT_LENGTH:
            raise ScalabilityValidationError("C3.6 context length must be exactly 4096")
        if self.architecture_id != "C" or not self.tied_weights:
            raise ScalabilityValidationError("C3.6 requires tied Architecture C")
        if self.precision != "bf16":
            raise ScalabilityValidationError("C3.6 reference compute precision is BF16")
        if self.seed != STREAM_SEED:
            raise ScalabilityValidationError(
                "C3.6 synthetic stream seed is baseline-pinned"
            )
        if (self.warmup_updates, self.measured_updates, self.repeats) != (5, 20, 1):
            raise ScalabilityValidationError(
                "C3.6 requires 5 warm-up updates, 20 measured updates, and one repeat"
            )
        if self.attention_backend != "auto":
            raise ScalabilityValidationError("C3.6 attention backend must remain auto")
        if self.activation_checkpointing != "none":
            raise ScalabilityValidationError(
                "C3.6 activation checkpointing must be none"
            )
        if self.optimizer_fixture != "C3.4 synthetic AdamW fixture":
            raise ScalabilityValidationError(
                "C3.6 optimizer fixture identity is pinned"
            )
        expected_tokens = self.microbatch_size * self.gradient_accumulation_steps * 4096
        if self.tokens_per_update != expected_tokens:
            raise ScalabilityValidationError(
                "tokens_per_update does not match the cell"
            )
        return self

    def to_dict(self) -> dict[str, object]:
        self.validate()
        value = asdict(self)
        value.update(
            {
                "baseline_id": BASELINE_ID,
                "baseline_sha256": BASELINE_SHA256,
                "parameter_count": BASELINE_APPROVED["parameter_count"],
                "tokenizer_identity": BASELINE_FROZEN["tokenizer_identity"],
                "tokens_per_update": self.tokens_per_update,
                "optimizer_fixture_settings": OPTIMIZER_FIXTURE,
                "memory_policy": {
                    "physical_gpu_index": 0,
                    "gpu_uuid": BASELINE_APPROVED["target_hardware"]["gpu_uuid"],
                    "cuda_visible_devices": "0",
                    "hard_process_budget_mib": GPU_MEMORY_BUDGET_MIB,
                    "process_peak_target_mib": PROCESS_PEAK_TARGET_MIB,
                    "torch_allocator_limit_mib": TORCH_ALLOCATOR_LIMIT_MIB,
                    "minimum_free_memory_mib": MINIMUM_FREE_MEMORY_MIB,
                },
                "final_training_optimizer_selected": False,
            }
        )
        return value


def approved_cells() -> tuple[CellConfig, ...]:
    """Return the nine immutable cells in the order recorded in the baseline."""

    cells = tuple(
        CellConfig(
            cell_id=f"cell-m{microbatch}-a{accumulation}",
            microbatch_size=microbatch,
            gradient_accumulation_steps=accumulation,
        ).validate()
        for microbatch in (1, 2, 4)
        for accumulation in (1, 2, 4)
    )
    baseline_cells = BASELINE_PROPOSAL["cells"]
    observed = [
        {
            "microbatch_size": cell.microbatch_size,
            "gradient_accumulation_steps": cell.gradient_accumulation_steps,
            "tokens_per_update": cell.tokens_per_update,
        }
        for cell in cells
    ]
    if observed != baseline_cells:
        raise ScalabilityValidationError(
            "runtime matrix differs from the approved baseline"
        )
    return cells


def validate_suite_id(suite_id: str) -> str:
    if not _SUITE_ID_PATTERN.fullmatch(suite_id):
        raise ScalabilityValidationError(
            "suite ID must use 1-128 letters, digits, dots, underscores, or hyphens"
        )
    if suite_id in {".", ".."}:
        raise ScalabilityValidationError("suite ID cannot be a path component")
    return suite_id


def suite_path(suite_id: str, scratch_root: Path = SCRATCH_ROOT) -> Path:
    return scratch_root / validate_suite_id(suite_id)


class SyntheticTokenStream:
    """A seeded, finite CPU token stream with stable sample-boundary offsets.

    Each fixed-length sample has context+1 token IDs. Samples are generated one
    row at a time so regrouping them into different microbatches preserves the
    exact token IDs for cells with the same effective batch.
    """

    algorithm = "torch.randint_cpu_int64_rowwise_v1"

    def __init__(
        self,
        torch_module: Any,
        *,
        seed: int = STREAM_SEED,
        context_length: int = CONTEXT_LENGTH,
        vocab_size: int = 24_576,
        sample_count: int = STREAM_SAMPLE_COUNT,
    ) -> None:
        if seed != STREAM_SEED or context_length != CONTEXT_LENGTH:
            raise ScalabilityValidationError(
                "synthetic stream identity is baseline-pinned"
            )
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        self.torch = torch_module
        self.seed = seed
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.sample_count = sample_count
        generator = torch_module.Generator(device="cpu")
        generator.manual_seed(seed)
        rows = []
        for _ in range(sample_count):
            rows.append(
                torch_module.randint(
                    0,
                    vocab_size,
                    (context_length + 1,),
                    generator=generator,
                    dtype=torch_module.long,
                    device="cpu",
                )
            )
        self.tokens = torch_module.stack(rows, dim=0).contiguous()
        self.sha256 = hashlib.sha256(self.tokens.numpy().tobytes()).hexdigest()
        self.next_sample_offset = 0

    @property
    def next_token_offset(self) -> int:
        return self.next_sample_offset * self.context_length

    def take_microbatches(
        self, microbatch_size: int, accumulation_steps: int
    ) -> list[Any]:
        count = microbatch_size * accumulation_steps
        end = self.next_sample_offset + count
        if end > self.sample_count:
            raise RuntimeError("synthetic token stream is exhausted")
        selected = self.tokens[self.next_sample_offset : end]
        self.next_sample_offset = end
        return list(selected.reshape(accumulation_steps, microbatch_size, -1).unbind(0))

    def seek(self, sample_offset: int, *, expected_sha256: str) -> None:
        if expected_sha256 != self.sha256:
            raise CheckpointIntegrityError("synthetic input-stream hash mismatch")
        if not 0 <= sample_offset <= self.sample_count:
            raise CheckpointIntegrityError("synthetic input-stream offset is invalid")
        self.next_sample_offset = sample_offset

    def provenance(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "context_length": self.context_length,
            "vocab_size": self.vocab_size,
            "sample_count": self.sample_count,
            "sha256": self.sha256,
            "next_sample_offset": self.next_sample_offset,
            "next_token_offset": self.next_token_offset,
        }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_checkpoint(
    checkpoint_root: Path,
    *,
    torch_module: Any,
    payload: dict[str, object],
) -> Path:
    """Commit a checksummed checkpoint directory without replacing prior state."""

    update_index = int(payload["update_index"])
    checkpoint_dir = (
        checkpoint_root / f"update-{update_index:06d}-{uuid.uuid4().hex[:8]}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    final_path = checkpoint_dir / "checkpoint.pt"
    temporary_path = checkpoint_dir / ".checkpoint.pt.tmp"
    with temporary_path.open("wb") as stream:
        torch_module.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, final_path)
    digest = sha256_file(final_path)
    _atomic_write_bytes(
        checkpoint_dir / "checkpoint.sha256",
        f"{digest}  checkpoint.pt\n".encode("ascii"),
    )
    verify_checkpoint_checksum(final_path)
    _fsync_directory(checkpoint_dir)
    pointer = canonical_json(
        {
            "schema_version": "c3.6-checkpoint-pointer-v1",
            "update_index": update_index,
            "directory": checkpoint_dir.name,
            "sha256": digest,
        }
    )
    _atomic_write_bytes(checkpoint_root / "latest.json", pointer)
    return final_path


def verify_checkpoint_checksum(checkpoint_path: Path) -> str:
    checksum_path = checkpoint_path.with_name("checkpoint.sha256")
    try:
        digest, filename = (
            checksum_path.read_text(encoding="ascii").strip().split("  ", 1)
        )
    except (OSError, ValueError) as exc:
        raise CheckpointIntegrityError(
            f"checkpoint checksum is unavailable: {exc}"
        ) from exc
    if filename != checkpoint_path.name:
        raise CheckpointIntegrityError("checkpoint checksum names a different file")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise CheckpointIntegrityError("checkpoint checksum is malformed")
    actual = sha256_file(checkpoint_path)
    if actual != digest:
        raise CheckpointIntegrityError(
            f"checkpoint checksum mismatch: expected {digest}, got {actual}"
        )
    return actual


def load_checkpoint(
    checkpoint_path: Path,
    *,
    torch_module: Any,
    expected_identity: dict[str, object],
) -> dict[str, object]:
    """Verify bytes and pinned identities before deserializing/applying state."""

    verify_checkpoint_checksum(checkpoint_path)
    try:
        with checkpoint_path.open("rb") as stream:
            payload = torch_module.load(
                io.BytesIO(stream.read()), map_location="cpu", weights_only=False
            )
    except Exception as exc:
        raise CheckpointIntegrityError(
            f"checkpoint could not be loaded: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CheckpointIntegrityError("checkpoint payload is not a mapping")
    if payload.get("schema_version") != "c3.6-checkpoint-v1":
        raise CheckpointIntegrityError("checkpoint schema version mismatch")
    if payload.get("identity") != expected_identity:
        raise CheckpointIntegrityError("checkpoint pinned identity mismatch")
    stream = payload.get("synthetic_stream")
    if not isinstance(stream, dict) or stream.get("sha256") != expected_identity.get(
        "synthetic_stream_sha256"
    ):
        raise CheckpointIntegrityError("checkpoint synthetic stream identity mismatch")
    if stream.get("seed") != expected_identity.get("synthetic_stream_seed"):
        raise CheckpointIntegrityError("checkpoint synthetic stream seed mismatch")
    required = {
        "model_state",
        "optimizer_state",
        "rng_states",
        "update_index",
        "synthetic_stream",
        "identity",
    }
    if not required.issubset(payload):
        raise CheckpointIntegrityError("checkpoint is missing required state fields")
    return payload


def _python_torch_numpy_rng_states(
    torch_module: Any, *, include_cuda: bool
) -> dict[str, Any]:
    states: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch_module.random.get_rng_state().clone(),
        "configured_seed": STREAM_SEED,
    }
    try:
        numpy = importlib.import_module("numpy")
    except ImportError:
        states["numpy"] = None
    else:
        states["numpy"] = numpy.random.get_state()
    if include_cuda:
        states["torch_cuda"] = [
            state.clone() for state in torch_module.cuda.get_rng_state_all()
        ]
    else:
        states["torch_cuda"] = None
    return states


def _restore_rng_states(torch_module: Any, states: dict[str, Any]) -> None:
    random.setstate(states["python"])
    if states.get("numpy") is not None:
        numpy = importlib.import_module("numpy")
        numpy.random.set_state(states["numpy"])
    torch_module.random.set_rng_state(states["torch_cpu"])
    if states.get("torch_cuda") is not None:
        torch_module.cuda.set_rng_state_all(states["torch_cuda"])


def _cpu_copy(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def write_state_snapshot(
    torch_module: Any, path: Path, payload: dict[str, object]
) -> str:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("wb") as stream:
            torch_module.save(_cpu_copy(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    digest = sha256_file(path)
    _atomic_write_bytes(
        path.with_suffix(path.suffix + ".sha256"),
        f"{digest}  {path.name}\n".encode("ascii"),
    )
    return digest


def read_state_snapshot(torch_module: Any, path: Path) -> dict[str, object]:
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    try:
        expected, filename = (
            checksum_path.read_text(encoding="ascii").strip().split("  ", 1)
        )
    except (OSError, ValueError) as exc:
        raise CheckpointIntegrityError(
            f"state snapshot checksum is unavailable: {exc}"
        ) from exc
    if filename != path.name or sha256_file(path) != expected:
        raise CheckpointIntegrityError("state snapshot checksum mismatch")
    with path.open("rb") as stream:
        value = torch_module.load(
            io.BytesIO(stream.read()), map_location="cpu", weights_only=False
        )
    if not isinstance(value, dict):
        raise CheckpointIntegrityError("state snapshot payload is not a mapping")
    return value


def _torch_and_nvml() -> tuple[Any, Any]:
    try:
        torch_module = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for C3.6 GPU execution") from exc
    try:
        nvml = importlib.import_module("pynvml")
    except ImportError as exc:
        raise RuntimeError(
            "NVML process monitoring is required for C3.6 GPU execution"
        ) from exc
    return torch_module, nvml


def _require_gpu_environment(environ: dict[str, str] | None = None) -> None:
    values = os.environ if environ is None else environ
    if values.get("CUDA_VISIBLE_DEVICES") != "0":
        raise GpuSafetyError(
            "set CUDA_VISIBLE_DEVICES=0 exactly before C3.6 GPU execution"
        )


def _benchmark_config(
    cell: CellConfig, scratch_root: Path, output_root: Path
) -> BenchmarkConfig:
    return BenchmarkConfig(
        architecture="C",
        context_length=CONTEXT_LENGTH,
        device="cuda:0",
        precision="bf16",
        microbatch_size=cell.microbatch_size,
        warmup_steps=WARMUP_UPDATES,
        timed_steps=MEASURED_UPDATES,
        seed=STREAM_SEED,
        attention_backend="auto",
        gpu_memory_budget_mib=GPU_MEMORY_BUDGET_MIB,
        safety_margin_mib=MINIMUM_FREE_MEMORY_MIB,
        output_root=output_root,
        scratch_root=scratch_root,
        lock_path=LOCK_PATH,
        tie_weights=True,
        activation_checkpointing="none",
    ).validate()


def _set_nondeterministic_benchmark_settings(torch_module: Any) -> None:
    torch_module.backends.cudnn.benchmark = False
    if hasattr(torch_module.backends.cuda.matmul, "allow_tf32"):
        torch_module.backends.cuda.matmul.allow_tf32 = False


def _configure_deterministic_restart(torch_module: Any) -> dict[str, object]:
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True
    if hasattr(torch_module.backends.cuda.matmul, "allow_tf32"):
        torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.use_deterministic_algorithms(True)
    return {
        "torch_use_deterministic_algorithms": bool(
            torch_module.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch_module.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch_module.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(
            getattr(torch_module.backends.cuda.matmul, "allow_tf32", False)
        ),
        "attention_backend": "auto",
        "backend_fallback_or_switch": False,
    }


def _create_model_and_optimizer(torch_module: Any, *, device: str = "cuda:0"):
    from cambacica.model.transformer import (
        CambacicaCausalLM,
        count_instantiated_parameters,
    )

    spec = get_architecture("C")
    model = CambacicaCausalLM(
        spec,
        tie_weights=True,
        seed=STREAM_SEED,
        attention_backend="auto",
        activation_dtype=torch_module.bfloat16,
        device="cpu",
    )
    count = count_instantiated_parameters(model)
    if count != 176_180_224:
        raise GpuSafetyError(f"Architecture C parameter count mismatch: {count}")
    if model.lm_head.weight is not model.token_embedding.weight:
        raise GpuSafetyError("Architecture C embedding/output weights are not tied")
    model = model.to(device)
    model.train()
    optimizer = torch_module.optim.AdamW(
        model.parameters(),
        lr=float(OPTIMIZER_FIXTURE["learning_rate"]),
        betas=tuple(OPTIMIZER_FIXTURE["betas"]),
        weight_decay=float(OPTIMIZER_FIXTURE["weight_decay"]),
        eps=float(OPTIMIZER_FIXTURE["epsilon"]),
    )
    return model, optimizer, count


def _autocast(torch_module: Any, device: str):
    if not device.startswith("cuda"):
        return nullcontext()
    return torch_module.autocast(device.split(":", 1)[0], dtype=torch_module.bfloat16)


def _optimizer_update(
    torch_module: Any,
    *,
    model: Any,
    optimizer: Any,
    microbatches: list[Any],
    device: str,
) -> float:
    """Accumulate equal-sized microbatch means and step once per optimizer update."""

    if not microbatches:
        raise ValueError("an optimizer update requires at least one microbatch")
    optimizer.zero_grad(set_to_none=True)
    losses: list[Any] = []
    accumulation_steps = len(microbatches)
    for token_ids in microbatches:
        input_ids = token_ids[:, :-1].to(device, non_blocking=False)
        labels = token_ids[:, 1:].to(device, non_blocking=False)
        with _autocast(torch_module, device):
            logits, loss = model(input_ids, labels=labels)
        if tuple(logits.shape) != (
            input_ids.shape[0],
            CONTEXT_LENGTH,
            int(model.spec.vocab_size),
        ):
            raise RuntimeError(f"unexpected logits shape: {tuple(logits.shape)}")
        if not bool(torch_module.isfinite(loss.detach()).all().item()):
            raise GpuSafetyError("non-finite loss detected during accumulation")
        (loss / accumulation_steps).backward()
        losses.append(loss.detach().float())
        del logits, input_ids, labels
    gradient_checks = [
        torch_module.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradient_checks:
        raise GpuSafetyError("no gradients were produced by the accumulated loss")
    numeric_stability_check(
        True, bool(torch_module.stack(gradient_checks).all().item())
    )
    mean_loss = torch_module.stack(losses).mean()
    loss_value = float(mean_loss.item())
    if not math.isfinite(loss_value):
        raise GpuSafetyError("non-finite accumulated loss detected")
    optimizer.step()
    return loss_value


def _tensor_bytes(tensors: Iterable[Any]) -> int:
    return sum(int(tensor.numel()) * int(tensor.element_size()) for tensor in tensors)


def _optimizer_moment_tensors(optimizer: Any) -> list[Any]:
    moments = []
    for state in optimizer.state.values():
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            tensor = state.get(name)
            if tensor is not None:
                moments.append(tensor)
    return moments


def _execute_cell(
    cell: CellConfig,
    *,
    torch_module: Any,
    nvml: Any,
    scratch_root: Path,
    output_root: Path,
) -> dict[str, object]:
    cell.validate()
    config = _benchmark_config(cell, scratch_root, output_root)
    diagnostics: dict[str, object] = {}
    snapshot, handle = preflight_gpu(torch_module, nvml, config)
    diagnostics["startup_gpu"] = snapshot.to_dict()
    diagnostics["visible_cuda_devices"] = int(torch_module.cuda.device_count())
    if not torch_module.cuda.is_bf16_supported():
        raise GpuSafetyError("BF16 is not supported by the visible physical GPU 0")
    _set_nondeterministic_benchmark_settings(torch_module)
    allocator_fraction = apply_allocator_limit(
        torch_module, config, snapshot.total_memory_mib
    )
    if TORCH_ALLOCATOR_LIMIT_MIB != config.allocator_limit_mib:
        raise GpuSafetyError(
            "PyTorch allocator ceiling differs from the approved 35840 MiB"
        )
    diagnostics["allocator_fraction"] = allocator_fraction
    diagnostics["allocator_limit_mib"] = config.allocator_limit_mib
    torch_module.empty(1, device="cuda:0")
    torch_module.cuda.synchronize(0)
    monitor = GpuSafetyMonitor(nvml, handle, config)
    samples = [monitor.sample()]
    model = optimizer = stream = None
    try:
        model, optimizer, parameter_count = _create_model_and_optimizer(torch_module)
        stream = SyntheticTokenStream(torch_module)
        diagnostics["synthetic_stream"] = stream.provenance()
        diagnostics["initial_process_memory"] = monitor.sample()
        attention = c3_runner._attention_profiler_result(
            torch_module,
            model,
            stream.tokens[: cell.microbatch_size, :-1].to("cuda:0"),
            "bf16",
        )
        diagnostics["attention_backend"] = attention
        del attention
        torch_module.cuda.empty_cache()
        torch_module.cuda.reset_peak_memory_stats(0)
        samples.append(monitor.sample())
        warmup_losses: list[float] = []
        measured_losses: list[float] = []
        for _ in range(cell.warmup_updates):
            microbatches = stream.take_microbatches(
                cell.microbatch_size, cell.gradient_accumulation_steps
            )
            warmup_losses.append(
                _optimizer_update(
                    torch_module,
                    model=model,
                    optimizer=optimizer,
                    microbatches=microbatches,
                    device="cuda:0",
                )
            )
            samples.append(monitor.sample())
        torch_module.cuda.synchronize(0)
        wall_start = time.perf_counter()
        events = []
        for _ in range(cell.measured_updates):
            start_event = torch_module.cuda.Event(enable_timing=True)
            end_event = torch_module.cuda.Event(enable_timing=True)
            start_event.record()
            microbatches = stream.take_microbatches(
                cell.microbatch_size, cell.gradient_accumulation_steps
            )
            measured_losses.append(
                _optimizer_update(
                    torch_module,
                    model=model,
                    optimizer=optimizer,
                    microbatches=microbatches,
                    device="cuda:0",
                )
            )
            end_event.record()
            events.append((start_event, end_event))
            samples.append(monitor.sample())
        torch_module.cuda.synchronize(0)
        wall_seconds = time.perf_counter() - wall_start
        event_times_ms = [float(start.elapsed_time(end)) for start, end in events]
        if any(not math.isfinite(value) for value in event_times_ms):
            raise GpuSafetyError("non-finite CUDA event timing detected")
        final_sample = monitor.sample()
        samples.append(final_sample)
        allocated_mib = torch_module.cuda.max_memory_allocated(0) / (1024 * 1024)
        reserved_mib = torch_module.cuda.max_memory_reserved(0) / (1024 * 1024)
        parameters = list(model.parameters())
        parameter_bytes = _tensor_bytes(parameters)
        gradient_bytes = _tensor_bytes(
            parameter.grad for parameter in parameters if parameter.grad is not None
        )
        moment_tensors = _optimizer_moment_tensors(optimizer)
        moment_bytes = _tensor_bytes(moment_tensors)
        mib_per_byte = 1 / (1024 * 1024)
        process_peak = monitor.peak_process_memory_mib
        free_values = [int(sample["free_memory_mib"]) for sample in samples]
        return {
            "schema_version": "c3.6-cell-result-v1",
            "status": "complete",
            "cell_id": cell.cell_id,
            "baseline_id": BASELINE_ID,
            "baseline_sha256": BASELINE_SHA256,
            "architecture_id": "C",
            "parameter_count": parameter_count,
            "weight_tying": True,
            "context_length": CONTEXT_LENGTH,
            "microbatch_size": cell.microbatch_size,
            "gradient_accumulation_steps": cell.gradient_accumulation_steps,
            "tokens_per_update": cell.tokens_per_update,
            "warmup_updates": cell.warmup_updates,
            "measured_updates": cell.measured_updates,
            "repeats": cell.repeats,
            "optimizer_fixture": OPTIMIZER_FIXTURE,
            "final_training_optimizer_selected": False,
            "activation_checkpointing": "none",
            "precision": "bf16_autocast",
            "attention_backend": diagnostics["attention_backend"],
            "synthetic_stream": stream.provenance(),
            "loss": {
                "warmup_mean": sum(warmup_losses) / len(warmup_losses),
                "measured_mean": sum(measured_losses) / len(measured_losses),
                "measured_first": measured_losses[0],
                "measured_last": measured_losses[-1],
                "measured_sequence": measured_losses,
                "finite": True,
            },
            "optimizer_update_latency_gpu_event_ms": summarize_step_times(
                event_times_ms
            ),
            "wall_time_seconds_including_safety_checks": wall_seconds,
            "wall_tokens_per_second_including_safety_checks": tokens_per_second(
                cell.tokens_per_update, cell.measured_updates, wall_seconds
            ),
            "peak_pytorch_allocated_mib": allocated_mib,
            "peak_pytorch_reserved_mib": reserved_mib,
            "sampled_cambacica_process_peak_mib": process_peak,
            "estimated_non_pytorch_peak_mib": max(0.0, process_peak - reserved_mib),
            "startup_free_memory_mib": snapshot.free_memory_mib,
            "minimum_free_memory_mib": min(free_values),
            "gpu": {
                "hostname": platform.node(),
                "physical_index": snapshot.physical_index,
                "uuid": snapshot.uuid,
                "name": snapshot.name,
                "pci_bus_id": snapshot.pci_bus_id,
                "mapping_method": snapshot.mapping_method,
                "driver_version": snapshot.driver_version,
                "visible_cuda_devices": int(torch_module.cuda.device_count()),
                "cuda_runtime_version": torch_module.version.cuda,
            },
            "runtime": runtime_identity(torch_module),
            "memory_accounting_bytes": {
                "parameter_bytes": parameter_bytes,
                "gradient_bytes": gradient_bytes,
                "adamw_moment_bytes": moment_bytes,
                "persistent_subtotal_bytes": parameter_bytes
                + gradient_bytes
                + moment_bytes,
                "parameter_mib": parameter_bytes * mib_per_byte,
                "gradient_mib": gradient_bytes * mib_per_byte,
                "adamw_moment_mib": moment_bytes * mib_per_byte,
                "persistent_subtotal_mib": (
                    parameter_bytes + gradient_bytes + moment_bytes
                )
                * mib_per_byte,
                "parameter_dtype": str(parameters[0].dtype),
                "gradient_dtypes": sorted(
                    {
                        str(parameter.grad.dtype)
                        for parameter in parameters
                        if parameter.grad is not None
                    }
                ),
                "adamw_moment_dtypes": sorted(
                    {str(tensor.dtype) for tensor in moment_tensors}
                ),
                "allocator_reserved_peak_bytes": int(reserved_mib * 1024 * 1024),
                "nvml_sampled_process_peak_mib": process_peak,
                "nvml_minus_allocator_reserved_estimate_mib": max(
                    0.0, process_peak - reserved_mib
                ),
            },
            "memory_policy": cell.to_dict()["memory_policy"],
            "numerical_status": "finite",
            "diagnostics": {
                **diagnostics,
                "last_gpu_memory_sample": final_sample,
                "minimum_free_memory_mib": min(free_values),
                "sampled_cambacica_process_peak_mib": process_peak,
            },
            "git": git_provenance(),
            "implementation_sha256": implementation_sha256(),
        }
    finally:
        del model, optimizer, stream
        import gc

        gc.collect()
        torch_module.cuda.empty_cache()


def write_cell_artifacts(
    cell_directory: Path,
    *,
    configuration: dict[str, object],
    result: dict[str, object],
) -> dict[str, str]:
    cell_directory.mkdir(parents=True, exist_ok=True)
    configuration_bytes = canonical_json(configuration)
    result_value = dict(result)
    configuration_sha = hashlib.sha256(configuration_bytes).hexdigest()
    result_value["configuration_sha256"] = configuration_sha
    result_bytes = canonical_json(result_value)
    _atomic_write_bytes(cell_directory / "configuration.json", configuration_bytes)
    _atomic_write_bytes(cell_directory / "result.json", result_bytes)
    hashes = {
        "configuration.json": configuration_sha,
        "result.json": hashlib.sha256(result_bytes).hexdigest(),
    }
    _atomic_write_bytes(
        cell_directory / "checksums.sha256",
        "".join(
            f"{digest}  {name}\n" for name, digest in sorted(hashes.items())
        ).encode("ascii"),
    )
    return hashes


def verify_cell_artifacts(cell_directory: Path) -> dict[str, object]:
    try:
        entries = {}
        for line in (
            (cell_directory / "checksums.sha256")
            .read_text(encoding="ascii")
            .splitlines()
        ):
            digest, name = line.split("  ", 1)
            entries[name] = digest
    except (OSError, ValueError) as exc:
        raise CheckpointIntegrityError(
            f"cell checksum ledger is invalid: {exc}"
        ) from exc
    if set(entries) != {"configuration.json", "result.json"}:
        raise CheckpointIntegrityError("cell checksum ledger has unexpected files")
    for name, digest in entries.items():
        if sha256_file(cell_directory / name) != digest:
            raise CheckpointIntegrityError(f"cell artifact checksum mismatch: {name}")
    configuration = json.loads(
        (cell_directory / "configuration.json").read_text(encoding="utf-8")
    )
    result = json.loads((cell_directory / "result.json").read_text(encoding="utf-8"))
    if result.get("configuration_sha256") != entries["configuration.json"]:
        raise CheckpointIntegrityError("result configuration identity mismatch")
    if result.get("cell_id") != configuration.get("cell_id"):
        raise CheckpointIntegrityError("cell artifact IDs do not match")
    return {"configuration": configuration, "result": result, "hashes": entries}


def _write_execution_failure(
    cell: CellConfig,
    cell_directory: Path,
    error: BaseException,
    *,
    diagnostics: dict[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": "c3.6-cell-result-v1",
        "status": "failed",
        "cell_id": cell.cell_id,
        "baseline_id": BASELINE_ID,
        "failure_kind": c3_runner._classify_failure(error),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
        },
        "diagnostics": diagnostics or {},
        "git": git_provenance(),
        "implementation_sha256": implementation_sha256(),
    }
    if isinstance(error, GpuSafetyError):
        result["diagnostics"].update(error.diagnostics)
    write_cell_artifacts(cell_directory, configuration=cell.to_dict(), result=result)
    return result


def _execute_cells_under_lock(
    cells: tuple[CellConfig, ...],
    *,
    suite_directory: Path,
    scratch_root: Path,
    output_root: Path,
    lock_path: Path,
) -> dict[str, object]:
    _require_gpu_environment()
    require_storage_layout(scratch_root, output_root)
    suite_directory.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        if (suite_directory / "cells" / cell.cell_id).exists():
            raise FileExistsError(
                f"cell artifacts already exist: {suite_directory / 'cells' / cell.cell_id}"
            )
    torch_module = nvml = None
    results: list[dict[str, object]] = []
    lock = BenchmarkLock(lock_path)
    started = time.time()
    try:
        try:
            lock.acquire()
        except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - persist lock errors
            first = cells[0]
            attempt_dir = (
                suite_directory / "startup_failures" / f"lock-{uuid.uuid4().hex}"
            )
            failed = _write_execution_failure(
                first,
                attempt_dir,
                exc,
            )
            return {
                "status": "failed",
                "suite_id": suite_directory.name,
                "completed_cells": [],
                "failed_cell": first.cell_id,
                "results": [failed],
                "elapsed_seconds": time.time() - started,
                "error": failed["error"],
            }
        try:
            torch_module, nvml = _torch_and_nvml()
        except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - persist startup errors
            first = cells[0]
            failed = _write_execution_failure(
                first,
                suite_directory / "cells" / first.cell_id,
                exc,
            )
            return {
                "status": "failed",
                "suite_id": suite_directory.name,
                "completed_cells": [first.cell_id],
                "failed_cell": first.cell_id,
                "results": [failed],
                "elapsed_seconds": time.time() - started,
                "error": failed["error"],
            }
        for cell in cells:
            cell_dir = suite_directory / "cells" / cell.cell_id
            if cell_dir.exists():
                raise FileExistsError(f"cell artifacts already exist: {cell_dir}")
            try:
                result = _execute_cell(
                    cell,
                    torch_module=torch_module,
                    nvml=nvml,
                    scratch_root=scratch_root,
                    output_root=output_root,
                )
                write_cell_artifacts(
                    cell_dir, configuration=cell.to_dict(), result=result
                )
                results.append(result)
            except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - persist runtime failures
                result = _write_execution_failure(cell, cell_dir, exc)
                results.append(result)
                return {
                    "status": "failed",
                    "suite_id": suite_directory.name,
                    "completed_cells": [item["cell_id"] for item in results],
                    "failed_cell": cell.cell_id,
                    "results": results,
                    "elapsed_seconds": time.time() - started,
                    "error": result["error"],
                }
        return {
            "status": "complete",
            "suite_id": suite_directory.name,
            "completed_cells": [item["cell_id"] for item in results],
            "results": results,
            "elapsed_seconds": time.time() - started,
            "gpu_execution": True,
        }
    finally:
        lock.release()
        del torch_module, nvml


def run_matrix(
    suite_id: str,
    *,
    scratch_root: Path = SCRATCH_ROOT,
    output_root: Path = OUTPUT_ROOT,
    lock_path: Path = LOCK_PATH,
) -> dict[str, object]:
    """Execute the nine approved cells sequentially under one local lock."""

    validate_suite_id(suite_id)
    suite_directory = suite_path(suite_id, scratch_root)
    if suite_directory.exists() and any(suite_directory.iterdir()):
        raise FileExistsError(f"suite already has local artifacts: {suite_directory}")
    return _execute_cells_under_lock(
        approved_cells(),
        suite_directory=suite_directory,
        scratch_root=scratch_root,
        output_root=output_root,
        lock_path=lock_path,
    )


def run_one_cell(
    suite_id: str,
    microbatch_size: int,
    gradient_accumulation_steps: int,
    *,
    scratch_root: Path = SCRATCH_ROOT,
    output_root: Path = OUTPUT_ROOT,
    lock_path: Path = LOCK_PATH,
) -> dict[str, object]:
    validate_suite_id(suite_id)
    cell = next(
        (
            item
            for item in approved_cells()
            if item.microbatch_size == microbatch_size
            and item.gradient_accumulation_steps == gradient_accumulation_steps
        ),
        None,
    )
    if cell is None:
        raise ScalabilityValidationError(
            "requested cell is outside the approved matrix"
        )
    suite_directory = suite_path(suite_id, scratch_root)
    return _execute_cells_under_lock(
        (cell,),
        suite_directory=suite_directory,
        scratch_root=scratch_root,
        output_root=output_root,
        lock_path=lock_path,
    )


def _configuration_identity(
    cell: CellConfig,
    *,
    implementation_digest: str,
    provenance: dict[str, object],
    runtime: dict[str, object],
    gpu_identity: dict[str, object],
    stream_hash: str,
    deterministic_settings: dict[str, object],
    attention_backend: dict[str, object],
) -> dict[str, object]:
    return {
        "baseline_id": BASELINE_ID,
        "baseline_sha256": BASELINE_SHA256,
        "baseline_approved_identity": BASELINE_APPROVED,
        "tokenizer_identity": BASELINE_FROZEN["tokenizer_identity"],
        "cell_configuration": cell.to_dict(),
        "cell_configuration_sha256": hashlib.sha256(
            canonical_json(cell.to_dict())
        ).hexdigest(),
        "optimizer_fixture": OPTIMIZER_FIXTURE,
        "synthetic_stream_seed": STREAM_SEED,
        "synthetic_stream_sha256": stream_hash,
        "git": provenance,
        "implementation_sha256": implementation_digest,
        "runtime": runtime,
        "hostname": platform.node(),
        "hardware": gpu_identity,
        "deterministic_kernel_settings": deterministic_settings,
        "attention_backend": {
            "requested": attention_backend.get("requested"),
            "observed": attention_backend.get("observed"),
        },
    }


def _restart_runtime(
    torch_module: Any,
    nvml: Any,
    *,
    scratch_root: Path,
    output_root: Path,
    lock_path: Path,
) -> tuple[
    CellConfig,
    BenchmarkConfig,
    Any,
    Any,
    GpuSafetyMonitor,
    dict[str, object],
]:
    cell = CellConfig("cell-m1-a1", 1, 1).validate()
    config = _benchmark_config(cell, scratch_root, output_root)
    snapshot, handle = preflight_gpu(torch_module, nvml, config)
    if not torch_module.cuda.is_bf16_supported():
        raise GpuSafetyError("BF16 is not supported by the visible physical GPU 0")
    apply_allocator_limit(torch_module, config, snapshot.total_memory_mib)
    settings = _configure_deterministic_restart(torch_module)
    random.seed(STREAM_SEED)
    try:
        numpy = importlib.import_module("numpy")
    except ImportError:
        pass
    else:
        numpy.random.seed(STREAM_SEED)
    torch_module.manual_seed(STREAM_SEED)
    torch_module.cuda.manual_seed_all(STREAM_SEED)
    torch_module.empty(1, device="cuda:0")
    torch_module.cuda.synchronize(0)
    monitor = GpuSafetyMonitor(nvml, handle, config)
    monitor.sample()
    return cell, config, handle, snapshot, monitor, settings


def _run_restart_updates(
    torch_module: Any,
    *,
    model: Any,
    optimizer: Any,
    stream: SyntheticTokenStream,
    count: int,
    monitor: GpuSafetyMonitor,
) -> list[float]:
    losses = []
    for _ in range(count):
        microbatches = stream.take_microbatches(1, 1)
        losses.append(
            _optimizer_update(
                torch_module,
                model=model,
                optimizer=optimizer,
                microbatches=microbatches,
                device="cuda:0",
            )
        )
        monitor.sample()
    return losses


def _state_comparison(
    torch_module: Any, expected: Any, actual: Any
) -> dict[str, object]:
    mismatches: list[dict[str, object]] = []
    tensor_count = 0
    max_abs = 0.0
    max_relative = 0.0

    def compare(left: Any, right: Any, path: str) -> None:
        nonlocal tensor_count, max_abs, max_relative
        if hasattr(left, "shape") and hasattr(right, "shape"):
            tensor_count += 1
            if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
                mismatches.append({"path": path, "kind": "tensor_metadata"})
                return
            left_bytes = (
                left.detach().cpu().contiguous().reshape(-1).view(torch_module.uint8)
            )
            right_bytes = (
                right.detach().cpu().contiguous().reshape(-1).view(torch_module.uint8)
            )
            if torch_module.equal(left_bytes, right_bytes):
                return
            if left.is_floating_point() or left.is_complex():
                comparison_dtype = (
                    torch_module.complex128
                    if left.is_complex()
                    else torch_module.float64
                )
                left_values = left.to(comparison_dtype)
                right_values = right.to(comparison_dtype)
                delta = (left_values - right_values).abs().to(torch_module.float64)
                absolute = float(delta.max().item()) if delta.numel() else 0.0
                denominator = torch_module.maximum(
                    left_values.abs().to(torch_module.float64),
                    torch_module.full_like(
                        delta, torch_module.finfo(torch_module.float64).tiny
                    ),
                )
                relative = (
                    float((delta / denominator).max().item()) if delta.numel() else 0.0
                )
                max_abs = max(max_abs, absolute)
                max_relative = max(max_relative, relative)
                mismatches.append(
                    {
                        "path": path,
                        "kind": "tensor_value",
                        "max_absolute_delta": absolute,
                        "max_relative_delta": relative,
                    }
                )
            else:
                mismatches.append({"path": path, "kind": "tensor_value"})
            return
        if isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                mismatches.append({"path": path, "kind": "mapping_keys"})
                return
            for key in left:
                compare(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
            if len(left) != len(right):
                mismatches.append({"path": path, "kind": "sequence_length"})
                return
            for index, (left_item, right_item) in enumerate(
                zip(left, right, strict=True)
            ):
                compare(left_item, right_item, f"{path}[{index}]")
            return
        if left != right:
            mismatches.append({"path": path, "kind": "scalar_value"})

    compare(expected, actual, "state")
    return {
        "bitwise_equal": not mismatches,
        "tensor_count": tensor_count,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:50],
        "maximum_absolute_delta": max_abs,
        "maximum_relative_delta": max_relative,
    }


def _same_float_sequence(left: list[float], right: list[float]) -> bool:
    import struct

    return len(left) == len(right) and all(
        struct.pack("!d", first) == struct.pack("!d", second)
        for first, second in zip(left, right, strict=True)
    )


def _restart_worker(plan_path: Path, phase: str) -> dict[str, object]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    scratch_root = Path(plan["scratch_root"])
    output_root = Path(plan["output_root"])
    lock_path = Path(plan["lock_path"])
    restart_dir = Path(plan["restart_dir"])
    if lock_path != LOCK_PATH:
        raise ScalabilityValidationError("restart worker must use the existing C3 lock")
    if scratch_root != SCRATCH_ROOT or output_root != OUTPUT_ROOT:
        raise ScalabilityValidationError("restart worker storage roots are pinned")
    _require_gpu_environment()
    require_storage_layout(scratch_root, output_root)
    lock = BenchmarkLock(lock_path)
    lock.acquire()
    cuda_context_ready = False
    torch_module = None
    model = optimizer = stream = None
    try:
        torch_module, nvml = _torch_and_nvml()
        cell, _config, _handle, snapshot, monitor, deterministic = _restart_runtime(
            torch_module,
            nvml,
            scratch_root=scratch_root,
            output_root=output_root,
            lock_path=lock_path,
        )
        cuda_context_ready = True
        model, optimizer, parameter_count = _create_model_and_optimizer(torch_module)
        stream = SyntheticTokenStream(torch_module, sample_count=STREAM_SAMPLE_COUNT)
        attention = c3_runner._attention_profiler_result(
            torch_module, model, stream.tokens[:1, :-1].to("cuda:0"), "bf16"
        )
        torch_module.cuda.synchronize(0)
        gpu_identity = {
            "physical_index": snapshot.physical_index,
            "uuid": snapshot.uuid,
            "name": snapshot.name,
            "pci_bus_id": snapshot.pci_bus_id,
            "mapping_method": snapshot.mapping_method,
            "driver_version": snapshot.driver_version,
            "visible_cuda_devices": int(torch_module.cuda.device_count()),
            "cuda_runtime_version": torch_module.version.cuda,
        }
        identity = _configuration_identity(
            cell,
            implementation_digest=implementation_sha256(),
            provenance=git_provenance(),
            runtime=runtime_identity(torch_module),
            gpu_identity=gpu_identity,
            stream_hash=stream.sha256,
            deterministic_settings=deterministic,
            attention_backend=attention,
        )
        expected_identity = plan.get("expected_identity")
        if expected_identity is not None and identity != expected_identity:
            raise CheckpointIntegrityError(
                "restart phase pinned identity differs from reference"
            )

        if phase == "reference":
            losses = _run_restart_updates(
                torch_module,
                model=model,
                optimizer=optimizer,
                stream=stream,
                count=4,
                monitor=monitor,
            )
            snapshot_payload = {
                "identity": identity,
                "losses": losses,
                "update_index": 4,
                "synthetic_stream": stream.provenance(),
                "attention_backend": attention,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            path = restart_dir / "reference_state.pt"
            digest = write_state_snapshot(torch_module, path, snapshot_payload)
            return {
                "status": "complete",
                "phase": phase,
                "identity": identity,
                "losses": losses,
                "reference_state_path": str(path),
                "reference_state_sha256": digest,
                "parameter_count": parameter_count,
            }

        if phase == "prepare":
            losses = _run_restart_updates(
                torch_module,
                model=model,
                optimizer=optimizer,
                stream=stream,
                count=2,
                monitor=monitor,
            )
            payload = {
                "schema_version": "c3.6-checkpoint-v1",
                "identity": identity,
                "model_state": _cpu_copy(model.state_dict()),
                "optimizer_state": _cpu_copy(optimizer.state_dict()),
                "rng_states": _python_torch_numpy_rng_states(
                    torch_module, include_cuda=True
                ),
                "update_index": 2,
                "synthetic_stream": stream.provenance(),
                "attention_backend": attention,
                "losses_before_checkpoint": losses,
                "configured_seed": STREAM_SEED,
                "deterministic_kernel_settings": deterministic,
                "scaler_state": None,
                "scheduler_state": None,
            }
            checkpoint_path = write_checkpoint(
                restart_dir / "checkpoints",
                torch_module=torch_module,
                payload=payload,
            )
            return {
                "status": "complete",
                "phase": phase,
                "identity": identity,
                "losses": losses,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": verify_checkpoint_checksum(checkpoint_path),
                "update_index": 2,
                "synthetic_stream": stream.provenance(),
            }

        if phase != "resume":
            raise ValueError(f"unknown restart worker phase: {phase}")
        checkpoint_path = Path(plan["checkpoint_path"])
        payload = load_checkpoint(
            checkpoint_path,
            torch_module=torch_module,
            expected_identity=identity,
        )
        if int(payload["update_index"]) != 2:
            raise CheckpointIntegrityError(
                "restart checkpoint update index must equal 2"
            )
        if payload.get("configured_seed") != STREAM_SEED:
            raise CheckpointIntegrityError("checkpoint configured seed mismatch")
        if payload["rng_states"].get("configured_seed") != STREAM_SEED:
            raise CheckpointIntegrityError("checkpoint RNG-state seed mismatch")
        if payload.get("deterministic_kernel_settings") != deterministic:
            raise CheckpointIntegrityError(
                "checkpoint deterministic-kernel settings mismatch"
            )
        stream_state = payload["synthetic_stream"]
        if int(stream_state["next_sample_offset"]) != int(payload["update_index"]):
            raise CheckpointIntegrityError(
                "checkpoint sample offset does not match completed update index"
            )
        expected_token_offset = int(stream_state["next_sample_offset"]) * CONTEXT_LENGTH
        if int(stream_state["next_token_offset"]) != expected_token_offset:
            raise CheckpointIntegrityError("checkpoint token offset is inconsistent")
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        stream.seek(
            int(stream_state["next_sample_offset"]),
            expected_sha256=str(stream_state["sha256"]),
        )
        _restore_rng_states(torch_module, payload["rng_states"])
        resumed_losses = _run_restart_updates(
            torch_module,
            model=model,
            optimizer=optimizer,
            stream=stream,
            count=2,
            monitor=monitor,
        )
        reference_path = Path(plan["reference_state_path"])
        if sha256_file(reference_path) != plan.get("reference_state_sha256"):
            raise CheckpointIntegrityError("reference state pinned checksum mismatch")
        reference = read_state_snapshot(torch_module, reference_path)
        if reference.get("identity") != identity:
            raise CheckpointIntegrityError("reference state pinned identity mismatch")
        expected_losses = list(reference["losses"])
        actual_losses = list(payload["losses_before_checkpoint"]) + resumed_losses
        loss_equal = _same_float_sequence(expected_losses, actual_losses)
        expected_state = {
            "model_state": reference["model_state"],
            "optimizer_state": reference["optimizer_state"],
        }
        actual_state = {
            "model_state": _cpu_copy(model.state_dict()),
            "optimizer_state": _cpu_copy(optimizer.state_dict()),
        }
        state_report = _state_comparison(torch_module, expected_state, actual_state)
        return {
            "status": "complete"
            if loss_equal and state_report["bitwise_equal"]
            else "failed",
            "phase": phase,
            "identity": identity,
            "reference_losses": expected_losses,
            "resumed_losses": actual_losses,
            "loss_sequences_bitwise_equal": loss_equal,
            "model_optimizer_comparison": state_report,
            "update_index": 4,
            "synthetic_stream": stream.provenance(),
            "attention_backend": attention,
            "deterministic_kernel_incompatibility": None,
        }
    finally:
        del model, optimizer, stream
        import gc

        gc.collect()
        try:
            if cuda_context_ready:
                torch_module.cuda.empty_cache()
        finally:
            lock.release()


def _run_worker_subprocess(plan_path: Path, phase: str) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "cambacica.benchmark.scalability_cli",
        "_restart-worker",
        "--execute",
        "--plan",
        str(plan_path),
        "--phase",
        phase,
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result = {
            "status": "failed",
            "phase": phase,
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-4_000:],
            "stderr": completed.stderr[-4_000:],
        }
    if completed.returncode != 0 and result.get("status") == "complete":
        result["status"] = "failed"
    if completed.stderr:
        result["stderr"] = completed.stderr[-4_000:]
    return result


def run_restart_check(
    suite_id: str,
    *,
    scratch_root: Path = SCRATCH_ROOT,
    output_root: Path = OUTPUT_ROOT,
    lock_path: Path = LOCK_PATH,
) -> dict[str, object]:
    """Run reference, save, and resume phases in three sequential fresh processes."""

    _require_gpu_environment()
    validate_suite_id(suite_id)
    require_storage_layout(scratch_root, output_root)
    suite_directory = suite_path(suite_id, scratch_root)
    suite_directory.mkdir(parents=True, exist_ok=True)
    restart_dir = suite_directory / "restart"
    if restart_dir.exists() and any(restart_dir.iterdir()):
        raise FileExistsError(f"restart artifacts already exist: {restart_dir}")
    restart_dir.mkdir(parents=True, exist_ok=True)
    plan_path = restart_dir / "worker-plan.json"
    plan: dict[str, object] = {
        "scratch_root": str(scratch_root),
        "output_root": str(output_root),
        "lock_path": str(lock_path),
        "restart_dir": str(restart_dir),
    }
    # Each fresh worker acquires and releases the shared host-local lock before
    # the next worker starts, preventing concurrent Cambacica GPU processes.
    _atomic_write_bytes(plan_path, canonical_json(plan))
    phases: list[dict[str, object]] = []
    failure: dict[str, object] | None = None
    for phase in ("reference", "prepare"):
        result = _run_worker_subprocess(plan_path, phase)
        phases.append(result)
        if result.get("status") != "complete":
            failure = result
            break
        if phase == "reference":
            plan["expected_identity"] = result["identity"]
            plan["reference_state_path"] = result["reference_state_path"]
            plan["reference_state_sha256"] = result["reference_state_sha256"]
        else:
            plan["checkpoint_path"] = result["checkpoint_path"]
            _atomic_write_bytes(restart_dir / "prepare.json", canonical_json(result))
        _atomic_write_bytes(plan_path, canonical_json(plan))
    if failure is None:
        resume = _run_worker_subprocess(plan_path, "resume")
        phases.append(resume)
        if resume.get("status") != "complete":
            failure = resume
    failure_text = str(failure.get("error", "")) if failure else ""
    report = {
        "schema_version": "c3.6-restart-verification-v1",
        "suite_id": suite_id,
        "status": "failed" if failure else "complete",
        "deterministic_restart_status": "FAILED" if failure else "PASSED",
        "protocol": {
            "reference_updates": 4,
            "updates_before_save": 2,
            "updates_after_resume": 2,
            "fresh_process_phases": ["reference", "prepare", "resume"],
            "optimizer_fixture": OPTIMIZER_FIXTURE,
            "synthetic_only": True,
            "gpu_execution": True,
            "deterministic_kernel_settings": {
                "torch_use_deterministic_algorithms": True,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "cuda_matmul_allow_tf32": False,
                "attention_backend": "auto",
                "backend_fallback_or_switch": False,
            },
        },
        "phases": phases,
        "comparison": phases[-1]
        if phases and phases[-1].get("phase") == "resume"
        else None,
        "deterministic_kernel_incompatibility": (
            failure_text if failure and "determin" in failure_text.lower() else None
        ),
        "failure": failure,
        "git": git_provenance(),
        "implementation_sha256": implementation_sha256(),
    }
    _atomic_write_bytes(
        restart_dir / "restart_verification.json", canonical_json(report)
    )
    _atomic_write_bytes(
        suite_directory / "restart_verification.json", canonical_json(report)
    )
    return report


def _suite_files(suite_directory: Path) -> list[Path]:
    """Return only finalized reports; checkpoints and state snapshots stay local."""

    return sorted(
        path
        for path in suite_directory.rglob("*")
        if path.is_file()
        and path.name not in {"manifest.json", "checksums.sha256"}
        and ".tmp" not in path.name
        and (
            path.relative_to(suite_directory).as_posix()
            in {"matrix.json", "results.csv", "restart_verification.json"}
            or (
                path.relative_to(suite_directory).parts[0] == "cells"
                and path.name
                in {"configuration.json", "result.json", "checksums.sha256"}
            )
        )
    )


def finalize_suite(
    suite_id: str,
    *,
    scratch_root: Path = SCRATCH_ROOT,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    """Validate all nine cell artifacts and atomically publish an immutable suite."""

    validate_suite_id(suite_id)
    suite_directory = suite_path(suite_id, scratch_root)
    expected_cells = approved_cells()
    verified = []
    for cell in expected_cells:
        cell_dir = suite_directory / "cells" / cell.cell_id
        entry = verify_cell_artifacts(cell_dir)
        if entry["configuration"] != cell.to_dict():
            raise CheckpointIntegrityError(f"configuration drift in {cell.cell_id}")
        if entry["result"].get("status") != "complete":
            raise CheckpointIntegrityError(f"cell {cell.cell_id} did not complete")
        verified.append(entry)
    restart_path = suite_directory / "restart_verification.json"
    if not restart_path.is_file():
        raise CheckpointIntegrityError(
            "restart_verification.json is required before finalization"
        )
    restart_report = json.loads(restart_path.read_text(encoding="utf-8"))
    if restart_report.get("status") not in {"complete", "failed"}:
        raise CheckpointIntegrityError("restart verification has no terminal status")
    matrix = {
        "schema_version": "c3.6-matrix-v1",
        "suite_id": suite_id,
        "baseline_id": BASELINE_ID,
        "baseline_sha256": BASELINE_SHA256,
        "status": "complete" if restart_report["status"] == "complete" else "failed",
        "execution_order": [cell.cell_id for cell in expected_cells],
        "concurrent_cells": 1,
        "context_length": CONTEXT_LENGTH,
        "warmup_updates": WARMUP_UPDATES,
        "measured_updates": MEASURED_UPDATES,
        "repeats": REPEATS,
        "cells": [cell.to_dict() for cell in expected_cells],
    }
    _atomic_write_bytes(suite_directory / "matrix.json", canonical_json(matrix))
    columns = [
        "cell_id",
        "status",
        "microbatch_size",
        "gradient_accumulation_steps",
        "tokens_per_update",
        "mean_update_ms",
        "median_update_ms",
        "p95_update_ms",
        "wall_tokens_per_second",
        "peak_pytorch_allocated_mib",
        "peak_pytorch_reserved_mib",
        "sampled_cambacica_process_peak_mib",
        "startup_free_memory_mib",
        "minimum_free_memory_mib",
        "input_stream_sha256",
    ]
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=columns)
    writer.writeheader()
    for cell, entry in zip(expected_cells, verified, strict=True):
        result = entry["result"]
        timing = result["optimizer_update_latency_gpu_event_ms"]
        stream = result["synthetic_stream"]
        writer.writerow(
            {
                "cell_id": cell.cell_id,
                "status": result["status"],
                "microbatch_size": cell.microbatch_size,
                "gradient_accumulation_steps": cell.gradient_accumulation_steps,
                "tokens_per_update": cell.tokens_per_update,
                "mean_update_ms": timing["mean_ms"],
                "median_update_ms": timing["median_ms"],
                "p95_update_ms": timing["p95_ms"],
                "wall_tokens_per_second": result[
                    "wall_tokens_per_second_including_safety_checks"
                ],
                "peak_pytorch_allocated_mib": result["peak_pytorch_allocated_mib"],
                "peak_pytorch_reserved_mib": result["peak_pytorch_reserved_mib"],
                "sampled_cambacica_process_peak_mib": result[
                    "sampled_cambacica_process_peak_mib"
                ],
                "startup_free_memory_mib": result["startup_free_memory_mib"],
                "minimum_free_memory_mib": result["minimum_free_memory_mib"],
                "input_stream_sha256": stream["sha256"],
            }
        )
    _atomic_write_bytes(suite_directory / "results.csv", csv_buffer.getvalue().encode())
    all_files = _suite_files(suite_directory)
    file_hashes = {
        path.relative_to(suite_directory).as_posix(): sha256_file(path)
        for path in all_files
    }
    manifest = {
        "schema_version": "c3.6-suite-manifest-v1",
        "suite_id": suite_id,
        "status": matrix["status"],
        "baseline_id": BASELINE_ID,
        "baseline_sha256": BASELINE_SHA256,
        "implementation_sha256": implementation_sha256(),
        "artifact_count": len(file_hashes),
        "artifacts": file_hashes,
    }
    _atomic_write_bytes(suite_directory / "manifest.json", canonical_json(manifest))
    ledger_paths = [*all_files, suite_directory / "manifest.json"]
    ledger = "".join(
        f"{sha256_file(path)}  {path.relative_to(suite_directory).as_posix()}\n"
        for path in ledger_paths
    )
    _atomic_write_bytes(suite_directory / "checksums.sha256", ledger.encode("ascii"))
    verify_suite_artifacts(suite_directory)
    require_storage_layout(scratch_root, output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / suite_id
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite completed C3.6 suite: {destination}"
        )
    temporary = Path(tempfile.mkdtemp(prefix=f".{suite_id}.partial-", dir=output_root))
    try:
        publishable = [
            *_suite_files(suite_directory),
            suite_directory / "manifest.json",
            suite_directory / "checksums.sha256",
        ]
        for source in publishable:
            relative = source.relative_to(suite_directory)
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
        _fsync_directory(temporary)
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite completed C3.6 suite: {destination}"
            )
        os.rename(temporary, destination)
        _fsync_directory(output_root)
        return destination
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_suite_artifacts(suite_directory: Path) -> dict[str, object]:
    ledger: dict[str, str] = {}
    try:
        for line in (
            (suite_directory / "checksums.sha256")
            .read_text(encoding="ascii")
            .splitlines()
        ):
            digest, name = line.split("  ", 1)
            ledger[name] = digest
    except (OSError, ValueError) as exc:
        raise CheckpointIntegrityError(
            f"suite checksum ledger is invalid: {exc}"
        ) from exc
    if not ledger:
        raise CheckpointIntegrityError("suite checksum ledger is empty")
    expected_names = {
        path.relative_to(suite_directory).as_posix()
        for path in _suite_files(suite_directory)
    } | {"manifest.json"}
    if set(ledger) != expected_names:
        raise CheckpointIntegrityError(
            "suite checksum ledger does not cover exactly the finalized reports"
        )
    for name, digest in ledger.items():
        target = suite_directory / name
        if not target.is_file() or sha256_file(target) != digest:
            raise CheckpointIntegrityError(f"suite artifact checksum mismatch: {name}")
    manifest = json.loads(
        (suite_directory / "manifest.json").read_text(encoding="utf-8")
    )
    if set(manifest["artifacts"]) != expected_names - {"manifest.json"}:
        raise CheckpointIntegrityError("suite manifest artifact set is incomplete")
    for name, digest in manifest["artifacts"].items():
        if ledger.get(name) != digest:
            raise CheckpointIntegrityError(f"manifest checksum mismatch: {name}")
    return {"status": "valid", "artifact_count": len(ledger), "manifest": manifest}


def dry_run_plan(suite_id: str) -> dict[str, object]:
    """Return a CPU-only validated plan and the exact opt-in GPU commands."""

    validate_suite_id(suite_id)
    cells = approved_cells()
    commands = [
        "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /srv/cambacica-c3/venv/bin/python "
        "-m cambacica.benchmark.scalability_cli run-cell "
        f"--suite-id {suite_id} --microbatch-size {cell.microbatch_size} "
        f"--gradient-accumulation-steps {cell.gradient_accumulation_steps} --execute"
        for cell in cells
    ]
    return {
        "schema_version": "c3.6-dry-run-v1",
        "status": "validated_only",
        "baseline_id": BASELINE_ID,
        "baseline_sha256": BASELINE_SHA256,
        "gpu_execution": False,
        "production_corpus_opened": False,
        "matrix": [cell.to_dict() for cell in cells],
        "commands": commands,
        "execute_matrix_command": (
            "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "
            "/srv/cambacica-c3/venv/bin/python "
            "-m cambacica.benchmark.scalability_cli matrix "
            f"--suite-id {suite_id} --execute"
        ),
        "restart_command": (
            "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /srv/cambacica-c3/venv/bin/python "
            "-m cambacica.benchmark.scalability_cli restart-check "
            f"--suite-id {suite_id} --execute"
        ),
        "finalize_command": (
            "PYTHONPATH=src /srv/cambacica-c3/venv/bin/python "
            "-m cambacica.benchmark.scalability_cli finalize-suite "
            f"--suite-id {suite_id}"
        ),
        "memory_policy": {
            "hard_process_budget_mib": GPU_MEMORY_BUDGET_MIB,
            "process_peak_target_mib": PROCESS_PEAK_TARGET_MIB,
            "torch_allocator_limit_mib": TORCH_ALLOCATOR_LIMIT_MIB,
            "minimum_free_memory_mib": MINIMUM_FREE_MEMORY_MIB,
        },
    }
