from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is an optional C3 dependency")

from cambacica.benchmark import scalability
from cambacica.benchmark.safety import GpuSafetyError
from cambacica.benchmark.scalability_cli import main as scalability_main


def test_approved_matrix_matches_baseline_and_fixes_safety_policy() -> None:
    cells = scalability.approved_cells()
    assert len(cells) == 9
    assert [cell.tokens_per_update for cell in cells] == [
        4096,
        8192,
        16384,
        8192,
        16384,
        32768,
        16384,
        32768,
        65536,
    ]
    policy = cells[-1].to_dict()["memory_policy"]
    assert policy == {
        "physical_gpu_index": 0,
        "gpu_uuid": "GPU-7562e34d-b0a2-a725-2ad5-ed1327be3cb0",
        "cuda_visible_devices": "0",
        "hard_process_budget_mib": 40_960,
        "process_peak_target_mib": 36_864,
        "torch_allocator_limit_mib": 35_840,
        "minimum_free_memory_mib": 4_096,
    }


def test_cell_validation_rejects_baseline_drift() -> None:
    with pytest.raises(scalability.ScalabilityValidationError, match="context length"):
        scalability.CellConfig("cell-m1-a1", 1, 1, context_length=2048).validate()
    with pytest.raises(
        scalability.ScalabilityValidationError, match="reference compute"
    ):
        scalability.CellConfig("cell-m1-a1", 1, 1, precision="fp32").validate()
    with pytest.raises(scalability.ScalabilityValidationError, match="3x3 matrix"):
        scalability.CellConfig("cell-m8-a1", 8, 1).validate()


def test_synthetic_stream_is_reproducible_and_rebatching_preserves_ids() -> None:
    first = scalability.SyntheticTokenStream(torch, sample_count=8)
    second = scalability.SyntheticTokenStream(torch, sample_count=8)
    one_by_two = first.take_microbatches(1, 2)
    two_by_one = second.take_microbatches(2, 1)
    assert first.sha256 == second.sha256
    assert torch.equal(torch.cat(one_by_two, dim=0), two_by_one[0])
    assert first.next_sample_offset == second.next_sample_offset == 2
    assert first.next_token_offset == 2 * scalability.CONTEXT_LENGTH
    first.seek(0, expected_sha256=second.sha256)
    assert first.next_sample_offset == 0
    with pytest.raises(scalability.CheckpointIntegrityError, match="hash mismatch"):
        first.seek(0, expected_sha256="0" * 64)


class _TinyLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.4, -0.3]))
        self.spec = SimpleNamespace(vocab_size=4)

    def forward(self, input_ids, labels):
        batch_size, sequence_length = input_ids.shape
        logits = self.bias.view(1, 1, 4).expand(batch_size, sequence_length, 4)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 4), labels.reshape(-1)
        )
        return logits, loss


class _CountingSGD:
    def __init__(self, parameters) -> None:
        self.parameters = list(parameters)
        self.step_count = 0

    def zero_grad(self, *, set_to_none: bool = False) -> None:
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    def step(self):
        self.step_count += 1
        with torch.no_grad():
            for parameter in self.parameters:
                parameter.add_(parameter.grad, alpha=-0.05)


def test_gradient_accumulation_matches_combined_batch_and_steps_once() -> None:
    combined_model = _TinyLM()
    accumulated_model = _TinyLM()
    accumulated_model.load_state_dict(combined_model.state_dict())
    combined_optimizer = _CountingSGD(combined_model.parameters())
    accumulated_optimizer = _CountingSGD(accumulated_model.parameters())
    token_ids = torch.randint(
        0, 4, (2, scalability.CONTEXT_LENGTH + 1), dtype=torch.long
    )

    combined_loss = scalability._optimizer_update(
        torch,
        model=combined_model,
        optimizer=combined_optimizer,
        microbatches=[token_ids],
        device="cpu",
    )
    accumulated_loss = scalability._optimizer_update(
        torch,
        model=accumulated_model,
        optimizer=accumulated_optimizer,
        microbatches=[token_ids[:1], token_ids[1:]],
        device="cpu",
    )
    assert combined_optimizer.step_count == 1
    assert accumulated_optimizer.step_count == 1
    assert combined_loss == pytest.approx(accumulated_loss, abs=1e-7)
    assert torch.allclose(
        combined_model.bias, accumulated_model.bias, atol=1e-7, rtol=1e-7
    )


def test_gpu_preflight_delegates_to_existing_pinned_safety_policy(monkeypatch) -> None:
    cell = scalability.approved_cells()[0]
    captured = {}

    def fake_preflight(torch_module, nvml, config):
        captured["config"] = config
        return SimpleNamespace(total_memory_mib=81_000, to_dict=dict), object()

    class FakeCuda:
        def device_count(self):
            return 1

        def is_bf16_supported(self):
            return False

    fake_torch = SimpleNamespace(cuda=FakeCuda())
    monkeypatch.setattr(scalability, "preflight_gpu", fake_preflight)
    with pytest.raises(GpuSafetyError, match="BF16 is not supported"):
        scalability._execute_cell(
            cell,
            torch_module=fake_torch,
            nvml=object(),
            scratch_root=Path("/tmp/c3-test-scratch"),
            output_root=Path("/tmp/c3-test-output"),
        )
    config = captured["config"]
    assert config.architecture == "C"
    assert config.context_length == 4096
    assert config.tie_weights is True
    assert config.gpu_memory_budget_mib == 40_960
    assert config.peak_target_mib == 36_864
    assert config.allocator_limit_mib == 35_840
    assert config.safety_margin_mib == 4_096


def test_cuda_environment_guard_refuses_any_other_mapping() -> None:
    scalability._require_gpu_environment({"CUDA_VISIBLE_DEVICES": "0"})
    with pytest.raises(GpuSafetyError, match="CUDA_VISIBLE_DEVICES=0"):
        scalability._require_gpu_environment({"CUDA_VISIBLE_DEVICES": "1"})
    with pytest.raises(GpuSafetyError, match="CUDA_VISIBLE_DEVICES=0"):
        scalability._require_gpu_environment({})


def test_failure_injection_stops_sequential_matrix_without_retry(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(scalability, "_require_gpu_environment", lambda: None)
    monkeypatch.setattr(scalability, "require_storage_layout", lambda *_: None)
    monkeypatch.setattr(scalability, "_torch_and_nvml", lambda: (object(), object()))
    calls = []

    def fail_preflight(cell, **_kwargs):
        calls.append(cell.cell_id)
        raise GpuSafetyError("injected UUID mismatch", diagnostics={"gpu": 0})

    monkeypatch.setattr(scalability, "_execute_cell", fail_preflight)
    result = scalability.run_matrix(
        "failure-injection",
        scratch_root=tmp_path / "scratch",
        output_root=tmp_path / "nfs",
        lock_path=tmp_path / "lock" / "c3.lock",
    )
    assert result["status"] == "failed"
    assert result["failed_cell"] == "cell-m1-a1"
    assert calls == ["cell-m1-a1"]
    failed = scalability.verify_cell_artifacts(
        tmp_path / "scratch/failure-injection/cells/cell-m1-a1"
    )
    assert failed["result"]["failure_kind"] == "gpu_safety_policy"
    assert failed["result"]["diagnostics"]["gpu"] == 0
    assert not (tmp_path / "scratch/failure-injection/cells/cell-m1-a2").exists()


def test_checkpoint_round_trip_validates_checksum_and_identity(tmp_path: Path) -> None:
    identity = {
        "baseline_id": scalability.BASELINE_ID,
        "code": "abc123",
        "synthetic_stream_sha256": "a" * 64,
        "synthetic_stream_seed": scalability.STREAM_SEED,
    }
    payload = {
        "schema_version": "c3.6-checkpoint-v1",
        "identity": identity,
        "model_state": {"weight": torch.arange(4, dtype=torch.float32)},
        "optimizer_state": {"step": 2, "moments": torch.ones(3)},
        "rng_states": {"python": (1, 2), "torch_cpu": torch.get_rng_state()},
        "update_index": 2,
        "synthetic_stream": {
            "seed": scalability.STREAM_SEED,
            "sha256": "a" * 64,
            "next_sample_offset": 2,
            "next_token_offset": 8192,
        },
    }
    path = scalability.write_checkpoint(
        tmp_path / "checkpoints", torch_module=torch, payload=payload
    )
    restored = scalability.load_checkpoint(
        path, torch_module=torch, expected_identity=identity
    )
    assert restored["update_index"] == 2
    assert torch.equal(
        restored["model_state"]["weight"], payload["model_state"]["weight"]
    )
    with pytest.raises(scalability.CheckpointIntegrityError, match="pinned identity"):
        scalability.load_checkpoint(
            path, torch_module=torch, expected_identity={"baseline_id": "other"}
        )
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(scalability.CheckpointIntegrityError, match="checksum mismatch"):
        scalability.load_checkpoint(
            path, torch_module=torch, expected_identity=identity
        )


def test_rng_capture_and_restore_preserves_cpu_python_and_numpy_states() -> None:
    import numpy

    random.seed(140)
    numpy.random.seed(240)
    torch.random.set_rng_state(
        torch.Generator(device="cpu").manual_seed(340).get_state()
    )
    captured = scalability._python_torch_numpy_rng_states(torch, include_cuda=False)
    python_expected = random.random()
    numpy_expected = float(numpy.random.random())
    torch_expected = torch.rand(1)

    random.random()
    numpy.random.random()
    torch.random.set_rng_state(
        torch.Generator(device="cpu").manual_seed(540).get_state()
    )
    scalability._restore_rng_states(torch, captured)

    assert random.random() == python_expected
    assert float(numpy.random.random()) == numpy_expected
    assert torch.equal(torch.rand(1), torch_expected)
    assert captured["torch_cuda"] is None


def test_restart_state_comparison_requires_bitwise_tensor_equality() -> None:
    expected = {"weight": torch.tensor([1.0, 0.0], dtype=torch.float32)}
    actual = {"weight": torch.tensor([1.01, -0.0], dtype=torch.float32)}
    report = scalability._state_comparison(torch, expected, actual)
    assert report["bitwise_equal"] is False
    assert report["mismatch_count"] == 1
    assert report["maximum_absolute_delta"] == pytest.approx(0.01, abs=1e-7)
    assert report["mismatches"][0]["path"] == "state.weight"


def test_cell_and_suite_artifact_integrity_detects_tampering(tmp_path: Path) -> None:
    cell = scalability.approved_cells()[0]
    cell_dir = tmp_path / "cells" / cell.cell_id
    scalability.write_cell_artifacts(
        cell_dir,
        configuration=cell.to_dict(),
        result={"cell_id": cell.cell_id, "status": "failed"},
    )
    assert scalability.verify_cell_artifacts(cell_dir)["result"]["status"] == "failed"
    with (cell_dir / "result.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(scalability.CheckpointIntegrityError, match="checksum mismatch"):
        scalability.verify_cell_artifacts(cell_dir)

    report = tmp_path / "suite"
    report.mkdir()
    active_checkpoint = report / "restart" / "checkpoints" / "checkpoint.pt"
    active_checkpoint.parent.mkdir(parents=True)
    active_checkpoint.write_bytes(b"local-only")
    matrix_path = report / "matrix.json"
    matrix_path.write_text("{}\n", encoding="utf-8")
    assert [path.name for path in scalability._suite_files(report)] == ["matrix.json"]
    matrix_hash = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "c3.6-suite-manifest-v1",
        "artifacts": {"matrix.json": matrix_hash},
    }
    manifest_path = report / "manifest.json"
    manifest_path.write_bytes(scalability.canonical_json(manifest))
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (report / "checksums.sha256").write_text(
        f"{matrix_hash}  matrix.json\n{manifest_hash}  manifest.json\n",
        encoding="ascii",
    )
    assert scalability.verify_suite_artifacts(report)["status"] == "valid"
    matrix_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(scalability.CheckpointIntegrityError, match="checksum mismatch"):
        scalability.verify_suite_artifacts(report)


def test_suite_finalization_publishes_only_reports_and_never_overwrites(
    monkeypatch, tmp_path: Path
) -> None:
    scratch = tmp_path / "scratch"
    output = tmp_path / "nfs"
    suite_id = "artifact-finalization"
    suite = scalability.suite_path(suite_id, scratch)
    for cell in scalability.approved_cells():
        timing = {
            "count": 20,
            "mean_ms": 1.0,
            "median_ms": 1.0,
            "p95_ms": 1.0,
        }
        result = {
            "cell_id": cell.cell_id,
            "status": "complete",
            "optimizer_update_latency_gpu_event_ms": timing,
            "wall_tokens_per_second_including_safety_checks": 4096.0,
            "peak_pytorch_allocated_mib": 100.0,
            "peak_pytorch_reserved_mib": 120.0,
            "sampled_cambacica_process_peak_mib": 150.0,
            "startup_free_memory_mib": 60_000,
            "minimum_free_memory_mib": 59_000,
            "synthetic_stream": {"sha256": "a" * 64},
        }
        scalability.write_cell_artifacts(
            suite / "cells" / cell.cell_id,
            configuration=cell.to_dict(),
            result=result,
        )
    (suite / "restart").mkdir(parents=True)
    (suite / "restart" / "checkpoint.pt").write_bytes(b"local checkpoint")
    (suite / "restart_verification.json").write_bytes(
        scalability.canonical_json({"status": "complete"})
    )
    monkeypatch.setattr(scalability, "require_storage_layout", lambda *_: None)

    destination = scalability.finalize_suite(
        suite_id, scratch_root=scratch, output_root=output
    )

    assert destination == output / suite_id
    assert (destination / "results.csv").is_file()
    assert (destination / "matrix.json").is_file()
    assert not (destination / "restart" / "checkpoint.pt").exists()
    assert scalability.verify_suite_artifacts(destination)["status"] == "valid"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        scalability.finalize_suite(suite_id, scratch_root=scratch, output_root=output)


def test_cli_dry_run_validates_plan_without_runtime_loading(
    monkeypatch, capsys
) -> None:
    def no_runtime():
        raise AssertionError("dry run must not load PyTorch or NVML")

    monkeypatch.setattr(scalability, "_torch_and_nvml", no_runtime)
    assert scalability_main(["matrix", "--suite-id", "cpu-dry-run", "--dry-run"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["gpu_execution"] is False
    assert output["production_corpus_opened"] is False
    assert len(output["commands"]) == 9
    assert all("--execute" in command for command in output["commands"])
