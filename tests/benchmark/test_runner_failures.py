from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from cambacica.benchmark import runner
from cambacica.benchmark.config import BenchmarkConfig
from cambacica.benchmark.safety import GpuSafetyError


def test_missing_runtime_dependency_is_reported_without_gpu_or_overwrite(
    monkeypatch, tmp_path: Path
) -> None:
    scratch = tmp_path / "local-scratch"
    output = tmp_path / "nfs-output"
    config = BenchmarkConfig(
        architecture="A",
        context_length=2048,
        output_root=output,
        scratch_root=scratch,
        lock_path=tmp_path / "lock" / "c3.lock",
    )
    monkeypatch.setattr(runner, "require_storage_layout", lambda *_: None)

    def missing_runtime():
        raise RuntimeError("optional PyTorch is unavailable")

    monkeypatch.setattr(runner, "_torch_and_nvml", missing_runtime)
    result = runner.run_benchmark(config)
    assert result["status"] == "failed"
    assert result["exit_code"] == 1
    assert result["failure_kind"] == "benchmark_error"
    artifact_path = Path(result["artifact_path"])
    assert artifact_path.parent == output / "architecture-A" / "context-2048"
    on_disk = json.loads((artifact_path / "result.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "failed"
    assert on_disk["error"]["message"] == "optional PyTorch is unavailable"
    assert (artifact_path / "checksums.sha256").is_file()


def test_oom_and_numerical_errors_publish_failure_without_retry(
    monkeypatch, tmp_path: Path
) -> None:
    torch_stub = SimpleNamespace(
        __version__="2.3.test",
        version=SimpleNamespace(cuda="12.test"),
    )
    nvml_stub = object()
    monkeypatch.setattr(runner, "require_storage_layout", lambda *_: None)
    monkeypatch.setattr(runner, "_torch_and_nvml", lambda: (torch_stub, nvml_stub))
    calls = []

    def oom(*_args, **_kwargs):
        calls.append("oom")
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(runner, "_execute_benchmark", oom)
    config = BenchmarkConfig(
        architecture="B",
        context_length=4096,
        output_root=tmp_path / "nfs-output",
        scratch_root=tmp_path / "local-scratch",
        lock_path=tmp_path / "lock" / "c3.lock",
    )
    oom_result = runner.run_benchmark(config)
    assert oom_result["failure_kind"] == "cuda_oom"
    assert calls == ["oom"]
    oom_artifact = json.loads(
        (Path(oom_result["artifact_path"]) / "result.json").read_text(encoding="utf-8")
    )
    assert oom_artifact["error"]["message"] == "CUDA out of memory"

    def nonfinite(*_args, **_kwargs):
        calls.append("nonfinite")
        raise GpuSafetyError("non-finite loss detected")

    monkeypatch.setattr(runner, "_execute_benchmark", nonfinite)
    second_config = BenchmarkConfig(
        architecture="C",
        context_length=2048,
        output_root=tmp_path / "nfs-output",
        scratch_root=tmp_path / "local-scratch",
        lock_path=tmp_path / "lock" / "c3.lock",
    )
    numerical_result = runner.run_benchmark(second_config)
    assert numerical_result["failure_kind"] == "numerical_failure"
    assert calls == ["oom", "nonfinite"]
