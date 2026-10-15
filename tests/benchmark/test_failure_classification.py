from __future__ import annotations

from cambacica.benchmark.runner import _classify_failure
from cambacica.benchmark.safety import GpuSafetyError


def test_oom_numerical_backend_and_safety_failures_are_distinguished() -> None:
    assert _classify_failure(RuntimeError("CUDA out of memory")) == "cuda_oom"
    assert _classify_failure(GpuSafetyError("non-finite gradients detected")) == "numerical_failure"
    assert _classify_failure(RuntimeError("No available kernel for requested backend")) == "incompatible_attention_backend"
    assert _classify_failure(GpuSafetyError("UUID mismatch")) == "gpu_safety_policy"
