from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is an optional C3 dependency")

from cambacica.benchmark.config import BenchmarkConfig  # noqa: E402
from cambacica.benchmark.runner import _synthetic_tokens  # noqa: E402


def test_synthetic_token_ids_are_seeded_and_context_shaped() -> None:
    config = BenchmarkConfig(architecture="B", context_length=2048, seed=91)
    first, first_hash = _synthetic_tokens(torch, config)
    second, second_hash = _synthetic_tokens(torch, config)
    assert first.device.type == "cpu"
    assert tuple(first.shape) == (1, 2049)
    assert torch.equal(first, second)
    assert first_hash == second_hash
