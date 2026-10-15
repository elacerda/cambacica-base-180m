from __future__ import annotations

import json

import pytest

from cambacica.benchmark.cli import parse_config
from cambacica.benchmark.config import BenchmarkConfig
from cambacica.benchmark.metrics import nearest_rank, summarize_step_times, tokens_per_second


def command(*extra: str) -> list[str]:
    return [
        "--architecture",
        "A",
        "--context-length",
        "2048",
        "--tie-weights",
        "tied",
        *extra,
    ]


def test_cli_defaults_match_primary_synthetic_methodology() -> None:
    config = parse_config(command())
    assert config.precision == "bf16"
    assert config.microbatch_size == 1
    assert config.warmup_steps == 20
    assert config.timed_steps == 100
    assert config.attention_backend == "auto"
    assert config.tie_weights is True
    assert config.activation_checkpointing == "none"
    assert config.gpu_memory_budget_mib == 40_960
    assert config.peak_target_mib == 36_864
    assert config.allocator_limit_mib == 35_840
    assert config.tokens_per_step == 2_048


@pytest.mark.parametrize(
    "extra, message",
    [
        (("--device", "cuda:1"), "restricted to cuda:0"),
        (("--gpu-memory-budget-mib", "40961"), "no greater than 40960"),
        (("--safety-margin-mib", "4095"), "cannot be smaller than 4096"),
        (("--timed-steps", "0"), "timed steps must be positive"),
        (("--activation-checkpointing", "full"), "invalid choice"),
    ],
)
def test_cli_rejects_invalid_execution_configuration(extra, message: str) -> None:
    with pytest.raises(SystemExit):
        parse_config(command(*extra))


def test_8192_context_requires_explicit_capacity_flag() -> None:
    with pytest.raises(ValueError, match="require --enable-8192-capacity-test"):
        BenchmarkConfig(architecture="B", context_length=8192).validate()
    config = parse_config(
        [
            "--architecture",
            "B",
            "--context-length",
            "8192",
            "--tie-weights",
            "untied",
            "--enable-8192-capacity-test",
        ]
    )
    assert config.context_length == 8192


def test_config_serializes_all_benchmark_identity_and_policy() -> None:
    config = parse_config(command("--attention-backend", "math", "--tie-weights", "tied"))
    encoded = json.dumps(config.to_dict(), sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["architecture"] == "A"
    assert decoded["attention_backend"] == "math"
    assert decoded["memory_policy"]["approved_hard_budget_mib"] == 40_960
    assert decoded["memory_policy"]["allocator_is_hardware_isolation"] is False
    assert decoded["tokenizer_identity"]["model_sha256"].startswith("c8982a23")


def test_step_metrics_use_median_and_nearest_rank_p95() -> None:
    assert nearest_rank([5, 1, 3, 2, 4], 0.5) == 3.0
    assert nearest_rank([5, 1, 3, 2, 4], 0.95) == 5.0
    assert summarize_step_times([1, 2, 3, 4]) == {
        "count": 4,
        "mean_ms": 2.5,
        "median_ms": 2.5,
        "p95_ms": 4.0,
        "min_ms": 1.0,
        "max_ms": 4.0,
    }
    assert tokens_per_second(2048, 100, 20) == 10_240
