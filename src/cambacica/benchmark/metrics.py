"""Small deterministic metric helpers used by C3 and its CPU-only tests."""

from __future__ import annotations

import math
from statistics import mean
from typing import Iterable


def nearest_rank(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def summarize_step_times(step_times_ms: Iterable[float]) -> dict[str, float | int | None]:
    values = [float(value) for value in step_times_ms]
    if not values:
        raise ValueError("at least one timed step is required")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("step times must be finite and non-negative")
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {
        "count": len(values),
        "mean_ms": mean(values),
        "median_ms": median,
        "p95_ms": nearest_rank(values, 0.95),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def tokens_per_second(tokens_per_step: int, timed_steps: int, wall_seconds: float) -> float:
    if tokens_per_step <= 0 or timed_steps <= 0 or wall_seconds <= 0:
        raise ValueError("token count, step count, and wall time must be positive")
    return tokens_per_step * timed_steps / wall_seconds
