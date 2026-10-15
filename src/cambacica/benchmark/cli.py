"""Command-line interface for explicit, bounded C3 GPU benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Sequence

from cambacica.benchmark.config import (
    DEFAULT_SAFETY_MARGIN_MIB,
    HARD_GPU_MEMORY_BUDGET_MIB,
    BenchmarkConfig,
)
from cambacica.benchmark.runner import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cambacica-benchmark",
        description=(
            "Run one synthetic C3 model benchmark on the approved physical GPU 0. "
            "The command fails closed when identity or memory preflight fails."
        ),
    )
    parser.add_argument("--architecture", required=True, choices=("A", "B", "C"))
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--timed-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument(
        "--attention-backend",
        choices=("auto", "flash", "efficient", "math"),
        default="auto",
    )
    parser.add_argument(
        "--gpu-memory-budget-mib",
        type=int,
        default=HARD_GPU_MEMORY_BUDGET_MIB,
        help="hard Cambacica process budget; values above 40960 MiB are rejected",
    )
    parser.add_argument(
        "--safety-margin-mib",
        type=int,
        default=DEFAULT_SAFETY_MARGIN_MIB,
        help="minimum reserved margin; cannot be smaller than 4096 MiB",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/data/cambacica-base-180m/c3-benchmarks"),
    )
    parser.add_argument("--tie-weights", choices=("tied", "untied"), required=True)
    parser.add_argument(
        "--activation-checkpointing",
        choices=("none",),
        default="none",
        help="activation checkpointing is not implemented in C3 v1",
    )
    parser.add_argument(
        "--enable-8192-capacity-test",
        action="store_true",
        help="separate explicit authorization after review of 4096-token results",
    )
    return parser


def parse_config(argv: Sequence[str] | None = None) -> BenchmarkConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = BenchmarkConfig(
        architecture=args.architecture,
        context_length=args.context_length,
        device=args.device,
        precision=args.precision,
        microbatch_size=args.microbatch_size,
        warmup_steps=args.warmup_steps,
        timed_steps=args.timed_steps,
        seed=args.seed,
        attention_backend=args.attention_backend,
        gpu_memory_budget_mib=args.gpu_memory_budget_mib,
        safety_margin_mib=args.safety_margin_mib,
        output_root=args.output_root,
        tie_weights=args.tie_weights == "tied",
        activation_checkpointing=args.activation_checkpointing,
        enable_8192_capacity_test=args.enable_8192_capacity_test,
    )
    try:
        return config.validate()
    except ValueError as exc:
        parser.error(str(exc))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        config = parse_config(argv)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        result = run_benchmark(config)
    except Exception as exc:
        result = {
            "status": "failed",
            "failure_kind": "artifact_or_startup_error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc)),
            },
            "exit_code": 1,
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return int(result["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
