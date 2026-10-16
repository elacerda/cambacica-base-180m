"""Opt-in C3.6 scalability runner CLI; planning and validation stay CPU-only."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from cambacica.benchmark.scalability import (
    CellConfig,
    _restart_worker,
    approved_cells,
    dry_run_plan,
    finalize_suite,
    run_matrix,
    run_one_cell,
    run_restart_check,
    validate_suite_id,
)


def _execution_choice(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and report the plan without importing PyTorch or accessing CUDA",
    )
    group.add_argument(
        "--execute",
        action="store_true",
        help="execute the explicit GPU workload on the approved physical GPU 0",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cambacica-c3-scalability",
        description=(
            "Validate, run, verify, and publish the approved C3.6 synthetic "
            "scalability suite. GPU execution requires --execute."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    matrix = subparsers.add_parser(
        "matrix", help="validate all nine cells or execute them sequentially"
    )
    matrix.add_argument("--suite-id", required=True)
    _execution_choice(matrix)

    cell = subparsers.add_parser(
        "run-cell", help="execute one approved cell into the named local suite"
    )
    cell.add_argument("--suite-id", required=True)
    cell.add_argument("--microbatch-size", type=int, choices=(1, 2, 4), required=True)
    cell.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        choices=(1, 2, 4),
        required=True,
    )
    _execution_choice(cell)

    restart = subparsers.add_parser(
        "restart-check", help="run the fresh-process 4 versus 2+save+resume+2 check"
    )
    restart.add_argument("--suite-id", required=True)
    _execution_choice(restart)

    finalize = subparsers.add_parser(
        "finalize-suite", help="verify and immutably publish a completed suite to NFS"
    )
    finalize.add_argument("--suite-id", required=True)

    worker = subparsers.add_parser("_restart-worker", help=argparse.SUPPRESS)
    worker.add_argument("--execute", action="store_true", required=True)
    worker.add_argument("--plan", type=Path, required=True)
    worker.add_argument(
        "--phase", choices=("reference", "prepare", "resume"), required=True
    )
    return parser


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "matrix":
            validate_suite_id(args.suite_id)
            if args.dry_run:
                result = dry_run_plan(args.suite_id)
            else:
                result = run_matrix(args.suite_id)
        elif args.command == "run-cell":
            validate_suite_id(args.suite_id)
            CellConfig(
                cell_id=(
                    f"cell-m{args.microbatch_size}-a{args.gradient_accumulation_steps}"
                ),
                microbatch_size=args.microbatch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            ).validate()
            if args.dry_run:
                result = {
                    "status": "validated_only",
                    "gpu_execution": False,
                    "configuration": next(
                        cell.to_dict()
                        for cell in approved_cells()
                        if cell.microbatch_size == args.microbatch_size
                        and cell.gradient_accumulation_steps
                        == args.gradient_accumulation_steps
                    ),
                }
            else:
                result = run_one_cell(
                    args.suite_id,
                    args.microbatch_size,
                    args.gradient_accumulation_steps,
                )
        elif args.command == "restart-check":
            validate_suite_id(args.suite_id)
            if args.dry_run:
                result = {
                    "status": "validated_only",
                    "gpu_execution": False,
                    "suite_id": args.suite_id,
                    "protocol": "four uninterrupted updates versus two, save, fresh-process resume, two",
                    "optimizer_fixture": "C3.4 synthetic AdamW fixture",
                }
            else:
                result = run_restart_check(args.suite_id)
        elif args.command == "finalize-suite":
            validate_suite_id(args.suite_id)
            destination = finalize_suite(args.suite_id)
            result = {
                "status": "published",
                "suite_id": args.suite_id,
                "artifact_path": str(destination),
                "gpu_execution": False,
            }
        else:
            result = _restart_worker(args.plan, args.phase)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - report CLI failures
        result = {
            "status": "failed",
            "failure_kind": type(exc).__name__,
            "error": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
            "gpu_execution": args.command
            in {"matrix", "run-cell", "restart-check", "_restart-worker"}
            and bool(getattr(args, "execute", False)),
        }
    _print(result)
    return (
        0 if result.get("status") in {"validated_only", "complete", "published"} else 1
    )


if __name__ == "__main__":
    sys.exit(main())
