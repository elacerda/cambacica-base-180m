"""Command line interface for C1-BD2 artifacts and gated BD3 integration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .calibration import (
    DEFAULT_CALIBRATION_ROOT,
    run_calibration,
    verify_calibration,
)
from .production import (
    DEFAULT_CALIBRATION_ROOT as DEFAULT_BD3_CALIBRATION_ROOT,
    DEFAULT_EXACT_ROOT,
    DEFAULT_SCAN_ROOT,
    run_bd3_scan,
    verify_bd3_scan,
)
from .snapshot import DEFAULT_SNAPSHOT_ROOT, build_snapshot, verify_snapshot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cambacica.corpus.decontamination",
        description="Pinned benchmark snapshots and read-only contamination matcher controls.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot_parser = subparsers.add_parser(
        "snapshot",
        help="fetch only the approved pinned benchmark files and build the registry",
    )
    snapshot_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_SNAPSHOT_ROOT
    )
    snapshot_parser.add_argument("--cache-dir", type=Path)

    verify_snapshot_parser = subparsers.add_parser(
        "verify-snapshot",
        help="verify source checksums, row schema, IDs, and snapshot artifacts",
    )
    verify_snapshot_parser.add_argument(
        "--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_ROOT
    )

    calibration_parser = subparsers.add_parser(
        "calibrate",
        help="run separated deterministic fixtures and emit candidate-policy reports",
    )
    calibration_parser.add_argument(
        "--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_ROOT
    )
    calibration_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_CALIBRATION_ROOT
    )
    calibration_parser.add_argument("--scratch-dir", type=Path)

    verify_calibration_parser = subparsers.add_parser(
        "verify-calibration", help="verify calibration checksums and fixture schema"
    )
    verify_calibration_parser.add_argument(
        "--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_ROOT
    )

    scan_parser = subparsers.add_parser(
        "scan",
        help="BD3 preflight by default; production matching requires --execute-bd3 and approved policy",
    )
    scan_parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    scan_parser.add_argument(
        "--calibration-dir", type=Path, default=DEFAULT_BD3_CALIBRATION_ROOT
    )
    scan_parser.add_argument("--input-root", type=Path, default=DEFAULT_EXACT_ROOT)
    scan_parser.add_argument("--output-dir", type=Path, default=DEFAULT_SCAN_ROOT)
    scan_parser.add_argument("--scratch-dir", type=Path)
    scan_parser.add_argument("--policy", type=Path)
    scan_parser.add_argument(
        "--execute-bd3",
        action="store_true",
        help="explicitly run the full read-only production scan (never used by BD2)",
    )

    verify_scan_parser = subparsers.add_parser(
        "verify-scan", help="verify BD3 artifacts and per-shard accounting"
    )
    verify_scan_parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_ROOT)
    verify_scan_parser.add_argument(
        "--input-root", type=Path, default=DEFAULT_EXACT_ROOT
    )
    verify_scan_parser.add_argument(
        "--verify-inputs",
        action="store_true",
        help="read source shard IDs and independently recompute every occurrence-ID digest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "snapshot":
            result = build_snapshot(args.output_dir, args.cache_dir)
        elif args.command == "verify-snapshot":
            result = verify_snapshot(args.snapshot_dir)
        elif args.command == "calibrate":
            result = run_calibration(
                args.snapshot_dir, args.output_dir, args.scratch_dir
            )
        elif args.command == "verify-calibration":
            result = verify_calibration(args.calibration_dir)
        elif args.command == "scan":
            if args.execute_bd3 and args.policy is None:
                raise ValueError(
                    "--policy with scientist-approved metadata is required with --execute-bd3"
                )
            result = run_bd3_scan(
                snapshot_dir=args.snapshot_dir,
                calibration_dir=args.calibration_dir,
                policy_path=args.policy,
                input_root=args.input_root,
                output_dir=args.output_dir,
                scratch_dir=args.scratch_dir,
                execute_bd3=args.execute_bd3,
            )
        elif args.command == "verify-scan":
            result = verify_bd3_scan(args.scan_dir, args.input_root, args.verify_inputs)
        else:
            raise ValueError(f"Unknown command: {args.command}")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
