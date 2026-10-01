"""Command-line interface for Cambacica corpus sampling and inspection.

Provides commands to sample candidate sources deterministically, inspect sample
diagnostics, and evaluate cross-source duplicate overlaps for Gate C1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional
import yaml

from cambacica.corpus.dedup.exact import find_cross_source_exact_duplicates
from cambacica.corpus.dedup.minhash import (
    MinHashConfig,
    find_minhash_near_duplicates,
)
from cambacica.corpus.metrics import format_report_text, inspect_sample_file
from cambacica.corpus.sources import SAMPLER_REGISTRY
from cambacica.corpus.sources.base import ensure_user_hf_cache

# Ensure cache is writable before any dataset or HF Hub call
ensure_user_hf_cache()


def load_yaml_config(config_path: Path | str) -> dict:
    """Load configuration from a YAML file.

    Parameters
    ----------
    config_path : Path or str
        Path to YAML file.

    Returns
    -------
    dict
        Parsed YAML dictionary.
    """
    path = Path(config_path)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def handle_sample(args: argparse.Namespace) -> int:
    """Handle the 'sample' subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    source_name = args.source.lower().replace("-", "_")
    sampler_cls = SAMPLER_REGISTRY.get(source_name)
    if not sampler_cls:
        print(f"[ERROR] Unknown source: '{args.source}'.")
        print(f"Supported sources: {', '.join(sorted(SAMPLER_REGISTRY.keys()))}")
        return 1

    config_data = load_yaml_config(args.config)
    source_cfg = config_data.get("sources", {}).get(source_name, {})

    # Determine default mode and size from config if not provided
    mode = args.mode
    if not mode:
        modes_dict = source_cfg.get("modes", {})
        if "representative" in modes_dict:
            mode = "representative"
        elif "candidate" in modes_dict:
            mode = "candidate"
        else:
            mode = list(modes_dict.keys())[0] if modes_dict else "representative"

    size = args.size
    if size is None:
        size = source_cfg.get("modes", {}).get(mode, {}).get("default_size", 1000)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("data/samples/gate_c1") / source_name

    print(f"--> Starting sampling for '{source_name}'")
    print(f"    Mode: {mode}")
    print(f"    Target Size: {size:,}")
    print(f"    Seed: {args.seed}")
    print(f"    Output Directory: {out_dir}")

    sampler = sampler_cls(config=source_cfg)

    # Dry-run / safety visibility mode
    if getattr(args, "dry_run", False):
        kwargs = {}
        if hasattr(args, "exclusions"):
            kwargs["exclusions_path"] = args.exclusions
        if hasattr(args, "min_length"):
            kwargs["min_length"] = args.min_length
        plan = sampler.plan(mode=mode, size=size, seed=args.seed, **kwargs)
        print("\n=== DRY RUN / SAFETY PLANNING REVIEW ===")
        print(f"Source:                {plan.get('source')}")
        print(f"Sampling Mode:         {plan.get('mode')}")
        print(f"Requested Sample Size: {plan.get('target_size'):,}")
        print(f"Upstream Identifier:   {plan.get('upstream_identifier')}")
        print(f"Human Revision/Config: {plan.get('upstream_revision')}")
        print(f"Immutable Commit SHA:  {plan.get('upstream_commit_sha') or 'N/A'}")
        print(f"Population Scope:      {plan.get('population_scope')}")
        print(f"Sampling Frame:        {plan.get('sampling_frame')}")
        print(f"Safety Limits:         {plan.get('safety_limits')}")
        print(f"Selected Partitions:   {plan.get('selected_partitions')}")
        print(f"Expected Transfer:     {plan.get('estimated_transfer')}")
        print("=========================================\n")
        print("[DRY-RUN] No downloads performed. Safety limits and revisions verified.")
        return 0

    try:
        sample_kwargs = {"exclusions_path": args.exclusions}
        if hasattr(args, "min_length"):
            sample_kwargs["min_length"] = args.min_length

        parquet_path, manifest = sampler.sample(
            mode=mode,
            size=size,
            seed=args.seed,
            output_dir=out_dir,
            **sample_kwargs,
        )
        manifest_file = parquet_path.parent / f"manifest_{mode}.json"
        print("\n[OK] Sample generated successfully!")
        print(f"     Parquet:  {parquet_path} ({parquet_path.stat().st_size:,} bytes)")
        print(f"     Manifest: {manifest_file}")
        print(f"     Documents collected: {manifest.document_count:,}")
        underfill = manifest.stats.get("underfill")
        if underfill:
            print(
                "\n[WARNING] Underfill detected: "
                + ", ".join(f"{k}={v}" for k, v in underfill.items())
            )
        return 0
    except Exception as e:
        print(f"\n[ERROR] Failed to sample '{source_name}': {e}", file=sys.stderr)
        return 1


def find_sample_parquet_files(target: Path) -> List[Path]:
    """Discover Parquet sample files, ignoring test and smoke directories.

    Parameters
    ----------
    target : Path
        Path to Parquet file or directory.

    Returns
    -------
    list of Path
        Sorted list of discovered Parquet files.
    """
    if target.is_file() and target.suffix == ".parquet":
        return [target]
    if not target.is_dir():
        return []

    found = []
    for p in target.glob("**/*.parquet"):
        parts = [part.lower() for part in p.parts]
        if any(
            part in ("tests", "test", "smoke", "temp", "tmp", "scratch")
            or part.startswith("smoke_")
            or part.startswith("test_")
            or part.startswith("tmp_")
            for part in parts
        ):
            continue
        if (
            len(p.parts) >= 3
            and p.parent.name in ("audit", "candidate")
            and p.parent.parent.name == "gigaverbo_v2"
        ):
            continue
        found.append(p)
    return sorted(found)


def handle_inspect(args: argparse.Namespace) -> int:
    """Handle the 'inspect' subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code.
    """
    target = Path(args.path)
    parquet_files = find_sample_parquet_files(target)

    if not parquet_files:
        if not target.exists():
            print(f"[ERROR] Target path '{target}' is not a Parquet file or directory.")
            return 1
        print(f"[WARNING] No .parquet files found in '{target}'.")
        return 0

    reports = []
    for pf in parquet_files:
        try:
            report = inspect_sample_file(pf)
            reports.append(report)
            if args.json:
                print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
            else:
                print(format_report_text(report))
                print("\n" + "=" * 70 + "\n")
        except Exception as e:
            print(f"[ERROR] Failed inspecting {pf}: {e}", file=sys.stderr)

    return 0


def handle_compare(args: argparse.Namespace) -> int:
    """Handle the 'compare' subcommand for cross-source duplicates.

    Classifies duplicates into three relationship types:

    A. WITHIN_FILE_DUPLICATES
       Exact duplicate text occurring multiple times inside one Parquet sample.
    B. SAME_SOURCE_CROSS_MODE_OVERLAP
       Same content in multiple files/modes for the same upstream source.
       Expected and not a contamination signal.
    C. CROSS_SOURCE_DUPLICATES
       Same content in genuinely different upstream sources.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code.
    """
    target_dir = Path(args.directory)
    if not target_dir.is_dir():
        print(f"[ERROR] Target '{target_dir}' is not a directory.")
        return 1

    parquet_files = find_sample_parquet_files(target_dir)
    if len(parquet_files) < 2:
        print(
            f"[WARNING] Need at least 2 Parquet files to compare, "
            f"found {len(parquet_files)}."
        )
        return 0

    print(f"--> Comparing {len(parquet_files)} sample files for exact duplicates...")
    exact_report = find_cross_source_exact_duplicates(parquet_files)

    print("\n=== Exact Duplicate Report (Classified by Relationship) ===")
    print(f"Total Unique Hashes:              {exact_report.total_unique_hashes:,}")
    print(f"Total Duplicate Hashes:           {exact_report.total_duplicate_hashes:,}")
    print(
        f"Total Redundant Doc Occurrences:  {exact_report.total_duplicate_documents:,}"
    )

    print("\n--- A. WITHIN_FILE_DUPLICATES ---")
    print(
        "  Exact duplicate text appearing multiple times inside one Parquet file.\n"
        "  These are genuine corpus-internal repetitions."
    )
    if exact_report.within_file_counts:
        for src, count in sorted(exact_report.within_file_counts.items()):
            print(f"  {src}: {count:,} within-file duplicate document(s)")
    else:
        print("  None detected.")

    print("\n--- B. SAME_SOURCE_CROSS_MODE_OVERLAP ---")
    print(
        "  Same content in multiple files/modes for the same upstream source.\n"
        "  Expected when audit vs candidate or representative vs diagnostic overlap.\n"
        "  This is NOT a corpus contamination signal."
    )
    if exact_report.same_source_cross_mode_overlap:
        for src, count in sorted(exact_report.same_source_cross_mode_overlap.items()):
            print(f"  {src}: {count:,} shared hash(es) across modes")
    else:
        print("  None detected.")

    print("\n--- C. CROSS_SOURCE_DUPLICATES ---")
    print(
        "  Same content appearing in genuinely different upstream sources.\n"
        "  These are actual corpus-level contamination candidates."
    )
    if exact_report.cross_source_pair_counts:
        for pair, count in sorted(exact_report.cross_source_pair_counts.items()):
            print(f"  {pair}: {count:,} shared document(s)")
    else:
        print("  None detected across tested samples.")

    if args.minhash:
        print(
            f"\n--> Running MinHash near-duplicate diagnostics "
            f"(threshold >= {args.threshold})..."
        )
        import pyarrow.parquet as pq

        docs_to_compare = []
        for pf in parquet_files:
            tbl = pq.read_table(pf, columns=["text", "source", "original_id"])
            for row in tbl.to_pylist()[:500]:  # Limit for diagnostic speed
                row["file_path"] = str(pf)
                docs_to_compare.append(row)

        config = MinHashConfig(num_permutations=64, ngram_size=5)
        candidates = find_minhash_near_duplicates(
            docs_to_compare, config, threshold=args.threshold, max_candidates=20
        )

        cross_source_near = [c for c in candidates if c.relationship == "CROSS_SOURCE"]
        cross_mode_near = [
            c for c in candidates if c.relationship == "SAME_SOURCE_CROSS_MODE"
        ]
        within_file_near = [c for c in candidates if c.relationship == "WITHIN_FILE"]

        print(f"\n--- D. NEAR_DUPLICATES (MinHash Jaccard >= {args.threshold}) ---")

        print("\n  [D1] CROSS_SOURCE Near-Duplicates:")
        print("       (Genuine cross-corpus similarity candidates)")
        if cross_source_near:
            print(
                f"       Found {len(cross_source_near)} cross-source candidate pair(s):"
            )
            for c in cross_source_near:
                print(
                    f"       [{c.source_1}:{c.doc_id_1}] <-> [{c.source_2}:{c.doc_id_2}] "
                    f"Jaccard={c.estimated_jaccard:.2f}"
                )
                print(f"         1: {c.snippet_1}")
                print(f"         2: {c.snippet_2}")
        else:
            print("       None detected (0 pairs across evaluated samples).")

        print("\n  [D2] SAME_SOURCE_CROSS_MODE Near-Duplicates:")
        print("       (Expected cross-mode overlaps between sample configurations)")
        if cross_mode_near:
            print(f"       Found {len(cross_mode_near)} cross-mode pair(s):")
            for c in cross_mode_near[:10]:
                print(
                    f"       [{c.source_1}:{c.doc_id_1}] <-> [{c.source_2}:{c.doc_id_2}] "
                    f"Jaccard={c.estimated_jaccard:.2f}"
                )
        else:
            print("       None detected.")

        print("\n  [D3] WITHIN_FILE Near-Duplicates:")
        print("       (Repeated or highly similar documents inside a single sample)")
        if within_file_near:
            print(f"       Found {len(within_file_near)} within-file pair(s):")
            for c in within_file_near[:10]:
                print(
                    f"       [{c.source_1}:{c.doc_id_1}] <-> [{c.source_2}:{c.doc_id_2}] "
                    f"Jaccard={c.estimated_jaccard:.2f}"
                )
        else:
            print("       None detected.")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured CLI parser.
    """
    parser = argparse.ArgumentParser(
        prog="cambacica.corpus",
        description="Gate C1 (CORPUS) sampling and inspection pipeline for Cambacica.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: sample
    sample_parser = subparsers.add_parser(
        "sample", help="Sample a candidate corpus source."
    )
    sample_parser.add_argument(
        "source",
        choices=[
            "carolina",
            "wikipedia_pt",
            "parlamento_pt",
            "gigaverbo_v2",
            "gutenberg_pt",
        ],
        help="Source identifier to sample.",
    )
    sample_parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Sample mode ('representative', 'diagnostic', 'audit', 'candidate').",
    )
    sample_parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Number of documents to sample.",
    )
    sample_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed (default: 42).",
    )
    sample_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save Parquet and manifest.",
    )
    sample_parser.add_argument(
        "--config",
        type=str,
        default="configs/corpus_sources.yaml",
        help="Path to corpus_sources.yaml.",
    )
    sample_parser.add_argument(
        "--exclusions",
        type=str,
        default="configs/gigaverbo_exclusions.yaml",
        help="Path to gigaverbo_exclusions.yaml.",
    )
    sample_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Output sampling plan, resolved revisions, safety limits, and "
            "estimated transfer without downloading."
        ),
    )
    sample_parser.add_argument(
        "--min-length",
        type=int,
        default=0,
        help=(
            "Optional minimum character filter (disabled by default; "
            "0 keeps all non-empty records)."
        ),
    )

    # Subcommand: inspect
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect diagnostic metrics for a sample Parquet file or directory.",
    )
    inspect_parser.add_argument(
        "path",
        type=str,
        help="Path to sample Parquet file or directory containing samples.",
    )
    inspect_parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON report.",
    )

    # Subcommand: compare
    compare_parser = subparsers.add_parser(
        "compare", help="Compare multiple sample files for exact and near duplicates."
    )
    compare_parser.add_argument(
        "directory",
        type=str,
        help="Directory containing Parquet sample files to compare.",
    )
    compare_parser.add_argument(
        "--minhash",
        action="store_true",
        help="Run MinHash near-duplicate candidate detection.",
    )
    compare_parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="MinHash Jaccard threshold (default: 0.80).",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint.

    Parameters
    ----------
    argv : list of str or None, optional
        Command line arguments.

    Returns
    -------
    int
        Exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "sample":
        return handle_sample(args)
    elif args.subcommand == "inspect":
        return handle_inspect(args)
    elif args.subcommand == "compare":
        return handle_compare(args)
    return 1


if __name__ == "__main__":
    import os

    sys.stdout.flush()
    sys.stderr.flush()
    ret = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(ret)
