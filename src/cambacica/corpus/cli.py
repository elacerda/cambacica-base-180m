"""Command-line interface for Cambacica corpus sampling and inspection.

Provides commands to sample candidate sources deterministically, inspect sample
diagnostics, evaluate cross-source duplicate overlaps, normalize frozen Gate C1
source pools, and characterize normalized word volumes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional
import yaml

from cambacica.corpus.dedup.exact import find_cross_source_exact_duplicates
from cambacica.corpus.dedup.exact_pipeline import (
    DEFAULT_EXACT_ROOT,
    DEFAULT_PILOT_SEED,
    DEFAULT_PILOT_SIZE,
    audit_occurrence_ids,
    build_exact_dedup,
    run_exact_dedup_pilot,
    verify_exact_dedup,
)
from cambacica.corpus.dedup.minhash import (
    MinHashConfig,
    find_minhash_near_duplicates,
)
from cambacica.corpus.dedup.near_pilot import (
    DEFAULT_EXACT_DATA_ROOT,
    DEFAULT_NEAR_PILOT_ROOT,
    DEFAULT_NEAR_PILOT_SEED,
    run_near_dedup_pilot,
)
from cambacica.corpus.dedup.near_calibration import (
    DEFAULT_BASE_PILOT_ROOT,
    DEFAULT_CALIBRATION_ROOT,
    run_near_dedup_calibration,
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


def handle_validate_mixes(args: argparse.Namespace) -> int:
    """Handle the 'validate-mixes' subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    from cambacica.corpus.mix import MixValidationError, validate_mix_files

    files = args.files or [
        "configs/corpus_mix_a.yaml",
        "configs/corpus_mix_b.yaml",
        "configs/corpus_mix_c.yaml",
    ]
    print(
        f"--> Validating {len(files)} Gate C1 candidate corpus mixture configuration(s)..."
    )
    try:
        validated_configs = validate_mix_files(files)
        for path_str, cfg in zip(files, validated_configs):
            name = cfg.get("name", "unknown")
            label = cfg.get("label", "unknown")
            sources = cfg.get("sources", {})
            shares = {k: v.get("share") for k, v in sources.items()}
            print(f"  [PASS] {path_str} ({name} — {label})")
            for src_k, share_v in shares.items():
                print(f"         - {src_k}: {share_v:.2%}")
        print("--> All mixture configurations passed Gate C1 scientific validation.")
        return 0
    except MixValidationError as err:
        print(f"[ERROR] Mix validation failed: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"[ERROR] Unexpected error during mix validation: {err}", file=sys.stderr)


def _require_clean_normalization_commit() -> None:
    """Refuse production normalization until its implementation is committed."""
    import subprocess
    from pathlib import Path

    repository_root = Path(__file__).resolve().parents[3]
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise RuntimeError(
            "Production normalization requires a clean worktree so its tool commit is frozen."
        )


def handle_normalize(args: argparse.Namespace) -> int:
    """Normalize one pinned raw source or verify a completed normalized pool."""
    from cambacica.corpus.normalization import (
        normalize_source,
        verify_normalized_source,
    )

    try:
        if args.verify_only:
            valid, errors = verify_normalized_source(
                args.source,
                output_root=args.output_root,
                raw_root=args.raw_root,
            )
            if valid:
                print(f"[PASS] Normalized source verified: {args.source}")
                return 0
            for error in errors:
                print(f"[FAIL] {error}", file=sys.stderr)
            return 1
        _require_clean_normalization_commit()
        manifest = normalize_source(
            args.source,
            raw_root=args.raw_root,
            output_root=args.output_root,
            shard_text_bytes=args.shard_text_bytes,
            resume=args.resume,
        )
        print(
            json.dumps(
                {
                    "source": manifest["source"],
                    "status": manifest["status"],
                    "source_document_count": manifest["source_document_count"],
                    "output_document_count": manifest["output_document_count"],
                    "normalized_files": len(manifest["normalized_files"]),
                    "normalized_bytes": manifest["total_normalized_bytes"],
                    "normalized_characters": manifest["total_normalized_characters"],
                    "normalized_words": manifest["total_normalized_words"],
                    "failures": manifest["normalization_failure_count"],
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0 if manifest["status"] == "COMPLETE" else 1
    except Exception as exc:
        print(f"[ERROR] Normalization failed: {exc}", file=sys.stderr)
        return 1


def handle_characterize(args: argparse.Namespace) -> int:
    """Characterize normalized source pools and candidate capacities."""
    from cambacica.corpus.characterize import characterize_sources

    try:
        summary = characterize_sources(
            normalized_root=args.normalized_root,
            target_words=args.target_words,
        )
        print(
            json.dumps(
                {
                    "normalized_root": str(args.normalized_root),
                    "characterization_dir": str(
                        args.normalized_root / "characterization"
                    ),
                    "total_normalized_words": summary["total_normalized_words"],
                    "target_normalized_words": summary["target_normalized_words"],
                    "candidate_mixes": {
                        name: {
                            "max_non_oversampled_total_normalized_words": report[
                                "max_non_oversampled_total_normalized_words"
                            ],
                            "limiting_component": report["limiting_component"],
                            "source_pool_status": report["source_pool_status"],
                            "feasibility_status": report["feasibility_status"],
                            "oversampling_required_at_target": report[
                                "oversampling_required_at_target"
                            ],
                        }
                        for name, report in summary["candidate_mix_feasibility"].items()
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] Characterization failed: {exc}", file=sys.stderr)
        return 1


def handle_exact_dedup(args: argparse.Namespace) -> int:
    """Build, verify, audit, or run the bounded Gate C1 exact-dedup pilot."""
    try:
        if args.mode == "audit-identities":
            report = audit_occurrence_ids(
                normalized_root=args.normalized_root,
                scratch_dir=args.audit_scratch_dir,
                include_upstream_identity_diagnostic=(
                    not args.skip_upstream_identity_diagnostic
                ),
            )
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return int(
                report["collisions"] != 0
                or report["total_occurrence_ids"] != report["unique_occurrence_ids"]
            )
        if args.mode == "verify":
            errors = verify_exact_dedup(
                normalized_root=args.normalized_root,
                output_root=args.output_root,
            )
            if errors:
                print("[ERROR] Exact-dedup verification failed:")
                for error in errors:
                    print(f"  - {error}")
                return 1
            print(f"[OK] Exact-dedup outputs verified: {args.output_root}")
            return 0
        if args.mode == "pilot":
            manifest = run_exact_dedup_pilot(
                normalized_root=args.normalized_root,
                output_root=args.output_root,
                size=args.pilot_size,
                seed=args.seed,
            )
            print(
                json.dumps(
                    {
                        "output_root": str(args.output_root),
                        "run_type": manifest["run_type"],
                        "input_record_count": manifest["input_record_count"],
                        "retained_record_count": manifest["retained_record_count"],
                        "eligible_duplicate_hash_groups": manifest[
                            "eligible_duplicate_hash_groups"
                        ],
                        "pilot": manifest["pilot"],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
        manifest = build_exact_dedup(
            normalized_root=args.normalized_root,
            output_root=args.output_root,
        )
        print(
            json.dumps(
                {
                    "output_root": str(args.output_root),
                    "run_type": manifest["run_type"],
                    "input_record_count": manifest["input_record_count"],
                    "retained_record_count": manifest["retained_record_count"],
                    "eligible_duplicate_hash_groups": manifest[
                        "eligible_duplicate_hash_groups"
                    ],
                    "dropped_eligible_record_count": manifest[
                        "dropped_eligible_record_count"
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] Exact-dedup {args.mode} failed: {exc}", file=sys.stderr)
        return 1


def handle_near_dedup_pilot(args: argparse.Namespace) -> int:
    """Run the non-destructive Gate C1 near-dedup pilot."""
    try:
        manifest = run_near_dedup_pilot(
            input_root=args.input_root,
            output_root=args.output_root,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "output_root": str(args.output_root),
                    "status": manifest["status"],
                    "input_exact_manifest_sha256": manifest[
                        "input_exact_manifest_sha256"
                    ],
                    "sample_records": manifest["resource_report"][
                        "sample_record_count"
                    ],
                    "review_pairs": manifest["review_validation"]["review_pair_count"],
                    "near_dedup_production": manifest["production"][
                        "near_dedup_production"
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] Near-dedup pilot failed: {exc}", file=sys.stderr)
        return 1


def handle_near_dedup_calibration(args: argparse.Namespace) -> int:
    """Run targeted D2b calibration without changing corpus rows."""
    try:
        manifest = run_near_dedup_calibration(
            input_root=args.input_root,
            base_pilot_root=args.base_pilot_root,
            output_root=args.output_root,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "output_root": str(args.output_root),
                    "status": manifest["status"],
                    "input_exact_manifest_sha256": manifest[
                        "input_exact_manifest_sha256"
                    ],
                    "calibration_records": (
                        manifest["sampling_design"]["D2_base_records"]
                        + manifest["sampling_design"]["targeted_family_records"]
                    ),
                    "gold_exhaustive_pairs": manifest["gold_subset"][
                        "exhaustive_pair_count"
                    ],
                    "near_dedup_production": manifest["production"][
                        "near_dedup_production"
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] Near-dedup calibration failed: {exc}", file=sys.stderr)
        return 1


def handle_materialize(args: argparse.Namespace) -> int:
    """Handle the 'materialize' subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    import time
    from cambacica.corpus.manifest import compute_file_sha256
    from cambacica.corpus.materialize import get_materializer

    source_name = args.source.lower().replace("-", "_")

    try:
        materializer = get_materializer(
            source=source_name,
            config_path=args.config,
            destination=args.destination,
            allow_custom_destination=args.allow_custom_destination,
        )
    except Exception as err:
        print(f"[ERROR] Failed to initialize materializer: {err}", file=sys.stderr)
        return 1

    # Dry-run mode
    if args.dry_run:
        plan = materializer.plan()
        print("\n=== DRY RUN / MATERIALIZATION PLAN REVIEW ===")
        print(f"Source:                {plan.get('source')}")
        print(f"Canonical Name:        {plan.get('canonical_name')}")
        print(f"Upstream Repository:   {plan.get('repository')}")
        print(f"Pinned Revision:       {plan.get('pinned_revision')}")
        print(
            f"Commit SHA / Snapshot: {plan.get('pinned_commit_sha') or plan.get('snapshot_date') or 'N/A'}"
        )
        print(
            f"Acquisition Mode:      {plan.get('acquisition_mode', 'full_catalog_snapshot')}"
        )
        print(f"Target Destination:    {plan.get('destination')}")
        print(f"Estimated Raw Size:    {plan.get('estimated_raw_size')}")
        if "expected_catalog_size" in plan:
            print(
                f"Expected Catalog Size: {plan.get('expected_catalog_size'):,} eBooks"
            )
        if "existing_text_files" in plan:
            print(f"Existing Text Files:   {plan.get('existing_text_files'):,}")
        if "checksum_provenance_requirements" in plan:
            print(
                f"Provenance Reqs:       {plan.get('checksum_provenance_requirements')}"
            )
        print("==============================================\n")
        print(
            "[DRY-RUN] No downloads performed. Safety boundaries and revisions verified."
        )
        return 0

    # Verify-only mode
    if args.verify_only:
        print(
            f"--> Verifying materialized source '{source_name}' at {materializer.destination}..."
        )
        is_valid, errors = materializer.verify()
        manifest_file = materializer.destination / "manifest.json"
        if is_valid and manifest_file.is_file():
            from cambacica.corpus.materialize import MaterializationManifest

            manifest = MaterializationManifest.load(manifest_file)
            manifest_sha = compute_file_sha256(manifest_file)
            print(
                f"\n[PASS] Materialization verification succeeded for '{source_name}'."
            )
            print(f"       Status:            {manifest.status}")
            print(f"       Total Files:       {manifest.total_files:,}")
            print(
                f"       Total Bytes:       {manifest.total_bytes:,} ({manifest.total_bytes / (1024 * 1024):.2f} MB)"
            )
            print(f"       Manifest SHA-256:  {manifest_sha}")
            print(
                "       All payload files exist, byte counts match, and SHA-256 verified."
            )
            print("       Zero orphaned .partial files detected.")
            return 0
        else:
            print(
                f"\n[FAIL] Materialization verification failed for '{source_name}':",
                file=sys.stderr,
            )
            for err in errors:
                print(f"       - {err}", file=sys.stderr)
            return 1

    # Live Materialization
    effective_concurrency = args.concurrency
    if source_name in {"gigaverbo", "gigaverbo_v2"}:
        effective_concurrency = 1
    print(f"--> Materializing source '{source_name}'...")
    print(f"    Config:          {args.config}")
    print(f"    Destination:     {materializer.destination}")
    if effective_concurrency == args.concurrency:
        print(f"    Concurrency:     {effective_concurrency}")
    else:
        print(
            f"    Concurrency:     {effective_concurrency} (serialized row-group stream; "
            f"requested {args.concurrency})"
        )
    print(f"    Timeout:         {args.timeout}s")
    print(f"    Max Retries:     {args.retries}")

    t0 = time.time()
    try:
        manifest = materializer.materialize(
            # Keep the requested value for provenance; GigaVerbo records its
            # serialized effective concurrency inside the source metadata.
            concurrency=args.concurrency,
            timeout=args.timeout,
            max_retries=args.retries,
        )
    except NotImplementedError as nie:
        print(f"[ERROR] {nie}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] Materialization aborted with exception: {e}", file=sys.stderr)
        return 1

    elapsed = time.time() - t0
    manifest_file = materializer.destination / "manifest.json"
    manifest_sha = (
        compute_file_sha256(manifest_file) if manifest_file.is_file() else "N/A"
    )

    print("\n" + "=" * 60)
    print(f"MATERIALIZATION RESULT: {manifest.status}")
    print("=" * 60)
    print(f"Source:                {manifest.source}")
    print(f"Status:                {manifest.status}")
    print(f"Total Files:           {manifest.total_files:,}")
    print(
        f"Total Bytes:           {manifest.total_bytes:,} ({manifest.total_bytes / (1024 * 1024):.2f} MB)"
    )
    print(f"Elapsed Time:          {elapsed:.2f}s")
    print(f"Manifest Path:         {manifest_file}")
    print(f"Manifest SHA-256:      {manifest_sha}")
    if manifest.failed_ids:
        print(f"Failed Items Count:    {len(manifest.failed_ids)}")
        print(f"Failed IDs:            {manifest.failed_ids}")

    return 0 if manifest.status == "COMPLETE" else 1


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

    # Subcommand: validate-mixes
    val_parser = subparsers.add_parser(
        "validate-mixes",
        help="Validate Gate C1 candidate corpus mixture configuration files.",
    )
    val_parser.add_argument(
        "files",
        nargs="*",
        default=[
            "configs/corpus_mix_a.yaml",
            "configs/corpus_mix_b.yaml",
            "configs/corpus_mix_c.yaml",
        ],
        help="Paths to mix YAML config files to validate (default: configs/corpus_mix_[a,b,c].yaml).",
    )

    # Subcommand: normalize
    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Normalize one complete Gate C1 raw source into versioned Parquet.",
    )
    normalize_parser.add_argument(
        "source",
        choices=[
            "gutenberg_pt",
            "parlamento_pt",
            "wikipedia_pt",
            "carolina",
            "gigaverbo_v2",
        ],
        help="Pinned raw source family to normalize.",
    )
    normalize_parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/mnt/data/cambacica-base-180m/raw"),
        help="Root containing the five verified raw source directories.",
    )
    normalize_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/data/cambacica-base-180m/normalized"),
        help="Normalized source output root.",
    )
    normalize_parser.add_argument(
        "--shard-text-bytes",
        type=int,
        default=256 * 1024 * 1024,
        help="Target normalized UTF-8 text bytes per Parquet shard (default: 256 MiB).",
    )
    normalize_parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume an interrupted run only when all frozen inputs/options match.",
    )
    normalize_parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify a completed normalized source without processing raw payloads.",
    )

    # Subcommand: characterize
    characterize_parser = subparsers.add_parser(
        "characterize",
        help="Characterize all completed normalized source pools and candidate capacities.",
    )
    characterize_parser.add_argument(
        "--normalized-root",
        type=Path,
        default=Path("/mnt/data/cambacica-base-180m/normalized"),
        help="Root containing the five normalized source directories.",
    )
    characterize_parser.add_argument(
        "--target-words",
        type=int,
        default=None,
        help="Optional candidate corpus size for explicit no-oversampling feasibility checks.",
    )

    # Subcommand: exact-dedup
    exact_parser = subparsers.add_parser(
        "exact-dedup",
        help="Build or verify exact-dedup outputs, or run a small normalized-pool pilot.",
    )
    exact_parser.add_argument(
        "mode",
        choices=["run", "pilot", "verify", "audit-identities"],
        help=(
            "Production build, deterministic pilot, output verification, or "
            "metadata-only occurrence identity audit."
        ),
    )
    exact_parser.add_argument(
        "--normalized-root",
        type=Path,
        default=Path("/mnt/data/cambacica-base-180m/normalized"),
        help="Root containing the five immutable normalized source pools.",
    )
    exact_parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_EXACT_ROOT,
        help="Exact-dedup output directory (must not already exist).",
    )
    exact_parser.add_argument(
        "--pilot-size",
        type=int,
        default=DEFAULT_PILOT_SIZE,
        help="Target stratified pilot record count (pilot mode only).",
    )
    exact_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_PILOT_SEED,
        help="Fixed deterministic pilot seed (pilot mode only).",
    )
    exact_parser.add_argument(
        "--audit-scratch-dir",
        type=Path,
        default=None,
        help="Optional scratch location for the metadata-only identity audit.",
    )
    exact_parser.add_argument(
        "--skip-upstream-identity-diagnostic",
        action="store_true",
        help="Skip the repeated upstream/provenance identity summary.",
    )

    near_pilot_parser = subparsers.add_parser(
        "near-dedup-pilot",
        help="Run a deterministic representative near-dedup pilot on exact/data.",
    )
    near_pilot_parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_EXACT_DATA_ROOT,
        help="Exact-deduplicated corpus root or its data/ directory; never the normalized pool.",
    )
    near_pilot_parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_NEAR_PILOT_ROOT,
        help="Separate pilot output directory; existing outputs are never overwritten.",
    )
    near_pilot_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_NEAR_PILOT_SEED,
        help="Fixed deterministic pilot seed.",
    )

    near_calibration_parser = subparsers.add_parser(
        "near-dedup-calibration",
        help="Run targeted D2b scientific near-dedup calibration.",
    )
    near_calibration_parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_EXACT_DATA_ROOT,
        help="Exact-deduplicated corpus root; source rows are read-only.",
    )
    near_calibration_parser.add_argument(
        "--base-pilot-root",
        type=Path,
        default=DEFAULT_BASE_PILOT_ROOT,
        help="Completed D2 pilot artifacts used as the starting calibration panel.",
    )
    near_calibration_parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_CALIBRATION_ROOT,
        help="New separate output directory; existing outputs are never overwritten.",
    )
    near_calibration_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_NEAR_PILOT_SEED,
        help="Fixed deterministic family and review selection seed.",
    )

    # Subcommand: materialize
    mat_parser = subparsers.add_parser(
        "materialize",
        help="Materialize raw source corpus data reproducibly.",
    )
    mat_parser.add_argument(
        "source",
        choices=[
            "gutenberg",
            "gutenberg_pt",
            "carolina",
            "wikipedia_pt",
            "wikipedia",
            "parlamento_pt",
            "parlamento",
            "gigaverbo_v2",
            "gigaverbo",
        ],
        help="Source identifier to materialize.",
    )
    mat_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Output materialization plan and safety parameters without downloading.",
    )
    mat_parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify previously materialized payload files and manifest integrity without downloading.",
    )
    mat_parser.add_argument(
        "--config",
        type=str,
        default="configs/corpus_materialization.yaml",
        help="Path to corpus_materialization.yaml.",
    )
    mat_parser.add_argument(
        "--destination",
        type=str,
        default=None,
        help="Destination directory override.",
    )
    mat_parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Worker threads for downloads (default: 4).",
    )
    mat_parser.add_argument(
        "--timeout",
        type=int,
        default=25,
        help="HTTP request timeout in seconds (default: 25).",
    )
    mat_parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Max retries per item (default: 3).",
    )
    mat_parser.add_argument(
        "--allow-custom-destination",
        action="store_true",
        help="Allow custom destination path outside /mnt/data for testing.",
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
    elif args.subcommand == "validate-mixes":
        return handle_validate_mixes(args)
    elif args.subcommand == "normalize":
        return handle_normalize(args)
    elif args.subcommand == "characterize":
        return handle_characterize(args)
    elif args.subcommand == "exact-dedup":
        return handle_exact_dedup(args)
    elif args.subcommand == "near-dedup-pilot":
        return handle_near_dedup_pilot(args)
    elif args.subcommand == "near-dedup-calibration":
        return handle_near_dedup_calibration(args)
    elif args.subcommand == "materialize":
        return handle_materialize(args)
    return 1


if __name__ == "__main__":
    import os

    sys.stdout.flush()
    sys.stderr.flush()
    ret = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(ret)
