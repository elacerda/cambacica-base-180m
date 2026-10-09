#!/usr/bin/env python3
"""Bounded, stratified performance profile for the C1 benchmark matcher."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import resource
import signal
import shutil
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cambacica.corpus.decontamination.matcher import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    MATCHER_VERSION,
    NORMALIZATION_VERSION,
    BenchmarkMatcher,
    CorpusDocument,
    fields_from_snapshot,
)
from cambacica.corpus.decontamination.production import (  # noqa: E402
    DEFAULT_EXACT_ROOT,
    PINNED_EXACT_MANIFEST_SHA256,
    _canonical_sha256,
    _rss_bytes,
    _write_anchor_evidence,
    _write_hits,
    inspect_bd3_inputs,
)
from cambacica.corpus.dedup.exact_pipeline import occurrence_id_v2  # noqa: E402
from cambacica.corpus.decontamination.snapshot import (  # noqa: E402
    DEFAULT_SNAPSHOT_ROOT,
    sha256_file,
    verify_snapshot,
)


class ProfileTimeout(TimeoutError):
    """Raised by the process timer to stop an over-budget profile safely."""


def _timeout_handler(_signum: int, _frame: Any) -> None:
    raise ProfileTimeout("preflight profiling wall-time limit reached")


def _read_proc_io() -> dict[str, int] | None:
    path = Path("/proc/self/io")
    if not path.is_file():
        return None
    result: dict[str, int] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        name, value = line.split(":", 1)
        result[name] = int(value.strip())
    return result


def _hash_rank(seed: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).digest()


def _sample_plan(
    data_files: list[dict[str, Any]], input_root: Path, seed: str
) -> dict[str, deque[dict[str, Any]]]:
    """Select row groups deterministically, interleaved across source strata."""
    strata: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in data_files:
        relative = str(record["relative"])
        normalized_shard = Path(relative).relative_to("data").as_posix()
        path = input_root / relative
        parquet = pq.ParquetFile(path)
        row_ordinal = 0
        for row_group in range(parquet.metadata.num_row_groups):
            metadata = parquet.metadata.row_group(row_group)
            stratum = "/".join(Path(normalized_shard).parts[:2])
            spec = {
                "relative": relative,
                "normalized_shard": normalized_shard,
                "row_group": row_group,
                "row_ordinal_start": row_ordinal,
                "row_count": metadata.num_rows,
                "rank": _hash_rank(seed, f"{relative}#{row_group}"),
            }
            strata[Path(normalized_shard).parts[0]][stratum].append(spec)
            row_ordinal += metadata.num_rows

    giga_subset_words: dict[str, int] = {}
    accounting_path = input_root / "accounting_by_source_subset.csv"
    with accounting_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["source"] == "gigaverbo_v2" and row["scope"] == "source_subset":
                giga_subset_words[row["subset"]] = int(row["normalized_words_after"])

    result: dict[str, deque[dict[str, Any]]] = {}
    for family, family_strata in sorted(strata.items()):
        for items in family_strata.values():
            items.sort(
                key=lambda item: (item["rank"], item["relative"], item["row_group"])
            )
        ordered: list[dict[str, Any]] = []
        if family == "gigaverbo_v2":
            stratum_names = sorted(
                family_strata,
                key=lambda name: (
                    -giga_subset_words.get(name.split("subset=", 1)[-1], 0),
                    _hash_rank(seed, name),
                ),
            )
        else:
            stratum_names = sorted(
                family_strata, key=lambda name: (_hash_rank(seed, name), name)
            )
        offset = 0
        while any(offset < len(family_strata[name]) for name in stratum_names):
            for name in stratum_names:
                if offset < len(family_strata[name]):
                    ordered.append(family_strata[name][offset])
            offset += 1
        result[family] = deque(ordered)
    return result


def _family_word_targets(input_root: Path, total_target: int) -> dict[str, int]:
    """Allocate most sample words in proportion to the pinned post-D1 mix."""
    accounting_path = input_root / "accounting_by_source_subset.csv"
    manifest = json.loads((input_root / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest["output_files"]["accounting_by_source_subset.csv"]["sha256"]
    if sha256_file(accounting_path) != expected:
        raise ValueError("Post-D1 source accounting checksum mismatch")
    words: dict[str, int] = {}
    with accounting_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["scope"] == "source_total":
                words[row["source"]] = int(row["normalized_words_after"])
    if len(words) != 5:
        raise ValueError(f"Expected five source word totals, found {sorted(words)}")
    minimum_per_family = 400_000
    if total_target < minimum_per_family * len(words):
        raise ValueError("Profiling target is too small to cover all source families")
    distributable = total_target - minimum_per_family * len(words)
    total_words = sum(words.values())
    exact = {
        family: distributable * family_words / total_words
        for family, family_words in words.items()
    }
    targets = {
        family: minimum_per_family + math.floor(value)
        for family, value in exact.items()
    }
    remainder = total_target - sum(targets.values())
    for family in sorted(
        exact, key=lambda item: (-(exact[item] - math.floor(exact[item])), item)
    )[:remainder]:
        targets[family] += 1
    return targets


def _group_compressed_bytes(
    parquet: pq.ParquetFile, row_group: int, column_names: set[str]
) -> int:
    metadata = parquet.metadata.row_group(row_group)
    names = parquet.schema_arrow.names
    return sum(
        metadata.column(index).total_compressed_size
        for index, name in enumerate(names)
        if name in column_names
    )


def _resource_peak_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024 if value < 1_000_000_000 else value)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _profile(
    *,
    input_root: Path,
    snapshot_dir: Path,
    output_dir: Path,
    scratch_parent: Path,
    seed: str,
    target_words: int,
    max_seconds: int,
    max_document_characters: int,
    stress_documents: int,
    persistent_checkpoint: bool,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite profile output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    scratch_parent.mkdir(parents=True, exist_ok=True)
    scratch_dir = scratch_parent / f"c1-bd25-profile-{os.getpid()}"
    scratch_dir.mkdir()
    scratch_free_bytes_at_start = shutil.disk_usage(scratch_dir).free
    snapshot_check = verify_snapshot(snapshot_dir)
    inputs = inspect_bd3_inputs(input_root)
    data_files = inputs.pop("data_file_inventory")
    fields = fields_from_snapshot(snapshot_dir / "match_fields.parquet")
    matcher_index_started = time.perf_counter()
    matcher = BenchmarkMatcher(fields)
    matcher_index_seconds = time.perf_counter() - matcher_index_started
    matcher_policy = matcher.policy.to_dict()
    matcher_policy_sha256 = hashlib.sha256(
        json.dumps(
            matcher_policy, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()

    sample_targets_by_family = _family_word_targets(input_root, target_words)
    sample_words: dict[str, int] = defaultdict(int)
    sample_docs: dict[str, int] = defaultdict(int)
    sample_read_seconds: dict[str, float] = defaultdict(float)
    sample_conversion_seconds: dict[str, float] = defaultdict(float)
    sample_match_wall: dict[str, float] = defaultdict(float)
    sample_match_cpu: dict[str, float] = defaultdict(float)
    sample_compressed_bytes: dict[str, int] = defaultdict(int)
    sample_word_lengths: dict[str, list[int]] = defaultdict(list)
    sample_id_hashes = defaultdict(hashlib.sha256)
    selected_groups: list[dict[str, Any]] = []
    skipped_oversize_documents = 0
    completed_words = 0
    completed_documents = 0
    hit_counts: dict[str, int] = defaultdict(int)
    sample_input_io_before = _read_proc_io()
    sampler = _sample_plan(data_files, input_root, seed)
    families = sorted(sampler)
    if len(families) != 5:
        raise ValueError(f"Expected five post-D1 source families, found {families}")
    sample_plan = [
        {
            "source_family": family,
            "normalized_shard": spec["normalized_shard"],
            "row_group": int(spec["row_group"]),
            "row_ordinal_start": int(spec["row_ordinal_start"]),
            "row_count": int(spec["row_count"]),
        }
        for family in families
        for spec in list(sampler[family])
    ]
    checkpoint_database_path = scratch_dir / "resumable-profile.sqlite3"
    checkpoint_identity = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "profile_run_id": f"bd2.6-profile-{os.getpid()}",
        "input_manifest_sha256": inputs["manifest_sha256"],
        "ordered_source_inventory_sha256": _canonical_sha256(
            [
                {
                    "relative": row["relative"],
                    "bytes": row["bytes"],
                    "rows": row["rows"],
                    "sha256": row["sha256"],
                }
                for row in data_files
            ]
        ),
        "benchmark_snapshot_manifest_sha256": snapshot_check["manifest_sha256"],
        "matcher_policy_sha256": matcher_policy_sha256,
        "matcher_policy": matcher_policy,
        "matcher_implementation_version": MATCHER_VERSION,
        "normalization_implementation_version": NORMALIZATION_VERSION,
        "sample_plan_sha256": _canonical_sha256(sample_plan),
    }

    profile_started = time.perf_counter()
    cpu_started = time.process_time()
    timer_installed = hasattr(signal, "setitimer")
    previous_handler = None
    if timer_installed:
        previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, max_seconds)

    run = matcher.start_run(
        scratch_dir=scratch_dir,
        profile=True,
        database_path=checkpoint_database_path if persistent_checkpoint else None,
        checkpoint_identity=checkpoint_identity if persistent_checkpoint else None,
    )
    run_status = "COMPLETE"
    sample_loop_started = time.perf_counter()
    try:
        with run:
            while any(
                sample_words[family] < sample_targets_by_family[family]
                and sampler[family]
                for family in families
            ):
                made_progress = False
                for family in families:
                    if completed_words >= 20_000_000:
                        break
                    if sample_words[family] >= sample_targets_by_family[family]:
                        continue
                    if not sampler[family]:
                        continue
                    spec = sampler[family].popleft()
                    parquet_path = input_root / spec["relative"]
                    parquet = pq.ParquetFile(parquet_path)
                    row_group = int(spec["row_group"])
                    payload_bytes = _group_compressed_bytes(
                        parquet,
                        row_group,
                        {"text", "source", "content_sha256"},
                    )
                    selected_groups.append(
                        {
                            "source_family": family,
                            "normalized_shard": spec["normalized_shard"],
                            "row_group": row_group,
                            "row_ordinal_start": int(spec["row_ordinal_start"]),
                            "row_count_in_group": int(spec["row_count"]),
                            "selected_columns_compressed_bytes_upper_bound": payload_bytes,
                            "documents_processed": 0,
                            "normalized_words_processed": 0,
                        }
                    )
                    group_record = selected_groups[-1]
                    sample_compressed_bytes[family] += payload_bytes
                    batches = iter(
                        parquet.iter_batches(
                            batch_size=16,
                            row_groups=[row_group],
                            columns=["text", "source", "content_sha256"],
                            use_threads=False,
                        )
                    )
                    row_ordinal = int(spec["row_ordinal_start"])
                    reached_family_target = False
                    while not reached_family_target:
                        read_started = time.perf_counter()
                        try:
                            batch = next(batches)
                        except StopIteration:
                            break
                        sample_read_seconds[family] += (
                            time.perf_counter() - read_started
                        )
                        made_progress = True
                        for index in range(batch.num_rows):
                            if (
                                completed_words >= 20_000_000
                                or sample_words[family]
                                >= sample_targets_by_family[family]
                            ):
                                reached_family_target = True
                                break
                            conversion_started = time.perf_counter()
                            text = batch.column(0)[index].as_py()
                            source = batch.column(1)[index].as_py()
                            content_sha256 = batch.column(2)[index].as_py()
                            if not isinstance(text, str) or not isinstance(source, str):
                                raise ValueError(
                                    f"Invalid pinned record: {spec['normalized_shard']}:{row_ordinal}"
                                )
                            actual_sha = hashlib.sha256(
                                text.encode("utf-8")
                            ).hexdigest()
                            if actual_sha != content_sha256:
                                raise ValueError(
                                    f"Content checksum mismatch: {spec['normalized_shard']}:{row_ordinal}"
                                )
                            sample_conversion_seconds[family] += (
                                time.perf_counter() - conversion_started
                            )
                            if len(text) > max_document_characters:
                                skipped_oversize_documents += 1
                                row_ordinal += 1
                                continue
                            record_id = occurrence_id_v2(
                                source, spec["normalized_shard"], row_ordinal
                            )
                            document = CorpusDocument(
                                doc_id=record_id,
                                text=text,
                                source_shard=spec["normalized_shard"],
                                source=source,
                                source_row_ordinal=row_ordinal,
                                input_manifest_sha256=inputs["manifest_sha256"],
                            )
                            matcher_started = time.perf_counter()
                            matcher_cpu_started = time.process_time()
                            run.add_document(document)
                            sample_match_wall[family] += (
                                time.perf_counter() - matcher_started
                            )
                            sample_match_cpu[family] += (
                                time.process_time() - matcher_cpu_started
                            )
                            sample_words[family] += run.last_document_tokens
                            completed_words += run.last_document_tokens
                            sample_docs[family] += 1
                            completed_documents += 1
                            sample_word_lengths[family].append(run.last_document_tokens)
                            encoded = record_id.encode("ascii")
                            sample_id_hashes[family].update(
                                len(encoded).to_bytes(8, "big") + encoded
                            )
                            group_record["documents_processed"] += 1
                            group_record["normalized_words_processed"] += (
                                run.last_document_tokens
                            )
                            row_ordinal += 1
                    if completed_words >= 20_000_000:
                        break
                if not made_progress:
                    break
            sample_loop_seconds = time.perf_counter() - sample_loop_started
            sample_target_met = all(
                sample_words[family] >= sample_targets_by_family[family]
                for family in families
            )
            run.finish()
            run_status = "COMPLETE" if sample_target_met else "COMPLETE_BELOW_TARGET"
            for result in run.iter_results():
                family = Path(result.source_shard).parts[0]
                hit_counts[family] += 1
            sample_write_started = time.perf_counter()
            sample_hit_count = _write_hits(
                output_dir / "profile_candidate_hits.parquet", run.iter_results()
            )
            sample_evidence_count = _write_anchor_evidence(
                output_dir / "profile_candidate_anchor_evidence.parquet",
                run.iter_anchor_evidence(),
            )
            sample_serialization_seconds = time.perf_counter() - sample_write_started
            sample_sqlite_bytes = run.scratch_bytes
            sample_sqlite_changes = run.connection.total_changes
            sample_profile_timings = run.profile_timings
            sample_anchor_evidence_rows_stored = run.anchor_evidence_rows
    except ProfileTimeout:
        run_status = "TIME_LIMIT_REACHED_BEFORE_FINALIZATION"
        sample_loop_seconds = time.perf_counter() - sample_loop_started
        sample_serialization_seconds = 0.0
        sample_sqlite_bytes = None
        sample_sqlite_changes = None
        sample_profile_timings = run.profile_timings
        sample_anchor_evidence_rows_stored = None
        sample_hit_count = None
        sample_evidence_count = None
        for path in output_dir.glob("profile_candidate_*.parquet"):
            path.unlink(missing_ok=True)
    finally:
        if timer_installed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)

    checkpoint_restart = {"status": "NOT_RUN_NONRESUMABLE_PROFILE"}
    if persistent_checkpoint and run_status.startswith("COMPLETE"):
        restart_started = time.perf_counter()
        with matcher.start_run(
            database_path=checkpoint_database_path,
            resume=True,
            checkpoint_identity=checkpoint_identity,
        ) as resumed_run:
            restart_seconds = time.perf_counter() - restart_started
            checkpoint_restart = {
                "status": "REOPENED_FINALIZED_CHECKPOINT",
                "restart_duration_seconds": round(restart_seconds, 6),
                "checkpoint_status": resumed_run.checkpoint_status,
                "completed_records": resumed_run.resume_position["documents_seen"],
            }

    stress_metrics = _run_synthetic_stress(
        matcher,
        fields,
        output_dir,
        scratch_dir,
        stress_documents,
        enabled=run_status.startswith("COMPLETE"),
    )
    recovery_metrics = (
        _run_synthetic_recovery(matcher, fields, output_dir, scratch_dir)
        if persistent_checkpoint and run_status.startswith("COMPLETE")
        else {"status": "NOT_RUN_NONRESUMABLE_PROFILE"}
    )
    output_serialization_seconds = sample_serialization_seconds + stress_metrics.get(
        "serialization_seconds", 0.0
    )
    elapsed_seconds = time.perf_counter() - profile_started
    cpu_seconds = time.process_time() - cpu_started
    input_io_after = _read_proc_io()
    io_delta = None
    if sample_input_io_before is not None and input_io_after is not None:
        io_delta = {
            key: input_io_after[key] - sample_input_io_before.get(key, 0)
            for key in input_io_after
        }

    by_source_rows = []
    for family in families:
        lengths = sorted(sample_word_lengths[family])
        by_source_rows.append(
            {
                "source_family": family,
                "documents_processed": sample_docs[family],
                "normalized_words_processed": sample_words[family],
                "input_compressed_payload_bytes_upper_bound": sample_compressed_bytes[
                    family
                ],
                "input_read_and_decompression_seconds": round(
                    sample_read_seconds[family], 6
                ),
                "python_conversion_and_content_hash_seconds": round(
                    sample_conversion_seconds[family], 6
                ),
                "matcher_wall_seconds": round(sample_match_wall[family], 6),
                "matcher_cpu_seconds": round(sample_match_cpu[family], 6),
                "matcher_words_per_second": round(
                    sample_words[family] / sample_match_wall[family], 3
                )
                if sample_match_wall[family]
                else None,
                "end_to_end_words_per_second": round(
                    sample_words[family]
                    / (
                        sample_read_seconds[family]
                        + sample_conversion_seconds[family]
                        + sample_match_wall[family]
                    ),
                    3,
                )
                if sample_words[family]
                else None,
                "short_document_count_under_32_words": sum(
                    length < 32 for length in lengths
                ),
                "long_document_count_at_least_8192_words": sum(
                    length >= 8192 for length in lengths
                ),
                "minimum_document_words": min(lengths) if lengths else None,
                "median_document_words": lengths[len(lengths) // 2]
                if lengths
                else None,
                "maximum_document_words": max(lengths) if lengths else None,
                "candidate_hits": hit_counts[family],
                "sample_id_sha256": sample_id_hashes[family].hexdigest(),
            }
        )
    by_source_path = output_dir / "performance_by_source.csv"
    _write_csv(by_source_path, by_source_rows, list(by_source_rows[0]))

    results = {
        "preflight_profile_version": "c1-bd2.6-recovery-profile-v1",
        "status": run_status,
        "production_scan": "NOT_RUN",
        "corpus_input": {
            "root": str(input_root.resolve()),
            "manifest_sha256": inputs["manifest_sha256"],
            "pinned_manifest_sha256": PINNED_EXACT_MANIFEST_SHA256,
            "retained_records_in_full_corpus": inputs["retained_records"],
            "full_corpus_normalized_words": 21_470_091_017,
            "full_corpus_compressed_bytes": inputs["compressed_bytes"],
        },
        "benchmark_snapshot": {
            "root": str(snapshot_dir.resolve()),
            "manifest_sha256": snapshot_check["manifest_sha256"],
            "records": snapshot_check["benchmark_rows"],
            "match_fields": snapshot_check["match_fields"],
        },
        "matcher_policy": matcher_policy,
        "matcher_policy_sha256": matcher_policy_sha256,
        "matcher_implementation": {
            "matcher_version": matcher.policy.matcher_version,
            "normalization_version": matcher.policy.normalization_version,
            "policy_frozen_for_bd3": False,
            "persistent_checkpoint_profile": persistent_checkpoint,
        },
        "sample_design": {
            "seed": seed,
            "selection": "SHA-256-ranked Parquet row groups, round-robin across five source families and GigaVerbo subsets",
            "target_words_total": target_words,
            "target_words_by_family": sample_targets_by_family,
            "maximum_document_characters": max_document_characters,
            "sampled_source_families": families,
            "sampled_records": completed_documents,
            "sampled_normalized_words": completed_words,
            "skipped_oversize_records": skipped_oversize_documents,
            "sample_id_sha256_by_family": {
                family: sample_id_hashes[family].hexdigest() for family in families
            },
            "selected_row_groups": selected_groups,
            "deterministic_sample_sha256": hashlib.sha256(
                json.dumps(
                    [
                        (row["normalized_shard"], row["row_group"])
                        for row in selected_groups
                    ],
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
        "measurements": {
            "documents_processed": completed_documents,
            "normalized_words_processed": completed_words,
            "input_compressed_bytes_read": None,
            "input_selected_column_compressed_bytes_upper_bound": sum(
                sample_compressed_bytes.values()
            ),
            "process_io_bytes_delta": io_delta,
            "elapsed_wall_seconds": round(elapsed_seconds, 6),
            "sample_processing_loop_seconds": round(sample_loop_seconds, 6),
            "cpu_seconds": round(cpu_seconds, 6),
            "average_words_per_second_end_to_end": round(
                completed_words / elapsed_seconds, 3
            )
            if elapsed_seconds
            else None,
            "average_words_per_second_scan_phase": round(
                completed_words
                / max(
                    sample_loop_seconds,
                    1e-9,
                ),
                3,
            ),
            "peak_rss_bytes": max(_rss_bytes(), _resource_peak_bytes()),
            "sqlite_database_bytes_at_finalization": sample_sqlite_bytes,
            "sqlite_total_changes": sample_sqlite_changes,
            "sqlite_anchor_evidence_rows_stored": sample_anchor_evidence_rows_stored,
            "candidate_hits": sample_hit_count,
            "candidate_anchor_evidence_rows_serialized": sample_evidence_count,
            "output_serialization_seconds": round(output_serialization_seconds, 6),
            "sample_output_files": [
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(output_dir.glob("profile_candidate_*.parquet"))
            ],
            "matcher_stage_wall_seconds": sample_profile_timings,
            "checkpointing": {
                "enabled": persistent_checkpoint,
                "interval_documents": matcher.policy.commit_every_documents,
                "commit_seconds": sample_profile_timings.get(
                    "checkpoint_commit_seconds", 0.0
                ),
                "commit_count": int(
                    sample_profile_timings.get("checkpoint_commits", 0)
                ),
                "sqlite_database_bytes_at_finalization": sample_sqlite_bytes,
                "restart": checkpoint_restart,
                "match_heavy_recovery_stress": recovery_metrics,
            },
            "matcher_index_build_seconds": matcher_index_seconds,
            "source_families": by_source_rows,
            "synthetic_match_heavy_stress": stress_metrics,
            "major_bottlenecks": sorted(
                [
                    ["input_read_and_decompression", sum(sample_read_seconds.values())],
                    [
                        "python_conversion_and_content_hash",
                        sum(sample_conversion_seconds.values()),
                    ],
                    ["matcher_add_document", sum(sample_match_wall.values())],
                    [
                        "matcher_finalization",
                        sample_profile_timings.get("finalization_seconds", 0.0),
                    ],
                    ["output_serialization", output_serialization_seconds],
                ],
                key=lambda row: (-row[1], row[0]),
            ),
        },
        "interpretation_limits": [
            "This is a stratified row-group sample, not a probability sample of corpus records.",
            "Each source family receives at least 400,000 words; the remaining budget follows pinned post-D1 word shares.",
            "Compressed-byte payload is a row-group metadata upper bound; process I/O counters are reported separately.",
            "Candidate anchor frequencies are final only within this bounded sample, not across the full corpus.",
        ],
        "runtime": {
            "scratch_directory": str(scratch_dir),
            "scratch_free_bytes_at_start": scratch_free_bytes_at_start,
            "host_name": platform.node(),
            "host_platform": platform.platform(),
            "profile_wall_limit_seconds": max_seconds,
            "matcher_policy_frozen": False,
        },
    }
    results_path = output_dir / "performance_results.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results


def _run_synthetic_recovery(
    matcher: BenchmarkMatcher,
    fields: list[Any],
    output_dir: Path,
    scratch_dir: Path,
) -> dict[str, Any]:
    """Measure restart and replay cost with dense, synthetic match evidence."""
    passage = max(
        (
            field
            for field in fields
            if field.matchable and field.field_role in {"context", "passage"}
        ),
        key=lambda field: len(matcher.field_token_values[field.field_id]),
    )
    recovery_policy = matcher.policy
    checkpoint_interval_documents = 16
    documents = [
        CorpusDocument(
            doc_id=f"synthetic-recovery-{index:03d}",
            text=f"Envelope {index}. {passage.original_text} Fecho {index}.",
            source_shard="synthetic-recovery",
            source="synthetic_match_heavy_recovery",
            source_row_ordinal=index,
        )
        for index in range(19)
    ]
    database_path = scratch_dir / "match-heavy-recovery.sqlite3"
    identity = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "profile_run_id": f"synthetic-recovery-{os.getpid()}",
        "source": "synthetic wrappers around one pinned benchmark passage",
        "matcher_policy_sha256": _canonical_sha256(recovery_policy.to_dict()),
        "matcher_policy": recovery_policy.to_dict(),
        "checkpoint_interval_documents": checkpoint_interval_documents,
        "matcher_implementation_version": MATCHER_VERSION,
        "normalization_implementation_version": NORMALIZATION_VERSION,
    }
    started = time.perf_counter()
    cpu_started = time.process_time()
    first_run = matcher.start_run(
        database_path=database_path,
        checkpoint_identity=identity,
        profile=True,
        checkpoint_interval_documents=checkpoint_interval_documents,
    )
    for document in documents:
        first_run.add_document(document)
    initial_timings = first_run.profile_timings
    initial_sqlite_bytes = first_run.scratch_bytes
    initial_attempted_records = first_run.documents_seen
    close_started = time.perf_counter()
    first_run.close()  # rolls back the three rows after the 16-record checkpoint
    interrupted_close_seconds = time.perf_counter() - close_started

    restart_started = time.perf_counter()
    resumed_run = matcher.start_run(
        database_path=database_path,
        resume=True,
        checkpoint_identity=identity,
        profile=True,
        checkpoint_interval_documents=checkpoint_interval_documents,
    )
    restart_seconds = time.perf_counter() - restart_started
    committed_records = resumed_run.resume_position["documents_seen"]
    replayed_records = initial_attempted_records - committed_records
    for document in documents[committed_records:]:
        resumed_run.add_document(document)
    accounting = resumed_run.finish()
    hit_count = _write_hits(
        output_dir / "recovery_synthetic_candidate_hits.parquet",
        resumed_run.iter_results(),
    )
    evidence_count = _write_anchor_evidence(
        output_dir / "recovery_synthetic_anchor_evidence.parquet",
        resumed_run.iter_anchor_evidence(),
    )
    sqlite_bytes = resumed_run.scratch_bytes
    resumed_timings = resumed_run.profile_timings
    finalization_seconds = resumed_timings.get("finalization_seconds", 0.0)
    resumed_run.close()
    elapsed = time.perf_counter() - started
    all_timings = {
        name: initial_timings.get(name, 0.0) + resumed_timings.get(name, 0.0)
        for name in set(initial_timings) | set(resumed_timings)
    }
    return {
        "status": "COMPLETE_SYNTHETIC_RECOVERY",
        "workload": "19 match-heavy wrappers; 16-record committed batch; three-record replay",
        "recovery_type": "connection restart after rollback of uncommitted work",
        "documents_processed": accounting.documents_seen,
        "normalized_words_processed": resumed_run.normalized_words_seen,
        "candidate_hits": hit_count,
        "candidate_anchor_evidence_rows": evidence_count,
        "elapsed_wall_seconds": round(elapsed, 6),
        "cpu_seconds": round(time.process_time() - cpu_started, 6),
        "words_per_second": round(resumed_run.normalized_words_seen / elapsed, 3)
        if elapsed
        else None,
        "checkpoint_interval_documents": checkpoint_interval_documents,
        "checkpoint_commit_seconds": round(
            all_timings.get("checkpoint_commit_seconds", 0.0), 6
        ),
        "checkpoint_commits": int(all_timings.get("checkpoint_commits", 0)),
        "sqlite_bytes_before_restart": initial_sqlite_bytes,
        "sqlite_bytes_after_finalization": sqlite_bytes,
        "interrupted_close_seconds": round(interrupted_close_seconds, 6),
        "restart_duration_seconds": round(restart_seconds, 6),
        "records_replayed_after_interruption": replayed_records,
        "maximum_records_replayed_by_interval": checkpoint_interval_documents - 1,
        "finalization_seconds": round(finalization_seconds, 6),
        "peak_rss_bytes": _resource_peak_bytes(),
        "source_accounting": [asdict(item) for item in accounting.shard_accounting],
        "artifact_checksums": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(output_dir.glob("recovery_synthetic_*.parquet"))
        ],
    }


def _run_synthetic_stress(
    matcher: BenchmarkMatcher,
    fields: list[Any],
    output_dir: Path,
    scratch_dir: Path,
    document_count: int,
    *,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"status": "NOT_RUN_AFTER_SAMPLE_TIMEOUT"}
    passage = max(
        (
            field
            for field in fields
            if field.matchable and field.field_role in {"context", "passage"}
        ),
        key=lambda field: len(matcher.field_token_values[field.field_id]),
    )
    if len(matcher.field_token_values[passage.field_id]) < 100:
        return {"status": "SKIPPED_NO_LONG_BENCHMARK_FIELD"}
    documents = [
        CorpusDocument(
            doc_id=f"synthetic-stress-{index:05d}",
            text=f"Envelope {index}. {passage.original_text} Fecho {index}.",
            source_shard="synthetic-stress",
            source="synthetic_match_heavy_stress",
            source_row_ordinal=index,
        )
        for index in range(document_count)
    ]
    stress_run = matcher.start_run(scratch_dir=scratch_dir / "stress", profile=True)
    (scratch_dir / "stress").mkdir(exist_ok=True)
    started = time.perf_counter()
    cpu_started = time.process_time()
    timer_installed = hasattr(signal, "setitimer")
    previous_handler = None
    if timer_installed:
        previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        with stress_run:
            accounting = stress_run.scan(documents)
            finalization_seconds = stress_run.profile_timings["finalization_seconds"]
            write_started = time.perf_counter()
            hit_count = _write_hits(
                output_dir / "synthetic_stress_candidate_hits.parquet",
                stress_run.iter_results(),
            )
            evidence_count = _write_anchor_evidence(
                output_dir / "synthetic_stress_anchor_evidence.parquet",
                stress_run.iter_anchor_evidence(),
            )
            serialization_seconds = time.perf_counter() - write_started
            sqlite_bytes = stress_run.scratch_bytes
            sqlite_changes = stress_run.connection.total_changes
            profile_timings = stress_run.profile_timings
    except ProfileTimeout:
        for path in output_dir.glob("synthetic_stress_*.parquet"):
            path.unlink(missing_ok=True)
        return {"status": "TIME_LIMIT_REACHED_DURING_SYNTHETIC_STRESS"}
    finally:
        if timer_installed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)
    elapsed = time.perf_counter() - started
    word_count = stress_run.normalized_words_seen
    return {
        "status": "COMPLETE_SYNTHETIC_ONLY",
        "source": "synthetic wrappers around one pinned benchmark passage",
        "documents_processed": accounting.documents_seen,
        "normalized_words_processed": word_count,
        "candidate_hits": hit_count,
        "candidate_anchor_evidence_rows": evidence_count,
        "elapsed_wall_seconds": round(elapsed, 6),
        "cpu_seconds": round(time.process_time() - cpu_started, 6),
        "words_per_second": round(word_count / elapsed, 3) if elapsed else None,
        "peak_rss_bytes": _resource_peak_bytes(),
        "sqlite_database_bytes": sqlite_bytes,
        "sqlite_total_changes": sqlite_changes,
        "finalization_seconds": finalization_seconds,
        "serialization_seconds": round(serialization_seconds, 6),
        "matcher_stage_wall_seconds": profile_timings,
        "output_files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(output_dir.glob("synthetic_stress_*.parquet"))
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_EXACT_ROOT)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--seed", default="c1-bd2.5-profile-v1")
    parser.add_argument("--target-words", type=int, default=12_000_000)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--max-document-characters", type=int, default=1_000_000)
    parser.add_argument("--stress-documents", type=int, default=64)
    parser.add_argument(
        "--persistent-checkpoint",
        action="store_true",
        help="profile the bounded sample with a durable WAL/FULL SQLite checkpoint",
    )
    args = parser.parse_args()
    if not 10_000_000 <= args.target_words <= 20_000_000:
        parser.error("--target-words must be between 10,000,000 and 20,000,000")
    if not 1 <= args.max_seconds <= 1800:
        parser.error("--max-seconds must be between 1 and 1,800")
    if args.max_document_characters < 1 or args.stress_documents < 1:
        parser.error("document and stress limits must be positive")
    try:
        result = _profile(
            input_root=args.input_root,
            snapshot_dir=args.snapshot_dir,
            output_dir=args.output_dir,
            scratch_parent=args.scratch_dir,
            seed=args.seed,
            target_words=args.target_words,
            max_seconds=args.max_seconds,
            max_document_characters=args.max_document_characters,
            stress_documents=args.stress_documents,
            persistent_checkpoint=args.persistent_checkpoint,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
