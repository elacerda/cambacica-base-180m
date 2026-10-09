"""Explicitly gated read-only C1-BD3 scan and independent verification."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import sys
import tempfile
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from cambacica.corpus.dedup.exact_pipeline import occurrence_id_v2

from .calibration import verify_calibration
from .matcher import (
    AnchorEvidence,
    BenchmarkMatcher,
    CHECKPOINT_SCHEMA_VERSION,
    CandidatePolicy,
    CorpusDocument,
    MATCHER_VERSION,
    MatchResult,
    NORMALIZATION_VERSION,
    fields_from_snapshot,
)
from .snapshot import (
    DEFAULT_SNAPSHOT_ROOT,
    canonical_json,
    sha256_file,
    verify_snapshot,
)


PINNED_EXACT_MANIFEST_SHA256 = (
    "57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428"
)
PINNED_RETAINED_RECORDS = 21_603_689
DEFAULT_EXACT_ROOT = Path("/mnt/data/cambacica-base-180m/deduplicated/exact")
DEFAULT_CALIBRATION_ROOT = Path(
    "/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2/calibration"
)
DEFAULT_SCAN_ROOT = Path(
    "/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd3/scan-v1"
)
SCAN_VERSION = "c1-bd3-read-only-scan-v2"
RUN_METADATA_VERSION = "c1-bd2-recovery-run-v1"

ACCOUNTING_SCHEMA = pa.schema(
    [
        pa.field("normalized_shard", pa.string(), nullable=False),
        pa.field("expected_rows", pa.int64(), nullable=False),
        pa.field("accounted_rows", pa.int64(), nullable=False),
        pa.field("record_id_sha256", pa.string(), nullable=False),
        pa.field("candidate_rows", pa.int64(), nullable=False),
    ]
)
HIT_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("source_shard", pa.string(), nullable=False),
        pa.field("source", pa.string()),
        pa.field("source_row_ordinal", pa.int64()),
        pa.field("input_manifest_sha256", pa.string()),
        pa.field("document_text_sha256", pa.string(), nullable=False),
        pa.field("benchmark_name", pa.string(), nullable=False),
        pa.field("example_id", pa.string(), nullable=False),
        pa.field("source_row_id", pa.string(), nullable=False),
        pa.field("source_file_sha256", pa.string(), nullable=False),
        pa.field("source_category", pa.string()),
        pa.field("field_id", pa.string(), nullable=False),
        pa.field("field_role", pa.string(), nullable=False),
        pa.field("decision_rule", pa.string(), nullable=False),
        pa.field("exact_match", pa.bool_(), nullable=False),
        pa.field("matched_tokens", pa.int32(), nullable=False),
        pa.field("contiguous_tokens", pa.int32(), nullable=False),
        pa.field("distinctive_anchor_count", pa.int32(), nullable=False),
        pa.field("distinctive_token_coverage", pa.float64(), nullable=False),
        pa.field("matched_anchor_document_frequency_min", pa.int64()),
        pa.field("matched_anchor_document_frequency_max", pa.int64()),
        pa.field("corpus_token_start", pa.int64(), nullable=False),
        pa.field("corpus_token_end", pa.int64(), nullable=False),
        pa.field("corpus_char_start", pa.int64(), nullable=False),
        pa.field("corpus_char_end", pa.int64(), nullable=False),
        pa.field("benchmark_token_start", pa.int64(), nullable=False),
        pa.field("benchmark_token_end", pa.int64(), nullable=False),
        pa.field("benchmark_char_start", pa.int64(), nullable=False),
        pa.field("benchmark_char_end", pa.int64(), nullable=False),
    ]
)
ANCHOR_EVIDENCE_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("benchmark_name", pa.string(), nullable=False),
        pa.field("example_id", pa.string(), nullable=False),
        pa.field("source_row_id", pa.string(), nullable=False),
        pa.field("source_file_sha256", pa.string(), nullable=False),
        pa.field("source_category", pa.string()),
        pa.field("field_id", pa.string(), nullable=False),
        pa.field("field_role", pa.string(), nullable=False),
        pa.field("anchor_sha256", pa.string(), nullable=False),
        pa.field("anchor_text", pa.string(), nullable=False),
        pa.field("anchor_document_frequency", pa.int64(), nullable=False),
        pa.field("corpus_token_start", pa.int64(), nullable=False),
        pa.field("corpus_token_end", pa.int64(), nullable=False),
        pa.field("corpus_char_start", pa.int64(), nullable=False),
        pa.field("corpus_char_end", pa.int64(), nullable=False),
        pa.field("benchmark_token_start", pa.int64(), nullable=False),
        pa.field("benchmark_token_end", pa.int64(), nullable=False),
        pa.field("benchmark_char_start", pa.int64(), nullable=False),
        pa.field("benchmark_char_end", pa.int64(), nullable=False),
    ]
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024 if value < 1_000_000_000 else value)


def _expected_data_files(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    data_records = [
        {"relative": path, **metadata}
        for path, metadata in manifest["output_files"].items()
        if path.startswith("data/") and path.endswith(".parquet")
    ]
    data_records.sort(key=lambda record: record["relative"])
    actual = {
        path.relative_to(root).as_posix() for path in (root / "data").rglob("*.parquet")
    }
    expected = {record["relative"] for record in data_records}
    if actual != expected:
        raise ValueError(
            f"Exact corpus file inventory differs from manifest: "
            f"missing={len(expected - actual)}, extra={len(actual - expected)}"
        )
    for record in data_records:
        path = root / record["relative"]
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(
                f"Exact Parquet byte count differs from manifest: {record['relative']}"
            )
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != int(record["rows"]):
            raise ValueError(
                f"Exact Parquet row count differs from manifest: {record['relative']}"
            )
        if not {"text", "source", "content_sha256"}.issubset(
            parquet.schema_arrow.names
        ):
            raise ValueError(
                f"Exact Parquet schema lacks matcher columns: {record['relative']}"
            )
    return data_records


def inspect_bd3_inputs(
    input_root: Path | str = DEFAULT_EXACT_ROOT,
    expected_manifest_sha256: str = PINNED_EXACT_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Check pinned input identity and shard metadata without reading row data."""
    root = Path(input_root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    actual_sha = sha256_file(manifest_path)
    if actual_sha != expected_manifest_sha256:
        raise ValueError(
            f"Exact input manifest SHA-256 mismatch: {actual_sha} != {expected_manifest_sha256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("run_type") != "production":
        raise ValueError(
            "Pinned exact corpus manifest is not a COMPLETE production output"
        )
    if manifest.get("retained_record_count") != PINNED_RETAINED_RECORDS:
        raise ValueError("Pinned exact corpus retained row count mismatch")
    data_files = _expected_data_files(root, manifest)
    total_rows = sum(int(row["rows"]) for row in data_files)
    total_bytes = sum(int(row["bytes"]) for row in data_files)
    if total_rows != PINNED_RETAINED_RECORDS:
        raise ValueError("Exact data Parquet row counts do not sum to the pinned total")
    return {
        "input_root": str(root),
        "manifest_sha256": actual_sha,
        "retained_records": total_rows,
        "data_files": len(data_files),
        "compressed_bytes": total_bytes,
        "data_file_inventory": data_files,
    }


def _verify_resume_prefix(
    input_root: Path,
    data_files: list[dict[str, Any]],
    next_shard_index: int,
    next_row_ordinal: int,
) -> dict[str, Any]:
    """Recheck byte identities for shards containing committed rows."""
    prefix_count = next_shard_index + int(next_row_ordinal > 0)
    checked_bytes = 0
    checked_shards = 0
    for record in data_files[:prefix_count]:
        expected_sha = record.get("sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError(
                f"Pinned shard has no valid SHA-256 in the corpus manifest: {record['relative']}"
            )
        actual_sha = sha256_file(input_root / record["relative"])
        if actual_sha != expected_sha:
            raise ValueError(
                f"Committed source shard checksum changed since checkpoint: {record['relative']}"
            )
        checked_shards += 1
        checked_bytes += int(record["bytes"])
    return {"shards": checked_shards, "bytes": checked_bytes}


def _corpus_documents(
    input_root: Path,
    data_files: list[dict[str, Any]],
    manifest_sha256: str,
    batch_size: int = 16,
    start_shard_index: int = 0,
    start_row_ordinal: int = 0,
) -> Iterator[CorpusDocument]:
    for shard_index, record in enumerate(data_files):
        if shard_index < start_shard_index:
            continue
        relative = record["relative"]
        path = input_root / relative
        normalized_shard = Path(relative).relative_to("data").as_posix()
        parquet = pq.ParquetFile(path)
        target_ordinal = start_row_ordinal if shard_index == start_shard_index else 0
        row_ordinal = 0
        for row_group in range(parquet.metadata.num_row_groups):
            row_group_start = row_ordinal
            row_group_rows = parquet.metadata.row_group(row_group).num_rows
            row_ordinal += row_group_rows
            if row_group_start + row_group_rows <= target_ordinal:
                continue
            batch_offset = 0
            for batch in parquet.iter_batches(
                batch_size=batch_size,
                row_groups=[row_group],
                columns=["text", "source", "content_sha256"],
                use_threads=False,
            ):
                text_column = batch.column(batch.schema.get_field_index("text"))
                source_column = batch.column(batch.schema.get_field_index("source"))
                hash_column = batch.column(
                    batch.schema.get_field_index("content_sha256")
                )
                for index in range(batch.num_rows):
                    current_ordinal = row_group_start + batch_offset + index
                    if current_ordinal < target_ordinal:
                        continue
                    text = text_column[index].as_py()
                    source = source_column[index].as_py()
                    content_sha256 = hash_column[index].as_py()
                    if not isinstance(text, str) or not source:
                        raise ValueError(
                            f"Invalid normalized record at {normalized_shard}:{current_ordinal}"
                        )
                    actual_content_sha = hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest()
                    if actual_content_sha != content_sha256:
                        raise ValueError(
                            f"Normalized content SHA mismatch at {normalized_shard}:{current_ordinal}"
                        )
                    record_id = occurrence_id_v2(
                        source, normalized_shard, current_ordinal
                    )
                    yield CorpusDocument(
                        doc_id=record_id,
                        text=text,
                        source_shard=normalized_shard,
                        source=source,
                        source_row_ordinal=current_ordinal,
                        input_manifest_sha256=manifest_sha256,
                        source_shard_index=shard_index,
                    )
                batch_offset += batch.num_rows
        if row_ordinal != int(record["rows"]):
            raise ValueError(
                f"Scanned row count differs for {normalized_shard}: "
                f"{row_ordinal} != {record['rows']}"
            )


def _write_hits(
    path: Path, results: Iterable[MatchResult], batch_size: int = 2048
) -> int:
    writer = pq.ParquetWriter(
        path,
        HIT_SCHEMA,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        version="2.6",
    )
    buffer: list[dict[str, Any]] = []
    count = 0
    try:
        for result in results:
            buffer.append(asdict(result))
            if len(buffer) >= batch_size:
                writer.write_table(pa.Table.from_pylist(buffer, schema=HIT_SCHEMA))
                count += len(buffer)
                buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=HIT_SCHEMA))
            count += len(buffer)
        writer.close()
    except Exception:
        writer.close()
        raise
    return count


def _write_anchor_evidence(
    path: Path, evidence: Iterable[AnchorEvidence], batch_size: int = 4096
) -> int:
    """Write all rare-anchor spans in bounded Parquet batches."""
    writer = pq.ParquetWriter(
        path,
        ANCHOR_EVIDENCE_SCHEMA,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        version="2.6",
    )
    buffer: list[dict[str, Any]] = []
    count = 0
    try:
        for item in evidence:
            buffer.append(asdict(item))
            if len(buffer) >= batch_size:
                writer.write_table(
                    pa.Table.from_pylist(buffer, schema=ANCHOR_EVIDENCE_SCHEMA)
                )
                count += len(buffer)
                buffer.clear()
        if buffer:
            writer.write_table(
                pa.Table.from_pylist(buffer, schema=ANCHOR_EVIDENCE_SCHEMA)
            )
            count += len(buffer)
        writer.close()
    except Exception:
        writer.close()
        raise
    return count


def _approved_policy(path: Path) -> tuple[CandidatePolicy, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "SCIENTIST_APPROVED_FOR_BD3":
        raise ValueError("Matcher policy lacks status SCIENTIST_APPROVED_FOR_BD3")
    approval = payload.get("scientist_approval")
    if (
        not isinstance(approval, dict)
        or not approval.get("reference")
        or not approval.get("approver")
    ):
        raise ValueError(
            "Approved matcher policy must identify approver and decision reference"
        )
    policy_values = {
        key: value
        for key, value in payload.items()
        if key in CandidatePolicy.__dataclass_fields__
    }
    policy = CandidatePolicy(**policy_values)
    return policy, approval


def _verify_output_records(
    root: Path,
    manifest: dict[str, Any],
    *,
    allow_incomplete_marker: bool = False,
) -> dict[str, Any]:
    """Verify a prepared or published output directory without source reads."""
    if (root / "INCOMPLETE.json").exists() and not allow_incomplete_marker:
        raise ValueError("BD3 output still has an INCOMPLETE marker")
    if manifest.get("scan_version") != SCAN_VERSION:
        raise ValueError("Unsupported BD3 scan manifest version")
    if manifest.get("status") not in {
        "BD3_SCAN_COMPLETE_PENDING_INDEPENDENT_VERIFICATION",
        "BD3_SCAN_VERIFIED_COMPLETE",
    }:
        raise ValueError("BD3 scan manifest is incomplete")
    expected_outputs = {
        "candidate_hits.parquet",
        "candidate_anchor_evidence.parquet",
        "scan_accounting.parquet",
    }
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != expected_outputs:
        raise ValueError("BD3 output manifest has an unexpected artifact inventory")
    for relative, record in outputs.items():
        path = (root / relative).resolve()
        if root.resolve() not in path.parents:
            raise ValueError(f"Invalid BD3 output artifact path: {relative}")
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError(f"BD3 output missing or size mismatch: {relative}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"BD3 output checksum mismatch: {relative}")
    manifest_hash_path = root / "manifest.sha256"
    if not manifest_hash_path.is_file():
        raise ValueError("BD3 manifest checksum sidecar is missing")
    expected_manifest_sha = manifest_hash_path.read_text(encoding="ascii").strip()
    if sha256_file(root / "manifest.json") != expected_manifest_sha:
        raise ValueError("BD3 manifest checksum mismatch")
    hits = pq.ParquetFile(root / "candidate_hits.parquet")
    evidence = pq.ParquetFile(root / "candidate_anchor_evidence.parquet")
    accounting = pq.ParquetFile(root / "scan_accounting.parquet")
    if not hits.schema_arrow.equals(HIT_SCHEMA):
        raise ValueError("BD3 candidate hit schema mismatch")
    if not evidence.schema_arrow.equals(ANCHOR_EVIDENCE_SCHEMA):
        raise ValueError("BD3 candidate anchor evidence schema mismatch")
    if not accounting.schema_arrow.equals(ACCOUNTING_SCHEMA):
        raise ValueError("BD3 scan accounting schema mismatch")
    if hits.metadata.num_rows != manifest.get("candidate_hit_rows"):
        raise ValueError("BD3 candidate hit count mismatch")
    if evidence.metadata.num_rows != manifest.get("candidate_anchor_evidence_rows"):
        raise ValueError("BD3 candidate anchor evidence count mismatch")
    return {
        "candidate_hit_rows": hits.metadata.num_rows,
        "candidate_anchor_evidence_rows": evidence.metadata.num_rows,
        "scan_accounting_rows": accounting.metadata.num_rows,
        "manifest_sha256": expected_manifest_sha,
    }


def _publish_prepared_stage(
    stage: Path,
    output: Path,
    *,
    run_id: str,
    checkpoint_identity_sha256: str,
) -> dict[str, Any]:
    """Verify and atomically publish a complete same-filesystem stage."""
    if stage.parent.resolve() != output.parent.resolve():
        raise ValueError("BD3 stage and output must share a parent filesystem")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing BD3 output: {output}")
    manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("run_id") != run_id
        or manifest.get("checkpoint_identity_sha256") != checkpoint_identity_sha256
    ):
        raise ValueError("Prepared finalization belongs to another run")
    marker = stage / "INCOMPLETE.json"
    if marker.exists():
        try:
            marker_identity = json.loads(marker.read_text(encoding="utf-8")).get(
                "checkpoint_identity_sha256"
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Finalization INCOMPLETE marker is corrupted") from exc
        if marker_identity != checkpoint_identity_sha256:
            raise ValueError("Finalization INCOMPLETE marker belongs to another run")
    _verify_output_records(stage, manifest, allow_incomplete_marker=True)
    marker.unlink(missing_ok=True)
    _fsync_directory(stage)
    os.replace(stage, output)
    _fsync_directory(output.parent)
    return manifest


def run_bd3_scan(
    *,
    snapshot_dir: Path | str = DEFAULT_SNAPSHOT_ROOT,
    calibration_dir: Path | str = DEFAULT_CALIBRATION_ROOT,
    policy_path: Path | str | None,
    input_root: Path | str = DEFAULT_EXACT_ROOT,
    output_dir: Path | str = DEFAULT_SCAN_ROOT,
    scratch_dir: Path | str | None = None,
    checkpoint_dir: Path | str | None = None,
    run_id: str | None = None,
    resume: bool = False,
    execute_bd3: bool = False,
) -> dict[str, Any]:
    """Preflight by default; scan only with explicit BD3 and scientist approvals."""
    if resume and not execute_bd3:
        raise ValueError(
            "--resume requires --execute-bd3; the default scan is a dry-run"
        )
    snapshot_check = verify_snapshot(snapshot_dir)
    calibration_check = verify_calibration(calibration_dir)
    inputs = inspect_bd3_inputs(input_root)
    data_files = inputs.pop("data_file_inventory")
    output = Path(output_dir).expanduser().resolve()
    if not execute_bd3:
        return {
            "mode": "DRY_RUN",
            "bd3_status": "NOT_RUN",
            "required_explicit_flag": "--execute-bd3",
            "snapshot": snapshot_check,
            "calibration": calibration_check,
            "inputs": inputs,
            "output_dir": str(output),
            "checkpointing": "Persistent resumable checkpoints require explicit execution.",
        }
    if policy_path is None:
        raise ValueError(
            "Scientist-approved policy path is required for an explicit BD3 scan"
        )
    policy, approval = _approved_policy(Path(policy_path))
    if not run_id or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) is None:
        raise ValueError(
            "A unique --run-id containing only letters, numbers, dot, underscore, "
            "or hyphen is required for BD3 execution"
        )
    if checkpoint_dir is not None and scratch_dir is not None:
        checkpoint_value = Path(checkpoint_dir).expanduser().resolve()
        scratch_value = Path(scratch_dir).expanduser().resolve()
        if checkpoint_value != scratch_value:
            raise ValueError(
                "--checkpoint-dir and the legacy --scratch-dir must identify the same directory"
            )
    checkpoint = checkpoint_dir if checkpoint_dir is not None else scratch_dir
    if checkpoint is None:
        raise ValueError(
            "Explicit --checkpoint-dir on persistent local scratch is required"
        )
    checkpoint = Path(checkpoint).expanduser().resolve()
    if output.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite existing BD3 output: {output}")
    if not output.exists() and resume and not checkpoint.is_dir():
        raise FileNotFoundError(
            f"Cannot resume; checkpoint directory is missing: {checkpoint}"
        )

    ordered_inventory = [
        {
            "relative": str(record["relative"]),
            "normalized_shard": Path(record["relative"]).relative_to("data").as_posix(),
            "bytes": int(record["bytes"]),
            "rows": int(record["rows"]),
            "sha256": str(record["sha256"]),
        }
        for record in data_files
    ]
    policy_sha256 = _canonical_sha256(policy.to_dict())
    checkpoint_identity = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_metadata_version": RUN_METADATA_VERSION,
        "run_id": run_id,
        "output_dir": str(output),
        "input_root": str(Path(input_root).expanduser().resolve()),
        "d1_corpus_manifest_sha256": inputs["manifest_sha256"],
        "ordered_source_shard_inventory": ordered_inventory,
        "ordered_source_shard_inventory_sha256": _canonical_sha256(ordered_inventory),
        "benchmark_snapshot_manifest_sha256": snapshot_check["manifest_sha256"],
        "benchmark_calibration_manifest_sha256": calibration_check["manifest_sha256"],
        "matcher_policy_sha256": policy_sha256,
        "matcher_policy": policy.to_dict(),
        "scientist_approval": approval,
        "matcher_implementation_version": MATCHER_VERSION,
        "normalization_implementation_version": NORMALIZATION_VERSION,
    }
    checkpoint_identity_sha256 = _canonical_sha256(checkpoint_identity)
    checkpoint_metadata_path = checkpoint / "run.json"
    checkpoint_database_path = checkpoint / "matcher.sqlite3"
    if resume:
        if not checkpoint.is_dir() or not checkpoint_metadata_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume; checkpoint metadata is missing under {checkpoint}"
            )
        try:
            checkpoint_metadata = json.loads(
                checkpoint_metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Checkpoint run metadata is missing or corrupted") from exc
        if (
            checkpoint_metadata.get("checkpoint_identity_sha256")
            != checkpoint_identity_sha256
        ):
            raise ValueError(
                "Checkpoint run metadata identity does not match this scan"
            )
    else:
        checkpoint.mkdir(parents=True, exist_ok=False)
        _write_json(
            checkpoint_metadata_path,
            {
                "run_metadata_version": RUN_METADATA_VERSION,
                "status": "INCOMPLETE",
                "run_id": run_id,
                "checkpoint_identity_sha256": checkpoint_identity_sha256,
                "checkpoint_identity": checkpoint_identity,
            },
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    matcher = BenchmarkMatcher(
        fields_from_snapshot(Path(snapshot_dir) / "match_fields.parquet"), policy
    )
    with matcher.start_run(
        database_path=checkpoint_database_path,
        resume=resume,
        checkpoint_identity=checkpoint_identity,
        shard_inventory=ordered_inventory,
        profile=True,
    ) as run:
        position = run.resume_position
        print(
            "BD3 recovery position: "
            f"completed_records={position['documents_seen']}, "
            f"next_shard_index={position['shard_index']}, "
            f"next_row_ordinal={position['row_ordinal']}, "
            f"checkpoint_status={run.checkpoint_status}",
            file=sys.stderr,
            flush=True,
        )
        resume_integrity = {"shards": 0, "bytes": 0}
        if resume and run.checkpoint_status in {
            "SCANNING",
            "READY_TO_FINALIZE",
            "FINALIZING",
        }:
            resume_integrity = _verify_resume_prefix(
                Path(input_root),
                data_files,
                position["shard_index"],
                position["row_ordinal"],
            )

        if output.exists():
            if not resume:
                raise FileExistsError(
                    f"Refusing to overwrite existing BD3 output: {output}"
                )
            if run.checkpoint_status not in {"FINALIZED", "PUBLISHED"}:
                raise ValueError(
                    "Output directory exists while its checkpoint is not finalized"
                )
            verified = verify_bd3_scan(output, input_root)
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            if (
                manifest.get("run_id") != run_id
                or manifest.get("checkpoint_identity_sha256")
                != checkpoint_identity_sha256
            ):
                raise ValueError("Existing output does not belong to this checkpoint")
            run.mark_published()
            _write_json(
                checkpoint_metadata_path,
                {
                    "run_metadata_version": RUN_METADATA_VERSION,
                    "status": "PUBLISHED",
                    "run_id": run_id,
                    "checkpoint_identity_sha256": checkpoint_identity_sha256,
                    "checkpoint_identity": checkpoint_identity,
                },
            )
            return {
                **manifest,
                "resume_result": "ALREADY_PUBLISHED",
                "verification": verified,
            }

        if run.checkpoint_status in {"FINALIZED", "PUBLISHED"}:
            if run.checkpoint_status == "PUBLISHED":
                raise ValueError(
                    "Checkpoint says output was published, but its output directory is missing"
                )
            accounting = run.finish()
        else:
            accounting = run.scan(
                _corpus_documents(
                    Path(input_root),
                    data_files,
                    inputs["manifest_sha256"],
                    start_shard_index=position["shard_index"],
                    start_row_ordinal=position["row_ordinal"],
                )
            )

        shard_candidates: dict[str, int] = {}
        for result in run.iter_results():
            shard_candidates[result.source_shard] = (
                shard_candidates.get(result.source_shard, 0) + 1
            )
        accounting_by_shard = {
            item.source_shard: item for item in accounting.shard_accounting
        }
        rows = []
        for record in ordered_inventory:
            item = accounting_by_shard.get(record["normalized_shard"])
            accounted_rows = item.document_count if item else 0
            if accounted_rows != record["rows"]:
                raise ValueError(
                    "BD3 per-shard accounting does not reconcile for "
                    f"{record['normalized_shard']}: {accounted_rows} != {record['rows']}"
                )
            rows.append(
                {
                    "normalized_shard": record["normalized_shard"],
                    "expected_rows": record["rows"],
                    "accounted_rows": accounted_rows,
                    "record_id_sha256": item.record_id_sha256
                    if item
                    else hashlib.sha256().hexdigest(),
                    "candidate_rows": shard_candidates.get(
                        record["normalized_shard"], 0
                    ),
                }
            )
        if accounting.documents_seen != PINNED_RETAINED_RECORDS:
            raise ValueError("BD3 scan did not account for every retained input record")

        stage = output.parent / f".{output.name}.{run_id}.incomplete"
        stage_ready = False
        if stage.exists():
            incomplete_marker = stage / "INCOMPLETE.json"
            if incomplete_marker.is_file():
                try:
                    stage_identity = json.loads(
                        incomplete_marker.read_text(encoding="utf-8")
                    ).get("checkpoint_identity_sha256")
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        "Interrupted finalization marker is corrupted"
                    ) from exc
                if stage_identity != checkpoint_identity_sha256:
                    raise ValueError("Interrupted finalization belongs to another run")
                shutil.rmtree(stage)
            else:
                try:
                    staged_manifest = json.loads(
                        (stage / "manifest.json").read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        "Unmarked finalization directory is not a verified prepared output"
                    ) from exc
                if (
                    staged_manifest.get("checkpoint_identity_sha256")
                    != checkpoint_identity_sha256
                ):
                    raise ValueError("Prepared finalization belongs to another run")
                _verify_output_records(stage, staged_manifest)
                stage_ready = True
        if not stage_ready:
            stage.mkdir(parents=False, exist_ok=False)
            _write_json(
                stage / "INCOMPLETE.json",
                {
                    "status": "INCOMPLETE",
                    "run_id": run_id,
                    "checkpoint_identity_sha256": checkpoint_identity_sha256,
                },
            )
            hit_count = _write_hits(
                stage / "candidate_hits.parquet", run.iter_results()
            )
            anchor_evidence_count = _write_anchor_evidence(
                stage / "candidate_anchor_evidence.parquet",
                run.iter_anchor_evidence(),
            )
            pq.write_table(
                pa.Table.from_pylist(rows, schema=ACCOUNTING_SCHEMA),
                stage / "scan_accounting.parquet",
                compression="zstd",
                version="2.6",
            )
            output_records = {}
            for filename in (
                "candidate_hits.parquet",
                "candidate_anchor_evidence.parquet",
                "scan_accounting.parquet",
            ):
                _fsync_file(stage / filename)
                output_records[filename] = {
                    "bytes": (stage / filename).stat().st_size,
                    "sha256": sha256_file(stage / filename),
                }
            scan_manifest = {
                "scan_version": SCAN_VERSION,
                "status": "BD3_SCAN_COMPLETE_PENDING_INDEPENDENT_VERIFICATION",
                "bd2_snapshot": "COMPLETE",
                "bd2_calibration": "COMPLETE",
                "bd3_production": "COMPLETE_PENDING_VERIFICATION",
                "bd4_review_exclusions": "NOT_RUN",
                "run_id": run_id,
                "checkpoint_identity_sha256": checkpoint_identity_sha256,
                "exact_manifest_sha256": inputs["manifest_sha256"],
                "ordered_source_shard_inventory_sha256": checkpoint_identity[
                    "ordered_source_shard_inventory_sha256"
                ],
                "expected_retained_records": PINNED_RETAINED_RECORDS,
                "accounted_records": accounting.documents_seen,
                "data_files": inputs["data_files"],
                "compressed_input_bytes": inputs["compressed_bytes"],
                "benchmark_snapshot_manifest_sha256": snapshot_check["manifest_sha256"],
                "calibration_manifest_sha256": calibration_check["manifest_sha256"],
                "matcher_version": policy.matcher_version,
                "matcher_implementation_version": MATCHER_VERSION,
                "normalization_version": policy.normalization_version,
                "normalization_implementation_version": NORMALIZATION_VERSION,
                "matcher_policy_sha256": policy_sha256,
                "policy": policy.to_dict(),
                "scientist_approval": approval,
                "candidate_hit_rows": hit_count,
                "candidate_anchor_evidence_rows": anchor_evidence_count,
                "outputs": output_records,
            }
            _write_json(stage / "manifest.json", scan_manifest)
            with (stage / "manifest.sha256").open("w", encoding="ascii") as stream:
                stream.write(sha256_file(stage / "manifest.json") + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            _verify_output_records(stage, scan_manifest, allow_incomplete_marker=True)
        else:
            scan_manifest = json.loads(
                (stage / "manifest.json").read_text(encoding="utf-8")
            )

        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing BD3 output: {output}"
            )
        scan_manifest = _publish_prepared_stage(
            stage,
            output,
            run_id=run_id,
            checkpoint_identity_sha256=checkpoint_identity_sha256,
        )
        run.mark_published()
        _write_json(
            checkpoint_metadata_path,
            {
                "run_metadata_version": RUN_METADATA_VERSION,
                "status": "PUBLISHED",
                "run_id": run_id,
                "checkpoint_identity_sha256": checkpoint_identity_sha256,
                "checkpoint_identity": checkpoint_identity,
                "resume_integrity": resume_integrity,
            },
        )
        return scan_manifest


def verify_bd3_scan(
    scan_dir: Path | str,
    input_root: Path | str = DEFAULT_EXACT_ROOT,
    verify_inputs: bool = False,
) -> dict[str, Any]:
    """Verify output artifacts and shard completeness; optionally rehash input IDs."""
    root = Path(scan_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_check = _verify_output_records(root, manifest)
    if _canonical_sha256(manifest.get("policy")) != manifest.get(
        "matcher_policy_sha256"
    ):
        raise ValueError("BD3 matcher policy checksum mismatch")
    inputs = inspect_bd3_inputs(input_root, manifest["exact_manifest_sha256"])
    expected = {
        row["relative"]: row
        for row in _expected_data_files(
            Path(input_root).expanduser().resolve(),
            json.loads(
                (Path(input_root) / "manifest.json").read_text(encoding="utf-8")
            ),
        )
    }
    accounting_table = pq.read_table(root / "scan_accounting.parquet")
    anchor_evidence_table = pq.read_table(root / "candidate_anchor_evidence.parquet")
    if accounting_table.num_rows != len(expected):
        raise ValueError("BD3 scan does not have one accounting row per source shard")
    accounting_rows = accounting_table.to_pylist()
    if len(accounting_rows) != len(expected):
        raise ValueError("BD3 scan does not have one accounting row per source shard")
    for row in accounting_rows:
        relative = f"data/{row['normalized_shard']}"
        expected_row = expected.get(relative)
        if expected_row is None:
            raise ValueError(f"Unknown source shard in accounting: {relative}")
        if (
            row["accounted_rows"] != expected_row["rows"]
            or row["expected_rows"] != expected_row["rows"]
        ):
            raise ValueError(f"BD3 accounted row count mismatch: {relative}")
    accounted_count = sum(row["accounted_rows"] for row in accounting_rows)
    if accounted_count != manifest.get("accounted_records"):
        raise ValueError("BD3 accounted rows do not match the final manifest")
    if accounted_count != manifest.get("expected_retained_records"):
        raise ValueError("BD3 accounted rows do not match the pinned record total")
    if accounted_count != PINNED_RETAINED_RECORDS:
        raise ValueError(
            "BD3 accounting does not sum to the pinned retained record count"
        )
    input_digest_verified = False
    if verify_inputs:
        expected_id_digest = {
            row["normalized_shard"]: row["record_id_sha256"] for row in accounting_rows
        }
        actual: dict[str, tuple[int, str]] = {}
        for relative in sorted(expected):
            source_relative = Path(relative).relative_to("data").as_posix()
            source = hashlib.sha256()
            parquet = pq.ParquetFile(Path(input_root) / relative)
            row_ordinal = 0
            for batch in parquet.iter_batches(
                batch_size=512,
                columns=["source"],
                use_threads=False,
            ):
                source_column = batch.column(0)
                for index in range(batch.num_rows):
                    record_id = occurrence_id_v2(
                        source_column[index].as_py(), source_relative, row_ordinal
                    )
                    encoded = record_id.encode("ascii")
                    source.update(len(encoded).to_bytes(8, "big"))
                    source.update(encoded)
                    row_ordinal += 1
            actual[source_relative] = (row_ordinal, source.hexdigest())
        for shard, digest in expected_id_digest.items():
            if actual[shard] != (expected[f"data/{shard}"]["rows"], digest):
                raise ValueError(
                    f"BD3 independent source-ID verification failed: {shard}"
                )
        input_digest_verified = True
    return {
        "status": "BD3_SCAN_VERIFIED_COMPLETE"
        if input_digest_verified
        else "BD3_OUTPUTS_AND_ACCOUNTING_VERIFIED",
        "accounted_records": sum(row["accounted_rows"] for row in accounting_rows),
        "candidate_hit_rows": artifact_check["candidate_hit_rows"],
        "candidate_anchor_evidence_rows": anchor_evidence_table.num_rows,
        "manifest_sha256": artifact_check["manifest_sha256"],
        "input_record_id_digests_verified": input_digest_verified,
        "input_data_files": inputs["data_files"],
    }
