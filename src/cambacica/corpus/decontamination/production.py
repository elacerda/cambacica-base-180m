"""Explicitly gated read-only C1-BD3 scan and independent verification."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import tempfile
import time
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from cambacica.corpus.dedup.exact_pipeline import occurrence_id_v2

from .calibration import verify_calibration
from .matcher import (
    AnchorEvidence,
    BenchmarkMatcher,
    CandidatePolicy,
    CorpusDocument,
    MatchResult,
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
SCAN_VERSION = "c1-bd3-read-only-scan-v1"

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
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


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


def _corpus_documents(
    input_root: Path,
    data_files: list[dict[str, Any]],
    manifest_sha256: str,
    batch_size: int = 16,
) -> Iterator[CorpusDocument]:
    for record in data_files:
        relative = record["relative"]
        path = input_root / relative
        normalized_shard = Path(relative).relative_to("data").as_posix()
        parquet = pq.ParquetFile(path)
        row_ordinal = 0
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=["text", "source", "content_sha256"],
            use_threads=False,
        ):
            text_column = batch.column(batch.schema.get_field_index("text"))
            source_column = batch.column(batch.schema.get_field_index("source"))
            hash_column = batch.column(batch.schema.get_field_index("content_sha256"))
            for index in range(batch.num_rows):
                text = text_column[index].as_py()
                source = source_column[index].as_py()
                content_sha256 = hash_column[index].as_py()
                if not isinstance(text, str) or not source:
                    raise ValueError(
                        f"Invalid normalized record at {normalized_shard}:{row_ordinal}"
                    )
                actual_content_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if actual_content_sha != content_sha256:
                    raise ValueError(
                        f"Normalized content SHA mismatch at {normalized_shard}:{row_ordinal}"
                    )
                record_id = occurrence_id_v2(source, normalized_shard, row_ordinal)
                yield CorpusDocument(
                    doc_id=record_id,
                    text=text,
                    source_shard=normalized_shard,
                    source=source,
                    source_row_ordinal=row_ordinal,
                    input_manifest_sha256=manifest_sha256,
                )
                row_ordinal += 1
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


def run_bd3_scan(
    *,
    snapshot_dir: Path | str = DEFAULT_SNAPSHOT_ROOT,
    calibration_dir: Path | str = DEFAULT_CALIBRATION_ROOT,
    policy_path: Path | str | None,
    input_root: Path | str = DEFAULT_EXACT_ROOT,
    output_dir: Path | str = DEFAULT_SCAN_ROOT,
    scratch_dir: Path | str | None = None,
    execute_bd3: bool = False,
) -> dict[str, Any]:
    """Preflight by default; scan only with explicit BD3 and scientist approvals."""
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
            "scan_estimated_temporary_storage": (
                "Not estimated from the fixtures; provision local scratch for the "
                "candidate index and review-output spill before manual approval."
            ),
        }
    if policy_path is None:
        raise ValueError(
            "Scientist-approved policy path is required for an explicit BD3 scan"
        )
    policy, approval = _approved_policy(Path(policy_path))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing BD3 output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".incomplete", dir=output.parent
        )
    )
    scratch = Path(scratch_dir).expanduser().resolve() if scratch_dir else None
    if scratch is not None:
        scratch.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        matcher = BenchmarkMatcher(
            fields_from_snapshot(Path(snapshot_dir) / "match_fields.parquet"), policy
        )
        with matcher.start_run(scratch_dir=scratch) as run:
            accounting = run.scan(
                _corpus_documents(
                    Path(input_root), data_files, inputs["manifest_sha256"]
                )
            )
            hit_count = _write_hits(
                stage / "candidate_hits.parquet", run.iter_results()
            )
            anchor_evidence_count = _write_anchor_evidence(
                stage / "candidate_anchor_evidence.parquet",
                run.iter_anchor_evidence(),
            )
            shard_candidates: dict[str, int] = {}
            for result in run.iter_results():
                shard_candidates[result.source_shard] = (
                    shard_candidates.get(result.source_shard, 0) + 1
                )
            rows = []
            for item in accounting.shard_accounting:
                matching_file = next(
                    row
                    for row in data_files
                    if Path(row["relative"]).relative_to("data").as_posix()
                    == item.source_shard
                )
                rows.append(
                    {
                        "normalized_shard": item.source_shard,
                        "expected_rows": int(matching_file["rows"]),
                        "accounted_rows": item.document_count,
                        "record_id_sha256": item.record_id_sha256,
                        "candidate_rows": shard_candidates.get(item.source_shard, 0),
                    }
                )
        pq.write_table(
            pa.Table.from_pylist(rows, schema=ACCOUNTING_SCHEMA),
            stage / "scan_accounting.parquet",
            compression="zstd",
            version="2.6",
        )
        if accounting.documents_seen != PINNED_RETAINED_RECORDS:
            raise ValueError("BD3 scan did not account for every retained input record")
        if any(row["expected_rows"] != row["accounted_rows"] for row in rows):
            raise ValueError("BD3 per-shard accounting does not reconcile")
        output_records = {
            filename: {
                "bytes": (stage / filename).stat().st_size,
                "sha256": sha256_file(stage / filename),
            }
            for filename in (
                "candidate_hits.parquet",
                "candidate_anchor_evidence.parquet",
                "scan_accounting.parquet",
            )
        }
        scan_manifest = {
            "scan_version": SCAN_VERSION,
            "status": "BD3_SCAN_COMPLETE_PENDING_INDEPENDENT_VERIFICATION",
            "bd2_snapshot": "COMPLETE",
            "bd2_calibration": "COMPLETE",
            "bd3_production": "COMPLETE_PENDING_VERIFICATION",
            "bd4_review_exclusions": "NOT_RUN",
            "exact_manifest_sha256": inputs["manifest_sha256"],
            "expected_retained_records": PINNED_RETAINED_RECORDS,
            "accounted_records": accounting.documents_seen,
            "data_files": inputs["data_files"],
            "compressed_input_bytes": inputs["compressed_bytes"],
            "benchmark_snapshot_manifest_sha256": snapshot_check["manifest_sha256"],
            "calibration_manifest_sha256": calibration_check["manifest_sha256"],
            "matcher_version": policy.matcher_version,
            "normalization_version": policy.normalization_version,
            "policy": policy.to_dict(),
            "scientist_approval": approval,
            "candidate_hit_rows": hit_count,
            "candidate_anchor_evidence_rows": anchor_evidence_count,
            "resource_usage": {
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "peak_rss_bytes_process_high_water": _rss_bytes(),
                "gpu_used": False,
                "scratch_directory": str(scratch)
                if scratch
                else str(Path(tempfile.gettempdir())),
            },
            "outputs": output_records,
        }
        _write_json(stage / "manifest.json", scan_manifest)
        os.replace(stage, output)
        return scan_manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_bd3_scan(
    scan_dir: Path | str,
    input_root: Path | str = DEFAULT_EXACT_ROOT,
    verify_inputs: bool = False,
) -> dict[str, Any]:
    """Verify output artifacts and shard completeness; optionally rehash input IDs."""
    root = Path(scan_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("scan_version") != SCAN_VERSION:
        raise ValueError("Unsupported BD3 scan manifest version")
    if manifest.get("status") not in {
        "BD3_SCAN_COMPLETE_PENDING_INDEPENDENT_VERIFICATION",
        "BD3_SCAN_VERIFIED_COMPLETE",
    }:
        raise ValueError("BD3 scan manifest is incomplete")
    for relative, record in manifest["outputs"].items():
        path = root / relative
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError(f"BD3 output missing or size mismatch: {relative}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"BD3 output checksum mismatch: {relative}")
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
    if not accounting_table.schema.equals(ACCOUNTING_SCHEMA):
        raise ValueError("BD3 scan accounting schema mismatch")
    anchor_evidence_table = pq.read_table(root / "candidate_anchor_evidence.parquet")
    if not anchor_evidence_table.schema.equals(ANCHOR_EVIDENCE_SCHEMA):
        raise ValueError("BD3 candidate anchor evidence schema mismatch")
    if anchor_evidence_table.num_rows != manifest.get("candidate_anchor_evidence_rows"):
        raise ValueError("BD3 candidate anchor evidence count mismatch")
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
    if sum(row["accounted_rows"] for row in accounting_rows) != PINNED_RETAINED_RECORDS:
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
        "candidate_hit_rows": int(
            pq.ParquetFile(root / "candidate_hits.parquet").metadata.num_rows
        ),
        "candidate_anchor_evidence_rows": anchor_evidence_table.num_rows,
        "input_record_id_digests_verified": input_digest_verified,
        "input_data_files": inputs["data_files"],
    }
