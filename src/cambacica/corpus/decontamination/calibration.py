"""Run separated fixture calibration and emit deterministic evidence artifacts."""

from __future__ import annotations

from dataclasses import asdict, replace
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import tempfile
import time
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .fixtures import (
    FixtureCase,
    benchmark_derived_cases,
    synthetic_cases,
    synthetic_fields,
)
from .matcher import (
    AnchorEvidence,
    BenchmarkMatcher,
    CandidatePolicy,
    MatchField,
    MatchResult,
    fields_from_snapshot,
)
from .snapshot import canonical_json, sha256_file, verify_snapshot


DEFAULT_CALIBRATION_ROOT = Path(
    "/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2/calibration"
)
CALIBRATION_VERSION = "c1-bd2-calibration-v1"

FIXTURE_RESULT_SCHEMA = pa.schema(
    [
        pa.field("policy_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("case_id", pa.string(), nullable=False),
        pa.field("control_source", pa.string(), nullable=False),
        pa.field("expected_match", pa.bool_(), nullable=False),
        pa.field("detected", pa.bool_(), nullable=False),
        pa.field("expected_example_id", pa.string()),
        pa.field("matched_example_ids_json", pa.string(), nullable=False),
        pa.field("matched_source_row_ids_json", pa.string(), nullable=False),
        pa.field("matched_fields_json", pa.string(), nullable=False),
        pa.field("matched_rules_json", pa.string(), nullable=False),
        pa.field("max_contiguous_tokens", pa.int32(), nullable=False),
        pa.field("max_distinctive_anchor_count", pa.int32(), nullable=False),
        pa.field("max_distinctive_token_coverage", pa.float64(), nullable=False),
        pa.field("matched_anchor_document_frequency_min", pa.int64()),
        pa.field("matched_anchor_document_frequency_max", pa.int64()),
        pa.field("matched_spans_json", pa.string(), nullable=False),
        pa.field("document_text_sha256", pa.string(), nullable=False),
        pa.field("document_characters", pa.int64(), nullable=False),
    ]
)

CASE_REPORT_FIELDS = (
    "policy_id",
    "split",
    "case_id",
    "control_source",
    "expected_match",
    "detected",
    "expected_example_id",
    "matched_example_ids_json",
    "matched_fields_json",
    "matched_rules_json",
    "matched_anchor_document_frequency_min",
    "matched_anchor_document_frequency_max",
)


def _load_registry(snapshot_dir: Path) -> list[dict[str, Any]]:
    return pq.read_table(snapshot_dir / "benchmark_registry.parquet").to_pylist()


def _rusage_peak_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return int(value * 1024 if value < 1_000_000_000 else value)


def _case_control_source(case: FixtureCase) -> str:
    return (
        "benchmark_derived"
        if case.document.source == "benchmark_derived_control"
        else "synthetic"
    )


def _case_rows(
    cases: list[FixtureCase],
    hits: list[MatchResult],
    anchor_evidence: list[AnchorEvidence],
    policy_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_doc: dict[str, list[MatchResult]] = {}
    for hit in hits:
        by_doc.setdefault(hit.doc_id, []).append(hit)
    evidence_by_doc: dict[str, list[AnchorEvidence]] = {}
    for evidence in anchor_evidence:
        evidence_by_doc.setdefault(evidence.doc_id, []).append(evidence)
    rows: list[dict[str, Any]] = []
    totals = {
        "cases": len(cases),
        "positive_controls": sum(case.expected_match for case in cases),
        "negative_controls": sum(not case.expected_match for case in cases),
        "true_positive_controls": 0,
        "missed_positive_controls": 0,
        "false_positive_controls": 0,
        "true_negative_controls": 0,
    }
    for case in sorted(cases, key=lambda item: item.case_id):
        case_hits = sorted(
            by_doc.get(case.document.doc_id, []),
            key=lambda item: (
                item.example_id,
                item.field_role,
                item.decision_rule,
                item.corpus_token_start,
            ),
        )
        case_anchor_evidence = sorted(
            evidence_by_doc.get(case.document.doc_id, []),
            key=lambda item: (
                item.example_id,
                item.field_role,
                item.benchmark_token_start,
                item.corpus_token_start,
                item.anchor_sha256,
            ),
        )
        detected = bool(case_hits)
        if case.expected_match and detected:
            totals["true_positive_controls"] += 1
        elif case.expected_match:
            totals["missed_positive_controls"] += 1
        elif detected:
            totals["false_positive_controls"] += 1
        else:
            totals["true_negative_controls"] += 1
        rows.append(
            {
                "policy_id": policy_id,
                "split": case.split,
                "case_id": case.case_id,
                "control_source": _case_control_source(case),
                "expected_match": case.expected_match,
                "detected": detected,
                "expected_example_id": case.expected_example_id,
                "matched_example_ids_json": canonical_json(
                    sorted({hit.example_id for hit in case_hits})
                ),
                "matched_source_row_ids_json": canonical_json(
                    sorted({hit.source_row_id for hit in case_hits})
                ),
                "matched_fields_json": canonical_json(
                    sorted({hit.field_id for hit in case_hits})
                ),
                "matched_rules_json": canonical_json(
                    sorted({hit.decision_rule for hit in case_hits})
                ),
                "max_contiguous_tokens": max(
                    (hit.contiguous_tokens for hit in case_hits), default=0
                ),
                "max_distinctive_anchor_count": max(
                    (hit.distinctive_anchor_count for hit in case_hits), default=0
                ),
                "max_distinctive_token_coverage": max(
                    (hit.distinctive_token_coverage for hit in case_hits), default=0.0
                ),
                "matched_anchor_document_frequency_min": min(
                    (
                        hit.matched_anchor_document_frequency_min
                        for hit in case_hits
                        if hit.matched_anchor_document_frequency_min is not None
                    ),
                    default=None,
                ),
                "matched_anchor_document_frequency_max": max(
                    (
                        hit.matched_anchor_document_frequency_max
                        for hit in case_hits
                        if hit.matched_anchor_document_frequency_max is not None
                    ),
                    default=None,
                ),
                "matched_spans_json": canonical_json(
                    [
                        {
                            "evidence_type": "candidate_hit_span",
                            "doc_id": hit.doc_id,
                            "example_id": hit.example_id,
                            "field_id": hit.field_id,
                            "field_role": hit.field_role,
                            "rule": hit.decision_rule,
                            "corpus_token_start": hit.corpus_token_start,
                            "corpus_token_end": hit.corpus_token_end,
                            "corpus_char_start": hit.corpus_char_start,
                            "corpus_char_end": hit.corpus_char_end,
                            "benchmark_token_start": hit.benchmark_token_start,
                            "benchmark_token_end": hit.benchmark_token_end,
                            "benchmark_char_start": hit.benchmark_char_start,
                            "benchmark_char_end": hit.benchmark_char_end,
                        }
                        for hit in case_hits
                    ]
                    + [
                        {
                            "evidence_type": "matched_anchor",
                            "doc_id": evidence.doc_id,
                            "example_id": evidence.example_id,
                            "field_id": evidence.field_id,
                            "field_role": evidence.field_role,
                            "anchor_sha256": evidence.anchor_sha256,
                            "anchor_text": evidence.anchor_text,
                            "anchor_document_frequency": evidence.anchor_document_frequency,
                            "corpus_token_start": evidence.corpus_token_start,
                            "corpus_token_end": evidence.corpus_token_end,
                            "corpus_char_start": evidence.corpus_char_start,
                            "corpus_char_end": evidence.corpus_char_end,
                            "benchmark_token_start": evidence.benchmark_token_start,
                            "benchmark_token_end": evidence.benchmark_token_end,
                            "benchmark_char_start": evidence.benchmark_char_start,
                            "benchmark_char_end": evidence.benchmark_char_end,
                        }
                        for evidence in case_anchor_evidence
                    ]
                ),
                "document_text_sha256": hashlib.sha256(
                    case.document.text.encode("utf-8")
                ).hexdigest(),
                "document_characters": len(case.document.text),
            }
        )
    return rows, totals


def _evaluate_policy(
    fields: list[MatchField],
    cases: list[FixtureCase],
    policy: CandidatePolicy,
    scratch_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], int, int, tuple[dict[str, Any], ...]]:
    rows: list[dict[str, Any]] = []
    totals: dict[str, Any] = {}
    peak_scratch = 0
    peak_docs = 0
    accounting: list[dict[str, Any]] = []
    for split in ("development", "heldout"):
        split_cases = [case for case in cases if case.split == split]
        if not split_cases:
            continue
        matcher = BenchmarkMatcher(fields, policy)
        with matcher.start_run(scratch_dir=scratch_dir) as run:
            scan_accounting = run.scan(case.document for case in split_cases)
            hits = list(run.iter_results())
            anchor_evidence = list(run.iter_anchor_evidence())
            rows_for_split, counts = _case_rows(
                split_cases, hits, anchor_evidence, policy.policy_version
            )
            rows.extend(rows_for_split)
            totals[split] = counts
            peak_scratch = max(peak_scratch, run.scratch_bytes)
            peak_docs = max(peak_docs, scan_accounting.documents_seen)
            accounting.append(
                {
                    "split": split,
                    "documents_seen": scan_accounting.documents_seen,
                    "candidate_rows": len(hits),
                    "shards": [
                        asdict(item) for item in scan_accounting.shard_accounting
                    ],
                }
            )
    return rows, totals, peak_scratch, peak_docs, tuple(accounting)


def _candidate_policies() -> tuple[tuple[str, CandidatePolicy], ...]:
    base = CandidatePolicy()
    baseline = replace(
        base,
        policy_version="c1-bd2-baseline-50-token-only-v1",
        distinctive_coverage=1.0,
        minimum_rare_anchors=100_000,
    )
    candidate = replace(base, policy_version="c1-bd2-candidate-50-or-80-2-v1")
    strict = replace(
        base,
        policy_version="c1-bd2-strict-50-or-90-3-v1",
        distinctive_coverage=0.90,
        minimum_rare_anchors=3,
    )
    return (
        ("baseline_50_only", baseline),
        ("candidate_50_or_80_2", candidate),
        ("strict_50_or_90_3", strict),
    )


def _json_write(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def verify_calibration(calibration_dir: Path | str) -> dict[str, Any]:
    root = Path(calibration_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing calibration manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("calibration_version") != CALIBRATION_VERSION:
        raise ValueError("Unsupported calibration artifact version")
    if manifest.get("status") != "BD2_CALIBRATION_COMPLETE":
        raise ValueError("Calibration artifacts are not complete")
    for relative, record in manifest["outputs"].items():
        path = root / relative
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError(
                f"Calibration output is missing or has changed: {relative}"
            )
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Calibration checksum mismatch: {relative}")
    table = pq.read_table(root / "fixture_results.parquet")
    if not table.schema.equals(FIXTURE_RESULT_SCHEMA):
        raise ValueError("Calibration result schema mismatch")
    if table.num_rows != manifest["fixture_result_rows"]:
        raise ValueError("Calibration fixture row count mismatch")
    return {
        "status": manifest["status"],
        "fixture_result_rows": table.num_rows,
        "manifest_sha256": sha256_file(manifest_path),
    }


def run_calibration(
    snapshot_dir: Path | str,
    output_dir: Path | str = DEFAULT_CALIBRATION_ROOT,
    scratch_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Compare the candidate policy on separated controls and publish artifacts."""
    snapshot = Path(snapshot_dir).expanduser().resolve()
    snapshot_check = verify_snapshot(snapshot)
    output = Path(output_dir).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        return verify_calibration(output)
    if scratch_dir is None:
        scratch = Path(tempfile.mkdtemp(prefix="cambacica-bd2-calibration-"))
        remove_scratch_dir = True
    else:
        scratch = Path(scratch_dir).expanduser().resolve()
        scratch.mkdir(parents=True, exist_ok=True)
        remove_scratch_dir = False
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".incomplete", dir=output.parent
        )
    )
    started = time.monotonic()
    try:
        registry_rows = _load_registry(snapshot)
        snapshot_fields = fields_from_snapshot(snapshot / "match_fields.parquet")
        synthetic = synthetic_fields()
        all_fields = snapshot_fields + synthetic
        cases = synthetic_cases() + benchmark_derived_cases(registry_rows)

        dev_expected = {
            case.expected_example_id
            for case in cases
            if case.split == "development" and case.expected_example_id
        }
        heldout_expected = {
            case.expected_example_id
            for case in cases
            if case.split == "heldout" and case.expected_example_id
        }
        if dev_expected & heldout_expected:
            raise ValueError("A benchmark example appears in both fixture partitions")
        if not any(
            case.case_id == "dev_taxonomy_shared_references_a" for case in cases
        ):
            raise ValueError("D2d shared-taxonomy-reference negative is missing")

        comparison_rows: list[dict[str, Any]] = []
        policy_metrics: dict[str, Any] = {}
        candidate_rows: list[dict[str, Any]] = []
        scratch_peak = 0
        docs_scanned_peak = 0
        accounting_by_policy: dict[str, Any] = {}
        policies = _candidate_policies()
        for label, policy in policies:
            rows, metrics, peak_bytes, docs_seen, accounting = _evaluate_policy(
                all_fields, cases, policy, scratch
            )
            policy_metrics[label] = metrics
            scratch_peak = max(scratch_peak, peak_bytes)
            docs_scanned_peak = max(docs_scanned_peak, docs_seen)
            accounting_by_policy[label] = accounting
            comparison_rows.extend(rows)
            if label == "candidate_50_or_80_2":
                candidate_rows = rows

        if not candidate_rows:
            raise ValueError("Candidate policy produced no fixture results")
        fp_fn = [
            row for row in candidate_rows if row["expected_match"] != row["detected"]
        ]
        false_positives = [row for row in fp_fn if row["detected"]]
        missed_positives = [row for row in fp_fn if not row["detected"]]
        if false_positives or missed_positives:
            raise ValueError(
                "Candidate policy failed one or more deterministic controls; "
                "inspect case-level outputs before marking BD2 calibration complete"
            )

        candidate_policy = next(
            policy for label, policy in policies if label == "candidate_50_or_80_2"
        )
        _json_write(
            stage / "matching_policy_candidate.json",
            {
                **candidate_policy.to_dict(),
                "status": "CANDIDATE_REQUIRES_SCIENTIST_APPROVAL",
                "freeze_for_bd3": False,
                "calibration_scope": "deterministic synthetic and selected benchmark-derived controls",
                "limitations": [
                    "Fixture results do not estimate population precision, recall, or corpus contamination prevalence.",
                    "max_ngram_df=100 is a candidate cutoff, not calibrated against corpus-wide document frequencies.",
                    "The two-rare-anchor and 80% coverage rule needs a larger, blinded source-stratified calibration before BD3.",
                    "Semantic paraphrase, translation, OCR/image text, and most answer-only inference remain outside detector coverage.",
                ],
            },
        )
        pq.write_table(
            pa.Table.from_pylist(comparison_rows, schema=FIXTURE_RESULT_SCHEMA),
            stage / "fixture_results.parquet",
            compression="zstd",
            version="2.6",
            use_dictionary=True,
        )
        summary = {
            "status": "BD2_CALIBRATION_COMPLETE",
            "candidate_policy": "candidate_50_or_80_2",
            "candidate_policy_predeclared": True,
            "policy_selection_note": (
                "The candidate thresholds were declared from the BD1 plan before "
                "fixture execution. Fixture results compare policies; they do not "
                "select or freeze a production threshold."
            ),
            "policy_comparison": policy_metrics,
            "development_case_ids": sorted(
                case.case_id for case in cases if case.split == "development"
            ),
            "heldout_case_ids": sorted(
                case.case_id for case in cases if case.split == "heldout"
            ),
            "development_example_ids": sorted(dev_expected),
            "heldout_example_ids": sorted(heldout_expected),
            "benchmark_partition_disjoint": not bool(dev_expected & heldout_expected),
            "provisional_frequency_cutoff": {
                "max_ngram_df": candidate_policy.max_ngram_df,
                "document_frequency_unit": "distinct corpus record IDs",
                "benchmark_rarity": f"anchor occurs in <= {candidate_policy.max_benchmark_item_df} benchmark example IDs",
                "frequency_threshold_status": "NOT SCIENTIFICALLY FROZEN",
            },
            "resource_usage": {
                "documents_per_policy_split": docs_scanned_peak,
                "total_fixture_documents_per_policy": len(cases),
                "peak_sqlite_scratch_bytes": scratch_peak,
                "peak_rss_bytes_process_high_water": _rusage_peak_bytes(),
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "gpu_used": False,
            },
            "accounting_by_policy": {
                label: list(items) for label, items in accounting_by_policy.items()
            },
            "non_population_claim": (
                "Fixture counts describe only these deterministic controls and do not estimate population-level matcher quality or contamination prevalence."
            ),
        }
        _json_write(stage / "calibration_summary.json", summary)
        for name, rows in (
            ("false_positive_cases.csv", false_positives),
            ("missed_positive_cases.csv", missed_positives),
        ):
            with (stage / name).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=CASE_REPORT_FIELDS)
                writer.writeheader()
                writer.writerows(
                    {key: row.get(key) for key in CASE_REPORT_FIELDS} for row in rows
                )
        artifact_names = (
            "matching_policy_candidate.json",
            "fixture_results.parquet",
            "calibration_summary.json",
            "false_positive_cases.csv",
            "missed_positive_cases.csv",
        )
        outputs = {
            name: {
                "bytes": (stage / name).stat().st_size,
                "sha256": sha256_file(stage / name),
            }
            for name in artifact_names
        }
        manifest = {
            "calibration_version": CALIBRATION_VERSION,
            "status": "BD2_CALIBRATION_COMPLETE",
            "bd2_snapshot": "COMPLETE",
            "bd3_production": "NOT_RUN",
            "bd4_review_exclusions": "NOT_RUN",
            "snapshot_manifest_sha256": snapshot_check["manifest_sha256"],
            "fixture_result_rows": len(comparison_rows),
            "candidate_policy": candidate_policy.policy_version,
            "candidate_false_positive_controls": len(false_positives),
            "candidate_missed_positive_controls": len(missed_positives),
            "outputs": outputs,
        }
        _json_write(stage / "manifest.json", manifest)
        verify_calibration(stage)
        os.replace(stage, output)
        return verify_calibration(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        if remove_scratch_dir:
            shutil.rmtree(scratch, ignore_errors=True)
