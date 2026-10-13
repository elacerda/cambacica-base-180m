#!/usr/bin/env python3
"""Create the fixed, deterministic C1-BD4 decision artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


CORPUS = Path("/mnt/data/cambacica-base-180m/deduplicated/exact")
SCAN = Path("/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd3/scan-bd3-20261009T201945Z-cenpesor")
SNAPSHOT = Path("/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2/snapshot")
VERIFY_LOG = Path("/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd3/logs/bd3-20261009T201945Z-cenpesor-verification.log")
REQUESTED_OUTPUT = Path("/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd4/review-v1")
D1_SHA = "57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428"
BD3_SHA = "20e55353720f75d4883889f7fd9b7109b3f82e18f0ec4cf05384765d7bae3010"
SNAPSHOT_SHA = "d9b5bfdec543af37fdb8733239b0f8b0255339454b5ebca8880371bb86f77cda"
DATE, REFERENCE = "2026-10-13", "C1-BD4-2026-10-13"
CLASSIFICATION = "SHORT_GENERIC_PHRASE_OVERLAP"
COUNTS = {
    "Qual é a composição química da água?": 15,
    "O que os cientistas queriam descobrir?": 3,
    "Onde foi realizada a cerimônia fúnebre?": 1,
    "Qual era o propósito da Grande Pirâmide?": 1,
}
RATIONALES = {
    "Qual é a composição química da água?": "Natural educational, chemistry, water-treatment, and philosophical contexts; some occurrences start longer hydrogen-peroxide questions.",
    "O que os cientistas queriam descobrir?": "Independent journalism about different scientific subjects.",
    "Onde foi realizada a cerimônia fúnebre?": "A historical account of a different ceremony in Belém do Pará.",
    "Qual era o propósito da Grande Pirâmide?": "An independently contextualized discussion of the pyramids of Giza.",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jbytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def occurrence_id(source: str, shard: str, ordinal: int) -> str:
    identity = {"source": source, "normalized_shard": shard, "row_ordinal": ordinal}
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(b"cambacica-occurrence-id-v2\0" + encoded).hexdigest()


def check_rows(corpus: Path, d1: dict, docs: dict[str, dict]) -> None:
    shards: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for doc_id, doc in docs.items():
        path = f"data/{doc['source_shard']}"
        entry = d1["output_files"].get(path)
        ordinal = doc["source_row_ordinal"]
        if entry is None or not 0 <= ordinal < entry["rows"]:
            raise ValueError(f"Document outside pinned D1 corpus: {doc_id}")
        shards[path].append((doc_id, ordinal))
    for path, requested in sorted(shards.items()):
        parquet = pq.ParquetFile(corpus / path)
        bounds, start = [], 0
        for index in range(parquet.num_row_groups):
            end = start + parquet.metadata.row_group(index).num_rows
            bounds.append((start, end))
            start = end
        if start != d1["output_files"][path]["rows"]:
            raise ValueError(f"D1 shard row count changed: {path}")
        groups: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for doc_id, ordinal in requested:
            group = next((i for i, (lo, hi) in enumerate(bounds) if lo <= ordinal < hi), None)
            if group is None:
                raise ValueError(f"Missing D1 row: {doc_id}")
            groups[group].append((doc_id, ordinal))
        for group, rows in sorted(groups.items()):
            batch = parquet.read_row_group(group, columns=["text", "source", "content_sha256"])
            for doc_id, ordinal in rows:
                row = batch.slice(ordinal - bounds[group][0], 1).to_pylist()[0]
                doc = docs[doc_id]
                text_sha = hashlib.sha256(row["text"].encode()).hexdigest()
                if (row["source"] != doc["source"] or row["content_sha256"] != text_sha
                    or doc["document_text_sha256"] != text_sha
                    or occurrence_id(doc["source"], doc["source_shard"], ordinal) != doc_id):
                    raise ValueError(f"D1 record identity/checksum mismatch: {doc_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, default=CORPUS)
    parser.add_argument("--scan-dir", type=Path, default=SCAN)
    parser.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    corpus, scan, snapshot, output = (p.resolve() for p in (args.corpus_root, args.scan_dir, args.snapshot_dir, args.output_dir))
    if any(output == p or p in output.parents for p in (corpus, scan, snapshot)):
        raise ValueError("Output must remain outside pinned inputs")

    d1_path, bd3_path = corpus / "manifest.json", scan / "manifest.json"
    snap_path, hits_path = snapshot / "benchmark_manifest.json", scan / "candidate_hits.parquet"
    anchor_path, accounting_path = scan / "candidate_anchor_evidence.parquet", scan / "scan_accounting.parquet"
    fields_path = snapshot / "match_fields.parquet"
    inputs = {
        "post_d1_manifest": d1_path,
        "bd3_manifest": bd3_path,
        "bd3_manifest_sha256": scan / "manifest.sha256",
        "candidate_hits": hits_path,
        "candidate_anchor_evidence": anchor_path,
        "scan_accounting": accounting_path,
        "benchmark_snapshot_manifest": snap_path,
        "benchmark_match_fields": fields_path,
        "verification_log": VERIFY_LOG,
    }
    before = {name: (p.stat().st_size, sha(p)) for name, p in inputs.items()}
    d1, bd3, snap = (json.loads(p.read_text(encoding="utf-8")) for p in (d1_path, bd3_path, snap_path))
    if before["post_d1_manifest"][1] != D1_SHA or d1.get("retained_record_count") != 21_603_689:
        raise ValueError("Post-D1 manifest identity differs from pin")
    if (before["bd3_manifest"][1] != BD3_SHA or bd3.get("exact_manifest_sha256") != D1_SHA
        or bd3.get("status") != "BD3_SCAN_COMPLETE_PENDING_INDEPENDENT_VERIFICATION"
        or inputs["bd3_manifest_sha256"].read_text().strip() != BD3_SHA):
        raise ValueError("Published BD3 manifest identity or historical status changed")
    if before["benchmark_snapshot_manifest"][1] != SNAPSHOT_SHA or snap.get("status") != "BD2_SNAPSHOT_COMPLETE":
        raise ValueError("Benchmark snapshot identity differs from pin")
    for name, path in (("candidate_hits.parquet", hits_path), ("candidate_anchor_evidence.parquet", anchor_path), ("scan_accounting.parquet", accounting_path)):
        expected = bd3["outputs"][name]
        if (path.stat().st_size, sha(path)) != (expected["bytes"], expected["sha256"]):
            raise ValueError(f"BD3 artifact checksum mismatch: {name}")
    expected = snap["outputs"]["match_fields.parquet"]
    if (fields_path.stat().st_size, sha(fields_path)) != (expected["bytes"], expected["sha256"]):
        raise ValueError("Benchmark fields checksum mismatch")

    verify_text = VERIFY_LOG.read_text(encoding="utf-8")
    verify = json.loads(verify_text.split("\n\tCommand being timed:", 1)[0])
    if ("VERIFICATION_EXIT_STATUS=0" not in verify_text or verify.get("status") != "BD3_SCAN_VERIFIED_COMPLETE"
        or verify.get("manifest_sha256") != BD3_SHA or verify.get("input_record_id_digests_verified") is not True
        or verify.get("accounted_records") != 21_603_689 or verify.get("input_data_files") != 532
        or verify.get("candidate_hit_rows") != 20 or verify.get("candidate_anchor_evidence_rows") != 0):
        raise ValueError("Independent BD3 verification result differs from pin")

    hits, accounting = pq.read_table(hits_path).to_pylist(), pq.read_table(accounting_path).to_pylist()
    if (len(hits) != 20 or pq.ParquetFile(anchor_path).metadata.num_rows != 0 or len(accounting) != 532
        or sum(r["accounted_rows"] for r in accounting) != 21_603_689
        or any(r["expected_rows"] != r["accounted_rows"] for r in accounting)
        or sum(r["candidate_rows"] for r in accounting) != 20):
        raise ValueError("BD3 output counts do not reconcile")

    benchmark = {r["benchmark_name"]: r for r in snap["benchmarks"]}["belebele_por_latn"]
    if (benchmark["repository"], benchmark["revision"], benchmark["logical_configuration"], benchmark["logical_split"]) != (
        "facebook/belebele", "d4c91dedc9de484dbea7b7d940f898f59fd135e9", "por_Latn", "test"):
        raise ValueError("Matched Belebele snapshot identity differs from pin")
    selected_fields = pq.read_table(fields_path, columns=[
        "benchmark_name", "example_id", "field_id", "field_role", "original_text",
        "source_row_id", "source_file_sha256", "token_count",
    ]).to_pylist()
    fields = {(r["benchmark_name"], r["example_id"], r["field_id"]): r for r in selected_fields
        if r["benchmark_name"] == "belebele_por_latn" and r["field_role"] == "question"
        and r["original_text"] in RATIONALES}
    per_example = Counter(r["example_id"] for r in hits)
    examples = {}
    for hit in hits:
        field = fields.get((hit["benchmark_name"], hit["example_id"], hit["field_id"]))
        if (field is None or hit["field_role"] != "question" or hit["decision_rule"] != "exact_distinctive_question"
            or hit["exact_match"] is not True or hit["source_row_id"] != field["source_row_id"]
            or hit["source_file_sha256"] != field["source_file_sha256"] or hit["matched_tokens"] not in (6, 7)
            or hit["matched_tokens"] != hit["contiguous_tokens"] or hit["matched_tokens"] != field["token_count"]
            or hit["source"] != "gigaverbo_v2" or hit["input_manifest_sha256"] != D1_SHA):
            raise ValueError("A candidate hit cannot be reconciled with its pinned question field")
        examples[hit["example_id"]] = {"example_id": hit["example_id"], "field_id": hit["field_id"],
            "question": field["original_text"], "source_row_id": field["source_row_id"],
            "source_file_sha256": field["source_file_sha256"], "hit_count": per_example[hit["example_id"]]}
    if (len(fields) != 4 or {examples[k]["question"]: n for k, n in per_example.items()} != COUNTS
        or {r["benchmark_name"] for r in hits} != {"belebele_por_latn"}):
        raise ValueError("BD3 candidate examples or counts differ from approved review")

    indexed = sorted(enumerate(hits), key=lambda x: (x[1]["doc_id"], x[1]["corpus_char_start"],
        x[1]["corpus_char_end"], x[1]["benchmark_name"], x[1]["example_id"], x[1]["field_id"]))
    reviewed, docs = [], {}
    for number, (source_ordinal, hit) in enumerate(indexed, 1):
        doc = {k: hit[k] for k in ("doc_id", "source", "source_shard", "source_row_ordinal", "document_text_sha256", "input_manifest_sha256")}
        if docs.setdefault(hit["doc_id"], doc) != doc:
            raise ValueError(f"Conflicting hit identity: {hit['doc_id']}")
        question = examples[hit["example_id"]]["question"]
        reviewed.append({"review_id": f"BD4-H{number:02d}", "bd3_output_row_ordinal": source_ordinal,
            "benchmark_question": question, "classification": CLASSIFICATION, "decision": "NO_EXCLUSION",
            "decision_date": DATE, "decision_reference": REFERENCE, "contextual_rationale": RATIONALES[question],
            "bd3_hit": hit})
    if len(docs) != 18 or {r["bd3_output_row_ordinal"] for r in reviewed} != set(range(20)):
        raise ValueError("Expected 18 documents and every BD3 row exactly once")
    check_rows(corpus, d1, docs)
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for row in reviewed:
        by_doc[row["bd3_hit"]["doc_id"]].append(row)
    decisions = []
    for doc_id, doc in sorted(docs.items()):
        rows = by_doc[doc_id]
        decisions.append({**doc, "decision": "RETAIN", "classification": CLASSIFICATION,
            "hit_count": len(rows), "review_ids_json": json.dumps([r["review_id"] for r in rows], separators=(",", ":")),
            "benchmark_example_ids_json": json.dumps(sorted({r["bd3_hit"]["example_id"] for r in rows}), separators=(",", ":"))})
    exclusion = {"schema": "cambacica.c1.bd4.exclusion-manifest.v1", "version": 1,
        "scope": "first_training_run", "decision_date": DATE, "decision_reference": REFERENCE,
        "source_post_d1_manifest_sha256": D1_SHA, "excluded_document_count": 0,
        "excluded_document_ids": [], "interpretation": "Zero document IDs are excluded by this decision."}

    output.mkdir(parents=True, exist_ok=True)
    hit_file, doc_file = output / "reviewed_hits-v1.jsonl", output / "document_decisions-v1.csv"
    exclusion_file, note_file = output / "exclusion_manifest-v1.json", output / "FINAL_REVIEW_README.md"
    manifest_file, sum_file = output / "review_manifest-v1.json", output / "review_manifest-v1.sha256"
    hit_file.write_bytes(b"".join(jbytes(r) for r in reviewed))
    with doc_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(decisions[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(decisions)
    exclusion_file.write_bytes(jbytes(exclusion))
    note_file.write_text("# BD4 final review-v1\n\nThe project scientist approved the contextual review on 2026-10-13. "
        "All 20 hits are `SHORT_GENERIC_PHRASE_OVERLAP`; all 18 documents are RETAIN and zero IDs are excluded.\n\n"
        "The 2026-10-10 candidate matrix and provisional recommendation are historical agent-proposed assessments. "
        "This decision supersedes them; the earlier packet remains unchanged.\n", encoding="utf-8", newline="\n")
    artifact_files = (hit_file, doc_file, exclusion_file, note_file)
    artifact_hashes = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in artifact_files}
    source_hashes = {name: {"path": str(p), "bytes": before[name][0], "sha256": before[name][1]} for name, p in inputs.items()}
    manifest = {"schema": "cambacica.c1.bd4.review-manifest.v1", "version": 1,
        "status": "COMPLETE_SCIENTIST_APPROVED",
        "decision": {"date": DATE, "reference": REFERENCE, "basis": "Scientist-approved contextual review",
            "classification": CLASSIFICATION, "document_action": "RETAIN", "excluded_document_count": 0},
        "requested_artifact_directory": str(REQUESTED_OUTPUT),
        "source_identities": {"post_d1_manifest_sha256": D1_SHA, "post_d1_retained_documents": 21_603_689,
            "bd3_run_id": bd3["run_id"], "bd3_manifest_sha256": BD3_SHA,
            "bd3_manifest_status_preserved": bd3["status"], "bd3_verification_status": verify["status"],
            "benchmark_snapshot_manifest_sha256": SNAPSHOT_SHA, "benchmark_repository": benchmark["repository"],
            "benchmark_revision": benchmark["revision"], "benchmark_configuration": benchmark["logical_configuration"],
            "benchmark_split": benchmark["logical_split"], "benchmark_declared_license": benchmark["license_declared"],
            "benchmark_provenance_caveat": benchmark["provenance_caveat"], "files": source_hashes},
        "counts": {"reviewed_hit_rows": 20, "distinct_documents": 18, "matched_benchmark_examples": 4,
            "excluded_documents": 0, "calame_pt_candidate_hits": 0, "approximate_anchor_evidence_rows": 0,
            "classification_counts": {CLASSIFICATION: 20},
            "question_counts": sorted(examples.values(), key=lambda r: (-r["hit_count"], r["question"]))},
        "validation": {"all_hit_rows_accounted_for_once": True, "all_documents_resolved_in_post_d1_corpus": True,
            "all_document_decisions_retain": True, "excluded_document_ids_explicitly_empty": True,
            "published_bd3_manifest_and_outputs_verified": True, "input_checksums_unchanged_during_generation": True,
            "serialization": "UTF-8 canonical JSON; stable row ordering"},
        "artifact_checksums": artifact_hashes,
        "methodological_limitations": [
            "The scientist reviewed contextual excerpts, not an independent full-document comparison against every benchmark passage.",
            "The reported hits are short exact question matches; lexical matching does not detect every paraphrase, translation, answer-only exposure, or other unobserved form.",
            "This decision concerns the available matching evidence and does not prove complete absence of benchmark contamination.",
        ]}
    manifest_file.write_bytes(jbytes(manifest))
    manifest_sha = sha(manifest_file)
    sum_file.write_text(f"{manifest_sha}  {manifest_file.name}\n", encoding="ascii")
    if {name: (p.stat().st_size, sha(p)) for name, p in inputs.items()} != before:
        raise ValueError("Pinned source files changed during BD4 generation")
    with doc_file.open(encoding="utf-8", newline="") as stream:
        written_docs = list(csv.DictReader(stream))
    if (len(hit_file.read_text(encoding="utf-8").splitlines()) != 20 or len(written_docs) != 18
        or any(row["decision"] != "RETAIN" for row in written_docs)
        or json.loads(exclusion_file.read_text(encoding="utf-8"))["excluded_document_ids"] != []):
        raise ValueError("Written BD4 artifacts failed validation")
    print(json.dumps({"review_manifest_sha256": manifest_sha, "reviewed_hit_rows": 20,
        "distinct_documents": 18, "matched_benchmark_examples": 4, "excluded_documents": 0,
        "artifact_checksums": artifact_hashes}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
