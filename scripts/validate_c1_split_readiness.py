"""Bounded C1-S1 evidence packet. No full-corpus assignment or transformation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import pyarrow.parquet as pq

from cambacica.corpus.manifest import compute_file_sha256
from cambacica.corpus.mix import validate_mix_files
from cambacica.corpus.split import (
    DEFAULT_INPUT,
    DEFAULT_POLICY,
    SOURCES,
    build,
    canonical,
    capacity_report,
    verify,
)

BD4 = Path("/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd4/review-v1")


def identities() -> dict:
    review = json.loads((BD4 / "review_manifest-v1.json").read_text())
    files = {
        name: Path(item["path"])
        for name, item in review["source_identities"]["files"].items()
    }
    files["bd4_review_manifest"] = BD4 / "review_manifest-v1.json"
    files.update({"bd4_" + name: BD4 / name for name in review["artifact_checksums"]})
    result = {
        name: dict(path=str(path), sha256=compute_file_sha256(path))
        for name, path in sorted(files.items())
    }
    for name, item in review["source_identities"]["files"].items():
        assert result[name]["sha256"] == item["sha256"]
    for name, item in review["artifact_checksums"].items():
        assert result["bd4_" + name]["sha256"] == item["sha256"]
    assert (
        result["bd4_review_manifest"]["sha256"]
        == "9889c448fd67a3728bdd508d1eb41061413bb0b42f13a38569c33a6b9d944be3"
    )
    manifest = json.loads((DEFAULT_INPUT / "manifest.json").read_text())
    result["d1_payload_stat_snapshot"] = {
        name: dict(
            bytes=(DEFAULT_INPUT / name).stat().st_size,
            mtime_ns=(DEFAULT_INPUT / name).stat().st_mtime_ns,
        )
        for name in sorted(manifest["output_files"])
    }
    assert review["counts"]["excluded_documents"] == 0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root
    before = identities()
    start = time.monotonic()
    manifest = build(DEFAULT_INPUT, output, DEFAULT_POLICY)
    build_seconds = time.monotonic() - start
    start = time.monotonic()
    verification = verify(DEFAULT_INPUT, output, DEFAULT_POLICY, bounded=True)
    verification["elapsed_seconds"] = round(time.monotonic() - start, 3)
    after = identities()
    assert before == after
    configs_paths = sorted(Path("configs").glob("corpus_mix_*.yaml"))
    configs = validate_mix_files(configs_paths)
    for config in configs:
        for source, info in config["sources"].items():
            source = "gigaverbo_v2" if source == "gigaverbo_v2_residual" else source
            if source != "gutenberg_pt":
                assert info["commit_sha"] == SOURCES[source]
            else:
                assert info["snapshot_date"] == "2026-10-01"
    accounting_path = DEFAULT_INPUT / "accounting_by_source_subset.csv"
    exact_manifest = json.loads((DEFAULT_INPUT / "manifest.json").read_text())
    assert (
        compute_file_sha256(accounting_path)
        == exact_manifest["output_files"][accounting_path.name]["sha256"]
    )
    accounting = list(csv.DictReader(accounting_path.open()))
    report = capacity_report(accounting, configs)
    table = pq.read_table(output / "assignments.parquet").to_pylist()
    split_ids = {
        s: {r["doc_id"] for r in table if r["split"] == s}
        for s in ("train", "validation", "test")
    }
    assert sum(map(len, split_ids.values())) == len(table) == manifest["documents"]
    assert not split_ids["train"] & split_ids["validation"]
    assert not split_ids["train"] & split_ids["test"]
    assert not split_ids["validation"] & split_ids["test"]
    groups = {}
    for row in table:
        groups.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(value) == 1 for value in groups.values())
    all_strata = {
        (r["source"], r["subset"]) for r in accounting if r["scope"] == "source_subset"
    }
    assert all_strata == {(r["source"], r["subset"]) for r in table}
    for stratum in all_strata:
        assert {
            r["split"] for r in table if (r["source"], r["subset"]) == stratum
        } == set(split_ids)
    books = {
        s: sum(
            r["normalized_words"]
            for r in table
            if r["source"] == "gutenberg_pt" and r["split"] == s
        )
        for s in split_ids
    }
    metadata = {}
    characterized = Path(
        "/mnt/data/cambacica-base-180m/normalized/characterization/sources"
    )
    for path in sorted(characterized.glob("*.json")):
        doc = json.loads(path.read_text())
        metadata[doc["source"]] = dict(
            evidence_sha256=compute_file_sha256(path),
            scope="pre-D1 full normalization characterization; not remeasured post-D1",
            metadata_coverage=doc["metrics"]["metadata_coverage"],
            word_length_distribution=doc["metrics"]["word_length_distribution"],
        )
    evidence = dict(
        status="BOUNDED_PASS_POLICY_PROPOSED",
        verification=verification,
        build_seconds=round(build_seconds, 3),
        candidate_config_sha256={str(p): compute_file_sha256(p) for p in configs_paths},
        eligible_accounting=accounting,
        gross_no_reuse_capacities=report,
        complete_gutenberg_split_words=books,
        gutenberg_train_word_ceiling_a=books["train"] * 100 // 8,
        gutenberg_train_word_ceiling_b=books["train"] * 100 // 4,
        normalized_metadata_evidence=metadata,
        unchanged_input_identities=after,
        validation=dict(
            unique_ids=len(table),
            unique_groups=len(groups),
            no_split_overlap=True,
            group_integrity=True,
            all_source_subset_strata_covered=len(all_strata),
            candidate_independent_heldouts=True,
            long_documents=sum(r["is_long"] for r in table),
            maximum_words=max(r["normalized_words"] for r in table),
        ),
    )
    (output / "readiness.json").write_text(canonical(evidence) + "\n")
    # SQLite is restart scratch, deliberately outside the scientific checksum set.
    names = [
        "state.json",
        "manifest.json",
        "assignments.parquet",
        "checksums.sha256",
        "readiness.json",
    ]
    artifact_manifest = {
        name: dict(
            bytes=(output / name).stat().st_size,
            sha256=compute_file_sha256(output / name),
        )
        for name in names
    }
    (output / "artifact_manifest.json").write_text(canonical(artifact_manifest) + "\n")
    (output / "artifact_manifest.sha256").write_text(
        f"{compute_file_sha256(output / 'artifact_manifest.json')}  artifact_manifest.json\n"
    )
    print(
        canonical(
            dict(
                status=evidence["status"],
                documents=len(table),
                validation=evidence["validation"],
                artifact_manifest_sha256=compute_file_sha256(
                    output / "artifact_manifest.json"
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
