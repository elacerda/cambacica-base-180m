"""Bounded synthetic tests for C1-S2 selection, retrieval, and verification."""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cambacica.corpus import tokenizer_sample as sample
from cambacica.corpus.dedup.exact_pipeline import occurrence_id_v2
from cambacica.corpus.manifest import compute_file_sha256


SOURCE_SCHEMA = pa.schema(
    [
        pa.field("text", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("source_revision", pa.string()),
        pa.field("subset", pa.string()),
        pa.field("original_id", pa.string()),
        pa.field("original_url", pa.string()),
        pa.field("license", pa.string()),
        pa.field("language", pa.string()),
        pa.field("language_score", pa.float32()),
        pa.field("variety", pa.string()),
        pa.field("quality_score", pa.float32()),
        pa.field("publication_date", pa.string()),
        pa.field("domain_category", pa.string()),
        pa.field("content_sha256", pa.string(), nullable=False),
        pa.field("title", pa.string()),
        pa.field("raw_source_file", pa.string(), nullable=False),
        pa.field("raw_record_identifier", pa.string(), nullable=False),
        pa.field("normalization_version", pa.string(), nullable=False),
        pa.field("_gv2_upstream_shard", pa.string()),
        pa.field("_gv2_upstream_row_group", pa.int32()),
        pa.field("_gv2_upstream_commit", pa.string()),
        pa.field("upstream_metadata_json", pa.string()),
    ]
)


def _text(words: int, marker: str) -> str:
    return " ".join([marker] + ["palavra"] * (words - 1))


def _source_row(source: str, subset: str, identity: str, text: str) -> dict:
    shard = "fixture.parquet"
    return dict(
        text=text,
        source=source,
        source_revision=sample.SOURCE_REVISIONS[source],
        subset=subset,
        original_id=identity,
        original_url=f"https://fixture.invalid/{identity}",
        license="fixture-license",
        language="pt",
        language_score=1.0,
        variety="fixture",
        quality_score=1.0,
        publication_date=None,
        domain_category="fixture_domain",
        content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        title=f"Título {identity}",
        raw_source_file=shard,
        raw_record_identifier=identity,
        normalization_version="fixture-normalization-v1",
        _gv2_upstream_shard="fixture-shard" if source == "gigaverbo_v2" else None,
        _gv2_upstream_row_group=0 if source == "gigaverbo_v2" else None,
        _gv2_upstream_commit=sample.SOURCE_REVISIONS[source]
        if source == "gigaverbo_v2"
        else None,
        upstream_metadata_json="{}" if source == "gigaverbo_v2" else None,
    )


def _group_id(source: str, subset: str, row: dict) -> str:
    key = row["content_sha256"] if source == "parlamento_pt" else row["original_id"]
    encoded = json.dumps(
        [source, subset, key], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(("cambacica-s1-group-v1\0" + encoded).encode()).hexdigest()


def _fixture(
    tmp_path: Path, target_words: int = 10_000, reverse_assignment: bool = False
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    policy = json.loads(sample.DEFAULT_POLICY.read_text())
    policy["target_normalized_words"] = target_words
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy, sort_keys=True))
    input_root = tmp_path / "exact"
    split_root = tmp_path / ("split-reversed" if reverse_assignment else "split")
    input_root.mkdir()
    split_root.mkdir()
    quotas = sample._quota_plan(policy)
    d1_entries = {}
    split_entries = {}
    assignment_rows = []
    test_doc_ids = set()

    for bucket, plan in quotas.items():
        source = plan["source"]
        subset = plan["subset"] or "fixture"
        if source == "gigaverbo_v2":
            source_shard = f"gigaverbo_v2/subset={subset}/part-00000.parquet"
        else:
            source_shard = f"{source}/part-00000.parquet"
        rows = []
        lengths = (
            int(plan["target"]) // 2,
            int(plan["target"]) - int(plan["target"]) // 2,
        )
        for index, words in enumerate(lengths):
            rows.append(
                _source_row(
                    source,
                    subset,
                    f"{bucket}-train-{index}",
                    _text(words, f"id{bucket}{index}"),
                )
            )
        rows.append(
            _source_row(source, subset, f"{bucket}-test", _text(25, f"heldout{bucket}"))
        )
        source_path = input_root / "data" / source_shard
        source_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(rows, schema=SOURCE_SCHEMA),
            source_path,
            row_group_size=2,
        )
        entry = dict(
            bytes=source_path.stat().st_size,
            rows=len(rows),
            sha256=compute_file_sha256(source_path),
        )
        d1_entries[f"data/{source_shard}"] = entry
        split_entries[f"data/{source_shard}"] = entry
        for ordinal, row in enumerate(rows):
            doc_id = occurrence_id_v2(source, source_shard, ordinal)
            assignment_rows.append(
                dict(
                    doc_id=doc_id,
                    group_id=_group_id(source, subset, row),
                    source=source,
                    subset=subset,
                    domain_category=row["domain_category"],
                    source_shard=source_shard,
                    source_row_ordinal=ordinal,
                    content_sha256=row["content_sha256"],
                    normalized_words=len(row["text"].split()),
                    is_long=False,
                    split="test" if ordinal == 2 else "train",
                )
            )
            if ordinal == 2:
                test_doc_ids.add(doc_id)

    d1_manifest = dict(
        schema_version=1,
        status="COMPLETE",
        retained_record_count=sum(item["rows"] for item in d1_entries.values()),
        output_files=d1_entries,
    )
    (input_root / "manifest.json").write_text(json.dumps(d1_manifest, sort_keys=True))
    d1_sha = compute_file_sha256(input_root / "manifest.json")
    if reverse_assignment:
        assignment_rows.reverse()
    assignments_path = split_root / "assignments.parquet"
    pq.write_table(
        pa.Table.from_pylist(assignment_rows), assignments_path, row_group_size=7
    )
    assignments_sha = compute_file_sha256(assignments_path)
    split_manifest = dict(
        version="c1-s1-v1",
        input_manifest_sha256=d1_sha,
        output_sha256=assignments_sha,
        documents=len(assignment_rows),
        input_files=split_entries,
    )
    (split_root / "manifest.json").write_text(
        json.dumps(split_manifest, sort_keys=True)
    )
    split_sha = compute_file_sha256(split_root / "manifest.json")
    (split_root / "checksums.sha256").write_text(
        f"{split_sha}  manifest.json\n{assignments_sha}  assignments.parquet\n"
    )
    policy["input_d1_manifest_sha256"] = d1_sha
    policy["input_split_manifest_sha256"] = split_sha
    policy["input_assignments_sha256"] = assignments_sha
    policy_path.write_text(json.dumps(policy, sort_keys=True))
    return dict(
        input_root=input_root,
        split_root=split_root,
        policy_path=policy_path,
        d1_sha=d1_sha,
        split_sha=split_sha,
        assignments_sha=assignments_sha,
        test_doc_ids=test_doc_ids,
        target_words=target_words,
    )


def _patch_pins(monkeypatch, fixture):
    monkeypatch.setattr(sample, "PINNED_D1_MANIFEST_SHA256", fixture["d1_sha"])
    monkeypatch.setattr(sample, "PINNED_SPLIT_MANIFEST_SHA256", fixture["split_sha"])
    monkeypatch.setattr(sample, "PINNED_ASSIGNMENTS_SHA256", fixture["assignments_sha"])
    monkeypatch.setattr(
        sample,
        "PINNED_POLICY_SHA256",
        compute_file_sha256(fixture["policy_path"]),
    )


def test_selection_is_order_and_batch_independent_and_train_only(tmp_path, monkeypatch):
    first = _fixture(tmp_path / "first")
    _patch_pins(monkeypatch, first)
    first_result = sample.select(
        first["input_root"],
        first["split_root"],
        tmp_path / "selection-a",
        tmp_path / "scratch-a",
        first["policy_path"],
        batch_size=1,
    )
    first_refs = pq.read_table(
        tmp_path / "selection-a" / "selected_references.parquet"
    ).to_pylist()
    assert first_result["status"] == "PASS"
    assert first_result["selected_normalized_words"] == first["target_words"]
    assert len(first_refs) == len({row["doc_id"] for row in first_refs})
    assert all(row["split"] == "train" for row in first_refs)
    assert not ({row["doc_id"] for row in first_refs} & first["test_doc_ids"])

    second = _fixture(tmp_path / "second", reverse_assignment=True)
    _patch_pins(monkeypatch, second)
    second_result = sample.select(
        second["input_root"],
        second["split_root"],
        tmp_path / "selection-b",
        tmp_path / "scratch-b",
        second["policy_path"],
        batch_size=5,
        reverse_row_groups=True,
    )
    second_refs_path = tmp_path / "selection-b" / "selected_references.parquet"
    assert second_result["status"] == "PASS"
    assert compute_file_sha256(
        tmp_path / "selection-a" / "selected_references.parquet"
    ) == compute_file_sha256(second_refs_path)
    assert first_refs == pq.read_table(second_refs_path).to_pylist()


def test_select_resume_rolls_back_partial_row_group(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path / "resume")
    _patch_pins(monkeypatch, fixture)
    original = sample._iter_assignment_batches

    def interrupted(*args, **kwargs):
        iterator = original(*args, **kwargs)
        yield next(iterator)
        raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(sample, "_iter_assignment_batches", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        sample.select(
            fixture["input_root"],
            fixture["split_root"],
            tmp_path / "resume-selection",
            tmp_path / "resume-scratch",
            fixture["policy_path"],
            batch_size=1,
        )
    monkeypatch.setattr(sample, "_iter_assignment_batches", original)
    result = sample.select(
        fixture["input_root"],
        fixture["split_root"],
        tmp_path / "resume-selection",
        tmp_path / "resume-scratch",
        fixture["policy_path"],
        batch_size=3,
    )
    assert result["status"] == "PASS"
    assert result["selected_normalized_words"] == fixture["target_words"]


def test_materialize_resume_preserves_complete_text_and_verify_checks_output(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path / "materialize")
    _patch_pins(monkeypatch, fixture)
    selection_dir = tmp_path / "selection"
    selected = sample.select(
        fixture["input_root"],
        fixture["split_root"],
        selection_dir,
        tmp_path / "selection-scratch",
        fixture["policy_path"],
        batch_size=4,
    )
    assert selected["status"] == "PASS"
    selected_refs = pq.read_table(
        selection_dir / "selected_references.parquet"
    ).to_pylist()
    assert {row["source_row_group"] for row in selected_refs} == {0}

    original_parquet_file = sample.pq.ParquetFile
    source_row_groups_read = []

    class TrackedParquetFile:
        def __init__(self, path):
            self.path = Path(path)
            self.inner = original_parquet_file(path)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def iter_batches(self, *args, **kwargs):
            if self.path.resolve().is_relative_to(fixture["input_root"].resolve()):
                source_row_groups_read.extend(kwargs.get("row_groups", []))
            return self.inner.iter_batches(*args, **kwargs)

    monkeypatch.setattr(sample.pq, "ParquetFile", TrackedParquetFile)

    original_writer = sample._write_materialized_shard
    call_count = 0

    def fail_on_second_shard(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("synthetic materialization interruption")
        return original_writer(*args, **kwargs)

    monkeypatch.setattr(sample, "_write_materialized_shard", fail_on_second_shard)
    sample_dir = tmp_path / "sample"
    with pytest.raises(RuntimeError, match="materialization interruption"):
        sample.materialize(
            fixture["input_root"],
            fixture["split_root"],
            selection_dir,
            sample_dir,
            tmp_path / "sample-scratch",
            fixture["policy_path"],
            batch_size=2,
        )
    assert not sample_dir.exists()
    assert sample_dir.with_name(sample_dir.name + ".partial").exists()

    monkeypatch.setattr(sample, "_write_materialized_shard", original_writer)
    materialized = sample.materialize(
        fixture["input_root"],
        fixture["split_root"],
        selection_dir,
        sample_dir,
        tmp_path / "sample-scratch",
        fixture["policy_path"],
        batch_size=3,
    )
    assert materialized["status"] == "MATERIALIZED"
    observed_rows = []
    for name in materialized["artifacts"]:
        observed_rows.extend(pq.read_table(sample_dir / name).to_pylist())
    assert len(observed_rows) == selected["selected_documents"]
    assert all(row["split"] == "train" for row in observed_rows)
    assert all(
        hashlib.sha256(row["text"].encode()).hexdigest() == row["content_sha256"]
        and len(row["text"].split()) == row["normalized_words"]
        for row in observed_rows
    )
    assert all(
        row["original_id"] and row["raw_record_identifier"] for row in observed_rows
    )
    assert source_row_groups_read and set(source_row_groups_read) == {0}

    verified = sample.verify(
        fixture["input_root"],
        fixture["split_root"],
        selection_dir,
        sample_dir,
        tmp_path / "verify-scratch",
        fixture["policy_path"],
        batch_size=3,
    )
    assert verified["status"] == "PASS"
    assert verified["normalized_words"] == fixture["target_words"]

    first_artifact = next(iter(materialized["artifacts"]))
    artifact_path = sample_dir / first_artifact
    artifact_path.write_bytes(artifact_path.read_bytes()[:-1] + b"X")
    with pytest.raises(ValueError, match="integrity failure"):
        sample.verify(
            fixture["input_root"],
            fixture["split_root"],
            selection_dir,
            sample_dir,
            tmp_path / "verify-scratch",
            fixture["policy_path"],
            batch_size=2,
        )


def test_quota_deviation_is_reported_and_fails_closed(tmp_path):
    policy = json.loads(sample.DEFAULT_POLICY.read_text())
    policy["target_normalized_words"] = 10_000
    quotas = sample._quota_plan(policy)
    summaries = {
        bucket: {
            "eligible_train_documents": 2,
            "eligible_train_words": plan["target"],
            "selected_documents": 1,
            "selected_words": plan["target"],
            "target_words": plan["target"],
        }
        for bucket, plan in quotas.items()
    }
    summaries["gutenberg_pt"]["selected_words"] = 0
    report = sample._quota_report(summaries, policy)
    gutenberg = next(
        item for item in report["categories"] if item["bucket"] == "gutenberg_pt"
    )
    assert report["status"] == "QUOTA_TOLERANCE_FAILED"
    assert gutenberg["deviation_words"] == -400
    assert not gutenberg["within_tolerance"]


def test_sqlite_scratch_is_rejected_on_nfs(tmp_path, monkeypatch):
    input_root = tmp_path / "input"
    split_root = tmp_path / "split"
    scratch = tmp_path / "scratch"
    output = tmp_path / "output"
    monkeypatch.setattr(sample, "_mount_type", lambda path: "nfs4")
    with pytest.raises(ValueError, match="local storage"):
        sample._require_separate_paths(output, scratch, (input_root, split_root))


def test_source_text_hash_mismatch_is_rejected(tmp_path):
    fixture = _fixture(tmp_path / "hash")
    refs = pq.read_table(fixture["split_root"] / "assignments.parquet").to_pylist()
    reference = next(row for row in refs if row["split"] == "train")
    source_path = fixture["input_root"] / "data" / reference["source_shard"]
    row = pq.read_table(source_path).to_pylist()[reference["source_row_ordinal"]]
    row["text"] = row["text"].replace("palavra", "palavrw", 1)
    with pytest.raises(ValueError, match="identity or normalized-text hash"):
        sample._check_source_row(row, reference)
