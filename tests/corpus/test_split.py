"""Scientific split invariants on synthetic frozen corpora; no acquisition."""

import csv
import json
from pathlib import Path
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cambacica.corpus import split
from cambacica.corpus.manifest import compute_file_sha256
from cambacica.corpus.mix import validate_mix_files


def document(source, identity, text="Texto português.", subset="fixture"):
    return dict(
        text=text,
        source=source,
        subset=subset,
        source_revision=split.SOURCES[source],
        original_id=identity,
        raw_source_file="upstream.fixture",
        raw_record_identifier=identity,
        content_sha256=split.hashlib.sha256(text.encode()).hexdigest(),
        domain_category="fixture_domain",
    )


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    root = tmp_path / "input"
    files = {}
    accounting = []
    count = 0
    for source in split.SOURCES:
        rows = [document(source, str(i), f"Texto português {i}.") for i in range(12)]
        # Preserve repeated text in Parlamento, and repeated native IDs in GV.
        if source == "parlamento_pt":
            rows[1] = document(source, "1", rows[0]["text"])
        if source == "gigaverbo_v2":
            rows[1]["original_id"] = rows[0]["original_id"]
        if source == "gutenberg_pt":
            rows[0] = document(source, "0", "palavra " * 100001)
        path = root / "data" / source / "part-00000.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=4)
        files[path.relative_to(root).as_posix()] = dict(
            bytes=path.stat().st_size, rows=len(rows), sha256=compute_file_sha256(path)
        )
        count += len(rows)
        for scope, subset in (("source_total", ""), ("source_subset", "fixture")):
            accounting.append(
                dict(
                    source=source,
                    subset=subset,
                    scope=scope,
                    documents_after=len(rows),
                    normalized_words_after=sum(len(r["text"].split()) for r in rows),
                )
            )
    accounting_path = root / "accounting_by_source_subset.csv"
    with accounting_path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(accounting[0]))
        writer.writeheader()
        writer.writerows(accounting)
    files[accounting_path.name] = dict(
        bytes=accounting_path.stat().st_size,
        sha256=compute_file_sha256(accounting_path),
    )
    (root / "manifest.json").write_text(
        json.dumps(dict(output_files=files, retained_record_count=count))
    )
    identity = compute_file_sha256(root / "manifest.json")
    policy = json.loads(split.DEFAULT_POLICY.read_text())
    policy["input_manifest_sha256"] = identity
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    monkeypatch.setattr(split, "PINNED_INPUT", identity)
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            dict(
                policy_sha256=compute_file_sha256(policy_path),
                input_manifest_sha256=identity,
                scientist="fixture",
                decision_reference="synthetic-only",
                action="approve_full_split_manifest_and_verification",
            )
        )
    )
    return root, policy_path, approval


def test_order_batches_restart_groups_and_accounting(corpus, tmp_path):
    root, policy, approval = corpus
    output = tmp_path / "a"
    first = split.build(
        root, output, policy, bounded=False, approval=approval, batch_size=1
    )
    second = split.build(
        root,
        tmp_path / "b",
        policy,
        bounded=False,
        approval=approval,
        batch_size=5,
        reverse=True,
    )
    assert first == second
    assert (
        split.verify(root, output, policy, bounded=False, approval=approval)["status"]
        == "PASS"
    )
    assert (
        split.build(
            root, output, policy, bounded=False, approval=approval, reverse=True
        )
        == first
    )
    table = pq.read_table(output / "assignments.parquet").to_pylist()
    assert len(table) == len({r["doc_id"] for r in table}) == 60
    assert sum(r["normalized_words"] for r in table) == sum(
        r["normalized_words"] for r in first["accounting"]
    )
    groups = {}
    for row in table:
        groups.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in groups.values())
    assert sum(r["is_long"] for r in table) == 1
    assert next(r for r in table if r["is_long"])["normalized_words"] == 100001
    for source in split.SOURCES:
        assert {r["split"] for r in table if r["source"] == source} == {
            "train",
            "validation",
            "test",
        }
    # Partial uncommitted shard work is rolled back before a restart.
    with sqlite3.connect(output / "index.sqlite") as conn:
        conn.execute("DELETE FROM refs WHERE source='carolina'")
        conn.execute("DELETE FROM completed WHERE path LIKE 'data/carolina/%'")
    assert split.build(root, output, policy, bounded=False, approval=approval) == first


@pytest.mark.parametrize(
    "n,expected",
    [
        (1, (1, 0, 0)),
        (2, (1, 1, 0)),
        (3, (1, 1, 1)),
        (26, (24, 1, 1)),
        (655, (641, 7, 7)),
        (1000, (980, 10, 10)),
    ],
)
def test_tiny_strata(n, expected):
    assert split.group_counts(n) == expected


@pytest.mark.parametrize("field", split.COLUMNS)
def test_missing_metadata_fails(field):
    policy = split.load_policy(split.DEFAULT_POLICY)
    row = document("carolina", "a")
    del row[field]
    with pytest.raises(ValueError, match="Missing required metadata"):
        split.reference(row, "carolina/fixture.parquet", 0, policy)


def test_bad_text_revision_policy_and_approval(tmp_path):
    policy = split.load_policy(split.DEFAULT_POLICY)
    row = document("carolina", "a")
    row["text"] += " changed"
    with pytest.raises(ValueError, match="checksum"):
        split.reference(row, "carolina/fixture.parquet", 0, policy)
    row = document("carolina", "a")
    row["source_revision"] = "main"
    with pytest.raises(ValueError, match="revision"):
        split.reference(row, "carolina/fixture.parquet", 0, policy)
    with pytest.raises(ValueError, match="approval"):
        split.authorize(split.DEFAULT_POLICY, None)
    policy["ratios_basis_points"]["train"] = 9000
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="Unsupported"):
        split.load_policy(p)


def test_changed_manifest_and_checksum_rejected(corpus, tmp_path):
    root, policy, _ = corpus
    output = tmp_path / "output"
    split.build(root, output, policy)
    (output / "checksums.sha256").write_text("wrong")
    with pytest.raises(ValueError, match="Checksum manifest"):
        split.verify(root, output, policy, bounded=True)
    (root / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="Pinned input"):
        split.build(root, output, policy)


def test_heldout_is_independent_of_candidate(corpus, tmp_path):
    root, policy, _ = corpus
    output = tmp_path / "output"
    split.build(root, output, policy)
    configs = validate_mix_files(sorted(Path("configs").glob("corpus_mix_*.yaml")))
    table = pq.read_table(output / "assignments.parquet").to_pylist()
    held = {r["doc_id"] for r in table if r["split"] != "train"}
    for config in configs:
        # Every view is permitted to reference only this shared training pool.
        sources = {
            "gigaverbo_v2" if s == "gigaverbo_v2_residual" else s
            for s in config["sources"]
        }
        train = {
            r["doc_id"]
            for r in table
            if r["split"] == "train" and r["source"] in sources
        }
        assert not train & held
        assert {r["doc_id"] for r in table if r["split"] != "train"} == held


def test_interrupted_shard_rolls_back_then_restarts(corpus, tmp_path, monkeypatch):
    root, policy, approval = corpus
    output = tmp_path / "interrupted"
    original = split.rows_from_file

    def interrupted(*args):
        yield next(original(*args))
        raise RuntimeError("synthetic process interruption")

    monkeypatch.setattr(split, "rows_from_file", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        split.build(root, output, policy, bounded=False, approval=approval)
    with sqlite3.connect(output / "index.sqlite") as connection:
        assert connection.execute("SELECT count(*) FROM refs").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM completed").fetchone()[0] == 0
    monkeypatch.setattr(split, "rows_from_file", original)
    assert (
        split.build(root, output, policy, bounded=False, approval=approval)["documents"]
        == 60
    )
    assert (
        split.verify(root, output, policy, bounded=False, approval=approval)["status"]
        == "PASS"
    )


def test_full_requires_approval_and_checks_every_payload(corpus, tmp_path):
    root, policy, approval = corpus
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="approval"):
        split.build(root, output, policy, bounded=False)
    assert not output.exists()
    payload = next((root / "data").rglob("*.parquet"))
    content = payload.read_bytes()
    payload.write_bytes(b"bad" + content[3:])
    with pytest.raises(ValueError, match="Input shard checksum"):
        split.build(root, output, policy, bounded=False, approval=approval)


def test_capacity_constraints_and_absent_source():
    configs = validate_mix_files(sorted(Path("configs").glob("corpus_mix_*.yaml")))
    rows = []
    for source in split.SOURCES:
        rows.append(
            dict(
                source=source,
                subset="",
                scope="source_total",
                normalized_words_after=1000,
            )
        )
    for subset in configs[0]["sources"]["gigaverbo_v2_residual"]["subsets"]:
        rows.append(
            dict(
                source="gigaverbo_v2",
                subset=subset,
                scope="source_subset",
                normalized_words_after=1000,
            )
        )
    report = split.capacity_report(rows, configs)
    assert report["corpus_mix_a"]["limiting"]["max_total_words_floor"] == 2500
    with pytest.raises(KeyError):
        split.capacity_report(rows[1:], configs)
