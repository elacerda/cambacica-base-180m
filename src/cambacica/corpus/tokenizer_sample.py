"""Deterministic, train-only whole-document C1 tokenizer sample generator.

The production workflow is deliberately split into three commands:
``select`` ranks C1-S1 references, ``materialize`` retrieves only selected D1
records, and ``verify`` independently rebuilds selection and checks the
published records.  SQLite state belongs on local scratch; published files
belong on the configured corpus filesystem.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from cambacica.corpus.dedup.exact_pipeline import occurrence_id_v2
from cambacica.corpus.manifest import compute_file_sha256

DEFAULT_INPUT = Path("/mnt/data/cambacica-base-180m/deduplicated/exact")
DEFAULT_SPLIT = Path("/mnt/data/cambacica-base-180m/splits/c1-s1-v1")
DEFAULT_POLICY = Path("configs/c1_tokenizer_sample_policy.json")
DEFAULT_MIX = Path("configs/corpus_mix_b.yaml")
DEFAULT_ROOT = Path("/mnt/data/cambacica-base-180m/tokenizer-sample/c1-s2-v1")

PINNED_D1_MANIFEST_SHA256 = (
    "57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428"
)
PINNED_SPLIT_MANIFEST_SHA256 = (
    "8b937f90d40487c643fe48baa6b5c2950713fe2e0d2d534f3e2a3c08e74dbe16"
)
PINNED_ASSIGNMENTS_SHA256 = (
    "ef52e3cf3033934df13e3d16c207e1783c509dff6fb91954fb8c25ab719ece1e"
)
PINNED_MIX_CONFIG_SHA256 = (
    "4ff5faac9dc49466a3c19a6f8e020e340aead613a339ee7bcd62820c5a759d95"
)
PINNED_POLICY_SHA256 = (
    "eb903e329d2f260ac117f4e8730171d728dc76df621f9a533e39903e25df8797"
)

SOURCE_REVISIONS = {
    "carolina": "55e63a519393c70a48dcfa14a558499c6bb0583b",
    "wikipedia_pt": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    "parlamento_pt": "08f13e7e63ab9bfbd8c0b40955defe3bb7f68c2b",
    "gutenberg_pt": "snapshot_2026-10-01",
    "gigaverbo_v2": "7058ccf19eaeaf4505a96fc7e5305a01fc441fd8",
}
NON_GIGAVERBO_SOURCES = (
    "carolina",
    "wikipedia_pt",
    "parlamento_pt",
    "gutenberg_pt",
)

ASSIGNMENT_COLUMNS = (
    "doc_id",
    "group_id",
    "source",
    "subset",
    "domain_category",
    "source_shard",
    "source_row_ordinal",
    "content_sha256",
    "normalized_words",
    "split",
)
REFERENCE_COLUMNS = (
    "doc_id",
    "group_id",
    "source",
    "subset",
    "domain_category",
    "source_shard",
    "source_row_ordinal",
    "source_row_group",
    "content_sha256",
    "normalized_words",
    "split",
    "sample_bucket",
    "rank_hash",
    "quota_target_words",
)
REFERENCE_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("group_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("subset", pa.string(), nullable=False),
        pa.field("domain_category", pa.string(), nullable=False),
        pa.field("source_shard", pa.string(), nullable=False),
        pa.field("source_row_ordinal", pa.int64(), nullable=False),
        pa.field("source_row_group", pa.int32(), nullable=False),
        pa.field("content_sha256", pa.string(), nullable=False),
        pa.field("normalized_words", pa.int64(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("sample_bucket", pa.string(), nullable=False),
        pa.field("rank_hash", pa.string(), nullable=False),
        pa.field("quota_target_words", pa.int64(), nullable=False),
    ]
)
SAMPLE_EXTRA_FIELDS = (
    pa.field("doc_id", pa.string(), nullable=False),
    pa.field("group_id", pa.string(), nullable=False),
    pa.field("source_shard", pa.string(), nullable=False),
    pa.field("source_row_ordinal", pa.int64(), nullable=False),
    pa.field("source_row_group", pa.int32(), nullable=False),
    pa.field("normalized_words", pa.int64(), nullable=False),
    pa.field("split", pa.string(), nullable=False),
    pa.field("selection_bucket", pa.string(), nullable=False),
    pa.field("selection_rank_hash", pa.string(), nullable=False),
    pa.field("quota_target_words", pa.int64(), nullable=False),
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (canonical(value) + "\n").encode("utf-8"))


def _write_checksums(path: Path, entries: Mapping[str, str]) -> None:
    body = "".join(f"{digest}  {name}\n" for name, digest in sorted(entries.items()))
    _atomic_bytes(path, body.encode("ascii"))


def _read_checksums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or parts[1] in result:
            raise ValueError(f"Malformed checksum manifest: {path}")
        result[parts[1]] = parts[0]
    return result


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=120)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _store_identity(
    connection: sqlite3.Connection, identity: Mapping[str, Any]
) -> None:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS run_identity (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    saved = dict(connection.execute("SELECT key,value FROM run_identity"))
    current = {key: canonical(value) for key, value in identity.items()}
    if saved and saved != current:
        raise ValueError(
            "Scratch database identity mismatch; use a fresh scratch directory"
        )
    if not saved:
        with connection:
            connection.executemany(
                "INSERT INTO run_identity(key,value) VALUES (?,?)", current.items()
            )


def _policy_sha(path: Path) -> str:
    digest = compute_file_sha256(path)
    if digest != PINNED_POLICY_SHA256:
        raise ValueError(
            "Sampling policy bytes differ from the frozen implementation policy; "
            "review and version a new policy before running production"
        )
    return digest


def load_policy(
    path: Path = DEFAULT_POLICY, mix_path: Path = DEFAULT_MIX
) -> dict[str, Any]:
    policy_sha = _policy_sha(path)
    if compute_file_sha256(mix_path) != PINNED_MIX_CONFIG_SHA256:
        raise ValueError("Reference corpus_mix_b.yaml checksum mismatch")
    policy = json.loads(path.read_text())
    with mix_path.open() as stream:
        mix = yaml.safe_load(stream)
    if (
        policy.get("input_d1_manifest_sha256") != PINNED_D1_MANIFEST_SHA256
        or policy.get("input_split_manifest_sha256") != PINNED_SPLIT_MANIFEST_SHA256
        or policy.get("input_assignments_sha256") != PINNED_ASSIGNMENTS_SHA256
    ):
        raise ValueError("Sampling policy does not name the pinned D1 and C1-S1 inputs")
    if policy.get("status") != "proposed_frozen_for_scientist_review":
        raise ValueError("Unexpected tokenizer sampling policy status")
    if (
        policy.get("mixture") != "B"
        or not isinstance(policy.get("target_normalized_words"), int)
        or policy["target_normalized_words"] < 1
    ):
        raise ValueError("Unsupported target or sampling mixture")
    if policy.get("sampling_seed") != 20261014:
        raise ValueError("Unsupported sampling seed")
    source_shares = {
        "carolina": 2800,
        "wikipedia_pt": 1800,
        "parlamento_pt": 1000,
        "gutenberg_pt": 400,
        "gigaverbo_v2": 4000,
    }
    mix_source_names = {
        "carolina": "carolina",
        "wikipedia_pt": "wikipedia_pt",
        "parlamento_pt": "parlamento_pt",
        "gutenberg_pt": "gutenberg_pt",
        "gigaverbo_v2_residual": "gigaverbo_v2",
    }
    actual_shares = {
        mix_source_names[name]: int(round(float(spec["share"]) * 10000))
        for name, spec in mix["sources"].items()
    }
    if (
        policy.get("source_share_basis_points") != source_shares
        or actual_shares != source_shares
    ):
        raise ValueError("B source shares differ from the frozen policy")
    gv_spec = mix["sources"]["gigaverbo_v2_residual"]
    subset_weights = {
        name: int(round(float(weight) * 10000))
        for name, weight in gv_spec["subsets"].items()
    }
    if (
        policy.get("gigaverbo_residual_share_basis_points") != 4000
        or policy.get("gigaverbo_subset_weight_basis_points") != subset_weights
        or sum(subset_weights.values()) != 10000
    ):
        raise ValueError("GigaVerbo subset weights differ from corpus_mix_b.yaml")
    if sum(source_shares.values()) != 10000:
        raise ValueError("Source shares must sum to 100 percent")
    frame = policy.get("materialization_frame", {})
    if (
        frame.get("algorithm") != "sha256"
        or frame.get("namespace") != "cambacica-c1-tokenizer-frame-v1"
        or frame.get("word_capacity_multiplier") != 10
    ):
        raise ValueError("Unsupported deterministic materialization frame")
    policy["_policy_sha256"] = policy_sha
    policy["_mix_config_sha256"] = PINNED_MIX_CONFIG_SHA256
    return policy


def _quota_plan(policy: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    target = int(policy["target_normalized_words"])
    quotas: dict[str, dict[str, Any]] = {}
    source_bps = policy["source_share_basis_points"]
    for source in NON_GIGAVERBO_SOURCES:
        product = target * int(source_bps[source])
        if product % 10000:
            raise ValueError(f"Target does not yield an integer quota for {source}")
        quotas[source] = dict(
            bucket=source, source=source, subset=None, target=product // 10000
        )
    residual_product = target * int(policy["gigaverbo_residual_share_basis_points"])
    if residual_product % 10000:
        raise ValueError("Target does not yield an integer GigaVerbo residual quota")
    residual_target = residual_product // 10000
    for subset, weight in policy["gigaverbo_subset_weight_basis_points"].items():
        product = residual_target * int(weight)
        if product % 10000:
            raise ValueError(f"Target does not yield an integer quota for {subset}")
        bucket = f"gigaverbo_v2/{subset}"
        quotas[bucket] = dict(
            bucket=bucket,
            source="gigaverbo_v2",
            subset=subset,
            target=product // 10000,
        )
    if sum(item["target"] for item in quotas.values()) != target:
        raise ValueError("Quota plan does not sum to the total target")
    return quotas


def _bucket(source: str, subset: str, quotas: Mapping[str, Any]) -> str:
    name = f"gigaverbo_v2/{subset}" if source == "gigaverbo_v2" else source
    if name not in quotas:
        raise ValueError(f"No frozen sampling quota exists for {source}/{subset}")
    return name


def _rank_hash(seed: int, doc_id: str) -> str:
    value = canonical([int(seed), doc_id])
    return hashlib.sha256(
        ("cambacica-c1-tokenizer-sample-v1\0" + value).encode("utf-8")
    ).hexdigest()


def _frame_rank_hash(seed: int, bucket: str, source_shard: str, row_group: int) -> str:
    value = canonical([int(seed), bucket, source_shard, int(row_group)])
    return hashlib.sha256(
        ("cambacica-c1-tokenizer-frame-v1\0" + value).encode("utf-8")
    ).hexdigest()


def _read_input_identities(
    input_root: Path,
    split_root: Path,
    *,
    verify_assignment_payload: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    d1_manifest_path = input_root / "manifest.json"
    split_manifest_path = split_root / "manifest.json"
    if compute_file_sha256(d1_manifest_path) != PINNED_D1_MANIFEST_SHA256:
        raise ValueError("Pinned D1 manifest SHA-256 mismatch")
    if compute_file_sha256(split_manifest_path) != PINNED_SPLIT_MANIFEST_SHA256:
        raise ValueError("Pinned C1-S1 manifest SHA-256 mismatch")
    d1_manifest = json.loads(d1_manifest_path.read_text())
    split_manifest = json.loads(split_manifest_path.read_text())
    if (
        d1_manifest.get("status") != "COMPLETE"
        or split_manifest.get("input_manifest_sha256") != PINNED_D1_MANIFEST_SHA256
        or split_manifest.get("output_sha256") != PINNED_ASSIGNMENTS_SHA256
    ):
        raise ValueError("D1 or C1-S1 manifest identity mismatch")
    checksum_file = split_root / "checksums.sha256"
    expected_checksums = {
        "manifest.json": PINNED_SPLIT_MANIFEST_SHA256,
        "assignments.parquet": PINNED_ASSIGNMENTS_SHA256,
    }
    if _read_checksums(checksum_file) != expected_checksums:
        raise ValueError("C1-S1 checksum manifest differs from pinned identities")
    assignments_path = split_root / "assignments.parquet"
    if not assignments_path.is_file():
        raise ValueError("Pinned C1-S1 assignments file is missing")
    if verify_assignment_payload:
        observed = compute_file_sha256(assignments_path)
        if observed != PINNED_ASSIGNMENTS_SHA256:
            raise ValueError("C1-S1 assignments payload checksum mismatch")
    return (
        d1_manifest,
        split_manifest,
        {
            "d1_manifest_sha256": PINNED_D1_MANIFEST_SHA256,
            "split_manifest_sha256": PINNED_SPLIT_MANIFEST_SHA256,
            "assignments_sha256": PINNED_ASSIGNMENTS_SHA256,
        },
    )


def _check_shard_entry(
    source: str,
    source_shard: str,
    d1_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    parts = PurePosixPath(source_shard).parts
    if (
        not parts
        or parts[0] != source
        or PurePosixPath(source_shard).is_absolute()
        or ".." in parts
    ):
        raise ValueError("Assignment source shard does not match its source")
    relative = f"data/{source_shard}"
    d1_entry = d1_manifest.get("output_files", {}).get(relative)
    split_entry = split_manifest.get("input_files", {}).get(relative)
    if not d1_entry or not split_entry:
        raise ValueError(
            f"Source shard is absent from pinned D1/S1 manifests: {relative}"
        )
    for key in ("bytes", "rows", "sha256"):
        if d1_entry.get(key) != split_entry.get(key):
            raise ValueError(f"D1 and C1-S1 source shard identities differ: {relative}")
    return dict(d1_entry)


def _read_row_group_layout(
    input_root: Path,
    d1_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> dict[str, list[int]]:
    """Read Parquet footers only and map each pinned source row to its row group."""
    layout: dict[str, list[int]] = {}
    paths = sorted(
        path
        for path in split_manifest.get("input_files", {})
        if path.startswith("data/")
    )
    if not paths:
        raise ValueError("C1-S1 manifest contains no source data files")
    for relative in paths:
        source_shard = relative.removeprefix("data/")
        source = PurePosixPath(source_shard).parts[0]
        entry = _check_shard_entry(source, source_shard, d1_manifest, split_manifest)
        path = input_root / relative
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
            raise ValueError(
                f"Pinned D1 source shard is missing or changed: {relative}"
            )
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != int(entry["rows"]):
            raise ValueError(
                f"D1 footer row count differs from pinned manifest: {relative}"
            )
        starts = [0]
        for index in range(parquet.num_row_groups):
            starts.append(starts[-1] + parquet.metadata.row_group(index).num_rows)
        if starts[-1] != parquet.metadata.num_rows:
            raise ValueError(f"D1 row-group footer counts do not sum: {relative}")
        layout[source_shard] = starts
    return layout


def _source_row_group(
    layout: Mapping[str, Sequence[int]], source_shard: str, ordinal: int
) -> int:
    starts = layout.get(source_shard)
    if starts is None or ordinal < 0 or ordinal >= starts[-1]:
        raise ValueError("Assignment coordinate does not exist in the pinned D1 footer")
    row_group = bisect_right(starts, ordinal) - 1
    if (
        row_group >= len(starts) - 1
        or not starts[row_group] <= ordinal < starts[row_group + 1]
    ):
        raise ValueError("Could not map assignment coordinate to a D1 row group")
    return row_group


def _validate_assignment_row(
    row: Mapping[str, Any],
    policy: Mapping[str, Any],
    quotas: Mapping[str, Any],
    d1_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    row_group_layout: Mapping[str, Sequence[int]],
    checked_shards: set[str],
) -> tuple[str, str, int]:
    missing = set(ASSIGNMENT_COLUMNS) - set(row)
    if missing:
        raise ValueError(f"C1-S1 assignment row is missing fields: {sorted(missing)}")
    source = row["source"]
    subset = row["subset"]
    if source not in SOURCE_REVISIONS or not isinstance(subset, str):
        raise ValueError("Unknown source or invalid subset in C1-S1 assignment")
    if row["split"] not in ("train", "validation", "test"):
        raise ValueError(f"Invalid C1-S1 split value: {row['split']!r}")
    if not isinstance(row["source_row_ordinal"], int) or row["source_row_ordinal"] < 0:
        raise ValueError("Invalid source row ordinal in C1-S1 assignment")
    if not isinstance(row["normalized_words"], int) or row["normalized_words"] < 0:
        raise ValueError("Invalid normalized word count in C1-S1 assignment")
    for key in ("doc_id", "group_id", "source_shard", "content_sha256"):
        if not isinstance(row[key], str) or not row[key]:
            raise ValueError(f"Missing {key} in C1-S1 assignment")
    if len(row["content_sha256"]) != 64:
        raise ValueError("Malformed content checksum in C1-S1 assignment")
    expected_doc_id = occurrence_id_v2(
        source, row["source_shard"], row["source_row_ordinal"]
    )
    if row["doc_id"] != expected_doc_id:
        raise ValueError("Stable document ID does not match its source coordinates")
    bucket = _bucket(source, subset, quotas)
    if (
        source == "gigaverbo_v2"
        and subset not in policy["gigaverbo_subset_weight_basis_points"]
    ):
        raise ValueError(f"Unexpected GigaVerbo subset in C1-S1 assignment: {subset}")
    if row["source_shard"] not in checked_shards:
        entry = _check_shard_entry(
            source, row["source_shard"], d1_manifest, split_manifest
        )
        if row["source_row_ordinal"] >= int(entry["rows"]):
            raise ValueError("Assignment row ordinal exceeds its pinned source shard")
        checked_shards.add(row["source_shard"])
    row_group = _source_row_group(
        row_group_layout, row["source_shard"], row["source_row_ordinal"]
    )
    return source, bucket, row_group


def _initialize_candidate_db(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS candidate (
          doc_id TEXT PRIMARY KEY,
          group_id TEXT NOT NULL,
          source TEXT NOT NULL,
          subset TEXT NOT NULL,
          domain_category TEXT NOT NULL,
          source_shard TEXT NOT NULL,
          source_row_ordinal INTEGER NOT NULL,
          source_row_group INTEGER NOT NULL,
          content_sha256 TEXT NOT NULL,
          normalized_words INTEGER NOT NULL,
          sample_bucket TEXT NOT NULL,
          rank_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS completed_row_groups (
          row_group INTEGER PRIMARY KEY,
          records INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS frame_group (
          sample_bucket TEXT NOT NULL,
          source_shard TEXT NOT NULL,
          source_row_group INTEGER NOT NULL,
          rank_hash TEXT NOT NULL,
          train_documents INTEGER NOT NULL,
          train_words INTEGER NOT NULL,
          in_frame INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY(sample_bucket,source_shard,source_row_group)
        );
        CREATE TABLE IF NOT EXISTS selected (
          doc_id TEXT PRIMARY KEY,
          sample_bucket TEXT NOT NULL,
          quota_target_words INTEGER NOT NULL,
          FOREIGN KEY(doc_id) REFERENCES candidate(doc_id)
        );
        CREATE INDEX IF NOT EXISTS frame_group_rank
          ON frame_group(sample_bucket,rank_hash,source_shard,source_row_group);
        """
    )


def _iter_assignment_batches(
    assignments_path: Path,
    batch_size: int,
    *,
    reverse_row_groups: bool = False,
    skip_row_groups: set[int] | None = None,
) -> Iterator[tuple[int, pa.RecordBatch]]:
    parquet = pq.ParquetFile(assignments_path)
    absent = set(ASSIGNMENT_COLUMNS) - set(parquet.schema_arrow.names)
    if absent:
        raise ValueError(f"C1-S1 assignments schema is missing {sorted(absent)}")
    row_groups: Iterable[int] = range(parquet.num_row_groups)
    if reverse_row_groups:
        row_groups = reversed(range(parquet.num_row_groups))
    for row_group in row_groups:
        if skip_row_groups and row_group in skip_row_groups:
            continue
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            row_groups=[row_group],
            columns=list(ASSIGNMENT_COLUMNS),
        ):
            yield row_group, batch


def _populate_candidates(
    connection: sqlite3.Connection,
    assignments_path: Path,
    policy: Mapping[str, Any],
    quotas: Mapping[str, Any],
    d1_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    row_group_layout: Mapping[str, Sequence[int]],
    *,
    batch_size: int,
    reverse_row_groups: bool,
) -> None:
    _initialize_candidate_db(connection)
    parquet = pq.ParquetFile(assignments_path)
    completed = {
        int(row[0])
        for row in connection.execute("SELECT row_group FROM completed_row_groups")
    }
    checked_shards: set[str] = set()
    active_group: int | None = None
    group_count = 0
    group_stats: dict[tuple[str, str, int], list[int]] = {}
    transaction = None

    def finish_group() -> None:
        if active_group is None:
            return
        for (bucket, source_shard, source_row_group), (
            documents,
            words,
        ) in group_stats.items():
            rank = _frame_rank_hash(
                int(policy["sampling_seed"]), bucket, source_shard, source_row_group
            )
            connection.execute(
                """INSERT INTO frame_group VALUES (?,?,?,?,?,?,0)
                   ON CONFLICT(sample_bucket,source_shard,source_row_group) DO UPDATE SET
                     train_documents=frame_group.train_documents+excluded.train_documents,
                     train_words=frame_group.train_words+excluded.train_words""",
                (bucket, source_shard, source_row_group, rank, documents, words),
            )
        connection.execute(
            "INSERT INTO completed_row_groups VALUES (?,?)",
            (active_group, group_count),
        )

    try:
        for row_group, batch in _iter_assignment_batches(
            assignments_path,
            batch_size,
            reverse_row_groups=reverse_row_groups,
            skip_row_groups=completed,
        ):
            if active_group != row_group:
                if transaction is not None:
                    finish_group()
                    connection.commit()
                connection.execute("BEGIN")
                transaction = True
                active_group = row_group
                group_count = 0
                group_stats = {}
            for row in batch.to_pylist():
                group_count += 1
                source, bucket, source_row_group = _validate_assignment_row(
                    row,
                    policy,
                    quotas,
                    d1_manifest,
                    split_manifest,
                    row_group_layout,
                    checked_shards,
                )
                if row["split"] != "train":
                    continue
                rank = _rank_hash(int(policy["sampling_seed"]), row["doc_id"])
                connection.execute(
                    """INSERT INTO candidate VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        row["doc_id"],
                        row["group_id"],
                        source,
                        row["subset"],
                        row["domain_category"],
                        row["source_shard"],
                        row["source_row_ordinal"],
                        source_row_group,
                        row["content_sha256"],
                        row["normalized_words"],
                        bucket,
                        rank,
                    ),
                )
                stats = group_stats.setdefault(
                    (bucket, row["source_shard"], source_row_group), [0, 0]
                )
                stats[0] += 1
                stats[1] += int(row["normalized_words"])
        if transaction is not None:
            finish_group()
            connection.commit()
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise
    if (
        connection.execute("SELECT count(*) FROM completed_row_groups").fetchone()[0]
        != parquet.num_row_groups
    ):
        raise ValueError("Selection checkpoint is incomplete")
    for row_group, records in connection.execute(
        "SELECT row_group,records FROM completed_row_groups"
    ):
        expected_records = parquet.metadata.row_group(int(row_group)).num_rows
        if int(records) != expected_records:
            raise ValueError("Selection checkpoint row count differs from assignments")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS candidate_rank ON candidate(sample_bucket,rank_hash,doc_id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS frame_group_rank"
        " ON frame_group(sample_bucket,rank_hash,source_shard,source_row_group)"
    )
    connection.commit()


def _choose_candidates(
    connection: sqlite3.Connection,
    quotas: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    connection.execute("DELETE FROM selected")
    connection.execute("UPDATE frame_group SET in_frame=0")
    summaries: dict[str, dict[str, int]] = {}
    multiplier = int(policy["materialization_frame"]["word_capacity_multiplier"])
    for bucket, plan in sorted(quotas.items()):
        target = int(plan["target"])
        selected_words = 0
        selected_documents = 0
        eligible_documents, eligible_words = connection.execute(
            "SELECT count(*),coalesce(sum(normalized_words),0) FROM candidate WHERE sample_bucket=?",
            (bucket,),
        ).fetchone()
        frame_documents = 0
        frame_words = 0
        frame_groups = 0
        ranked_groups = connection.execute(
            """SELECT source_shard,source_row_group,train_documents,train_words
               FROM frame_group WHERE sample_bucket=?
               ORDER BY rank_hash,source_shard,source_row_group""",
            (bucket,),
        ).fetchall()
        for source_shard, source_row_group, documents, words in ranked_groups:
            connection.execute(
                """UPDATE frame_group SET in_frame=1 WHERE sample_bucket=?
                   AND source_shard=? AND source_row_group=?""",
                (bucket, source_shard, source_row_group),
            )
            frame_documents += int(documents)
            frame_words += int(words)
            frame_groups += 1
            if frame_words >= target * multiplier:
                break
        cursor = connection.execute(
            """SELECT c.doc_id,c.normalized_words FROM candidate c
               JOIN frame_group g ON g.sample_bucket=c.sample_bucket
                 AND g.source_shard=c.source_shard
                 AND g.source_row_group=c.source_row_group
               WHERE c.sample_bucket=? AND g.in_frame=1
               ORDER BY c.rank_hash,c.doc_id""",
            (bucket,),
        )
        for doc_id, normalized_words in cursor:
            word_count = int(normalized_words)
            if word_count <= 0 or word_count > target - selected_words:
                continue
            connection.execute(
                "INSERT INTO selected VALUES (?,?,?)",
                (doc_id, bucket, target),
            )
            selected_words += word_count
            selected_documents += 1
        summaries[bucket] = {
            "eligible_train_documents": int(eligible_documents),
            "eligible_train_words": int(eligible_words),
            "selected_documents": selected_documents,
            "selected_words": selected_words,
            "target_words": target,
            "materialization_frame_documents": frame_documents,
            "materialization_frame_words": frame_words,
            "materialization_frame_row_groups": frame_groups,
        }
    connection.commit()
    return summaries


def _quota_report(
    summaries: Mapping[str, Mapping[str, int]], policy: Mapping[str, Any]
) -> dict[str, Any]:
    limits = policy["quota_tolerance"]
    categories: list[dict[str, Any]] = []
    source_actual: dict[str, int] = {}
    source_targets: dict[str, int] = {}
    passed = True
    for bucket, counts in sorted(summaries.items()):
        source = (
            bucket.split("/", 1)[0] if bucket.startswith("gigaverbo_v2/") else bucket
        )
        target = int(counts["target_words"])
        actual = int(counts["selected_words"])
        source_actual[source] = source_actual.get(source, 0) + actual
        source_targets[source] = source_targets.get(source, 0) + target
        if bucket == "gutenberg_pt":
            tolerance = int(
                limits["gutenberg_pt_maximum_source_shortfall_basis_points"]
            )
            level = "source"
        elif bucket.startswith("gigaverbo_v2/"):
            tolerance = int(limits["maximum_gigaverbo_subset_shortfall_basis_points"])
            level = "gigaverbo_subset"
        else:
            tolerance = int(limits["maximum_source_shortfall_basis_points"])
            level = "source"
        shortfall = max(0, target - actual)
        allowed = math.ceil(target * tolerance / 10000)
        item_pass = actual <= target and shortfall <= allowed
        passed = passed and item_pass
        categories.append(
            dict(
                level=level,
                bucket=bucket,
                source=source,
                subset=bucket.split("/", 1)[1]
                if bucket.startswith("gigaverbo_v2/")
                else None,
                target_words=target,
                actual_words=actual,
                deviation_words=actual - target,
                shortfall_basis_points=(
                    math.ceil(shortfall * 10000 / target) if target else 0
                ),
                maximum_shortfall_basis_points=tolerance,
                within_tolerance=item_pass,
                selected_documents=int(counts["selected_documents"]),
                eligible_train_documents=int(counts["eligible_train_documents"]),
                eligible_train_words=int(counts["eligible_train_words"]),
            )
        )
    source_tolerance = int(limits["maximum_source_shortfall_basis_points"])
    for source in sorted(source_targets):
        target, actual = source_targets[source], source_actual.get(source, 0)
        tol = (
            int(limits["gutenberg_pt_maximum_source_shortfall_basis_points"])
            if source == "gutenberg_pt"
            else source_tolerance
        )
        shortfall = max(0, target - actual)
        source_pass = actual <= target and shortfall <= math.ceil(target * tol / 10000)
        passed = passed and source_pass
        categories.append(
            dict(
                level="source_total",
                bucket=source,
                source=source,
                subset=None,
                target_words=target,
                actual_words=actual,
                deviation_words=actual - target,
                shortfall_basis_points=(
                    math.ceil(shortfall * 10000 / target) if target else 0
                ),
                maximum_shortfall_basis_points=tol,
                within_tolerance=source_pass,
            )
        )
    total_target = int(policy["target_normalized_words"])
    total_actual = sum(int(counts["selected_words"]) for counts in summaries.values())
    total_shortfall = max(0, total_target - total_actual)
    total_tolerance = int(limits["maximum_total_shortfall_basis_points"])
    total_pass = total_actual <= total_target and total_shortfall <= math.ceil(
        total_target * total_tolerance / 10000
    )
    passed = passed and total_pass
    categories.append(
        dict(
            level="total",
            bucket="TOTAL",
            source=None,
            subset=None,
            target_words=total_target,
            actual_words=total_actual,
            deviation_words=total_actual - total_target,
            shortfall_basis_points=(
                math.ceil(total_shortfall * 10000 / total_target) if total_target else 0
            ),
            maximum_shortfall_basis_points=total_tolerance,
            maximum_overshoot_words=int(limits["maximum_overshoot_words"]),
            within_tolerance=total_pass,
        )
    )
    return dict(
        status="PASS" if passed else "QUOTA_TOLERANCE_FAILED", categories=categories
    )


def _selected_reference_rows(
    connection: sqlite3.Connection,
) -> Iterator[dict[str, Any]]:
    cursor = connection.execute(
        """SELECT c.doc_id,c.group_id,c.source,c.subset,c.domain_category,
                  c.source_shard,c.source_row_ordinal,c.source_row_group,c.content_sha256,
                  c.normalized_words,'train',s.sample_bucket,c.rank_hash,
                  s.quota_target_words
           FROM selected s JOIN candidate c USING(doc_id)
           ORDER BY c.doc_id"""
    )
    for values in cursor:
        yield dict(zip(REFERENCE_COLUMNS, values))


def _write_reference_file(connection: sqlite3.Connection, path: Path) -> dict[str, Any]:
    temporary = path.with_name(path.name + ".partial")
    count = 0
    with pq.ParquetWriter(
        temporary, REFERENCE_SCHEMA, compression="zstd", version="2.6"
    ) as writer:
        buffer: list[dict[str, Any]] = []
        for row in _selected_reference_rows(connection):
            buffer.append(row)
            if len(buffer) >= 4096:
                writer.write_table(
                    pa.Table.from_pylist(buffer, schema=REFERENCE_SCHEMA),
                    row_group_size=4096,
                )
                count += len(buffer)
                buffer.clear()
        if buffer:
            writer.write_table(
                pa.Table.from_pylist(buffer, schema=REFERENCE_SCHEMA),
                row_group_size=4096,
            )
            count += len(buffer)
    os.replace(temporary, path)
    return dict(rows=count, bytes=path.stat().st_size, sha256=compute_file_sha256(path))


def _stage_for(output: Path, identity: Mapping[str, Any]) -> Path:
    output = output.resolve()
    stage = output.with_name(output.name + ".partial")
    if output.exists():
        raise FileExistsError(f"Published output already exists: {output}")
    stage.mkdir(parents=True, exist_ok=True)
    identity_path = stage / "state.json"
    current = canonical(identity) + "\n"
    if identity_path.exists() and identity_path.read_text() != current:
        raise ValueError(f"Staging directory identity mismatch: {stage}")
    if not identity_path.exists():
        _atomic_bytes(identity_path, current.encode("utf-8"))
    return stage


def _publish_stage(stage: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to replace published output: {output}")
    os.replace(stage, output)
    descriptor = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _paths_overlap(first: Path, second: Path) -> bool:
    left, right = first.resolve(), second.resolve()
    return left == right or left in right.parents or right in left.parents


def _mount_type(path: Path) -> str:
    target = path.resolve()
    try:
        lines = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError as error:
        raise ValueError(
            "Cannot validate local scratch and output filesystems"
        ) from error
    matches: list[tuple[int, str]] = []
    for line in lines:
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        fields = before.split()
        tail = after.split()
        if len(fields) < 5 or not tail:
            continue
        mount_text = re.sub(
            r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), fields[4]
        )
        mount_point = Path(mount_text)
        if target == mount_point or mount_point in target.parents:
            matches.append((len(mount_text), tail[0]))
    if not matches:
        raise ValueError(f"Cannot determine filesystem type for {path}")
    return max(matches)[1]


def _require_separate_paths(
    output: Path,
    scratch: Path,
    protected: Sequence[Path],
) -> None:
    for path in protected:
        if _paths_overlap(output, path):
            raise ValueError(
                f"Published output must be separate from protected input: {path}"
            )
        if _paths_overlap(scratch, path):
            raise ValueError(
                f"SQLite scratch must be separate from protected path: {path}"
            )
    if _paths_overlap(output, scratch):
        raise ValueError("SQLite scratch and published output must be separate")
    input_filesystem = _mount_type(protected[0])
    output_filesystem = _mount_type(output)
    scratch_filesystem = _mount_type(scratch)
    if output_filesystem != input_filesystem:
        raise ValueError("Published output must use the pinned input filesystem")
    if scratch_filesystem in {"nfs", "nfs4"}:
        raise ValueError("SQLite scratch must use local storage, not NFS")


def _existing_selection(
    output: Path, identity: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not output.exists():
        return None
    manifest_path = output / "selection_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Existing selection output is incomplete")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("run_identity") != identity:
        raise ValueError("Existing selection output has a different input identity")
    checksums = _read_checksums(output / "checksums.sha256")
    if set(checksums) != {"selection_manifest.json", "selected_references.parquet"}:
        raise ValueError("Existing selection checksum manifest has unexpected entries")
    if checksums.get("selection_manifest.json") != compute_file_sha256(manifest_path):
        raise ValueError("Existing selection manifest checksum mismatch")
    artifact = output / "selected_references.parquet"
    if checksums.get(artifact.name) != compute_file_sha256(artifact):
        raise ValueError("Existing selected references checksum mismatch")
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    if actual_files != {
        "selection_manifest.json",
        "selected_references.parquet",
        "checksums.sha256",
    }:
        raise ValueError("Existing selection output contains unexpected files")
    return manifest


def select(
    input_root: Path = DEFAULT_INPUT,
    split_root: Path = DEFAULT_SPLIT,
    output_dir: Path = DEFAULT_ROOT / "selection",
    scratch_dir: Path = Path("/tmp/c1-tokenizer-sample-v1"),
    policy_path: Path = DEFAULT_POLICY,
    *,
    batch_size: int = 4096,
    reverse_row_groups: bool = False,
) -> dict[str, Any]:
    """Build deterministic references from train assignments using local SQLite."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    _require_separate_paths(output_dir, scratch_dir, (input_root, split_root))
    policy = load_policy(policy_path)
    quotas = _quota_plan(policy)
    d1_manifest, split_manifest, input_identity = _read_input_identities(
        input_root, split_root, verify_assignment_payload=True
    )
    row_group_layout = _read_row_group_layout(input_root, d1_manifest, split_manifest)
    identity = {
        **input_identity,
        "policy_sha256": policy["_policy_sha256"],
        "mix_config_sha256": policy["_mix_config_sha256"],
        "sampling_seed": int(policy["sampling_seed"]),
        "target_normalized_words": int(policy["target_normalized_words"]),
        "implementation_sha256": compute_file_sha256(Path(__file__)),
        "operation": "select-v1",
    }
    existing = _existing_selection(output_dir, identity)
    if existing is not None:
        return existing
    assignments_path = split_root / "assignments.parquet"
    before = (assignments_path.stat().st_size, assignments_path.stat().st_mtime_ns)
    parquet = pq.ParquetFile(assignments_path)
    if parquet.metadata.num_rows != int(split_manifest["documents"]):
        raise ValueError("C1-S1 assignments row count differs from its manifest")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    database_path = scratch_dir / "selection.sqlite3"
    connection = _connect(database_path)
    try:
        _store_identity(
            connection,
            {**identity, "reverse_row_groups": bool(reverse_row_groups)},
        )
        _populate_candidates(
            connection,
            assignments_path,
            policy,
            quotas,
            d1_manifest,
            split_manifest,
            row_group_layout,
            batch_size=batch_size,
            reverse_row_groups=reverse_row_groups,
        )
        candidate_count = int(
            connection.execute("SELECT count(*) FROM candidate").fetchone()[0]
        )
        summaries = _choose_candidates(connection, quotas, policy)
        quota_report = _quota_report(summaries, policy)
        if (
            assignments_path.stat().st_size,
            assignments_path.stat().st_mtime_ns,
        ) != before:
            raise ValueError("C1-S1 assignments changed during selection")
        stage_identity = {**identity, "operation": "select-v1"}
        stage = _stage_for(output_dir, stage_identity)
        selected_file = _write_reference_file(
            connection, stage / "selected_references.parquet"
        )
        manifest = dict(
            schema_version=1,
            status=quota_report["status"],
            created_by="cambacica.corpus.tokenizer_sample",
            run_identity=identity,
            policy=policy,
            input_documents=int(split_manifest["documents"]),
            eligible_train_documents=candidate_count,
            selected_documents=int(
                connection.execute("SELECT count(*) FROM selected").fetchone()[0]
            ),
            selected_normalized_words=sum(
                item["selected_words"] for item in summaries.values()
            ),
            quotas=summaries,
            quota_report=quota_report,
            artifacts={"selected_references.parquet": selected_file},
            assignment_order="reverse_row_groups"
            if reverse_row_groups
            else "forward_row_groups",
            assignment_batch_size=int(batch_size),
        )
        _atomic_json(stage / "selection_manifest.json", manifest)
        _write_checksums(
            stage / "checksums.sha256",
            {
                "selection_manifest.json": compute_file_sha256(
                    stage / "selection_manifest.json"
                ),
                "selected_references.parquet": selected_file["sha256"],
            },
        )
        (stage / "state.json").unlink(missing_ok=True)
        _publish_stage(stage, output_dir.resolve())
        if quota_report["status"] == "PASS":
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.close()
            connection = None
            database_path.unlink(missing_ok=True)
            Path(str(database_path) + "-wal").unlink(missing_ok=True)
            Path(str(database_path) + "-shm").unlink(missing_ok=True)
        return manifest
    finally:
        if connection is not None:
            connection.close()


def _load_selection(
    selection_dir: Path,
    policy: Mapping[str, Any],
    input_identity: Mapping[str, str],
) -> tuple[dict[str, Any], str, str]:
    manifest_path = selection_dir / "selection_manifest.json"
    refs_path = selection_dir / "selected_references.parquet"
    manifest = json.loads(manifest_path.read_text())
    checksums = _read_checksums(selection_dir / "checksums.sha256")
    manifest_sha = compute_file_sha256(manifest_path)
    refs_sha = compute_file_sha256(refs_path)
    if (
        checksums != {"selection_manifest.json": manifest_sha, refs_path.name: refs_sha}
        or manifest.get("status") != "PASS"
        or manifest.get("run_identity", {}).get("d1_manifest_sha256")
        != input_identity["d1_manifest_sha256"]
        or manifest.get("run_identity", {}).get("split_manifest_sha256")
        != input_identity["split_manifest_sha256"]
        or manifest.get("run_identity", {}).get("assignments_sha256")
        != input_identity["assignments_sha256"]
        or manifest.get("run_identity", {}).get("policy_sha256")
        != policy["_policy_sha256"]
    ):
        raise ValueError("Selection artifact identity, status, or checksum mismatch")
    expected = manifest.get("artifacts", {}).get(refs_path.name, {})
    if (
        expected.get("sha256") != refs_sha
        or expected.get("bytes") != refs_path.stat().st_size
    ):
        raise ValueError("Selected-reference artifact does not match its manifest")
    return manifest, manifest_sha, refs_sha


def _initialize_materialize_db(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS selected_ref (
          doc_id TEXT PRIMARY KEY,
          group_id TEXT NOT NULL,
          source TEXT NOT NULL,
          subset TEXT NOT NULL,
          domain_category TEXT NOT NULL,
          source_shard TEXT NOT NULL,
          source_row_ordinal INTEGER NOT NULL,
          source_row_group INTEGER NOT NULL,
          content_sha256 TEXT NOT NULL,
          normalized_words INTEGER NOT NULL,
          split TEXT NOT NULL,
          sample_bucket TEXT NOT NULL,
          rank_hash TEXT NOT NULL,
          quota_target_words INTEGER NOT NULL,
          UNIQUE(source_shard,source_row_ordinal)
        );
        CREATE INDEX IF NOT EXISTS selected_ref_coordinate
          ON selected_ref(source_shard,source_row_ordinal);
        CREATE TABLE IF NOT EXISTS completed_shards (
          source_shard TEXT PRIMARY KEY,
          artifact_path TEXT NOT NULL,
          rows INTEGER NOT NULL,
          bytes INTEGER NOT NULL,
          sha256 TEXT NOT NULL
        );
        """
    )


def _load_selected_refs(
    connection: sqlite3.Connection, path: Path, policy: Mapping[str, Any]
) -> int:
    parquet = pq.ParquetFile(path)
    if parquet.schema_arrow.names != list(REFERENCE_COLUMNS):
        raise ValueError("Selected-reference schema mismatch")
    _initialize_materialize_db(connection)
    quotas = _quota_plan(policy)
    with connection:
        for batch in parquet.iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                if row["split"] != "train":
                    raise ValueError("Selection contains a validation or test record")
                if row["normalized_words"] <= 0:
                    raise ValueError("Selection contains a zero-word record")
                shard_parts = PurePosixPath(row["source_shard"]).parts
                if (
                    PurePosixPath(row["source_shard"]).is_absolute()
                    or ".." in shard_parts
                    or not shard_parts
                    or shard_parts[0] != row["source"]
                    or int(row["source_row_group"]) < 0
                ):
                    raise ValueError(
                        "Selected reference contains unsafe source coordinates"
                    )
                bucket = _bucket(row["source"], row["subset"], quotas)
                if (
                    row["sample_bucket"] != bucket
                    or row["rank_hash"]
                    != _rank_hash(int(policy["sampling_seed"]), row["doc_id"])
                    or int(row["quota_target_words"]) != int(quotas[bucket]["target"])
                    or row["doc_id"]
                    != occurrence_id_v2(
                        row["source"], row["source_shard"], row["source_row_ordinal"]
                    )
                ):
                    raise ValueError("Selected reference ranking or identity mismatch")
                connection.execute(
                    "INSERT INTO selected_ref VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    tuple(row[name] for name in REFERENCE_COLUMNS),
                )
    return int(connection.execute("SELECT count(*) FROM selected_ref").fetchone()[0])


def _sample_schema(source_schema: pa.Schema) -> pa.Schema:
    existing = set(source_schema.names)
    duplicate = existing.intersection(field.name for field in SAMPLE_EXTRA_FIELDS)
    if duplicate:
        raise ValueError(
            f"D1 input already contains reserved sample fields: {sorted(duplicate)}"
        )
    return pa.schema(
        list(source_schema) + list(SAMPLE_EXTRA_FIELDS), metadata=source_schema.metadata
    )


def _check_source_row(
    row: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    text = row.get("text")
    if not isinstance(text, str):
        raise ValueError("Selected D1 document text is not a string")
    text_sha = _sha256_text(text)
    if (
        text_sha != reference["content_sha256"]
        or row.get("content_sha256") != text_sha
        or len(text.split()) != int(reference["normalized_words"])
        or row.get("source") != reference["source"]
        or row.get("subset") != reference["subset"]
        or row.get("domain_category") != reference["domain_category"]
    ):
        raise ValueError(
            "Selected D1 document identity or normalized-text hash mismatch"
        )
    expected_revision = SOURCE_REVISIONS[reference["source"]]
    if row.get("source_revision") != expected_revision:
        raise ValueError("Selected D1 source revision differs from the pinned source")
    for key in ("original_id", "raw_source_file", "raw_record_identifier"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"Selected D1 provenance field is missing: {key}")
    expected_doc_id = occurrence_id_v2(
        reference["source"], reference["source_shard"], reference["source_row_ordinal"]
    )
    if expected_doc_id != reference["doc_id"]:
        raise ValueError("Selected stable document identity mismatch")
    group_key = (
        row["content_sha256"]
        if reference["source"] == "parlamento_pt"
        else row["original_id"]
    )
    expected_group = hashlib.sha256(
        (
            "cambacica-s1-group-v1\0"
            + canonical([reference["source"], reference["subset"], group_key])
        ).encode("utf-8")
    ).hexdigest()
    if expected_group != reference["group_id"]:
        raise ValueError("Selected original document group identity mismatch")
    return dict(
        doc_id=reference["doc_id"],
        group_id=reference["group_id"],
        source_shard=reference["source_shard"],
        source_row_ordinal=int(reference["source_row_ordinal"]),
        source_row_group=int(reference["source_row_group"]),
        normalized_words=int(reference["normalized_words"]),
        split=reference["split"],
        selection_bucket=reference["sample_bucket"],
        selection_rank_hash=reference["rank_hash"],
        quota_target_words=int(reference["quota_target_words"]),
    )


def _refs_for_batch(
    connection: sqlite3.Connection,
    source_shard: str,
    source_row_group: int,
    first: int,
    last: int,
) -> dict[int, dict[str, Any]]:
    cursor = connection.execute(
        """SELECT doc_id,group_id,source,subset,domain_category,source_shard,
                  source_row_ordinal,source_row_group,content_sha256,normalized_words,split,
                  sample_bucket,rank_hash,quota_target_words
           FROM selected_ref WHERE source_shard=? AND source_row_group=?
             AND source_row_ordinal>=? AND source_row_ordinal<?
           ORDER BY source_row_ordinal""",
        (source_shard, source_row_group, first, last),
    )
    return {
        int(values[6]): dict(zip(REFERENCE_COLUMNS, values))
        for values in cursor.fetchall()
    }


def _source_rows_for_shard(
    connection: sqlite3.Connection,
    input_root: Path,
    source_shard: str,
    *,
    batch_size: int,
) -> tuple[pa.Schema, Iterator[dict[str, Any]], dict[str, Any]]:
    shard_path = PurePosixPath(source_shard)
    if shard_path.is_absolute() or ".." in shard_path.parts:
        raise ValueError("Unsafe source shard path in selected references")
    path = input_root / "data" / source_shard
    if not path.is_file():
        raise ValueError(f"Selected source shard is missing: {path}")
    before_stat = path.stat()
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    starts: list[int] = []
    cursor = 0
    for index in range(parquet.num_row_groups):
        starts.append(cursor)
        cursor += parquet.metadata.row_group(index).num_rows
    selected_count = int(
        connection.execute(
            "SELECT count(*) FROM selected_ref WHERE source_shard=?", (source_shard,)
        ).fetchone()[0]
    )
    if not selected_count:
        raise ValueError("Materialization was asked to retrieve an empty source shard")

    def records() -> Iterator[dict[str, Any]]:
        found = 0
        row_groups = [
            int(row[0])
            for row in connection.execute(
                "SELECT DISTINCT source_row_group FROM selected_ref WHERE source_shard=? ORDER BY source_row_group",
                (source_shard,),
            )
        ]
        for row_group in row_groups:
            if row_group < 0 or row_group >= parquet.num_row_groups:
                raise ValueError(
                    "Selected source row group is outside the D1 shard footer"
                )
            start = starts[row_group]
            offset = 0
            for batch in parquet.iter_batches(
                batch_size=batch_size, row_groups=[row_group]
            ):
                first = start + offset
                refs = _refs_for_batch(
                    connection,
                    source_shard,
                    row_group,
                    first,
                    first + batch.num_rows,
                )
                if refs:
                    for relative_index, row in enumerate(batch.to_pylist()):
                        ordinal = first + relative_index
                        reference = refs.get(ordinal)
                        if reference is None:
                            continue
                        extras = _check_source_row(row, reference)
                        yield {**row, **extras}
                        found += 1
                offset += batch.num_rows
        if found != selected_count:
            raise ValueError(
                f"Retrieved {found} of {selected_count} selected documents from {source_shard}"
            )
        after_stat = path.stat()
        if (before_stat.st_size, before_stat.st_mtime_ns) != (
            after_stat.st_size,
            after_stat.st_mtime_ns,
        ):
            raise ValueError(
                f"Selected source shard changed during retrieval: {source_shard}"
            )

    return schema, records(), dict(rows=selected_count, before_stat=before_stat)


def _selected_shards(connection: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT source_shard FROM selected_ref ORDER BY source_shard"
        )
    ]


def _write_materialized_shard(
    connection: sqlite3.Connection,
    input_root: Path,
    source_shard: str,
    output_path: Path,
    *,
    batch_size: int,
    expected_schema: pa.Schema | None,
) -> tuple[pa.Schema, dict[str, Any]]:
    source_schema, rows, details = _source_rows_for_shard(
        connection, input_root, source_shard, batch_size=batch_size
    )
    if expected_schema is not None and not source_schema.equals(
        expected_schema, check_metadata=True
    ):
        raise ValueError("D1 source shard schemas are inconsistent")
    schema = _sample_schema(source_schema)
    temporary = output_path.with_name(output_path.name + ".partial")
    writer = pq.ParquetWriter(temporary, schema, compression="zstd", version="2.6")
    row_count = 0
    buffer: list[dict[str, Any]] = []
    try:
        for row in rows:
            buffer.append(row)
            if len(buffer) >= 1024:
                writer.write_table(
                    pa.Table.from_pylist(buffer, schema=schema), row_group_size=1024
                )
                row_count += len(buffer)
                buffer.clear()
        if buffer:
            writer.write_table(
                pa.Table.from_pylist(buffer, schema=schema), row_group_size=1024
            )
            row_count += len(buffer)
    finally:
        writer.close()
    if row_count != details["rows"]:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Materialized row count mismatch for {source_shard}")
    os.replace(temporary, output_path)
    result = dict(
        source_shard=source_shard,
        rows=row_count,
        bytes=output_path.stat().st_size,
        sha256=compute_file_sha256(output_path),
    )
    return source_schema, result


def _sample_stage_identity(
    selection_manifest_sha: str,
    refs_sha: str,
    input_identity: Mapping[str, str],
    policy_sha: str,
) -> dict[str, Any]:
    return {
        **input_identity,
        "selection_manifest_sha256": selection_manifest_sha,
        "selected_references_sha256": refs_sha,
        "policy_sha256": policy_sha,
        "operation": "materialize-v1",
    }


def _validate_published_sample(output_dir: Path, selection_sha: str) -> dict[str, Any]:
    manifest_path = output_dir / "sample_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checksums = _read_checksums(output_dir / "checksums.sha256")
    if (
        manifest.get("status") != "MATERIALIZED"
        or manifest.get("selection_manifest_sha256") != selection_sha
    ):
        raise ValueError("Published sample was built from a different selection")
    if checksums.get("sample_manifest.json") != compute_file_sha256(manifest_path):
        raise ValueError("Published sample manifest checksum mismatch")
    expected_names = {"sample_manifest.json"}
    for name, record in manifest.get("artifacts", {}).items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe published sample artifact path")
        expected_names.add(name)
        path = output_dir / name
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or compute_file_sha256(path) != record.get("sha256")
            or checksums.get(name) != record.get("sha256")
        ):
            raise ValueError(f"Published sample artifact integrity failure: {name}")
    if set(checksums) != expected_names:
        raise ValueError("Published sample checksum manifest has unexpected entries")
    actual_files = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_names | {"checksums.sha256"}:
        raise ValueError("Published sample output contains unexpected files")
    return manifest


def materialize(
    input_root: Path = DEFAULT_INPUT,
    split_root: Path = DEFAULT_SPLIT,
    selection_dir: Path = DEFAULT_ROOT / "selection",
    output_dir: Path = DEFAULT_ROOT / "sample",
    scratch_dir: Path = Path("/tmp/c1-tokenizer-sample-v1"),
    policy_path: Path = DEFAULT_POLICY,
    *,
    batch_size: int = 1024,
) -> dict[str, Any]:
    """Retrieve selected source rows only and publish an atomic Parquet sample."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    _require_separate_paths(
        output_dir, scratch_dir, (input_root, split_root, selection_dir)
    )
    policy = load_policy(policy_path)
    d1_manifest, split_manifest, input_identity = _read_input_identities(
        input_root, split_root, verify_assignment_payload=False
    )
    selection_manifest, selection_sha, refs_sha = _load_selection(
        selection_dir, policy, input_identity
    )
    if Path(output_dir).exists():
        return _validate_published_sample(Path(output_dir), selection_sha)
    identity = _sample_stage_identity(
        selection_sha, refs_sha, input_identity, policy["_policy_sha256"]
    )
    scratch_dir.mkdir(parents=True, exist_ok=True)
    connection = _connect(scratch_dir / "materialize.sqlite3")
    stage: Path | None = None
    try:
        _store_identity(connection, identity)
        _initialize_materialize_db(connection)
        if not connection.execute("SELECT 1 FROM selected_ref LIMIT 1").fetchone():
            _load_selected_refs(
                connection,
                selection_dir / "selected_references.parquet",
                policy,
            )
        if int(
            connection.execute("SELECT count(*) FROM selected_ref").fetchone()[0]
        ) != int(selection_manifest["selected_documents"]):
            raise ValueError("Local selected-reference state count mismatch")
        stage = _stage_for(output_dir, {**identity, "operation": "materialize-v1"})
        records_dir = stage / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        shards = _selected_shards(connection)
        expected_schema: pa.Schema | None = None
        artifact_records: dict[str, dict[str, Any]] = {}
        for source_shard in shards:
            relative_data_path = f"data/{source_shard}"
            expected_entry = d1_manifest.get("output_files", {}).get(relative_data_path)
            split_entry = split_manifest.get("input_files", {}).get(relative_data_path)
            if not expected_entry or not split_entry:
                raise ValueError(f"Selected source shard is not pinned: {source_shard}")
            if any(
                expected_entry.get(key) != split_entry.get(key)
                for key in ("bytes", "rows", "sha256")
            ):
                raise ValueError(
                    f"Selected source shard identity mismatch: {source_shard}"
                )
            source_path = input_root / "data" / source_shard
            if source_path.stat().st_size != int(expected_entry["bytes"]):
                raise ValueError(
                    f"Selected source shard byte count differs from pinned D1: {source_shard}"
                )
            target_name = f"records/part-{_sha256_text(source_shard)[:16]}.parquet"
            already = connection.execute(
                "SELECT artifact_path,rows,bytes,sha256 FROM completed_shards WHERE source_shard=?",
                (source_shard,),
            ).fetchone()
            if already:
                artifact_path = stage / already[0]
                if (
                    artifact_path.is_file()
                    and artifact_path.stat().st_size == already[2]
                    and compute_file_sha256(artifact_path) == already[3]
                ):
                    artifact_records[already[0]] = dict(
                        source_shard=source_shard,
                        rows=int(already[1]),
                        bytes=int(already[2]),
                        sha256=already[3],
                    )
                    if expected_schema is None:
                        completed_sample = pq.ParquetFile(artifact_path)
                        extra_names = {field.name for field in SAMPLE_EXTRA_FIELDS}
                        expected_schema = pa.schema(
                            [
                                field
                                for field in completed_sample.schema_arrow
                                if field.name not in extra_names
                            ],
                            metadata=completed_sample.schema_arrow.metadata,
                        )
                    continue
                with connection:
                    connection.execute(
                        "DELETE FROM completed_shards WHERE source_shard=?",
                        (source_shard,),
                    )
                artifact_path.unlink(missing_ok=True)
            result_schema, artifact = _write_materialized_shard(
                connection,
                input_root,
                source_shard,
                stage / target_name,
                batch_size=batch_size,
                expected_schema=expected_schema,
            )
            if expected_schema is None:
                expected_schema = result_schema
            artifact_records[target_name] = artifact
            with connection:
                connection.execute(
                    "INSERT OR REPLACE INTO completed_shards VALUES (?,?,?,?,?)",
                    (
                        source_shard,
                        target_name,
                        artifact["rows"],
                        artifact["bytes"],
                        artifact["sha256"],
                    ),
                )
        if sum(record["rows"] for record in artifact_records.values()) != int(
            selection_manifest["selected_documents"]
        ):
            raise ValueError("Published sample row count differs from selection")
        by_source_subset: dict[str, dict[str, int]] = {}
        total_words = 0
        total_rows = 0
        for source, subset, rows, words in connection.execute(
            """SELECT source,subset,count(*),sum(normalized_words)
               FROM selected_ref GROUP BY source,subset ORDER BY source,subset"""
        ):
            key = f"{source}/{subset}" if subset else source
            by_source_subset[key] = {
                "documents": int(rows),
                "normalized_words": int(words),
            }
            total_rows += int(rows)
            total_words += int(words)
        manifest = dict(
            schema_version=1,
            status="MATERIALIZED",
            created_by="cambacica.corpus.tokenizer_sample",
            selection_manifest_sha256=selection_sha,
            selected_references_sha256=refs_sha,
            input_identity=input_identity,
            policy_sha256=policy["_policy_sha256"],
            documents=total_rows,
            normalized_words=total_words,
            by_source_subset=by_source_subset,
            quota_report=selection_manifest["quota_report"],
            artifacts=artifact_records,
        )
        _atomic_json(stage / "sample_manifest.json", manifest)
        _write_checksums(
            stage / "checksums.sha256",
            {
                "sample_manifest.json": compute_file_sha256(
                    stage / "sample_manifest.json"
                ),
                **{name: record["sha256"] for name, record in artifact_records.items()},
            },
        )
        (stage / "state.json").unlink(missing_ok=True)
        _publish_stage(stage, output_dir.resolve())
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return manifest
    finally:
        connection.close()


def _load_refs_into_verification_db(
    connection: sqlite3.Connection, refs_path: Path
) -> int:
    _initialize_materialize_db(connection)
    # Independent selection reconstruction validates the reference stream before
    # it is admitted to the record comparison database.
    return int(pq.ParquetFile(refs_path).metadata.num_rows)


def _verify_reconstructed_selection(
    input_root: Path,
    split_root: Path,
    selection_dir: Path,
    scratch_dir: Path,
    policy: Mapping[str, Any],
    d1_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    input_identity: Mapping[str, str],
    *,
    batch_size: int,
) -> tuple[sqlite3.Connection, dict[str, Any]]:
    """Rebuild ranking state in a separate local database, then compare refs."""
    assignments_path = split_root / "assignments.parquet"
    identity = {
        **input_identity,
        "policy_sha256": policy["_policy_sha256"],
        "implementation_sha256": compute_file_sha256(Path(__file__)),
        "sampling_seed": int(policy["sampling_seed"]),
        "operation": "independent-verify-selection-v1",
    }
    connection = _connect(scratch_dir / "verification.sqlite3")
    _store_identity(connection, identity)
    quotas = _quota_plan(policy)
    row_group_layout = _read_row_group_layout(input_root, d1_manifest, split_manifest)
    _populate_candidates(
        connection,
        assignments_path,
        policy,
        quotas,
        d1_manifest,
        split_manifest,
        row_group_layout,
        batch_size=batch_size,
        reverse_row_groups=False,
    )
    candidate_count = int(
        connection.execute("SELECT count(*) FROM candidate").fetchone()[0]
    )
    summaries = _choose_candidates(connection, quotas, policy)
    report = _quota_report(summaries, policy)
    if report["status"] != "PASS":
        connection.close()
        raise ValueError("Independent reconstruction fails the frozen quota tolerances")
    published_path = selection_dir / "selected_references.parquet"
    published = pq.ParquetFile(published_path)
    candidate_rows = _selected_reference_rows(connection)
    published_rows = (
        row
        for batch in published.iter_batches(batch_size=4096)
        for row in batch.to_pylist()
    )
    compared = 0
    sentinel = object()
    while True:
        expected = next(candidate_rows, sentinel)
        actual = next(published_rows, sentinel)
        if expected is sentinel and actual is sentinel:
            break
        if expected is sentinel or actual is sentinel or expected != actual:
            connection.close()
            raise ValueError(
                "Published selection differs from independent deterministic reconstruction"
            )
        compared += 1
    if compared != int(
        connection.execute("SELECT count(*) FROM selected").fetchone()[0]
    ):
        connection.close()
        raise ValueError("Independent selection comparison count mismatch")
    published_manifest = json.loads(
        (selection_dir / "selection_manifest.json").read_text()
    )
    if candidate_count != int(published_manifest.get("eligible_train_documents", -1)):
        connection.close()
        raise ValueError("Independent reconstruction training-pool count mismatch")
    if summaries != published_manifest.get("quotas"):
        connection.close()
        raise ValueError(
            "Selection bucket counts or materialization frame report mismatch"
        )
    if report != published_manifest.get("quota_report"):
        connection.close()
        raise ValueError(
            "Selection quota report differs from independent reconstruction"
        )
    return connection, dict(
        status="PASS",
        eligible_train_documents=candidate_count,
        independently_selected_documents=compared,
        quotas=summaries,
        quota_report=report,
    )


def _compare_published_shard(
    connection: sqlite3.Connection,
    input_root: Path,
    sample_path: Path,
    source_shard: str,
    *,
    batch_size: int,
) -> tuple[int, int, dict[str, dict[str, int]]]:
    source_schema, original_rows, details = _source_rows_for_shard(
        connection, input_root, source_shard, batch_size=batch_size
    )
    sample_parquet = pq.ParquetFile(sample_path)
    expected_names = list(_sample_schema(source_schema).names)
    if sample_parquet.schema_arrow.names != expected_names:
        raise ValueError(f"Published sample schema mismatch for {source_shard}")
    output_rows = (
        row
        for batch in sample_parquet.iter_batches(batch_size=1024)
        for row in batch.to_pylist()
    )
    count = 0
    words = 0
    by_source_subset: dict[str, dict[str, int]] = {}
    sentinel = object()
    for expected in original_rows:
        observed = next(output_rows, sentinel)
        if observed is sentinel or expected != observed:
            raise ValueError(
                f"Published record differs from pinned source: {source_shard}"
            )
        if observed["split"] != "train":
            raise ValueError("Published sample contains a held-out record")
        text = observed["text"]
        if _sha256_text(text) != observed["content_sha256"]:
            raise ValueError("Published document text checksum mismatch")
        if len(text.split()) != observed["normalized_words"]:
            raise ValueError("Published normalized word count mismatch")
        count += 1
        word_count = int(observed["normalized_words"])
        words += word_count
        key = (
            f"{observed['source']}/{observed['subset']}"
            if observed["subset"]
            else observed["source"]
        )
        subtotal = by_source_subset.setdefault(
            key, {"documents": 0, "normalized_words": 0}
        )
        subtotal["documents"] += 1
        subtotal["normalized_words"] += word_count
    if next(output_rows, sentinel) is not sentinel or count != details["rows"]:
        raise ValueError(f"Published record count mismatch for {source_shard}")
    return count, words, by_source_subset


def verify(
    input_root: Path = DEFAULT_INPUT,
    split_root: Path = DEFAULT_SPLIT,
    selection_dir: Path = DEFAULT_ROOT / "selection",
    sample_dir: Path = DEFAULT_ROOT / "sample",
    scratch_dir: Path = Path("/tmp/c1-tokenizer-sample-v1"),
    policy_path: Path = DEFAULT_POLICY,
    *,
    batch_size: int = 4096,
) -> dict[str, Any]:
    """Independently rebuild selection and verify published sample provenance."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    _require_separate_paths(
        sample_dir, scratch_dir, (input_root, split_root, selection_dir)
    )
    policy = load_policy(policy_path)
    d1_manifest, split_manifest, input_identity = _read_input_identities(
        input_root, split_root, verify_assignment_payload=True
    )
    selection_manifest, selection_sha, _ = _load_selection(
        selection_dir, policy, input_identity
    )
    sample_manifest = _validate_published_sample(sample_dir, selection_sha)
    if (
        sample_manifest.get("input_identity") != input_identity
        or sample_manifest.get("policy_sha256") != policy["_policy_sha256"]
        or sample_manifest.get("quota_report") != selection_manifest.get("quota_report")
    ):
        raise ValueError("Published sample manifest identity or quota report mismatch")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    reconstruction, reconstruction_result = _verify_reconstructed_selection(
        input_root,
        split_root,
        selection_dir,
        scratch_dir,
        policy,
        d1_manifest,
        split_manifest,
        input_identity,
        batch_size=batch_size,
    )
    try:
        reference_db_path = scratch_dir / "verification_refs.sqlite3"
        reference_db = _connect(reference_db_path)
        try:
            _store_identity(
                reference_db,
                {
                    **input_identity,
                    "selection_manifest_sha256": selection_sha,
                    "operation": "verify-materialized-records-v1",
                },
            )
            _initialize_materialize_db(reference_db)
            if not reference_db.execute(
                "SELECT 1 FROM selected_ref LIMIT 1"
            ).fetchone():
                _load_selected_refs(
                    reference_db,
                    selection_dir / "selected_references.parquet",
                    policy,
                )
            actual_totals: dict[str, dict[str, int]] = {}
            actual_by_source_subset: dict[str, dict[str, int]] = {}
            total_documents = 0
            total_words = 0
            artifact_rows = 0
            for relative_path, artifact in sorted(sample_manifest["artifacts"].items()):
                source_shard = artifact["source_shard"]
                if (
                    PurePosixPath(relative_path).is_absolute()
                    or ".." in PurePosixPath(relative_path).parts
                ):
                    raise ValueError("Unsafe sample artifact path")
                source_path = input_root / "data" / source_shard
                expected_entry = d1_manifest.get("output_files", {}).get(
                    f"data/{source_shard}"
                )
                split_entry = split_manifest.get("input_files", {}).get(
                    f"data/{source_shard}"
                )
                if (
                    not expected_entry
                    or not split_entry
                    or any(
                        expected_entry.get(key) != split_entry.get(key)
                        for key in ("bytes", "rows", "sha256")
                    )
                    or source_path.stat().st_size != int(expected_entry["bytes"])
                ):
                    raise ValueError(
                        f"Published sample references an unpinned source shard: {source_shard}"
                    )
                count, words, source_subset_counts = _compare_published_shard(
                    reference_db,
                    input_root,
                    sample_dir / relative_path,
                    source_shard,
                    batch_size=batch_size,
                )
                if count != int(artifact["rows"]):
                    raise ValueError(
                        f"Sample artifact row count mismatch: {relative_path}"
                    )
                actual_totals[source_shard] = {
                    "documents": count,
                    "normalized_words": words,
                }
                for key, subtotal in source_subset_counts.items():
                    aggregate = actual_by_source_subset.setdefault(
                        key, {"documents": 0, "normalized_words": 0}
                    )
                    aggregate["documents"] += subtotal["documents"]
                    aggregate["normalized_words"] += subtotal["normalized_words"]
                total_documents += count
                total_words += words
                artifact_rows += int(artifact["rows"])
            if (
                total_documents != int(selection_manifest["selected_documents"])
                or total_words != int(selection_manifest["selected_normalized_words"])
                or total_documents != int(sample_manifest["documents"])
                or total_words != int(sample_manifest["normalized_words"])
                or artifact_rows != total_documents
                or actual_by_source_subset != sample_manifest.get("by_source_subset")
            ):
                raise ValueError(
                    "Published sample totals differ from independently checked records"
                )
            actual_by_bucket = {
                bucket: (int(documents), int(words))
                for bucket, documents, words in reference_db.execute(
                    """SELECT sample_bucket,count(*),sum(normalized_words)
                       FROM selected_ref GROUP BY sample_bucket"""
                )
            }
            recalculated = _quota_report(
                {
                    bucket: {
                        "selected_words": actual_by_bucket.get(bucket, (0, 0))[1],
                        "selected_documents": actual_by_bucket.get(bucket, (0, 0))[0],
                        "target_words": int(plan["target"]),
                        "eligible_train_documents": actual_by_bucket.get(
                            bucket, (0, 0)
                        )[0],
                        "eligible_train_words": actual_by_bucket.get(bucket, (0, 0))[1],
                    }
                    for bucket, plan in _quota_plan(policy).items()
                },
                policy,
            )
            expected_categories = selection_manifest["quota_report"]["categories"]
            observed_categories = recalculated["categories"]
            category_keys = (
                "level",
                "bucket",
                "target_words",
                "actual_words",
                "deviation_words",
                "within_tolerance",
            )
            expected_simple = [
                {key: row[key] for key in category_keys} for row in expected_categories
            ]
            observed_simple = [
                {key: row[key] for key in category_keys} for row in observed_categories
            ]
            if recalculated["status"] != "PASS" or expected_simple != observed_simple:
                raise ValueError(
                    "Published sample quota report differs from actual records"
                )
            return dict(
                status="PASS",
                documents=total_documents,
                normalized_words=total_words,
                independent_selection=reconstruction_result,
                verified_source_shards=len(actual_totals),
                input_identity=input_identity,
                selection_manifest_sha256=selection_sha,
                sample_manifest_sha256=compute_file_sha256(
                    sample_dir / "sample_manifest.json"
                ),
                quota_report=selection_manifest["quota_report"],
            )
        finally:
            reference_db.close()
    finally:
        reconstruction.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for operation in ("select", "materialize", "verify"):
        child = commands.add_parser(operation)
        child.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
        child.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT)
        child.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
        child.add_argument("--scratch-dir", type=Path, required=True)
        child.add_argument(
            "--batch-size",
            type=int,
            default=4096 if operation != "materialize" else 1024,
        )
        if operation == "select":
            child.add_argument(
                "--output-dir", type=Path, default=DEFAULT_ROOT / "selection"
            )
        elif operation == "materialize":
            child.add_argument(
                "--selection-dir", type=Path, default=DEFAULT_ROOT / "selection"
            )
            child.add_argument(
                "--output-dir", type=Path, default=DEFAULT_ROOT / "sample"
            )
        else:
            child.add_argument(
                "--selection-dir", type=Path, default=DEFAULT_ROOT / "selection"
            )
            child.add_argument(
                "--sample-dir", type=Path, default=DEFAULT_ROOT / "sample"
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = dict(
        input_root=args.input_root,
        split_root=args.split_root,
        scratch_dir=args.scratch_dir,
        policy_path=args.policy,
        batch_size=args.batch_size,
    )
    if args.operation == "select":
        result = select(output_dir=args.output_dir, **common)
    elif args.operation == "materialize":
        result = materialize(
            selection_dir=args.selection_dir, output_dir=args.output_dir, **common
        )
    else:
        result = verify(
            selection_dir=args.selection_dir, sample_dir=args.sample_dir, **common
        )
    print(canonical(result))
    return (
        0
        if result.get("status") in ("PASS", "MATERIALIZED", "ALREADY_PUBLISHED")
        else 3
    )


if __name__ == "__main__":
    raise SystemExit(main())
