"""External-memory exact deduplication for the frozen Gate C1 pools.

The implementation first indexes projected Parquet metadata in SQLite, chooses
deterministic representatives, then streams text only for word accounting and
retained-row materialization. SQLite keeps the record index and verifier state
on disk so corpus size does not determine Python object memory use.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import resource
import shutil
import sqlite3
import tempfile
import time
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from cambacica.corpus.manifest import compute_file_sha256
from cambacica.corpus.normalization import (
    DEFAULT_NORMALIZED_ROOT,
    NORMALIZATION_VERSION,
    NORMALIZED_SCHEMA,
    SOURCE_CONFIG,
)


EXACT_DEDUP_VERSION = "1.0.0"
DEFAULT_EXACT_ROOT = Path("/mnt/data/cambacica-base-180m/deduplicated/exact")
DEFAULT_PILOT_SIZE = 1_200
DEFAULT_PILOT_SEED = 20261004
OUTPUT_ROW_GROUP_ROWS = 4_096
OUTPUT_ROW_BUFFER_TEXT_BYTES = 64 * 1024 * 1024
PRESERVE_SOURCE = "parlamento_pt"
PRIMARY_SOURCES = {"carolina", "gutenberg_pt", "wikipedia_pt"}
GV_SUBSET_TIERS: dict[str, int] = {
    # The tier order follows the documented C1 composition buckets.
    "finepdfs_por_Latn": 1,  # Formal / technical
    "crawlPT_dedup": 2,  # Curated / native Portuguese web
    "quati": 2,
    "blogset": 2,
    "fineweb_2_pt": 3,  # Modern general web
    "mc4_pt": 4,  # Legacy web
    "hplt2_pt": 4,
    "hplt1_pt": 4,
    "common_crawl": 4,
    "oscar": 4,
    "culturax": 4,
}

INDEX_COLUMNS = [
    "content_sha256",
    "source",
    "subset",
    "source_revision",
    "original_id",
    "raw_source_file",
    "raw_record_identifier",
    "_gv2_upstream_shard",
    "_gv2_upstream_row_group",
    "_gv2_upstream_commit",
]
MATERIALIZE_COLUMNS = list(NORMALIZED_SCHEMA.names)

RESOLUTION_SCHEMA = pa.schema(
    [
        pa.field("record_id", pa.string(), nullable=False),
        pa.field("cluster_id", pa.string()),
        pa.field("content_sha256", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("subset", pa.string(), nullable=False),
        pa.field("normalized_words", pa.int64(), nullable=False),
        pa.field("normalized_shard", pa.string(), nullable=False),
        pa.field("raw_source_file", pa.string(), nullable=False),
        pa.field("raw_record_identifier", pa.string(), nullable=False),
        pa.field("representative_record_id", pa.string(), nullable=False),
        pa.field("disposition", pa.string(), nullable=False),
        pa.field("ownership_rule", pa.string(), nullable=False),
        pa.field("selection_role", pa.string()),
    ]
)
EDGE_SCHEMA = pa.schema(
    [
        pa.field("cluster_id", pa.string(), nullable=False),
        pa.field("content_sha256", pa.string(), nullable=False),
        pa.field("dropped_record_id", pa.string(), nullable=False),
        pa.field("retained_record_id", pa.string(), nullable=False),
        pa.field("dropped_source", pa.string(), nullable=False),
        pa.field("dropped_subset", pa.string(), nullable=False),
        pa.field("retained_source", pa.string(), nullable=False),
        pa.field("retained_subset", pa.string(), nullable=False),
        pa.field("ownership_rule", pa.string(), nullable=False),
    ]
)
CLUSTER_SCHEMA = pa.schema(
    [
        pa.field("cluster_id", pa.string(), nullable=False),
        pa.field("content_sha256", pa.string(), nullable=False),
        pa.field("retained_record_id", pa.string(), nullable=False),
        pa.field("eligible_record_count", pa.int64(), nullable=False),
        pa.field("duplicate_record_count", pa.int64(), nullable=False),
        pa.field("source_count", pa.int32(), nullable=False),
        pa.field("source_counts_json", pa.string(), nullable=False),
        pa.field("subset_counts_json", pa.string(), nullable=False),
        pa.field("ownership_rule", pa.string(), nullable=False),
    ]
)
PARLAMENTO_DIAGNOSTIC_SCHEMA = pa.schema(
    [
        pa.field("content_sha256", pa.string(), nullable=False),
        pa.field("record_count", pa.int64(), nullable=False),
        pa.field("surplus_record_count", pa.int64(), nullable=False),
    ]
)
PILOT_INDEX_SCHEMA = pa.schema(
    [
        pa.field("record_id", pa.string(), nullable=False),
        pa.field("content_sha256", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("subset", pa.string(), nullable=False),
        pa.field("source_revision", pa.string()),
        pa.field("original_id", pa.string()),
        pa.field("_gv2_upstream_shard", pa.string()),
        pa.field("_gv2_upstream_row_group", pa.int32()),
        pa.field("_gv2_upstream_commit", pa.string()),
        pa.field("normalized_words", pa.int64(), nullable=False),
        pa.field("normalized_shard", pa.string(), nullable=False),
        pa.field("raw_source_file", pa.string(), nullable=False),
        pa.field("raw_record_identifier", pa.string(), nullable=False),
        pa.field("selection_role", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class ExactInputFile:
    """One Parquet file to index, with its immutable normalized location."""

    source: str
    path: Path
    normalized_shard: str
    relative_path: str
    selection_role: str | None = None


@dataclass
class _ParquetSink:
    path: Path
    schema: pa.Schema
    batch_size: int = 4_096

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.partial_path = self.path.with_name(f"{self.path.name}.partial")
        self.partial_path.unlink(missing_ok=True)
        self.writer = pq.ParquetWriter(
            self.partial_path,
            self.schema,
            compression="zstd",
            compression_level=6,
            use_dictionary=True,
            write_statistics=True,
            version="2.6",
        )
        self.rows: list[dict[str, Any]] = []
        self.row_count = 0
        self.closed = False

    def append(self, row: Mapping[str, Any]) -> None:
        self.rows.append(dict(row))
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        self.writer.write_table(table, row_group_size=self.batch_size)
        self.row_count += len(self.rows)
        self.rows.clear()

    def close(self) -> None:
        if self.closed:
            return
        self.flush()
        self.writer.close()
        with self.partial_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(self.partial_path, self.path)
        self.closed = True


def stable_record_id(row: Mapping[str, Any]) -> str:
    """Return the stable identity digest for one normalized source record."""
    identity = {
        key: row.get(key)
        for key in (
            "source",
            "source_revision",
            "subset",
            "original_id",
            "raw_source_file",
            "raw_record_identifier",
            "_gv2_upstream_shard",
            "_gv2_upstream_row_group",
            "_gv2_upstream_commit",
        )
    }
    payload = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(b"cambacica-exact-record-v1\0" + payload).hexdigest()


def exact_cluster_id(content_sha256: str) -> str:
    """Return a namespaced cluster identifier for an eligible exact hash."""
    return hashlib.sha256(
        b"cambacica-exact-cluster-v1\0" + content_sha256.encode("ascii")
    ).hexdigest()


def _count_words_bounded(text: str) -> int:
    """Count Python Unicode-whitespace-delimited words without a split list."""
    count = 0
    in_word = False
    for character in text:
        if character.isspace():
            in_word = False
        elif not in_word:
            count += 1
            in_word = True
    return count


def _ownership_key(source: str, subset: str, record_id: str) -> tuple[Any, ...]:
    if source in PRIMARY_SOURCES:
        # Alphabetical source order is the documented tie-break among native
        # primary records; source, subset and record identity complete the key.
        return (0, source, subset, record_id)
    if source == "gigaverbo_v2":
        return (1, GV_SUBSET_TIERS.get(subset, 5), subset, record_id)
    return (2, source, subset, record_id)


def _ownership_rule(source: str, duplicate_count: int) -> str:
    if duplicate_count == 0:
        return "unique_content_hash"
    if source in PRIMARY_SOURCES:
        return "primary_source_precedence_then_source_subset_record_id"
    if source == "gigaverbo_v2":
        return "gigaverbo_provenance_tier_subset_record_id"
    return "source_subset_record_id"


def _sqlite_connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute("PRAGMA mmap_size=0")
    return connection


def _create_index_db(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE records (
            record_id TEXT PRIMARY KEY,
            content_sha256 TEXT NOT NULL,
            source TEXT NOT NULL,
            subset TEXT NOT NULL,
            source_revision TEXT,
            original_id TEXT,
            raw_source_file TEXT NOT NULL,
            raw_record_identifier TEXT NOT NULL,
            normalized_shard TEXT NOT NULL,
            gv_upstream_shard TEXT,
            gv_upstream_row_group INTEGER,
            gv_upstream_commit TEXT,
            normalized_words INTEGER,
            representative_record_id TEXT,
            cluster_id TEXT,
            disposition TEXT,
            ownership_rule TEXT,
            selection_role TEXT,
            sort_kind INTEGER NOT NULL,
            subset_tier INTEGER NOT NULL
        );
        CREATE INDEX records_by_hash ON records(content_sha256);
        CREATE INDEX records_by_source_subset ON records(source, subset);
        CREATE TABLE materialized_ids (record_id TEXT PRIMARY KEY);
        """
    )


def _insert_index_rows(
    connection: sqlite3.Connection, rows: Sequence[tuple[Any, ...]]
) -> int:
    insert = """
        INSERT INTO records (
            record_id, content_sha256, source, subset, source_revision,
            original_id, raw_source_file, raw_record_identifier,
            normalized_shard, gv_upstream_shard, gv_upstream_row_group,
            gv_upstream_commit, selection_role, sort_kind, subset_tier
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    connection.executemany(insert, rows)
    return len(rows)


def _index_files(
    connection: sqlite3.Connection,
    input_files: Sequence[ExactInputFile],
    *,
    batch_size: int = 2_048,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    uncommitted = 0
    for item in sorted(
        input_files, key=lambda value: (value.source, value.relative_path)
    ):
        parquet = pq.ParquetFile(item.path)
        names = set(parquet.schema_arrow.names)
        missing = set(INDEX_COLUMNS) - names
        if missing:
            raise ValueError(f"{item.path} is missing index columns: {sorted(missing)}")
        columns = list(INDEX_COLUMNS)
        if "pilot_selection_role" in names:
            columns.append("pilot_selection_role")
        indexed: list[tuple[Any, ...]] = []
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                source = str(row.get("source") or "")
                if source != item.source:
                    raise ValueError(
                        f"Source column mismatch in {item.path}: {source!r} != {item.source!r}"
                    )
                subset = str(row.get("subset") or "")
                content_hash = str(row.get("content_sha256") or "")
                if len(content_hash) != 64 or any(
                    char not in "0123456789abcdef" for char in content_hash
                ):
                    raise ValueError(
                        f"Invalid content_sha256 at {item.path}: {content_hash!r}"
                    )
                for field in ("raw_source_file", "raw_record_identifier"):
                    if not row.get(field):
                        raise ValueError(f"Missing {field} in {item.path}")
                record_id = stable_record_id(row)
                role = row.get("pilot_selection_role") or item.selection_role
                key = _ownership_key(source, subset, record_id)
                indexed.append(
                    (
                        record_id,
                        content_hash,
                        source,
                        subset,
                        row.get("source_revision"),
                        row.get("original_id"),
                        str(row["raw_source_file"]),
                        str(row["raw_record_identifier"]),
                        item.normalized_shard,
                        row.get("_gv2_upstream_shard"),
                        row.get("_gv2_upstream_row_group"),
                        row.get("_gv2_upstream_commit"),
                        role,
                        int(key[0]),
                        int(key[1]) if key[0] == 1 else 0,
                    )
                )
                if len(indexed) >= batch_size:
                    counts[source] += _insert_index_rows(connection, indexed)
                    uncommitted += len(indexed)
                    indexed.clear()
                    if uncommitted >= 50_000:
                        connection.commit()
                        uncommitted = 0
        if indexed:
            counts[item.source] += _insert_index_rows(connection, indexed)
            uncommitted += len(indexed)
            if uncommitted >= 50_000:
                connection.commit()
                uncommitted = 0
    connection.commit()
    return dict(counts)


def _resolve_hash_groups(
    connection: sqlite3.Connection,
    stage: Path,
) -> dict[str, Any]:
    edge_sink = _ParquetSink(stage / "duplicate_edges.parquet", EDGE_SCHEMA)
    cluster_sink = _ParquetSink(stage / "cluster_index.parquet", CLUSTER_SCHEMA)
    cursor = connection.execute(
        """
        SELECT record_id, content_sha256, source, subset, normalized_shard,
               raw_source_file, raw_record_identifier, sort_kind, subset_tier
        FROM records
        WHERE source <> ?
        ORDER BY content_sha256, sort_kind, subset_tier, source, subset, record_id
        """,
        (PRESERVE_SOURCE,),
    )

    group_hash: str | None = None
    winner: tuple[Any, ...] | None = None
    group_count = 0
    source_counts: Counter[str] = Counter()
    subset_counts: Counter[str] = Counter()
    dropped_records = 0

    def finish_group() -> None:
        nonlocal group_hash, winner, group_count, source_counts, subset_counts
        if group_hash is None or winner is None:
            return
        cluster = exact_cluster_id(group_hash)
        rule = _ownership_rule(str(winner[2]), max(1, group_count - 1))
        cluster_sink.append(
            {
                "cluster_id": cluster,
                "content_sha256": group_hash,
                "retained_record_id": winner[0],
                "eligible_record_count": group_count,
                "duplicate_record_count": group_count - 1,
                "source_count": len(source_counts),
                "source_counts_json": json.dumps(
                    dict(sorted(source_counts.items())),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "subset_counts_json": json.dumps(
                    dict(sorted(subset_counts.items())),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "ownership_rule": rule,
            }
        )
        source_counts = Counter()
        subset_counts = Counter()

    for record in cursor:
        record_id, content_hash, source, subset = record[:4]
        if group_hash != content_hash:
            finish_group()
            group_hash = content_hash
            winner = record
            group_count = 0
        group_count += 1
        source_counts[source] += 1
        subset_counts[f"{source}/{subset}"] += 1
        cluster = exact_cluster_id(content_hash)
        rule = _ownership_rule(winner[2], max(1, group_count - 1))
        disposition = "retained" if record_id == winner[0] else "dropped"
        connection.execute(
            """
            UPDATE records
            SET representative_record_id=?, cluster_id=?, disposition=?, ownership_rule=?
            WHERE record_id=?
            """,
            (winner[0], cluster, disposition, rule, record_id),
        )
        if disposition == "dropped":
            dropped_records += 1
            edge_sink.append(
                {
                    "cluster_id": cluster,
                    "content_sha256": content_hash,
                    "dropped_record_id": record_id,
                    "retained_record_id": winner[0],
                    "dropped_source": source,
                    "dropped_subset": subset,
                    "retained_source": winner[2],
                    "retained_subset": winner[3],
                    "ownership_rule": rule,
                }
            )
        if group_count % 50_000 == 0:
            connection.commit()
    finish_group()
    connection.commit()
    edge_sink.close()
    cluster_sink.close()

    duplicate_rows = connection.execute(
        "SELECT COUNT(*) FROM records WHERE source<>? AND disposition='dropped'",
        (PRESERVE_SOURCE,),
    ).fetchone()[0]
    duplicate_groups = connection.execute(
        "SELECT COUNT(*) FROM (SELECT content_sha256 FROM records "
        "WHERE source<>? GROUP BY content_sha256 HAVING COUNT(*)>1)",
        (PRESERVE_SOURCE,),
    ).fetchone()[0]
    cross_source_groups = connection.execute(
        "SELECT COUNT(*) FROM (SELECT content_sha256 FROM records "
        "WHERE source<>? GROUP BY content_sha256 HAVING COUNT(DISTINCT source)>1)",
        (PRESERVE_SOURCE,),
    ).fetchone()[0]
    return {
        "eligible_duplicate_hash_groups": int(duplicate_groups),
        "eligible_dropped_records": int(duplicate_rows),
        "eligible_cross_source_duplicate_hash_groups": int(cross_source_groups),
    }


def _set_parlamento_self_resolution(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        UPDATE records
        SET representative_record_id=record_id,
            cluster_id=NULL,
            disposition='preserved_diagnostic',
            ownership_rule='parlamento_preserve_all'
        WHERE source=?
        """,
        (PRESERVE_SOURCE,),
    )
    connection.commit()


def _parlamento_diagnostics(
    connection: sqlite3.Connection, stage: Path
) -> dict[str, int]:
    sink = _ParquetSink(
        stage / "parlamento_duplicate_hashes.parquet", PARLAMENTO_DIAGNOSTIC_SCHEMA
    )
    rows = connection.execute(
        "SELECT content_sha256, COUNT(*) AS n FROM records WHERE source=? "
        "GROUP BY content_sha256 HAVING COUNT(*)>1 ORDER BY content_sha256",
        (PRESERVE_SOURCE,),
    )
    groups = 0
    repeated_records = 0
    surplus_records = 0
    for content_hash, count in rows:
        groups += 1
        repeated_records += int(count)
        surplus_records += int(count) - 1
        sink.append(
            {
                "content_sha256": content_hash,
                "record_count": int(count),
                "surplus_record_count": int(count) - 1,
            }
        )
    sink.close()
    return {
        "duplicate_hash_groups": groups,
        "records_in_repeated_hash_groups": repeated_records,
        "surplus_repeated_records": surplus_records,
    }


def _materialize_and_count(
    connection: sqlite3.Connection,
    input_files: Sequence[ExactInputFile],
    stage: Path,
    *,
    batch_size: int = 32,
) -> tuple[dict[str, Any], list[Path]]:
    source_counts: Counter[str] = Counter()
    source_words: Counter[str] = Counter()
    subset_counts: Counter[tuple[str, str]] = Counter()
    subset_words: Counter[tuple[str, str]] = Counter()
    data_paths: list[Path] = []
    lookup = connection.cursor()
    update = connection.cursor()
    for item in sorted(
        input_files, key=lambda value: (value.source, value.relative_path)
    ):
        parquet = pq.ParquetFile(item.path)
        file_names = set(parquet.schema_arrow.names)
        read_columns = list(MATERIALIZE_COLUMNS)
        if "pilot_selection_role" in file_names:
            read_columns.append("pilot_selection_role")
        relative = PurePosixPath(item.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe normalized shard path: {item.relative_path}")
        output_file = stage / "data" / item.source / Path(*relative.parts)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = output_file.with_name(f"{output_file.name}.partial")
        writer: pq.ParquetWriter | None = None
        kept_rows: list[dict[str, Any]] = []
        kept_text_bytes = 0

        def flush_kept_rows() -> None:
            nonlocal writer, kept_text_bytes
            if not kept_rows:
                return
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_file,
                    NORMALIZED_SCHEMA,
                    compression="zstd",
                    compression_level=6,
                    use_dictionary=True,
                    write_statistics=True,
                    version="2.6",
                )
            writer.write_table(
                pa.Table.from_pylist(kept_rows, schema=NORMALIZED_SCHEMA),
                row_group_size=OUTPUT_ROW_GROUP_ROWS,
            )
            kept_rows.clear()
            kept_text_bytes = 0

        for batch in parquet.iter_batches(batch_size=batch_size, columns=read_columns):
            for row in batch.to_pylist():
                record_id = stable_record_id(row)
                found = lookup.execute(
                    "SELECT representative_record_id, disposition FROM records WHERE record_id=?",
                    (record_id,),
                ).fetchone()
                if found is None:
                    raise ValueError(
                        f"Input row is absent from compact index: {record_id}"
                    )
                representative_id, disposition = found
                text = row["text"] or ""
                words = _count_words_bounded(text)
                subset = str(row.get("subset") or "")
                source_counts[item.source] += 1
                source_words[item.source] += words
                subset_counts[(item.source, subset)] += 1
                subset_words[(item.source, subset)] += words
                update.execute(
                    "UPDATE records SET normalized_words=? WHERE record_id=?",
                    (words, record_id),
                )
                if disposition != "dropped":
                    materialized = {name: row.get(name) for name in MATERIALIZE_COLUMNS}
                    kept_rows.append(materialized)
                    kept_text_bytes += len(text.encode("utf-8"))
                if (
                    len(kept_rows) >= OUTPUT_ROW_GROUP_ROWS
                    or kept_text_bytes >= OUTPUT_ROW_BUFFER_TEXT_BYTES
                ):
                    flush_kept_rows()
        flush_kept_rows()
        if writer is not None:
            writer.close()
            with temporary_file.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary_file, output_file)
            data_paths.append(output_file)
        else:
            temporary_file.unlink(missing_ok=True)
        connection.commit()

    indexed_sources = {
        source: int(count)
        for source, count in connection.execute(
            "SELECT source, COUNT(*) FROM records GROUP BY source"
        )
    }
    if dict(source_counts) != indexed_sources:
        raise ValueError("Text materialization row counts do not match compact index.")
    return (
        {
            "documents_by_source": dict(sorted(source_counts.items())),
            "words_by_source": dict(sorted(source_words.items())),
            "documents_by_source_subset": {
                f"{source}/{subset}": int(count)
                for (source, subset), count in sorted(subset_counts.items())
            },
            "words_by_source_subset": {
                f"{source}/{subset}": int(count)
                for (source, subset), count in sorted(subset_words.items())
            },
        },
        data_paths,
    )


def _write_resolution_files(
    connection: sqlite3.Connection,
    stage: Path,
    *,
    pilot: bool,
) -> None:
    resolution_sink = _ParquetSink(
        stage / "record_resolution.parquet", RESOLUTION_SCHEMA
    )
    pilot_sink = (
        _ParquetSink(stage / "pilot_input_index.parquet", PILOT_INDEX_SCHEMA)
        if pilot
        else None
    )
    query = connection.execute(
        """
        SELECT record_id, cluster_id, content_sha256, source, subset,
               normalized_words, normalized_shard, raw_source_file,
               raw_record_identifier, representative_record_id, disposition,
               ownership_rule, selection_role, source_revision, original_id,
               gv_upstream_shard, gv_upstream_row_group, gv_upstream_commit
        FROM records ORDER BY source, subset, record_id
        """
    )
    for row in query:
        if row[5] is None or row[9] is None or row[10] is None or row[11] is None:
            raise ValueError(f"Incomplete resolution row for record {row[0]}")
        resolution_sink.append(
            dict(
                zip(
                    RESOLUTION_SCHEMA.names,
                    row[: len(RESOLUTION_SCHEMA.names)],
                    strict=True,
                )
            )
        )
        if pilot_sink is not None:
            pilot_sink.append(
                {
                    "record_id": row[0],
                    "content_sha256": row[2],
                    "source": row[3],
                    "subset": row[4],
                    "source_revision": row[13],
                    "original_id": row[14],
                    "_gv2_upstream_shard": row[15],
                    "_gv2_upstream_row_group": row[16],
                    "_gv2_upstream_commit": row[17],
                    "normalized_words": row[5],
                    "normalized_shard": row[6],
                    "raw_source_file": row[7],
                    "raw_record_identifier": row[8],
                    "selection_role": row[12] or "stratified",
                }
            )
    resolution_sink.close()
    if pilot_sink is not None:
        pilot_sink.close()


def _write_accounting(
    connection: sqlite3.Connection,
    stage: Path,
    source_counts: Mapping[str, int],
) -> list[dict[str, Any]]:
    aggregates: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {
            "documents_before": 0,
            "normalized_words_before": 0,
            "documents_after": 0,
            "normalized_words_after": 0,
        }
    )
    for source, subset, words, disposition in connection.execute(
        "SELECT source, subset, normalized_words, disposition FROM records"
    ):
        key = (source, subset)
        aggregates[key]["documents_before"] += 1
        aggregates[key]["normalized_words_before"] += int(words)
        if disposition != "dropped":
            aggregates[key]["documents_after"] += 1
            aggregates[key]["normalized_words_after"] += int(words)
    rows: list[dict[str, Any]] = []
    for source in sorted(source_counts):
        source_subset_keys = sorted(key for key in aggregates if key[0] == source)
        totals = {
            field: sum(aggregates[key][field] for key in source_subset_keys)
            for field in (
                "documents_before",
                "normalized_words_before",
                "documents_after",
                "normalized_words_after",
            )
        }
        rows.append(_accounting_row(source, "", "source_total", totals))
        for _, subset in source_subset_keys:
            rows.append(
                _accounting_row(
                    source, subset, "source_subset", aggregates[(source, subset)]
                )
            )
    with (stage / "accounting_by_source_subset.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fields = [
            "source",
            "subset",
            "scope",
            "documents_before",
            "normalized_words_before",
            "documents_after",
            "normalized_words_after",
            "documents_removed",
            "words_removed",
            "loss_fraction",
            "document_loss_fraction",
            "word_loss_fraction",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    return rows


def _accounting_row(
    source: str, subset: str, scope: str, totals: Mapping[str, int]
) -> dict[str, Any]:
    removed_documents = int(totals["documents_before"]) - int(totals["documents_after"])
    removed_words = int(totals["normalized_words_before"]) - int(
        totals["normalized_words_after"]
    )
    before_documents = int(totals["documents_before"])
    return {
        "source": source,
        "subset": subset,
        "scope": scope,
        "documents_before": before_documents,
        "normalized_words_before": int(totals["normalized_words_before"]),
        "documents_after": int(totals["documents_after"]),
        "normalized_words_after": int(totals["normalized_words_after"]),
        "documents_removed": removed_documents,
        "words_removed": removed_words,
        "loss_fraction": (
            round(removed_documents / before_documents, 12) if before_documents else 0.0
        ),
        "document_loss_fraction": (
            round(removed_documents / before_documents, 12) if before_documents else 0.0
        ),
        "word_loss_fraction": (
            round(removed_words / int(totals["normalized_words_before"]), 12)
            if int(totals["normalized_words_before"])
            else 0.0
        ),
    }


def _file_inventory(stage: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(p for p in stage.rglob("*") if p.is_file()):
        relative = path.relative_to(stage).as_posix()
        if relative in {"manifest.json", "manifest.json.partial"}:
            continue
        record: dict[str, Any] = {
            "sha256": compute_file_sha256(path),
            "bytes": path.stat().st_size,
        }
        if path.suffix == ".parquet":
            record["rows"] = int(pq.ParquetFile(path).metadata.num_rows)
        inventory[relative] = record
    return inventory


def _atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _input_catalog(
    normalized_root: Path,
) -> tuple[dict[str, dict[str, Any]], list[ExactInputFile]]:
    manifests: dict[str, dict[str, Any]] = {}
    input_files: list[ExactInputFile] = []
    for source in sorted(SOURCE_CONFIG):
        source_dir = normalized_root / SOURCE_CONFIG[source]["output_dir"]
        manifest_path = source_dir / "manifest.json"
        progress_path = source_dir / "manifest.in_progress.json"
        if not manifest_path.is_file() or progress_path.exists():
            raise ValueError(
                f"Completed normalized manifest is unavailable for {source}."
            )
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest.get("source") != source or manifest.get("status") != "COMPLETE":
            raise ValueError(f"Normalized manifest for {source} is not COMPLETE.")
        if manifest.get("normalization_schema_version") != NORMALIZATION_VERSION:
            raise ValueError(f"Normalization version mismatch for {source}.")
        if int(manifest.get("normalization_failure_count", -1)) != 0:
            raise ValueError(f"Normalized source {source} has recorded failures.")
        manifests[source] = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "status": manifest["status"],
            "normalization_version": manifest["normalization_schema_version"],
            "document_count": int(manifest["output_document_count"]),
            "normalized_words": int(manifest["total_normalized_words"]),
            "normalized_bytes": int(manifest["total_normalized_bytes"]),
            "normalized_files": manifest.get("normalized_files", []),
        }
        seen_paths: set[str] = set()
        for item in manifest.get("normalized_files", []):
            relative = str(item.get("relative_path") or "")
            safe = PurePosixPath(relative)
            if not relative or safe.is_absolute() or ".." in safe.parts:
                raise ValueError(
                    f"Unsafe normalized path in {source} manifest: {relative!r}"
                )
            if relative in seen_paths:
                raise ValueError(
                    f"Duplicate normalized path in {source} manifest: {relative}"
                )
            seen_paths.add(relative)
            path = source_dir / Path(*safe.parts)
            if not path.is_file() or path.stat().st_size != int(item.get("bytes", -1)):
                raise ValueError(f"Normalized shard missing or size mismatch: {path}")
            parquet = pq.ParquetFile(path)
            if parquet.schema_arrow.remove_metadata() != NORMALIZED_SCHEMA:
                raise ValueError(f"Normalized schema mismatch: {path}")
            if parquet.metadata.num_rows != int(item.get("documents", -1)):
                raise ValueError(f"Normalized row count mismatch: {path}")
            input_files.append(
                ExactInputFile(
                    source=source,
                    path=path,
                    normalized_shard=f"{source}/{relative}",
                    relative_path=relative,
                )
            )
        if (
            sum(
                int(item["documents"]) for item in manifests[source]["normalized_files"]
            )
            != manifests[source]["document_count"]
        ):
            raise ValueError(
                f"Normalized shard inventory does not reconcile for {source}."
            )
    return manifests, sorted(
        input_files, key=lambda item: (item.source, item.relative_path)
    )


def _input_manifest_identity(
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        source: {
            "manifest_sha256": item["manifest_sha256"],
            "normalization_version": item["normalization_version"],
            "document_count": item["document_count"],
            "normalized_words": item["normalized_words"],
            "normalized_files": [
                {
                    "relative_path": file["relative_path"],
                    "sha256": file["sha256"],
                    "documents": int(file["documents"]),
                }
                for file in item["normalized_files"]
            ],
        }
        for source, item in sorted(manifests.items())
    }


def _stage_output(
    *,
    normalized_root: Path,
    output_root: Path,
    input_manifests: Mapping[str, Mapping[str, Any]],
    input_files: Sequence[ExactInputFile],
    run_type: str,
    pilot_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing exact output: {output_root}"
        )
    if normalized_root == output_root or normalized_root in output_root.parents:
        raise ValueError("Exact output must not be inside immutable normalized pools.")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.partial-", dir=output_root.parent)
    )
    db_path = stage / "index.sqlite3"
    try:
        connection = _sqlite_connect(db_path)
        _create_index_db(connection)
        source_counts = _index_files(connection, input_files)
        if source_counts.get(PRESERVE_SOURCE, 0) == 0:
            raise ValueError("ParlamentoPT input is empty; expected preserve-all rows.")
        if run_type == "production":
            for source, item in input_manifests.items():
                if source_counts.get(source, 0) != int(item["document_count"]):
                    raise ValueError(
                        f"Compact-index count differs from normalized manifest for {source}."
                    )
        _set_parlamento_self_resolution(connection)
        exact_counts = _resolve_hash_groups(connection, stage)
        parlamento_counts = _parlamento_diagnostics(connection, stage)
        run_counts, data_paths = _materialize_and_count(connection, input_files, stage)
        if run_type == "production":
            for source, item in input_manifests.items():
                if run_counts["words_by_source"].get(source, 0) != int(
                    item["normalized_words"]
                ):
                    raise ValueError(
                        f"Streamed word count differs from normalized manifest for {source}."
                    )
        _write_resolution_files(connection, stage, pilot=run_type == "pilot")
        accounting_rows = _write_accounting(connection, stage, source_counts)
        pilot_report = _pilot_report(connection) if run_type == "pilot" else None
        connection.close()
        db_path.unlink(missing_ok=True)

        files = _file_inventory(stage)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "exact_dedup_version": EXACT_DEDUP_VERSION,
            "status": "COMPLETE",
            "run_type": run_type,
            "normalization_version": NORMALIZATION_VERSION,
            "normalized_root_identity": _input_manifest_identity(input_manifests),
            "input_record_count": sum(source_counts.values()),
            "input_counts_by_source": dict(sorted(source_counts.items())),
            "input_words_by_source": run_counts["words_by_source"],
            "retained_record_count": int(
                sum(
                    row["documents_after"]
                    for row in accounting_rows
                    if row["scope"] == "source_total"
                )
            ),
            "dropped_eligible_record_count": exact_counts["eligible_dropped_records"],
            "eligible_duplicate_hash_groups": exact_counts[
                "eligible_duplicate_hash_groups"
            ],
            "eligible_cross_source_duplicate_hash_groups": exact_counts[
                "eligible_cross_source_duplicate_hash_groups"
            ],
            "parlamento_diagnostic": parlamento_counts,
            "output_files": files,
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "peak_rss_bytes": _peak_rss_bytes(),
        }
        if pilot_metadata is not None:
            manifest["pilot"] = {**dict(pilot_metadata), **(pilot_report or {})}
        _atomic_json(stage / "manifest.json", manifest)
        if output_root.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing exact output: {output_root}"
            )
        os.replace(stage, output_root)
        parent_fd = os.open(output_root.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return manifest
    except BaseException:
        try:
            shutil.rmtree(stage)
        except OSError:
            pass
        raise


def _pilot_report(connection: sqlite3.Connection) -> dict[str, Any]:
    result: dict[str, Any] = {}
    roles = {
        str(role): int(count)
        for role, count in connection.execute(
            "SELECT COALESCE(selection_role, 'stratified'), COUNT(*) "
            "FROM records GROUP BY COALESCE(selection_role, 'stratified')"
        )
    }
    result["selection_role_counts"] = roles
    stratified_count = roles.get("stratified", 0)
    result["stratified_base_record_count"] = stratified_count
    result["duplicate_enrichment_record_count"] = roles.get("duplicate_enrichment", 0)
    duplicates = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT content_sha256 FROM records
            WHERE source<>? AND COALESCE(selection_role, 'stratified')='stratified'
            GROUP BY content_sha256 HAVING COUNT(*)>1
        )
        """,
        (PRESERVE_SOURCE,),
    ).fetchone()[0]
    duplicate_records = connection.execute(
        """
        SELECT COALESCE(SUM(n-1), 0) FROM (
            SELECT COUNT(*) AS n FROM records
            WHERE source<>? AND COALESCE(selection_role, 'stratified')='stratified'
            GROUP BY content_sha256 HAVING COUNT(*)>1
        )
        """,
        (PRESERVE_SOURCE,),
    ).fetchone()[0]
    cross_source = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT content_sha256 FROM records
            WHERE source<>? AND COALESCE(selection_role, 'stratified')='stratified'
            GROUP BY content_sha256 HAVING COUNT(DISTINCT source)>1
        )
        """,
        (PRESERVE_SOURCE,),
    ).fetchone()[0]
    length_counts = {
        name: int(count)
        for name, count in connection.execute(
            """
            SELECT CASE
                WHEN normalized_words < 20 THEN 'short_under_20_words'
                WHEN normalized_words >= 100000 THEN 'long_at_least_100000_words'
                ELSE 'medium'
            END, COUNT(*)
            FROM records GROUP BY 1 ORDER BY 1
            """
        )
    }
    eligible_base_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM records WHERE source<>? "
            "AND COALESCE(selection_role, 'stratified')='stratified'",
            (PRESERVE_SOURCE,),
        ).fetchone()[0]
    )
    stratified_accounting: dict[str, dict[str, int]] = {}
    accounting_query = connection.execute(
        """
        WITH eligible AS (
            SELECT source, subset, normalized_words,
                   ROW_NUMBER() OVER (
                       PARTITION BY content_sha256
                       ORDER BY sort_kind, subset_tier, source, subset, record_id
                   ) AS owner_rank
            FROM records
            WHERE source<>? AND COALESCE(selection_role, 'stratified')='stratified'
        ), base AS (
            SELECT source, subset, normalized_words, owner_rank FROM eligible
            UNION ALL
            SELECT source, subset, normalized_words, 1 AS owner_rank
            FROM records
            WHERE source=? AND COALESCE(selection_role, 'stratified')='stratified'
        )
        SELECT source, subset, COUNT(*), SUM(normalized_words),
               SUM(owner_rank=1),
               SUM(CASE WHEN owner_rank=1 THEN normalized_words ELSE 0 END)
        FROM base GROUP BY source, subset ORDER BY source, subset
        """,
        (PRESERVE_SOURCE, PRESERVE_SOURCE),
    )
    for (
        source,
        subset,
        before_docs,
        before_words,
        after_docs,
        after_words,
    ) in accounting_query:
        removed_docs = int(before_docs) - int(after_docs)
        removed_words = int(before_words) - int(after_words)
        stratified_accounting[f"{source}/{subset}"] = {
            "documents_before": int(before_docs),
            "normalized_words_before": int(before_words),
            "documents_after": int(after_docs),
            "normalized_words_after": int(after_words),
            "documents_removed": removed_docs,
            "words_removed": removed_words,
            "document_loss_fraction": (
                round(removed_docs / int(before_docs), 12) if before_docs else 0.0
            ),
            "word_loss_fraction": (
                round(removed_words / int(before_words), 12) if before_words else 0.0
            ),
        }
    result.update(
        {
            "stratified_duplicate_hash_groups": int(duplicates),
            "stratified_duplicate_records_removed": int(duplicate_records),
            "stratified_cross_source_duplicate_hash_groups": int(cross_source),
            "stratified_eligible_duplicate_fraction": (
                round(int(duplicate_records) / eligible_base_count, 8)
                if eligible_base_count
                else 0.0
            ),
            "stratified_accounting_by_source_subset": stratified_accounting,
            "sample_length_band_counts": length_counts,
            "rate_estimate_scope": "stratified rows only; duplicate-enrichment rows excluded",
            "rate_estimate_warning": "Pilot rates are diagnostic and are not authoritative production rates.",
        }
    )
    return result


def _peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS and the BSDs report bytes.
    return int(usage * 1024 if os.name == "posix" and Path("/proc").exists() else usage)


def build_exact_dedup(
    *,
    normalized_root: Path | str = DEFAULT_NORMALIZED_ROOT,
    output_root: Path | str = DEFAULT_EXACT_ROOT,
) -> dict[str, Any]:
    """Build full exact-dedup outputs without changing normalized source pools."""
    root = Path(normalized_root).resolve()
    output = Path(output_root).resolve()
    manifests, input_files = _input_catalog(root)
    return _stage_output(
        normalized_root=root,
        output_root=output,
        input_manifests=manifests,
        input_files=input_files,
        run_type="production",
    )


@dataclass(frozen=True)
class _PilotCandidate:
    record_id: str
    content_sha256: str
    source: str
    subset: str
    normalized_words: int
    normalized_shard: str
    raw_source_file: str
    raw_record_identifier: str
    source_relative_path: str
    row_group: int
    row_index: int
    selection_role: str
    sample_rank: str
    stratum: str
    length_band: str


def _rank(seed: int, text: str) -> str:
    return hashlib.sha256(f"{seed}\0{text}".encode("utf-8")).hexdigest()


def _pilot_stratum(source: str, subset: str) -> str:
    return f"gigaverbo_v2/{subset}" if source == "gigaverbo_v2" else source


def _create_pilot_db(path: Path) -> sqlite3.Connection:
    connection = _sqlite_connect(path)
    connection.executescript(
        """
        CREATE TABLE candidates (
            record_id TEXT PRIMARY KEY,
            content_sha256 TEXT NOT NULL,
            source TEXT NOT NULL,
            subset TEXT NOT NULL,
            normalized_words INTEGER NOT NULL,
            normalized_shard TEXT NOT NULL,
            raw_source_file TEXT NOT NULL,
            raw_record_identifier TEXT NOT NULL,
            source_relative_path TEXT NOT NULL,
            row_group INTEGER NOT NULL,
            row_index INTEGER NOT NULL,
            sample_rank TEXT NOT NULL,
            stratum TEXT NOT NULL,
            length_band TEXT NOT NULL
        );
        CREATE INDEX candidates_by_stratum_band_rank
            ON candidates(stratum, length_band, sample_rank, record_id);
        CREATE INDEX candidates_by_hash ON candidates(content_sha256);
        CREATE TABLE selected (record_id TEXT PRIMARY KEY, selection_role TEXT NOT NULL);
        """
    )
    return connection


def _row_groups_for_strata(
    input_files: Sequence[ExactInputFile],
    manifests: Mapping[str, Mapping[str, Any]],
    seed: int,
) -> dict[str, tuple[ExactInputFile, int]]:
    by_stratum: dict[str, list[tuple[ExactInputFile, int]]] = defaultdict(list)
    for item in input_files:
        parquet = pq.ParquetFile(item.path)
        subset = ""
        if item.source == "gigaverbo_v2":
            manifest_file = next(
                entry
                for entry in manifests[item.source]["normalized_files"]
                if entry["relative_path"] == item.relative_path
            )
            subset = str(manifest_file.get("subset") or "")
        stratum = _pilot_stratum(item.source, subset)
        for row_group in range(parquet.metadata.num_row_groups):
            by_stratum[stratum].append((item, row_group))
    selected: dict[str, tuple[ExactInputFile, int]] = {}
    for stratum, groups in sorted(by_stratum.items()):
        selected[stratum] = min(
            groups,
            key=lambda pair: _rank(
                seed,
                f"{stratum}\0{pair[0].relative_path}\0{pair[1]}",
            ),
        )
    return selected


def _scan_pilot_row_groups(
    connection: sqlite3.Connection,
    selected_groups: Mapping[str, tuple[ExactInputFile, int]],
    seed: int,
    batch_size: int = 128,
) -> None:
    insert_sql = """
        INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    for stratum, (item, row_group) in sorted(selected_groups.items()):
        parquet = pq.ParquetFile(item.path)
        columns = list(INDEX_COLUMNS) + ["text"]
        row_index = 0
        pending: list[tuple[Any, ...]] = []
        for batch in parquet.iter_batches(
            row_groups=[row_group], batch_size=batch_size, columns=columns
        ):
            for row in batch.to_pylist():
                subset = str(row.get("subset") or "")
                source = str(row["source"])
                if _pilot_stratum(source, subset) != stratum:
                    raise ValueError(
                        f"Unexpected source/subset in pilot shard {item.path}"
                    )
                words = _count_words_bounded(row.get("text") or "")
                band = (
                    "short" if words < 20 else "long" if words >= 100_000 else "medium"
                )
                record_id = stable_record_id(row)
                normalized_shard = item.normalized_shard
                pending.append(
                    (
                        record_id,
                        row["content_sha256"],
                        source,
                        subset,
                        words,
                        normalized_shard,
                        row["raw_source_file"],
                        row["raw_record_identifier"],
                        item.relative_path,
                        row_group,
                        row_index,
                        _rank(seed, record_id),
                        stratum,
                        band,
                    )
                )
                row_index += 1
                if len(pending) >= batch_size:
                    connection.executemany(insert_sql, pending)
                    pending.clear()
        if pending:
            connection.executemany(insert_sql, pending)
        connection.commit()


def _select_pilot_ids(
    connection: sqlite3.Connection,
    *,
    size: int,
    duplicate_group_limit: int = 50,
    duplicate_record_limit: int = 2_000,
) -> tuple[int, int, dict[str, int]]:
    strata = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT stratum FROM candidates ORDER BY stratum"
        )
    ]
    if not strata:
        raise ValueError("Pilot row-group selection produced no records.")
    base_quota, remainder = divmod(size, len(strata))
    selected_count = 0
    for index, stratum in enumerate(strata):
        quota = base_quota + int(index < remainder)
        if quota <= 0:
            continue
        short_quota = min(quota, max(1, quota // 5))
        long_quota = min(quota - short_quota, max(1, quota // 10))
        band_quotas = {"short": short_quota, "long": long_quota}
        band_quotas["medium"] = max(
            0, quota - band_quotas["short"] - band_quotas["long"]
        )
        chosen: set[str] = set()
        for band in ("short", "long", "medium"):
            limit = band_quotas[band]
            candidates = connection.execute(
                "SELECT record_id FROM candidates WHERE stratum=? AND length_band=? "
                "ORDER BY sample_rank, record_id LIMIT ?",
                (stratum, band, limit),
            ).fetchall()
            for (record_id,) in candidates:
                if record_id not in chosen:
                    chosen.add(record_id)
        if len(chosen) < quota:
            extra = connection.execute(
                "SELECT record_id FROM candidates WHERE stratum=? "
                "ORDER BY sample_rank, record_id LIMIT ?",
                (stratum, quota * 3),
            ).fetchall()
            for (record_id,) in extra:
                if record_id not in chosen:
                    chosen.add(record_id)
                    if len(chosen) >= quota:
                        break
        connection.executemany(
            "INSERT OR IGNORE INTO selected(record_id, selection_role) VALUES (?, 'stratified')",
            [(record_id,) for record_id in sorted(chosen)],
        )
        selected_count += len(chosen)
    selected_count = int(
        connection.execute("SELECT COUNT(*) FROM selected").fetchone()[0]
    )
    if selected_count < size:
        missing = size - selected_count
        extras = connection.execute(
            "SELECT c.record_id FROM candidates c LEFT JOIN selected s USING(record_id) "
            "WHERE s.record_id IS NULL ORDER BY c.sample_rank, c.record_id LIMIT ?",
            (missing,),
        ).fetchall()
        connection.executemany(
            "INSERT INTO selected(record_id, selection_role) VALUES (?, 'stratified')",
            extras,
        )
    connection.commit()

    # Include a bounded number of complete duplicate groups discovered in the
    # selected row groups. This is diagnostic enrichment and stays labelled.
    enriched = 0
    duplicate_groups = connection.execute(
        """
        SELECT content_sha256, COUNT(*) AS n,
               COUNT(DISTINCT CASE WHEN source<>? THEN source END) AS source_n
        FROM candidates GROUP BY content_sha256 HAVING COUNT(*)>1
        ORDER BY CASE WHEN COUNT(DISTINCT CASE WHEN source<>? THEN source END)>1 THEN 0 ELSE 1 END,
                 content_sha256
        """,
        (PRESERVE_SOURCE, PRESERVE_SOURCE),
    )
    groups_added = 0
    for content_hash, group_size, _source_n in duplicate_groups:
        if (
            groups_added >= duplicate_group_limit
            or enriched + int(group_size) > duplicate_record_limit
        ):
            continue
        members = connection.execute(
            "SELECT record_id FROM candidates WHERE content_sha256=? ORDER BY record_id",
            (content_hash,),
        ).fetchall()
        connection.executemany(
            "INSERT OR IGNORE INTO selected(record_id, selection_role) "
            "VALUES (?, 'duplicate_enrichment')",
            members,
        )
        groups_added += 1
    connection.commit()
    enriched = int(
        connection.execute(
            "SELECT COUNT(*) FROM selected WHERE selection_role='duplicate_enrichment'"
        ).fetchone()[0]
    )
    selected_count = int(
        connection.execute("SELECT COUNT(*) FROM selected").fetchone()[0]
    )
    band_counts = {
        band: int(count)
        for band, count in connection.execute(
            """
            SELECT c.length_band, COUNT(*) FROM selected s JOIN candidates c USING(record_id)
            GROUP BY c.length_band ORDER BY c.length_band
            """
        )
    }
    return selected_count - enriched, enriched, band_counts


def _write_pilot_inputs(
    connection: sqlite3.Connection,
    selected_groups: Mapping[str, tuple[ExactInputFile, int]],
    stage: Path,
) -> list[ExactInputFile]:
    by_file_group: dict[tuple[str, str, int], tuple[ExactInputFile, set[int]]] = {}
    for row in connection.execute(
        """
        SELECT c.source_relative_path, c.row_group, c.row_index, c.source,
               c.normalized_shard, c.stratum
        FROM candidates c JOIN selected s USING(record_id)
        ORDER BY c.source, c.source_relative_path, c.row_group, c.row_index
        """
    ):
        relative, row_group, row_index, source, normalized_shard, stratum = row
        selected_file, expected_group = selected_groups[stratum]
        if expected_group != int(row_group) or selected_file.relative_path != relative:
            raise ValueError("Pilot candidate escaped its selected row group.")
        key = (source, relative, int(row_group))
        if key not in by_file_group:
            by_file_group[key] = (selected_file, set())
        by_file_group[key][1].add(int(row_index))

    output_inputs: list[ExactInputFile] = []
    selected_role = {
        record_id: role
        for record_id, role in connection.execute(
            "SELECT record_id, selection_role FROM selected"
        )
    }
    for (_source, relative, row_group), (item, wanted_rows) in sorted(
        by_file_group.items(), key=lambda entry: (entry[0][0], entry[0][1], entry[0][2])
    ):
        destination = (
            stage / "_pilot_input" / item.source / Path(*PurePosixPath(relative).parts)
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f"{destination.name}.partial")
        writer: pq.ParquetWriter | None = None
        row_index = 0
        selected_records: set[str] = set()
        source_parquet = pq.ParquetFile(item.path)
        columns = MATERIALIZE_COLUMNS
        for batch in source_parquet.iter_batches(
            row_groups=[row_group], batch_size=64, columns=columns
        ):
            picked: list[dict[str, Any]] = []
            for row in batch.to_pylist():
                if row_index in wanted_rows:
                    row_id = stable_record_id(row)
                    role = selected_role[row_id]
                    selected_records.add(row_id)
                    enriched_row = dict(row)
                    enriched_row["pilot_selection_role"] = role
                    picked.append(enriched_row)
                row_index += 1
            if picked:
                if writer is None:
                    sample_schema = NORMALIZED_SCHEMA.append(
                        pa.field("pilot_selection_role", pa.string(), nullable=False)
                    )
                    writer = pq.ParquetWriter(
                        temporary,
                        sample_schema,
                        compression="zstd",
                        compression_level=6,
                        use_dictionary=True,
                        write_statistics=True,
                        version="2.6",
                    )
                table = pa.Table.from_pylist(picked, schema=sample_schema)
                writer.write_table(table, row_group_size=64)
        if writer is None:
            raise ValueError(f"Selected pilot rows are missing from {item.path}.")
        writer.close()
        os.replace(temporary, destination)
        if len(selected_records) != len(wanted_rows):
            raise ValueError(f"Pilot sample row count mismatch in {item.path}.")
        output_inputs.append(
            ExactInputFile(
                source=item.source,
                path=destination,
                normalized_shard=item.normalized_shard,
                relative_path=relative,
            )
        )
    return output_inputs


def _build_pilot_sample(
    *,
    root: Path,
    manifests: Mapping[str, Mapping[str, Any]],
    input_files: Sequence[ExactInputFile],
    stage: Path,
    size: int,
    seed: int,
) -> tuple[list[ExactInputFile], dict[str, Any]]:
    strata_expected = {source for source in SOURCE_CONFIG if source != "gigaverbo_v2"}
    strata_expected.update(f"gigaverbo_v2/{subset}" for subset in GV_SUBSET_TIERS)
    actual_groups = _row_groups_for_strata(input_files, manifests, seed)
    missing = strata_expected - set(actual_groups)
    if missing:
        raise ValueError(
            f"Pilot cannot cover required source/subset strata: {sorted(missing)}"
        )
    scratch_path = stage / "pilot-selection.sqlite3"
    connection = _create_pilot_db(scratch_path)
    _scan_pilot_row_groups(connection, actual_groups, seed)
    base_count, enriched_count, band_counts = _select_pilot_ids(connection, size=size)
    selected_count = base_count + enriched_count
    sample_inputs = _write_pilot_inputs(connection, actual_groups, stage)
    selected_row_groups = {
        stratum: {
            "normalized_shard": item.normalized_shard,
            "row_group": row_group,
        }
        for stratum, (item, row_group) in sorted(actual_groups.items())
    }
    connection.close()
    scratch_path.unlink(missing_ok=True)
    metadata = {
        "seed": seed,
        "requested_stratified_sample_size": size,
        "stratified_base_record_count": base_count,
        "duplicate_enrichment_record_count": enriched_count,
        "selected_record_count": selected_count,
        "required_strata": sorted(strata_expected),
        "selected_row_groups": selected_row_groups,
        "selected_length_band_counts": band_counts,
        "duplicate_enrichment_policy": {
            "maximum_groups": 50,
            "maximum_records": 2_000,
            "selection": "complete duplicate groups found in sampled row groups, cross-source groups first",
        },
        "sample_selection": "deterministic row-group selection plus bounded hash-ranked reservoirs by source/subset and length band",
        "near_dedup_input": "data/ retained pilot sample Parquet files",
        "rate_estimate_warning": "Duplicate-enriched rows are excluded from representative rate summaries; all pilot fractions remain non-authoritative.",
    }
    return sample_inputs, metadata


def run_exact_dedup_pilot(
    *,
    normalized_root: Path | str = DEFAULT_NORMALIZED_ROOT,
    output_root: Path | str,
    size: int = DEFAULT_PILOT_SIZE,
    seed: int = DEFAULT_PILOT_SEED,
) -> dict[str, Any]:
    """Select and exact-deduplicate a small deterministic local pilot sample."""
    if size < len(SOURCE_CONFIG) - 1 + len(GV_SUBSET_TIERS):
        raise ValueError(
            "Pilot size must allow at least one stratified row per required stratum."
        )
    if size > 200_000:
        raise ValueError(
            "Pilot size is capped at 200,000 records for bounded local inspection."
        )
    root = Path(normalized_root).resolve()
    output = Path(output_root).resolve()
    manifests, full_input_files = _input_catalog(root)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing exact pilot output: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    selection_stage = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.select-", dir=output.parent)
    )
    try:
        selection_started = time.perf_counter()
        pilot_files, pilot_metadata = _build_pilot_sample(
            root=root,
            manifests=manifests,
            input_files=full_input_files,
            stage=selection_stage,
            size=size,
            seed=seed,
        )
        pilot_metadata["selection_runtime_seconds"] = round(
            time.perf_counter() - selection_started, 3
        )
        result = _stage_output(
            normalized_root=root,
            output_root=output,
            input_manifests=manifests,
            input_files=pilot_files,
            run_type="pilot",
            pilot_metadata=pilot_metadata,
        )
        shutil.rmtree(selection_stage, ignore_errors=True)
        return result
    finally:
        shutil.rmtree(selection_stage, ignore_errors=True)


def _output_file_schema(path: Path) -> pa.Schema | None:
    if path.name == "record_resolution.parquet":
        return RESOLUTION_SCHEMA
    if path.name == "duplicate_edges.parquet":
        return EDGE_SCHEMA
    if path.name == "cluster_index.parquet":
        return CLUSTER_SCHEMA
    if path.name == "parlamento_duplicate_hashes.parquet":
        return PARLAMENTO_DIAGNOSTIC_SCHEMA
    if path.name == "pilot_input_index.parquet":
        return PILOT_INDEX_SCHEMA
    if path.parts and path.parts[0] == "data":
        return NORMALIZED_SCHEMA
    return None


def _resolution_row_to_tuple(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[name] for name in RESOLUTION_SCHEMA.names)


def _verify_expected_inputs(
    normalized_root: Path,
    output_root: Path,
    manifest: Mapping[str, Any],
    connection: sqlite3.Connection,
) -> None:
    run_type = manifest.get("run_type")
    seen = 0
    lookup = connection.cursor()
    if run_type == "pilot":
        pilot_index_path = output_root / "pilot_input_index.parquet"
        if not pilot_index_path.is_file():
            raise ValueError("Pilot output lacks pilot_input_index.parquet")
        parquet = pq.ParquetFile(pilot_index_path)
        for batch in parquet.iter_batches(batch_size=4_096):
            for row in batch.to_pylist():
                record_id = row["record_id"]
                if stable_record_id(row) != record_id:
                    raise ValueError(
                        f"Pilot input record identity mismatch for {record_id}"
                    )
                found = lookup.execute(
                    "SELECT content_sha256, source, subset, selection_role "
                    "FROM resolution WHERE record_id=?",
                    (record_id,),
                ).fetchone()
                if found is None or found != (
                    row["content_sha256"],
                    row["source"],
                    row["subset"],
                    row["selection_role"],
                ):
                    raise ValueError(
                        f"Pilot input index mapping mismatch for {record_id}"
                    )
                connection.execute("INSERT INTO input_seen VALUES (?)", (record_id,))
                seen += 1
    else:
        _manifests, input_files = _input_catalog(normalized_root)
        for item in input_files:
            parquet = pq.ParquetFile(item.path)
            for batch in parquet.iter_batches(batch_size=4_096, columns=INDEX_COLUMNS):
                for row in batch.to_pylist():
                    record_id = stable_record_id(row)
                    found = lookup.execute(
                        "SELECT content_sha256, source, subset FROM resolution WHERE record_id=?",
                        (record_id,),
                    ).fetchone()
                    source = row["source"]
                    subset = str(row.get("subset") or "")
                    if found != (row["content_sha256"], source, subset):
                        raise ValueError(
                            f"Input-to-resolution mapping mismatch for {record_id}"
                        )
                    connection.execute(
                        "INSERT INTO input_seen VALUES (?)", (record_id,)
                    )
                    seen += 1
    resolution_count = int(
        connection.execute("SELECT COUNT(*) FROM resolution").fetchone()[0]
    )
    if seen != resolution_count:
        raise ValueError(
            f"Input/resolution counts differ: {seen} != {resolution_count}"
        )


def verify_exact_dedup(
    *,
    normalized_root: Path | str = DEFAULT_NORMALIZED_ROOT,
    output_root: Path | str = DEFAULT_EXACT_ROOT,
) -> list[str]:
    """Verify output hashes, source identities, mappings, accounting and ownership."""
    root = Path(output_root).resolve()
    normalized = Path(normalized_root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return [f"Exact-dedup manifest is missing: {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("status") != "COMPLETE":
        errors.append(
            f"Exact output status is {manifest.get('status')!r}, expected COMPLETE."
        )
    if manifest.get("exact_dedup_version") != EXACT_DEDUP_VERSION:
        errors.append("Exact-dedup contract version mismatch.")
    try:
        current_manifests, _ = _input_catalog(normalized)
        current_identity = _input_manifest_identity(current_manifests)
        if current_identity != manifest.get("normalized_root_identity"):
            errors.append("Normalized input manifest identities changed.")
    except Exception as exc:
        errors.append(f"Cannot validate normalized input manifests: {exc}")

    output_inventory = manifest.get("output_files", {})
    for relative, item in sorted(output_inventory.items()):
        safe = PurePosixPath(relative)
        path = (root / Path(*safe.parts)).resolve()
        if safe.is_absolute() or ".." in safe.parts or root not in path.parents:
            errors.append(f"Unsafe exact output file path: {relative}")
            continue
        if not path.is_file():
            errors.append(f"Exact output file is missing: {relative}")
            continue
        if path.stat().st_size != int(item["bytes"]):
            errors.append(f"Exact output byte count mismatch: {relative}")
        if compute_file_sha256(path) != item["sha256"]:
            errors.append(f"Exact output SHA-256 mismatch: {relative}")
        if path.suffix == ".parquet":
            try:
                parquet = pq.ParquetFile(path)
                expected_schema = _output_file_schema(path)
                if (
                    expected_schema is not None
                    and parquet.schema_arrow.remove_metadata() != expected_schema
                ):
                    errors.append(f"Exact output schema mismatch: {relative}")
                if parquet.metadata.num_rows != int(item["rows"]):
                    errors.append(f"Exact output row count mismatch: {relative}")
            except Exception as exc:
                errors.append(f"Cannot read exact output Parquet {relative}: {exc}")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_files != set(output_inventory):
        errors.append(
            "Exact output inventory mismatch: "
            f"unlisted={sorted(actual_files - set(output_inventory))[:5]}, "
            f"missing={sorted(set(output_inventory) - actual_files)[:5]}"
        )
    if errors:
        return errors

    scratch = Path(tempfile.mkdtemp(prefix="cambacica-exact-verify-"))
    connection = _sqlite_connect(scratch / "verify.sqlite3")
    try:
        connection.executescript(
            """
            CREATE TABLE resolution (
                record_id TEXT PRIMARY KEY,
                cluster_id TEXT,
                content_sha256 TEXT NOT NULL,
                source TEXT NOT NULL,
                subset TEXT NOT NULL,
                normalized_words INTEGER NOT NULL,
                normalized_shard TEXT NOT NULL,
                raw_source_file TEXT NOT NULL,
                raw_record_identifier TEXT NOT NULL,
                representative_record_id TEXT NOT NULL,
                disposition TEXT NOT NULL,
                ownership_rule TEXT NOT NULL,
                selection_role TEXT
            );
            CREATE INDEX resolution_hash ON resolution(content_sha256);
            CREATE TABLE edges (
                dropped_record_id TEXT PRIMARY KEY,
                cluster_id TEXT NOT NULL,
                retained_record_id TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                dropped_source TEXT NOT NULL,
                dropped_subset TEXT NOT NULL,
                retained_source TEXT NOT NULL,
                retained_subset TEXT NOT NULL,
                ownership_rule TEXT NOT NULL
            );
            CREATE TABLE materialized (record_id TEXT PRIMARY KEY);
            CREATE TABLE input_seen (record_id TEXT PRIMARY KEY);
            """
        )
        resolution_path = root / "record_resolution.parquet"
        for batch in pq.ParquetFile(resolution_path).iter_batches(batch_size=4_096):
            records = batch.to_pylist()
            connection.executemany(
                "INSERT INTO resolution VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [_resolution_row_to_tuple(row) for row in records],
            )
        edge_path = root / "duplicate_edges.parquet"
        for batch in pq.ParquetFile(edge_path).iter_batches(batch_size=4_096):
            rows = batch.to_pylist()
            connection.executemany(
                "INSERT INTO edges VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (
                        row["dropped_record_id"],
                        row["cluster_id"],
                        row["retained_record_id"],
                        row["content_sha256"],
                        row["dropped_source"],
                        row["dropped_subset"],
                        row["retained_source"],
                        row["retained_subset"],
                        row["ownership_rule"],
                    )
                    for row in rows
                ],
            )
        connection.commit()

        _verify_expected_inputs(normalized, root, manifest, connection)
        expected_parliament_count = int(
            manifest.get("input_counts_by_source", {}).get(PRESERVE_SOURCE, -1)
        )
        actual_parliament_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM resolution WHERE source=?", (PRESERVE_SOURCE,)
            ).fetchone()[0]
        )
        if actual_parliament_count != expected_parliament_count:
            errors.append(
                "ParlamentoPT retained count does not match normalized input count: "
                f"{actual_parliament_count} != {expected_parliament_count}."
            )
        _verify_manifest_counts(manifest, connection, errors)
        unresolved = connection.execute(
            "SELECT COUNT(*) FROM resolution r LEFT JOIN resolution p "
            "ON p.record_id=r.representative_record_id WHERE p.record_id IS NULL"
        ).fetchone()[0]
        if unresolved:
            errors.append(f"{unresolved} records point to missing representatives.")
        invalid_representatives = connection.execute(
            "SELECT COUNT(*) FROM resolution r JOIN resolution p "
            "ON p.record_id=r.representative_record_id "
            "WHERE (r.disposition='dropped' AND p.disposition<>'retained') "
            "OR (r.disposition IN ('retained','preserved_diagnostic') AND r.record_id<>p.record_id)"
        ).fetchone()[0]
        if invalid_representatives:
            errors.append(
                f"{invalid_representatives} resolution rows have invalid representatives."
            )
        parliament_dropped = connection.execute(
            "SELECT COUNT(*) FROM resolution WHERE source=? AND disposition='dropped'",
            (PRESERVE_SOURCE,),
        ).fetchone()[0]
        if parliament_dropped:
            errors.append("ParlamentoPT contains dropped resolution rows.")
        parliament_self = connection.execute(
            "SELECT COUNT(*) FROM resolution WHERE source=? "
            "AND (representative_record_id<>record_id OR disposition<>'preserved_diagnostic')",
            (PRESERVE_SOURCE,),
        ).fetchone()[0]
        if parliament_self:
            errors.append("ParlamentoPT rows do not all map to themselves.")
        retained_duplicates = connection.execute(
            "SELECT COUNT(*) FROM (SELECT content_sha256 FROM resolution "
            "WHERE source<>? AND disposition='retained' GROUP BY content_sha256 HAVING COUNT(*)>1)",
            (PRESERVE_SOURCE,),
        ).fetchone()[0]
        if retained_duplicates:
            errors.append(
                "Eligible retained records still contain exact duplicate hashes."
            )
        dropped_edge_mismatch = connection.execute(
            "SELECT COUNT(*) FROM resolution r LEFT JOIN edges e "
            "ON e.dropped_record_id=r.record_id "
            "WHERE (r.disposition='dropped' AND (e.dropped_record_id IS NULL "
            "OR e.cluster_id<>r.cluster_id "
            "OR e.retained_record_id<>r.representative_record_id "
            "OR e.content_sha256<>r.content_sha256 "
            "OR e.dropped_source<>r.source OR e.dropped_subset<>r.subset "
            "OR e.ownership_rule<>r.ownership_rule "
            "OR e.retained_source<>(SELECT p.source FROM resolution p WHERE p.record_id=r.representative_record_id) "
            "OR e.retained_subset<>(SELECT p.subset FROM resolution p WHERE p.record_id=r.representative_record_id))) "
            "OR (r.disposition<>'dropped' AND e.dropped_record_id IS NOT NULL)"
        ).fetchone()[0]
        if dropped_edge_mismatch:
            errors.append(
                "Dropped-record edge mappings are missing, repeated or inconsistent."
            )
        dropped_count = int(
            connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        )
        dropped_resolution_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM resolution WHERE disposition='dropped'"
            ).fetchone()[0]
        )
        if dropped_count != dropped_resolution_count:
            errors.append("Dropped-edge count does not match dropped resolution count.")

        # Every non-dropped row must appear once in materialized data.
        for data_path in (
            sorted((root / "data").rglob("*.parquet"))
            if (root / "data").exists()
            else []
        ):
            parquet = pq.ParquetFile(data_path)
            for batch in parquet.iter_batches(batch_size=4_096, columns=INDEX_COLUMNS):
                for row in batch.to_pylist():
                    connection.execute(
                        "INSERT INTO materialized(record_id) VALUES (?)",
                        (stable_record_id(row),),
                    )
        connection.commit()
        missing_materialized = connection.execute(
            "SELECT COUNT(*) FROM resolution r LEFT JOIN materialized m USING(record_id) "
            "WHERE r.disposition<>'dropped' AND m.record_id IS NULL"
        ).fetchone()[0]
        extra_materialized = connection.execute(
            "SELECT COUNT(*) FROM materialized m LEFT JOIN resolution r USING(record_id) "
            "WHERE r.record_id IS NULL OR r.disposition='dropped'"
        ).fetchone()[0]
        if missing_materialized or extra_materialized:
            errors.append(
                f"Materialized retained mapping mismatch: missing={missing_materialized}, extra={extra_materialized}."
            )
        if int(
            connection.execute("SELECT COUNT(*) FROM materialized").fetchone()[0]
        ) != int(manifest["retained_record_count"]):
            errors.append(
                "Retained materialized count differs from the exact manifest."
            )

        _verify_accounting(root / "accounting_by_source_subset.csv", connection, errors)
        _verify_deterministic_ownership(connection, errors)
        _verify_cluster_index(root / "cluster_index.parquet", connection, errors)
        _verify_parlamento_diagnostics(
            root / "parlamento_duplicate_hashes.parquet", connection, errors
        )
    except Exception as exc:
        errors.append(
            f"Exact output semantic verification failed: {type(exc).__name__}: {exc}"
        )
    finally:
        connection.close()
        shutil.rmtree(scratch, ignore_errors=True)
    return errors


def _verify_deterministic_ownership(
    connection: sqlite3.Connection, errors: list[str]
) -> None:
    cursor = connection.execute(
        """
        SELECT content_sha256, record_id, source, subset,
               representative_record_id, disposition, ownership_rule, cluster_id
        FROM resolution WHERE source<>?
        ORDER BY content_sha256,
            CASE
                WHEN source IN ('carolina', 'gutenberg_pt', 'wikipedia_pt') THEN 0
                WHEN source='gigaverbo_v2' THEN 1
                ELSE 2
            END,
            CASE WHEN source='gigaverbo_v2' THEN CASE subset
                WHEN 'finepdfs_por_Latn' THEN 1
                WHEN 'crawlPT_dedup' THEN 2
                WHEN 'quati' THEN 2
                WHEN 'blogset' THEN 2
                WHEN 'fineweb_2_pt' THEN 3
                WHEN 'mc4_pt' THEN 4
                WHEN 'hplt2_pt' THEN 4
                WHEN 'hplt1_pt' THEN 4
                WHEN 'common_crawl' THEN 4
                WHEN 'oscar' THEN 4
                WHEN 'culturax' THEN 4
                ELSE 5
            END ELSE 0 END,
            source, subset, record_id
        """,
        (PRESERVE_SOURCE,),
    )
    content_hash: str | None = None
    winner_id: str | None = None
    owner_source: str | None = None
    expected_cluster: str | None = None
    for row in cursor:
        if row[0] != content_hash:
            content_hash = row[0]
            winner_id = row[1]
            owner_source = row[2]
            expected_cluster = exact_cluster_id(content_hash)
        expected_rule = _ownership_rule(owner_source or row[2], 1)
        expected_disposition = "retained" if row[1] == winner_id else "dropped"
        if row[4] != winner_id:
            errors.append(f"Nondeterministic representative for exact hash {row[0]}.")
            return
        if row[5] != expected_disposition:
            errors.append(f"Incorrect disposition for exact hash {row[0]}.")
            return
        if row[6] != expected_rule:
            errors.append(f"Ownership rule mismatch for exact hash {row[0]}.")
            return
        if row[7] != expected_cluster:
            errors.append(f"Cluster ID mismatch for exact hash {row[0]}.")
            return


def _verify_accounting(
    path: Path, connection: sqlite3.Connection, errors: list[str]
) -> None:
    expected: dict[tuple[str, str, str], tuple[int, int, int, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["source"], row["subset"], row["scope"])
            values = (
                int(row["documents_before"]),
                int(row["normalized_words_before"]),
                int(row["documents_after"]),
                int(row["normalized_words_after"]),
            )
            if key in expected:
                errors.append(f"Duplicate accounting row: {key}")
            expected[key] = values
            if values[0] != values[2] + int(row["documents_removed"]):
                errors.append(f"Document accounting does not reconcile for {key}.")
            if values[1] != values[3] + int(row["words_removed"]):
                errors.append(f"Word accounting does not reconcile for {key}.")
            if int(row["documents_removed"]) < 0 or int(row["words_removed"]) < 0:
                errors.append(f"Negative removal total in accounting row {key}.")
            expected_document_fraction = (
                int(row["documents_removed"]) / values[0] if values[0] else 0.0
            )
            expected_word_fraction = (
                int(row["words_removed"]) / values[1] if values[1] else 0.0
            )
            if abs(float(row["loss_fraction"]) - expected_document_fraction) > 1e-12:
                errors.append(f"Document loss fraction does not reconcile for {key}.")
            if (
                abs(float(row["document_loss_fraction"]) - expected_document_fraction)
                > 1e-12
            ):
                errors.append(
                    f"Explicit document loss fraction does not reconcile for {key}."
                )
            if abs(float(row["word_loss_fraction"]) - expected_word_fraction) > 1e-12:
                errors.append(f"Word loss fraction does not reconcile for {key}.")

    actual: dict[tuple[str, str, str], tuple[int, int, int, int]] = {}
    subsets = connection.execute(
        "SELECT source, subset, COUNT(*), SUM(normalized_words), "
        "SUM(disposition<>'dropped'), SUM(CASE WHEN disposition<>'dropped' THEN normalized_words ELSE 0 END) "
        "FROM resolution GROUP BY source, subset"
    )
    by_source: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    for source, subset, before_n, before_w, after_n, after_w in subsets:
        values = (int(before_n), int(before_w), int(after_n), int(after_w))
        actual[(source, subset, "source_subset")] = values
        by_source[source].append(values)
    for source, rows in by_source.items():
        actual[(source, "", "source_total")] = tuple(
            sum(row[i] for row in rows) for i in range(4)
        )
    if actual != expected:
        errors.append("Source/subset accounting totals do not match record_resolution.")


def _verify_manifest_counts(
    manifest: Mapping[str, Any], connection: sqlite3.Connection, errors: list[str]
) -> None:
    input_counts = {
        source: int(count)
        for source, count in connection.execute(
            "SELECT source, COUNT(*) FROM resolution GROUP BY source"
        )
    }
    input_words = {
        source: int(words)
        for source, words in connection.execute(
            "SELECT source, SUM(normalized_words) FROM resolution GROUP BY source"
        )
    }
    if input_counts != manifest.get("input_counts_by_source"):
        errors.append(
            "Exact manifest input document counts differ from record_resolution."
        )
    if input_words != manifest.get("input_words_by_source"):
        errors.append("Exact manifest input word counts differ from record_resolution.")

    retained_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM resolution WHERE disposition<>'dropped'"
        ).fetchone()[0]
    )
    dropped_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM resolution WHERE disposition='dropped'"
        ).fetchone()[0]
    )
    duplicate_groups = int(
        connection.execute(
            "SELECT COUNT(*) FROM (SELECT content_sha256 FROM resolution "
            "WHERE source<>? GROUP BY content_sha256 HAVING COUNT(*)>1)",
            (PRESERVE_SOURCE,),
        ).fetchone()[0]
    )
    cross_source_groups = int(
        connection.execute(
            "SELECT COUNT(*) FROM (SELECT content_sha256 FROM resolution "
            "WHERE source<>? GROUP BY content_sha256 HAVING COUNT(DISTINCT source)>1)",
            (PRESERVE_SOURCE,),
        ).fetchone()[0]
    )
    for field, actual in (
        ("retained_record_count", retained_count),
        ("dropped_eligible_record_count", dropped_count),
        ("eligible_duplicate_hash_groups", duplicate_groups),
        ("eligible_cross_source_duplicate_hash_groups", cross_source_groups),
    ):
        if int(manifest.get(field, -1)) != actual:
            errors.append(f"Exact manifest {field} differs from record_resolution.")

    repeated_parliament = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(n), 0), COALESCE(SUM(n-1), 0) FROM "
        "(SELECT COUNT(*) AS n FROM resolution WHERE source=? "
        "GROUP BY content_sha256 HAVING COUNT(*)>1)",
        (PRESERVE_SOURCE,),
    ).fetchone()
    expected_parliament = {
        "duplicate_hash_groups": int(repeated_parliament[0]),
        "records_in_repeated_hash_groups": int(repeated_parliament[1]),
        "surplus_repeated_records": int(repeated_parliament[2]),
    }
    if expected_parliament != manifest.get("parlamento_diagnostic"):
        errors.append("ParlamentoPT manifest diagnostics differ from resolution rows.")


def _verify_cluster_index(
    path: Path, connection: sqlite3.Connection, errors: list[str]
) -> None:
    groups = connection.execute(
        "SELECT content_sha256, source, subset, COUNT(*), "
        "SUM(disposition='dropped'), MIN(representative_record_id), "
        "MIN(ownership_rule) FROM resolution WHERE source<>? "
        "GROUP BY content_sha256, source, subset ORDER BY content_sha256, source, subset",
        (PRESERVE_SOURCE,),
    )
    parquet_rows = (
        row
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4_096)
        for row in batch.to_pylist()
    )
    actual = next(parquet_rows, None)
    active_hash: str | None = None
    record_count = duplicate_count = 0
    representative_id: str | None = None
    ownership_rule: str | None = None
    source_counts: Counter[str] = Counter()
    subset_counts: Counter[str] = Counter()
    seen = 0

    def finish_expected_group() -> bool:
        nonlocal actual, seen
        if active_hash is None:
            return True
        if actual is None:
            errors.append(f"Cluster index is missing exact hash {active_hash}.")
            return False
        expected_sources = dict(sorted(source_counts.items()))
        expected_subsets = dict(sorted(subset_counts.items()))
        expected = {
            "cluster_id": exact_cluster_id(active_hash),
            "content_sha256": active_hash,
            "retained_record_id": representative_id,
            "eligible_record_count": record_count,
            "duplicate_record_count": duplicate_count,
            "source_count": len(source_counts),
            "source_counts_json": json.dumps(
                expected_sources,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "subset_counts_json": json.dumps(
                expected_subsets,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "ownership_rule": ownership_rule,
        }
        if actual != expected:
            errors.append(
                f"Cluster index contents differ for exact hash {active_hash}."
            )
            return False
        seen += 1
        actual = next(parquet_rows, None)
        return True

    for content_hash, source, subset, count, drops, rep_id, rule in groups:
        if content_hash != active_hash:
            if not finish_expected_group():
                return
            active_hash = content_hash
            record_count = 0
            duplicate_count = 0
            representative_id = rep_id
            ownership_rule = rule
            source_counts = Counter()
            subset_counts = Counter()
        record_count += int(count)
        duplicate_count += int(drops)
        source_counts[source] += int(count)
        subset_counts[f"{source}/{subset}"] += int(count)
        if rep_id != representative_id or rule != ownership_rule:
            errors.append(
                f"Inconsistent representative metadata for exact hash {content_hash}."
            )
            return
    if not finish_expected_group():
        return
    if actual is not None:
        errors.append(
            f"Cluster index contains unexpected exact hash {actual['content_sha256']}."
        )
    expected_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM (SELECT content_sha256 FROM resolution WHERE source<>? GROUP BY content_sha256)",
            (PRESERVE_SOURCE,),
        ).fetchone()[0]
    )
    if seen != expected_count:
        errors.append(f"Cluster index count mismatch: {seen} != {expected_count}.")


def _verify_parlamento_diagnostics(
    path: Path, connection: sqlite3.Connection, errors: list[str]
) -> None:
    actual_rows = (
        row
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4_096)
        for row in batch.to_pylist()
    )
    actual = next(actual_rows, None)
    expected_rows = connection.execute(
        "SELECT content_sha256, COUNT(*) FROM resolution WHERE source=? "
        "GROUP BY content_sha256 HAVING COUNT(*)>1 ORDER BY content_sha256",
        (PRESERVE_SOURCE,),
    )
    for content_hash, count in expected_rows:
        expected = {
            "content_sha256": content_hash,
            "record_count": int(count),
            "surplus_record_count": int(count) - 1,
        }
        if actual != expected:
            errors.append(
                "Parlamento duplicate-hash diagnostics do not match resolution rows."
            )
            return
        actual = next(actual_rows, None)
    if actual is not None:
        errors.append(
            "Parlamento duplicate-hash diagnostics contain an unexpected hash."
        )
