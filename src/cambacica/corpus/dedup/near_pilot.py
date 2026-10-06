"""Deterministic, non-destructive near-deduplication pilot for Gate C1.

This module samples only the exact-deduplicated corpus.  It never writes a
retained corpus or changes the input tree.  The pilot deliberately keeps
ParlamentoPT rows and treats similarity there as diagnostic evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import csv
import hashlib
import heapq
import itertools
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import sqlite3
import struct
import tempfile
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import xxhash

from cambacica.corpus.manifest import compute_file_sha256


NEAR_PILOT_VERSION = "0.1.0"
DEFAULT_NEAR_PILOT_ROOT = Path("/mnt/data/cambacica-base-180m/dedup-pilots/near-v1")
DEFAULT_EXACT_DATA_ROOT = Path("/mnt/data/cambacica-base-180m/deduplicated/exact")
DEFAULT_NEAR_PILOT_SEED = 20261006
THRESHOLDS = (0.80, 0.85, 0.90, 0.92, 0.95)
REVIEW_BANDS = (
    (0.75, 0.80),
    (0.80, 0.85),
    (0.85, 0.90),
    (0.90, 0.92),
    (0.92, 0.95),
    (0.95, 1.000001),
)
LENGTH_QUOTAS = {"short": 48, "medium": 80, "long": 48, "giant": 12}
EXPECTED_SOURCES = {
    "carolina",
    "wikipedia_pt",
    "gutenberg_pt",
    "parlamento_pt",
    "gigaverbo_v2",
}
EXPECTED_GIGAVERBO_SUBSETS = {
    "finepdfs_por_Latn",
    "crawlPT_dedup",
    "quati",
    "blogset",
    "fineweb_2_pt",
    "mc4_pt",
    "hplt2_pt",
    "hplt1_pt",
    "common_crawl",
    "oscar",
    "culturax",
}
BUCKET_MEMBER_CAP = 256
WORD_COUNT_CHUNK_CHARS = 1 << 20
SIGNATURE_BATCH_SHINGLES = 4096
MAX_EXACT_REVIEW_SHINGLES = 500_000

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class MinHashConfig:
    """One stable word-shingle MinHash representation."""

    name: str
    ngram_size: int
    num_permutations: int
    seed: int = 20261006


@dataclass(frozen=True)
class LSHConfig:
    """Banding arrangement applied to a MinHash signature prefix."""

    name: str
    representation: str
    num_permutations: int
    bands: int
    rows_per_band: int


SIGNATURE_CONFIGS = (
    MinHashConfig("word3_128", 3, 128),
    MinHashConfig("word5_256", 5, 256),
    MinHashConfig("word7_128", 7, 128),
)
LSH_CONFIGS = (
    LSHConfig("word5_64_16x4", "word5_256", 64, 16, 4),
    LSHConfig("word5_64_8x8", "word5_256", 64, 8, 8),
    LSHConfig("word5_128_32x4", "word5_256", 128, 32, 4),
    LSHConfig("word5_128_16x8", "word5_256", 128, 16, 8),
    LSHConfig("word5_128_8x16", "word5_256", 128, 8, 16),
    LSHConfig("word5_256_32x8", "word5_256", 256, 32, 8),
    LSHConfig("word5_256_16x16", "word5_256", 256, 16, 16),
    LSHConfig("word5_256_8x32", "word5_256", 256, 8, 32),
    LSHConfig("word3_128_32x4", "word3_128", 128, 32, 4),
    LSHConfig("word7_128_32x4", "word7_128", 128, 32, 4),
)


def stable_hash64(value: str | bytes, seed: int = 0) -> int:
    """Return a stable unsigned xxHash64 value."""
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return xxhash.xxh64(raw, seed=seed & 0xFFFFFFFF).intdigest()


def occurrence_id(relative_path: str, row_ordinal: int) -> str:
    """Name an occurrence in the immutable exact-output layout."""
    payload = f"near-pilot-occurrence-v1\0{relative_path}\0{row_ordinal}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def count_normalized_words_bounded(text: str) -> int:
    """Count Unicode whitespace-delimited words with bounded temporary memory."""
    count = 0
    previous_chunk_ended_in_word = False
    for offset in range(0, len(text), WORD_COUNT_CHUNK_CHARS):
        chunk = text[offset : offset + WORD_COUNT_CHUNK_CHARS]
        if not chunk:
            continue
        count += len(chunk.split())
        if previous_chunk_ended_in_word and not chunk[0].isspace():
            count -= 1
        previous_chunk_ended_in_word = not chunk[-1].isspace()
    return count


def length_stratum(word_count: int) -> str:
    """Map the required pilot length ranges to stable labels."""
    if word_count < 20:
        return "short"
    if word_count < 1_000:
        return "medium"
    if word_count < 100_000:
        return "long"
    return "giant"


def iter_word_shingle_hashes(
    text: str,
    ngram_size: int,
    seed: int = 0,
) -> Iterator[int]:
    """Yield stable hashes of normalized, lowercased word shingles.

    A text shorter than the shingle size becomes one whole-document shingle.
    This avoids making every short text share a single generic padding token.
    The tokenizer is deliberately independent of the future Cambacica tokenizer.
    """
    if ngram_size < 1:
        raise ValueError("ngram_size must be positive")
    window: deque[str] = deque(maxlen=ngram_size)
    seen_word = False
    for match in TOKEN_RE.finditer(text):
        token = match.group(0).lower()
        window.append(token)
        seen_word = True
        if len(window) == ngram_size:
            shingle = " ".join(window).encode("utf-8")
            yield xxhash.xxh64(shingle, seed=seed & 0xFFFFFFFF).intdigest()
    if seen_word and len(window) < ngram_size:
        shingle = ("SHORT\0" + " ".join(window)).encode("utf-8")
        yield xxhash.xxh64(shingle, seed=seed & 0xFFFFFFFF).intdigest()


def _permutation_coefficients(
    count: int, seed: int, representation_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Create reproducible 64-bit affine hash coefficients."""
    a: list[int] = []
    b: list[int] = []
    for permutation in range(count):
        prefix = (
            f"cambacica-near-minhash-v1\0{seed}\0{representation_name}\0{permutation}"
        )
        a.append(stable_hash64("a\0" + prefix, seed=seed) | 1)
        b.append(stable_hash64("b\0" + prefix, seed=seed))
    return np.asarray(a, dtype=np.uint64), np.asarray(b, dtype=np.uint64)


def compute_minhash_signature(text: str, config: MinHashConfig) -> list[int]:
    """Compute an order-independent MinHash signature in bounded batches."""
    if config.num_permutations < 1:
        raise ValueError("num_permutations must be positive")
    coefficients_a, coefficients_b = _permutation_coefficients(
        config.num_permutations, config.seed, config.name
    )
    minima = np.full(config.num_permutations, np.iinfo(np.uint64).max, dtype=np.uint64)
    batch: list[int] = []

    def update(values: list[int]) -> None:
        if not values:
            return
        base = np.asarray(values, dtype=np.uint64)
        transformed = coefficients_a[:, None] * base[None, :]
        transformed += coefficients_b[:, None]
        minima[:] = np.minimum(minima, transformed.min(axis=1))

    for shingle_hash in iter_word_shingle_hashes(
        text, config.ngram_size, seed=config.seed
    ):
        batch.append(shingle_hash)
        if len(batch) == SIGNATURE_BATCH_SHINGLES:
            update(batch)
            batch.clear()
    update(batch)
    return [int(value) for value in minima]


def estimate_jaccard_signature(
    signature_a: Sequence[int], signature_b: Sequence[int]
) -> float:
    """Estimate Jaccard similarity from equal-length MinHash signatures."""
    if not signature_a or len(signature_a) != len(signature_b):
        return 0.0
    matches = sum(a == b for a, b in zip(signature_a, signature_b))
    return matches / len(signature_a)


def exact_jaccard_from_text(
    text_a: str,
    text_b: str,
    ngram_size: int = 5,
    seed: int = 0,
    max_shingles: int = MAX_EXACT_REVIEW_SHINGLES,
) -> dict[str, Any]:
    """Compute exact hashed-shingle overlap unless either set exceeds a cap."""
    set_a: set[int] = set()
    for item in iter_word_shingle_hashes(text_a, ngram_size, seed):
        set_a.add(item)
        if len(set_a) > max_shingles:
            return {"exact_available": False, "reason": "shingle_cap_exceeded"}
    set_b: set[int] = set()
    for item in iter_word_shingle_hashes(text_b, ngram_size, seed):
        set_b.add(item)
        if len(set_b) > max_shingles:
            return {"exact_available": False, "reason": "shingle_cap_exceeded"}
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return {
        "exact_available": True,
        "unique_shingles_a": len(set_a),
        "unique_shingles_b": len(set_b),
        "shared_shingles": intersection,
        "union_shingles": union,
        "exact_jaccard": intersection / union if union else 0.0,
        "containment": intersection / min(len(set_a), len(set_b))
        if min(len(set_a), len(set_b))
        else 0.0,
    }


class _BottomK:
    """Order-independent, bounded bottom-k reservoir keyed by a stable rank."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.seed = seed
        self.heap: list[tuple[int, str, dict[str, Any]]] = []
        self.seen = 0

    def add(self, key: str, item: dict[str, Any]) -> None:
        self.seen += 1
        rank = stable_hash64(key, self.seed)
        entry = (-rank, key, item)
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, entry)
            return
        worst_rank = -self.heap[0][0]
        if (rank, key) < (worst_rank, self.heap[0][1]):
            heapq.heapreplace(self.heap, entry)

    def values(self) -> list[dict[str, Any]]:
        return [
            item
            for _negative_rank, _key, item in sorted(
                self.heap, key=lambda entry: (-entry[0], entry[1])
            )
        ]


def _source_subset_key(source: str, subset: str | None) -> str:
    return f"{source}/{subset or ''}"


def _resolution_join_key(
    row: Mapping[str, Any], columns: Sequence[str]
) -> tuple[str, ...]:
    """Build the provenance key shared by resolution and retained data rows."""
    return tuple("" if row.get(name) is None else str(row[name]) for name in columns)


def _word_count_weight(frame_count: int, sample_count: int) -> float | None:
    return frame_count / sample_count if sample_count else None


def _domain(url: str | None) -> str:
    if not url:
        return ""
    try:
        host = (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""
    return host.removeprefix("www.")


class _EnrichmentCollector:
    """Bounded metadata-only collection of deliberately interesting pairs."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.groups: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
        self.group_limits = {
            "same_domain": 8_000,
            "same_title": 12_000,
            "exact_hash": 6_000,
        }
        self.neighbors = _BottomK(2_000, seed ^ 0x4E454947)

    def _remember_group(self, kind: str, key: str, record: dict[str, Any]) -> None:
        max_groups = self.group_limits[kind]
        if not key:
            return
        group_key = (kind, key)
        values = self.groups.get(group_key)
        if values is None:
            if len(self.groups) >= max_groups:
                return
            values = []
            self.groups[group_key] = values
        rank = stable_hash64(record["pilot_occurrence_id"], self.seed ^ 0xD1A6)
        values.append((rank, record))
        values.sort(key=lambda item: (item[0], item[1]["pilot_occurrence_id"]))
        del values[2:]

    def add(self, record: dict[str, Any]) -> None:
        domain = _domain(record.get("original_url"))
        self._remember_group("same_domain", domain, record)
        title = (record.get("title") or "").casefold().strip()
        self._remember_group("same_title", title, record)
        content_hash = record.get("content_sha256") or ""
        if record.get("source") == "parlamento_pt" and content_hash:
            self._remember_group("exact_hash", content_hash, record)

    def add_neighbor(self, previous: dict[str, Any], current: dict[str, Any]) -> None:
        if current.get("source") != "gigaverbo_v2":
            return
        if current.get("subset") != previous.get("subset"):
            return
        if current.get("_gv2_upstream_shard") != previous.get("_gv2_upstream_shard"):
            return
        if current.get("_gv2_upstream_row_group") != previous.get(
            "_gv2_upstream_row_group"
        ):
            return
        pair_key = "|".join(
            sorted(
                (
                    previous["pilot_occurrence_id"],
                    current["pilot_occurrence_id"],
                )
            )
        )
        self.neighbors.add(
            pair_key,
            {
                "id_a": min(
                    previous["pilot_occurrence_id"],
                    current["pilot_occurrence_id"],
                ),
                "id_b": max(
                    previous["pilot_occurrence_id"],
                    current["pilot_occurrence_id"],
                ),
                "reason": "neighboring_crawl_records",
                "record_a": previous,
                "record_b": current,
            },
        )

    def pairs(self) -> list[dict[str, str]]:
        grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for (kind, key), members in sorted(self.groups.items()):
            if len(members) < 2:
                continue
            records = [item[1] for item in members]
            left, right = records[0], records[1]
            if kind == "same_title" and left["source"] == right["source"]:
                cross = next(
                    (
                        (a, b)
                        for a, b in itertools.combinations(records, 2)
                        if a["source"] != b["source"]
                    ),
                    None,
                )
                if cross:
                    left, right = cross
            if kind == "same_domain":
                reason = "same_url_domain"
            elif kind == "same_title":
                reason = "same_title"
            else:
                reason = "parlamento_exact_hash_relative"
            pair = {
                "id_a": min(left["pilot_occurrence_id"], right["pilot_occurrence_id"]),
                "id_b": max(left["pilot_occurrence_id"], right["pilot_occurrence_id"]),
                "reason": reason,
            }
            grouped[reason].append(
                (
                    stable_hash64(f"{key}\0{pair['id_a']}\0{pair['id_b']}", self.seed),
                    pair["id_a"],
                    pair["id_b"],
                )
            )
        reason_caps = {
            "same_url_domain": 180,
            "same_title": 180,
            "parlamento_exact_hash_relative": 120,
        }
        reason_priority = {
            "parlamento_exact_hash_relative": 0,
            "same_title": 1,
            "same_url_domain": 2,
            "neighboring_crawl_records": 3,
        }
        result: list[dict[str, str]] = []
        for reason, pairs in sorted(grouped.items()):
            for _rank, id_a, id_b in sorted(pairs)[: reason_caps[reason]]:
                result.append({"id_a": id_a, "id_b": id_b, "reason": reason})
        neighbor_rows = [
            {
                "id_a": item["id_a"],
                "id_b": item["id_b"],
                "reason": item["reason"],
            }
            for item in self.neighbors.values()
        ]
        result.extend(neighbor_rows[: max(0, 700 - len(result))])
        result.sort(
            key=lambda item: (
                reason_priority[item["reason"]],
                item["id_a"],
                item["id_b"],
            )
        )
        # Keep at most 700 reason-tagged anchors; one pair can show several cues.
        unique: dict[tuple[str, str, str], dict[str, str]] = {}
        for item in result:
            unique[(item["id_a"], item["id_b"], item["reason"])] = item
        return list(unique.values())[:700]


def _candidate_metadata(
    row: Mapping[str, Any], relative_path: str, row_ordinal: int, words: int
) -> dict[str, Any]:
    source = str(row.get("source") or "")
    subset = str(row.get("subset") or "")
    pilot_id = occurrence_id(relative_path, row_ordinal)
    return {
        "pilot_occurrence_id": pilot_id,
        "source": source,
        "subset": subset,
        "normalized_words": words,
        "length_stratum": length_stratum(words),
        "data_relative_path": relative_path,
        "data_row_ordinal": row_ordinal,
        "content_sha256": row.get("content_sha256"),
        "original_id": row.get("original_id"),
        "original_url": row.get("original_url"),
        "title": row.get("title"),
        "domain_category": row.get("domain_category"),
        "_gv2_upstream_shard": row.get("_gv2_upstream_shard"),
        "_gv2_upstream_row_group": row.get("_gv2_upstream_row_group"),
    }


def _iter_data_files(input_root: Path) -> list[Path]:
    data_root = input_root / "data" if (input_root / "data").is_dir() else input_root
    files = sorted(data_root.rglob("*.parquet"))
    if not files:
        raise ValueError(f"No exact-deduplicated Parquet data found under {data_root}")
    return files


def _projected_compressed_bytes(parquet: pq.ParquetFile, columns: Sequence[str]) -> int:
    """Sum Parquet compressed column-chunk sizes for a projected scan."""
    wanted = set(columns)
    total = 0
    for row_group_index in range(parquet.metadata.num_row_groups):
        row_group = parquet.metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            if column.path_in_schema in wanted:
                total += int(column.total_compressed_size)
    return total


def _resolution_band_array(words: pa.Array) -> pa.Array:
    """Vectorized counterpart of ``length_stratum`` for Arrow word counts."""
    return pc.if_else(
        pc.less(words, 20),
        pa.scalar("short"),
        pc.if_else(
            pc.less(words, 1_000),
            pa.scalar("medium"),
            pc.if_else(pc.less(words, 100_000), pa.scalar("long"), pa.scalar("giant")),
        ),
    )


def _append_sample_rows(
    destination: list[dict[str, Any]],
    table: pa.Table,
    mask: pa.Array,
    limit: int,
) -> None:
    """Append at most ``limit`` matching sidecar rows as Python objects."""
    remaining = limit - len(destination)
    if remaining <= 0:
        return
    indices = pc.indices_nonzero(mask).slice(0, remaining)
    if len(indices):
        destination.extend(table.take(indices).to_pylist())


def _scan_and_select(
    input_root: Path,
    files: Sequence[Path],
    *,
    seed: int,
    quotas: Mapping[str, int],
    batch_size: int = 32,
    require_full_coverage: bool = True,
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, str]],
    dict[tuple[str, str], int],
    dict[tuple[str, str], int],
    dict[tuple[str, str, str], int],
    dict[tuple[str, str, str], int],
    dict[str, Any],
]:
    """Use exact retained-row accounting, then enrich from exact-data metadata.

    The exact resolution sidecar records normalized word counts and occurrence
    IDs for every row.  It defines the sampling frame; selected document text
    is still retrieved exclusively from exact/data in the following pass. Its
    source/subset/record-ID order permits a seeded circular sample around a
    per-cell pivot without hashing every corpus row in Python.
    """
    data_root = input_root / "data" if (input_root / "data").is_dir() else input_root
    resolution_path = input_root / "record_resolution.parquet"
    if not resolution_path.is_file():
        raise FileNotFoundError(f"Missing exact resolution sidecar: {resolution_path}")
    subset_counts: Counter[tuple[str, str]] = Counter()
    subset_words: Counter[tuple[str, str]] = Counter()
    frame_counts: Counter[tuple[str, str, str]] = Counter()
    frame_words: Counter[tuple[str, str, str]] = Counter()
    sample_after_pivot: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    sample_before_pivot: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    seen_sources: set[str] = set()
    seen_gv_subsets: set[str] = set()
    input_bytes = sum(path.stat().st_size for path in files)
    resolution_wall_started = time.perf_counter()
    resolution_cpu_started = time.process_time()
    resolution = pq.ParquetFile(resolution_path)
    resolution_columns = [
        "record_id",
        "content_sha256",
        "source",
        "subset",
        "normalized_words",
        "disposition",
        "normalized_shard",
        "source_revision",
        "original_id",
        "raw_source_file",
        "raw_record_identifier",
        "_gv2_upstream_shard",
        "_gv2_upstream_row_group",
        "_gv2_upstream_commit",
        "normalized_row_ordinal",
    ]
    resolution_projection_bytes = _projected_compressed_bytes(
        resolution, resolution_columns
    )
    retained_count = 0
    join_columns = (
        "source",
        "subset",
        "content_sha256",
        "source_revision",
        "original_id",
        "raw_source_file",
        "raw_record_identifier",
        "_gv2_upstream_shard",
        "_gv2_upstream_row_group",
        "_gv2_upstream_commit",
    )
    for batch in resolution.iter_batches(
        batch_size=16_384,
        columns=resolution_columns,
    ):
        retained = batch.filter(pc.not_equal(batch["disposition"], "dropped"))
        if not retained.num_rows:
            continue
        augmented = pa.Table.from_batches(
            [
                retained.append_column(
                    "_pilot_band", _resolution_band_array(retained["normalized_words"])
                )
            ]
        )
        grouped = (
            augmented.select(
                ["source", "subset", "_pilot_band", "normalized_words", "record_id"]
            )
            .group_by(["source", "subset", "_pilot_band"])
            .aggregate([("normalized_words", "sum"), ("record_id", "count")])
        )
        for summary in grouped.to_pylist():
            source = str(summary["source"])
            subset = str(summary["subset"] or "")
            band = str(summary["_pilot_band"])
            cell = (source, subset, band)
            count = int(summary["record_id_count"])
            words = int(summary["normalized_words_sum"])
            frame_counts[cell] += count
            frame_words[cell] += words
            subset_counts[(source, subset)] += count
            subset_words[(source, subset)] += words
            retained_count += count
            seen_sources.add(source)
            if source == "gigaverbo_v2":
                seen_gv_subsets.add(subset)

            cell_mask = pc.and_kleene(
                pc.and_kleene(
                    pc.equal(augmented["source"], source),
                    pc.equal(augmented["subset"], subset),
                ),
                pc.equal(augmented["_pilot_band"], band),
            )
            pivot_value = stable_hash64(
                f"near-pilot-sample-pivot-v1\0{seed}\0{source}\0{subset}\0{band}",
                seed,
            )
            pivot = f"{pivot_value:016x}" * 4
            after_mask = pc.and_kleene(
                cell_mask, pc.greater_equal(augmented["record_id"], pivot)
            )
            before_mask = pc.and_kleene(
                cell_mask, pc.less(augmented["record_id"], pivot)
            )
            quota = quotas[band]
            _append_sample_rows(sample_after_pivot[cell], augmented, after_mask, quota)
            _append_sample_rows(
                sample_before_pivot[cell], augmented, before_mask, quota
            )
    resolution_wall = time.perf_counter() - resolution_wall_started
    resolution_cpu = time.process_time() - resolution_cpu_started

    base_by_join_key: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    base_record_ids: set[str] = set()
    for cell in sorted(frame_counts):
        quota = quotas[cell[2]]
        after = sample_after_pivot[cell][:quota]
        before = sample_before_pivot[cell][: max(0, quota - len(after))]
        sample_count = min(frame_counts[cell], quota)
        for row in [*after, *before]:
            record = {
                "resolution_record_id": row["record_id"],
                "source": str(row["source"]),
                "subset": str(row["subset"] or ""),
                "normalized_words": int(row["normalized_words"]),
                "length_stratum": str(row["_pilot_band"]),
                "normalized_shard": row["normalized_shard"],
                "normalized_row_ordinal": int(row["normalized_row_ordinal"]),
                "data_relative_path": row["normalized_shard"],
                "content_sha256": row["content_sha256"],
                "_resolution_join_key": _resolution_join_key(row, join_columns),
                "selection_role": "stratified_base",
                "sample_tags": ["stratified_base"],
                "sampling_frame_population": frame_counts[cell],
                "sampling_frame_sample_n": sample_count,
                "base_sampling_weight": _word_count_weight(
                    frame_counts[cell], sample_count
                ),
            }
            base_by_join_key[record["_resolution_join_key"]].append(record)
            base_record_ids.add(record["resolution_record_id"])
    for records in base_by_join_key.values():
        records.sort(key=lambda row: row["resolution_record_id"])
    lookup_positions: Counter[tuple[str, ...]] = Counter()
    selected: dict[str, dict[str, Any]] = {}
    mapped_base_ids: set[str] = set()
    enrichment = _EnrichmentCollector(seed)
    metadata_wall_started = time.perf_counter()
    metadata_cpu_started = time.process_time()
    metadata_projection_bytes = 0
    exact_data_rows = 0
    selected_hashes = pa.array(
        sorted(
            {
                record["content_sha256"]
                for records in base_by_join_key.values()
                for record in records
                if record.get("content_sha256")
            }
        ),
        type=pa.string(),
    )
    for path in files:
        relative_path = path.relative_to(data_root).as_posix()
        parquet = pq.ParquetFile(path)
        ordinal = 0
        exact_data_rows += parquet.metadata.num_rows
        columns = [
            "source",
            "subset",
            "content_sha256",
            "source_revision",
            "original_id",
            "raw_source_file",
            "raw_record_identifier",
            "_gv2_upstream_shard",
            "_gv2_upstream_row_group",
            "_gv2_upstream_commit",
            "original_url",
            "title",
            "domain_category",
        ]
        metadata_projection_bytes += _projected_compressed_bytes(parquet, columns)
        for batch in parquet.iter_batches(batch_size=4096, columns=columns):
            match_mask = pc.is_in(batch["content_sha256"], value_set=selected_hashes)
            match_indices = pc.indices_nonzero(match_mask).to_pylist()
            match_rows = batch.filter(match_mask).to_pylist()
            for row_index, row in zip(match_indices, match_rows, strict=True):
                join_key = _resolution_join_key(row, join_columns)
                position = lookup_positions[join_key]
                base_rows = base_by_join_key.get(join_key, [])
                base_record = base_rows[position] if position < len(base_rows) else None
                if base_record is None:
                    continue
                lookup_positions[join_key] += 1
                words = int(base_record["normalized_words"])
                record = _candidate_metadata(
                    row, relative_path, ordinal + row_index, words
                )
                record.update(
                    {
                        "selection_role": "stratified_base",
                        "sample_tags": ["stratified_base"],
                        "sampling_frame_population": base_record[
                            "sampling_frame_population"
                        ],
                        "sampling_frame_sample_n": base_record[
                            "sampling_frame_sample_n"
                        ],
                        "base_sampling_weight": base_record["base_sampling_weight"],
                    }
                )
                selected[record["pilot_occurrence_id"]] = record
                mapped_base_ids.add(base_record["resolution_record_id"])
                enrichment.add(record)
                if record["source"] == "gigaverbo_v2":
                    for neighbor_index in (row_index - 1, row_index + 1):
                        if not 0 <= neighbor_index < batch.num_rows:
                            continue
                        neighbor_row = batch.take(
                            pa.array([neighbor_index], type=pa.int64())
                        ).to_pylist()[0]
                        if (
                            neighbor_row.get("source") != record["source"]
                            or (neighbor_row.get("subset") or "") != record["subset"]
                            or neighbor_row.get("_gv2_upstream_shard")
                            != record.get("_gv2_upstream_shard")
                            or neighbor_row.get("_gv2_upstream_row_group")
                            != record.get("_gv2_upstream_row_group")
                        ):
                            continue
                        neighbor_ordinal = ordinal + neighbor_index
                        neighbor = _candidate_metadata(
                            neighbor_row, relative_path, neighbor_ordinal, 0
                        )
                        neighbor.update(
                            {
                                "selection_role": "diagnostic_enrichment",
                                "sample_tags": ["diagnostic_enrichment"],
                                "sampling_frame_population": None,
                                "sampling_frame_sample_n": None,
                                "base_sampling_weight": None,
                            }
                        )
                        selected.setdefault(neighbor["pilot_occurrence_id"], neighbor)
                        enrichment.add_neighbor(record, neighbor)
            ordinal += len(batch)
    metadata_wall = time.perf_counter() - metadata_wall_started
    metadata_cpu = time.process_time() - metadata_cpu_started
    if mapped_base_ids != base_record_ids:
        raise ValueError(
            "Could not map every selected exact occurrence to exact/data: "
            f"{len(mapped_base_ids)} of {len(base_record_ids)}"
        )
    if exact_data_rows != retained_count:
        raise ValueError(
            "Exact/data metadata row count differs from retained resolution: "
            f"{exact_data_rows} != {retained_count}"
        )

    enrichment_pairs = enrichment.pairs()
    enrichment_metadata = _enrichment_records(enrichment)
    enrichment_ids = {
        pilot_id
        for pair in enrichment_pairs
        for pilot_id in (pair["id_a"], pair["id_b"])
    }
    for pilot_id in sorted(enrichment_ids):
        record = enrichment_metadata[pilot_id]
        if pilot_id in selected:
            selected[pilot_id]["sample_tags"].append("diagnostic_enrichment")
            continue
        record = dict(record)
        record["selection_role"] = "diagnostic_enrichment"
        record["sample_tags"] = ["diagnostic_enrichment"]
        record["sampling_frame_population"] = None
        record["sampling_frame_sample_n"] = None
        record["base_sampling_weight"] = None
        selected[pilot_id] = record

    if require_full_coverage and seen_sources != EXPECTED_SOURCES:
        raise ValueError(f"Exact data source coverage mismatch: {sorted(seen_sources)}")
    if require_full_coverage and seen_gv_subsets != EXPECTED_GIGAVERBO_SUBSETS:
        raise ValueError(
            f"Exact data GigaVerbo subset coverage mismatch: {sorted(seen_gv_subsets)}"
        )
    metrics = {
        "records_scanned": retained_count,
        "resolution_rows_including_exact_dropped": resolution.metadata.num_rows,
        "exact_metadata_rows_scanned": exact_data_rows,
        "resolution_file_bytes": resolution_path.stat().st_size,
        "resolution_projection_compressed_bytes": resolution_projection_bytes,
        "exact_data_metadata_projection_compressed_bytes": metadata_projection_bytes,
        "input_data_bytes": input_bytes,
        "resolution_scan_wall_seconds": resolution_wall,
        "resolution_scan_cpu_seconds": resolution_cpu,
        "metadata_scan_wall_seconds": metadata_wall,
        "metadata_scan_cpu_seconds": metadata_cpu,
        "scan_wall_seconds": resolution_wall + metadata_wall,
        "scan_cpu_seconds": resolution_cpu + metadata_cpu,
        "frame_counts": {
            f"{source}/{subset}/{band}": count
            for (source, subset, band), count in sorted(frame_counts.items())
        },
        "frame_words": {
            f"{source}/{subset}/{band}": count
            for (source, subset, band), count in sorted(frame_words.items())
        },
        "seen_sources": sorted(seen_sources),
        "seen_gigaverbo_subsets": sorted(seen_gv_subsets),
    }
    return (
        selected,
        enrichment_pairs,
        dict(subset_counts),
        dict(subset_words),
        dict(frame_counts),
        dict(frame_words),
        metrics,
    )


def _enrichment_records(collector: _EnrichmentCollector) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for members in collector.groups.values():
        for _rank, record in members:
            records[record["pilot_occurrence_id"]] = record
    for record in collector.neighbors.values():
        records[record["record_a"]["pilot_occurrence_id"]] = record["record_a"]
        records[record["record_b"]["pilot_occurrence_id"]] = record["record_b"]
    return records


def _materialize_pilot_records(
    input_root: Path,
    files: Sequence[Path],
    selected: Mapping[str, dict[str, Any]],
    output_path: Path,
    *,
    batch_size: int = 32,
) -> tuple[int, int]:
    """Re-read selected rows only, preserving all selected Parlamento rows."""
    data_root = input_root / "data" if (input_root / "data").is_dir() else input_root
    selected_by_path: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in selected.values():
        selected_by_path[record["data_relative_path"]][
            int(record["data_row_ordinal"])
        ] = record
    first_schema = pq.read_schema(files[0])
    output_schema = first_schema.append(
        pa.field("pilot_occurrence_id", pa.string(), nullable=False)
    )
    output_schema = output_schema.append(
        pa.field("normalized_words", pa.int64(), nullable=False)
    )
    output_schema = output_schema.append(
        pa.field("length_stratum", pa.string(), nullable=False)
    )
    output_schema = output_schema.append(
        pa.field("selection_role", pa.string(), nullable=False)
    )
    output_schema = output_schema.append(
        pa.field("sample_tags", pa.list_(pa.string()), nullable=False)
    )
    output_schema = output_schema.append(
        pa.field("data_relative_path", pa.string(), nullable=False)
    )
    output_schema = output_schema.append(
        pa.field("data_row_ordinal", pa.int64(), nullable=False)
    )
    output_schema = output_schema.append(
        pa.field("sampling_frame_population", pa.int64())
    )
    output_schema = output_schema.append(
        pa.field("sampling_frame_sample_n", pa.int64())
    )
    output_schema = output_schema.append(pa.field("base_sampling_weight", pa.float64()))
    output_schema = output_schema.append(
        pa.field("pilot_file_row_ordinal", pa.int64(), nullable=False)
    )
    temporary_path = output_path.with_name(f"{output_path.name}.partial")
    writer = pq.ParquetWriter(
        temporary_path,
        output_schema,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )
    pending: list[dict[str, Any]] = []
    pending_text_bytes = 0
    seen_ids: set[str] = set()
    written_ordinal = 0

    def flush() -> None:
        nonlocal pending_text_bytes, written_ordinal
        if not pending:
            return
        writer.write_table(pa.Table.from_pylist(pending, schema=output_schema))
        written_ordinal += len(pending)
        pending.clear()
        pending_text_bytes = 0

    try:
        for path in files:
            relative_path = path.relative_to(data_root).as_posix()
            wanted = selected_by_path.get(relative_path)
            if not wanted:
                continue
            parquet = pq.ParquetFile(path)
            offset = 0
            for row_group in range(parquet.metadata.num_row_groups):
                rows_in_group = parquet.metadata.row_group(row_group).num_rows
                high = offset + rows_in_group
                if not any(offset <= ordinal < high for ordinal in wanted):
                    offset = high
                    continue
                local_wanted = {
                    ordinal - offset: record
                    for ordinal, record in wanted.items()
                    if offset <= ordinal < high
                }
                wanted_ordinals = sorted(local_wanted)
                wanted_cursor = 0
                row_index = 0
                for batch in parquet.iter_batches(
                    row_groups=[row_group], batch_size=batch_size
                ):
                    batch_end = row_index + batch.num_rows
                    positions: list[int] = []
                    while (
                        wanted_cursor < len(wanted_ordinals)
                        and wanted_ordinals[wanted_cursor] < batch_end
                    ):
                        positions.append(wanted_ordinals[wanted_cursor] - row_index)
                        wanted_cursor += 1
                    if positions:
                        selected_rows = batch.take(
                            pa.array(positions, type=pa.int64())
                        ).to_pylist()
                        for position, row in zip(positions, selected_rows, strict=True):
                            selected_meta = local_wanted[row_index + position]
                            enriched = dict(row)
                            row_ordinal = offset + row_index + position
                            pilot_id = occurrence_id(relative_path, row_ordinal)
                            if pilot_id != selected_meta["pilot_occurrence_id"]:
                                raise ValueError(
                                    "Pilot occurrence identity changed on reread"
                                )
                            actual_words = count_normalized_words_bounded(
                                row.get("text") or ""
                            )
                            expected_words = int(selected_meta["normalized_words"])
                            if (
                                selected_meta["selection_role"] == "stratified_base"
                                and actual_words != expected_words
                            ):
                                raise ValueError(
                                    "Resolution word count differs from sampled exact text "
                                    f"for {pilot_id}: {expected_words} != {actual_words}"
                                )
                            enriched.update(
                                {
                                    "pilot_occurrence_id": pilot_id,
                                    "normalized_words": actual_words,
                                    "length_stratum": length_stratum(actual_words),
                                    "selection_role": selected_meta["selection_role"],
                                    "sample_tags": sorted(
                                        set(selected_meta.get("sample_tags", []))
                                    ),
                                    "data_relative_path": relative_path,
                                    "data_row_ordinal": row_ordinal,
                                    "sampling_frame_population": selected_meta.get(
                                        "sampling_frame_population"
                                    ),
                                    "sampling_frame_sample_n": selected_meta.get(
                                        "sampling_frame_sample_n"
                                    ),
                                    "base_sampling_weight": selected_meta.get(
                                        "base_sampling_weight"
                                    ),
                                    "pilot_file_row_ordinal": written_ordinal
                                    + len(pending),
                                }
                            )
                            pending.append(enriched)
                            pending_text_bytes += len(
                                (row.get("text") or "").encode("utf-8")
                            )
                            seen_ids.add(pilot_id)
                            if (
                                len(pending) >= 32
                                or pending_text_bytes >= 64 * 1024 * 1024
                            ):
                                flush()
                    row_index = batch_end
                if row_index != rows_in_group:
                    raise ValueError(f"Row count changed while reading {path}")
                offset = high
        flush()
        writer.close()
        if len(seen_ids) != len(selected):
            missing = sorted(set(selected) - seen_ids)
            raise ValueError(f"Pilot re-read omitted {len(missing)} selected rows")
        os.replace(temporary_path, output_path)
    except BaseException:
        try:
            writer.close()
        except Exception:
            pass
        temporary_path.unlink(missing_ok=True)
        raise
    return len(seen_ids), output_path.stat().st_size


def _write_signatures(
    pilot_records_path: Path,
    output_path: Path,
    *,
    seed: int = DEFAULT_NEAR_PILOT_SEED,
    batch_size: int = 1,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, list[int]]], dict[str, Any]]:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    parquet = pq.ParquetFile(pilot_records_path)
    configs = tuple(
        MinHashConfig(item.name, item.ngram_size, item.num_permutations, seed)
        for item in SIGNATURE_CONFIGS
    )
    signatures: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    records_per_config = Counter()
    seconds_by_config = Counter()
    words_by_config = Counter()
    seconds_by_cell_config: dict[str, Counter[str]] = defaultdict(Counter)
    words_by_cell_config: dict[str, Counter[str]] = defaultdict(Counter)
    records_by_cell_config: dict[str, Counter[str]] = defaultdict(Counter)
    schema = pa.schema(
        [
            pa.field("pilot_occurrence_id", pa.string(), nullable=False),
            pa.field("word3_128", pa.list_(pa.uint64()), nullable=False),
            pa.field("word5_256", pa.list_(pa.uint64()), nullable=False),
            pa.field("word7_128", pa.list_(pa.uint64()), nullable=False),
        ]
    )
    temporary = output_path.with_name(f"{output_path.name}.partial")
    writer = pq.ParquetWriter(
        temporary,
        schema,
        compression="zstd",
        compression_level=6,
        use_dictionary=False,
        write_statistics=True,
        version="2.6",
    )
    pending: list[dict[str, Any]] = []
    try:
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=[
                "pilot_occurrence_id",
                "text",
                "source",
                "subset",
                "normalized_words",
                "length_stratum",
                "selection_role",
                "sample_tags",
                "original_url",
                "title",
                "data_relative_path",
                "data_row_ordinal",
                "pilot_file_row_ordinal",
                "content_sha256",
            ],
        ):
            for row in batch.to_pylist():
                pilot_id = row["pilot_occurrence_id"]
                metadata[pilot_id] = {
                    key: value for key, value in row.items() if key != "text"
                }
                metadata[pilot_id]["source"] = str(row.get("source") or "")
                metadata[pilot_id]["subset"] = str(row.get("subset") or "")
                record_signatures: dict[str, list[int]] = {}
                cell = (
                    f"{_source_subset_key(row['source'], row['subset'])}/"
                    f"{row['length_stratum']}/{row['selection_role']}"
                )
                for config in configs:
                    config_started = time.process_time()
                    signature = compute_minhash_signature(row["text"], config)
                    elapsed = time.process_time() - config_started
                    seconds_by_config[config.name] += elapsed
                    words_by_config[config.name] += int(row["normalized_words"])
                    records_per_config[config.name] += 1
                    seconds_by_cell_config[config.name][cell] += elapsed
                    words_by_cell_config[config.name][cell] += int(
                        row["normalized_words"]
                    )
                    records_by_cell_config[config.name][cell] += 1
                    record_signatures[config.name] = signature
                signatures[pilot_id] = record_signatures
                pending.append(
                    {
                        "pilot_occurrence_id": pilot_id,
                        "word3_128": record_signatures["word3_128"],
                        "word5_256": record_signatures["word5_256"],
                        "word7_128": record_signatures["word7_128"],
                    }
                )
                if len(pending) >= 256:
                    writer.write_table(pa.Table.from_pylist(pending, schema=schema))
                    pending.clear()
        if pending:
            writer.write_table(pa.Table.from_pylist(pending, schema=schema))
        writer.close()
        os.replace(temporary, output_path)
    except BaseException:
        try:
            writer.close()
        except Exception:
            pass
        temporary.unlink(missing_ok=True)
        raise
    timing = {
        "signature_wall_seconds": time.perf_counter() - started_wall,
        "signature_cpu_seconds": time.process_time() - started_cpu,
        "seconds_by_configuration": dict(seconds_by_config),
        "words_processed_by_configuration": dict(words_by_config),
        "records_per_configuration": dict(records_per_config),
        "cpu_seconds_by_cell_and_role": {
            config: dict(values) for config, values in seconds_by_cell_config.items()
        },
        "words_by_cell_and_role": {
            config: dict(values) for config, values in words_by_cell_config.items()
        },
        "records_by_cell_and_role": {
            config: dict(values) for config, values in records_by_cell_config.items()
        },
        "signatures_per_second_by_configuration": {
            config.name: records_per_config[config.name]
            / seconds_by_config[config.name]
            if seconds_by_config[config.name]
            else 0.0
            for config in configs
        },
    }
    return metadata, signatures, timing


def _create_lsh_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.executescript(
        """
        CREATE TABLE lsh_buckets (
            config_id TEXT NOT NULL,
            band INTEGER NOT NULL,
            bucket_key BLOB NOT NULL,
            record_id TEXT NOT NULL,
            rank BLOB NOT NULL,
            PRIMARY KEY(config_id, band, bucket_key, record_id)
        ) WITHOUT ROWID;
        CREATE INDEX buckets_by_group
            ON lsh_buckets(config_id, band, bucket_key);
        CREATE TABLE candidate_by_config (
            config_id TEXT NOT NULL,
            id_a TEXT NOT NULL,
            id_b TEXT NOT NULL,
            PRIMARY KEY(config_id, id_a, id_b)
        ) WITHOUT ROWID;
        CREATE TABLE enrichment_pairs (
            id_a TEXT NOT NULL,
            id_b TEXT NOT NULL,
            reason TEXT NOT NULL,
            PRIMARY KEY(id_a, id_b, reason)
        ) WITHOUT ROWID;
        CREATE TABLE pair_scores (
            id_a TEXT NOT NULL,
            id_b TEXT NOT NULL,
            generators TEXT NOT NULL,
            enrichment_reasons TEXT NOT NULL,
            estimate_word3_128 REAL NOT NULL,
            estimate_word5_64 REAL NOT NULL,
            estimate_word5_128 REAL NOT NULL,
            estimate_word5_256 REAL NOT NULL,
            estimate_word7_128 REAL NOT NULL,
            lsh_generated INTEGER NOT NULL,
            PRIMARY KEY(id_a, id_b)
        ) WITHOUT ROWID;
        """
    )
    return connection


def _band_key(
    signature: Sequence[int], start: int, rows: int, config_name: str, band: int
) -> bytes:
    packed = struct.pack(f"<{rows}Q", *signature[start : start + rows])
    seed = stable_hash64(f"{config_name}\0{band}") & 0xFFFFFFFF
    value = xxhash.xxh64(packed, seed=seed).intdigest()
    return value.to_bytes(8, "big")


def _run_lsh(
    connection: sqlite3.Connection,
    signatures: Mapping[str, Mapping[str, Sequence[int]]],
    *,
    seed: int,
    bucket_cap: int = BUCKET_MEMBER_CAP,
) -> dict[str, Any]:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    bucket_insert = []
    for config in LSH_CONFIGS:
        for pilot_id, item_signatures in sorted(signatures.items()):
            signature = item_signatures[config.representation][
                : config.num_permutations
            ]
            for band in range(config.bands):
                key = _band_key(
                    signature,
                    band * config.rows_per_band,
                    config.rows_per_band,
                    config.name,
                    band,
                )
                rank = hashlib.sha256(
                    f"{seed}\0{config.name}\0{band}\0{key.hex()}\0{pilot_id}".encode(
                        "utf-8"
                    )
                ).digest()[:8]
                bucket_insert.append((config.name, band, key, pilot_id, rank))
        connection.executemany(
            "INSERT INTO lsh_buckets VALUES (?, ?, ?, ?, ?)", bucket_insert
        )
        bucket_insert.clear()
        connection.commit()

    by_config: dict[str, Any] = {}
    candidate_insert: list[tuple[str, str, str]] = []
    for config in LSH_CONFIGS:
        sizes = [
            int(row[0])
            for row in connection.execute(
                "SELECT COUNT(*) FROM lsh_buckets WHERE config_id=? "
                "GROUP BY band, bucket_key",
                (config.name,),
            )
        ]
        sizes.sort()
        size_histogram = Counter(
            "1"
            if size == 1
            else "2-5"
            if size <= 5
            else "6-20"
            if size <= 20
            else "21-64"
            if size <= 64
            else "65-256"
            if size <= bucket_cap
            else f">{bucket_cap}"
            for size in sizes
        )
        too_large = [size for size in sizes if size > bucket_cap]
        possible_pairs = sum(size * (size - 1) // 2 for size in sizes)
        emitted_pairs = 0
        for band, bucket_key, bucket_size in connection.execute(
            "SELECT band, bucket_key, COUNT(*) FROM lsh_buckets "
            "WHERE config_id=? GROUP BY band, bucket_key ORDER BY band, bucket_key",
            (config.name,),
        ):
            if int(bucket_size) < 2:
                continue
            members = [
                row[0]
                for row in connection.execute(
                    "SELECT record_id FROM lsh_buckets WHERE config_id=? "
                    "AND band=? AND bucket_key=? ORDER BY rank, record_id LIMIT ?",
                    (config.name, band, bucket_key, bucket_cap),
                )
            ]
            for id_a, id_b in itertools.combinations(sorted(members), 2):
                candidate_insert.append((config.name, id_a, id_b))
                emitted_pairs += 1
                if len(candidate_insert) >= 50_000:
                    connection.executemany(
                        "INSERT OR IGNORE INTO candidate_by_config VALUES (?, ?, ?)",
                        candidate_insert,
                    )
                    candidate_insert.clear()
                    connection.commit()
        if candidate_insert:
            connection.executemany(
                "INSERT OR IGNORE INTO candidate_by_config VALUES (?, ?, ?)",
                candidate_insert,
            )
            candidate_insert.clear()
            connection.commit()
        unique_candidates = int(
            connection.execute(
                "SELECT COUNT(*) FROM candidate_by_config WHERE config_id=?",
                (config.name,),
            ).fetchone()[0]
        )
        by_config[config.name] = {
            "permutations": config.num_permutations,
            "bands": config.bands,
            "rows_per_band": config.rows_per_band,
            "bucket_count": len(sizes),
            "bucket_size_distribution": dict(sorted(size_histogram.items())),
            "bucket_size_p50": _quantile(sizes, 0.50),
            "bucket_size_p90": _quantile(sizes, 0.90),
            "bucket_size_p99": _quantile(sizes, 0.99),
            "largest_bucket": max(sizes, default=0),
            "pathological_bucket_count": len(too_large),
            "pathological_bucket_cap": bucket_cap,
            "omitted_pair_upper_bound_from_capped_buckets": sum(
                size * (size - 1) // 2 - bucket_cap * (bucket_cap - 1) // 2
                for size in too_large
            ),
            "all_bucket_pair_occurrences_before_dedup": possible_pairs,
            "emitted_pair_occurrences_before_dedup": emitted_pairs,
            "unique_candidate_pairs": unique_candidates,
        }
    return {
        "configurations": by_config,
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "bucket_index_entries": int(
            connection.execute("SELECT COUNT(*) FROM lsh_buckets").fetchone()[0]
        ),
        "candidate_config_entries": int(
            connection.execute("SELECT COUNT(*) FROM candidate_by_config").fetchone()[0]
        ),
        "bucket_database_bytes": Path(
            connection.execute("PRAGMA database_list").fetchone()[2]
        )
        .stat()
        .st_size,
    }


def _quantile(sorted_values: Sequence[int], q: float) -> int:
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))
    return int(sorted_values[index])


def _write_pair_artifacts(
    connection: sqlite3.Connection,
    metadata: Mapping[str, Mapping[str, Any]],
    signatures: Mapping[str, Mapping[str, Sequence[int]]],
    enrichment_pairs: Sequence[Mapping[str, str]],
    output_candidates: Path,
    output_scored: Path,
) -> dict[str, Any]:
    for pair in enrichment_pairs:
        if pair["id_a"] == pair["id_b"]:
            continue
        connection.execute(
            "INSERT OR IGNORE INTO enrichment_pairs VALUES (?, ?, ?)",
            (pair["id_a"], pair["id_b"], pair["reason"]),
        )
    connection.commit()
    connection.executescript(
        """
        CREATE TABLE candidate_union AS
        SELECT id_a, id_b, GROUP_CONCAT(config_id, ',') AS generators
        FROM candidate_by_config GROUP BY id_a, id_b;
        CREATE UNIQUE INDEX candidate_union_ids ON candidate_union(id_a, id_b);
        CREATE TABLE pair_union AS
        SELECT id_a, id_b, generators FROM candidate_union
        UNION
        SELECT e.id_a, e.id_b, '' AS generators FROM enrichment_pairs e
        WHERE NOT EXISTS (
            SELECT 1 FROM candidate_union c
            WHERE c.id_a=e.id_a AND c.id_b=e.id_b
        );
        CREATE UNIQUE INDEX pair_union_ids ON pair_union(id_a, id_b);
        """
    )
    candidate_schema = pa.schema(
        [
            pa.field("id_a", pa.string(), nullable=False),
            pa.field("id_b", pa.string(), nullable=False),
            pa.field("generator_configs", pa.list_(pa.string()), nullable=False),
            pa.field("enrichment_reasons", pa.list_(pa.string()), nullable=False),
            pa.field("lsh_generated", pa.bool_(), nullable=False),
            pa.field("cross_source", pa.bool_(), nullable=False),
            pa.field("same_source_cross_subset", pa.bool_(), nullable=False),
        ]
    )
    score_schema = candidate_schema.append(
        pa.field("estimate_word3_128", pa.float32(), nullable=False)
    )
    score_schema = score_schema.append(
        pa.field("estimate_word5_64", pa.float32(), nullable=False)
    )
    score_schema = score_schema.append(
        pa.field("estimate_word5_128", pa.float32(), nullable=False)
    )
    score_schema = score_schema.append(
        pa.field("estimate_word5_256", pa.float32(), nullable=False)
    )
    score_schema = score_schema.append(
        pa.field("estimate_word7_128", pa.float32(), nullable=False)
    )
    score_schema = score_schema.append(
        pa.field("minhash_256_standard_error", pa.float32(), nullable=False)
    )
    score_schema = score_schema.append(
        pa.field("source_a", pa.string(), nullable=False)
    )
    score_schema = score_schema.append(
        pa.field("subset_a", pa.string(), nullable=False)
    )
    score_schema = score_schema.append(pa.field("words_a", pa.int64(), nullable=False))
    score_schema = score_schema.append(
        pa.field("source_b", pa.string(), nullable=False)
    )
    score_schema = score_schema.append(
        pa.field("subset_b", pa.string(), nullable=False)
    )
    score_schema = score_schema.append(pa.field("words_b", pa.int64(), nullable=False))

    candidate_tmp = output_candidates.with_name(f"{output_candidates.name}.partial")
    scored_tmp = output_scored.with_name(f"{output_scored.name}.partial")
    candidate_writer = pq.ParquetWriter(
        candidate_tmp, candidate_schema, compression="zstd"
    )
    score_writer = pq.ParquetWriter(scored_tmp, score_schema, compression="zstd")
    candidate_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    candidate_count = 0
    scored_count = 0
    score_insert: list[tuple[Any, ...]] = []
    try:
        pair_cursor = connection.execute(
            "SELECT p.id_a, p.id_b, COALESCE(c.generators,''), "
            "COALESCE(GROUP_CONCAT(e.reason,','), '') "
            "FROM pair_union p LEFT JOIN candidate_union c USING(id_a,id_b) "
            "LEFT JOIN enrichment_pairs e USING(id_a,id_b) "
            "GROUP BY p.id_a,p.id_b,c.generators ORDER BY p.id_a,p.id_b"
        )
        for id_a, id_b, generators_text, reasons_text in pair_cursor:
            if id_a not in metadata or id_b not in metadata:
                continue
            generators = sorted(item for item in generators_text.split(",") if item)
            reasons = sorted(item for item in reasons_text.split(",") if item)
            row_a, row_b = metadata[id_a], metadata[id_b]
            sig_a, sig_b = signatures[id_a], signatures[id_b]
            est3 = estimate_jaccard_signature(sig_a["word3_128"], sig_b["word3_128"])
            est5_64 = estimate_jaccard_signature(
                sig_a["word5_256"][:64], sig_b["word5_256"][:64]
            )
            est5_128 = estimate_jaccard_signature(
                sig_a["word5_256"][:128], sig_b["word5_256"][:128]
            )
            est5_256 = estimate_jaccard_signature(
                sig_a["word5_256"], sig_b["word5_256"]
            )
            est7 = estimate_jaccard_signature(sig_a["word7_128"], sig_b["word7_128"])
            score = {
                "id_a": id_a,
                "id_b": id_b,
                "generator_configs": generators,
                "enrichment_reasons": reasons,
                "lsh_generated": bool(generators),
                "cross_source": row_a["source"] != row_b["source"],
                "same_source_cross_subset": row_a["source"] == row_b["source"]
                and row_a["subset"] != row_b["subset"],
                "estimate_word3_128": est3,
                "estimate_word5_64": est5_64,
                "estimate_word5_128": est5_128,
                "estimate_word5_256": est5_256,
                "estimate_word7_128": est7,
                "minhash_256_standard_error": math.sqrt(
                    max(0.0, est5_256 * (1 - est5_256) / 256)
                ),
                "source_a": row_a["source"],
                "subset_a": row_a["subset"],
                "words_a": int(row_a["normalized_words"]),
                "source_b": row_b["source"],
                "subset_b": row_b["subset"],
                "words_b": int(row_b["normalized_words"]),
            }
            candidate_rows.append(
                {
                    key: score[key]
                    for key in (
                        "id_a",
                        "id_b",
                        "generator_configs",
                        "enrichment_reasons",
                        "lsh_generated",
                        "cross_source",
                        "same_source_cross_subset",
                    )
                }
            )
            score_rows.append(score)
            score_insert.append(
                (
                    id_a,
                    id_b,
                    ",".join(generators),
                    ",".join(reasons),
                    est3,
                    est5_64,
                    est5_128,
                    est5_256,
                    est7,
                    int(bool(generators)),
                )
            )
            candidate_count += 1
            scored_count += 1
            if len(candidate_rows) >= 4_000:
                candidate_writer.write_table(
                    pa.Table.from_pylist(candidate_rows, schema=candidate_schema)
                )
                score_writer.write_table(
                    pa.Table.from_pylist(score_rows, schema=score_schema)
                )
                candidate_rows.clear()
                score_rows.clear()
            if len(score_insert) >= 20_000:
                connection.executemany(
                    "INSERT OR REPLACE INTO pair_scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    score_insert,
                )
                connection.commit()
                score_insert.clear()
        if candidate_rows:
            candidate_writer.write_table(
                pa.Table.from_pylist(candidate_rows, schema=candidate_schema)
            )
            score_writer.write_table(
                pa.Table.from_pylist(score_rows, schema=score_schema)
            )
        if score_insert:
            connection.executemany(
                "INSERT OR REPLACE INTO pair_scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                score_insert,
            )
            connection.commit()
        candidate_writer.close()
        score_writer.close()
        os.replace(candidate_tmp, output_candidates)
        os.replace(scored_tmp, output_scored)
    except BaseException:
        try:
            candidate_writer.close()
            score_writer.close()
        except Exception:
            pass
        candidate_tmp.unlink(missing_ok=True)
        scored_tmp.unlink(missing_ok=True)
        raise
    return {
        "pair_union_count": candidate_count,
        "lsh_generated_pair_count": int(
            connection.execute("SELECT COUNT(*) FROM candidate_union").fetchone()[0]
        ),
        "diagnostic_enrichment_only_pair_count": int(
            connection.execute(
                "SELECT COUNT(*) FROM pair_union p LEFT JOIN candidate_union c USING(id_a,id_b) WHERE c.id_a IS NULL"
            ).fetchone()[0]
        ),
        "scored_pair_count": scored_count,
    }


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        self.size.setdefault(value, 1)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.size[root_left] < self.size[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.size[root_left] += self.size[root_right]

    def groups(self) -> list[list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for value in self.parent:
            grouped[self.find(value)].append(value)
        return [sorted(items) for items in grouped.values() if len(items) > 1]


def _owner_order(record: Mapping[str, Any], policy: str) -> tuple[Any, ...]:
    if record["source"] == "parlamento_pt":
        return (9, record["source"], record["subset"], record["pilot_occurrence_id"])
    if policy == "native_then_aggregator":
        native = {"carolina": 0, "gutenberg_pt": 1, "wikipedia_pt": 2}
        if record["source"] in native:
            return (
                0,
                native[record["source"]],
                record["subset"],
                record["pilot_occurrence_id"],
            )
        subset_tier = {
            "finepdfs_por_Latn": 0,
            "crawlPT_dedup": 1,
            "quati": 1,
            "blogset": 1,
            "fineweb_2_pt": 2,
            "mc4_pt": 3,
            "hplt2_pt": 3,
            "hplt1_pt": 3,
            "common_crawl": 3,
            "oscar": 3,
            "culturax": 3,
        }
        return (
            1,
            subset_tier.get(record["subset"], 5),
            record["subset"],
            record["pilot_occurrence_id"],
        )
    if policy == "longest_non_parlamento":
        return (
            -int(record["normalized_words"]),
            record["source"],
            record["subset"],
            record["pilot_occurrence_id"],
        )
    raise ValueError(f"Unknown ownership policy: {policy}")


def _cluster_metrics(
    pairs: Iterable[tuple[str, str]],
    metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[list[list[str]], dict[str, dict[str, int]]]:
    union_find = _UnionFind()
    for left, right in pairs:
        union_find.union(left, right)
    clusters = union_find.groups()
    policy_removals: dict[str, dict[str, int]] = {
        "native_then_aggregator": defaultdict(int),
        "longest_non_parlamento": defaultdict(int),
    }
    for cluster in clusters:
        non_parliament = [
            metadata[item]
            for item in cluster
            if metadata[item]["source"] != "parlamento_pt"
        ]
        if len(non_parliament) < 2:
            continue
        for policy in policy_removals:
            owner = min(non_parliament, key=lambda record: _owner_order(record, policy))
            for item in non_parliament:
                if item["pilot_occurrence_id"] != owner["pilot_occurrence_id"]:
                    policy_removals[policy][
                        _source_subset_key(item["source"], item["subset"])
                    ] += 1
    return clusters, {
        policy: dict(counts) for policy, counts in policy_removals.items()
    }


def _summary_rows(
    connection: sqlite3.Connection,
    metadata: Mapping[str, Mapping[str, Any]],
    thresholds: Sequence[float] = THRESHOLDS,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[float, list[list[str]]]]
]:
    """Summarize threshold sweeps for every LSH setup and the 5-gram union."""
    summary_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    cluster_sets: dict[str, dict[float, list[list[str]]]] = defaultdict(dict)
    all_cells = sorted(
        {_source_subset_key(row["source"], row["subset"]) for row in metadata.values()}
    )
    setup_specs: list[tuple[str, str | None]] = [
        (config.name, config.name) for config in LSH_CONFIGS
    ]
    setup_specs.append(("word5_lsh_union", None))
    for setup_name, config_name in setup_specs:
        if config_name is None:
            candidate_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM (SELECT id_a,id_b FROM candidate_by_config "
                    "WHERE config_id LIKE 'word5_%' GROUP BY id_a,id_b)"
                ).fetchone()[0]
            )
            candidate_cursor_sql = (
                "SELECT DISTINCT id_a,id_b FROM candidate_by_config "
                "WHERE config_id LIKE 'word5_%' ORDER BY id_a,id_b"
            )
            score_column = "estimate_word5_256"
            candidate_filter = (
                "SELECT DISTINCT c.id_a,c.id_b,s."
                + score_column
                + " FROM candidate_by_config c JOIN pair_scores s USING(id_a,id_b) "
                + "WHERE c.config_id LIKE 'word5_%' AND s."
                + score_column
                + ">=? "
                + "ORDER BY c.id_a,c.id_b"
            )
        else:
            candidate_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM candidate_by_config WHERE config_id=?",
                    (config_name,),
                ).fetchone()[0]
            )
            candidate_cursor_sql = "SELECT id_a,id_b FROM candidate_by_config WHERE config_id=? ORDER BY id_a,id_b"
            config = next(item for item in LSH_CONFIGS if item.name == config_name)
            score_column = (
                "estimate_word5_64"
                if config.representation == "word5_256"
                and config.num_permutations == 64
                else "estimate_word5_128"
                if config.representation == "word5_256"
                and config.num_permutations == 128
                else "estimate_word5_256"
                if config.representation == "word5_256"
                else f"estimate_{config.representation}"
            )
            candidate_filter = (
                "SELECT c.id_a,c.id_b,s."
                + score_column
                + " FROM candidate_by_config c JOIN pair_scores s USING(id_a,id_b) "
                + "WHERE c.config_id=? AND s."
                + score_column
                + ">=? "
                + "ORDER BY c.id_a,c.id_b"
            )
        candidate_endpoints: Counter[str] = Counter()
        candidate_parliament = 0
        candidate_parliament_short = 0
        candidate_iter = (
            connection.execute(candidate_cursor_sql)
            if config_name is None
            else connection.execute(candidate_cursor_sql, (config_name,))
        )
        for left, right in candidate_iter:
            candidate_endpoints[
                _source_subset_key(metadata[left]["source"], metadata[left]["subset"])
            ] += 1
            candidate_endpoints[
                _source_subset_key(metadata[right]["source"], metadata[right]["subset"])
            ] += 1
            row_a, row_b = metadata[left], metadata[right]
            if row_a["source"] == "parlamento_pt" or row_b["source"] == "parlamento_pt":
                candidate_parliament += 1
                if (
                    row_a["source"] == "parlamento_pt"
                    and row_a["normalized_words"] < 20
                ) or (
                    row_b["source"] == "parlamento_pt"
                    and row_b["normalized_words"] < 20
                ):
                    candidate_parliament_short += 1
        for threshold in thresholds:
            accepted_cursor = (
                connection.execute(candidate_filter, (threshold,))
                if config_name is None
                else connection.execute(candidate_filter, (config_name, threshold))
            )
            accepted_count = 0
            within = 0
            cross = 0
            cross_subset = 0
            parliament_pairs = 0
            parliament_short = 0
            accepted_endpoints: Counter[str] = Counter()
            accepted_incident: Counter[str] = Counter()
            union_find = _UnionFind()
            for left, right, _estimate in accepted_cursor:
                accepted_count += 1
                row_a, row_b = metadata[left], metadata[right]
                cell_a = _source_subset_key(row_a["source"], row_a["subset"])
                cell_b = _source_subset_key(row_b["source"], row_b["subset"])
                accepted_endpoints[cell_a] += 1
                accepted_endpoints[cell_b] += 1
                accepted_incident[cell_a] += 1
                if cell_b != cell_a:
                    accepted_incident[cell_b] += 1
                if row_a["source"] == row_b["source"]:
                    within += 1
                    if row_a["subset"] != row_b["subset"]:
                        cross_subset += 1
                else:
                    cross += 1
                if (
                    row_a["source"] == "parlamento_pt"
                    or row_b["source"] == "parlamento_pt"
                ):
                    parliament_pairs += 1
                    if (
                        row_a["source"] == "parlamento_pt"
                        and row_a["normalized_words"] < 20
                    ) or (
                        row_b["source"] == "parlamento_pt"
                        and row_b["normalized_words"] < 20
                    ):
                        parliament_short += 1
                union_find.union(left, right)
            clusters = union_find.groups()
            removal_counts: dict[str, Counter[str]] = {
                "native_then_aggregator": Counter(),
                "longest_non_parlamento": Counter(),
            }
            for cluster in clusters:
                participants = [metadata[item] for item in cluster]
                non_parlamento = [
                    row for row in participants if row["source"] != "parlamento_pt"
                ]
                if len(non_parlamento) < 2:
                    continue
                for policy in removal_counts:
                    owner = min(
                        non_parlamento, key=lambda row: _owner_order(row, policy)
                    )
                    for row in non_parlamento:
                        if row["pilot_occurrence_id"] != owner["pilot_occurrence_id"]:
                            removal_counts[policy][
                                _source_subset_key(row["source"], row["subset"])
                            ] += 1
            participant_ids = {item for cluster in clusters for item in cluster}
            removals_total = {
                policy: sum(counts.values())
                for policy, counts in removal_counts.items()
            }
            summary_rows.append(
                {
                    "candidate_setup": setup_name,
                    "threshold": threshold,
                    "candidate_pairs": candidate_count,
                    "accepted_near_duplicate_pairs": accepted_count,
                    "clusters": len(clusters),
                    "documents_participating": len(participant_ids),
                    "within_source_pairs": within,
                    "cross_source_pairs": cross,
                    "same_source_cross_subset_pairs": cross_subset,
                    "candidate_parlamento_involving_pairs": candidate_parliament,
                    "candidate_parlamento_short_formulaic_pairs": candidate_parliament_short,
                    "parlamento_involving_pairs": parliament_pairs,
                    "parlamento_short_formulaic_pairs": parliament_short,
                    "estimated_removable_native_then_aggregator": removals_total[
                        "native_then_aggregator"
                    ],
                    "estimated_removable_longest_non_parlamento": removals_total[
                        "longest_non_parlamento"
                    ],
                }
            )
            cluster_sets[setup_name][threshold] = clusters
            for source_subset in all_cells:
                sample_count = sum(
                    _source_subset_key(row["source"], row["subset"]) == source_subset
                    for row in metadata.values()
                )
                source_rows.append(
                    {
                        "candidate_setup": setup_name,
                        "threshold": threshold,
                        "source_subset": source_subset,
                        "sample_documents": sample_count,
                        "candidate_pair_endpoints": candidate_endpoints[source_subset],
                        "accepted_pair_endpoints": accepted_endpoints[source_subset],
                        "accepted_pairs_incident": accepted_incident[source_subset],
                        "documents_participating": sum(
                            _source_subset_key(
                                metadata[item]["source"], metadata[item]["subset"]
                            )
                            == source_subset
                            for item in participant_ids
                        ),
                        "estimated_removable_native_then_aggregator": removal_counts[
                            "native_then_aggregator"
                        ][source_subset],
                        "estimated_removable_longest_non_parlamento": removal_counts[
                            "longest_non_parlamento"
                        ][source_subset],
                    }
                )
    return summary_rows, source_rows, cluster_sets


def _write_cluster_files(
    output_dir: Path,
    cluster_sets: Mapping[str, Mapping[float, Sequence[Sequence[str]]]],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    setup: str = "word5_lsh_union",
) -> int:
    target = output_dir / "clusters_by_threshold"
    target.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("threshold", pa.float32(), nullable=False),
            pa.field("candidate_setup", pa.string(), nullable=False),
            pa.field("cluster_id", pa.string(), nullable=False),
            pa.field("pilot_occurrence_id", pa.string(), nullable=False),
            pa.field("source", pa.string(), nullable=False),
            pa.field("subset", pa.string(), nullable=False),
            pa.field("normalized_words", pa.int64(), nullable=False),
            pa.field("owner_native_then_aggregator", pa.bool_(), nullable=False),
            pa.field(
                "hypothetically_removable_native_then_aggregator",
                pa.bool_(),
                nullable=False,
            ),
            pa.field("owner_longest_non_parlamento", pa.bool_(), nullable=False),
            pa.field(
                "hypothetically_removable_longest_non_parlamento",
                pa.bool_(),
                nullable=False,
            ),
            pa.field("parlamento_preserved", pa.bool_(), nullable=False),
        ]
    )
    total_rows = 0
    for threshold, clusters in cluster_sets.get(setup, {}).items():
        path = target / f"threshold_{threshold:.2f}.parquet"
        rows: list[dict[str, Any]] = []
        for cluster in clusters:
            records = [metadata[item] for item in cluster]
            non_parliament = [
                row for row in records if row["source"] != "parlamento_pt"
            ]
            owner_native = (
                min(
                    non_parliament,
                    key=lambda item: _owner_order(item, "native_then_aggregator"),
                )["pilot_occurrence_id"]
                if non_parliament
                else None
            )
            owner_longest = (
                min(
                    non_parliament,
                    key=lambda item: _owner_order(item, "longest_non_parlamento"),
                )["pilot_occurrence_id"]
                if non_parliament
                else None
            )
            cluster_id = hashlib.sha256(
                "\0".join(sorted(cluster)).encode("utf-8")
            ).hexdigest()
            for row in records:
                parliament = row["source"] == "parlamento_pt"
                rows.append(
                    {
                        "threshold": threshold,
                        "candidate_setup": setup,
                        "cluster_id": cluster_id,
                        "pilot_occurrence_id": row["pilot_occurrence_id"],
                        "source": row["source"],
                        "subset": row["subset"],
                        "normalized_words": int(row["normalized_words"]),
                        "owner_native_then_aggregator": row["pilot_occurrence_id"]
                        == owner_native,
                        "hypothetically_removable_native_then_aggregator": not parliament
                        and row["pilot_occurrence_id"] != owner_native,
                        "owner_longest_non_parlamento": row["pilot_occurrence_id"]
                        == owner_longest,
                        "hypothetically_removable_longest_non_parlamento": not parliament
                        and row["pilot_occurrence_id"] != owner_longest,
                        "parlamento_preserved": parliament,
                    }
                )
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schema), path, compression="zstd"
        )
        total_rows += len(rows)
    return total_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _iter_word_shingles(
    text: str, ngram_size: int, seed: int
) -> Iterator[tuple[int, str]]:
    window: deque[str] = deque(maxlen=ngram_size)
    seen_word = False
    for match in TOKEN_RE.finditer(text):
        window.append(match.group(0).lower())
        seen_word = True
        if len(window) == ngram_size:
            shingle = " ".join(window)
            yield (
                xxhash.xxh64(
                    shingle.encode("utf-8"), seed=seed & 0xFFFFFFFF
                ).intdigest(),
                shingle,
            )
    if seen_word and len(window) < ngram_size:
        shingle = "SHORT\0" + " ".join(window)
        yield (
            xxhash.xxh64(shingle.encode("utf-8"), seed=seed & 0xFFFFFFFF).intdigest(),
            shingle,
        )


def _exact_review_overlap(
    text_a: str, text_b: str, cap: int = MAX_EXACT_REVIEW_SHINGLES
) -> dict[str, Any]:
    set_a: set[int] = set()
    examples_a: list[tuple[int, str]] = []
    for shingle_hash, phrase in _iter_word_shingles(text_a, 5, 20261006):
        if shingle_hash not in set_a:
            set_a.add(shingle_hash)
            if len(examples_a) < 8:
                examples_a.append((shingle_hash, phrase))
        if len(set_a) > cap:
            return {
                "exact_available": False,
                "exact_unavailable_reason": "shingle_cap_exceeded",
            }
    set_b: set[int] = set()
    examples_b: list[tuple[int, str]] = []
    common_examples: list[str] = []
    for shingle_hash, phrase in _iter_word_shingles(text_b, 5, 20261006):
        if shingle_hash not in set_b:
            set_b.add(shingle_hash)
            if len(examples_b) < 8:
                examples_b.append((shingle_hash, phrase))
        if (
            shingle_hash in set_a
            and phrase not in common_examples
            and len(common_examples) < 3
        ):
            common_examples.append(phrase.replace("SHORT\0", ""))
        if len(set_b) > cap:
            return {
                "exact_available": False,
                "exact_unavailable_reason": "shingle_cap_exceeded",
            }
    shared = len(set_a & set_b)
    union = len(set_a | set_b)
    distinct_a = [phrase for value, phrase in examples_a if value not in set_b][:3]
    distinct_b = [phrase for value, phrase in examples_b if value not in set_a][:3]
    min_size = min(len(set_a), len(set_b))
    return {
        "exact_available": True,
        "unique_shingles_a": len(set_a),
        "unique_shingles_b": len(set_b),
        "shared_shingles": shared,
        "union_shingles": union,
        "exact_jaccard": shared / union if union else 0.0,
        "containment": shared / min_size if min_size else 0.0,
        "common_shingle_examples": common_examples,
        "differing_shingle_examples_a": distinct_a,
        "differing_shingle_examples_b": distinct_b,
    }


def _excerpt_around(text: str, phrase: str | None, width: int = 260) -> str:
    if phrase:
        match = re.search(re.escape(phrase), text, flags=re.IGNORECASE)
        if match:
            start = max(0, match.start() - width // 3)
            end = min(len(text), match.end() + width // 2)
            excerpt = text[start:end]
        else:
            excerpt = text[:width]
    else:
        excerpt = text[:width]
    return re.sub(r"\s+", " ", excerpt).strip()[:width]


def _review_categories(
    row_a: Mapping[str, Any],
    row_b: Mapping[str, Any],
    reasons: Sequence[str],
    estimate: float,
) -> list[str]:
    sources = {row_a["source"], row_b["source"]}
    domains = {_domain(row_a.get("original_url")), _domain(row_b.get("original_url"))}
    categories: list[str] = []
    if "parlamento_pt" in sources and (
        (row_a["source"] == "parlamento_pt" and row_a["normalized_words"] < 20)
        or (row_b["source"] == "parlamento_pt" and row_b["normalized_words"] < 20)
    ):
        categories.append("parlamento_short_formulaic")
    if sources == {"carolina", "wikipedia_pt"}:
        categories.append("wikipedia_carolina")
    if row_a["source"] == row_b["source"] == "gigaverbo_v2":
        categories.append("web_web")
    curated = {"carolina", "wikipedia_pt", "gutenberg_pt", "parlamento_pt"}
    if (row_a["source"] in curated and row_b["source"] == "gigaverbo_v2") or (
        row_b["source"] in curated and row_a["source"] == "gigaverbo_v2"
    ):
        categories.append("curated_web")
    if "same_url_domain" in reasons or (len(domains) == 1 and "" not in domains):
        categories.append("same_url_domain")
    title_a = (row_a.get("title") or "").casefold().strip()
    title_b = (row_b.get("title") or "").casefold().strip()
    if title_a and title_a == title_b:
        categories.append("same_title_or_syndication")
        if domains and len(domains) > 1:
            categories.append("syndicated_article_candidate")
    if "neighboring_crawl_records" in reasons:
        categories.append("neighboring_crawl_records")
    if row_a["normalized_words"] < 20 or row_b["normalized_words"] < 20:
        categories.append("short_text")
    if row_a["normalized_words"] >= 100_000 or row_b["normalized_words"] >= 100_000:
        categories.append("giant_text")
    elif row_a["normalized_words"] >= 5_000 or row_b["normalized_words"] >= 5_000:
        categories.append("long_text")
    if "parlamento_exact_hash_relative" in reasons:
        categories.append("parlamento_repeated_exact_phrase")
    if estimate >= 0.90:
        categories.append("high_similarity_manual_review")
    return sorted(set(categories))


def _select_review_pairs(
    connection: sqlite3.Connection,
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    pools: dict[tuple[str, str], _BottomK] = {}
    for row in connection.execute(
        "SELECT id_a,id_b,generators,enrichment_reasons,estimate_word5_256,lsh_generated "
        "FROM pair_scores ORDER BY id_a,id_b"
    ):
        id_a, id_b, _generators, reasons_text, estimate, lsh_generated = row
        estimate = float(estimate)
        reasons = [item for item in reasons_text.split(",") if item]
        row_a, row_b = metadata[id_a], metadata[id_b]
        band = next(
            (
                f"{lower:.2f}-{upper:.2f}"
                for lower, upper in REVIEW_BANDS
                if lower <= estimate < upper
            ),
            "under_0.75",
        )
        categories = _review_categories(row_a, row_b, reasons, estimate)
        pair = {
            "id_a": id_a,
            "id_b": id_b,
            "similarity_band": band,
            "estimate_word5_256": estimate,
            "lsh_generated": bool(lsh_generated),
            "enrichment_reasons": reasons,
            "review_categories": categories,
        }
        pair_key = f"{id_a}\0{id_b}"
        labels = categories or ["uncategorized"]
        for label in labels:
            key = (band, label)
            pool = pools.setdefault(key, _BottomK(2, seed ^ stable_hash64(label)))
            pool.add(pair_key, pair)
        overall = pools.setdefault((band, "_overall"), _BottomK(5, seed ^ 0xA11))
        overall.add(pair_key, pair)
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for (band, label), pool in sorted(pools.items()):
        for pair in pool.values():
            selected[(pair["id_a"], pair["id_b"])] = pair
    by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in selected.values():
        by_band[pair["similarity_band"]].append(pair)
    result: list[dict[str, Any]] = []
    for band, items in sorted(by_band.items()):
        priority = {
            "parlamento_short_formulaic": 0,
            "wikipedia_carolina": 1,
            "syndicated_article_candidate": 2,
            "same_url_domain": 3,
            "neighboring_crawl_records": 4,
            "curated_web": 5,
            "web_web": 6,
            "short_text": 7,
            "long_text": 8,
            "giant_text": 9,
        }
        items.sort(
            key=lambda pair: (
                min(
                    (priority.get(item, 50) for item in pair["review_categories"]),
                    default=99,
                ),
                pair["id_a"],
                pair["id_b"],
            )
        )
        result.extend(items[:24])
    return result


def _read_pair_texts(
    parquet: pq.ParquetFile,
    metadata: Mapping[str, Mapping[str, Any]],
    id_a: str,
    id_b: str,
) -> tuple[str, str]:
    offsets: list[tuple[int, int, str]] = []
    cursor = 0
    for row_group in range(parquet.metadata.num_row_groups):
        count = parquet.metadata.row_group(row_group).num_rows
        offsets.append((cursor, cursor + count, str(row_group)))
        cursor += count
    wanted = {id_a, id_b}
    texts: dict[str, str] = {}
    groups_needed: set[int] = set()
    for pilot_id in wanted:
        ordinal = int(metadata[pilot_id]["pilot_file_row_ordinal"])
        for start, end, group in offsets:
            if start <= ordinal < end:
                groups_needed.add(int(group))
                break
    for row_group in sorted(groups_needed):
        for batch in parquet.iter_batches(
            row_groups=[row_group],
            batch_size=16,
            columns=["pilot_occurrence_id", "text"],
        ):
            ids = batch.column(0).to_pylist()
            values = batch.column(1).to_pylist()
            for pilot_id, text_value in zip(ids, values):
                if pilot_id in wanted:
                    texts[pilot_id] = text_value or ""
    if set(texts) != wanted:
        raise ValueError("Could not read selected review text from pilot records")
    return texts[id_a], texts[id_b]


def _write_review_pairs(
    pilot_records_path: Path,
    connection: sqlite3.Connection,
    metadata: Mapping[str, Mapping[str, Any]],
    output_path: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    selected = _select_review_pairs(connection, metadata, seed=seed)
    parquet = pq.ParquetFile(pilot_records_path)
    output: list[dict[str, Any]] = []
    for pair in selected:
        id_a, id_b = pair["id_a"], pair["id_b"]
        row_a, row_b = metadata[id_a], metadata[id_b]
        text_a, text_b = _read_pair_texts(parquet, metadata, id_a, id_b)
        overlap = _exact_review_overlap(text_a, text_b)
        common = overlap.get("common_shingle_examples", [])
        common_phrase = common[0].replace("SHORT\0", "") if common else None
        excerpt_a = _excerpt_around(text_a, common_phrase)
        excerpt_b = _excerpt_around(text_b, common_phrase)
        categories = list(pair["review_categories"])
        if overlap.get("exact_available") and float(overlap["exact_jaccard"]) >= 0.80:
            if text_a[:240] != text_b[:240] or text_a[-240:] != text_b[-240:]:
                categories.append("boundary_variant_candidate")
        if any(
            marker in text_a.casefold() or marker in text_b.casefold()
            for marker in (
                "política de privacidade",
                "todos os direitos reservados",
                "aceitar cookies",
                "termos de uso",
            )
        ):
            categories.append("boilerplate_candidate")
        output.append(
            {
                "id_a": id_a,
                "id_b": id_b,
                "similarity_band": pair["similarity_band"],
                "estimated_jaccard_word5_256": pair["estimate_word5_256"],
                "exact_jaccard_word5": overlap.get("exact_jaccard"),
                "exact_overlap_available": bool(overlap.get("exact_available")),
                "exact_unavailable_reason": overlap.get("exact_unavailable_reason"),
                "unique_shingles_a": overlap.get("unique_shingles_a"),
                "unique_shingles_b": overlap.get("unique_shingles_b"),
                "shared_shingles": overlap.get("shared_shingles"),
                "union_shingles": overlap.get("union_shingles"),
                "containment_of_smaller_shingle_set": overlap.get("containment"),
                "common_shingle_examples": common,
                "differing_shingle_examples_a": overlap.get(
                    "differing_shingle_examples_a", []
                ),
                "differing_shingle_examples_b": overlap.get(
                    "differing_shingle_examples_b", []
                ),
                "source_a": row_a["source"],
                "subset_a": row_a["subset"],
                "source_b": row_b["source"],
                "subset_b": row_b["subset"],
                "same_source": row_a["source"] == row_b["source"],
                "same_source_cross_subset": row_a["source"] == row_b["source"]
                and row_a["subset"] != row_b["subset"],
                "cross_source": row_a["source"] != row_b["source"],
                "occurrence_id_a": id_a,
                "occurrence_id_b": id_b,
                "normalized_words_a": int(row_a["normalized_words"]),
                "normalized_words_b": int(row_b["normalized_words"]),
                "original_url_a": row_a.get("original_url"),
                "original_url_b": row_b.get("original_url"),
                "title_a": row_a.get("title"),
                "title_b": row_b.get("title"),
                "review_categories": sorted(set(categories)),
                "lsh_generated": pair["lsh_generated"],
                "enrichment_reasons": pair["enrichment_reasons"],
                "excerpt_a": excerpt_a,
                "excerpt_b": excerpt_b,
            }
        )
    table = pa.Table.from_pylist(output)
    pq.write_table(table, output_path, compression="zstd")
    per_band = Counter(item["similarity_band"] for item in output)
    return {
        "review_pair_count": len(output),
        "review_pairs_by_band": dict(sorted(per_band.items())),
        "review_categories": dict(
            sorted(
                Counter(
                    category
                    for item in output
                    for category in item["review_categories"]
                ).items()
            )
        ),
        "pairs_with_exact_jaccard": sum(
            item["exact_overlap_available"] for item in output
        ),
        "pairs_unavailable_due_to_shingle_cap": sum(
            item["exact_unavailable_reason"] == "shingle_cap_exceeded"
            for item in output
        ),
    }


def _synthetic_cases() -> list[dict[str, Any]]:
    def tokens(prefix: str, start: int, count: int) -> str:
        return " ".join(f"{prefix}{index:04d}" for index in range(start, start + count))

    paragraph_a = tokens("alpha", 0, 120)
    paragraph_b = tokens("beta", 0, 120)
    paragraph_c = tokens("gamma", 0, 120)
    base_article = f"{paragraph_a}\n\n{paragraph_b}\n\n{paragraph_c}"
    article_changed = f"{paragraph_a}\n\n{tokens('changed', 0, 12)} {tokens('beta', 12, 108)}\n\n{paragraph_c}"
    reordered = f"{paragraph_c}\n\n{paragraph_a}\n\n{paragraph_b}"
    header_footer = (
        f"Portal informativo atualizado. {base_article} Consulte nosso arquivo digital."
    )
    header_footer_changed = (
        f"Página institucional. {base_article} Consulte os demais conteúdos."
    )
    template = tokens("template", 0, 600)
    template_body_a = template + " " + tokens("bodya", 0, 20)
    template_body_b = template + " " + tokens("bodyb", 0, 20)
    containment_small = f"{tokens('copied', 0, 160)}"
    containment_large = f"{tokens('outside', 0, 1_400)}\n\n{containment_small}\n\n{tokens('tail', 0, 1_200)}"
    long_overlap_a = f"{tokens('left', 0, 1_100)}\n\n{tokens('fragment', 0, 90)}"
    long_overlap_b = f"{tokens('right', 0, 1_100)}\n\n{tokens('fragment', 0, 90)}"
    parliament_a = (
        "O Senhor Presidente declarou aberta a sessão da Assembleia da República."
    )
    parliament_b = (
        "O Senhor Presidente declarou aberta a sessão da Assembleia da República."
    )
    short_generic_a = "A sessão foi aberta."
    short_generic_b = "A sessão foi iniciada."
    parliament_variant_a = "O Senhor Presidente informa que está aberta a sessão da Assembleia da República hoje."
    parliament_variant_b = "O Senhor Presidente informa que está suspensa a sessão da Assembleia da República hoje."
    return [
        {
            "case_id": "header_footer_tiny_change",
            "text_a": header_footer,
            "text_b": header_footer_changed,
            "expected_near_duplicate": True,
            "unit_level_note": "same body with small boundary text changes",
        },
        {
            "case_id": "same_article_one_paragraph_changed",
            "text_a": base_article,
            "text_b": article_changed,
            "expected_near_duplicate": True,
            "unit_level_note": "one paragraph has limited edits",
        },
        {
            "case_id": "same_template_unrelated_body",
            "text_a": template_body_a,
            "text_b": template_body_b,
            "expected_near_duplicate": False,
            "unit_level_note": "template overlap must not define document identity",
        },
        {
            "case_id": "short_generic_sentence_edit",
            "text_a": short_generic_a,
            "text_b": short_generic_b,
            "expected_near_duplicate": True,
            "unit_level_note": "near-copy short sentence",
        },
        {
            "case_id": "parlamento_repeated_procedural_occurrence",
            "text_a": parliament_a,
            "text_b": parliament_b,
            "expected_near_duplicate": False,
            "unit_level_note": "same procedural utterance at distinct occurrences; preserve ParlamentoPT",
        },
        {
            "case_id": "parlamento_formula_variant",
            "text_a": parliament_variant_a,
            "text_b": parliament_variant_b,
            "expected_near_duplicate": False,
            "unit_level_note": "procedural frame with different event meaning",
        },
        {
            "case_id": "paragraph_order_changed",
            "text_a": base_article,
            "text_b": reordered,
            "expected_near_duplicate": True,
            "unit_level_note": "word-shingle set ignores paragraph order",
        },
        {
            "case_id": "long_document_small_fragment",
            "text_a": long_overlap_a,
            "text_b": long_overlap_b,
            "expected_near_duplicate": False,
            "unit_level_note": "small shared passage is not whole-document duplication",
        },
        {
            "case_id": "document_contained_in_larger_document",
            "text_a": containment_small,
            "text_b": containment_large,
            "expected_near_duplicate": False,
            "unit_level_note": "containment requires separate handling",
        },
    ]


def _synthetic_candidate_set(
    signatures: Mapping[str, Mapping[str, Sequence[int]]], config: LSHConfig
) -> set[tuple[str, str]]:
    buckets: dict[tuple[int, bytes], list[str]] = defaultdict(list)
    for pilot_id, reps in signatures.items():
        signature = reps[config.representation][: config.num_permutations]
        for band in range(config.bands):
            key = _band_key(
                signature,
                band * config.rows_per_band,
                config.rows_per_band,
                config.name,
                band,
            )
            buckets[(band, key)].append(pilot_id)
    pairs: set[tuple[str, str]] = set()
    for members in buckets.values():
        for left, right in itertools.combinations(sorted(members), 2):
            pairs.add((left, right))
    return pairs


def _run_synthetic_validation(output_path: Path, *, seed: int) -> dict[str, Any]:
    cases = _synthetic_cases()
    docs: dict[str, str] = {}
    for case in cases:
        docs[f"{case['case_id']}:a"] = case["text_a"]
        docs[f"{case['case_id']}:b"] = case["text_b"]
    configs = tuple(
        MinHashConfig(item.name, item.ngram_size, item.num_permutations, seed)
        for item in SIGNATURE_CONFIGS
    )
    reps: dict[str, dict[str, list[int]]] = {}
    for pilot_id, text in docs.items():
        reps[pilot_id] = {
            config.name: compute_minhash_signature(text, config) for config in configs
        }
    exact_by_case = {
        case["case_id"]: _exact_review_overlap(case["text_a"], case["text_b"])
        for case in cases
    }
    rows: list[dict[str, Any]] = []
    for lsh_config in LSH_CONFIGS:
        candidates = _synthetic_candidate_set(reps, lsh_config)
        for case in cases:
            id_a = f"{case['case_id']}:a"
            id_b = f"{case['case_id']}:b"
            pair = tuple(sorted((id_a, id_b)))
            estimate = estimate_jaccard_signature(
                reps[id_a][lsh_config.representation][: lsh_config.num_permutations],
                reps[id_b][lsh_config.representation][: lsh_config.num_permutations],
            )
            exact = exact_by_case[case["case_id"]]
            for threshold in THRESHOLDS:
                candidate = pair in candidates
                detected = candidate and estimate >= threshold
                expected = bool(case["expected_near_duplicate"])
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "lsh_configuration": lsh_config.name,
                        "representation": lsh_config.representation,
                        "threshold": threshold,
                        "expected_near_duplicate": expected,
                        "exact_jaccard_word5": exact.get("exact_jaccard"),
                        "exact_containment": exact.get("containment"),
                        "estimated_jaccard": estimate,
                        "lsh_candidate": candidate,
                        "detected_by_lsh_and_threshold": detected,
                        "false_positive": detected and not expected,
                        "false_negative": expected and not detected,
                        "parlamento_preserve_all_applies": case["case_id"].startswith(
                            "parlamento_"
                        ),
                        "unit_level_note": case["unit_level_note"],
                    }
                )
    pq.write_table(pa.Table.from_pylist(rows), output_path, compression="zstd")
    aggregate = {
        "synthetic_case_count": len(cases),
        "configurations_tested": len(LSH_CONFIGS),
        "thresholds_tested": list(THRESHOLDS),
        "false_positive_decisions": sum(row["false_positive"] for row in rows),
        "false_negative_decisions": sum(row["false_negative"] for row in rows),
        "case_results": {
            case["case_id"]: {
                "expected_near_duplicate": case["expected_near_duplicate"],
                "exact_word5_jaccard": exact_by_case[case["case_id"]].get(
                    "exact_jaccard"
                ),
                "containment": exact_by_case[case["case_id"]].get("containment"),
                "exact_overlap_available": exact_by_case[case["case_id"]].get(
                    "exact_available"
                ),
                "note": case["unit_level_note"],
            }
            for case in cases
        },
    }
    return aggregate


EXPECTED_EXACT_MANIFEST_SHA256 = (
    "57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428"
)
POST_EXACT_RECORD_COUNT = 21_603_689
POST_EXACT_WORD_COUNT = 21_470_091_017


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024 if os.name == "posix" else value)


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_exact_root(input_root: Path) -> tuple[Path, Path, Path, dict[str, Any], str]:
    resolved = input_root.resolve()
    if resolved.name == "data" and (resolved / "carolina").is_dir():
        exact_root = resolved.parent
        data_root = resolved
    else:
        exact_root = resolved
        data_root = exact_root / "data"
    manifest_path = exact_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing exact-dedup manifest: {manifest_path}")
    manifest_sha = compute_file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("run_type") != "production":
        raise ValueError(
            "Near-dedup pilot requires a COMPLETE exact production manifest"
        )
    if not data_root.is_dir():
        raise FileNotFoundError(f"Missing exact-deduplicated data: {data_root}")
    return exact_root, data_root, manifest_path, manifest, manifest_sha


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        inventory.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": compute_file_sha256(path),
            }
        )
    return inventory


def verify_near_pilot_manifest(output_root: Path | str) -> list[str]:
    """Check all output sizes and checksums recorded by a completed pilot."""
    root = Path(output_root)
    path = root / "manifest.json"
    if not path.is_file():
        return ["manifest.json is missing"]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"manifest.json cannot be read: {exc}"]
    errors: list[str] = []
    if manifest.get("status") != "COMPLETE":
        errors.append("pilot manifest status is not COMPLETE")
    seen: set[str] = set()
    for artifact in manifest.get("artifacts", []):
        relative = str(artifact.get("path") or "")
        artifact_path = (root / relative).resolve()
        if not relative or root.resolve() not in artifact_path.parents:
            errors.append(f"unsafe artifact path: {relative!r}")
            continue
        if relative in seen:
            errors.append(f"duplicate artifact entry: {relative}")
            continue
        seen.add(relative)
        if not artifact_path.is_file():
            errors.append(f"artifact is missing: {relative}")
            continue
        if artifact_path.stat().st_size != int(artifact.get("bytes", -1)):
            errors.append(f"artifact size mismatch: {relative}")
        if compute_file_sha256(artifact_path) != artifact.get("sha256"):
            errors.append(f"artifact checksum mismatch: {relative}")
    return errors


def _production_estimate(
    *,
    scan_metrics: Mapping[str, Any],
    signature_metrics: Mapping[str, Any],
    lsh_metrics: Mapping[str, Any],
    metadata: Mapping[str, Mapping[str, Any]],
    signature_file_bytes: int,
    base_candidate_pairs: int,
) -> dict[str, Any]:
    frame_counts: dict[str, int] = scan_metrics["frame_counts"]
    frame_words: dict[str, int] = scan_metrics["frame_words"]
    seconds_by_cell = signature_metrics["cpu_seconds_by_cell_and_role"].get(
        "word5_256", {}
    )
    words_by_cell = signature_metrics["words_by_cell_and_role"].get("word5_256", {})
    records_by_cell = signature_metrics["records_by_cell_and_role"].get("word5_256", {})
    projected_256_word_scaled = 0.0
    projected_256_record_scaled = 0.0
    covered_words = 0
    for key, seconds in seconds_by_cell.items():
        if not key.endswith("/stratified_base"):
            continue
        cell_role = key[: -len("/stratified_base")]
        cell, band = cell_role.rsplit("/", 1)
        frame_key = f"{cell}/{band}"
        sampled_words = int(words_by_cell.get(key, 0))
        sampled_records = int(records_by_cell.get(key, 0))
        population_words = int(frame_words.get(frame_key, 0))
        population_records = int(frame_counts.get(frame_key, 0))
        if sampled_words:
            projected_256_word_scaled += (
                float(seconds) * population_words / sampled_words
            )
            covered_words += population_words
        if sampled_records:
            projected_256_record_scaled += (
                float(seconds) * population_records / sampled_records
            )
    # The two scalings bound different per-document/per-word cost profiles.
    low_256 = min(projected_256_word_scaled, projected_256_record_scaled)
    high_256 = max(projected_256_word_scaled, projected_256_record_scaled)
    sample_count = len(metadata)
    signatures_per_row_raw = (128 + 256 + 128) * 8
    measured_compression = (
        signature_file_bytes / (sample_count * signatures_per_row_raw)
        if sample_count
        else 1.0
    )
    candidate_rate = base_candidate_pairs / max(
        1,
        sum(row["selection_role"] == "stratified_base" for row in metadata.values()),
    )
    lsh_entries = int(lsh_metrics["bucket_index_entries"])
    bytes_per_entry = float(lsh_metrics["bucket_database_bytes"]) / max(1, lsh_entries)
    baseline_bands = 16
    projected_index_bytes = bytes_per_entry * POST_EXACT_RECORD_COUNT * baseline_bands
    return {
        "classification": "extrapolated_scenarios_not_production_measurements",
        "exact_retained_documents": POST_EXACT_RECORD_COUNT,
        "exact_retained_normalized_words": POST_EXACT_WORD_COUNT,
        "signature_strategy": "word-level 5-gram, 128 permutations, 64-bit signature values",
        "signature_bytes_raw_for_5gram_128": POST_EXACT_RECORD_COUNT * 128 * 8,
        "signature_parquet_bytes_estimate_for_5gram_128": int(
            POST_EXACT_RECORD_COUNT * 128 * 8 * measured_compression
        ),
        "measured_signature_parquet_compression_ratio_for_all_three_representations": measured_compression,
        "word5_256_cpu_seconds_projection_word_scaled": round(
            projected_256_word_scaled, 1
        ),
        "word5_256_cpu_seconds_projection_record_scaled": round(
            projected_256_record_scaled, 1
        ),
        "word5_128_cpu_hours_scenario_range": [
            round(low_256 * 0.50 / 3600, 1),
            round(high_256 * 0.80 / 3600, 1),
        ],
        "projection_coverage_normalized_words": covered_words,
        "lsh_working_index_bytes_per_pilot_bucket_entry_including_sqlite_overhead": round(
            bytes_per_entry, 2
        ),
        "word5_128_16_band_sqlite_index_bytes_scenario": [
            int(projected_index_bytes * 0.75),
            int(projected_index_bytes * 1.50),
        ],
        "base_sample_word5_candidate_pairs": base_candidate_pairs,
        "base_sample_candidate_pairs_per_1000_documents": round(
            candidate_rate * 1_000, 3
        ),
        "linear_candidate_pair_count_scenario_low": int(
            candidate_rate * POST_EXACT_RECORD_COUNT
        ),
        "linear_candidate_pair_count_scenario_high_x10": int(
            candidate_rate * POST_EXACT_RECORD_COUNT * 10
        ),
        "candidate_pair_extrapolation_warning": (
            "The pilot is deliberately stratified and diagnostically enriched. Pair counts and mega-bucket tails are not population estimates; the linear and x10 figures are sensitivity scenarios only."
        ),
        "full_retained_frame_metadata_enumeration_measured": True,
        "exact_data_total_file_footprint_bytes": int(scan_metrics["input_data_bytes"]),
        "resolution_projection_compressed_bytes_measured": int(
            scan_metrics["resolution_projection_compressed_bytes"]
        ),
        "exact_data_metadata_projection_compressed_bytes_measured": int(
            scan_metrics["exact_data_metadata_projection_compressed_bytes"]
        ),
    }


def run_near_dedup_pilot(
    *,
    input_root: Path | str = DEFAULT_EXACT_DATA_ROOT,
    output_root: Path | str = DEFAULT_NEAR_PILOT_ROOT,
    seed: int = DEFAULT_NEAR_PILOT_SEED,
    quotas: Mapping[str, int] | None = None,
    require_full_coverage: bool = True,
    expected_manifest_sha256: str | None = EXPECTED_EXACT_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Build, score and document one deterministic pilot without deleting data."""
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    requested_input = Path(input_root)
    exact_root, data_root, _manifest_path, exact_manifest, exact_manifest_sha = (
        _load_exact_root(requested_input)
    )
    if expected_manifest_sha256 and exact_manifest_sha != expected_manifest_sha256:
        raise ValueError(
            "Exact input manifest SHA-256 differs from the authoritative Gate C1 value: "
            f"{exact_manifest_sha}"
        )
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing near-dedup pilot: {output}"
        )
    if (
        output == exact_root
        or exact_root in output.parents
        or output in data_root.parents
    ):
        raise ValueError("Pilot output must be separate from the exact production tree")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    connection: sqlite3.Connection | None = None
    try:
        files = _iter_data_files(exact_root)
        (
            selected,
            enrichment_pairs,
            subset_counts,
            subset_words,
            frame_counts,
            frame_words,
            scan_metrics,
        ) = _scan_and_select(
            exact_root,
            files,
            seed=seed,
            quotas=dict(quotas or LENGTH_QUOTAS),
            require_full_coverage=require_full_coverage,
        )
        expected_records = exact_manifest.get("retained_record_count")
        if expected_records is not None and scan_metrics["records_scanned"] != int(
            expected_records
        ):
            raise ValueError(
                "Exact retained-data row count differs from its manifest: "
                f"{scan_metrics['records_scanned']} != {expected_records}"
            )
        if (
            require_full_coverage
            and scan_metrics["records_scanned"] != POST_EXACT_RECORD_COUNT
        ):
            raise ValueError(
                "Exact corpus row count differs from the supplied Gate C1 baseline"
            )

        pilot_records_path = stage / "pilot_records.parquet"
        sample_record_count, pilot_data_bytes = _materialize_pilot_records(
            exact_root, files, selected, pilot_records_path
        )
        signature_path = stage / "minhash_signatures.parquet"
        metadata, signatures, signature_metrics = _write_signatures(
            pilot_records_path, signature_path, seed=seed
        )
        db_path = stage / ".near-pilot-work.sqlite3"
        connection = _create_lsh_database(db_path)
        lsh_metrics = _run_lsh(connection, signatures, seed=seed)
        pair_metrics = _write_pair_artifacts(
            connection,
            metadata,
            signatures,
            enrichment_pairs,
            stage / "candidate_pairs.parquet",
            stage / "scored_pairs.parquet",
        )
        summary_rows, source_rows, cluster_sets = _summary_rows(connection, metadata)
        sample_docs_by_cell_role: Counter[tuple[str, str]] = Counter()
        sample_words_by_cell_role: Counter[tuple[str, str]] = Counter()
        sample_by_cell_band_role: Counter[tuple[str, str, str, str]] = Counter()
        for row in metadata.values():
            cell = _source_subset_key(row["source"], row["subset"])
            role = row["selection_role"]
            sample_docs_by_cell_role[(cell, role)] += 1
            sample_words_by_cell_role[(cell, role)] += int(row["normalized_words"])
            sample_by_cell_band_role[
                (row["source"], row["subset"], row["length_stratum"], role)
            ] += 1
        enriched_source_rows: list[dict[str, Any]] = []
        for row in source_rows:
            source, subset = row["source_subset"].split("/", 1)
            row = dict(row)
            row["population_documents_frame"] = sum(
                count
                for (frame_source, frame_subset, _band), count in frame_counts.items()
                if frame_source == source and frame_subset == subset
            )
            row["population_words_frame"] = sum(
                count
                for (frame_source, frame_subset, _band), count in frame_words.items()
                if frame_source == source and frame_subset == subset
            )
            row["stratified_base_sample_documents"] = sample_docs_by_cell_role[
                (row["source_subset"], "stratified_base")
            ]
            row["diagnostic_enrichment_documents"] = sample_docs_by_cell_role[
                (row["source_subset"], "diagnostic_enrichment")
            ]
            row["stratified_base_sample_words"] = sample_words_by_cell_role[
                (row["source_subset"], "stratified_base")
            ]
            row["population_rate_inference"] = (
                "not reported; quotas intentionally oversample strata"
            )
            enriched_source_rows.append(row)
        _write_csv(stage / "threshold_summary.csv", summary_rows)
        _write_csv(stage / "source_subset_summary.csv", enriched_source_rows)
        cluster_member_count = _write_cluster_files(stage, cluster_sets, metadata)
        review_metrics = _write_review_pairs(
            pilot_records_path,
            connection,
            metadata,
            stage / "review_pairs.parquet",
            seed=seed,
        )
        synthetic_metrics = _run_synthetic_validation(
            stage / "synthetic_validation.parquet", seed=seed
        )

        base_candidate_pairs = 0
        for left, right in connection.execute(
            "SELECT id_a,id_b FROM candidate_union "
            "WHERE generators LIKE 'word5_%' OR generators LIKE '%,word5_%' "
            "ORDER BY id_a,id_b"
        ):
            if (
                metadata[left]["selection_role"] == "stratified_base"
                and metadata[right]["selection_role"] == "stratified_base"
            ):
                base_candidate_pairs += 1
        connection.close()
        connection = None
        db_path.unlink(missing_ok=True)

        composition: dict[str, int] = Counter()
        composition_words: dict[str, int] = Counter()
        for row in metadata.values():
            key = "/".join(
                (
                    row["source"],
                    row["subset"],
                    row["length_stratum"],
                    row["selection_role"],
                )
            )
            composition[key] += 1
            composition_words[key] += int(row["normalized_words"])
        sample_profile = {
            key: {
                "documents": composition[key],
                "normalized_words": composition_words[key],
            }
            for key in sorted(composition)
        }
        parliament_rows = sum(
            row["source"] == "parlamento_pt" for row in metadata.values()
        )
        parliament_deleted = sum(
            row["source"] == "parlamento_pt" and row["selection_role"] == "removed"
            for row in metadata.values()
        )
        if parliament_deleted:
            raise AssertionError("ParlamentoPT rows must never be marked for deletion")

        total_wall = time.perf_counter() - started_wall
        total_cpu = time.process_time() - started_cpu
        report = {
            "measurement_scope": "measured pilot process unless labeled extrapolated",
            "wall_seconds_total": round(total_wall, 3),
            "cpu_seconds_total": round(total_cpu, 3),
            "peak_rss_bytes": _peak_rss_bytes(),
            "input_data_files": len(files),
            "input_data_compressed_bytes": scan_metrics["input_data_bytes"],
            "scan": scan_metrics,
            "sample_record_count": sample_record_count,
            "sample_data_parquet_bytes": pilot_data_bytes,
            "signature_file_bytes": signature_path.stat().st_size,
            "signatures": signature_metrics,
            "lsh": lsh_metrics,
            "pairs": pair_metrics,
            "review": review_metrics,
            "base_sample_lsh_candidate_pairs_for_word5_sweep": base_candidate_pairs,
            "disk_footprint_bytes_before_report_and_manifest": sum(
                path.stat().st_size for path in stage.rglob("*") if path.is_file()
            ),
        }
        report["production_resource_scenarios"] = _production_estimate(
            scan_metrics=scan_metrics,
            signature_metrics=signature_metrics,
            lsh_metrics=lsh_metrics,
            metadata=metadata,
            signature_file_bytes=signature_path.stat().st_size,
            base_candidate_pairs=base_candidate_pairs,
        )
        _json_write(stage / "resource_report.json", report)
        artifact_bytes = sum(
            path.stat().st_size
            for path in stage.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        )
        report["disk_footprint_bytes_before_manifest"] = artifact_bytes
        _json_write(stage / "resource_report.json", report)
        sample_stratum_profile = {
            f"{source}/{subset}/{band}": {
                "population_documents": frame_counts.get((source, subset, band), 0),
                "population_words": frame_words.get((source, subset, band), 0),
                "stratified_base_sample_documents": sample_by_cell_band_role.get(
                    (source, subset, band, "stratified_base"), 0
                ),
                "diagnostic_enrichment_documents": sample_by_cell_band_role.get(
                    (source, subset, band, "diagnostic_enrichment"), 0
                ),
            }
            for source, subset, band in sorted(
                set(frame_counts)
                | {
                    (row["source"], row["subset"], row["length_stratum"])
                    for row in metadata.values()
                }
            )
        }
        artifacts = _artifact_inventory(stage)
        manifest = {
            "schema_version": 1,
            "near_dedup_pilot_version": NEAR_PILOT_VERSION,
            "status": "COMPLETE",
            "gate": "C1",
            "task": "D2 representative near-dedup pilot",
            "input_exact_root": str(exact_root),
            "input_exact_data_root": str(data_root),
            "input_exact_manifest_sha256": exact_manifest_sha,
            "input_exact_dedup_version": exact_manifest.get("exact_dedup_version"),
            "input_retained_records": scan_metrics["records_scanned"],
            "input_exact_retained_words_reference": POST_EXACT_WORD_COUNT,
            "input_data_files": len(files),
            "input_data_bytes": scan_metrics["input_data_bytes"],
            "seed": seed,
            "sample_design": {
                "frame": "all retained exact/data rows; per-cell seeded circular sample over SHA-256 record IDs in the exact resolution sidecar's source/subset/record-ID order",
                "length_strata": {
                    "short": "0-19 words",
                    "medium": "20-999 words",
                    "long": "1,000-99,999 words",
                    "giant": "at least 100,000 words",
                },
                "quotas_per_nonempty_source_subset_stratum": dict(
                    quotas or LENGTH_QUOTAS
                ),
                "population_rate_estimates": False,
                "base_sample_profile": sample_profile,
                "stratum_population_and_sample_counts": sample_stratum_profile,
                "diagnostic_enrichment_pair_count": len(enrichment_pairs),
                "enrichment_reasons": dict(
                    sorted(Counter(pair["reason"] for pair in enrichment_pairs).items())
                ),
            },
            "document_representations": [
                {
                    "name": item.name,
                    "unit": "lowercased Unicode word shingles from Python re \\w+ tokens",
                    "ngram_size": item.ngram_size,
                    "num_permutations": item.num_permutations,
                    "permutation_family": "stable 64-bit affine hashes over stable xxHash64 shingle hashes",
                    "short_document_rule": "fewer than n words becomes one whole-document shingle; empty text gets the all-max sentinel signature",
                }
                for item in SIGNATURE_CONFIGS
            ],
            "lsh_configurations": [
                {
                    "name": item.name,
                    "representation": item.representation,
                    "permutations": item.num_permutations,
                    "bands": item.bands,
                    "rows_per_band": item.rows_per_band,
                }
                for item in LSH_CONFIGS
            ],
            "thresholds_evaluated": list(THRESHOLDS),
            "ownership": {
                "frozen": False,
                "hypotheses": ["native_then_aggregator", "longest_non_parlamento"],
                "parlamento_policy": "all sampled ParlamentoPT records remain preserved under every hypothetical policy",
                "transitive_cluster_warning": "Connected components can chain pairs that are not mutually similar; review cluster members before any ownership action.",
            },
            "production": {
                "near_dedup_production": "NOT RUN",
                "benchmark_decontamination": "PENDING",
                "C1": "IN PROGRESS",
                "C2": "PENDING",
                "final_training_token_budget": "NOT CHOSEN",
            },
            "parlamento_pt": {
                "sampled_rows": parliament_rows,
                "rows_marked_removed": parliament_deleted,
                "all_rows_preserved_in_pilot_artifact": parliament_rows
                == sum(row["source"] == "parlamento_pt" for row in metadata.values()),
                "formulaic_matching_is_diagnostic": True,
                "short_utterance_definition": "fewer than 20 normalized words",
            },
            "synthetic_validation": synthetic_metrics,
            "review_validation": review_metrics,
            "cluster_member_rows_written": cluster_member_count,
            "resource_report": report,
            "artifacts": artifacts,
        }
        _json_write(stage / "manifest.json", manifest)
        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing near-dedup pilot: {output}"
            )
        os.replace(stage, output)
        errors = verify_near_pilot_manifest(output)
        if errors:
            raise ValueError(
                "Near-pilot artifact verification failed: " + "; ".join(errors)
            )
        return manifest
    except BaseException:
        if connection is not None:
            connection.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise
