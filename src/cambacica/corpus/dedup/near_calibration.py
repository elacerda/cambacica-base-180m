"""Targeted, non-destructive D2b near-dedup calibration.

The calibration starts from the immutable post-exact corpus and the completed
D2 panel. It enriches that panel with deterministic metadata families, scores
their real text, and compares LSH against exhaustive comparisons inside small
gold families. It never writes or removes production corpus rows.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cambacica.corpus.dedup import near_pilot as near
from cambacica.corpus.dedup.exact_pipeline import GV_SUBSET_TIERS
from cambacica.corpus.manifest import compute_file_sha256


CALIBRATION_VERSION = "1.0.0"
DEFAULT_CALIBRATION_ROOT = Path(
    "/tmp/cambacica-base-180m/dedup-pilots/near-calibration-v1"
)
DEFAULT_BASE_PILOT_ROOT = near.DEFAULT_NEAR_PILOT_ROOT
DEFAULT_CALIBRATION_SEED = near.DEFAULT_NEAR_PILOT_SEED
EXACT_THRESHOLDS = (0.80, 0.85, 0.90, 0.92, 0.95)
EXACT_NGRAMS = (3, 5, 7)
MAX_EXACT_SHINGLES = near.MAX_EXACT_REVIEW_SHINGLES
TITLE_KEY_SAMPLE_MODULUS = 512
URL_KEY_SAMPLE_MODULUS = 2_048
DOMAIN_KEY_SAMPLE_MODULUS = 64
MAX_FAMILY_KEYS = 50_000
MAX_FAMILY_MEMBERS = 8
MAX_CANDIDATES_PER_REASON = 1_200
GOLD_FAMILY_LIMITS = {
    "same_title": 128,
    "same_url_variant": 96,
    "same_url_domain": 72,
    "neighboring_crawl_records": 192,
}
DISTINCTIVE_DF_CUTOFF = 3
DISTINCTIVE_REFERENCE_CAP = 3_000
REVIEW_PAIR_CAP = 100
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class _FamilyCollector:
    """Order-independent bounded index of metadata-selected record families."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.groups: dict[str, dict[str, Any]] = {}
        self.group_counts: Counter[str] = Counter()
        self.moduli = {
            "same_title": TITLE_KEY_SAMPLE_MODULUS,
            "same_url_variant": URL_KEY_SAMPLE_MODULUS,
            "same_url_domain": DOMAIN_KEY_SAMPLE_MODULUS,
        }
        self.neighbors = near._BottomK(1_200, seed ^ 0x4E454947)
        self.seen_neighbor_groups = 0

    def _remember(self, kind: str, key: str, record: Mapping[str, Any]) -> None:
        if not key:
            return
        modulus = self.moduli[kind]
        if near.stable_hash64(f"{kind}\0{key}", self.seed) % modulus:
            return
        family_id = f"{kind}:{hashlib.sha256(key.encode('utf-8')).hexdigest()[:20]}"
        group = self.groups.get(family_id)
        if group is None:
            if self.group_counts[kind] >= MAX_FAMILY_KEYS:
                raise ValueError(
                    f"{kind} family-key cap reached; increase its deterministic "
                    "sampling modulus before calibration"
                )
            group = {
                "family_id": family_id,
                "kind": kind,
                "key": key,
                "members": {},
            }
            self.groups[family_id] = group
            self.group_counts[kind] += 1
        record_id = str(record["pilot_occurrence_id"])
        group["members"][record_id] = dict(record)
        if len(group["members"]) > MAX_FAMILY_MEMBERS:
            ranked = sorted(
                group["members"],
                key=lambda item: (
                    near.stable_hash64(item, self.seed ^ 0xD1A6),
                    item,
                ),
            )
            group["members"] = {
                item: group["members"][item] for item in ranked[:MAX_FAMILY_MEMBERS]
            }

    def add(self, record: Mapping[str, Any]) -> None:
        title = _normalized_title(record.get("title"))
        if title:
            self._remember("same_title", title, record)
        canonical_url = _canonical_url(record.get("original_url"))
        if canonical_url:
            self._remember("same_url_variant", canonical_url, record)
        domain = near._domain(record.get("original_url"))
        if domain:
            self._remember("same_url_domain", domain, record)

    def add_neighbor(
        self, previous: Mapping[str, Any] | None, current: Mapping[str, Any]
    ) -> None:
        if not previous or current.get("source") != "gigaverbo_v2":
            return
        if current.get("subset") != previous.get("subset"):
            return
        if current.get("_gv2_upstream_shard") != previous.get("_gv2_upstream_shard"):
            return
        if current.get("_gv2_upstream_row_group") != previous.get(
            "_gv2_upstream_row_group"
        ):
            return
        left, right = sorted(
            (
                str(previous["pilot_occurrence_id"]),
                str(current["pilot_occurrence_id"]),
            )
        )
        records_by_id = {
            str(previous["pilot_occurrence_id"]): dict(previous),
            str(current["pilot_occurrence_id"]): dict(current),
        }
        self.seen_neighbor_groups += 1
        self.neighbors.add(
            f"{left}\0{right}",
            {
                "family_id": "neighbor:"
                + hashlib.sha256(f"{left}\0{right}".encode("utf-8")).hexdigest()[:20],
                "kind": "neighboring_crawl_records",
                "members": [records_by_id[left], records_by_id[right]],
            },
        )

    def finalize(self) -> dict[str, Any]:
        candidate_pools = {
            kind: near._BottomK(
                MAX_CANDIDATES_PER_REASON,
                self.seed ^ near.stable_hash64(kind, self.seed),
            )
            for kind in (
                "same_title",
                "same_url_variant",
                "same_url_domain",
                "neighboring_crawl_records",
            )
        }
        gold_pools = {
            kind: near._BottomK(
                limit, self.seed ^ near.stable_hash64(f"gold:{kind}", self.seed)
            )
            for kind, limit in GOLD_FAMILY_LIMITS.items()
        }

        for family_id, group in sorted(self.groups.items()):
            members = sorted(group["members"].values(), key=_record_rank)
            if len(members) < 2:
                continue
            kind = str(group["kind"])
            gold_pools[kind].add(
                family_id,
                {
                    "family_id": family_id,
                    "kind": kind,
                    "members": members,
                },
            )
            for left, right in itertools.combinations(members, 2):
                id_a, id_b = sorted(
                    (left["pilot_occurrence_id"], right["pilot_occurrence_id"])
                )
                candidate_pools[kind].add(
                    f"{family_id}\0{id_a}\0{id_b}",
                    {
                        "id_a": id_a,
                        "id_b": id_b,
                        "reason": kind,
                        "family_id": family_id,
                        "record_a": left,
                        "record_b": right,
                    },
                )

        for item in self.neighbors.values():
            candidate_pools["neighboring_crawl_records"].add(
                f"{item['family_id']}\0{item['members'][0]['pilot_occurrence_id']}",
                {
                    "id_a": item["members"][0]["pilot_occurrence_id"],
                    "id_b": item["members"][1]["pilot_occurrence_id"],
                    "reason": "neighboring_crawl_records",
                    "family_id": item["family_id"],
                    "record_a": item["members"][0],
                    "record_b": item["members"][1],
                },
            )
            gold_pools["neighboring_crawl_records"].add(item["family_id"], item)

        candidate_pairs: dict[tuple[str, str], dict[str, set[str]]] = {}
        target_records: dict[str, Mapping[str, Any]] = {}
        for kind, pool in candidate_pools.items():
            for row in pool.values():
                key = (str(row["id_a"]), str(row["id_b"]))
                item = candidate_pairs.setdefault(
                    key, {"reasons": set(), "family_ids": set()}
                )
                item["reasons"].add(kind)
                item["family_ids"].add(str(row["family_id"]))
                for field in ("record_a", "record_b"):
                    record = row.get(field)
                    if record is not None:
                        target_records[str(record["pilot_occurrence_id"])] = record

        gold_groups: list[dict[str, Any]] = []
        for kind, pool in gold_pools.items():
            gold_groups.extend(pool.values())
        gold_pairs: dict[tuple[str, str], set[str]] = {}
        for group in gold_groups:
            members = group["members"]
            for left, right in itertools.combinations(members, 2):
                key = tuple(
                    sorted(
                        (
                            str(left["pilot_occurrence_id"]),
                            str(right["pilot_occurrence_id"]),
                        )
                    )
                )
                gold_pairs.setdefault(key, set()).add(str(group["family_id"]))

        target_tags: dict[str, set[str]] = defaultdict(set)
        for (id_a, id_b), item in candidate_pairs.items():
            for record_id in (id_a, id_b):
                target_tags[record_id].update(item["reasons"])
            for field in ("record_a", "record_b"):
                record = item.get(field)
                if record is not None:
                    target_records[str(record["pilot_occurrence_id"])] = record
        for group in gold_groups:
            for record in group["members"]:
                record_id = str(record["pilot_occurrence_id"])
                target_tags[record_id].add(group["kind"])
                target_records[record_id] = record

        for pair, item in candidate_pairs.items():
            record_ids = {record_id for record_id in pair}
            if not record_ids <= target_records.keys():
                raise AssertionError("Candidate pair lost an endpoint descriptor")
            for record_id in record_ids:
                target_tags[record_id].update(item["reasons"])

        return {
            "candidate_pairs": candidate_pairs,
            "gold_pairs": gold_pairs,
            "gold_groups": gold_groups,
            "target_tags": target_tags,
            "target_records": target_records,
            "family_key_counts": dict(
                sorted(Counter(row["kind"] for row in self.groups.values()).items())
            ),
            "neighbor_pair_candidates_seen": self.seen_neighbor_groups,
        }


def _record_rank(record: Mapping[str, Any]) -> tuple[int, str]:
    record_id = str(record["pilot_occurrence_id"])
    return (near.stable_hash64(record_id, 0xD2B), record_id)


def _normalized_title(value: Any) -> str:
    if not value:
        return ""
    tokens = [item.casefold() for item in TOKEN_RE.findall(str(value))]
    if len(tokens) < 3 or sum(len(item) for item in tokens) < 12:
        return ""
    return " ".join(tokens)


def _canonical_url(value: Any) -> str:
    if not value:
        return ""
    try:
        parsed = urlsplit(str(value).strip())
        host = (parsed.hostname or "").casefold().removeprefix("www.").rstrip(".")
        if not host:
            return ""
        path = re.sub(r"/{2,}", "/", parsed.path or "/").rstrip("/") or "/"
        return f"{host}{path}"
    except ValueError:
        return ""


def _scan_metadata_families(
    files: Sequence[Path], *, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    collector = _FamilyCollector(seed)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    scanned_rows = 0
    projected_bytes = 0
    source_counts: Counter[str] = Counter()
    subset_counts: Counter[str] = Counter()
    all_sources: set[str] = set()
    all_subsets: set[str] = set()
    columns = [
        "source",
        "subset",
        "original_url",
        "title",
        "content_sha256",
        "_gv2_upstream_shard",
        "_gv2_upstream_row_group",
    ]
    for path in files:
        relative_path = path.relative_to(files[0].parents[1]).as_posix()
        parquet = pq.ParquetFile(path)
        projected_bytes += near._projected_compressed_bytes(parquet, columns)
        ordinal = 0
        previous: dict[str, Any] | None = None
        for batch in parquet.iter_batches(batch_size=4096, columns=columns):
            for row_index, row in enumerate(batch.to_pylist()):
                row_source = str(row.get("source") or "")
                row_subset = str(row.get("subset") or "")
                record = near._candidate_metadata(
                    row, relative_path, ordinal + row_index, 0
                )
                record["source"] = row_source
                record["subset"] = row_subset
                collector.add(record)
                collector.add_neighbor(previous, record)
                previous = record
                scanned_rows += 1
                source_counts[row_source] += 1
                subset_counts[f"{row_source}/{row_subset}"] += 1
                all_sources.add(row_source)
                if row_source == "gigaverbo_v2":
                    all_subsets.add(row_subset)
            ordinal += batch.num_rows
    if scanned_rows != near.POST_EXACT_RECORD_COUNT:
        raise ValueError(
            f"Metadata scan found {scanned_rows} rows; expected "
            f"{near.POST_EXACT_RECORD_COUNT}"
        )
    if all_sources != near.EXPECTED_SOURCES:
        raise ValueError(f"Source coverage mismatch: {sorted(all_sources)}")
    if all_subsets != near.EXPECTED_GIGAVERBO_SUBSETS:
        raise ValueError(f"GigaVerbo coverage mismatch: {sorted(all_subsets)}")
    families = collector.finalize()
    metrics = {
        "records_scanned": scanned_rows,
        "source_counts": dict(sorted(source_counts.items())),
        "source_subset_counts": dict(sorted(subset_counts.items())),
        "seen_sources": sorted(all_sources),
        "seen_gigaverbo_subsets": sorted(all_subsets),
        "metadata_projected_compressed_bytes": projected_bytes,
        "wall_seconds": round(time.perf_counter() - started_wall, 3),
        "cpu_seconds": round(time.process_time() - started_cpu, 3),
    }
    return families, metrics


def _materialize_calibration_records(
    *,
    files: Sequence[Path],
    base_records_path: Path,
    target_records: Mapping[str, Mapping[str, Any]],
    target_tags: Mapping[str, set[str]],
    output_path: Path,
) -> dict[str, Any]:
    base_parquet = pq.ParquetFile(base_records_path)
    base_schema = base_parquet.schema_arrow
    if "calibration_families" in base_schema.names:
        raise ValueError("D2 base panel unexpectedly has calibration_families")
    output_schema = base_schema.append(
        pa.field("calibration_families", pa.list_(pa.string()), nullable=False)
    )
    temporary = output_path.with_name(f"{output_path.name}.partial")
    writer = pq.ParquetWriter(
        temporary,
        output_schema,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )
    base_ids: set[str] = set()
    base_count = 0
    added_count = 0
    target_by_path: dict[str, dict[int, tuple[str, Mapping[str, Any]]]] = defaultdict(
        dict
    )
    for record_id, record in target_records.items():
        target_by_path[str(record["data_relative_path"])][
            int(record["data_row_ordinal"])
        ] = (record_id, record)

    try:
        for batch in base_parquet.iter_batches(batch_size=256):
            table = pa.Table.from_batches([batch])
            id_values = table["pilot_occurrence_id"].to_pylist()
            roles = table["selection_role"].to_pylist()
            families = []
            for record_id, role in zip(id_values, roles, strict=True):
                base_ids.add(str(record_id))
                tags = {"d2_reference_panel"}
                if role == "diagnostic_enrichment":
                    tags.add("d2_diagnostic_enrichment")
                tags.update(target_tags.get(str(record_id), set()))
                families.append(sorted(tags))
            table = table.append_column(
                pa.field(
                    "calibration_families",
                    pa.list_(pa.string()),
                    nullable=False,
                ),
                pa.array(families, type=pa.list_(pa.string())),
            )
            writer.write_table(table)
            base_count += table.num_rows

        data_root = files[0].parents[1]
        for path in files:
            relative_path = path.relative_to(data_root).as_posix()
            wanted = target_by_path.get(relative_path)
            if not wanted:
                continue
            parquet = pq.ParquetFile(path)
            offset = 0
            for row_group in range(parquet.metadata.num_row_groups):
                group_rows = parquet.metadata.row_group(row_group).num_rows
                high = offset + group_rows
                local = {
                    ordinal - offset: value
                    for ordinal, value in wanted.items()
                    if offset <= ordinal < high
                }
                if not local:
                    offset = high
                    continue
                ordered = sorted(local)
                cursor = 0
                row_index = 0
                for batch in parquet.iter_batches(
                    row_groups=[row_group],
                    batch_size=32,
                    columns=base_schema.names[: len(near.NORMALIZED_SCHEMA.names)]
                    if hasattr(near, "NORMALIZED_SCHEMA")
                    else None,
                ):
                    batch_end = row_index + batch.num_rows
                    positions: list[int] = []
                    while cursor < len(ordered) and ordered[cursor] < batch_end:
                        positions.append(ordered[cursor] - row_index)
                        cursor += 1
                    if positions:
                        rows = batch.take(
                            pa.array(positions, type=pa.int64())
                        ).to_pylist()
                        pending: list[dict[str, Any]] = []
                        for position, row in zip(positions, rows, strict=True):
                            record_id, family_record = local[row_index + position]
                            pilot_id = near.occurrence_id(
                                relative_path, offset + row_index + position
                            )
                            if pilot_id != record_id:
                                raise ValueError(
                                    f"Target occurrence identity changed: {record_id}"
                                )
                            words = near.count_normalized_words_bounded(
                                row.get("text") or ""
                            )
                            row.update(
                                {
                                    "pilot_occurrence_id": pilot_id,
                                    "normalized_words": words,
                                    "length_stratum": near.length_stratum(words),
                                    "selection_role": "targeted_family",
                                    "sample_tags": ["targeted_family"],
                                    "data_relative_path": relative_path,
                                    "data_row_ordinal": offset + row_index + position,
                                    "sampling_frame_population": None,
                                    "sampling_frame_sample_n": None,
                                    "base_sampling_weight": None,
                                    "pilot_file_row_ordinal": base_count + added_count,
                                    "calibration_families": sorted(
                                        target_tags.get(pilot_id, set())
                                    ),
                                }
                            )
                            pending.append(row)
                            added_count += 1
                        writer.write_table(
                            pa.Table.from_pylist(pending, schema=output_schema)
                        )
                    row_index = batch_end
                if row_index != group_rows:
                    raise ValueError(f"Row count changed while reading {path}")
                offset = high
        writer.close()
        os.replace(temporary, output_path)
    except BaseException:
        try:
            writer.close()
        except Exception:
            pass
        temporary.unlink(missing_ok=True)
        raise

    if len(base_ids) != base_count:
        raise AssertionError("D2 base panel contains duplicate occurrence IDs")
    if base_ids.intersection(target_records):
        raise AssertionError("Target panel must omit already-present D2 occurrences")
    if added_count != len(target_records):
        raise ValueError(
            f"Materialized {added_count} targeted rows; expected {len(target_records)}"
        )
    return {
        "d2_base_records": base_count,
        "targeted_records": added_count,
        "calibration_record_count": base_count + added_count,
        "parquet_bytes": output_path.stat().st_size,
    }


def _read_texts(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=128, columns=["pilot_occurrence_id", "text"]
    ):
        for record_id, text_value in zip(
            batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
        ):
            result[str(record_id)] = text_value or ""
    return result


def _shingle_set(
    text: str, ngram: int, seed: int = DEFAULT_CALIBRATION_SEED
) -> set[int] | None:
    result: set[int] = set()
    for value in near.iter_word_shingle_hashes(text, ngram, seed=seed):
        result.add(value)
        if len(result) > MAX_EXACT_SHINGLES:
            return None
    return result


def _directional_pair_metrics(
    text_a: str,
    text_b: str,
    df_common: set[int],
    *,
    seed: int = DEFAULT_CALIBRATION_SEED,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for ngram in EXACT_NGRAMS:
        set_a = _shingle_set(text_a, ngram, seed)
        set_b = _shingle_set(text_b, ngram, seed)
        prefix = f"word{ngram}"
        if set_a is None or set_b is None:
            result[f"exact_available_{prefix}"] = False
            for name in (
                "exact_jaccard",
                "shared_shingles",
                "unique_shingles_a",
                "unique_shingles_b",
                "containment_a_in_b",
                "containment_b_in_a",
            ):
                result[f"{name}_{prefix}"] = None
            continue
        shared = len(set_a & set_b)
        union = len(set_a | set_b)
        result[f"exact_available_{prefix}"] = True
        result[f"exact_jaccard_{prefix}"] = shared / union if union else 0.0
        result[f"shared_shingles_{prefix}"] = shared
        result[f"unique_shingles_a_{prefix}"] = len(set_a)
        result[f"unique_shingles_b_{prefix}"] = len(set_b)
        result[f"containment_a_in_b_{prefix}"] = shared / len(set_a) if set_a else 0.0
        result[f"containment_b_in_a_{prefix}"] = shared / len(set_b) if set_b else 0.0
        if ngram == 5:
            distinctive_a = set_a - df_common
            distinctive_b = set_b - df_common
            shared_distinctive = len(distinctive_a & distinctive_b)
            union_distinctive = len(distinctive_a | distinctive_b)
            result.update(
                {
                    "distinctive_shared_shingles_word5": shared_distinctive,
                    "distinctive_union_shingles_word5": union_distinctive,
                    "distinctive_jaccard_word5": (
                        shared_distinctive / union_distinctive
                        if union_distinctive
                        else 0.0
                    ),
                    "distinctive_containment_word5": (
                        shared_distinctive / min(len(distinctive_a), len(distinctive_b))
                        if min(len(distinctive_a), len(distinctive_b))
                        else 0.0
                    ),
                    "common_fraction_of_shared_word5": (
                        (shared - shared_distinctive) / shared if shared else 0.0
                    ),
                }
            )
    return result


def _document_frequency_reference(
    metadata: Mapping[str, Mapping[str, Any]],
    texts: Mapping[str, str],
    *,
    cap: int = DISTINCTIVE_REFERENCE_CAP,
    seed: int = DEFAULT_CALIBRATION_SEED,
) -> tuple[set[int], dict[str, Any]]:
    """Find shingles shared by at least three calibration-reference documents."""
    eligible = [
        (record_id, row)
        for record_id, row in metadata.items()
        if 20 <= int(row.get("normalized_words") or 0) <= 5_000 and record_id in texts
    ]
    family_eligible = [
        item
        for item in eligible
        if "same_url_domain" in item[1].get("calibration_families", [])
    ]
    source = family_eligible if len(family_eligible) >= 32 else eligible
    ranked = sorted(
        source,
        key=lambda item: (
            near.stable_hash64(item[0], seed ^ 0xDF),
            item[0],
        ),
    )[:cap]
    chunks: list[np.ndarray] = []
    for document_index, (record_id, _row) in enumerate(ranked):
        values = np.fromiter(
            set(near.iter_word_shingle_hashes(texts[record_id], 5, seed=seed)),
            dtype=np.uint64,
        )
        if not len(values):
            continue
        chunks.append(values)
    if not chunks:
        return set(), {
            "reference_documents": 0,
            "reference_shingle_occurrences": 0,
            "frequent_shingles_at_cutoff": 0,
            "document_frequency_cutoff": DISTINCTIVE_DF_CUTOFF,
        }
    all_hashes = np.concatenate(chunks)
    all_hashes.sort()
    unique, counts = np.unique(all_hashes, return_counts=True)
    common = set(map(int, unique[counts >= DISTINCTIVE_DF_CUTOFF]))
    return common, {
        "reference_documents": len(ranked),
        "reference_shingle_occurrences": int(len(all_hashes)),
        "frequent_shingles_at_cutoff": len(common),
        "document_frequency_cutoff": DISTINCTIVE_DF_CUTOFF,
        "reference_selection": (
            "deterministic same-domain family records when at least 32 qualify; "
            "otherwise all eligible calibration records"
        ),
    }


def _canonical_owner(
    row_a: Mapping[str, Any], row_b: Mapping[str, Any]
) -> tuple[str, str]:
    """Return a provisional provenance-based owner without changing data."""
    source_a, source_b = str(row_a["source"]), str(row_b["source"])
    if {source_a, source_b} == {"wikipedia_pt", "carolina"}:
        carolina = row_a if source_a == "carolina" else row_b
        wikipedia = row_a if source_a == "wikipedia_pt" else row_b
        is_wik = "wik" in str(carolina.get("subset") or "").casefold() or (
            "wikipedia" in str(carolina.get("original_url") or "").casefold()
        )
        if is_wik:
            return (
                str(wikipedia["pilot_occurrence_id"]),
                "Wikipedia PT preferred for a Carolina Wikipedia-derived pair",
            )

    def order(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
        source = str(row["source"])
        subset = str(row.get("subset") or "")
        if source in {"carolina", "gutenberg_pt", "wikipedia_pt"}:
            source_rank = {
                "gutenberg_pt": 0,
                "wikipedia_pt": 1,
                "carolina": 2,
            }[source]
            provenance_tier = 0
        elif source == "gigaverbo_v2":
            provenance_tier = GV_SUBSET_TIERS.get(subset, 5)
            source_rank = 3
        else:
            provenance_tier = 9
            source_rank = 9
        return (
            provenance_tier,
            source_rank,
            -int(row.get("normalized_words") or 0),
            str(row["pilot_occurrence_id"]),
        )

    winner = min((row_a, row_b), key=order)
    reason = (
        "curated/native provenance before GigaVerbo tier; deterministic occurrence "
        "ID is the final tie-break"
    )
    if source_a == source_b == "gigaverbo_v2":
        reason = "stronger documented GigaVerbo provenance tier"
    return str(winner["pilot_occurrence_id"]), reason


def _short_document_action(
    words_a: int,
    words_b: int,
    source_a: str,
    source_b: str,
) -> str:
    """Return a calibration guardrail proposal for a pair's shorter document."""
    if "parlamento_pt" in {source_a, source_b}:
        return "preserve_all_diagnostic"
    shorter = min(words_a, words_b)
    if shorter < 20:
        return "exact_only_or_diagnostic"
    if shorter < 100:
        return "review_only"
    return "normal_candidate_generation"


def _score_pair(
    id_a: str,
    id_b: str,
    metadata: Mapping[str, Mapping[str, Any]],
    texts: Mapping[str, str],
    signatures: Mapping[str, Mapping[str, Sequence[int]]],
    df_common: set[int],
    *,
    seed: int = DEFAULT_CALIBRATION_SEED,
) -> dict[str, Any]:
    row_a, row_b = metadata[id_a], metadata[id_b]
    words_a = int(row_a.get("normalized_words") or 0)
    words_b = int(row_b.get("normalized_words") or 0)
    maximum = max(words_a, words_b)
    ratio = min(words_a, words_b) / maximum if maximum else 1.0
    exact = _directional_pair_metrics(texts[id_a], texts[id_b], df_common, seed=seed)
    sig_a, sig_b = signatures[id_a], signatures[id_b]
    estimates = {
        "minhash_estimate_word3_128": near.estimate_jaccard_signature(
            sig_a["word3_128"], sig_b["word3_128"]
        ),
        "minhash_estimate_word5_64": near.estimate_jaccard_signature(
            sig_a["word5_256"][:64], sig_b["word5_256"][:64]
        ),
        "minhash_estimate_word5_128": near.estimate_jaccard_signature(
            sig_a["word5_256"][:128], sig_b["word5_256"][:128]
        ),
        "minhash_estimate_word5_256": near.estimate_jaccard_signature(
            sig_a["word5_256"], sig_b["word5_256"]
        ),
        "minhash_estimate_word7_128": near.estimate_jaccard_signature(
            sig_a["word7_128"], sig_b["word7_128"]
        ),
    }
    owner_id, owner_reason = _canonical_owner(row_a, row_b)
    return {
        "id_a": id_a,
        "id_b": id_b,
        "source_a": str(row_a.get("source") or ""),
        "subset_a": str(row_a.get("subset") or ""),
        "source_b": str(row_b.get("source") or ""),
        "subset_b": str(row_b.get("subset") or ""),
        "words_a": words_a,
        "words_b": words_b,
        "length_ratio_min_max": ratio,
        "short_document_action": _short_document_action(
            words_a,
            words_b,
            str(row_a.get("source") or ""),
            str(row_b.get("source") or ""),
        ),
        "containment_review_flag": (
            min(words_a, words_b) >= 100
            and ratio < 0.50
            and max(
                float(exact.get("containment_a_in_b_word5") or 0.0),
                float(exact.get("containment_b_in_a_word5") or 0.0),
            )
            >= 0.90
        ),
        "same_source": row_a.get("source") == row_b.get("source"),
        "same_source_cross_subset": row_a.get("source") == row_b.get("source")
        and row_a.get("subset") != row_b.get("subset"),
        "cross_source": row_a.get("source") != row_b.get("source"),
        "original_url_a": row_a.get("original_url"),
        "original_url_b": row_b.get("original_url"),
        "title_a": row_a.get("title"),
        "title_b": row_b.get("title"),
        "provisional_owner_id": owner_id,
        "provisional_owner_reason": owner_reason,
        "would_remove": False,
        **exact,
        **estimates,
    }


def _collect_lsh_candidates(
    connection: Any, signatures: Mapping[str, Mapping[str, Sequence[int]]], seed: int
) -> dict[tuple[str, str], set[str]]:
    metrics = near._run_lsh(connection, signatures, seed=seed)
    found: dict[tuple[str, str], set[str]] = defaultdict(set)
    for config, id_a, id_b in connection.execute(
        "SELECT config_id,id_a,id_b FROM candidate_by_config ORDER BY id_a,id_b,config_id"
    ):
        found[(id_a, id_b)].add(config)
    return found, metrics


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if rows:
        table = pa.Table.from_pylist([dict(row) for row in rows])
    else:
        table = pa.table({"empty": pa.array([], type=pa.string())})
    pq.write_table(table, path, compression="zstd", compression_level=6)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict, set, tuple))
                else value
                for key, value in row.items()
            }
            writer.writerow(normalized)


def _threshold_rows(
    gold_pairs: Mapping[tuple[str, str], set[str]],
    gold_scores: Mapping[tuple[str, str], Mapping[str, Any]],
    lsh_candidates: Mapping[tuple[str, str], set[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configs: list[tuple[str, str, set[tuple[str, str]]]] = []
    for config in near.LSH_CONFIGS:
        generated = {
            pair
            for pair, generators in lsh_candidates.items()
            if config.name in generators
        }
        configs.append((config.name, config.representation, generated))
    for representation in ("word3_128", "word5_256", "word7_128"):
        represented = [
            config
            for config in near.LSH_CONFIGS
            if config.representation == representation
        ]
        union = set().union(
            *(
                {
                    pair
                    for pair, generators in lsh_candidates.items()
                    if config.name in generators
                }
                for config in represented
            )
        )
        configs.append((f"{representation}_union", representation, union))

    exact_column = {
        "word3_128": "exact_jaccard_word3",
        "word5_256": "exact_jaccard_word5",
        "word7_128": "exact_jaccard_word7",
    }
    for config_name, representation, generated in configs:
        gold_generated = set(gold_pairs) & generated
        for threshold in EXACT_THRESHOLDS:
            positives = {
                pair
                for pair in gold_pairs
                if gold_scores[pair].get(exact_column[representation]) is not None
                and gold_scores[pair][exact_column[representation]] >= threshold
            }
            hits = positives & gold_generated
            rows.append(
                {
                    "lsh_configuration": config_name,
                    "representation": representation,
                    "threshold": threshold,
                    "exhaustive_gold_positive_pairs": len(positives),
                    "gold_pairs_generated_by_lsh": len(gold_generated),
                    "positive_pairs_recalled": len(hits),
                    "recall": len(hits) / len(positives) if positives else None,
                    "candidate_precision_within_gold_families": (
                        len(hits) / len(gold_generated) if gold_generated else None
                    ),
                }
            )
    return rows


def _ratio_bin(ratio: float) -> str:
    if ratio < 0.10:
        return "0.00-0.10"
    if ratio < 0.25:
        return "0.10-0.25"
    if ratio < 0.50:
        return "0.25-0.50"
    if ratio < 0.75:
        return "0.50-0.75"
    if ratio < 0.90:
        return "0.75-0.90"
    return "0.90-1.00"


def _length_ratio_rows(scored: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in scored:
        if row.get("exact_jaccard_word5") is not None:
            groups[_ratio_bin(float(row["length_ratio_min_max"]))].append(row)
    rows: list[dict[str, Any]] = []
    for band in (
        "0.00-0.10",
        "0.10-0.25",
        "0.25-0.50",
        "0.50-0.75",
        "0.75-0.90",
        "0.90-1.00",
    ):
        values = groups.get(band, [])
        similarities = sorted(float(row["exact_jaccard_word5"]) for row in values)
        containments = sorted(
            max(
                float(row["containment_a_in_b_word5"] or 0.0),
                float(row["containment_b_in_a_word5"] or 0.0),
            )
            for row in values
        )
        rows.append(
            {
                "length_ratio_band": band,
                "scored_pairs": len(values),
                "median_exact_jaccard_word5": near._quantile(similarities, 0.50),
                "mean_exact_jaccard_word5": (
                    sum(similarities) / len(similarities) if similarities else None
                ),
                "median_max_directional_containment_word5": near._quantile(
                    containments, 0.50
                ),
                "pairs_jaccard_at_least_0_80": sum(
                    value >= 0.80 for value in similarities
                ),
                "pairs_jaccard_at_least_0_90": sum(
                    value >= 0.90 for value in similarities
                ),
            }
        )
    return rows


def _containment_rows(scored: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in scored:
        if row.get("exact_jaccard_word5") is None:
            continue
        words_min = min(int(row["words_a"]), int(row["words_b"]))
        ratio = float(row["length_ratio_min_max"])
        if ratio >= 0.80:
            bucket = "near_equal_length"
        elif ratio >= 0.25:
            bucket = "moderate_length_difference"
        elif ratio >= 0.10:
            bucket = "embedded_document"
        else:
            bucket = "short_excerpt_in_long_document"
        if words_min < 20:
            bucket += "_shorter_under_20_words"
        elif words_min < 100:
            bucket += "_shorter_20_to_99_words"
        elif words_min >= 100_000:
            bucket += "_shorter_giant"
        buckets[bucket].append(row)
    rows: list[dict[str, Any]] = []
    for bucket, values in sorted(buckets.items()):
        containments = [
            max(
                float(row["containment_a_in_b_word5"] or 0.0),
                float(row["containment_b_in_a_word5"] or 0.0),
            )
            for row in values
        ]
        similarities = [float(row["exact_jaccard_word5"]) for row in values]
        rows.append(
            {
                "length_relationship": bucket,
                "scored_pairs": len(values),
                "containment_at_least_0_80": sum(
                    value >= 0.80 for value in containments
                ),
                "containment_at_least_0_95": sum(
                    value >= 0.95 for value in containments
                ),
                "jaccard_below_0_50": sum(value < 0.50 for value in similarities),
                "median_containment": near._quantile(sorted(containments), 0.50),
                "median_jaccard_word5": near._quantile(sorted(similarities), 0.50),
                "policy": "diagnostic only; containment is not an automatic duplicate decision",
            }
        )
    return rows


def _short_document_rows(scored: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    bands = (
        ("under_20_words", 0, 20),
        ("20_to_99_words", 20, 100),
        ("100_to_999_words", 100, 1_000),
        ("1k_to_99k_words", 1_000, 100_000),
        ("100k_plus_words", 100_000, None),
    )
    rows: list[dict[str, Any]] = []
    for label, lower, upper in bands:
        values = [
            row
            for row in scored
            if min(int(row["words_a"]), int(row["words_b"])) >= lower
            and (upper is None or min(int(row["words_a"]), int(row["words_b"])) < upper)
            and row.get("exact_jaccard_word5") is not None
        ]
        errors = {
            permutations: [
                abs(
                    float(row["exact_jaccard_word5"])
                    - float(row[f"minhash_estimate_word5_{permutations}"])
                )
                for row in values
                if row.get(f"minhash_estimate_word5_{permutations}") is not None
            ]
            for permutations in (64, 128, 256)
        }
        rows.append(
            {
                "shorter_document_length_band": label,
                "scored_pairs": len(values),
                "exact_jaccard_at_least_0_80": sum(
                    float(row["exact_jaccard_word5"]) >= 0.80 for row in values
                ),
                "exact_jaccard_at_least_0_90": sum(
                    float(row["exact_jaccard_word5"]) >= 0.90 for row in values
                ),
                "mean_absolute_minhash_error_64": (
                    sum(errors[64]) / len(errors[64]) if errors[64] else None
                ),
                "mean_absolute_minhash_error_128": (
                    sum(errors[128]) / len(errors[128]) if errors[128] else None
                ),
                "mean_absolute_minhash_error_256": (
                    sum(errors[256]) / len(errors[256]) if errors[256] else None
                ),
                "proposed_action": (
                    "exact-only or diagnostic"
                    if lower == 0
                    else "review-only"
                    if lower == 20
                    else "normal candidate scoring with containment diagnostics"
                ),
            }
        )
    return rows


def _provisional_label(
    row: Mapping[str, Any],
    metadata: Mapping[str, Mapping[str, Any]],
    reasons: Sequence[str],
) -> str:
    source_a, source_b = row["source_a"], row["source_b"]
    if "parlamento_pt" in {source_a, source_b}:
        return "ambiguous"
    jaccard = row.get("exact_jaccard_word5")
    if jaccard is None:
        return "ambiguous"
    containment = max(
        float(row.get("containment_a_in_b_word5") or 0.0),
        float(row.get("containment_b_in_a_word5") or 0.0),
    )
    if containment >= 0.90 and float(jaccard) < 0.50:
        return "containment case"
    distinctive = float(row.get("distinctive_jaccard_word5") or 0.0)
    common_fraction = float(row.get("common_fraction_of_shared_word5") or 0.0)
    if float(jaccard) >= 0.75 and (distinctive < 0.30 or common_fraction >= 0.75):
        return "boilerplate false positive"
    ratio = float(row["length_ratio_min_max"])
    if float(jaccard) >= 0.90 and ratio >= 0.70 and distinctive >= 0.70:
        return "true duplicate"
    if float(jaccard) >= 0.75 and ratio >= 0.50 and distinctive >= 0.50:
        return "probable duplicate"
    title_a = _normalized_title(metadata[row["id_a"]].get("title"))
    title_b = _normalized_title(metadata[row["id_b"]].get("title"))
    if (title_a and title_a == title_b) or "same_url_domain" in reasons:
        return "related but distinct"
    return "ambiguous"


def _human_review_rows(
    scored: Sequence[Mapping[str, Any]],
    candidates: Mapping[tuple[str, str], Mapping[str, Any]],
    gold_pairs: Mapping[tuple[str, str], set[str]],
    metadata: Mapping[str, Mapping[str, Any]],
    texts: Mapping[str, str],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str], Mapping[str, Any]] = {}
    pools: dict[str, near._BottomK] = {}
    for row in scored:
        pair = (str(row["id_a"]), str(row["id_b"]))
        candidate = candidates.get(pair, {})
        reasons = sorted(candidate.get("reasons", set()))
        jaccard = float(row.get("exact_jaccard_word5") or 0.0)
        containment = max(
            float(row.get("containment_a_in_b_word5") or 0.0),
            float(row.get("containment_b_in_a_word5") or 0.0),
        )
        if (
            float(row.get("common_fraction_of_shared_word5") or 0.0) >= 0.50
            and jaccard >= 0.65
        ):
            categories = ["boilerplate"]
        else:
            categories = []
        if containment >= 0.80 and jaccard < 0.65:
            categories.append("containment")
        if {row["source_a"], row["source_b"]} == {"carolina", "wikipedia_pt"}:
            categories.append("wikipedia_carolina")
        if min(int(row["words_a"]), int(row["words_b"])) < 100:
            categories.append("short_text")
        if jaccard >= 0.75:
            categories.append("high_similarity")
        if pair in gold_pairs:
            categories.append("exhaustive_gold")
        if not categories:
            categories.append("candidate_boundary")
        for category in categories:
            pool = pools.setdefault(
                category,
                near._BottomK(
                    12, seed ^ near.stable_hash64(f"review:{category}", seed)
                ),
            )
            pool.add(f"{pair[0]}\0{pair[1]}", dict(row))
    for pool in pools.values():
        for row in pool.values():
            selected[(str(row["id_a"]), str(row["id_b"]))] = row
    prioritized = sorted(
        selected.values(),
        key=lambda row: (
            0
            if "parlamento_pt" in {row["source_a"], row["source_b"]}
            else 1
            if {row["source_a"], row["source_b"]} == {"carolina", "wikipedia_pt"}
            else 2
            if float(row.get("common_fraction_of_shared_word5") or 0.0) >= 0.50
            else 3
            if max(
                float(row.get("containment_a_in_b_word5") or 0.0),
                float(row.get("containment_b_in_a_word5") or 0.0),
            )
            >= 0.80
            else 4,
            -float(row.get("exact_jaccard_word5") or 0.0),
            row["id_a"],
            row["id_b"],
        ),
    )[:REVIEW_PAIR_CAP]
    output: list[dict[str, Any]] = []
    for scored_row in prioritized:
        pair = (str(scored_row["id_a"]), str(scored_row["id_b"]))
        candidate = candidates.get(pair, {})
        reasons = sorted(candidate.get("reasons", set()))
        row = dict(scored_row)
        row.update(
            {
                "generator_configs": sorted(candidate.get("generators", set())),
                "family_reasons": reasons,
                "gold_family_ids": sorted(gold_pairs.get(pair, set())),
                "provisional_classification": _provisional_label(
                    scored_row, metadata, reasons
                ),
                "label_source": "evidence-based triage; human confirmation required",
                "human_label": None,
                "excerpt_a": near._excerpt_around(texts[pair[0]], None, width=300),
                "excerpt_b": near._excerpt_around(texts[pair[1]], None, width=300),
                "excerpt_tail_a": near._excerpt_around(
                    texts[pair[0]][-600:], None, width=300
                ),
                "excerpt_tail_b": near._excerpt_around(
                    texts[pair[1]][-600:], None, width=300
                ),
                "normalized_url_a": metadata[pair[0]].get("original_url"),
                "normalized_url_b": metadata[pair[1]].get("original_url"),
                "title_a": metadata[pair[0]].get("title"),
                "title_b": metadata[pair[1]].get("title"),
                "parlamento_policy": (
                    "preserve_all_diagnostic"
                    if "parlamento_pt" in {row["source_a"], row["source_b"]}
                    else None
                ),
            }
        )
        output.append(row)
    return output


def _configuration_rows(
    scored: Sequence[Mapping[str, Any]],
    lsh_metrics: Mapping[str, Any],
    *,
    signature_metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    representation_columns = {
        "word3_128": ("exact_jaccard_word3", "minhash_estimate_word3_128"),
        "word5_256": ("exact_jaccard_word5", "minhash_estimate_word5_256"),
        "word7_128": ("exact_jaccard_word7", "minhash_estimate_word7_128"),
    }
    for config in near.LSH_CONFIGS:
        exact_column, estimate_column = representation_columns[config.representation]
        pairs = [
            row
            for row in scored
            if row.get(exact_column) is not None
            and row.get(estimate_column) is not None
        ]
        errors = [
            abs(float(row[exact_column]) - float(row[estimate_column])) for row in pairs
        ]
        config_metrics = lsh_metrics["configurations"][config.name]
        rows.append(
            {
                "configuration": config.name,
                "representation": config.representation,
                "ngram_size": {"word3_128": 3, "word5_256": 5, "word7_128": 7}[
                    config.representation
                ],
                "permutations": config.num_permutations,
                "bands": config.bands,
                "rows_per_band": config.rows_per_band,
                "scored_candidate_pairs": len(pairs),
                "mean_absolute_minhash_error": (
                    sum(errors) / len(errors) if errors else None
                ),
                "candidate_pairs": config_metrics["unique_candidate_pairs"],
                "largest_lsh_bucket": config_metrics["largest_bucket"],
                "signature_seconds_cpu": signature_metrics[
                    "seconds_by_configuration"
                ].get(config.representation),
                "estimated_signature_bytes_per_record": (config.num_permutations * 8),
                "interpretation": "calibration-set diagnostic; not prevalence",
            }
        )
    for permutations in (64, 128, 256):
        exact_column = "exact_jaccard_word5"
        estimate_column = f"minhash_estimate_word5_{permutations}"
        pairs = [
            row
            for row in scored
            if row.get(exact_column) is not None
            and row.get(estimate_column) is not None
        ]
        errors = [
            abs(float(row[exact_column]) - float(row[estimate_column])) for row in pairs
        ]
        rows.append(
            {
                "configuration": f"word5_{permutations}_permutation_estimate",
                "representation": "word5_256_prefix",
                "ngram_size": 5,
                "permutations": permutations,
                "bands": None,
                "rows_per_band": None,
                "scored_candidate_pairs": len(pairs),
                "mean_absolute_minhash_error": (
                    sum(errors) / len(errors) if errors else None
                ),
                "candidate_pairs": None,
                "largest_lsh_bucket": None,
                "signature_seconds_cpu": None,
                "estimated_signature_bytes_per_record": permutations * 8,
                "interpretation": "prefix length comparison on identical 5-gram signature",
            }
        )
    return rows


def _synthetic_boilerplate_row(seed: int) -> dict[str, Any]:
    case = next(
        item
        for item in near._synthetic_cases()
        if item["case_id"] == "same_template_unrelated_body"
    )
    template = case["text_a"].split(" bodya", maxsplit=1)[0]
    texts = [
        case["text_a"],
        case["text_b"],
        template + " bodyc0000 bodyc0001 bodyc0002 bodyc0003",
        template + " bodyd0000 bodyd0001 bodyd0002 bodyd0003",
    ]
    frequencies: Counter[int] = Counter()
    for text in texts:
        frequencies.update(set(near.iter_word_shingle_hashes(text, 5, seed=seed)))
    common = {
        value for value, count in frequencies.items() if count >= DISTINCTIVE_DF_CUTOFF
    }
    metrics = _directional_pair_metrics(texts[0], texts[1], common, seed=seed)
    return {
        "analysis_scope": "adversarial synthetic control",
        "case_id": "same_template_unrelated_body_with_four_page_df_reference",
        "source_a": "synthetic",
        "subset_a": "",
        "source_b": "synthetic",
        "subset_b": "",
        "words_a": len(texts[0].split()),
        "words_b": len(texts[1].split()),
        "length_ratio_min_max": min(len(texts[0].split()), len(texts[1].split()))
        / max(len(texts[0].split()), len(texts[1].split())),
        "exact_jaccard_word5": metrics["exact_jaccard_word5"],
        "distinctive_jaccard_word5": metrics["distinctive_jaccard_word5"],
        "distinctive_shared_shingles_word5": metrics[
            "distinctive_shared_shingles_word5"
        ],
        "common_fraction_of_shared_word5": metrics["common_fraction_of_shared_word5"],
        "family_reasons": ["same template; four document reference"],
        "review_label": "boilerplate false positive",
        "document_frequency_cutoff": DISTINCTIVE_DF_CUTOFF,
        "lsh_estimate_word5_256": near.estimate_jaccard_signature(
            near.compute_minhash_signature(
                texts[0], near.MinHashConfig("word5_256", 5, 256, seed)
            ),
            near.compute_minhash_signature(
                texts[1], near.MinHashConfig("word5_256", 5, 256, seed)
            ),
        ),
    }


def _policy_rows(
    scored: Sequence[Mapping[str, Any]],
    review: Sequence[Mapping[str, Any]],
    gold_pairs: Mapping[tuple[str, str], set[str]],
    gold_scores: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    policies = (
        ("P1_global_jaccard", False, False, False),
        ("P2_jaccard_short_guard", True, False, False),
        ("P3_short_guard_containment_diagnostic", True, True, False),
        ("P4_short_guard_distinctive_and_containment", True, True, True),
    )
    positive_labels = {"true duplicate", "probable duplicate"}
    review_positive_total = sum(
        row["provisional_classification"] in positive_labels for row in review
    )
    gold_positive = {
        pair
        for pair, row in gold_scores.items()
        if row.get("exact_jaccard_word5") is not None
        and float(row["exact_jaccard_word5"]) >= 0.85
    }
    rows: list[dict[str, Any]] = []
    for name, short_guard, containment_diag, distinctive_guard in policies:
        accepted_review: list[Mapping[str, Any]] = []
        for row in review:
            if "parlamento_pt" in {row["source_a"], row["source_b"]}:
                continue
            if float(row.get("exact_jaccard_word5") or 0.0) < 0.85:
                continue
            if short_guard and min(int(row["words_a"]), int(row["words_b"])) < 20:
                continue
            if (
                distinctive_guard
                and float(row.get("distinctive_jaccard_word5") or 0.0) < 0.75
            ):
                continue
            accepted_review.append(row)
        accepted_positive = sum(
            row["provisional_classification"] in positive_labels
            for row in accepted_review
        )
        accepted_gold = {
            pair
            for pair in gold_positive
            if float(gold_scores[pair].get("exact_jaccard_word5") or 0.0) >= 0.85
            and (
                not short_guard
                or min(
                    int(gold_scores[pair]["words_a"]),
                    int(gold_scores[pair]["words_b"]),
                )
                >= 20
            )
            and (
                not distinctive_guard
                or float(gold_scores[pair].get("distinctive_jaccard_word5") or 0.0)
                >= 0.75
            )
        }
        rows.append(
            {
                "policy": name,
                "reference_threshold": 0.85,
                "review_pairs_accepted": len(accepted_review),
                "review_positive_triage_pairs": accepted_positive,
                "parlamento_diagnostic_pairs_at_or_above_0_80": sum(
                    "parlamento_pt" in {row["source_a"], row["source_b"]}
                    and float(row.get("exact_jaccard_word5") or 0.0) >= 0.80
                    for row in review
                ),
                "review_precision_against_provisional_triage": (
                    accepted_positive / len(accepted_review)
                    if accepted_review
                    else None
                ),
                "review_recall_against_provisional_triage": (
                    accepted_positive / review_positive_total
                    if review_positive_total
                    else None
                ),
                "gold_similarity_positives_at_0_85": len(gold_positive),
                "gold_similarity_positives_retained": len(accepted_gold),
                "gold_similarity_recall": (
                    len(accepted_gold) / len(gold_positive) if gold_positive else None
                ),
                "containment_diagnostic_pair_count": (
                    sum(bool(row.get("containment_review_flag")) for row in scored)
                    if containment_diag
                    else 0
                ),
                "gold_containment_diagnostic_pair_count": (
                    sum(
                        bool(row.get("containment_review_flag"))
                        for pair, row in gold_scores.items()
                        if pair in gold_pairs
                    )
                    if containment_diag
                    else 0
                ),
                "containment_is_delete_rule": False,
                "precision_scope": "small, family-enriched provisional review artifact",
                "population_prevalence_inference": False,
                "complexity": {
                    "P1_global_jaccard": "one 5-gram MinHash and LSH pass",
                    "P2_jaccard_short_guard": "P1 plus an inexpensive word-count guard",
                    "P3_short_guard_containment_diagnostic": "P2 plus exact directional overlap on candidates",
                    "P4_short_guard_distinctive_and_containment": "P3 plus document-frequency counts and distinctive overlap",
                }[name],
                "cpu_memory_storage_implications": {
                    "P1_global_jaccard": "lowest; 256-value signature is about 2 KiB per row before compression",
                    "P2_jaccard_short_guard": "same signature and index cost as P1",
                    "P3_short_guard_containment_diagnostic": "small candidate-pair scoring cost; no additional full-corpus signature",
                    "P4_short_guard_distinctive_and_containment": "highest; needs shingle document-frequency aggregation and extra pair scoring",
                }[name],
                "major_failure_modes": {
                    "P1_global_jaccard": "template boilerplate false positives; short edits and containment misses",
                    "P2_jaccard_short_guard": "template boilerplate remains; short near copies are missed by automatic scoring",
                    "P3_short_guard_containment_diagnostic": "boilerplate remains; containment can mistake quotations or excerpts for duplicates",
                    "P4_short_guard_distinctive_and_containment": "sample-local frequency can misidentify boilerplate; threshold and source effects remain uncalibrated",
                }[name],
                "scientific_risks": "metrics are from enriched calibration families and cannot estimate corpus prevalence",
            }
        )
    return rows


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": compute_file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]


def verify_calibration_manifest(output_root: Path | str) -> list[str]:
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
        errors.append("calibration manifest status is not COMPLETE")
    for artifact in manifest.get("artifacts", []):
        relative = str(artifact.get("path") or "")
        path = (root / relative).resolve()
        if not relative or root.resolve() not in path.parents:
            errors.append(f"unsafe artifact path: {relative!r}")
            continue
        if not path.is_file():
            errors.append(f"artifact is missing: {relative}")
            continue
        if path.stat().st_size != int(artifact.get("bytes", -1)):
            errors.append(f"artifact size mismatch: {relative}")
        elif compute_file_sha256(path) != artifact.get("sha256"):
            errors.append(f"artifact checksum mismatch: {relative}")
    return errors


def augment_gold_with_d2_positive_anchors(
    output_root: Path | str,
    *,
    base_pilot_root: Path | str = DEFAULT_BASE_PILOT_ROOT,
) -> dict[str, Any]:
    """Add prior D2 real high-Jaccard pairs to the exhaustive-gold accounting.

    The prior D2 review artifact is an independent source of a true-candidate
    anchor. It is explicitly reported as an anchor and does not turn the
    family-enriched sample into a prevalence estimate.
    """
    root = Path(output_root).resolve()
    base_root = Path(base_pilot_root).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing_gold = pq.read_table(root / "gold_pairs.parquet").to_pylist()
    gold_pairs: dict[tuple[str, str], set[str]] = {
        tuple(sorted((str(row["id_a"]), str(row["id_b"])))): set(
            row.get("gold_family_ids") or []
        )
        for row in existing_gold
    }
    anchors: list[tuple[tuple[str, str], str]] = []
    for row in pq.read_table(base_root / "review_pairs.parquet").to_pylist():
        exact = row.get("exact_jaccard_word5")
        if exact is None or float(exact) < 0.80:
            continue
        if "parlamento_pt" in {row.get("source_a"), row.get("source_b")}:
            continue
        pair = tuple(sorted((str(row["id_a"]), str(row["id_b"]))))
        family_id = (
            "d2_real_high_similarity_anchor:"
            + hashlib.sha256(f"{pair[0]}\0{pair[1]}".encode("utf-8")).hexdigest()[:20]
        )
        if family_id not in gold_pairs.setdefault(pair, set()):
            gold_pairs[pair].add(family_id)
            anchors.append((pair, family_id))

    if not anchors:
        return manifest

    scored_rows = pq.read_table(root / "scored_pairs.parquet").to_pylist()
    score_by_pair = {(str(row["id_a"]), str(row["id_b"])): row for row in scored_rows}
    for pair, _family_id in anchors:
        if pair not in score_by_pair:
            raise ValueError(f"D2 positive anchor is missing from D2b scores: {pair}")
    for pair, score in score_by_pair.items():
        families = gold_pairs.get(pair)
        if families:
            score["exhaustive_gold"] = True
            score["gold_family_ids"] = sorted(families)

    candidate_rows = pq.read_table(root / "candidate_pairs.parquet").to_pylist()
    candidate_by_pair = {
        (str(row["id_a"]), str(row["id_b"])): row for row in candidate_rows
    }
    lsh_candidates: dict[tuple[str, str], set[str]] = {}
    for pair, row in candidate_by_pair.items():
        generators = set(row.get("generator_configs") or [])
        if generators:
            lsh_candidates[pair] = generators
        if pair in gold_pairs:
            row["exhaustive_gold"] = True

    gold_rows = []
    for pair, family_ids in sorted(gold_pairs.items()):
        row = dict(score_by_pair[pair])
        row["gold_family_ids"] = sorted(family_ids)
        row["gold_subset_scope"] = (
            "prior D2 real high-similarity positive anchor; rescored exhaustively in D2b"
            if any(
                item.startswith("d2_real_high_similarity_anchor:")
                for item in family_ids
            )
            else "all unordered pairs within a selected D2b metadata family"
        )
        gold_rows.append(row)

    gold_scores = {pair: score_by_pair[pair] for pair in gold_pairs}
    review_rows = pq.read_table(root / "human_review_pairs.parquet").to_pylist()
    recall_rows = _threshold_rows(gold_pairs, gold_scores, lsh_candidates)
    policy_rows = _policy_rows(scored_rows, review_rows, gold_pairs, gold_scores)
    _write_parquet(root / "scored_pairs.parquet", scored_rows)
    _write_parquet(root / "candidate_pairs.parquet", candidate_rows)
    _write_parquet(root / "gold_pairs.parquet", gold_rows)
    _write_csv(root / "lsh_recall_by_threshold.csv", recall_rows)
    _write_csv(root / "policy_decision_matrix.csv", policy_rows)

    exact_positive_counts = {
        representation: {
            f"{threshold:.2f}": sum(
                1
                for row in gold_rows
                if row.get(f"exact_jaccard_{representation}") is not None
                and float(row[f"exact_jaccard_{representation}"]) >= threshold
            )
            for threshold in EXACT_THRESHOLDS
        }
        for representation in ("word3", "word5", "word7")
    }
    gold_record_ids = {record_id for pair in gold_pairs for record_id in pair}
    anchor_count = sum(
        family_id.startswith("d2_real_high_similarity_anchor:")
        for family_ids in gold_pairs.values()
        for family_id in family_ids
    )
    manifest["gold_subset"].update(
        {
            "family_group_count": (
                int(manifest["gold_subset"]["family_group_count"]) + anchor_count
            ),
            "record_count": len(gold_record_ids),
            "exhaustive_pair_count": len(gold_rows),
            "D2_real_high_similarity_anchor_count": anchor_count,
            "exact_positive_pairs_by_representation_and_threshold": exact_positive_counts,
            "scope": (
                "all unordered pairs within selected metadata families plus "
                "independently identified D2 high-Jaccard real-pair anchors"
            ),
        }
    )
    manifest["lsh"]["recall_by_threshold_artifact"] = (
        "lsh_recall_by_threshold.csv; denominator includes explicitly marked D2 anchors"
    )
    resource_report = dict(manifest["resource_report"])
    resource_report["exhaustive_gold_pair_count"] = len(gold_rows)
    resource_report["D2_real_high_similarity_anchor_count"] = anchor_count
    (root / "resource_report.json").write_text(
        json.dumps(resource_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest["resource_report"] = resource_report
    manifest["artifacts"] = _artifact_inventory(root)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    errors = verify_calibration_manifest(root)
    if errors:
        raise ValueError(
            "D2 anchor augmentation failed verification: " + "; ".join(errors)
        )
    return manifest


def run_near_dedup_calibration(
    *,
    input_root: Path | str = near.DEFAULT_EXACT_DATA_ROOT,
    base_pilot_root: Path | str = DEFAULT_BASE_PILOT_ROOT,
    output_root: Path | str = DEFAULT_CALIBRATION_ROOT,
    seed: int = DEFAULT_CALIBRATION_SEED,
    require_full_coverage: bool = True,
) -> dict[str, Any]:
    """Run targeted D2b calibration without changing production corpus rows."""
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    exact_root, _data_root, _manifest_path, exact_manifest, exact_manifest_sha = (
        near._load_exact_root(Path(input_root))
    )
    base_root = Path(base_pilot_root).resolve()
    base_errors = near.verify_near_pilot_manifest(base_root)
    if base_errors:
        raise ValueError("D2 base pilot verification failed: " + "; ".join(base_errors))
    base_manifest = json.loads(
        (base_root / "manifest.json").read_text(encoding="utf-8")
    )
    if base_manifest.get("input_exact_manifest_sha256") != exact_manifest_sha:
        raise ValueError("D2 base pilot and selected exact corpus manifest differ")
    if base_manifest.get("production", {}).get("near_dedup_production") != "NOT RUN":
        raise ValueError("D2 base pilot does not attest that production was not run")

    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite D2b output: {output}")
    if (
        output == exact_root
        or exact_root in output.parents
        or output in exact_root.parents
    ):
        raise ValueError("D2b output must be separate from the exact corpus")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    connection = None
    try:
        files = near._iter_data_files(exact_root)
        families, scan_metrics = _scan_metadata_families(files, seed=seed)
        base_records_path = base_root / "pilot_records.parquet"
        base_ids: set[str] = set()
        for batch in pq.ParquetFile(base_records_path).iter_batches(
            batch_size=512, columns=["pilot_occurrence_id"]
        ):
            base_ids.update(map(str, batch.column(0).to_pylist()))
        target_records = {
            record_id: record
            for record_id, record in families["target_records"].items()
            if record_id not in base_ids
        }
        materialize_metrics = _materialize_calibration_records(
            files=files,
            base_records_path=base_records_path,
            target_records=target_records,
            target_tags=families["target_tags"],
            output_path=stage / "calibration_records.parquet",
        )
        calibration_path = stage / "calibration_records.parquet"
        signatures_path = stage / "minhash_signatures.parquet"
        metadata, signatures, signature_metrics = near._write_signatures(
            calibration_path, signatures_path, seed=seed
        )
        for record_id, row in metadata.items():
            row["calibration_families"] = []
        for batch in pq.ParquetFile(calibration_path).iter_batches(
            batch_size=512,
            columns=["pilot_occurrence_id", "calibration_families"],
        ):
            for record_id, tags in zip(
                batch.column(0).to_pylist(),
                batch.column(1).to_pylist(),
                strict=True,
            ):
                metadata[str(record_id)]["calibration_families"] = tags or []
        texts = _read_texts(calibration_path)
        if set(metadata) != set(texts):
            raise ValueError("Calibration record metadata/text ID coverage differs")

        df_common, df_metrics = _document_frequency_reference(
            metadata, texts, seed=seed
        )
        connection = near._create_lsh_database(stage / ".near-calibration.sqlite3")
        lsh_candidates, lsh_metrics = _collect_lsh_candidates(
            connection, signatures, seed
        )
        d2_candidate_anchor_pairs = 0
        for row in pq.read_table(base_root / "candidate_pairs.parquet").to_pylist():
            pair = tuple(sorted((str(row["id_a"]), str(row["id_b"]))))
            generators = set(row.get("generator_configs") or [])
            if generators:
                lsh_candidates.setdefault(pair, set()).update(generators)
            d2_candidate_anchor_pairs += 1
        d2_review_anchor_pairs = 0
        for row in pq.read_table(base_root / "review_pairs.parquet").to_pylist():
            pair = tuple(sorted((str(row["id_a"]), str(row["id_b"]))))
            d2_review_anchor_pairs += 1
        combined_candidates: dict[tuple[str, str], dict[str, set[str]]] = {}
        for pair, generators in lsh_candidates.items():
            combined_candidates[pair] = {
                "generators": set(generators),
                "reasons": set(),
                "family_ids": set(),
            }
        for pair, family_item in families["candidate_pairs"].items():
            current = combined_candidates.setdefault(
                pair, {"generators": set(), "reasons": set(), "family_ids": set()}
            )
            current["reasons"].update(family_item["reasons"])
            current["family_ids"].update(family_item["family_ids"])
        for row in pq.read_table(base_root / "candidate_pairs.parquet").to_pylist():
            pair = tuple(sorted((str(row["id_a"]), str(row["id_b"]))))
            current = combined_candidates.setdefault(
                pair, {"generators": set(), "reasons": set(), "family_ids": set()}
            )
            current["generators"].update(row.get("generator_configs") or [])
            current["reasons"].add("d2_candidate_anchor")
            current["reasons"].update(
                f"d2_{reason}" for reason in row.get("enrichment_reasons") or []
            )
        for row in pq.read_table(base_root / "review_pairs.parquet").to_pylist():
            pair = tuple(sorted((str(row["id_a"]), str(row["id_b"]))))
            current = combined_candidates.setdefault(
                pair, {"generators": set(), "reasons": set(), "family_ids": set()}
            )
            current["reasons"].add("d2_review_anchor")

        all_pairs = set(combined_candidates) | set(families["gold_pairs"])
        scored: list[dict[str, Any]] = []
        score_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
        for id_a, id_b in sorted(all_pairs):
            row = _score_pair(
                id_a, id_b, metadata, texts, signatures, df_common, seed=seed
            )
            candidate = combined_candidates.get(
                (id_a, id_b), {"generators": set(), "reasons": set()}
            )
            row.update(
                {
                    "lsh_candidate": bool(candidate.get("generators")),
                    "family_candidate": bool(candidate.get("reasons")),
                    "exhaustive_gold": (id_a, id_b) in families["gold_pairs"],
                    "generator_configs": sorted(candidate.get("generators", set())),
                    "family_reasons": sorted(candidate.get("reasons", set())),
                    "gold_family_ids": sorted(
                        families["gold_pairs"].get((id_a, id_b), set())
                    ),
                    "would_remove": False,
                }
            )
            scored.append(row)
            score_by_pair[(id_a, id_b)] = row
        gold_scores = {pair: score_by_pair[pair] for pair in families["gold_pairs"]}
        review = _human_review_rows(
            scored,
            combined_candidates,
            families["gold_pairs"],
            metadata,
            texts,
            seed=seed,
        )

        candidate_rows = []
        for pair, item in sorted(combined_candidates.items()):
            row_a, row_b = metadata[pair[0]], metadata[pair[1]]
            candidate_rows.append(
                {
                    "id_a": pair[0],
                    "id_b": pair[1],
                    "generator_configs": sorted(item["generators"]),
                    "family_reasons": sorted(item["reasons"]),
                    "family_ids": sorted(item["family_ids"]),
                    "lsh_generated": bool(item["generators"]),
                    "exhaustive_gold": pair in families["gold_pairs"],
                    "source_a": row_a["source"],
                    "subset_a": row_a["subset"],
                    "source_b": row_b["source"],
                    "subset_b": row_b["subset"],
                    "words_a": int(row_a["normalized_words"]),
                    "words_b": int(row_b["normalized_words"]),
                    "would_remove": False,
                }
            )
        gold_rows = [
            {
                **score_by_pair[pair],
                "gold_family_ids": sorted(family_ids),
                "gold_subset_scope": "all unordered pairs within selected metadata family",
            }
            for pair, family_ids in sorted(families["gold_pairs"].items())
        ]
        recall_rows = _threshold_rows(
            families["gold_pairs"], gold_scores, lsh_candidates
        )
        length_rows = _length_ratio_rows(scored)
        short_rows = _short_document_rows(scored)
        containment_rows = _containment_rows(scored)
        configuration_rows = _configuration_rows(
            scored, lsh_metrics, signature_metrics=signature_metrics
        )
        policy_rows = _policy_rows(scored, review, families["gold_pairs"], gold_scores)
        boilerplate_rows = []
        for row in scored:
            if (
                float(row.get("exact_jaccard_word5") or 0.0) >= 0.60
                or float(row.get("common_fraction_of_shared_word5") or 0.0) >= 0.50
                or float(row.get("distinctive_jaccard_word5") or 0.0) < 0.30
            ):
                boilerplate_rows.append(
                    {
                        "id_a": row["id_a"],
                        "id_b": row["id_b"],
                        "source_a": row["source_a"],
                        "subset_a": row["subset_a"],
                        "source_b": row["source_b"],
                        "subset_b": row["subset_b"],
                        "words_a": row["words_a"],
                        "words_b": row["words_b"],
                        "length_ratio_min_max": row["length_ratio_min_max"],
                        "exact_jaccard_word5": row["exact_jaccard_word5"],
                        "distinctive_jaccard_word5": row["distinctive_jaccard_word5"],
                        "distinctive_shared_shingles_word5": row[
                            "distinctive_shared_shingles_word5"
                        ],
                        "common_fraction_of_shared_word5": row[
                            "common_fraction_of_shared_word5"
                        ],
                        "family_reasons": row["family_reasons"],
                        "review_label": _provisional_label(
                            row, metadata, row["family_reasons"]
                        ),
                    }
                )
        boilerplate_rows.append(_synthetic_boilerplate_row(seed))

        _write_parquet(stage / "candidate_pairs.parquet", candidate_rows)
        _write_parquet(stage / "scored_pairs.parquet", scored)
        _write_parquet(stage / "gold_pairs.parquet", gold_rows)
        _write_parquet(stage / "human_review_pairs.parquet", review)
        _write_csv(stage / "lsh_recall_by_threshold.csv", recall_rows)
        _write_csv(stage / "length_ratio_analysis.csv", length_rows)
        _write_csv(stage / "short_document_analysis.csv", short_rows)
        _write_csv(stage / "containment_analysis.csv", containment_rows)
        _write_csv(stage / "boilerplate_analysis.csv", boilerplate_rows)
        _write_csv(stage / "configuration_summary.csv", configuration_rows)
        _write_csv(stage / "policy_decision_matrix.csv", policy_rows)
        connection.close()
        connection = None
        (stage / ".near-calibration.sqlite3").unlink(missing_ok=True)

        reason_counts = Counter(
            reason
            for item in combined_candidates.values()
            for reason in item["reasons"]
        )
        label_counts = Counter(row["provisional_classification"] for row in review)
        source_counts = Counter(row["source"] for row in metadata.values())
        subset_counts = Counter(
            f"{row['source']}/{row['subset']}" for row in metadata.values()
        )
        gold_counts = {
            representation: {
                f"{threshold:.2f}": sum(
                    1
                    for row in gold_rows
                    if row.get(f"exact_jaccard_{representation}") is not None
                    and row[f"exact_jaccard_{representation}"] >= threshold
                )
                for threshold in EXACT_THRESHOLDS
            }
            for representation in ("word3", "word5", "word7")
        }
        resource_report = {
            "measurement_scope": "this D2b calibration process and selected artifacts",
            "wall_seconds_total": round(time.perf_counter() - started_wall, 3),
            "cpu_seconds_total": round(time.process_time() - started_cpu, 3),
            "peak_rss_bytes": near._peak_rss_bytes(),
            "metadata_scan": scan_metrics,
            "materialized_records": materialize_metrics,
            "signatures": signature_metrics,
            "lsh": lsh_metrics,
            "document_frequency_reference": df_metrics,
            "scored_pair_count": len(scored),
            "exhaustive_gold_pair_count": len(gold_rows),
            "candidate_pair_count": len(candidate_rows),
            "d2_candidate_anchor_pairs": d2_candidate_anchor_pairs,
            "d2_review_anchor_pairs": d2_review_anchor_pairs,
            "artifact_bytes_before_manifest": sum(
                item.stat().st_size
                for item in stage.rglob("*")
                if item.is_file() and not item.name.startswith(".")
            ),
        }
        (stage / "resource_report.json").write_text(
            json.dumps(resource_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        production_states = {
            "near_dedup_production": "NOT RUN",
            "benchmark_decontamination": "PENDING",
            "C1": "IN PROGRESS",
            "C2": "PENDING",
            "final_training_token_budget": "NOT CHOSEN",
            "parlamento_pt": "preserve all; diagnostic only",
        }
        manifest = {
            "schema_version": 1,
            "calibration_version": CALIBRATION_VERSION,
            "status": "COMPLETE",
            "gate": "C1",
            "task": "D2b targeted scientific near-dedup calibration",
            "input_exact_root": str(exact_root),
            "input_exact_manifest_sha256": exact_manifest_sha,
            "input_exact_dedup_version": exact_manifest.get("exact_dedup_version"),
            "input_retained_records": scan_metrics["records_scanned"],
            "base_d2_pilot_root": str(base_root),
            "base_d2_manifest_sha256": compute_file_sha256(base_root / "manifest.json"),
            "seed": seed,
            "sampling_design": {
                "D2_base_records": materialize_metrics["d2_base_records"],
                "targeted_family_records": materialize_metrics["targeted_records"],
                "candidate_families": dict(sorted(reason_counts.items())),
                "family_key_counts_after_deterministic_sampling": families[
                    "family_key_counts"
                ],
                "neighbor_pair_candidates_seen": families[
                    "neighbor_pair_candidates_seen"
                ],
                "source_counts": dict(sorted(source_counts.items())),
                "source_subset_counts": dict(sorted(subset_counts.items())),
                "all_gigaverbo_subsets_covered": (
                    set(near.EXPECTED_GIGAVERBO_SUBSETS)
                    <= {
                        subset
                        for source_subset in subset_counts
                        if source_subset.startswith("gigaverbo_v2/")
                        for subset in [source_subset.split("/", 1)[1]]
                    }
                ),
                "population_prevalence_inference": False,
            },
            "gold_subset": {
                "family_group_count": len(families["gold_groups"]),
                "record_count": len(
                    {
                        row["pilot_occurrence_id"]
                        for group in families["gold_groups"]
                        for row in group["members"]
                    }
                ),
                "exhaustive_pair_count": len(gold_rows),
                "exact_positive_pairs_by_representation_and_threshold": gold_counts,
                "scope": "every unordered pair within each selected title, URL, domain, or crawl-neighbor family",
            },
            "lsh": {
                "candidate_pairs_across_configurations": sum(
                    bool(value) for value in lsh_candidates.values()
                ),
                "D2_candidate_anchor_pairs": d2_candidate_anchor_pairs,
                "D2_review_anchor_pairs": d2_review_anchor_pairs,
                "configurations": [item.name for item in near.LSH_CONFIGS],
                "thresholds": list(EXACT_THRESHOLDS),
                "recall_by_threshold_artifact": "lsh_recall_by_threshold.csv",
            },
            "human_review": {
                "pair_count": len(review),
                "provisional_label_counts": dict(sorted(label_counts.items())),
                "labels_are_not_automated_truth": True,
                "human_label_column_blank_for_review": True,
            },
            "ownership": {
                "frozen": False,
                "policy": "provisional provenance hierarchy; no row dispositions emitted",
                "wikipedia_carolina": "Wikipedia PT is preferred when Carolina metadata identifies the row as Wikipedia-derived",
                "deterministic_final_tie_break": "occurrence ID after provenance and documented GigaVerbo subset tier",
            },
            "parlamento_pt": {
                "sampled_rows": sum(
                    row["source"] == "parlamento_pt" for row in metadata.values()
                ),
                "rows_marked_removed": 0,
                "production_policy": "preserve all; diagnostic only",
            },
            "production": production_states,
            "resource_report": resource_report,
            "artifacts": _artifact_inventory(stage),
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite D2b output: {output}")
        os.replace(stage, output)
        augment_gold_with_d2_positive_anchors(output, base_pilot_root=base_root)
        errors = verify_calibration_manifest(output)
        if errors:
            raise ValueError("D2b artifact verification failed: " + "; ".join(errors))
        return json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    except BaseException:
        if connection is not None:
            connection.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise
