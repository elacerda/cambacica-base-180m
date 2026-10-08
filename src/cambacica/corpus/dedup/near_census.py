"""Read-only, staged full-corpus MinHash/LSH census for Gate C1 D2c.

The census fingerprints records and measures LSH candidate populations.  It
never selects owners, edits the exact corpus, or writes a retained corpus.
Stages are published as complete directories with checksummed manifests so a
later command can resume from the last completed stage.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import ExitStack, closing, contextmanager
from datetime import datetime, timezone
import argparse
import csv
import fcntl
import hashlib
import heapq
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import struct
import tempfile
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import urlsplit
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xxhash

from cambacica.corpus.dedup import near_pilot as near
from cambacica.corpus.manifest import compute_file_sha256


CENSUS_VERSION = "1.0.0"
DEFAULT_INPUT_ROOT = Path("/mnt/data/cambacica-base-180m/deduplicated/exact")
DEFAULT_OUTPUT_ROOT = Path("/mnt/data/cambacica-base-180m/dedup-census/near-v1")
DEFAULT_SCRATCH_ROOT = Path("/tmp/cambacica-near-census")
EXPECTED_EXACT_MANIFEST_SHA256 = (
    "57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428"
)
EXPECTED_RECORD_COUNT = 21_603_689
SEED = 20261006
SIGNATURE_CONFIG = near.MinHashConfig("word5_256", 5, 128, seed=SEED)
LSH_CONFIGS = (
    near.LSHConfig("word5_128_32x4", "word5_256", 128, 32, 4),
    near.LSHConfig("word5_128_8x16", "word5_256", 128, 8, 16),
)
LSH_CONFIG_MASK = {item.name: 1 << index for index, item in enumerate(LSH_CONFIGS)}
BUCKET_MEMBER_CAP = 256
FINGERPRINT_BATCH_SIZE = 128
EXACT_SAMPLE_SEED = 0xD2C20261006
RARE_SOURCE_PAIR_ALL_LIMIT = 24
REVIEW_PANEL_LIMIT = 300
DOMAIN_STRATA_COUNT = 64
EXACT_MEMORY_SHINGLE_LIMIT = 300_000
EXTERNAL_SORT_SHINGLE_BATCH = 100_000
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
SIGNATURE_BYTES = SIGNATURE_CONFIG.num_permutations * 8

LENGTH_BANDS = (
    ("lt20", 0, 20),
    ("20_99", 20, 100),
    ("100_999", 100, 1_000),
    ("1000_99999", 1_000, 100_000),
    ("ge100000", 100_000, None),
)
SIMILARITY_BANDS = (
    ("lt0.70", 0.0, 0.70),
    ("0.70_0.80", 0.70, 0.80),
    ("0.80_0.85", 0.80, 0.85),
    ("0.85_0.90", 0.85, 0.90),
    ("0.90_0.92", 0.90, 0.92),
    ("0.92_0.95", 0.92, 0.95),
    ("ge0.95", 0.95, 1.0000001),
)
EXACT_TAIL_BANDS = (
    ("0.80_0.85", 0.80, 0.85),
    ("0.85_0.90", 0.85, 0.90),
    ("0.90_0.92", 0.90, 0.92),
    ("0.92_0.95", 0.92, 0.95),
    ("ge0.95", 0.95, 1.0000001),
)
LENGTH_RATIO_BANDS = (
    ("lt0.10", 0.0, 0.10),
    ("0.10_0.25", 0.10, 0.25),
    ("0.25_0.50", 0.25, 0.50),
    ("0.50_0.75", 0.50, 0.75),
    ("0.75_0.90", 0.75, 0.90),
    ("0.90_1.00", 0.90, 1.0000001),
)
BUCKET_SIZE_BANDS = (
    ("2_5", 2, 6),
    ("6_20", 6, 21),
    ("21_64", 21, 65),
    ("65_256", 65, 257),
    ("gt256", 257, None),
)


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _json_read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_id(relative_path: str, row_ordinal: int) -> bytes:
    payload = f"near-census-v1\0{relative_path}\0{row_ordinal}".encode("utf-8")
    return hashlib.sha256(payload).digest()[:16]


def occurrence_id_hex(relative_path: str, row_ordinal: int) -> str:
    """Return a stable, compact occurrence identifier for one exact row."""
    return _sha256_id(relative_path, row_ordinal).hex()


def length_band(word_count: int) -> str:
    """Return the required five-way normalized-word length class."""
    for name, lower, upper in LENGTH_BANDS:
        if word_count >= lower and (upper is None or word_count < upper):
            return name
    raise AssertionError("unreachable length band")


def similarity_band(similarity: float) -> str:
    """Map a MinHash estimate or exact Jaccard to a named half-open band."""
    for name, lower, upper in SIMILARITY_BANDS:
        if lower <= similarity < upper:
            return name
    return "ge0.95" if similarity >= 1.0 else "lt0.70"


def exact_tail_band(similarity: float) -> str | None:
    for name, lower, upper in EXACT_TAIL_BANDS:
        if lower <= similarity < upper:
            return name
    return None


def length_ratio_band(ratio: float) -> str:
    for name, lower, upper in LENGTH_RATIO_BANDS:
        if lower <= ratio < upper:
            return name
    return "0.90_1.00" if ratio >= 1.0 else "lt0.10"


def bucket_size_band(size: int) -> str:
    for name, lower, upper in BUCKET_SIZE_BANDS:
        if size >= lower and (upper is None or size < upper):
            return name
    return "2_5"


def containment_flags(
    *,
    jaccard: float,
    containment_a_in_b: float,
    containment_b_in_a: float,
    length_ratio: float,
    shared_shingles: int,
) -> list[str]:
    """Return diagnostic containment flags; these never imply deletion."""
    maximum_containment = max(containment_a_in_b, containment_b_in_a)
    flags: list[str] = []
    if maximum_containment >= 0.90 and jaccard < 0.80:
        flags.append("containment_ge0.90_jaccard_lt0.80")
    if maximum_containment >= 0.95 and length_ratio < 0.50:
        flags.append("containment_ge0.95_length_ratio_lt0.50")
    if shared_shingles >= 100 and jaccard < 0.80:
        flags.append("shared_ge100_jaccard_lt0.80")
    return flags


def _domain(url: str | None) -> str:
    if not url:
        return ""
    try:
        host = (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""
    return host.removeprefix("www.")


def _source_subset(record: Mapping[str, Any]) -> str:
    source = str(record.get("source") or "")
    subset = str(record.get("subset") or "")
    return f"{source}/{subset}" if subset else source


def _relation(source_a: str, source_b: str) -> str:
    return "within_source" if source_a == source_b else "cross_source"


def _ratio(words_a: int, words_b: int) -> float:
    low, high = sorted((int(words_a), int(words_b)))
    if high == 0:
        return 1.0
    return low / high


class _FingerprintEngine:
    """Reuse the pilot's shingle/permutation family without per-row setup."""

    def __init__(self) -> None:
        self.a, self.b = near._permutation_coefficients(
            SIGNATURE_CONFIG.num_permutations,
            SIGNATURE_CONFIG.seed,
            SIGNATURE_CONFIG.name,
        )

    def signature(self, text: str) -> tuple[bytes, bool]:
        minima = np.full(128, np.iinfo(np.uint64).max, dtype=np.uint64)
        batch: list[int] = []

        def update(values: list[int]) -> None:
            if not values:
                return
            base = np.asarray(values, dtype=np.uint64)
            transformed = self.a[:, None] * base[None, :]
            transformed += self.b[:, None]
            minima[:] = np.minimum(minima, transformed.min(axis=1))

        has_shingles = False
        for shingle_hash in near.iter_word_shingle_hashes(
            text, SIGNATURE_CONFIG.ngram_size, seed=SIGNATURE_CONFIG.seed
        ):
            has_shingles = True
            batch.append(shingle_hash)
            if len(batch) >= near.SIGNATURE_BATCH_SHINGLES:
                update(batch)
                batch.clear()
        update(batch)
        return struct.pack("<128Q", *(int(value) for value in minima)), has_shingles


def _data_root_and_manifest(
    input_root: Path | str,
    expected_manifest_sha256: str | None = EXPECTED_EXACT_MANIFEST_SHA256,
) -> tuple[Path, Path, dict[str, Any], str]:
    requested = Path(input_root).resolve()
    if requested.name == "data" and requested.is_dir():
        data_root = requested
        exact_root = requested.parent
    else:
        exact_root = requested
        data_root = exact_root / "data"
    manifest_path = exact_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing exact-dedup manifest: {manifest_path}")
    manifest_sha = compute_file_sha256(manifest_path)
    if expected_manifest_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", expected_manifest_sha256
    ):
        raise ValueError(
            "Expected exact-manifest SHA-256 must be 64 lowercase hex characters"
        )
    if (
        expected_manifest_sha256 is not None
        and manifest_sha != expected_manifest_sha256
    ):
        raise ValueError(
            "Exact input manifest SHA-256 differs from the authoritative Gate C1 "
            f"value: {manifest_sha}"
        )
    manifest = _json_read(manifest_path)
    if manifest.get("status") != "COMPLETE" or manifest.get("run_type") != "production":
        raise ValueError("Near census requires a COMPLETE exact production manifest")
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Missing exact-deduplicated data directory: {data_root}"
        )
    return exact_root, data_root, manifest, manifest_sha


def _iter_data_files(data_root: Path) -> list[Path]:
    files = sorted(
        (path for path in data_root.rglob("*.parquet") if path.is_file()),
        key=lambda path: path.relative_to(data_root).as_posix(),
    )
    if not files:
        raise ValueError(f"No Parquet data found under {data_root}")
    return files


def _assert_output_separate(exact_root: Path, output_root: Path) -> None:
    resolved = output_root.resolve()
    exact = exact_root.resolve()
    if resolved == exact or exact in resolved.parents:
        raise ValueError("Census output must be outside the immutable exact corpus")


def _inventory(root: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": compute_file_sha256(path),
            }
        )
    return artifacts


def _verify_stage(stage_root: Path, *, check_hashes: bool) -> list[str]:
    manifest_path = stage_root / "manifest.json"
    if not manifest_path.is_file():
        return [f"{stage_root}: manifest.json is missing"]
    try:
        manifest = _json_read(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{stage_root}: manifest.json cannot be read: {exc}"]
    errors: list[str] = []
    if manifest.get("status") != "COMPLETE":
        errors.append(f"{stage_root}: stage status is not COMPLETE")
    for item in manifest.get("artifacts", []):
        artifact = stage_root / item["path"]
        if not artifact.is_file():
            errors.append(f"{stage_root}: missing artifact {item['path']}")
            continue
        if artifact.stat().st_size != item.get("bytes"):
            errors.append(f"{stage_root}: size mismatch for {item['path']}")
        elif check_hashes and compute_file_sha256(artifact) != item.get("sha256"):
            errors.append(f"{stage_root}: SHA-256 mismatch for {item['path']}")
    return errors


@contextmanager
def _stage_lock(output_root: Path, name: str) -> Iterator[None]:
    lock_root = output_root / ".locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"{name}.lock"
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Stage {name} is already running") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_stage_dir(stage_root: Path) -> Path:
    stage_root.parent.mkdir(parents=True, exist_ok=True)
    for partial in stage_root.parent.glob(f".{stage_root.name}.partial-*"):
        if partial.is_dir():
            shutil.rmtree(partial)
        else:
            partial.unlink(missing_ok=True)
    return Path(
        tempfile.mkdtemp(prefix=f".{stage_root.name}.partial-", dir=stage_root.parent)
    )


def _write_stage_manifest(
    stage_dir: Path,
    *,
    stage_name: str,
    input_manifest_sha256: str,
    input_signature_manifest_sha256: str | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "census_version": CENSUS_VERSION,
        "stage": stage_name,
        "status": "COMPLETE",
        "created_at_utc": _utc_now(),
        "input_exact_manifest_sha256": input_manifest_sha256,
        "input_signature_manifest_sha256": input_signature_manifest_sha256,
        "metrics": dict(metrics or {}),
        "artifacts": _inventory(stage_dir),
    }
    _json_write(stage_dir / "manifest.json", manifest)
    return manifest


def _update_root_manifest(
    output_root: Path,
    *,
    exact_root: Path,
    data_root: Path,
    exact_manifest_sha256: str,
    exact_manifest: Mapping[str, Any],
    status: str = "IN_PROGRESS",
) -> None:
    stage_status = {}
    for stage_name, relative in (
        ("fingerprints", "signatures"),
        ("lsh_index", "lsh/index"),
        ("candidates", "lsh/candidates"),
        ("candidate_summary", "candidate_summary"),
        ("exact_sample", "candidate_samples"),
    ):
        manifest_path = output_root / relative / "manifest.json"
        if manifest_path.is_file():
            try:
                stage_status[stage_name] = _json_read(manifest_path).get("status")
            except (OSError, json.JSONDecodeError):
                stage_status[stage_name] = "INVALID"
        else:
            stage_status[stage_name] = "NOT RUN"
    if all(value == "COMPLETE" for value in stage_status.values()):
        status = "COMPLETE"
    value = {
        "schema_version": 1,
        "census_version": CENSUS_VERSION,
        "status": status,
        "gate": "C1 / D2c",
        "measurement_only": True,
        "created_or_updated_at_utc": _utc_now(),
        "input_exact_root": str(exact_root),
        "input_exact_data_root": str(data_root),
        "input_exact_manifest_sha256": exact_manifest_sha256,
        "input_retained_record_count_expected": exact_manifest.get(
            "retained_record_count"
        ),
        "input_identity_verified_before_stage_work": True,
        "stage_status": stage_status,
        "candidate_generation": {
            "representation": "lowercased Unicode Python re \\w+ word 5-grams",
            "minhash_permutations": 128,
            "minhash_seed": SEED,
            "lsh_configurations": [
                {
                    "name": item.name,
                    "bands": item.bands,
                    "rows_per_band": item.rows_per_band,
                }
                for item in LSH_CONFIGS
            ],
            "bucket_member_cap": BUCKET_MEMBER_CAP,
        },
        "production_deletion_or_retained_corpus": "NOT PART OF THIS CENSUS",
        "near_dedup_production": "NOT RUN",
        "benchmark_decontamination": "PENDING",
        "C1": "IN PROGRESS",
        "C2": "PENDING",
    }
    temp_path = output_root / ".manifest.json.partial"
    _json_write(temp_path, value)
    os.replace(temp_path, output_root / "manifest.json")


def run_fingerprints(
    *,
    input_root: Path | str = DEFAULT_INPUT_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    expected_manifest_sha256: str | None = EXPECTED_EXACT_MANIFEST_SHA256,
    expected_record_count: int | None = EXPECTED_RECORD_COUNT,
    batch_size: int = FINGERPRINT_BATCH_SIZE,
) -> dict[str, Any]:
    """Fingerprint the complete exact corpus in one streaming pass."""
    exact_root, data_root, exact_manifest, exact_manifest_sha = _data_root_and_manifest(
        input_root, expected_manifest_sha256
    )
    output = Path(output_root).resolve()
    _assert_output_separate(exact_root, output)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    stage_root = output / "signatures"
    output.mkdir(parents=True, exist_ok=True)
    with _stage_lock(output, "fingerprints"):
        if stage_root.exists():
            errors = _verify_stage(stage_root, check_hashes=False)
            manifest = _json_read(stage_root / "manifest.json") if not errors else {}
            if (
                not errors
                and manifest.get("input_exact_manifest_sha256") == exact_manifest_sha
                and manifest.get("stage") == "fingerprints"
            ):
                _update_root_manifest(
                    output,
                    exact_root=exact_root,
                    data_root=data_root,
                    exact_manifest_sha256=exact_manifest_sha,
                    exact_manifest=exact_manifest,
                )
                print("[reuse] fingerprints stage is already COMPLETE", flush=True)
                return manifest
            raise FileExistsError(
                f"Fingerprint stage exists but does not match this input: {stage_root}"
            )
        stage_dir = _atomic_stage_dir(stage_root)
        started = time.perf_counter()
        files = _iter_data_files(data_root)
        engine = _FingerprintEngine()
        population_by_source_subset: Counter[tuple[str, str]] = Counter()
        population_by_length_band: Counter[str] = Counter()
        schema = pa.schema(
            [
                pa.field("record_id", pa.binary(16), nullable=False),
                pa.field("source", pa.string(), nullable=False),
                pa.field("subset", pa.string()),
                pa.field("normalized_words", pa.int64(), nullable=False),
                pa.field("length_band", pa.string(), nullable=False),
                pa.field("has_shingles", pa.bool_(), nullable=False),
                pa.field("data_relative_path", pa.string(), nullable=False),
                pa.field("row_ordinal", pa.int64(), nullable=False),
                pa.field("row_group", pa.int32(), nullable=False),
                pa.field("row_offset", pa.int32(), nullable=False),
                pa.field("original_id", pa.string()),
                pa.field("original_url", pa.string()),
                pa.field("domain", pa.string()),
                pa.field("title", pa.string()),
                pa.field("domain_category", pa.string()),
                pa.field("signature", pa.binary(SIGNATURE_BYTES), nullable=False),
            ]
        )
        parquet_path = stage_dir / "fingerprints.parquet"
        writer = pq.ParquetWriter(
            parquet_path,
            schema,
            compression="zstd",
            compression_level=6,
            use_dictionary=True,
            write_statistics=True,
        )
        record_count = 0
        empty_count = 0
        batch_rows: list[dict[str, Any]] = []
        try:
            for file_index, path in enumerate(files, start=1):
                relative = path.relative_to(data_root).as_posix()
                parquet = pq.ParquetFile(path)
                required_columns = [
                    "text",
                    "source",
                    "subset",
                    "original_id",
                    "original_url",
                    "title",
                    "domain_category",
                ]
                available = set(parquet.schema_arrow.names)
                columns = [name for name in required_columns if name in available]
                if "text" not in columns or "source" not in columns:
                    raise ValueError(f"Missing text/source columns in {path}")
                file_ordinal = 0
                for row_group in range(parquet.metadata.num_row_groups):
                    row_offset = 0
                    for batch in parquet.iter_batches(
                        batch_size=batch_size,
                        columns=columns,
                        row_groups=[row_group],
                    ):
                        values = batch.to_pylist()
                        for offset, row in enumerate(values):
                            text = row.get("text") or ""
                            words = near.count_normalized_words_bounded(text)
                            signature, has_shingles = engine.signature(text)
                            record_id = _sha256_id(relative, file_ordinal)
                            batch_rows.append(
                                {
                                    "record_id": record_id,
                                    "source": str(row.get("source") or "unknown"),
                                    "subset": row.get("subset"),
                                    "normalized_words": words,
                                    "length_band": length_band(words),
                                    "has_shingles": has_shingles,
                                    "data_relative_path": relative,
                                    "row_ordinal": file_ordinal,
                                    "row_group": row_group,
                                    "row_offset": row_offset + offset,
                                    "original_id": row.get("original_id"),
                                    "original_url": row.get("original_url"),
                                    "domain": _domain(row.get("original_url")),
                                    "title": row.get("title"),
                                    "domain_category": row.get("domain_category"),
                                    "signature": signature,
                                }
                            )
                            record_count += 1
                            empty_count += int(not has_shingles)
                            population_by_length_band[length_band(words)] += 1
                            subset_value = str(row.get("subset") or "")
                            source_value = str(row.get("source") or "unknown")
                            population_by_source_subset[
                                (source_value, subset_value)
                            ] += 1
                            file_ordinal += 1
                            if len(batch_rows) >= batch_size:
                                writer.write_table(
                                    pa.Table.from_pylist(batch_rows, schema=schema)
                                )
                                batch_rows.clear()
                        row_offset += batch.num_rows
                        if record_count and record_count % 100_000 == 0:
                            elapsed = max(time.perf_counter() - started, 0.001)
                            print(
                                f"[fingerprints] records={record_count:,} "
                                f"rate={record_count / elapsed:,.1f}/s "
                                f"file={file_index}/{len(files)}",
                                flush=True,
                            )
                print(
                    f"[fingerprints] completed {file_index}/{len(files)} "
                    f"{relative} rows={file_ordinal:,}",
                    flush=True,
                )
            if batch_rows:
                writer.write_table(pa.Table.from_pylist(batch_rows, schema=schema))
                batch_rows.clear()
        except BaseException:
            writer.close()
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise
        else:
            writer.close()
        declared_count = exact_manifest.get("retained_record_count")
        target_count = (
            expected_record_count
            if expected_record_count is not None
            else declared_count
        )
        if target_count is not None and record_count != int(target_count):
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise ValueError(
                f"Fingerprint row count {record_count} differs from expected {target_count}"
            )
        elapsed = time.perf_counter() - started
        _json_write(
            stage_dir / "resource_report.json",
            {
                "records": record_count,
                "files": len(files),
                "population_by_length_band": dict(
                    sorted(population_by_length_band.items())
                ),
                "population_by_source_subset": {
                    f"{source}/{subset}" if subset else source: count
                    for (source, subset), count in sorted(
                        population_by_source_subset.items()
                    )
                },
                "empty_or_no_word_shingle_records": empty_count,
                "elapsed_wall_seconds": round(elapsed, 3),
                "fingerprints_per_second": round(record_count / max(elapsed, 0.001), 2),
                "signature_raw_bytes_per_record": SIGNATURE_BYTES,
                "signature_raw_bytes_total": record_count * SIGNATURE_BYTES,
                "signature_parquet_bytes": parquet_path.stat().st_size,
                "peak_rss_bytes": near._peak_rss_bytes(),
                "input_data_bytes": sum(path.stat().st_size for path in files),
            },
        )
        manifest = _write_stage_manifest(
            stage_dir,
            stage_name="fingerprints",
            input_manifest_sha256=exact_manifest_sha,
            metrics={
                "records": record_count,
                "files": len(files),
                "population_by_length_band": dict(
                    sorted(population_by_length_band.items())
                ),
                "population_by_source_subset": {
                    f"{source}/{subset}" if subset else source: count
                    for (source, subset), count in sorted(
                        population_by_source_subset.items()
                    )
                },
            },
        )
        os.replace(stage_dir, stage_root)
        _update_root_manifest(
            output,
            exact_root=exact_root,
            data_root=data_root,
            exact_manifest_sha256=exact_manifest_sha,
            exact_manifest=exact_manifest,
        )
        return manifest


def _sqlite_connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-262144")
    connection.execute("PRAGMA mmap_size=1073741824")
    return connection


def _create_bucket_database(path: Path) -> sqlite3.Connection:
    connection = _sqlite_connect(path)
    connection.execute(
        "CREATE TABLE buckets ("
        "config_id INTEGER NOT NULL, band INTEGER NOT NULL, "
        "bucket_key BLOB NOT NULL, record_id BLOB NOT NULL, "
        "PRIMARY KEY(config_id, band, bucket_key, record_id)) WITHOUT ROWID"
    )
    return connection


def _bounded_bucket_members(
    member_ids: Iterable[bytes], cap: int
) -> tuple[list[bytes], int]:
    """Keep deterministic first members while counting an arbitrarily large bucket."""
    if cap < 1:
        raise ValueError("bucket cap must be positive")
    selected: list[bytes] = []
    size = 0
    for record_id in member_ids:
        size += 1
        if len(selected) < cap:
            selected.append(bytes(record_id))
    return selected, size


def _band_key_for_signature(
    signature: bytes, config: near.LSHConfig, band: int
) -> bytes:
    values = struct.unpack("<128Q", signature)
    return near._band_key(
        values,
        band * config.rows_per_band,
        config.rows_per_band,
        config.name,
        band,
    )


def run_lsh_index(
    *,
    input_root: Path | str = DEFAULT_INPUT_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    expected_manifest_sha256: str | None = EXPECTED_EXACT_MANIFEST_SHA256,
    expected_record_count: int | None = EXPECTED_RECORD_COUNT,
) -> dict[str, Any]:
    """Build a disk-backed, capped-ready bucket index from fingerprints."""
    exact_root, data_root, exact_manifest, exact_sha = _data_root_and_manifest(
        input_root, expected_manifest_sha256
    )
    output = Path(output_root).resolve()
    _assert_output_separate(exact_root, output)
    signature_root = output / "signatures"
    errors = _verify_stage(signature_root, check_hashes=False)
    if errors:
        raise ValueError("Fingerprint stage is not ready: " + "; ".join(errors))
    fingerprint_manifest = _json_read(signature_root / "manifest.json")
    if fingerprint_manifest.get("input_exact_manifest_sha256") != exact_sha:
        raise ValueError("Fingerprint stage belongs to a different exact corpus")
    fingerprint_path = signature_root / "fingerprints.parquet"
    stage_root = output / "lsh" / "index"
    output.mkdir(parents=True, exist_ok=True)
    with _stage_lock(output, "lsh-index"):
        if stage_root.exists():
            errors = _verify_stage(stage_root, check_hashes=False)
            manifest = _json_read(stage_root / "manifest.json") if not errors else {}
            if (
                not errors
                and manifest.get("input_exact_manifest_sha256") == exact_sha
                and manifest.get("input_signature_manifest_sha256")
                == compute_file_sha256(signature_root / "manifest.json")
            ):
                _update_root_manifest(
                    output,
                    exact_root=exact_root,
                    data_root=data_root,
                    exact_manifest_sha256=exact_sha,
                    exact_manifest=exact_manifest,
                )
                print("[reuse] LSH bucket index is already COMPLETE", flush=True)
                return manifest
            raise FileExistsError(
                f"LSH index exists but does not match input: {stage_root}"
            )
        stage_dir = _atomic_stage_dir(stage_root)
        started = time.perf_counter()
        index_path = stage_dir / "bucket_index.sqlite3"
        connection = _create_bucket_database(index_path)
        parquet = pq.ParquetFile(fingerprint_path)
        inserted = 0
        eligible = 0
        try:
            for batch_number, batch in enumerate(
                parquet.iter_batches(
                    batch_size=FINGERPRINT_BATCH_SIZE,
                    columns=["record_id", "has_shingles", "signature"],
                ),
                start=1,
            ):
                rows = batch.to_pylist()
                insert_rows: list[tuple[int, int, bytes, bytes]] = []
                for row in rows:
                    if not row["has_shingles"]:
                        continue
                    eligible += 1
                    record_id = row["record_id"]
                    signature = row["signature"]
                    for config_id, config in enumerate(LSH_CONFIGS):
                        for band in range(config.bands):
                            insert_rows.append(
                                (
                                    config_id,
                                    band,
                                    _band_key_for_signature(signature, config, band),
                                    record_id,
                                )
                            )
                if insert_rows:
                    connection.executemany(
                        "INSERT INTO buckets VALUES (?, ?, ?, ?)", insert_rows
                    )
                    inserted += len(insert_rows)
                if batch_number % 256 == 0:
                    connection.commit()
                    elapsed = max(time.perf_counter() - started, 0.001)
                    print(
                        f"[lsh-index] eligible={eligible:,} entries={inserted:,} "
                        f"rate={eligible / elapsed:,.1f} records/s",
                        flush=True,
                    )
            connection.commit()
        except BaseException:
            connection.close()
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise
        connection.close()
        declared_count = expected_record_count
        fingerprint_count = int(
            fingerprint_manifest.get("metrics", {}).get("records", 0)
        )
        if declared_count is not None and fingerprint_count != declared_count:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise ValueError(
                f"Fingerprint count {fingerprint_count} differs from expected {declared_count}"
            )
        elapsed = time.perf_counter() - started
        metrics = {
            "fingerprinted_records": fingerprint_count,
            "indexed_records_with_shingles": eligible,
            "empty_or_no_shingle_records_not_lsh_indexed": fingerprint_count - eligible,
            "bucket_index_entries": inserted,
            "bytes": index_path.stat().st_size,
            "elapsed_wall_seconds": round(elapsed, 3),
            "entries_per_second": round(inserted / max(elapsed, 0.001), 2),
            "peak_rss_bytes": near._peak_rss_bytes(),
            "bucket_member_cap": BUCKET_MEMBER_CAP,
        }
        _json_write(stage_dir / "resource_report.json", metrics)
        manifest = _write_stage_manifest(
            stage_dir,
            stage_name="lsh_index",
            input_manifest_sha256=exact_sha,
            input_signature_manifest_sha256=compute_file_sha256(
                signature_root / "manifest.json"
            ),
            metrics=metrics,
        )
        os.replace(stage_dir, stage_root)
        _update_root_manifest(
            output,
            exact_root=exact_root,
            data_root=data_root,
            exact_manifest_sha256=exact_sha,
            exact_manifest=exact_manifest,
        )
        return manifest


def _new_candidate_database(path: Path) -> sqlite3.Connection:
    connection = _sqlite_connect(path)
    connection.executescript(
        """
        CREATE TABLE pairs (
            id_a BLOB NOT NULL,
            id_b BLOB NOT NULL,
            config_mask INTEGER NOT NULL,
            bucket_hits INTEGER NOT NULL,
            max_bucket_size INTEGER NOT NULL,
            estimated_similarity REAL,
            PRIMARY KEY(id_a, id_b)
        ) WITHOUT ROWID;
        CREATE TABLE endpoint_ids (
            record_id BLOB PRIMARY KEY
        ) WITHOUT ROWID;
        CREATE TABLE records (
            record_id BLOB PRIMARY KEY,
            source TEXT NOT NULL,
            subset TEXT,
            normalized_words INTEGER NOT NULL,
            length_band TEXT NOT NULL,
            data_relative_path TEXT NOT NULL,
            row_ordinal INTEGER NOT NULL,
            row_group INTEGER NOT NULL,
            row_offset INTEGER NOT NULL,
            original_id TEXT,
            original_url TEXT,
            domain TEXT,
            title TEXT,
            domain_category TEXT
        ) WITHOUT ROWID;
        """
    )
    return connection


class _BloomFilter:
    """Fixed-memory Bloom filter used only to prefilter endpoint lookups."""

    def __init__(self, expected_items: int) -> None:
        target_bits = max(1 << 20, expected_items * 16)
        self.bit_count = min(1 << 31, 1 << (target_bits - 1).bit_length())
        self.data = bytearray(self.bit_count // 8)
        self.hash_count = 7

    def _positions(self, value: bytes) -> Iterator[int]:
        first = xxhash.xxh64(value, seed=0xA53C).intdigest()
        second = xxhash.xxh64(value, seed=0xD2C).intdigest() | 1
        mask = self.bit_count - 1
        for index in range(self.hash_count):
            yield (first + index * second) & mask

    def add(self, value: bytes) -> None:
        for position in self._positions(value):
            self.data[position >> 3] |= 1 << (position & 7)

    def might_contain(self, value: bytes) -> bool:
        return all(
            self.data[position >> 3] & (1 << (position & 7))
            for position in self._positions(value)
        )


def _source_pair_key(record_a: Mapping[str, Any], record_b: Mapping[str, Any]) -> str:
    key_a, key_b = sorted((_source_subset(record_a), _source_subset(record_b)))
    return f"{key_a} <> {key_b}"


def _candidate_join_query() -> str:
    return """
        SELECT p.id_a, p.id_b, p.config_mask, p.bucket_hits, p.max_bucket_size,
               p.estimated_similarity,
               a.source, a.subset, a.normalized_words, a.length_band,
               a.data_relative_path, a.row_ordinal, a.row_group, a.row_offset,
               a.original_id, a.original_url, a.domain, a.title, a.domain_category,
               b.source, b.subset, b.normalized_words, b.length_band,
               b.data_relative_path, b.row_ordinal, b.row_group, b.row_offset,
               b.original_id, b.original_url, b.domain, b.title, b.domain_category
        FROM pairs AS p
        JOIN records AS a ON a.record_id=p.id_a
        JOIN records AS b ON b.record_id=p.id_b
    """


def _unpack_candidate_row(row: Sequence[Any]) -> dict[str, Any]:
    names = (
        "id_a",
        "id_b",
        "config_mask",
        "bucket_hits",
        "max_bucket_size",
        "estimated_similarity",
        "source_a",
        "subset_a",
        "words_a",
        "length_band_a",
        "path_a",
        "ordinal_a",
        "row_group_a",
        "row_offset_a",
        "original_id_a",
        "url_a",
        "domain_a",
        "title_a",
        "domain_category_a",
        "source_b",
        "subset_b",
        "words_b",
        "length_band_b",
        "path_b",
        "ordinal_b",
        "row_group_b",
        "row_offset_b",
        "original_id_b",
        "url_b",
        "domain_b",
        "title_b",
        "domain_category_b",
    )
    row_dict = dict(zip(names, row))
    row_dict["id_a"] = bytes(row_dict["id_a"]).hex()
    row_dict["id_b"] = bytes(row_dict["id_b"]).hex()
    row_dict["candidate_configs"] = [
        config.name
        for index, config in enumerate(LSH_CONFIGS)
        if int(row_dict["config_mask"]) & (1 << index)
    ]
    row_dict["source_subset_a"] = _source_subset(
        {"source": row_dict["source_a"], "subset": row_dict["subset_a"]}
    )
    row_dict["source_subset_b"] = _source_subset(
        {"source": row_dict["source_b"], "subset": row_dict["subset_b"]}
    )
    row_dict["source_pair"] = _source_pair_key(
        {"source": row_dict["source_a"], "subset": row_dict["subset_a"]},
        {"source": row_dict["source_b"], "subset": row_dict["subset_b"]},
    )
    row_dict["pair_relation"] = _relation(
        str(row_dict["source_a"]), str(row_dict["source_b"])
    )
    row_dict["length_ratio"] = _ratio(row_dict["words_a"], row_dict["words_b"])
    row_dict["length_ratio_band"] = length_ratio_band(row_dict["length_ratio"])
    row_dict["estimated_similarity"] = float(row_dict["estimated_similarity"] or 0.0)
    row_dict["estimated_similarity_band"] = similarity_band(
        row_dict["estimated_similarity"]
    )
    row_dict["bucket_size_band"] = bucket_size_band(int(row_dict["max_bucket_size"]))
    row_dict["short_pair"] = (
        min(int(row_dict["words_a"]), int(row_dict["words_b"])) < 20
    )
    row_dict["parlamento_diagnostic_only"] = (
        row_dict["source_a"] == "parlamento_pt"
        or row_dict["source_b"] == "parlamento_pt"
    )
    row_dict["removal_eligible"] = False
    return row_dict


def _candidate_sample_strata(row: Mapping[str, Any]) -> list[tuple[str, str, int]]:
    """Return marginal strata and their deterministic bottom-k limits."""
    strata: list[tuple[str, str, int]] = [
        ("estimated_similarity", str(row["estimated_similarity_band"]), 120),
        ("source_subset_pair", str(row["source_pair"]), 24),
        ("relationship", str(row["pair_relation"]), 200),
        ("length_ratio", str(row["length_ratio_band"]), 100),
        (
            "length_class_pair",
            " <> ".join(sorted((str(row["length_band_a"]), str(row["length_band_b"])))),
            60,
        ),
        ("bucket_size", str(row["bucket_size_band"]), 100),
    ]
    for name in row["candidate_configs"]:
        strata.append(("configuration", name, 200))
    top_level_pair = " <> ".join(sorted((str(row["source_a"]), str(row["source_b"]))))
    strata.append(("top_level_source_pair", top_level_pair, 64))
    domains = (str(row.get("domain_a") or ""), str(row.get("domain_b") or ""))
    if domains[0] and domains[0] == domains[1]:
        domain_bin = (
            near.stable_hash64(domains[0], EXACT_SAMPLE_SEED) % DOMAIN_STRATA_COUNT
        )
        strata.append(("same_domain_hash_bin", str(domain_bin), 50))
    source_pair = {str(row["source_a"]), str(row["source_b"])}
    if "gigaverbo_v2" in source_pair and source_pair & {
        "carolina",
        "wikipedia_pt",
        "gutenberg_pt",
    }:
        strata.append(("curated_web", top_level_pair, 100))
    if source_pair == {"carolina", "wikipedia_pt"}:
        strata.append(("carolina_wikipedia", "carolina_wikipedia", 100))
    if source_pair == {"gutenberg_pt", "gigaverbo_v2"}:
        strata.append(("gutenberg_web", "gutenberg_web", 100))
    if source_pair == {"gigaverbo_v2"}:
        strata.append(("web_web", top_level_pair, 100))
    if int(row["max_bucket_size"]) > 20:
        strata.append(("wide_bucket", "bucket_size_gt20", 200))
    if int(row["max_bucket_size"]) >= 6 or int(row["bucket_hits"]) >= 3:
        strata.append(("boilerplate_bucket_proxy", "repeated_or_wide_bucket", 200))
    return strata


class _DeterministicReservoir:
    """A bottom-k sample whose result does not depend on input traversal order."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.seed = seed
        self.heap: list[tuple[int, str, tuple[str, str]]] = []

    def add(self, key: str, pair: tuple[str, str]) -> None:
        rank = near.stable_hash64(key, self.seed)
        entry = (-rank, key, pair)
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, entry)
            return
        worst = (-self.heap[0][0], self.heap[0][1])
        if (rank, key) < worst:
            heapq.heapreplace(self.heap, entry)

    def values(self) -> list[tuple[str, str]]:
        return [
            item
            for _rank, _key, item in sorted(
                self.heap, key=lambda row: (-row[0], row[1])
            )
        ]


def run_candidates(
    *,
    input_root: Path | str = DEFAULT_INPUT_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    expected_manifest_sha256: str | None = EXPECTED_EXACT_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Generate the capped LSH union and estimate candidate similarities."""
    exact_root, data_root, exact_manifest, exact_sha = _data_root_and_manifest(
        input_root, expected_manifest_sha256
    )
    output = Path(output_root).resolve()
    _assert_output_separate(exact_root, output)
    index_root = output / "lsh" / "index"
    index_errors = _verify_stage(index_root, check_hashes=False)
    signature_root = output / "signatures"
    signature_errors = _verify_stage(signature_root, check_hashes=False)
    if index_errors or signature_errors:
        raise ValueError(
            "LSH candidates require completed fingerprints and index: "
            + "; ".join(index_errors + signature_errors)
        )
    index_manifest = _json_read(index_root / "manifest.json")
    signature_manifest = _json_read(signature_root / "manifest.json")
    expected_signature_manifest_sha = compute_file_sha256(
        signature_root / "manifest.json"
    )
    if (
        index_manifest.get("input_exact_manifest_sha256") != exact_sha
        or index_manifest.get("input_signature_manifest_sha256")
        != expected_signature_manifest_sha
    ):
        raise ValueError("LSH index belongs to a different fingerprint stage")
    stage_root = output / "lsh" / "candidates"
    stage_root.parent.mkdir(parents=True, exist_ok=True)
    with _stage_lock(output, "candidates"):
        if stage_root.exists():
            errors = _verify_stage(stage_root, check_hashes=False)
            manifest = _json_read(stage_root / "manifest.json") if not errors else {}
            if (
                not errors
                and manifest.get("input_exact_manifest_sha256") == exact_sha
                and manifest.get("input_signature_manifest_sha256")
                == expected_signature_manifest_sha
            ):
                _update_root_manifest(
                    output,
                    exact_root=exact_root,
                    data_root=data_root,
                    exact_manifest_sha256=exact_sha,
                    exact_manifest=exact_manifest,
                )
                print(
                    "[reuse] candidate generation stage is already COMPLETE", flush=True
                )
                return manifest
            raise FileExistsError(
                f"Candidate stage exists but does not match: {stage_root}"
            )
        stage_dir = _atomic_stage_dir(stage_root)
        started = time.perf_counter()
        candidate_db_path = stage_dir / "candidate_index.sqlite3"
        candidates = _new_candidate_database(candidate_db_path)
        index_connection = sqlite3.connect(
            f"file:{index_root / 'bucket_index.sqlite3'}?mode=ro", uri=True
        )
        bucket_metrics: dict[str, dict[str, Any]] = {
            config.name: {
                "bucket_count": 0,
                "bucket_size_distribution": Counter(),
                "largest_bucket": 0,
                "overflow_bucket_count": 0,
                "omitted_pair_upper_bound_from_capped_buckets": 0,
                "possible_pair_occurrences": 0,
                "emitted_pair_occurrences": 0,
            }
            for config in LSH_CONFIGS
        }
        pair_batch: list[tuple[bytes, bytes, int, int, int, None]] = []
        endpoint_batch: list[tuple[bytes]] = []
        last_config = -1
        processed_buckets = 0

        def flush_pairs() -> None:
            if not pair_batch:
                return
            candidates.executemany(
                "INSERT INTO pairs VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id_a,id_b) DO UPDATE SET "
                "config_mask=(pairs.config_mask | excluded.config_mask), "
                "bucket_hits=pairs.bucket_hits + excluded.bucket_hits, "
                "max_bucket_size=MAX(pairs.max_bucket_size, excluded.max_bucket_size)",
                pair_batch,
            )
            pair_batch.clear()
            candidates.commit()

        def flush_endpoints() -> None:
            if endpoint_batch:
                candidates.executemany(
                    "INSERT OR IGNORE INTO endpoint_ids VALUES (?)", endpoint_batch
                )
                endpoint_batch.clear()

        try:
            cursor = index_connection.execute(
                "SELECT config_id, band, bucket_key, record_id FROM buckets "
                "ORDER BY config_id, band, bucket_key, record_id"
            )
            for group_key, rows_iter in itertools.groupby(
                cursor, key=lambda row: (int(row[0]), int(row[1]), bytes(row[2]))
            ):
                config_id, _band, _bucket_key = group_key
                config = LSH_CONFIGS[config_id]
                metric = bucket_metrics[config.name]
                if config_id != last_config:
                    last_config = config_id
                    print(f"[candidates] enumerating {config.name}", flush=True)
                member_ids, bucket_size = _bounded_bucket_members(
                    (bytes(row[3]) for row in rows_iter), BUCKET_MEMBER_CAP
                )
                metric["bucket_count"] += 1
                metric["largest_bucket"] = max(metric["largest_bucket"], bucket_size)
                if bucket_size <= 1:
                    distribution_key = "1"
                elif bucket_size <= 5:
                    distribution_key = "2-5"
                elif bucket_size <= 20:
                    distribution_key = "6-20"
                elif bucket_size <= 64:
                    distribution_key = "21-64"
                elif bucket_size <= BUCKET_MEMBER_CAP:
                    distribution_key = "65-256"
                else:
                    distribution_key = f">{BUCKET_MEMBER_CAP}"
                    metric["overflow_bucket_count"] += 1
                    metric["omitted_pair_upper_bound_from_capped_buckets"] += (
                        bucket_size * (bucket_size - 1) // 2
                        - BUCKET_MEMBER_CAP * (BUCKET_MEMBER_CAP - 1) // 2
                    )
                metric["bucket_size_distribution"][distribution_key] += 1
                metric["possible_pair_occurrences"] += (
                    bucket_size * (bucket_size - 1) // 2
                )
                emitted_size = min(bucket_size, BUCKET_MEMBER_CAP)
                metric["emitted_pair_occurrences"] += (
                    emitted_size * (emitted_size - 1) // 2
                )
                if bucket_size >= 2:
                    member_ids.sort()
                    for id_a, id_b in itertools.combinations(member_ids, 2):
                        pair_batch.append(
                            (
                                id_a,
                                id_b,
                                LSH_CONFIG_MASK[config.name],
                                1,
                                bucket_size,
                                None,
                            )
                        )
                        endpoint_batch.extend(((id_a,), (id_b,)))
                        if len(pair_batch) >= 20_000:
                            flush_pairs()
                        if len(endpoint_batch) >= 40_000:
                            flush_endpoints()
                processed_buckets += 1
                if processed_buckets % 2_000_000 == 0:
                    flush_pairs()
                    flush_endpoints()
                    print(
                        f"[candidates] bucket_groups={processed_buckets:,} "
                        f"config={config.name}",
                        flush=True,
                    )
            flush_pairs()
            flush_endpoints()
            endpoint_count = int(
                candidates.execute("SELECT COUNT(*) FROM endpoint_ids").fetchone()[0]
            )
            pair_count = int(
                candidates.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
            )
            bloom = _BloomFilter(endpoint_count)
            for (record_id,) in candidates.execute(
                "SELECT record_id FROM endpoint_ids"
            ):
                bloom.add(bytes(record_id))
            candidates.commit()

            # Recover endpoint metadata and signatures in one streaming pass. The
            # Bloom filter has fixed memory; exact membership remains disk-backed.
            endpoint_signatures_path = stage_dir / ".endpoint_signatures.sqlite3"
            endpoint_signatures = _sqlite_connect(endpoint_signatures_path)
            endpoint_signatures.execute(
                "CREATE TABLE signatures (record_id BLOB PRIMARY KEY, signature BLOB NOT NULL) WITHOUT ROWID"
            )
            signature_parquet = pq.ParquetFile(signature_root / "fingerprints.parquet")
            records_insert: list[tuple[Any, ...]] = []
            signature_insert: list[tuple[bytes, bytes]] = []
            recovered = 0
            signature_fields = [
                "record_id",
                "source",
                "subset",
                "normalized_words",
                "length_band",
                "data_relative_path",
                "row_ordinal",
                "row_group",
                "row_offset",
                "original_id",
                "original_url",
                "domain",
                "title",
                "domain_category",
                "signature",
            ]
            for batch in signature_parquet.iter_batches(
                batch_size=FINGERPRINT_BATCH_SIZE,
                columns=signature_fields,
            ):
                for row in batch.to_pylist():
                    record_id = row["record_id"]
                    if not bloom.might_contain(record_id):
                        continue
                    found = candidates.execute(
                        "SELECT 1 FROM endpoint_ids WHERE record_id=?", (record_id,)
                    ).fetchone()
                    if not found:
                        continue
                    records_insert.append(
                        (
                            record_id,
                            row["source"],
                            row["subset"],
                            row["normalized_words"],
                            row["length_band"],
                            row["data_relative_path"],
                            row["row_ordinal"],
                            row["row_group"],
                            row["row_offset"],
                            row["original_id"],
                            row["original_url"],
                            row["domain"],
                            row["title"],
                            row["domain_category"],
                        )
                    )
                    signature_insert.append((record_id, row["signature"]))
                    recovered += 1
                    if len(records_insert) >= 2_000:
                        candidates.executemany(
                            "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            records_insert,
                        )
                        records_insert.clear()
                        candidates.commit()
                    if len(signature_insert) >= 2_000:
                        endpoint_signatures.executemany(
                            "INSERT INTO signatures VALUES (?,?)", signature_insert
                        )
                        signature_insert.clear()
                        endpoint_signatures.commit()
                if recovered and recovered % 100_000 == 0:
                    print(
                        f"[candidates] endpoint_metadata={recovered:,}/{endpoint_count:,}",
                        flush=True,
                    )
            if records_insert:
                candidates.executemany(
                    "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    records_insert,
                )
            if signature_insert:
                endpoint_signatures.executemany(
                    "INSERT INTO signatures VALUES (?,?)", signature_insert
                )
            candidates.commit()
            endpoint_signatures.commit()
            if recovered != endpoint_count:
                raise ValueError(
                    f"Recovered {recovered} candidate endpoints, expected {endpoint_count}"
                )

            # MinHash estimates are computed for every generated candidate, but
            # exact word-5 scores are sampled in the next stage.
            candidates.execute(
                "ATTACH DATABASE ? AS endpoint_sig", (str(endpoint_signatures_path),)
            )
            score_cursor = candidates.execute(
                "SELECT p.id_a,p.id_b,sa.signature,sb.signature "
                "FROM pairs AS p "
                "JOIN endpoint_sig.signatures AS sa ON sa.record_id=p.id_a "
                "JOIN endpoint_sig.signatures AS sb ON sb.record_id=p.id_b "
                "ORDER BY p.id_a,p.id_b"
            )
            updates: list[tuple[float, bytes, bytes]] = []
            scored = 0
            for id_a, id_b, signature_a, signature_b in score_cursor:
                values_a = struct.unpack("<128Q", signature_a)
                values_b = struct.unpack("<128Q", signature_b)
                estimate = (
                    sum(left == right for left, right in zip(values_a, values_b)) / 128
                )
                updates.append((estimate, id_a, id_b))
                scored += 1
                if len(updates) >= 20_000:
                    candidates.executemany(
                        "UPDATE pairs SET estimated_similarity=? WHERE id_a=? AND id_b=?",
                        updates,
                    )
                    updates.clear()
                    candidates.commit()
                if scored and scored % 500_000 == 0:
                    print(
                        f"[candidates] MinHash-scored={scored:,}/{pair_count:,}",
                        flush=True,
                    )
            if updates:
                candidates.executemany(
                    "UPDATE pairs SET estimated_similarity=? WHERE id_a=? AND id_b=?",
                    updates,
                )
            candidates.commit()
            candidates.execute("DETACH DATABASE endpoint_sig")
            endpoint_signatures.close()
            endpoint_signatures_path.unlink(missing_ok=True)
            if scored != pair_count:
                raise ValueError(
                    f"MinHash-scored {scored} candidates, expected {pair_count}"
                )
            config_pair_counts = {
                config.name: int(
                    candidates.execute(
                        "SELECT COUNT(*) FROM pairs WHERE (config_mask & ?) != 0",
                        (LSH_CONFIG_MASK[config.name],),
                    ).fetchone()[0]
                )
                for config in LSH_CONFIGS
            }
            overflow_summary = {}
            for config_name, values in bucket_metrics.items():
                values["bucket_size_distribution"] = dict(
                    sorted(values["bucket_size_distribution"].items())
                )
                values["unique_candidate_pairs"] = config_pair_counts[config_name]
                overflow_summary[config_name] = values
            candidate_metrics = {
                "unique_union_candidate_pairs": pair_count,
                "candidate_endpoints": endpoint_count,
                "candidate_pairs_by_configuration": config_pair_counts,
                "bucket_member_cap": BUCKET_MEMBER_CAP,
                "bucket_metrics": overflow_summary,
                "elapsed_wall_seconds": round(time.perf_counter() - started, 3),
                "peak_rss_bytes": near._peak_rss_bytes(),
                "candidate_database_bytes": candidate_db_path.stat().st_size,
                "signature_manifest_sha256": compute_file_sha256(
                    signature_root / "manifest.json"
                ),
                "population_by_length_band": signature_manifest.get("metrics", {}).get(
                    "population_by_length_band", {}
                ),
                "population_by_source_subset": signature_manifest.get(
                    "metrics", {}
                ).get("population_by_source_subset", {}),
                "candidate_generation_is_capped": True,
            }
            _json_write(stage_dir / "resource_report.json", candidate_metrics)
            candidates.close()
            index_connection.close()
            manifest = _write_stage_manifest(
                stage_dir,
                stage_name="candidates",
                input_manifest_sha256=exact_sha,
                input_signature_manifest_sha256=expected_signature_manifest_sha,
                metrics=candidate_metrics,
            )
            os.replace(stage_dir, stage_root)
            _update_root_manifest(
                output,
                exact_root=exact_root,
                data_root=data_root,
                exact_manifest_sha256=exact_sha,
                exact_manifest=exact_manifest,
            )
            return manifest
        except BaseException:
            try:
                candidates.close()
            except Exception:
                pass
            try:
                index_connection.close()
            except Exception:
                pass
            try:
                endpoint_signatures.close()
            except (UnboundLocalError, Exception):
                pass
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise


def _csv_write(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _csv_safe(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _summarize_candidates(database_path: Path, summary_root: Path) -> dict[str, Any]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    total = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])
    by_config = {
        config.name: int(
            connection.execute(
                "SELECT COUNT(*) FROM pairs WHERE (config_mask & ?) != 0",
                (LSH_CONFIG_MASK[config.name],),
            ).fetchone()[0]
        )
        for config in LSH_CONFIGS
    }
    by_source_subset_pairs: Counter[tuple[str, str]] = Counter()
    by_source_subset_endpoints: Counter[tuple[str, str]] = Counter()
    by_source_pair: Counter[tuple[str, str, str, str]] = Counter()
    by_similarity: Counter[tuple[str, str, str]] = Counter()
    by_length: Counter[tuple[str, str]] = Counter()
    by_ratio: Counter[str] = Counter()
    by_relation: Counter[str] = Counter()
    by_source: Counter[tuple[str, str]] = Counter()
    by_subset_pair_lenratio: Counter[tuple[str, str, str, str]] = Counter()
    parliament_any = 0
    parliament_only = 0
    short_pairs = 0
    high_estimate = 0
    scanned_pairs = 0
    for raw in connection.execute(_candidate_join_query()):
        row = _unpack_candidate_row(tuple(raw))
        scanned_pairs += 1
        source_a, source_b = row["source_a"], row["source_b"]
        rel = row["pair_relation"]
        by_relation[rel] += 1
        parliament_partition = (
            "with_parlamento"
            if row["parlamento_diagnostic_only"]
            else "without_parlamento"
        )
        by_source[(str(source_a), parliament_partition)] += 1
        if source_a != source_b:
            by_source[(str(source_b), parliament_partition)] += 1
        by_ratio[row["length_ratio_band"]] += 1
        by_subset_pair_lenratio[
            (
                row["source_subset_a"],
                row["source_subset_b"],
                row["length_ratio_band"],
                rel,
            )
        ] += 1
        by_length[(row["source_subset_a"], row["length_band_a"])] += 1
        by_length[(row["source_subset_b"], row["length_band_b"])] += 1
        by_source_subset_endpoints[(row["source_subset_a"], rel)] += 1
        by_source_subset_endpoints[(row["source_subset_b"], rel)] += 1
        by_source_subset_pairs[(row["source_subset_a"], rel)] += 1
        if row["source_subset_a"] != row["source_subset_b"]:
            by_source_subset_pairs[(row["source_subset_b"], rel)] += 1
        left, right = sorted((row["source_subset_a"], row["source_subset_b"]))
        by_source_pair[(left, right, rel, row["length_ratio_band"])] += 1
        contains_parliament = row["parlamento_diagnostic_only"]
        by_similarity[
            (
                row["estimated_similarity_band"],
                "with_parlamento" if contains_parliament else "without_parlamento",
                "short" if row["short_pair"] else "normal_or_long",
            )
        ] += 1
        parliament_any += int(contains_parliament)
        parliament_only += int(
            source_a == "parlamento_pt" and source_b == "parlamento_pt"
        )
        short_pairs += int(row["short_pair"])
        high_estimate += int(row["estimated_similarity"] >= 0.80)
        if scanned_pairs % 500_000 == 0:
            print(
                f"[summarize] candidates_scanned={scanned_pairs:,}/{total:,}",
                flush=True,
            )
    connection.close()

    summary_root.mkdir(parents=True, exist_ok=True)
    rows_source_subset = [
        {
            "source_subset": key[0],
            "relationship": key[1],
            "candidate_pairs_touching": by_source_subset_pairs[key],
            "candidate_endpoint_incidences": by_source_subset_endpoints[key],
        }
        for key in sorted(by_source_subset_pairs)
    ]
    _csv_write(
        summary_root / "candidate_counts_by_source_subset.csv",
        (
            "source_subset",
            "relationship",
            "candidate_pairs_touching",
            "candidate_endpoint_incidences",
        ),
        rows_source_subset,
    )
    rows_pair = [
        {
            "source_subset_a": key[0],
            "source_subset_b": key[1],
            "relationship": key[2],
            "length_ratio_band": key[3],
            "candidate_pairs": count,
        }
        for key, count in sorted(by_source_pair.items())
    ]
    _csv_write(
        summary_root / "candidate_counts_by_pair.csv",
        (
            "source_subset_a",
            "source_subset_b",
            "relationship",
            "length_ratio_band",
            "candidate_pairs",
        ),
        rows_pair,
    )
    rows_similarity = [
        {
            "estimated_similarity_band": key[0],
            "parlamento_partition": key[1],
            "length_partition": key[2],
            "candidate_pairs": count,
            "interpretation": "128-value MinHash estimate; not exact Jaccard",
        }
        for key, count in sorted(by_similarity.items())
    ]
    _csv_write(
        summary_root / "candidate_counts_by_similarity_band.csv",
        (
            "estimated_similarity_band",
            "parlamento_partition",
            "length_partition",
            "candidate_pairs",
            "interpretation",
        ),
        rows_similarity,
    )
    rows_length = [
        {
            "source_subset": key[0],
            "document_length_band": key[1],
            "candidate_endpoint_incidences": count,
        }
        for key, count in sorted(by_length.items())
    ]
    _csv_write(
        summary_root / "candidate_counts_by_length_band.csv",
        ("source_subset", "document_length_band", "candidate_endpoint_incidences"),
        rows_length,
    )
    rows_ratio = [
        {"length_ratio_band": key, "candidate_pairs": value}
        for key, value in sorted(by_ratio.items())
    ]
    _csv_write(
        summary_root / "candidate_counts_by_length_ratio_band.csv",
        ("length_ratio_band", "candidate_pairs"),
        rows_ratio,
    )
    rows_relation = [
        {"relationship": key, "candidate_pairs": value}
        for key, value in sorted(by_relation.items())
    ]
    _csv_write(
        summary_root / "candidate_counts_by_relationship.csv",
        ("relationship", "candidate_pairs"),
        rows_relation,
    )
    rows_source = [
        {
            "source": key[0],
            "parlamento_partition": key[1],
            "candidate_pairs_touching_source": value,
        }
        for key, value in sorted(by_source.items())
    ]
    _csv_write(
        summary_root / "candidate_counts_by_source.csv",
        ("source", "parlamento_partition", "candidate_pairs_touching_source"),
        rows_source,
    )
    rows_pair_ratio = [
        {
            "source_subset_a": key[0],
            "source_subset_b": key[1],
            "length_ratio_band": key[2],
            "relationship": key[3],
            "candidate_pairs": count,
        }
        for key, count in sorted(by_subset_pair_lenratio.items())
    ]
    _csv_write(
        summary_root / "candidate_counts_by_pair_and_length_ratio.csv",
        (
            "source_subset_a",
            "source_subset_b",
            "length_ratio_band",
            "relationship",
            "candidate_pairs",
        ),
        rows_pair_ratio,
    )
    candidate_manifest = _json_read(database_path.parent / "manifest.json")
    summary = {
        "schema_version": 1,
        "census_version": CENSUS_VERSION,
        "measurement_scope": "complete candidate population generated by the two capped LSH configurations",
        "input_exact_manifest_sha256": candidate_manifest[
            "input_exact_manifest_sha256"
        ],
        "input_records": candidate_manifest["metrics"].get("fingerprinted_records"),
        "unique_union_candidate_pairs": total,
        "candidate_pairs_by_configuration": by_config,
        "candidate_pairs_with_parlamento_pt": parliament_any,
        "parlamento_pt_within_source_candidate_pairs": parliament_only,
        "candidate_pairs_without_parlamento_pt": total - parliament_any,
        "short_candidate_pairs_shorter_record_lt20_words": short_pairs,
        "candidate_pairs_estimate_ge0.80": high_estimate,
        "candidate_pairs_by_relationship": dict(sorted(by_relation.items())),
        "length_band_definitions": {
            "lt20": "fewer than 20 words",
            "20_99": "20 through 99 words",
            "100_999": "100 through 999 words",
            "1000_99999": "1,000 through 99,999 words",
            "ge100000": "at least 100,000 words",
        },
        "corpus_population_by_length_band": candidate_manifest["metrics"].get(
            "population_by_length_band", {}
        ),
        "corpus_population_by_source_subset": candidate_manifest["metrics"].get(
            "population_by_source_subset", {}
        ),
        "candidate_pairs_by_length_ratio_band": dict(sorted(by_ratio.items())),
        "bucket_metrics": candidate_manifest["metrics"].get("bucket_metrics", {}),
        "candidate_generation_capped": True,
        "candidate_overflow_note": (
            "Buckets above the cap use the first 256 cryptographic occurrence IDs; "
            "omitted-pair upper bounds are recorded by configuration. Counts describe "
            "the bounded LSH candidate generator, not all possible corpus pairs."
        ),
        "minhash_note": "Estimated similarities use 128 permutations and must not be interpreted as exact Jaccard.",
        "parlamento_policy": "diagnostic-only; every row is preserve-all; removal_eligible is false for all candidates",
        "production_decision": "NOT MADE BY CENSUS IMPLEMENTATION",
        "decision_support_outcomes": [
            "A: consider production only if the reviewed high-confidence population and a safe rule support it",
            "B: limit any later policy to source/provenance/length classes supported by review evidence",
            "C: skip global near-dedup if demonstrated duplication is sparse or ambiguous relative to false-positive risk",
        ],
    }
    _json_write(summary_root / "census_summary.json", summary)
    summary["candidate_pairs_by_source_subset"] = {
        f"{key[0]}|{key[1]}": count
        for key, count in sorted(by_source_subset_pairs.items())
    }
    summary["candidate_pairs_by_source_subset_pair_length_ratio"] = {
        "|".join(key): count for key, count in sorted(by_subset_pair_lenratio.items())
    }
    return summary


def run_summarize(
    *,
    input_root: Path | str = DEFAULT_INPUT_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    expected_manifest_sha256: str | None = EXPECTED_EXACT_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Write deterministic complete-candidate population summaries."""
    exact_root, data_root, exact_manifest, exact_sha = _data_root_and_manifest(
        input_root, expected_manifest_sha256
    )
    output = Path(output_root).resolve()
    _assert_output_separate(exact_root, output)
    candidate_stage = output / "lsh" / "candidates"
    errors = _verify_stage(candidate_stage, check_hashes=False)
    if errors:
        raise ValueError("Candidate stage is not ready: " + "; ".join(errors))
    stage_root = output / "candidate_summary"
    with _stage_lock(output, "summarize"):
        if stage_root.exists():
            errors = _verify_stage(stage_root, check_hashes=False)
            if not errors:
                _update_root_manifest(
                    output,
                    exact_root=exact_root,
                    data_root=data_root,
                    exact_manifest_sha256=exact_sha,
                    exact_manifest=exact_manifest,
                )
                print("[reuse] candidate summaries are already COMPLETE", flush=True)
                return _json_read(stage_root / "census_summary.json")
            raise FileExistsError(
                f"Candidate summary exists but is invalid: {stage_root}"
            )
        stage_dir = _atomic_stage_dir(stage_root)
        candidates_db = candidate_stage / "candidate_index.sqlite3"
        started = time.perf_counter()
        try:
            summary = _summarize_candidates(candidates_db, stage_dir)
            summary["summary_elapsed_seconds"] = round(time.perf_counter() - started, 3)
            _json_write(stage_dir / "census_summary.json", summary)
            _json_write(
                stage_dir / "resource_report.json",
                {
                    "summary_elapsed_seconds": summary["summary_elapsed_seconds"],
                    "candidate_pairs": summary["unique_union_candidate_pairs"],
                    "peak_rss_bytes": near._peak_rss_bytes(),
                },
            )
            _write_stage_manifest(
                stage_dir,
                stage_name="candidate_summary",
                input_manifest_sha256=exact_sha,
                input_signature_manifest_sha256=compute_file_sha256(
                    (output / "signatures" / "manifest.json")
                ),
                metrics={
                    "unique_union_candidate_pairs": summary[
                        "unique_union_candidate_pairs"
                    ]
                },
            )
            os.replace(stage_dir, stage_root)
            _update_root_manifest(
                output,
                exact_root=exact_root,
                data_root=data_root,
                exact_manifest_sha256=exact_sha,
                exact_manifest=exact_manifest,
            )
            return summary
        except BaseException:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise


def _iter_u64_file(path: Path) -> Iterator[int]:
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 16_384):
            usable = len(chunk) - (len(chunk) % 8)
            values = np.frombuffer(chunk[:usable], dtype="<u8")
            yield from (int(value) for value in values)


def _write_sorted_unique_runs(
    text: str, scratch: Path, *, prefix: str
) -> tuple[list[Path], int]:
    runs: list[Path] = []
    values: list[int] = []
    total = 0

    def flush() -> None:
        if not values:
            return
        unique = sorted(set(values))
        run_path = scratch / f"{prefix}-run-{len(runs):06d}.u64"
        np.asarray(unique, dtype="<u8").tofile(run_path)
        runs.append(run_path)
        values.clear()

    for value in near.iter_word_shingle_hashes(
        text, SIGNATURE_CONFIG.ngram_size, seed=SIGNATURE_CONFIG.seed
    ):
        values.append(value)
        total += 1
        if len(values) >= EXTERNAL_SORT_SHINGLE_BATCH:
            flush()
    flush()
    if not runs:
        empty = scratch / f"{prefix}-empty.u64"
        empty.write_bytes(b"")
        runs.append(empty)
    return runs, total


def _merge_unique_runs(runs: Sequence[Path], target: Path) -> int:
    count = 0
    previous: int | None = None
    with ExitStack() as resources:
        iterators = [
            resources.enter_context(closing(_iter_u64_file(path))) for path in runs
        ]
        merged = resources.enter_context(closing(heapq.merge(*iterators)))
        output = resources.enter_context(target.open("wb"))
        for value in merged:
            if value == previous:
                continue
            output.write(struct.pack("<Q", value))
            previous = value
            count += 1
    return count


def _merge_runs_with_bounded_fan_in(
    runs: Sequence[Path],
    target: Path,
    scratch: Path,
    *,
    prefix: str,
    fan_in: int = 64,
) -> int:
    """Merge arbitrarily many sorted runs with a bounded open-file count."""
    current = list(runs)
    generation = 0
    while len(current) > fan_in:
        next_runs: list[Path] = []
        for group_index, start in enumerate(range(0, len(current), fan_in)):
            group = current[start : start + fan_in]
            merged_path = (
                scratch / f"{prefix}-merge-{generation:03d}-{group_index:06d}.u64"
            )
            _merge_unique_runs(group, merged_path)
            next_runs.append(merged_path)
            for old_path in group:
                old_path.unlink(missing_ok=True)
        current = next_runs
        generation += 1
    return _merge_unique_runs(current, target)


def _external_exact_pair_metrics(
    text_a: str,
    text_b: str,
    scratch_parent: Path,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="pair-exact-", dir=scratch_parent) as tmp:
        scratch = Path(tmp)
        runs_a, _total_a = _write_sorted_unique_runs(text_a, scratch, prefix="a")
        runs_b, _total_b = _write_sorted_unique_runs(text_b, scratch, prefix="b")
        unique_a_path = scratch / "a-unique.u64"
        unique_b_path = scratch / "b-unique.u64"
        unique_a = _merge_runs_with_bounded_fan_in(
            runs_a, unique_a_path, scratch, prefix="a"
        )
        unique_b = _merge_runs_with_bounded_fan_in(
            runs_b, unique_b_path, scratch, prefix="b"
        )
        with ExitStack() as resources:
            iter_a = resources.enter_context(closing(_iter_u64_file(unique_a_path)))
            iter_b = resources.enter_context(closing(_iter_u64_file(unique_b_path)))
            value_a = next(iter_a, None)
            value_b = next(iter_b, None)
            shared = 0
            while value_a is not None and value_b is not None:
                if value_a == value_b:
                    shared += 1
                    value_a = next(iter_a, None)
                    value_b = next(iter_b, None)
                elif value_a < value_b:
                    value_a = next(iter_a, None)
                else:
                    value_b = next(iter_b, None)
        union = unique_a + unique_b - shared
        min_unique = min(unique_a, unique_b)
        return {
            "unique_shingles_a": unique_a,
            "unique_shingles_b": unique_b,
            "shared_shingles": shared,
            "union_shingles": union,
            "exact_jaccard": shared / union if union else 0.0,
            "containment_a_in_b": shared / unique_a if unique_a else 0.0,
            "containment_b_in_a": shared / unique_b if unique_b else 0.0,
            "containment_smaller_in_larger": shared / min_unique if min_unique else 0.0,
            "exact_scoring_method": "external_sort",
        }


def _exact_pair_metrics(
    text_a: str,
    text_b: str,
    scratch_parent: Path,
    *,
    expected_words_a: int,
    expected_words_b: int,
) -> dict[str, Any]:
    if max(expected_words_a, expected_words_b) <= EXACT_MEMORY_SHINGLE_LIMIT:
        metrics = near.exact_jaccard_from_text(
            text_a,
            text_b,
            ngram_size=5,
            seed=SIGNATURE_CONFIG.seed,
            max_shingles=EXACT_MEMORY_SHINGLE_LIMIT + 1,
        )
        if metrics.get("exact_available"):
            unique_a = int(metrics["unique_shingles_a"])
            unique_b = int(metrics["unique_shingles_b"])
            shared = int(metrics["shared_shingles"])
            union = int(metrics["union_shingles"])
            return {
                "unique_shingles_a": unique_a,
                "unique_shingles_b": unique_b,
                "shared_shingles": shared,
                "union_shingles": union,
                "exact_jaccard": float(metrics["exact_jaccard"]),
                "containment_a_in_b": shared / unique_a if unique_a else 0.0,
                "containment_b_in_a": shared / unique_b if unique_b else 0.0,
                "containment_smaller_in_larger": float(metrics["containment"]),
                "exact_scoring_method": "in_memory_set",
            }
    return _external_exact_pair_metrics(text_a, text_b, scratch_parent)


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _prepare_exact_scratch_root(
    scratch_root: Path | str,
    *,
    protected_paths: Sequence[Path],
) -> Path:
    root = Path(scratch_root).expanduser().resolve()
    for protected in protected_paths:
        protected_resolved = protected.resolve()
        if _paths_overlap(root, protected_resolved):
            raise ValueError(
                f"Exact-sample scratch root overlaps protected path: {protected_resolved}"
            )
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or not os.access(root, os.W_OK | os.X_OK):
        raise PermissionError(f"Exact-sample scratch root is not writable: {root}")
    # Probe actual creation/removal so a bad scratch mount fails before the
    # candidate sample stage is started. There is deliberately no TMPDIR/NFS
    # fallback if this check fails.
    with tempfile.TemporaryDirectory(prefix=".near-census-write-check-", dir=root):
        pass
    return root


def _score_exact_sample_pairs(
    *,
    data_root: Path,
    candidate_db_path: Path,
    selected_pairs: Sequence[tuple[str, str]],
    stage_dir: Path,
    scratch_root: Path,
) -> tuple[list[dict[str, Any]], int]:
    candidate_db_path_uri = f"file:{candidate_db_path}?mode=ro"
    with ExitStack() as resources:
        candidate_db = resources.enter_context(
            closing(sqlite3.connect(candidate_db_path_uri, uri=True))
        )
        candidate_db.row_factory = sqlite3.Row
        selected_ids = {
            bytes.fromhex(value) for pair in selected_pairs for value in pair
        }
        print(
            f"[exact-sample] selected_pairs={len(selected_pairs):,} "
            f"selected_endpoints={len(selected_ids):,}",
            flush=True,
        )
        text_store_path = stage_dir / ".selected_texts.sqlite3"
        text_store, text_count = _read_selected_texts(
            data_root,
            candidate_db,
            selected_ids,
            text_store_path,
        )
        resources.callback(text_store_path.unlink, missing_ok=True)
        text_store = resources.enter_context(closing(text_store))

        scored_rows: list[dict[str, Any]] = []
        query = _candidate_join_query() + " WHERE p.id_a=? AND p.id_b=?"
        with tempfile.TemporaryDirectory(
            prefix="exact-sample-run-", dir=scratch_root
        ) as run_scratch:
            run_scratch_path = Path(run_scratch)
            for index, pair in enumerate(selected_pairs, start=1):
                raw_a, raw_b = bytes.fromhex(pair[0]), bytes.fromhex(pair[1])
                base = candidate_db.execute(query, (raw_a, raw_b)).fetchone()
                if base is None:
                    raise ValueError(
                        f"Selected candidate pair disappeared from index: {pair}"
                    )
                row = _unpack_candidate_row(tuple(base))
                text_a = zlib.decompress(
                    text_store.execute(
                        "SELECT text_zlib FROM selected_texts WHERE record_id=?",
                        (raw_a,),
                    ).fetchone()[0]
                ).decode("utf-8")
                text_b = zlib.decompress(
                    text_store.execute(
                        "SELECT text_zlib FROM selected_texts WHERE record_id=?",
                        (raw_b,),
                    ).fetchone()[0]
                ).decode("utf-8")
                exact = _exact_pair_metrics(
                    text_a,
                    text_b,
                    run_scratch_path,
                    expected_words_a=int(row["words_a"]),
                    expected_words_b=int(row["words_b"]),
                )
                ratio = _ratio(int(row["words_a"]), int(row["words_b"]))
                flags = containment_flags(
                    jaccard=float(exact["exact_jaccard"]),
                    containment_a_in_b=float(exact["containment_a_in_b"]),
                    containment_b_in_a=float(exact["containment_b_in_a"]),
                    length_ratio=ratio,
                    shared_shingles=int(exact["shared_shingles"]),
                )
                title_sim = _title_similarity(row.get("title_a"), row.get("title_b"))
                high_jaccard = float(exact["exact_jaccard"]) >= 0.80
                boilerplate = bool(
                    high_jaccard
                    and title_sim is not None
                    and title_sim < 0.20
                    and text_a[:300].casefold() != text_b[:300].casefold()
                )
                excerpts_a = _text_excerpt_fields(text_a)
                excerpts_b = _text_excerpt_fields(text_b)
                exact_row: dict[str, Any] = {
                    "record_id_a": row["id_a"],
                    "record_id_b": row["id_b"],
                    "candidate_configs": row["candidate_configs"],
                    "bucket_hits": row["bucket_hits"],
                    "max_bucket_size": row["max_bucket_size"],
                    "bucket_size_band": row["bucket_size_band"],
                    "estimated_similarity": row["estimated_similarity"],
                    "estimated_similarity_band": row["estimated_similarity_band"],
                    "exact_jaccard": exact["exact_jaccard"],
                    "exact_similarity_band": exact_tail_band(exact["exact_jaccard"])
                    or similarity_band(exact["exact_jaccard"]),
                    "containment_a_in_b": exact["containment_a_in_b"],
                    "containment_b_in_a": exact["containment_b_in_a"],
                    "containment_smaller_in_larger": exact[
                        "containment_smaller_in_larger"
                    ],
                    "unique_shingles_a": exact["unique_shingles_a"],
                    "unique_shingles_b": exact["unique_shingles_b"],
                    "shared_shingles": exact["shared_shingles"],
                    "union_shingles": exact["union_shingles"],
                    "exact_scoring_method": exact["exact_scoring_method"],
                    "source_a": row["source_a"],
                    "subset_a": row["subset_a"],
                    "source_b": row["source_b"],
                    "subset_b": row["subset_b"],
                    "source_subset_a": row["source_subset_a"],
                    "source_subset_b": row["source_subset_b"],
                    "source_pair": row["source_pair"],
                    "pair_relation": row["pair_relation"],
                    "words_a": row["words_a"],
                    "words_b": row["words_b"],
                    "length_band_a": row["length_band_a"],
                    "length_band_b": row["length_band_b"],
                    "length_ratio": ratio,
                    "length_ratio_band": length_ratio_band(ratio),
                    "domain_a": row["domain_a"],
                    "domain_b": row["domain_b"],
                    "url_a": row["url_a"],
                    "url_b": row["url_b"],
                    "title_a": row["title_a"],
                    "title_b": row["title_b"],
                    "title_token_jaccard": title_sim,
                    "domain_category_a": row["domain_category_a"],
                    "domain_category_b": row["domain_category_b"],
                    "data_relative_path_a": row["path_a"],
                    "data_relative_path_b": row["path_b"],
                    "row_ordinal_a": row["ordinal_a"],
                    "row_ordinal_b": row["ordinal_b"],
                    "row_group_a": row["row_group_a"],
                    "row_group_b": row["row_group_b"],
                    "row_offset_a": row["row_offset_a"],
                    "row_offset_b": row["row_offset_b"],
                    "original_id_a": row["original_id_a"],
                    "original_id_b": row["original_id_b"],
                    "containment_flags": flags,
                    "boilerplate_suspect": boilerplate,
                    "boilerplate_suspect_basis": (
                        "high exact Jaccard with low title-token overlap and differing opening excerpt"
                        if boilerplate
                        else ""
                    ),
                    "parlamento_diagnostic_only": row["parlamento_diagnostic_only"],
                    "removal_eligible": False,
                    "excerpt_start_a": excerpts_a["excerpt_start"],
                    "excerpt_middle_a": excerpts_a["excerpt_middle"],
                    "excerpt_end_a": excerpts_a["excerpt_end"],
                    "excerpt_start_b": excerpts_b["excerpt_start"],
                    "excerpt_middle_b": excerpts_b["excerpt_middle"],
                    "excerpt_end_b": excerpts_b["excerpt_end"],
                }
                exact_row["review_categories"] = _review_categories(exact_row)
                scored_rows.append(exact_row)
                if index % 500 == 0:
                    print(
                        f"[exact-sample] exact-scored={index:,}/"
                        f"{len(selected_pairs):,}",
                        flush=True,
                    )
        return scored_rows, text_count


def _bounded_excerpt(text: str, start: int, width: int = 360) -> str:
    excerpt = text[max(0, start) : max(0, start) + width]
    return " ".join(excerpt.split())[:width]


def _text_excerpt_fields(text: str) -> dict[str, str]:
    middle = max(0, (len(text) - 360) // 2)
    end = max(0, len(text) - 360)
    return {
        "excerpt_start": _bounded_excerpt(text, 0),
        "excerpt_middle": _bounded_excerpt(text, middle),
        "excerpt_end": _bounded_excerpt(text, end),
    }


def _title_similarity(title_a: str | None, title_b: str | None) -> float | None:
    if not title_a or not title_b:
        return None
    tokens_a = set(TOKEN_RE.findall(title_a.casefold()))
    tokens_b = set(TOKEN_RE.findall(title_b.casefold()))
    if not tokens_a and not tokens_b:
        return 1.0
    union = tokens_a | tokens_b
    return len(tokens_a & tokens_b) / len(union) if union else 0.0


def _select_exact_sample(
    database_path: Path,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    reservoirs: dict[tuple[str, str], _DeterministicReservoir] = {}
    stratum_counts: Counter[tuple[str, str]] = Counter()
    candidate_pairs_seen = 0
    query = _candidate_join_query() + " ORDER BY p.id_a,p.id_b"
    for raw in connection.execute(query):
        row = _unpack_candidate_row(tuple(raw))
        pair = (row["id_a"], row["id_b"])
        candidate_pairs_seen += 1
        key = f"{pair[0]}:{pair[1]}"
        for dimension, value, capacity in _candidate_sample_strata(row):
            stratum = (dimension, value)
            stratum_counts[stratum] += 1
            reservoir = reservoirs.setdefault(
                stratum,
                _DeterministicReservoir(
                    capacity,
                    EXACT_SAMPLE_SEED
                    ^ near.stable_hash64(f"{dimension}\0{value}", EXACT_SAMPLE_SEED),
                ),
            )
            reservoir.add(key, pair)
        if candidate_pairs_seen and candidate_pairs_seen % 500_000 == 0:
            print(
                f"[exact-sample] scanned={candidate_pairs_seen:,} "
                f"strata={len(reservoirs):,}",
                flush=True,
            )
    selected: set[tuple[str, str]] = set()
    for reservoir in reservoirs.values():
        selected.update(reservoir.values())
    rare_strata = {
        key: count
        for key, count in stratum_counts.items()
        if key[0] in {"source_subset_pair", "top_level_source_pair"}
        and count
        <= (RARE_SOURCE_PAIR_ALL_LIMIT if key[0] == "source_subset_pair" else 64)
    }
    selected_set = set(selected)
    sampled_stratum_counts: Counter[tuple[str, str]] = Counter()
    for raw in connection.execute(query):
        row = _unpack_candidate_row(tuple(raw))
        pair = (row["id_a"], row["id_b"])
        if pair in selected_set:
            for dimension, value, _capacity in _candidate_sample_strata(row):
                sampled_stratum_counts[(dimension, value)] += 1
    stratum_rows = []
    for key, population_count in sorted(stratum_counts.items()):
        sampled_count = sampled_stratum_counts[key]
        rare_limit = (
            RARE_SOURCE_PAIR_ALL_LIMIT if key[0] == "source_subset_pair" else 64
        )
        included_all_as_rare_stratum = (
            key in rare_strata and sampled_count == population_count
        )
        stratum_rows.append(
            {
                "dimension": key[0],
                "stratum": key[1],
                "candidate_population_count": population_count,
                "exact_scored_sample_count": sampled_count,
                "descriptive_sample_fraction": (
                    sampled_count / population_count if population_count else 0.0
                ),
                "all_candidates_included_as_rare_pair_stratum": included_all_as_rare_stratum,
                "rare_pair_inclusion_limit": rare_limit if key in rare_strata else None,
            }
        )
    # Marginal strata overlap. Their sample fractions describe realized coverage
    # and are not independent inclusion probabilities or prevalence weights.
    connection.close()
    by_dimension = Counter(dimension for dimension, _value in stratum_counts)
    return sorted(selected), {
        "candidate_pairs_seen": candidate_pairs_seen,
        "sampled_candidate_pairs": len(selected),
        "strata_by_dimension": dict(sorted(by_dimension.items())),
        "rare_source_pair_strata_included_all": len(rare_strata),
        "rare_source_pair_stratum_candidate_inclusions": sum(rare_strata.values()),
        "stratum_rows": stratum_rows,
    }


def _read_selected_texts(
    data_root: Path,
    candidate_db: sqlite3.Connection,
    selected_ids: set[bytes],
    text_store_path: Path,
) -> tuple[sqlite3.Connection, int]:
    """Fetch only selected endpoint texts by recorded row-group locators."""
    text_store = _sqlite_connect(text_store_path)
    try:
        recovered = _populate_selected_text_store(
            data_root, candidate_db, selected_ids, text_store
        )
    except BaseException:
        text_store.close()
        raise
    return text_store, recovered


def _populate_selected_text_store(
    data_root: Path,
    candidate_db: sqlite3.Connection,
    selected_ids: set[bytes],
    text_store: sqlite3.Connection,
) -> int:
    text_store.execute(
        "CREATE TABLE selected_texts (record_id BLOB PRIMARY KEY, text_zlib BLOB) WITHOUT ROWID"
    )
    locators: dict[tuple[str, int, int], list[bytes]] = defaultdict(list)
    selected_values = sorted(selected_ids)
    for start in range(0, len(selected_values), 800):
        chunk = selected_values[start : start + 800]
        placeholders = ",".join("?" for _ in chunk)
        query = (
            "SELECT record_id,data_relative_path,row_group,row_offset FROM records "
            f"WHERE record_id IN ({placeholders})"
        )
        for record_id, relative, row_group, row_offset in candidate_db.execute(
            query, chunk
        ):
            raw_id = bytes(record_id)
            locator = (str(relative), int(row_group), int(row_offset))
            locators[locator].append(raw_id)
    wanted_by_group: dict[tuple[str, int], list[tuple[int, bytes]]] = defaultdict(list)
    for (relative, group, offset), ids in locators.items():
        for record_id in ids:
            wanted_by_group[(relative, group)].append((offset, record_id))
    recovered = 0
    text_inserts: list[tuple[bytes, bytes]] = []
    for group_index, ((relative, row_group), targets) in enumerate(
        sorted(wanted_by_group.items()), start=1
    ):
        path = data_root / relative
        parquet = pq.ParquetFile(path)
        wanted = {offset: record_id for offset, record_id in targets}
        group_offset = 0
        for batch in parquet.iter_batches(
            batch_size=8,
            columns=["text"],
            row_groups=[row_group],
        ):
            texts = batch.column(0).to_pylist()
            for local, text in enumerate(texts):
                ordinal = group_offset + local
                record_id = wanted.get(ordinal)
                if record_id is None:
                    continue
                text_inserts.append(
                    (record_id, zlib.compress((text or "").encode("utf-8"), 3))
                )
                recovered += 1
                if len(text_inserts) >= 128:
                    text_store.executemany(
                        "INSERT INTO selected_texts VALUES (?,?)", text_inserts
                    )
                    text_inserts.clear()
                    text_store.commit()
            group_offset += batch.num_rows
        if group_index % 100 == 0:
            print(
                f"[exact-sample] text_row_groups={group_index:,}/"
                f"{len(wanted_by_group):,} endpoints={recovered:,}",
                flush=True,
            )
    if text_inserts:
        text_store.executemany("INSERT INTO selected_texts VALUES (?,?)", text_inserts)
        text_store.commit()
    expected = len(selected_ids)
    if recovered != expected:
        raise ValueError(f"Recovered {recovered} selected texts, expected {expected}")
    return recovered


def _write_parquet_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if rows:
        table = pa.Table.from_pylist([dict(row) for row in rows])
    else:
        table = pa.table({"record_id_a": pa.array([], type=pa.string())})
    pq.write_table(table, path, compression="zstd", compression_level=6)


def _review_categories(row: Mapping[str, Any]) -> list[str]:
    categories: list[str] = []
    if row.get("boundary_distance") is not None:
        categories.append("similarity_boundary")
    if float(row.get("exact_jaccard", 0.0)) >= 0.80:
        categories.append("high_exact_jaccard")
    if row.get("pair_relation") == "cross_source":
        categories.append("cross_source")
    if row.get("domain_a") and row.get("domain_a") == row.get("domain_b"):
        categories.append("same_domain")
    if row.get("boilerplate_suspect"):
        categories.append("boilerplate_suspect")
    if row.get("containment_flags"):
        categories.append("containment")
    curated = {"carolina", "wikipedia_pt", "gutenberg_pt"}
    if (row.get("source_a") in curated and row.get("source_b") == "gigaverbo_v2") or (
        row.get("source_b") in curated and row.get("source_a") == "gigaverbo_v2"
    ):
        categories.append("curated_web")
    if row.get("source_a") == "gigaverbo_v2" and row.get("source_b") == "gigaverbo_v2":
        categories.append("web_web")
    if {row.get("source_a"), row.get("source_b")} == {"carolina", "wikipedia_pt"}:
        categories.append("carolina_wikipedia")
    if "gutenberg_pt" in {
        row.get("source_a"),
        row.get("source_b"),
    } and "gigaverbo_v2" in {
        row.get("source_a"),
        row.get("source_b"),
    }:
        categories.append("gutenberg_web")
    if min(int(row.get("words_a", 0)), int(row.get("words_b", 0))) < 20:
        categories.append("short_text")
    if max(int(row.get("words_a", 0)), int(row.get("words_b", 0))) >= 100_000:
        categories.append("giant_document")
    if row.get("parlamento_diagnostic_only"):
        categories.append("parlamento_diagnostic_only")
    return categories


def _select_review_panel(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}

    def keep(row: dict[str, Any], category: str) -> None:
        key = (str(row["record_id_a"]), str(row["record_id_b"]))
        current = selected.setdefault(key, dict(row))
        categories = set(current.get("review_categories", []))
        categories.add(category)
        current["review_categories"] = sorted(categories)

    boundaries = (0.80, 0.85, 0.90, 0.92, 0.95)
    for threshold in boundaries:
        near_rows = sorted(
            rows,
            key=lambda row: (
                abs(float(row["exact_jaccard"]) - threshold),
                row["record_id_a"],
                row["record_id_b"],
            ),
        )[:8]
        for row in near_rows:
            row["boundary_distance"] = abs(float(row["exact_jaccard"]) - threshold)
            row["boundary_threshold"] = threshold
            keep(row, f"boundary_{threshold:.2f}")
    for row in sorted(
        rows,
        key=lambda item: (
            -float(item["exact_jaccard"]),
            item["record_id_a"],
            item["record_id_b"],
        ),
    )[:40]:
        keep(row, "highest_exact_jaccard")
    category_caps = {
        "cross_source": 40,
        "same_domain": 30,
        "boilerplate_suspect": 40,
        "containment": 35,
        "curated_web": 25,
        "web_web": 25,
        "carolina_wikipedia": 20,
        "gutenberg_web": 25,
        "short_text": 20,
        "giant_document": 20,
        "parlamento_diagnostic_only": 20,
    }
    for category, cap in category_caps.items():
        eligible = [row for row in rows if category in _review_categories(row)]
        eligible.sort(
            key=lambda row: (
                near.stable_hash64(
                    f"{category}\0{row['record_id_a']}\0{row['record_id_b']}",
                    EXACT_SAMPLE_SEED,
                ),
                row["record_id_a"],
                row["record_id_b"],
            )
        )
        for row in eligible[:cap]:
            keep(row, category)
    values = list(selected.values())
    if len(values) > REVIEW_PANEL_LIMIT:

        def priority(row: Mapping[str, Any]) -> tuple[int, int]:
            categories = set(row.get("review_categories", []))
            if any(category.startswith("boundary_") for category in categories):
                rank = 0
            elif "highest_exact_jaccard" in categories:
                rank = 1
            elif "containment" in categories:
                rank = 2
            elif "boilerplate_suspect" in categories:
                rank = 3
            else:
                rank = 10
            return rank, near.stable_hash64(
                f"review\0{row['record_id_a']}\0{row['record_id_b']}",
                EXACT_SAMPLE_SEED,
            )

        values.sort(key=priority)
        values = values[:REVIEW_PANEL_LIMIT]
    for row in values:
        row["review_categories"] = sorted(set(row.get("review_categories", [])))
        row["bounded_excerpt_characters_per_excerpt"] = 360
    return sorted(values, key=lambda row: (row["record_id_a"], row["record_id_b"]))


def _exact_summaries(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    tail_stats: Counter[tuple[str, str, str]] = Counter()
    tail_estimates: defaultdict[tuple[str, str, str], list[float]] = defaultdict(list)
    contain_stats: Counter[tuple[str, str]] = Counter()
    family_stats: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        tail = exact_tail_band(float(row["exact_jaccard"]))
        if tail:
            partition = (
                "with_parlamento"
                if row["parlamento_diagnostic_only"]
                else "without_parlamento"
            )
            key = (tail, str(row["source_pair"]), partition)
            tail_stats[key] += 1
            tail_estimates[key].append(float(row["estimated_similarity"]))
            for domain in {
                str(row.get("domain_a") or ""),
                str(row.get("domain_b") or ""),
            }:
                if domain:
                    family_stats[(tail, "domain", domain)] += 1
            for source in {
                str(row.get("source_a") or ""),
                str(row.get("source_b") or ""),
            }:
                if source:
                    family_stats[(tail, "source", source)] += 1
        flags = row.get("containment_flags") or []
        for flag in flags:
            contain_stats[(flag, str(row["source_pair"]))] += 1
    score_rows = []
    for key, count in sorted(tail_stats.items()):
        estimates = tail_estimates[key]
        score_rows.append(
            {
                "exact_similarity_band": key[0],
                "source_subset_pair": key[1],
                "parlamento_partition": key[2],
                "sampled_candidate_pairs": count,
                "mean_minhash_estimate": sum(estimates) / len(estimates),
                "interpretation": "stratified candidate sample; not a corpus prevalence estimate",
            }
        )
    containment_rows = [
        {
            "diagnostic_flag": key[0],
            "source_subset_pair": key[1],
            "sampled_candidate_pairs": count,
            "automatic_duplicate_classification": False,
        }
        for key, count in sorted(contain_stats.items())
    ]
    family_rows = [
        {
            "exact_similarity_band": key[0],
            "family_dimension": key[1],
            "source_or_domain": key[2],
            "sampled_pair_endpoint_occurrences": count,
            "interpretation": "review-sample concentration only",
        }
        for key, count in sorted(family_stats.items())
    ]
    return score_rows, containment_rows, family_rows


def run_exact_sample(
    *,
    input_root: Path | str = DEFAULT_INPUT_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    scratch_root: Path | str = DEFAULT_SCRATCH_ROOT,
    expected_manifest_sha256: str | None = EXPECTED_EXACT_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Select deterministic strata and exact-score only the candidate sample."""
    exact_root, data_root, exact_manifest, exact_sha = _data_root_and_manifest(
        input_root, expected_manifest_sha256
    )
    output = Path(output_root).resolve()
    _assert_output_separate(exact_root, output)
    candidate_root = output / "lsh" / "candidates"
    errors = _verify_stage(candidate_root, check_hashes=False)
    if errors:
        raise ValueError("Candidate stage is not ready: " + "; ".join(errors))
    stage_root = output / "candidate_samples"
    with _stage_lock(output, "exact-sample"):
        if stage_root.exists():
            errors = _verify_stage(stage_root, check_hashes=False)
            if not errors:
                _update_root_manifest(
                    output,
                    exact_root=exact_root,
                    data_root=data_root,
                    exact_manifest_sha256=exact_sha,
                    exact_manifest=exact_manifest,
                )
                print("[reuse] exact-score sample is already COMPLETE", flush=True)
                return _json_read(stage_root / "sample_summary.json")
            raise FileExistsError(
                f"Exact-sample stage exists but is invalid: {stage_root}"
            )
        exact_scratch_root = _prepare_exact_scratch_root(
            scratch_root,
            protected_paths=(exact_root, output),
        )
        stage_dir = _atomic_stage_dir(stage_root)
        started = time.perf_counter()
        candidate_db_path = candidate_root / "candidate_index.sqlite3"
        selected_pairs, sampling_metrics = _select_exact_sample(candidate_db_path)
        sampling_stratum_rows = sampling_metrics.pop("stratum_rows")
        scored_rows, text_count = _score_exact_sample_pairs(
            data_root=data_root,
            candidate_db_path=candidate_db_path,
            selected_pairs=selected_pairs,
            stage_dir=stage_dir,
            scratch_root=exact_scratch_root,
        )

        # The complete sample is deterministic and finite; sorting by pair ID
        # gives stable Parquet row order even if sampling strata are revisited.
        scored_rows.sort(key=lambda row: (row["record_id_a"], row["record_id_b"]))
        review_rows = _select_review_panel(scored_rows)
        _write_parquet_rows(stage_dir / "exact_scored_sample.parquet", scored_rows)
        _write_parquet_rows(stage_dir / "review_pairs.parquet", review_rows)
        _csv_write(
            stage_dir / "exact_sample_strata.csv",
            (
                "dimension",
                "stratum",
                "candidate_population_count",
                "exact_scored_sample_count",
                "descriptive_sample_fraction",
                "all_candidates_included_as_rare_pair_stratum",
                "rare_pair_inclusion_limit",
            ),
            sampling_stratum_rows,
        )
        score_summary, containment_summary, family_summary = _exact_summaries(
            scored_rows
        )
        _csv_write(
            stage_dir / "exact_score_summary.csv",
            (
                "exact_similarity_band",
                "source_subset_pair",
                "parlamento_partition",
                "sampled_candidate_pairs",
                "mean_minhash_estimate",
                "interpretation",
            ),
            score_summary,
        )
        _csv_write(
            stage_dir / "containment_summary.csv",
            (
                "diagnostic_flag",
                "source_subset_pair",
                "sampled_candidate_pairs",
                "automatic_duplicate_classification",
            ),
            containment_summary,
        )
        _csv_write(
            stage_dir / "high_similarity_family_summary.csv",
            (
                "exact_similarity_band",
                "family_dimension",
                "source_or_domain",
                "sampled_pair_endpoint_occurrences",
                "interpretation",
            ),
            family_summary,
        )
        summary_stage_path = output / "candidate_summary" / "census_summary.json"
        candidate_summary = (
            _json_read(summary_stage_path) if summary_stage_path.is_file() else {}
        )
        parliament_scored = sum(
            bool(row["parlamento_diagnostic_only"]) for row in scored_rows
        )
        tail_counts = Counter(
            row["exact_similarity_band"]
            for row in scored_rows
            if exact_tail_band(float(row["exact_jaccard"]))
        )
        sample_summary = {
            "schema_version": 1,
            "census_version": CENSUS_VERSION,
            "input_exact_manifest_sha256": exact_sha,
            "sampling_design": sampling_metrics,
            "exact_scored_pair_count": len(scored_rows),
            "review_pair_count": len(review_rows),
            "selected_text_endpoint_count": text_count,
            "exact_tail_pair_counts_in_stratified_sample": dict(
                sorted(tail_counts.items())
            ),
            "parlamento_candidates_exact_scored": parliament_scored,
            "parlamento_removal_eligible_count": 0,
            "parlamento_policy": "preserve-all; diagnostic-only; removal_eligible=false",
            "short_text_candidate_pairs_exact_scored": sum(
                min(int(row["words_a"]), int(row["words_b"])) < 20
                for row in scored_rows
            ),
            "containment_flagged_exact_scored_pairs": sum(
                bool(row["containment_flags"]) for row in scored_rows
            ),
            "boilerplate_suspect_exact_scored_pairs": sum(
                bool(row["boilerplate_suspect"]) for row in scored_rows
            ),
            "distinctive_overlap_document_frequency": {
                "computed": False,
                "reason": "D2c intentionally does not build a corpus-wide shingle document-frequency table or freeze a frequency cutoff.",
            },
            "high_similarity_tail_interpretation": (
                "Exact-score rows are a deterministic stratified candidate sample, not a probability sample. "
                "The tail supports human characterization but does not by itself estimate population prevalence."
            ),
            "decision_support": {
                "A_near_dedup_production_justified": "requires a material reviewed high-confidence population and a separately calibrated safe rule",
                "B_conservative_limited_near_dedup": "consider only if reviewed evidence supports particular source/provenance/length classes",
                "C_skip_global_near_dedup": "supported when demonstrated duplication is sparse or ambiguous relative to false-positive risk",
                "decision_frozen": False,
            },
            "candidate_population_summary": candidate_summary,
            "elapsed_wall_seconds": round(time.perf_counter() - started, 3),
            "peak_rss_bytes": near._peak_rss_bytes(),
            "production": {
                "near_dedup_production": "NOT RUN",
                "benchmark_decontamination": "PENDING",
                "C1": "IN PROGRESS",
                "C2": "PENDING",
            },
        }
        _json_write(stage_dir / "sample_summary.json", sample_summary)
        _json_write(stage_dir / "census_summary.json", sample_summary)
        signature_report_path = output / "signatures" / "resource_report.json"
        index_report_path = output / "lsh" / "index" / "resource_report.json"
        candidate_report_path = candidate_root / "resource_report.json"
        signature_report = (
            _json_read(signature_report_path) if signature_report_path.is_file() else {}
        )
        index_report = (
            _json_read(index_report_path) if index_report_path.is_file() else {}
        )
        candidate_report = (
            _json_read(candidate_report_path) if candidate_report_path.is_file() else {}
        )
        expected_records = int(exact_manifest.get("retained_record_count", 0))
        raw_signature_bytes = expected_records * SIGNATURE_BYTES
        resource_report = {
            "classification": "measured smoke or full-stage values plus explicitly labeled planning estimates",
            "fingerprint_stage": signature_report,
            "lsh_index_stage": index_report,
            "candidate_stage": candidate_report,
            "exact_sample_stage": {
                "elapsed_wall_seconds": sample_summary["elapsed_wall_seconds"],
                "peak_rss_bytes": sample_summary["peak_rss_bytes"],
                "exact_scored_pairs": len(scored_rows),
                "selected_text_endpoints": text_count,
            },
            "full_corpus_planning": {
                "records": expected_records,
                "raw_128_value_signature_bytes": raw_signature_bytes,
                "raw_128_value_signature_gib": round(raw_signature_bytes / 2**30, 2),
                "signature_parquet_size_reference_bytes": 21_000_000_000,
                "lsh_index_estimate_bytes_for_selected_40_bands": [
                    150_000_000_000,
                    300_000_000_000,
                ],
                "lsh_index_estimate_basis": (
                    "D2 pilot's 60-120 GB 16-band scenario scaled linearly to 40 total bands; "
                    "not a measured census index."
                ),
                "peak_working_disk_estimate_bytes_excluding_input": [
                    175_000_000_000,
                    350_000_000_000,
                ],
                "ram_gib_planning_range": [3, 6],
                "wall_runtime_planning_range_hours": [24, 72],
                "runtime_basis": (
                    "D2's 4.6-7.4 CPU-hour fingerprint projection is inherited from pilot scaling; "
                    "full LSH index construction, endpoint recovery, and candidate volumes have not "
                    "been measured. Allow 1-3 days wall time as a cautious operational window."
                ),
                "candidate_population_sensitivity_from_d2": [650_000, 6_530_000],
                "candidate_population_warning": (
                    "D2's linear and 10x candidate scenarios used a deliberately stratified/enriched "
                    "pilot and are not prevalence estimates or capacity guarantees."
                ),
            },
        }
        _json_write(stage_dir / "resource_report.json", resource_report)
        _write_stage_manifest(
            stage_dir,
            stage_name="exact_sample",
            input_manifest_sha256=exact_sha,
            input_signature_manifest_sha256=compute_file_sha256(
                (output / "signatures" / "manifest.json")
            ),
            metrics={
                "exact_scored_pairs": len(scored_rows),
                "review_pairs": len(review_rows),
                "selected_text_endpoints": text_count,
            },
        )
        os.replace(stage_dir, stage_root)
        _update_root_manifest(
            output,
            exact_root=exact_root,
            data_root=data_root,
            exact_manifest_sha256=exact_sha,
            exact_manifest=exact_manifest,
        )
        return sample_summary


def verify_census(
    *,
    input_root: Path | str = DEFAULT_INPUT_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    expected_manifest_sha256: str | None = EXPECTED_EXACT_MANIFEST_SHA256,
    require_complete: bool = True,
) -> list[str]:
    """Verify input identity, every completed stage manifest and artifact hash."""
    exact_root, data_root, exact_manifest, exact_sha = _data_root_and_manifest(
        input_root, expected_manifest_sha256
    )
    output = Path(output_root).resolve()
    _assert_output_separate(exact_root, output)
    errors: list[str] = []
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        return [f"Census root manifest is missing: {manifest_path}"]
    try:
        root_manifest = _json_read(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Census root manifest cannot be read: {exc}"]
    if root_manifest.get("input_exact_manifest_sha256") != exact_sha:
        errors.append("root manifest exact-input SHA-256 does not match")
    stages = (
        ("fingerprints", output / "signatures"),
        ("lsh index", output / "lsh" / "index"),
        ("candidates", output / "lsh" / "candidates"),
        ("candidate summary", output / "candidate_summary"),
        ("exact sample", output / "candidate_samples"),
    )
    for label, path in stages:
        if not path.exists():
            if require_complete:
                errors.append(f"{label} stage is missing")
            continue
        errors.extend(_verify_stage(path, check_hashes=True))
        stage_manifest_path = path / "manifest.json"
        if stage_manifest_path.is_file():
            stage_manifest = _json_read(stage_manifest_path)
            if stage_manifest.get("input_exact_manifest_sha256") != exact_sha:
                errors.append(f"{label} stage exact-input SHA-256 does not match")
    if require_complete and root_manifest.get("status") != "COMPLETE":
        errors.append("root manifest status is not COMPLETE")
    if root_manifest.get("near_dedup_production") != "NOT RUN":
        errors.append(
            "root manifest does not preserve near-dedup production as NOT RUN"
        )
    if root_manifest.get("C2") != "PENDING":
        errors.append("root manifest does not preserve C2 as PENDING")
    signature_manifest = output / "signatures" / "manifest.json"
    if signature_manifest.is_file():
        fingerprint_file = output / "signatures" / "fingerprints.parquet"
        expected = int(exact_manifest.get("retained_record_count", 0))
        if fingerprint_file.is_file() and expected:
            actual = pq.ParquetFile(fingerprint_file).metadata.num_rows
            if actual != expected:
                errors.append(
                    f"fingerprint row count {actual} differs from exact manifest {expected}"
                )
    return errors


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="near-census",
        description="Read-only, staged full-corpus near-duplication census",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for name in (
        "fingerprints",
        "lsh-index",
        "candidates",
        "lsh",
        "summarize",
        "exact-sample",
        "verify",
    ):
        child = subparsers.add_parser(name)
        child.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
        child.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        child.add_argument(
            "--expected-manifest-sha256",
            default=EXPECTED_EXACT_MANIFEST_SHA256,
            help="authoritative exact manifest digest (default: Gate C1 pinned value)",
        )
        if name == "fingerprints":
            child.add_argument("--batch-size", type=int, default=FINGERPRINT_BATCH_SIZE)
        if name == "exact-sample":
            child.add_argument(
                "--scratch-root",
                type=Path,
                default=DEFAULT_SCRATCH_ROOT,
                help=(
                    "local directory for temporary exact pair-scoring files "
                    f"(default: {DEFAULT_SCRATCH_ROOT})"
                ),
            )
        if name == "verify":
            child.add_argument(
                "--allow-incomplete",
                action="store_true",
                help="verify completed stages without requiring the final sample stage",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    try:
        if args.stage == "fingerprints":
            result = run_fingerprints(
                input_root=args.input_root,
                output_root=args.output_root,
                expected_manifest_sha256=args.expected_manifest_sha256,
                batch_size=args.batch_size,
            )
            print(json.dumps(result.get("metrics", {}), indent=2, sort_keys=True))
        elif args.stage == "lsh-index":
            index = run_lsh_index(
                input_root=args.input_root,
                output_root=args.output_root,
                expected_manifest_sha256=args.expected_manifest_sha256,
            )
            print(json.dumps(index.get("metrics", {}), indent=2, sort_keys=True))
        elif args.stage == "candidates":
            candidates = run_candidates(
                input_root=args.input_root,
                output_root=args.output_root,
                expected_manifest_sha256=args.expected_manifest_sha256,
            )
            print(json.dumps(candidates.get("metrics", {}), indent=2, sort_keys=True))
        elif args.stage == "lsh":
            index = run_lsh_index(
                input_root=args.input_root,
                output_root=args.output_root,
                expected_manifest_sha256=args.expected_manifest_sha256,
            )
            candidates = run_candidates(
                input_root=args.input_root,
                output_root=args.output_root,
                expected_manifest_sha256=args.expected_manifest_sha256,
            )
            print(
                json.dumps(
                    {
                        "index": index.get("metrics", {}),
                        "candidates": candidates.get("metrics", {}),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.stage == "summarize":
            print(
                json.dumps(
                    run_summarize(
                        input_root=args.input_root,
                        output_root=args.output_root,
                        expected_manifest_sha256=args.expected_manifest_sha256,
                    ),
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        elif args.stage == "exact-sample":
            print(
                json.dumps(
                    run_exact_sample(
                        input_root=args.input_root,
                        output_root=args.output_root,
                        scratch_root=args.scratch_root,
                        expected_manifest_sha256=args.expected_manifest_sha256,
                    ),
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            errors = verify_census(
                input_root=args.input_root,
                output_root=args.output_root,
                expected_manifest_sha256=args.expected_manifest_sha256,
                require_complete=not args.allow_incomplete,
            )
            if errors:
                for error in errors:
                    print(f"[FAIL] {error}", flush=True)
                return 1
            print("[PASS] near-census input identity, stages and artifacts verified")
        return 0
    except Exception as exc:
        print(f"[ERROR] near-census {args.stage} failed: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
