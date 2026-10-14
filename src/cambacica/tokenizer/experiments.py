"""Train and compare the fixed C2 SentencePiece tokenizer candidates.

The CLI is deliberately limited to ``train``, ``evaluate`` and ``compare``.
Evaluation selection checkpoints only validation references on local scratch;
it reads text from D1 only for the row groups containing selected documents.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import hashlib
import heapq
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import unicodedata
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from cambacica.corpus.dedup.exact_pipeline import occurrence_id_v2
from cambacica.corpus.manifest import compute_file_sha256

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs/c2_tokenizer_experiments.json"
DATA_ROOT = Path("/mnt/data/cambacica-base-180m")
DEFAULT_SAMPLE_ROOT = DATA_ROOT / "tokenizer-sample/c1-s2-v1/sample"
DEFAULT_SPLIT_ROOT = DATA_ROOT / "splits/c1-s1-v1"
DEFAULT_D1_ROOT = DATA_ROOT / "deduplicated/exact"
DEFAULT_TOKENIZER_ROOT = DATA_ROOT / "tokenizers/c2-v1"
DEFAULT_EVALUATION_ROOT = DATA_ROOT / "tokenizer-evaluations/c2-v1"
DEFAULT_SCRATCH_ROOT = Path("/tmp/cambacica-c2-v1")

PINNED_CONFIG_SHA256 = "2552ea368fd448dec29843d015210caf24f261b5495ab9292c373795801e1375"
PINNED_TRAIN_SAMPLE_SHA256 = (
    "93c6c5400115c80aa5f5a63b4b593a5bc45bc46ecea429c78ffa5718f978e97c"
)
PINNED_SPLIT_MANIFEST_SHA256 = (
    "8b937f90d40487c643fe48baa6b5c2950713fe2e0d2d534f3e2a3c08e74dbe16"
)
PINNED_SPLIT_ASSIGNMENTS_SHA256 = (
    "ef52e3cf3033934df13e3d16c207e1783c509dff6fb91954fb8c25ab719ece1e"
)
PINNED_D1_MANIFEST_SHA256 = (
    "57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428"
)

SAMPLE_MANIFEST_NAME = "sample_manifest.json"
ASSIGNMENTS_NAME = "assignments.parquet"
MANIFEST_NAME = "training_manifest.json"
CHECKSUMS_NAME = "checksums.sha256"
ASSIGNMENT_COLUMNS = (
    "doc_id",
    "source",
    "subset",
    "domain_category",
    "source_shard",
    "source_row_ordinal",
    "content_sha256",
    "normalized_words",
    "split",
)
EVAL_WORD_RE = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)
NUMBER_RE = re.compile(
    r"(?<![\w])[-+]?(?:\d+(?:[.,]\d+)*|[.,]\d+)(?:[eE][-+]?\d+)?%?(?!\w)"
)
SCIENTIFIC_RE = re.compile(r"(?<!\w)[-+]?\d+(?:[.,]\d+)?[eE][-+]?\d+(?!\w)")
ACCENTED = frozenset("áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, (canonical_json(value) + "\n").encode("utf-8"))


def read_checksum_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or len(digest) != 64 or name in result:
            raise ValueError(f"Malformed checksum manifest: {path}")
        result[name] = digest
    return result


def verify_checksum_entries(root: Path, expected: Mapping[str, str]) -> None:
    checksum_path = root / CHECKSUMS_NAME
    checksums = read_checksum_file(checksum_path)
    if checksums != dict(expected):
        raise ValueError(f"Checksum manifest entries differ from expected files: {root}")
    for name, digest in sorted(expected.items()):
        path = root / name
        if not path.is_file() or compute_file_sha256(path) != digest:
            raise ValueError(f"Artifact checksum mismatch: {path}")


def load_config(path: Path = DEFAULT_CONFIG, *, verify_pin: bool = True) -> dict[str, Any]:
    digest = compute_file_sha256(path)
    if verify_pin and digest != PINNED_CONFIG_SHA256:
        raise ValueError(
            f"C2 configuration checksum mismatch: expected {PINNED_CONFIG_SHA256}, got {digest}"
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "c2-tokenizer-experiments-v1":
        raise ValueError("Unsupported C2 configuration schema")
    if pa.__version__ != config.get("pyarrow_version"):
        raise RuntimeError(
            f"PyArrow version mismatch: expected {config.get('pyarrow_version')}, got {pa.__version__}"
        )
    if config.get("inputs") != {
        "training_sample_manifest_sha256": PINNED_TRAIN_SAMPLE_SHA256,
        "split_manifest_sha256": PINNED_SPLIT_MANIFEST_SHA256,
        "split_assignments_sha256": PINNED_SPLIT_ASSIGNMENTS_SHA256,
        "d1_manifest_sha256": PINNED_D1_MANIFEST_SHA256,
    }:
        raise ValueError("C2 configuration does not name the frozen C1 inputs")
    candidates = config.get("candidates", [])
    expected = {
        (model_type, vocab_size)
        for model_type in ("bpe", "unigram")
        for vocab_size in (16384, 24576, 32768)
    }
    actual = {(item.get("model_type"), item.get("vocab_size")) for item in candidates}
    if actual != expected or len(candidates) != 6:
        raise ValueError("C2 candidate matrix must contain the six pinned configurations")
    if len({item["id"] for item in candidates}) != 6:
        raise ValueError("C2 candidate IDs must be unique")
    return config


def candidate_config(config: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    for candidate in config["candidates"]:
        if candidate["id"] == candidate_id:
            return dict(candidate)
    raise ValueError(f"Unknown C2 candidate: {candidate_id}")


def _require_sentencepiece(expected_version: str):
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError(
            "Install the pinned tokenizer extra with `pip install -e '.[tokenizer]'`"
        ) from exc
    actual = getattr(spm, "__version__", "unknown")
    if actual != expected_version:
        raise RuntimeError(
            f"SentencePiece version mismatch: expected {expected_version}, got {actual}"
        )
    return spm


def _filesystem_type(path: Path) -> str:
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    result = subprocess.run(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(target)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def require_storage_layout(scratch_dir: Path, finalized_root: Path) -> None:
    scratch_type = _filesystem_type(scratch_dir)
    final_type = _filesystem_type(finalized_root)
    if scratch_type.startswith("nfs"):
        raise ValueError(f"Scratch must use local storage, found {scratch_type}")
    if not final_type.startswith("nfs"):
        raise ValueError(f"Final artifacts must use project NFS, found {final_type}")
    scratch_resolved = scratch_dir.resolve()
    output_resolved = finalized_root.resolve()
    if (
        scratch_resolved == output_resolved
        or scratch_resolved in output_resolved.parents
        or output_resolved in scratch_resolved.parents
    ):
        raise ValueError("Scratch and finalized output directories must be separate")


def _safe_relative(name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe artifact path: {name}")
    return Path(*relative.parts)


def validate_sample_manifest(sample_root: Path, expected_sha256: str) -> dict[str, Any]:
    manifest_path = sample_root / SAMPLE_MANIFEST_NAME
    actual = compute_file_sha256(manifest_path)
    if actual != expected_sha256:
        raise ValueError(f"Frozen C1-S2 sample manifest mismatch: {actual}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "MATERIALIZED":
        raise ValueError("C1-S2 sample is not marked MATERIALIZED")
    expected_files = {SAMPLE_MANIFEST_NAME: actual}
    for name, entry in manifest.get("artifacts", {}).items():
        relative = _safe_relative(name)
        path = sample_root / relative
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"C1-S2 sample artifact missing or changed size: {path}")
        expected_files[name] = str(entry["sha256"])
    verify_checksum_entries(sample_root, expected_files)
    return manifest


def _training_identity(
    config: Mapping[str, Any], candidate: Mapping[str, Any], config_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": "c2-training-run-v1",
        "candidate": dict(candidate),
        "config_sha256": config_sha256,
        "training_sample_manifest_sha256": PINNED_TRAIN_SAMPLE_SHA256,
        "training_seed": int(config["training_seed"]),
        "sentencepiece_version": config["sentencepiece_version"],
    }


def _write_training_text(
    sample_root: Path,
    manifest: Mapping[str, Any],
    corpus_path: Path,
    max_sentence_length: int,
) -> dict[str, Any]:
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = corpus_path.with_name(corpus_path.name + ".partial")
    words = documents = codepoints = 0
    largest_sentence_bytes = 0
    source_documents: Counter[str] = Counter()
    source_subset_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"documents": 0, "normalized_words": 0}
    )
    with temporary.open("wb") as stream:
        for name in sorted(manifest["artifacts"]):
            path = sample_root / _safe_relative(name)
            parquet = pq.ParquetFile(path)
            if {"source", "subset"} - set(parquet.schema_arrow.names):
                raise ValueError(f"C1-S2 sample lacks source/subset metadata: {path}")
            for batch in parquet.iter_batches(batch_size=512, columns=["text", "source", "subset"]):
                texts = batch.column(0).to_pylist()
                sources = batch.column(1).to_pylist()
                subsets = batch.column(2).to_pylist()
                for text, source, subset in zip(texts, sources, subsets, strict=True):
                    if not isinstance(text, str) or not text:
                        raise ValueError(f"C1-S2 contains invalid text: {path}")
                    if "\x00" in text:
                        raise ValueError("C1-S2 text contains U+0000, unsupported by SentencePiece")
                    encoded = text.encode("utf-8", errors="strict")
                    largest = max((len(part) for part in encoded.split(b"\n")), default=0)
                    largest_sentence_bytes = max(largest_sentence_bytes, largest)
                    if largest > max_sentence_length:
                        raise ValueError(
                            "C1-S2 contains a newline-delimited segment longer than "
                            "the configured SentencePiece maximum"
                        )
                    stream.write(encoded)
                    stream.write(b"\n")
                    documents += 1
                    words += len(text.split())
                    codepoints += len(text)
                    source_documents[str(source)] += 1
                    key = f"{source}/{subset}" if subset else str(source)
                    source_subset_counts[key]["documents"] += 1
                    source_subset_counts[key]["normalized_words"] += len(text.split())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, corpus_path)
    if documents != int(manifest["documents"]):
        raise ValueError("C1-S2 document count differs from its pinned manifest")
    if words != int(manifest["normalized_words"]):
        raise ValueError("C1-S2 word count differs from its pinned manifest")
    if dict(sorted(source_subset_counts.items())) != manifest.get("by_source_subset"):
        raise ValueError("C1-S2 source/subset counts differ from its pinned manifest")
    return {
        "documents": documents,
        "normalized_words": words,
        "unicode_codepoints": codepoints,
        "largest_sentence_bytes": largest_sentence_bytes,
        "source_documents": dict(sorted(source_documents.items())),
        "source_subset": dict(sorted(source_subset_counts.items())),
        "training_text_bytes": corpus_path.stat().st_size,
        "training_text_sha256": compute_file_sha256(corpus_path),
    }


def _validate_trained_model(
    model_path: Path,
    candidate: Mapping[str, Any],
    spm: Any,
) -> dict[str, Any]:
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    size = int(processor.get_piece_size())
    expected_size = int(candidate["vocab_size"])
    if size != expected_size:
        raise ValueError(f"Trained vocabulary has {size} pieces, expected {expected_size}")
    if (processor.unk_id(), processor.bos_id(), processor.eos_id(), processor.pad_id()) != (
        0,
        1,
        2,
        -1,
    ):
        raise ValueError("Trained special-token IDs differ from the C2 contract")
    byte_ids = [processor.piece_to_id(f"<0x{value:02X}>") for value in range(256)]
    if len(set(byte_ids)) != 256 or any(
        processor.id_to_piece(token_id) != f"<0x{value:02X}>"
        for value, token_id in enumerate(byte_ids)
    ):
        raise ValueError("Trained model lacks all 256 byte fallback pieces")
    return {
        "vocab_size": size,
        "special_tokens": {
            "unk": {"id": processor.unk_id(), "piece": processor.id_to_piece(0)},
            "bos": {"id": processor.bos_id(), "piece": processor.id_to_piece(1)},
            "eos": {"id": processor.eos_id(), "piece": processor.id_to_piece(2)},
            "pad": None,
        },
        "byte_fallback_piece_count": 256,
    }


def _verify_published_training(path: Path, expected_identity: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_identity") != dict(expected_identity):
        raise ValueError(f"Existing tokenizer artifact has a different identity: {path}")
    entries = dict(manifest.get("files", {}))
    for name in entries:
        _safe_relative(name)
    entries[MANIFEST_NAME] = compute_file_sha256(manifest_path)
    verify_checksum_entries(path, entries)
    expected_names = {*entries, CHECKSUMS_NAME}
    actual_names = {item.name for item in path.iterdir() if item.is_file()}
    if actual_names != expected_names:
        raise ValueError(f"Published tokenizer artifact has unexpected files: {path}")
    return manifest


def _publish_training_artifact(
    work_dir: Path,
    output_root: Path,
    candidate: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_id = str(candidate["id"])
    final_dir = output_root / candidate_id
    stage_dir = output_root / f".{candidate_id}.partial"
    output_root.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        return _verify_published_training(final_dir, run_identity)
    if stage_dir.exists():
        identity_path = stage_dir / ".run-identity.json"
        manifest_path = stage_dir / MANIFEST_NAME
        if identity_path.is_file():
            if json.loads(identity_path.read_text()) != dict(run_identity):
                raise ValueError(
                    f"Existing publication stage has a different identity: {stage_dir}"
                )
        elif not manifest_path.is_file():
            raise ValueError(
                f"Publication stage has no identity or manifest: {stage_dir}"
            )
        if manifest_path.is_file():
            if not identity_path.is_file():
                staged_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if staged_manifest.get("run_identity") != dict(run_identity):
                    raise ValueError(
                        f"Existing publication stage has a different identity: {stage_dir}"
                    )
            identity_path.unlink(missing_ok=True)
            try:
                manifest = _verify_published_training(stage_dir, run_identity)
            except (OSError, ValueError, json.JSONDecodeError):
                shutil.rmtree(stage_dir)
            else:
                (stage_dir / ".run-identity.json").unlink(missing_ok=True)
                stage_dir.rename(final_dir)
                return manifest
        else:
            shutil.rmtree(stage_dir)
    stage_dir.mkdir()
    atomic_json(stage_dir / ".run-identity.json", dict(run_identity))
    for name in ("tokenizer.model", "tokenizer.vocab"):
        shutil.copyfile(work_dir / name, stage_dir / name)
        with (stage_dir / name).open("rb") as stream:
            os.fsync(stream.fileno())
    files = {
        name: compute_file_sha256(stage_dir / name)
        for name in ("tokenizer.model", "tokenizer.vocab")
    }
    manifest = {
        "schema_version": "c2-tokenizer-artifact-v1",
        "candidate": dict(candidate),
        "run_identity": dict(run_identity),
        "files": files,
        "model": dict(metadata),
        "runtime": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "pyarrow": pa.__version__,
        },
        "implementation_sha256": compute_file_sha256(Path(__file__)),
    }
    atomic_json(stage_dir / MANIFEST_NAME, manifest)
    checksum_entries = dict(files)
    checksum_entries[MANIFEST_NAME] = compute_file_sha256(stage_dir / MANIFEST_NAME)
    checksum_text = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(checksum_entries.items())
    )
    atomic_write(stage_dir / CHECKSUMS_NAME, checksum_text.encode("ascii"))
    (stage_dir / ".run-identity.json").unlink(missing_ok=True)
    stage_dir.rename(final_dir)
    return manifest


def train_candidate(
    candidate_id: str,
    sample_root: Path,
    output_root: Path,
    scratch_dir: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
    require_nfs: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    candidate = candidate_config(config, candidate_id)
    config_sha = compute_file_sha256(config_path)
    identity = _training_identity(config, candidate, config_sha)
    if require_nfs:
        require_storage_layout(scratch_dir, output_root)
    finalized = output_root / candidate_id
    if finalized.exists():
        manifest = _verify_published_training(finalized, identity)
        return {
            "status": "ALREADY_PUBLISHED",
            "candidate": candidate_id,
            "output": str(finalized),
            "vocab_size": manifest["model"]["vocab_size"],
            "training_sample_manifest_sha256": PINNED_TRAIN_SAMPLE_SHA256,
        }
    spm = _require_sentencepiece(config["sentencepiece_version"])
    sample_manifest = validate_sample_manifest(
        sample_root, config["inputs"]["training_sample_manifest_sha256"]
    )
    work_dir = scratch_dir / candidate_id
    work_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = work_dir / "training.txt"
    state_path = work_dir / "run_identity.json"
    if state_path.exists():
        saved_identity = json.loads(state_path.read_text(encoding="utf-8"))
        if saved_identity != identity:
            raise ValueError(f"Scratch identity mismatch; use a fresh directory: {work_dir}")
    else:
        atomic_json(state_path, identity)
    statistics_path = work_dir / "training_input.json"
    if corpus_path.exists() and statistics_path.exists():
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
        if compute_file_sha256(corpus_path) != statistics["training_text_sha256"]:
            raise ValueError("Scratch training text checksum mismatch")
    else:
        statistics = _write_training_text(
            sample_root,
            sample_manifest,
            corpus_path,
            int(config["sentencepiece"]["max_sentence_length"]),
        )
        atomic_json(statistics_path, statistics)
    model_path = work_dir / "tokenizer.model"
    vocab_path = work_dir / "tokenizer.vocab"
    try:
        if not vocab_path.is_file():
            raise ValueError("SentencePiece vocabulary file is missing")
        model_metadata = _validate_trained_model(model_path, candidate, spm)
        _validate_vocab_file(vocab_path, candidate)
    except (OSError, ValueError):
        model_path.unlink(missing_ok=True)
        vocab_path.unlink(missing_ok=True)
        trainer_args = dict(config["sentencepiece"])
        trainer_args.update(
            {
                "input": str(corpus_path),
                "model_prefix": str(work_dir / "tokenizer"),
                "model_type": candidate["model_type"],
                "vocab_size": int(candidate["vocab_size"]),
                "minloglevel": 2,
            }
        )
        spm.set_random_generator_seed(int(config["training_seed"]))
        spm.SentencePieceTrainer.train(**trainer_args)
        model_metadata = _validate_trained_model(model_path, candidate, spm)
        _validate_vocab_file(vocab_path, candidate)
    model_metadata.update(statistics)
    published = _publish_training_artifact(
        work_dir, output_root, candidate, identity, model_metadata
    )
    return {
        "status": "PUBLISHED",
        "candidate": candidate_id,
        "output": str(output_root / candidate_id),
        "vocab_size": published["model"]["vocab_size"],
        "training_sample_manifest_sha256": PINNED_TRAIN_SAMPLE_SHA256,
    }


def _validate_vocab_file(vocab_path: Path, candidate: Mapping[str, Any]) -> None:
    with vocab_path.open(encoding="utf-8") as stream:
        count = sum(1 for _ in stream)
    if count != int(candidate["vocab_size"]):
        raise ValueError(
            f"SentencePiece .vocab has {count} entries, expected {candidate['vocab_size']}"
        )


def _validate_split_inputs(
    split_root: Path, d1_root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    split_manifest_path = split_root / "manifest.json"
    split_sha = compute_file_sha256(split_manifest_path)
    if split_sha != config["inputs"]["split_manifest_sha256"]:
        raise ValueError("C1-S1 split manifest checksum mismatch")
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    assignments_path = split_root / ASSIGNMENTS_NAME
    if split_manifest.get("output_sha256") != config["inputs"]["split_assignments_sha256"]:
        raise ValueError("C1-S1 assignment checksum differs from pinned identity")
    checksum_file = read_checksum_file(split_root / CHECKSUMS_NAME)
    if (
        checksum_file.get("manifest.json") != split_sha
        or checksum_file.get(ASSIGNMENTS_NAME)
        != config["inputs"]["split_assignments_sha256"]
    ):
        raise ValueError("C1-S1 published checksum manifest mismatch")
    d1_manifest_path = d1_root / "manifest.json"
    d1_sha = compute_file_sha256(d1_manifest_path)
    if d1_sha != config["inputs"]["d1_manifest_sha256"]:
        raise ValueError("Pinned D1 manifest checksum mismatch")
    d1_manifest = json.loads(d1_manifest_path.read_text(encoding="utf-8"))
    return split_manifest, d1_manifest, assignments_path


def _eval_index_identity(
    assignments_path: Path,
    config: Mapping[str, Any],
    config_sha256: str,
) -> dict[str, Any]:
    stat = assignments_path.stat()
    return {
        "schema_version": "c2-validation-index-v1",
        "config_sha256": config_sha256,
        "split_manifest_sha256": PINNED_SPLIT_MANIFEST_SHA256,
        "split_assignments_sha256": PINNED_SPLIT_ASSIGNMENTS_SHA256,
        "assignments_bytes": stat.st_size,
        "assignments_mtime_ns": stat.st_mtime_ns,
        "evaluation_seed": int(config["evaluation_seed"]),
        "max_documents_per_stratum": int(
            config["evaluation"]["max_documents_per_stratum"]
        ),
        "stratum_fields": list(config["evaluation"]["stratum_fields"]),
    }


def _open_eval_index(path: Path, identity: Mapping[str, Any]) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=120)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-32768")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS selected_refs(
          doc_id TEXT PRIMARY KEY,
          rank TEXT NOT NULL,
          source TEXT NOT NULL,
          subset TEXT NOT NULL,
          domain_category TEXT NOT NULL,
          source_shard TEXT NOT NULL,
          source_row_ordinal INTEGER NOT NULL,
          content_sha256 TEXT NOT NULL,
          normalized_words INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS selected_order
          ON selected_refs(source_shard,source_row_ordinal);
        CREATE TABLE IF NOT EXISTS selection_state(
          singleton INTEGER PRIMARY KEY CHECK(singleton=1),
          completed_row_groups INTEGER NOT NULL
        );
        """
    )
    saved = {key: json.loads(value) for key, value in db.execute("SELECT key,value FROM meta")}
    current = {"identity": dict(identity)}
    if saved and saved != current:
        db.close()
        raise ValueError("Evaluation scratch identity mismatch; use a fresh scratch directory")
    if not saved:
        db.execute(
            "INSERT INTO meta(key,value) VALUES ('identity',?)",
            (canonical_json(dict(identity)),),
        )
        db.execute("INSERT INTO selection_state VALUES (1,0)")
        db.commit()
    return db


def _priority(seed: int, doc_id: str, namespace: str) -> str:
    payload = f"{namespace}\0{seed}\0{doc_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _invert_text(value: str) -> str:
    return "".join(chr(0x10FFFF - ord(char)) for char in value)


def _prepare_validation_index(
    assignments_path: Path,
    scratch_dir: Path,
    config: Mapping[str, Any],
    config_sha256: str,
) -> tuple[sqlite3.Connection, dict[str, Any]]:
    expected_assignment_sha = config["inputs"]["split_assignments_sha256"]
    index_path = scratch_dir / "c2-validation-index.sqlite3"
    identity = _eval_index_identity(assignments_path, config, config_sha256)
    db = _open_eval_index(index_path, identity)
    meta = dict(db.execute("SELECT key,value FROM meta"))
    checksum_state = json.loads(meta.get("assignment_checksum_sha256", "null"))
    stat = assignments_path.stat()
    checksum_fingerprint = (stat.st_size, stat.st_mtime_ns)
    if checksum_state != {"fingerprint": list(checksum_fingerprint), "sha256": expected_assignment_sha}:
        actual_assignment_sha = compute_file_sha256(assignments_path)
        if actual_assignment_sha != expected_assignment_sha:
            db.close()
            raise ValueError("C1-S1 assignment payload checksum mismatch")
        db.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES ('assignment_checksum_sha256',?)",
            (
                canonical_json(
                    {"fingerprint": list(checksum_fingerprint), "sha256": actual_assignment_sha}
                ),
            ),
        )
        db.commit()
    parquet = pq.ParquetFile(assignments_path)
    required = set(ASSIGNMENT_COLUMNS)
    if required - set(parquet.schema_arrow.names):
        db.close()
        raise ValueError("C1-S1 assignments are missing required validation fields")
    state = db.execute(
        "SELECT completed_row_groups FROM selection_state WHERE singleton=1"
    ).fetchone()
    completed_groups = int(state[0]) if state else 0
    if completed_groups > parquet.num_row_groups:
        db.close()
        raise ValueError("Evaluation checkpoint exceeds assignment row-group count")
    heaps: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    refs = db.execute(
        "SELECT doc_id,rank,source,subset,domain_category,source_shard,"
        "source_row_ordinal,content_sha256,normalized_words FROM selected_refs"
    )
    cap = int(config["evaluation"]["max_documents_per_stratum"])
    for ref in refs:
        row = {
            "doc_id": ref[0],
            "rank": ref[1],
            "source": ref[2],
            "subset": ref[3],
            "domain_category": ref[4],
            "source_shard": ref[5],
            "source_row_ordinal": int(ref[6]),
            "content_sha256": ref[7],
            "normalized_words": int(ref[8]),
        }
        stratum_key = canonical_json([row["source"], row["subset"], row["domain_category"]])
        heapq.heappush(
            heaps[stratum_key],
            (-int(row["rank"], 16), _invert_text(row["doc_id"]), row["doc_id"]),
        )
    for group_index in range(completed_groups, parquet.num_row_groups):
        with db:
            for batch in parquet.iter_batches(
                batch_size=8192,
                row_groups=[group_index],
                columns=list(ASSIGNMENT_COLUMNS),
            ):
                for row in batch.to_pylist():
                    if row["split"] != "validation":
                        continue
                    source = str(row["source"])
                    subset = str(row["subset"] or "")
                    domain = str(row["domain_category"] or "unknown")
                    stratum_key = canonical_json([source, subset, domain])
                    rank = _priority(
                        int(config["evaluation_seed"]),
                        str(row["doc_id"]),
                        config["evaluation"]["selection_namespace"],
                    )
                    doc_id = str(row["doc_id"])
                    heap = heaps[stratum_key]
                    candidate_key = (int(rank, 16), doc_id)
                    reference = {
                        "doc_id": doc_id,
                        "rank": rank,
                        "source": source,
                        "subset": subset,
                        "domain_category": domain,
                        "source_shard": str(row["source_shard"]),
                        "source_row_ordinal": int(row["source_row_ordinal"]),
                        "content_sha256": str(row["content_sha256"]),
                        "normalized_words": int(row["normalized_words"]),
                    }
                    add = len(heap) < cap
                    old_doc = None
                    if not add:
                        worst_rank = -heap[0][0]
                        worst_doc = heap[0][2]
                        if candidate_key < (worst_rank, worst_doc):
                            old_doc = worst_doc
                            heapq.heapreplace(
                                heap,
                                (-candidate_key[0], _invert_text(doc_id), doc_id),
                            )
                            add = True
                    else:
                        heapq.heappush(
                            heap, (-candidate_key[0], _invert_text(doc_id), doc_id)
                        )
                    if add:
                        if old_doc is not None:
                            db.execute("DELETE FROM selected_refs WHERE doc_id=?", (old_doc,))
                        db.execute(
                            "INSERT INTO selected_refs VALUES (?,?,?,?,?,?,?,?,?)",
                            (
                                reference["doc_id"],
                                reference["rank"],
                                reference["source"],
                                reference["subset"],
                                reference["domain_category"],
                                reference["source_shard"],
                                reference["source_row_ordinal"],
                                reference["content_sha256"],
                                reference["normalized_words"],
                            ),
                        )
            db.execute(
                "UPDATE selection_state SET completed_row_groups=? WHERE singleton=1",
                (group_index + 1,),
            )
    db.commit()
    counts = {
        "documents": int(db.execute("SELECT count(*) FROM selected_refs").fetchone()[0]),
        "strata": int(
            db.execute(
                "SELECT count(*) FROM (SELECT source,subset,domain_category FROM selected_refs GROUP BY 1,2,3)"
            ).fetchone()[0]
        ),
        "assignment_row_groups": int(parquet.num_row_groups),
        "assignment_rows": int(parquet.metadata.num_rows),
    }
    if counts["documents"] == 0:
        db.close()
        raise ValueError("Pinned C1-S1 validation split yielded no evaluation references")
    if counts["strata"] > 128:
        db.close()
        raise ValueError("Evaluation sample exceeds the frozen 128-stratum bound")
    return db, counts


def _quantile(values: Sequence[int], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return float(ordered[index])


class MetricAccumulator:
    def __init__(self, technical_terms: set[str]):
        self.technical_terms = technical_terms
        self.documents = 0
        self.words = 0
        self.codepoints = 0
        self.tokens = 0
        self.eos_tokens = 0
        self.roundtrip_mismatches = 0
        self.unknown_tokens = 0
        self.byte_fallback_tokens = 0
        self.word_token_counts: list[int] = []
        self.accented_word_token_counts: list[int] = []
        self.number_token_counts: list[int] = []
        self.scientific_token_counts: list[int] = []
        self.punctuation_characters = 0
        self.punctuation_single_piece = 0
        self.lexical_frequency: Counter[str] = Counter()
        self.lexical_piece_count: dict[str, int] = {}
        self.lexical_byte_count: dict[str, int] = {}
        self.technical_frequency: Counter[str] = Counter()
        self.technical_piece_count: dict[str, int] = {}
        self.technical_byte_count: dict[str, int] = {}
        self.encoding_seconds = 0.0

    def add(
        self,
        text: str,
        ids: Sequence[int],
        word_token_counts: Sequence[int],
        lexical_observations: Sequence[tuple[str, int, int]],
        technical_terms: set[str],
        accent_counts: Sequence[int],
        number_counts: Sequence[int],
        scientific_counts: Sequence[int],
        punctuation_chars: int,
        punctuation_single: int,
        roundtrip_mismatch: bool,
        encode_seconds: float,
        unk_id: int,
        byte_ids: set[int],
    ) -> None:
        self.documents += 1
        self.words += len(text.split())
        self.codepoints += len(text)
        self.tokens += len(ids)
        self.eos_tokens += 1
        self.roundtrip_mismatches += int(roundtrip_mismatch)
        self.unknown_tokens += sum(token == unk_id for token in ids)
        self.byte_fallback_tokens += sum(token in byte_ids for token in ids)
        self.word_token_counts.extend(word_token_counts)
        self.accented_word_token_counts.extend(accent_counts)
        self.number_token_counts.extend(number_counts)
        self.scientific_token_counts.extend(scientific_counts)
        self.punctuation_characters += punctuation_chars
        self.punctuation_single_piece += punctuation_single
        self.encoding_seconds += encode_seconds
        for word, pieces, byte_count in lexical_observations:
            normalized = word.casefold()
            self.lexical_frequency[normalized] += 1
            self.lexical_piece_count[normalized] = pieces
            self.lexical_byte_count[normalized] = byte_count
            if normalized in technical_terms:
                self.technical_frequency[normalized] += 1
                self.technical_piece_count[normalized] = pieces
                self.technical_byte_count[normalized] = byte_count

    def result(self) -> dict[str, Any]:
        token_denominator = self.tokens
        rare = [
            word
            for word, frequency in self.lexical_frequency.items()
            if frequency <= 2
        ]
        rare_occurrences = sum(self.lexical_frequency[word] for word in rare)
        rare_pieces = sum(
            self.lexical_piece_count[word] * self.lexical_frequency[word]
            for word in rare
        )
        rare_bytes = sum(
            self.lexical_byte_count[word] * self.lexical_frequency[word]
            for word in rare
        )
        rare_single = sum(self.lexical_piece_count[word] == 1 for word in rare)
        technical_types = len(self.technical_frequency)
        technical_occurrences = sum(self.technical_frequency.values())
        technical_pieces = sum(
            self.technical_piece_count[word] * frequency
            for word, frequency in self.technical_frequency.items()
        )
        technical_bytes = sum(
            self.technical_byte_count[word] * frequency
            for word, frequency in self.technical_frequency.items()
        )
        technical_single = sum(
            self.technical_piece_count[word] == 1 for word in self.technical_frequency
        )
        return {
            "documents": self.documents,
            "normalized_words": self.words,
            "unicode_codepoints": self.codepoints,
            "raw_text_tokens": self.tokens,
            "eos_tokens": self.eos_tokens,
            "tokens_per_normalized_word": self.tokens / self.words if self.words else None,
            "tokens_per_word_including_eos": (
                (self.tokens + self.eos_tokens) / self.words if self.words else None
            ),
            "tokens_per_unicode_codepoint": (
                self.tokens / self.codepoints if self.codepoints else None
            ),
            "tokens_per_word_distribution": {
                "p50": _quantile(self.word_token_counts, 0.50),
                "p95": _quantile(self.word_token_counts, 0.95),
                "p99": _quantile(self.word_token_counts, 0.99),
            },
            "accented_portuguese_words": {
                "count": len(self.accented_word_token_counts),
                "mean_tokens": (
                    sum(self.accented_word_token_counts) / len(self.accented_word_token_counts)
                    if self.accented_word_token_counts
                    else None
                ),
                "single_piece_rate": (
                    sum(value == 1 for value in self.accented_word_token_counts)
                    / len(self.accented_word_token_counts)
                    if self.accented_word_token_counts
                    else None
                ),
                "p50_tokens": _quantile(self.accented_word_token_counts, 0.50),
                "p95_tokens": _quantile(self.accented_word_token_counts, 0.95),
            },
            "rare_sample_words": {
                "frequency_threshold": 2,
                "unique_forms": len(rare),
                "occurrences": rare_occurrences,
                "single_piece_type_rate": (
                    rare_single / len(rare) if rare else None
                ),
                "mean_tokens_per_occurrence": (
                    rare_pieces / rare_occurrences if rare_occurrences else None
                ),
                "byte_fallback_piece_rate": (
                    rare_bytes / rare_pieces if rare_pieces else None
                ),
            },
            "technical_lexicon": {
                "unique_terms_observed": technical_types,
                "term_occurrences": technical_occurrences,
                "single_piece_type_rate": (
                    technical_single / technical_types if technical_types else None
                ),
                "mean_tokens_per_occurrence": (
                    technical_pieces / technical_occurrences
                    if technical_occurrences
                    else None
                ),
                "byte_fallback_piece_rate": (
                    technical_bytes / technical_pieces if technical_pieces else None
                ),
                "covered_terms": sorted(self.technical_frequency),
            },
            "numbers": {
                "spans": len(self.number_token_counts),
                "mean_tokens_per_span": (
                    sum(self.number_token_counts) / len(self.number_token_counts)
                    if self.number_token_counts
                    else None
                ),
                "p95_tokens_per_span": _quantile(self.number_token_counts, 0.95),
                "scientific_notation_spans": len(self.scientific_token_counts),
                "scientific_mean_tokens_per_span": (
                    sum(self.scientific_token_counts) / len(self.scientific_token_counts)
                    if self.scientific_token_counts
                    else None
                ),
            },
            "punctuation": {
                "characters": self.punctuation_characters,
                "single_piece_rate": (
                    self.punctuation_single_piece / self.punctuation_characters
                    if self.punctuation_characters
                    else None
                ),
            },
            "round_trip": {
                "documents_exact": self.documents - self.roundtrip_mismatches,
                "documents_mismatched": self.roundtrip_mismatches,
                "exact_rate": (
                    (self.documents - self.roundtrip_mismatches) / self.documents
                    if self.documents
                    else None
                ),
            },
            "unknown_token_rate": (
                self.unknown_tokens / token_denominator if token_denominator else None
            ),
            "byte_fallback_token_rate": (
                self.byte_fallback_tokens / token_denominator if token_denominator else None
            ),
            "encoding_seconds": self.encoding_seconds,
            "documents_per_second": (
                self.documents / self.encoding_seconds if self.encoding_seconds else None
            ),
            "characters_per_second": (
                self.codepoints / self.encoding_seconds if self.encoding_seconds else None
            ),
            "tokens_per_second": (
                self.tokens / self.encoding_seconds if self.encoding_seconds else None
            ),
        }


def _nested(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _selected_refs(db: sqlite3.Connection) -> list[dict[str, Any]]:
    columns = (
        "doc_id,rank,source,subset,domain_category,source_shard,"
        "source_row_ordinal,content_sha256,normalized_words"
    )
    return [
        {
            "doc_id": row[0],
            "rank": row[1],
            "source": row[2],
            "subset": row[3],
            "domain_category": row[4],
            "source_shard": row[5],
            "source_row_ordinal": int(row[6]),
            "content_sha256": row[7],
            "normalized_words": int(row[8]),
        }
        for row in db.execute(f"SELECT {columns} FROM selected_refs ORDER BY doc_id")
    ]


def _measure_validation_document(
    row: Mapping[str, Any],
    reference: Mapping[str, Any],
    source_shard: str,
    ordinal: int,
    processor: Any,
    technical_terms: set[str],
    unk_id: int,
    byte_ids: set[int],
) -> tuple[tuple[str, str, str], tuple[Any, ...]]:
    text = row["text"]
    source = str(row["source"])
    subset = str(row["subset"] or "")
    domain = str(row["domain_category"] or "unknown")
    if (source, subset, domain) != (
        reference["source"],
        reference["subset"],
        reference["domain_category"],
    ):
        raise ValueError("C1-S1 validation labels differ from D1 record")
    if sha256_text(text) != reference["content_sha256"]:
        raise ValueError("C1-S1 validation text checksum differs from D1")
    if occurrence_id_v2(source, source_shard, ordinal) != reference["doc_id"]:
        raise ValueError("C1-S1 validation document ID differs from D1 coordinates")
    words = text.split()
    if len(words) != reference["normalized_words"]:
        raise ValueError("C1-S1 validation word count differs from D1 text")
    encode_start = time.perf_counter()
    ids = processor.encode(text, out_type=int)
    encode_seconds = time.perf_counter() - encode_start
    roundtrip_mismatch = processor.decode(ids) != text
    word_token_counts = [
        len(processor.encode(word, out_type=int)) for word in words
    ]
    lexical_observations = []
    accent_counts = []
    for match in EVAL_WORD_RE.finditer(text):
        word = match.group(0)
        word_ids = processor.encode(word, out_type=int)
        pieces = len(word_ids)
        byte_count = sum(token in byte_ids for token in word_ids)
        lexical_observations.append((word, pieces, byte_count))
        if any(char in ACCENTED for char in word):
            accent_counts.append(pieces)
    number_counts = [
        len(processor.encode(match.group(0), out_type=int))
        for match in NUMBER_RE.finditer(text)
    ]
    scientific_counts = [
        len(processor.encode(match.group(0), out_type=int))
        for match in SCIENTIFIC_RE.finditer(text)
    ]
    punctuation = [
        char for char in text if unicodedata.category(char).startswith("P")
    ]
    punctuation_single = sum(
        len(processor.encode(char, out_type=int)) == 1 for char in punctuation
    )
    observation = (
        text,
        ids,
        word_token_counts,
        lexical_observations,
        technical_terms,
        accent_counts,
        number_counts,
        scientific_counts,
        len(punctuation),
        punctuation_single,
        roundtrip_mismatch,
        encode_seconds,
        unk_id,
        byte_ids,
    )
    return (source, subset, domain), observation


def _evaluate_texts(
    model_path: Path,
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
    d1_root: Path,
    d1_manifest: Mapping[str, Any],
    db: sqlite3.Connection,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spm = _require_sentencepiece(config["sentencepiece_version"])
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    expected_vocab = int(candidate["vocab_size"])
    if processor.get_piece_size() != expected_vocab:
        raise ValueError("Tokenizer model vocabulary differs from candidate identity")
    byte_ids = {
        processor.piece_to_id(f"<0x{value:02X}>") for value in range(256)
    }
    unk_id = int(processor.unk_id())
    technical_terms = {term.casefold() for term in config["evaluation"]["technical_terms"]}
    references = _selected_refs(db)
    expected_sample_sha = sha256_text(
        canonical_json(
            [
                [
                    row["doc_id"],
                    row["rank"],
                    row["source"],
                    row["subset"],
                    row["domain_category"],
                    row["source_shard"],
                    row["source_row_ordinal"],
                    row["content_sha256"],
                    row["normalized_words"],
                ]
                for row in references
            ]
        )
    )
    selected_by_shard: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for reference in references:
        selected_by_shard[reference["source_shard"]][
            reference["source_row_ordinal"]
        ] = reference
    stratum_accumulators: dict[tuple[str, str, str], MetricAccumulator] = {}
    source_accumulators: dict[str, MetricAccumulator] = {}
    candidate_ids = set()
    selected_source_shards: list[dict[str, Any]] = []
    total_read_row_groups = total_compressed_bytes = 0
    evaluated_documents = 0
    for source_shard in sorted(selected_by_shard):
        relative = f"data/{source_shard}"
        entry = d1_manifest.get("output_files", {}).get(relative)
        if entry is None:
            raise ValueError(f"Selected validation shard absent from pinned D1: {source_shard}")
        source_path = d1_root / relative
        if source_path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"Selected D1 shard changed size: {source_path}")
        selected_source_shards.append(
            {
                "source_shard": source_shard,
                "bytes": int(entry["bytes"]),
                "sha256": str(entry["sha256"]),
            }
        )
        parquet = pq.ParquetFile(source_path)
        starts = []
        cursor = 0
        for group in range(parquet.num_row_groups):
            starts.append(cursor)
            cursor += parquet.metadata.row_group(group).num_rows
        group_references: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
        for ordinal, reference in selected_by_shard[source_shard].items():
            if ordinal < 0 or ordinal >= cursor:
                raise ValueError("Validation reference points outside its D1 shard")
            group_index = max(0, bisect_right(starts, ordinal) - 1)
            group_references[group_index][ordinal] = reference
        for group_index in sorted(group_references):
            group_metadata = parquet.metadata.row_group(group_index)
            total_read_row_groups += 1
            wanted_columns = {
                "text",
                "source",
                "subset",
                "domain_category",
                "content_sha256",
            }
            total_compressed_bytes += sum(
                group_metadata.column(index).total_compressed_size
                for index in range(group_metadata.num_columns)
                if group_metadata.column(index).path_in_schema in wanted_columns
            )
            group_start = starts[group_index]
            last_selected_ordinal = max(group_references[group_index])
            batch_start = group_start
            for batch in parquet.iter_batches(
                batch_size=64,
                row_groups=[group_index],
                columns=sorted(wanted_columns),
            ):
                for local_ordinal, row in enumerate(batch.to_pylist()):
                    ordinal = batch_start + local_ordinal
                    reference = group_references[group_index].get(ordinal)
                    if reference is None:
                        continue
                    key, observation = _measure_validation_document(
                        row,
                        reference,
                        source_shard,
                        ordinal,
                        processor,
                        technical_terms,
                        unk_id,
                        byte_ids,
                    )
                    source, _, _ = key
                    if key not in stratum_accumulators:
                        stratum_accumulators[key] = MetricAccumulator(technical_terms)
                    if source not in source_accumulators:
                        source_accumulators[source] = MetricAccumulator(technical_terms)
                    stratum_accumulators[key].add(*observation)
                    source_accumulators[source].add(*observation)
                    candidate_ids.add(reference["doc_id"])
                    evaluated_documents += 1
                batch_start += batch.num_rows
                if batch_start > last_selected_ordinal:
                    break
    if evaluated_documents != len(references):
        raise ValueError("Not all bounded validation references were evaluated")
    by_stratum = {
        f"{source}/{subset}/{domain}": accumulator.result()
        for (source, subset, domain), accumulator in sorted(stratum_accumulators.items())
    }
    by_source = {
        source: accumulator.result()
        for source, accumulator in sorted(source_accumulators.items())
    }
    numeric_macro_keys = (
        "tokens_per_normalized_word",
        "tokens_per_word_including_eos",
        "tokens_per_unicode_codepoint",
        "unknown_token_rate",
        "byte_fallback_token_rate",
    )
    source_macro = {
        metric: sum(float(result[metric]) for result in by_source.values() if result[metric] is not None)
        / sum(result[metric] is not None for result in by_source.values())
        for metric in numeric_macro_keys
    }
    for metric in (
        "accented_portuguese_words.mean_tokens",
        "accented_portuguese_words.single_piece_rate",
        "rare_sample_words.single_piece_type_rate",
        "rare_sample_words.mean_tokens_per_occurrence",
        "technical_lexicon.single_piece_type_rate",
        "technical_lexicon.mean_tokens_per_occurrence",
        "numbers.mean_tokens_per_span",
        "numbers.scientific_mean_tokens_per_span",
        "punctuation.single_piece_rate",
        "round_trip.exact_rate",
    ):
        values = [
            _nested(result, metric)
            for result in by_source.values()
            if _nested(result, metric) is not None
        ]
        source_macro[metric.replace(".", "_")] = (
            sum(float(value) for value in values) / len(values) if values else None
        )
    for quantile in ("p50", "p95", "p99"):
        source_macro[f"word_{quantile}"] = sum(
            float(result["tokens_per_word_distribution"][quantile])
            for result in by_source.values()
        ) / len(by_source)
    source_results = list(by_source.values())
    word_token_counts = [
        count
        for accumulator in stratum_accumulators.values()
        for count in accumulator.word_token_counts
    ]
    accented_token_counts = [
        count
        for accumulator in stratum_accumulators.values()
        for count in accumulator.accented_word_token_counts
    ]
    number_token_counts = [
        count
        for accumulator in stratum_accumulators.values()
        for count in accumulator.number_token_counts
    ]
    scientific_token_counts = [
        count
        for accumulator in stratum_accumulators.values()
        for count in accumulator.scientific_token_counts
    ]
    total_words = sum(int(result["normalized_words"]) for result in source_results)
    total_tokens = sum(int(result["raw_text_tokens"]) for result in source_results)
    total_chars = sum(int(result["unicode_codepoints"]) for result in source_results)
    total_documents = sum(int(result["documents"]) for result in source_results)
    total_eos = sum(int(result["eos_tokens"]) for result in source_results)
    encoding_seconds = sum(float(result["encoding_seconds"]) for result in source_results)
    punctuation_characters = sum(
        accumulator.punctuation_characters for accumulator in stratum_accumulators.values()
    )
    punctuation_single_piece = sum(
        accumulator.punctuation_single_piece
        for accumulator in stratum_accumulators.values()
    )
    overall = {
        "documents": total_documents,
        "normalized_words": total_words,
        "unicode_codepoints": total_chars,
        "raw_text_tokens": total_tokens,
        "eos_tokens": total_eos,
        "tokens_per_normalized_word": total_tokens / total_words if total_words else None,
        "tokens_per_word_including_eos": (
            (total_tokens + total_eos) / total_words if total_words else None
        ),
        "tokens_per_unicode_codepoint": total_tokens / total_chars if total_chars else None,
        "encoding_seconds": encoding_seconds,
        "documents_per_second": (
            total_documents / encoding_seconds if encoding_seconds else None
        ),
        "characters_per_second": (
            total_chars / encoding_seconds if encoding_seconds else None
        ),
        "tokens_per_second": total_tokens / encoding_seconds if encoding_seconds else None,
        "tokens_per_word_distribution": {
            "p50": _quantile(word_token_counts, 0.50),
            "p95": _quantile(word_token_counts, 0.95),
            "p99": _quantile(word_token_counts, 0.99),
        },
        "accented_portuguese_words": {
            "count": len(accented_token_counts),
            "mean_tokens": (
                sum(accented_token_counts) / len(accented_token_counts)
                if accented_token_counts
                else None
            ),
            "single_piece_rate": (
                sum(value == 1 for value in accented_token_counts)
                / len(accented_token_counts)
                if accented_token_counts
                else None
            ),
            "p50_tokens": _quantile(accented_token_counts, 0.50),
            "p95_tokens": _quantile(accented_token_counts, 0.95),
        },
        "numbers": {
            "spans": len(number_token_counts),
            "mean_tokens_per_span": (
                sum(number_token_counts) / len(number_token_counts)
                if number_token_counts
                else None
            ),
            "p95_tokens_per_span": _quantile(number_token_counts, 0.95),
            "scientific_notation_spans": len(scientific_token_counts),
            "scientific_mean_tokens_per_span": (
                sum(scientific_token_counts) / len(scientific_token_counts)
                if scientific_token_counts
                else None
            ),
        },
        "punctuation": {
            "characters": punctuation_characters,
            "single_piece_rate": (
                punctuation_single_piece / punctuation_characters
                if punctuation_characters
                else None
            ),
        },
        "unknown_token_rate": sum(
            float(result["unknown_token_rate"] or 0.0) * int(result["raw_text_tokens"])
            for result in source_results
        )
        / total_tokens
        if total_tokens
        else None,
        "byte_fallback_token_rate": sum(
            float(result["byte_fallback_token_rate"] or 0.0)
            * int(result["raw_text_tokens"])
            for result in source_results
        )
        / total_tokens
        if total_tokens
        else None,
        "round_trip_exact_documents": sum(
            int(result["round_trip"]["documents_exact"]) for result in source_results
        ),
        "round_trip_mismatched_documents": sum(
            int(result["round_trip"]["documents_mismatched"])
            for result in source_results
        ),
        "round_trip_exact_rate": (
            sum(int(result["round_trip"]["documents_exact"]) for result in source_results)
            / total_documents
            if total_documents
            else None
        ),
    }
    return (
        {
            "overall": overall,
            "by_source": by_source,
            "by_source_subset_domain": by_stratum,
            "source_balanced_macro": {
                "source_count": len(by_source),
                "aggregation": "arithmetic mean of source-level metrics; equal weight per source",
                "metrics": source_macro,
            },
        },
        {
            "evaluation_sample_sha256": expected_sample_sha,
            "selected_documents": len(candidate_ids),
            "selected_strata": len(by_stratum),
            "max_documents_per_stratum": int(
                config["evaluation"]["max_documents_per_stratum"]
            ),
            "selection_seed": int(config["evaluation_seed"]),
            "selection_namespace": config["evaluation"]["selection_namespace"],
            "selected_source_shards": selected_source_shards,
            "d1_row_groups_read": total_read_row_groups,
            "d1_compressed_row_group_bytes_read": total_compressed_bytes,
        },
    )


def _tokenizer_artifact_identity(tokenizer_dir: Path, candidate_id: str) -> tuple[dict, Path]:
    manifest_path = tokenizer_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("candidate", {}).get("id") != candidate_id:
        raise ValueError("Tokenizer artifact candidate ID mismatch")
    checksums = dict(manifest.get("files", {}))
    for name in checksums:
        _safe_relative(name)
    checksums[MANIFEST_NAME] = compute_file_sha256(manifest_path)
    verify_checksum_entries(tokenizer_dir, checksums)
    expected_names = {*checksums, CHECKSUMS_NAME}
    actual_names = {item.name for item in tokenizer_dir.iterdir() if item.is_file()}
    if actual_names != expected_names:
        raise ValueError("Tokenizer artifact has unexpected files")
    return manifest, tokenizer_dir / "tokenizer.model"


def _write_evaluation_artifact(
    output_path: Path,
    candidate: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    sample_identity: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        identity = existing.get("evaluation_identity")
        if identity == dict(sample_identity) and existing.get("candidate") == dict(candidate):
            return existing
        raise ValueError(f"Refusing to overwrite evaluation artifact: {output_path}")
    result = {
        "schema_version": "c2-tokenizer-evaluation-v1",
        "candidate": dict(candidate),
        "evaluation_identity": dict(sample_identity),
        "metrics": dict(evaluation),
        "runtime": dict(runtime),
    }
    atomic_json(output_path, result)
    return result


def evaluate_candidate(
    candidate_id: str,
    tokenizer_root: Path,
    split_root: Path,
    d1_root: Path,
    evaluation_root: Path,
    scratch_dir: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
    require_nfs: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    candidate = candidate_config(config, candidate_id)
    if require_nfs:
        require_storage_layout(scratch_dir, evaluation_root)
    _, d1_manifest, assignments_path = _validate_split_inputs(
        split_root, d1_root, config
    )
    tokenizer_manifest, model_path = _tokenizer_artifact_identity(
        tokenizer_root / candidate_id, candidate_id
    )
    config_sha = compute_file_sha256(config_path)
    expected_training_identity = _training_identity(config, candidate, config_sha)
    if tokenizer_manifest.get("run_identity") != expected_training_identity:
        raise ValueError("Tokenizer artifact was trained with a different C2 identity")
    db, selection = _prepare_validation_index(
        assignments_path, scratch_dir, config, config_sha
    )
    try:
        metrics, read_info = _evaluate_texts(
            model_path, candidate, config, d1_root, d1_manifest, db
        )
        model_sha = compute_file_sha256(model_path)
        eval_identity = {
            "config_sha256": config_sha,
            "training_sample_manifest_sha256": PINNED_TRAIN_SAMPLE_SHA256,
            "split_manifest_sha256": PINNED_SPLIT_MANIFEST_SHA256,
            "split_assignments_sha256": PINNED_SPLIT_ASSIGNMENTS_SHA256,
            "d1_manifest_sha256": PINNED_D1_MANIFEST_SHA256,
            "evaluation_sample_sha256": read_info["evaluation_sample_sha256"],
            "evaluation_seed": config["evaluation_seed"],
            "tokenizer_model_sha256": model_sha,
        }
        evaluation = {
            "candidate": candidate,
            "tokenizer_model_sha256": model_sha,
            "tokenizer_vocab_size": int(candidate["vocab_size"]),
            "model_parameter_cost": {
                "reference_hidden_width": int(config["model_budget"]["reference_hidden_width"]),
                "embedding_matrix_parameters": int(candidate["vocab_size"])
                * int(config["model_budget"]["reference_hidden_width"]),
                "output_head_parameters_if_untied": int(candidate["vocab_size"])
                * int(config["model_budget"]["reference_hidden_width"]),
            },
            "tokenizer_artifact_bytes": sum(
                int(path.stat().st_size)
                for path in (tokenizer_root / candidate_id).iterdir()
                if path.is_file()
            ),
            "evaluation_sample": {**selection, **read_info},
            **metrics,
        }
        evaluation_root.mkdir(parents=True, exist_ok=True)
        result = _write_evaluation_artifact(
            evaluation_root / f"{candidate_id}.json",
            candidate,
            evaluation,
            eval_identity,
            {
                "python": sys.version,
                "sentencepiece": config["sentencepiece_version"],
                "pyarrow": pa.__version__,
                "platform": platform.platform(),
            },
        )
        return {
            "status": "EVALUATED",
            "candidate": candidate_id,
            "output": str(evaluation_root / f"{candidate_id}.json"),
            "selected_documents": selection["documents"],
            "tokens_per_normalized_word": result["metrics"]["overall"][
                "tokens_per_normalized_word"
            ],
        }
    finally:
        db.close()


def _flat_metric_rows(
    results: Sequence[Mapping[str, Any]], level: str
) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        candidate_id = result["candidate"]["id"]
        if level == "summary":
            metrics = result["metrics"]
            rows.append(
                {
                    "candidate": candidate_id,
                    "algorithm": result["candidate"]["model_type"],
                    "vocab_size": result["candidate"]["vocab_size"],
                    "tokens_per_normalized_word": metrics["overall"][
                        "tokens_per_normalized_word"
                    ],
                    "tokens_per_word_including_eos": metrics["overall"][
                        "tokens_per_word_including_eos"
                    ],
                    "tokens_per_unicode_codepoint": metrics["overall"][
                        "tokens_per_unicode_codepoint"
                    ],
                    "word_p50": metrics["overall"]["tokens_per_word_distribution"]["p50"],
                    "word_p95": metrics["overall"]["tokens_per_word_distribution"]["p95"],
                    "word_p99": metrics["overall"]["tokens_per_word_distribution"]["p99"],
                    "accented_word_mean_tokens": metrics["overall"][
                        "accented_portuguese_words"
                    ]["mean_tokens"],
                    "accented_word_single_piece_rate": metrics["overall"][
                        "accented_portuguese_words"
                    ]["single_piece_rate"],
                    "unknown_token_rate": metrics["overall"]["unknown_token_rate"],
                    "byte_fallback_token_rate": metrics["overall"][
                        "byte_fallback_token_rate"
                    ],
                    "number_mean_tokens_per_span": metrics["overall"]["numbers"][
                        "mean_tokens_per_span"
                    ],
                    "scientific_mean_tokens_per_span": metrics["overall"]["numbers"][
                        "scientific_mean_tokens_per_span"
                    ],
                    "punctuation_single_piece_rate": metrics["overall"][
                        "punctuation"
                    ]["single_piece_rate"],
                    "source_macro_word_p50": metrics["source_balanced_macro"]["metrics"][
                        "word_p50"
                    ],
                    "source_macro_word_p95": metrics["source_balanced_macro"]["metrics"][
                        "word_p95"
                    ],
                    "source_macro_word_p99": metrics["source_balanced_macro"]["metrics"][
                        "word_p99"
                    ],
                    "source_macro_tokens_per_word": metrics[
                        "source_balanced_macro"
                    ]["metrics"]["tokens_per_normalized_word"],
                    "source_macro_accent_word_tokens": metrics[
                        "source_balanced_macro"
                    ]["metrics"].get("accented_portuguese_words_mean_tokens"),
                    "source_macro_rare_single_piece_rate": metrics[
                        "source_balanced_macro"
                    ]["metrics"].get("rare_sample_words_single_piece_type_rate"),
                    "source_macro_technical_single_piece_rate": metrics[
                        "source_balanced_macro"
                    ]["metrics"].get("technical_lexicon_single_piece_type_rate"),
                    "round_trip_exact_documents": metrics["overall"][
                        "round_trip_exact_documents"
                    ],
                    "round_trip_mismatched_documents": metrics["overall"][
                        "round_trip_mismatched_documents"
                    ],
                    "round_trip_exact_rate": metrics["overall"][
                        "round_trip_exact_rate"
                    ],
                    "documents_per_second": metrics["overall"]["documents_per_second"],
                    "characters_per_second": metrics["overall"]["characters_per_second"],
                    "tokens_per_second": metrics["overall"]["tokens_per_second"],
                    "tokenizer_artifact_bytes": result["metrics"][
                        "tokenizer_artifact_bytes"
                    ],
                    "reference_hidden_width": metrics["model_parameter_cost"][
                        "reference_hidden_width"
                    ],
                    "embedding_matrix_parameters": metrics["model_parameter_cost"][
                        "embedding_matrix_parameters"
                    ],
                    "output_head_parameters_if_untied": metrics[
                        "model_parameter_cost"
                    ]["output_head_parameters_if_untied"],
                }
            )
        elif level == "source":
            for source, metrics in result["metrics"]["by_source"].items():
                rows.append(
                    {
                        "candidate": candidate_id,
                        "source": source,
                        "documents": metrics["documents"],
                        "normalized_words": metrics["normalized_words"],
                        "tokens_per_normalized_word": metrics[
                            "tokens_per_normalized_word"
                        ],
                        "tokens_per_unicode_codepoint": metrics[
                            "tokens_per_unicode_codepoint"
                        ],
                        "unknown_token_rate": metrics["unknown_token_rate"],
                        "byte_fallback_token_rate": metrics["byte_fallback_token_rate"],
                        "word_p50": metrics["tokens_per_word_distribution"]["p50"],
                        "word_p95": metrics["tokens_per_word_distribution"]["p95"],
                        "word_p99": metrics["tokens_per_word_distribution"]["p99"],
                        "accented_word_mean_tokens": metrics[
                            "accented_portuguese_words"
                        ]["mean_tokens"],
                        "accented_word_single_piece_rate": metrics[
                            "accented_portuguese_words"
                        ]["single_piece_rate"],
                        "rare_sample_single_piece_rate": metrics[
                            "rare_sample_words"
                        ]["single_piece_type_rate"],
                        "rare_sample_mean_tokens": metrics["rare_sample_words"][
                            "mean_tokens_per_occurrence"
                        ],
                        "rare_sample_byte_fallback_piece_rate": metrics[
                            "rare_sample_words"
                        ]["byte_fallback_piece_rate"],
                        "technical_lexicon_single_piece_rate": metrics[
                            "technical_lexicon"
                        ]["single_piece_type_rate"],
                        "technical_lexicon_mean_tokens": metrics["technical_lexicon"][
                            "mean_tokens_per_occurrence"
                        ],
                        "technical_lexicon_byte_fallback_piece_rate": metrics[
                            "technical_lexicon"
                        ]["byte_fallback_piece_rate"],
                        "number_mean_tokens_per_span": metrics["numbers"][
                            "mean_tokens_per_span"
                        ],
                        "scientific_mean_tokens": metrics["numbers"][
                            "scientific_mean_tokens_per_span"
                        ],
                        "punctuation_single_piece_rate": metrics["punctuation"][
                            "single_piece_rate"
                        ],
                        "round_trip_exact_rate": metrics["round_trip"]["exact_rate"],
                        "documents_per_second": metrics["documents_per_second"],
                        "characters_per_second": metrics["characters_per_second"],
                        "tokens_per_second": metrics["tokens_per_second"],
                    }
                )
        else:
            for stratum, metrics in result["metrics"]["by_source_subset_domain"].items():
                rows.append(
                    {
                        "candidate": candidate_id,
                        "source_subset_domain": stratum,
                        "documents": metrics["documents"],
                        "normalized_words": metrics["normalized_words"],
                        "unicode_codepoints": metrics["unicode_codepoints"],
                        "tokens_per_normalized_word": metrics[
                            "tokens_per_normalized_word"
                        ],
                        "tokens_per_unicode_codepoint": metrics[
                            "tokens_per_unicode_codepoint"
                        ],
                        "unknown_token_rate": metrics["unknown_token_rate"],
                        "byte_fallback_token_rate": metrics["byte_fallback_token_rate"],
                        "word_p50": metrics["tokens_per_word_distribution"]["p50"],
                        "word_p95": metrics["tokens_per_word_distribution"]["p95"],
                        "word_p99": metrics["tokens_per_word_distribution"]["p99"],
                        "accented_word_mean_tokens": metrics[
                            "accented_portuguese_words"
                        ]["mean_tokens"],
                        "accented_word_single_piece_rate": metrics[
                            "accented_portuguese_words"
                        ]["single_piece_rate"],
                        "rare_sample_single_piece_rate": metrics[
                            "rare_sample_words"
                        ]["single_piece_type_rate"],
                        "rare_sample_mean_tokens": metrics["rare_sample_words"][
                            "mean_tokens_per_occurrence"
                        ],
                        "rare_sample_byte_fallback_piece_rate": metrics[
                            "rare_sample_words"
                        ]["byte_fallback_piece_rate"],
                        "technical_lexicon_single_piece_rate": metrics[
                            "technical_lexicon"
                        ]["single_piece_type_rate"],
                        "technical_lexicon_mean_tokens": metrics["technical_lexicon"][
                            "mean_tokens_per_occurrence"
                        ],
                        "technical_lexicon_byte_fallback_piece_rate": metrics[
                            "technical_lexicon"
                        ]["byte_fallback_piece_rate"],
                        "number_spans": metrics["numbers"]["spans"],
                        "number_mean_tokens_per_span": metrics["numbers"][
                            "mean_tokens_per_span"
                        ],
                        "scientific_notation_spans": metrics["numbers"][
                            "scientific_notation_spans"
                        ],
                        "scientific_mean_tokens_per_span": metrics["numbers"][
                            "scientific_mean_tokens_per_span"
                        ],
                        "punctuation_characters": metrics["punctuation"]["characters"],
                        "punctuation_single_piece_rate": metrics["punctuation"][
                            "single_piece_rate"
                        ],
                        "round_trip_exact_rate": metrics["round_trip"]["exact_rate"],
                        "documents_per_second": metrics["documents_per_second"],
                        "characters_per_second": metrics["characters_per_second"],
                        "tokens_per_second": metrics["tokens_per_second"],
                    }
                )
    return rows


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        return b""
    fieldnames = list(rows[0])
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def compare_evaluations(
    evaluation_root: Path,
    output_dir: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
    require_nfs: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    if require_nfs:
        require_storage_layout(DEFAULT_SCRATCH_ROOT, output_dir)
    expected_candidates = {item["id"]: item for item in config["candidates"]}
    evaluations: list[dict[str, Any]] = []
    for candidate_id, candidate in sorted(expected_candidates.items()):
        path = evaluation_root / f"{candidate_id}.json"
        if not path.is_file():
            raise ValueError(f"Missing evaluation artifact for {candidate_id}: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("candidate") != candidate:
            raise ValueError(f"Evaluation candidate identity mismatch: {path}")
        if result.get("evaluation_identity", {}).get("config_sha256") != compute_file_sha256(
            config_path
        ):
            raise ValueError(f"Evaluation config identity mismatch: {path}")
        evaluations.append(result)
    shared_identity_fields = (
        "config_sha256",
        "training_sample_manifest_sha256",
        "split_manifest_sha256",
        "split_assignments_sha256",
        "d1_manifest_sha256",
        "evaluation_sample_sha256",
        "evaluation_seed",
    )
    first_identity = evaluations[0]["evaluation_identity"]
    for result in evaluations[1:]:
        if any(
            result["evaluation_identity"].get(field) != first_identity.get(field)
            for field in shared_identity_fields
        ):
            raise ValueError("Evaluation artifacts do not share one fixed validation sample")
    if output_dir.exists():
        raise ValueError(f"Refusing to overwrite comparison directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.with_name(output_dir.name + ".partial")
    if stage.exists():
        raise ValueError(f"Incomplete comparison stage exists; inspect and remove: {stage}")
    stage.mkdir()
    summary_rows = _flat_metric_rows(evaluations, "summary")
    source_rows = _flat_metric_rows(evaluations, "source")
    stratum_rows = _flat_metric_rows(evaluations, "stratum")
    table_files = {
        "summary.csv": summary_rows,
        "by_source.csv": source_rows,
        "by_source_subset_domain.csv": stratum_rows,
    }
    for name, rows in table_files.items():
        atomic_write(stage / name, _csv_bytes(rows))
    manifest = {
        "schema_version": "c2-tokenizer-comparison-v1",
        "config_sha256": compute_file_sha256(config_path),
        "candidate_count": len(evaluations),
        "candidate_ids": [item["candidate"]["id"] for item in evaluations],
        "evaluation_artifact_sha256": {
            item["candidate"]["id"]: compute_file_sha256(
                evaluation_root / f"{item['candidate']['id']}.json"
            )
            for item in evaluations
        },
        "winner_declared": False,
        "tables": sorted(table_files),
    }
    atomic_json(stage / "comparison_manifest.json", manifest)
    names = [*table_files, "comparison_manifest.json"]
    checksum_entries = {
        name: compute_file_sha256(stage / name) for name in names
    }
    checksums = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(checksum_entries.items())
    )
    atomic_write(stage / CHECKSUMS_NAME, checksums.encode("ascii"))
    stage.rename(output_dir)
    return {
        "status": "COMPARED",
        "candidates": len(evaluations),
        "output": str(output_dir),
        "winner_declared": False,
    }


def _add_candidate_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidate", required=True, help="Pinned candidate ID")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    train_parser = actions.add_parser("train", help="Train one pinned C2 candidate")
    _add_candidate_cli_args(train_parser)
    train_parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    train_parser.add_argument("--output-root", type=Path, default=DEFAULT_TOKENIZER_ROOT)
    train_parser.add_argument("--scratch-dir", type=Path, default=DEFAULT_SCRATCH_ROOT / "train")

    evaluate_parser = actions.add_parser(
        "evaluate", help="Evaluate one tokenizer on the fixed C1-S1 validation sample"
    )
    _add_candidate_cli_args(evaluate_parser)
    evaluate_parser.add_argument("--tokenizer-root", type=Path, default=DEFAULT_TOKENIZER_ROOT)
    evaluate_parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    evaluate_parser.add_argument("--d1-root", type=Path, default=DEFAULT_D1_ROOT)
    evaluate_parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    evaluate_parser.add_argument("--scratch-dir", type=Path, default=DEFAULT_SCRATCH_ROOT / "evaluate")

    compare_parser = actions.add_parser(
        "compare", help="Build tables from the six pinned evaluation artifacts"
    )
    compare_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    compare_parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    compare_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_EVALUATION_ROOT / "comparison-v1"
    )
    compare_parser.add_argument("--scratch-dir", type=Path, default=DEFAULT_SCRATCH_ROOT / "compare")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.action == "train":
        result = train_candidate(
            args.candidate,
            args.sample_root,
            args.output_root,
            args.scratch_dir,
            config_path=args.config,
        )
    elif args.action == "evaluate":
        result = evaluate_candidate(
            args.candidate,
            args.tokenizer_root,
            args.split_root,
            args.d1_root,
            args.evaluation_root,
            args.scratch_dir,
            config_path=args.config,
        )
    else:
        require_storage_layout(args.scratch_dir, args.output_dir)
        result = compare_evaluations(
            args.evaluation_root,
            args.output_dir,
            config_path=args.config,
            require_nfs=False,
        )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
