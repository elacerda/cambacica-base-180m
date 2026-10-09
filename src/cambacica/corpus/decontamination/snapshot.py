"""Build and verify small, immutable benchmark snapshots."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from .matcher import normalize_match_text, tokenize_with_offsets


SNAPSHOT_VERSION = "c1-bd2-snapshot-v1"
CALAME_REVISION = "353671bc95cc3d94d488201f67d41b640eb80c55"
BELEBELE_REVISION = "d4c91dedc9de484dbea7b7d940f898f59fd135e9"

DEFAULT_SNAPSHOT_ROOT = Path(
    "/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2/snapshot"
)

REGISTRY_SCHEMA = pa.schema(
    [
        pa.field("benchmark_name", pa.string(), nullable=False),
        pa.field("repository", pa.string(), nullable=False),
        pa.field("revision", pa.string(), nullable=False),
        pa.field("configuration", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("source_configuration", pa.string(), nullable=False),
        pa.field("source_split", pa.string(), nullable=False),
        pa.field("source_row_id", pa.string(), nullable=False),
        pa.field("example_id", pa.string(), nullable=False),
        pa.field("row_ordinal", pa.int32(), nullable=False),
        pa.field("source_file", pa.string(), nullable=False),
        pa.field("source_file_sha256", pa.string(), nullable=False),
        pa.field("row_sha256", pa.string(), nullable=False),
        pa.field("source_category", pa.string()),
        pa.field("context", pa.string()),
        pa.field("target_word", pa.string()),
        pa.field("passage", pa.string()),
        pa.field("question", pa.string()),
        pa.field("options_json", pa.string()),
        pa.field("answer", pa.string()),
        pa.field("answer_source_value", pa.string()),
        pa.field("answer_encoding", pa.string()),
        pa.field("public_answer_available", pa.bool_(), nullable=False),
        pa.field("field_roles_json", pa.string(), nullable=False),
        pa.field("license_declared", pa.string(), nullable=False),
        pa.field("attribution", pa.string(), nullable=False),
        pa.field("provenance_caveat", pa.string(), nullable=False),
        pa.field("raw_row_json", pa.string(), nullable=False),
    ]
)

MATCH_FIELD_SCHEMA = pa.schema(
    [
        pa.field("benchmark_name", pa.string(), nullable=False),
        pa.field("example_id", pa.string(), nullable=False),
        pa.field("source_row_id", pa.string(), nullable=False),
        pa.field("source_file_sha256", pa.string(), nullable=False),
        pa.field("source_category", pa.string()),
        pa.field("field_id", pa.string(), nullable=False),
        pa.field("field_role", pa.string(), nullable=False),
        pa.field("matchable", pa.bool_(), nullable=False),
        pa.field("original_text", pa.string(), nullable=False),
        pa.field("normalized_text", pa.string(), nullable=False),
        pa.field("normalized_tokens_json", pa.string(), nullable=False),
        pa.field("normalized_sha256", pa.string(), nullable=False),
        pa.field("token_count", pa.int32(), nullable=False),
    ]
)

CALAME_PROVENANCE = (
    "The CALAME introduction reports 406 handwritten examples and 1,670 "
    "GPT-3.5-generated examples derived from Portuguese Wikipedia, OSCAR, and "
    "Arquivo.pt, followed by human review. Per-example upstream source IDs are "
    "not provided. The pinned dataset README declares MIT; that label does not "
    "independently establish rights to embedded third-party material."
)
BELEBELE_PROVENANCE = (
    "The dataset card describes four-choice questions over FLORES-200 passages; "
    "the row link field is retained as the source passage identifier. The "
    "CC-BY-SA-4.0 label does not independently resolve rights to every cited "
    "upstream passage or linked source page."
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _example_id(
    benchmark: str, revision: str, config: str, split: str, row_id: str, row_sha: str
) -> str:
    identity = "\0".join((benchmark, revision, config, split, row_id, row_sha))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _add_match_field(
    fields: list[dict[str, Any]],
    *,
    benchmark: str,
    example_id: str,
    row_id: str,
    source_sha: str,
    category: str | None,
    role: str,
    text: str,
    matchable: bool,
) -> None:
    tokens = tokenize_with_offsets(text)
    norm = normalize_match_text(text)
    field_id = f"{example_id}:{role}"
    fields.append(
        {
            "benchmark_name": benchmark,
            "example_id": example_id,
            "source_row_id": row_id,
            "source_file_sha256": source_sha,
            "source_category": category,
            "field_id": field_id,
            "field_role": role,
            "matchable": matchable,
            "original_text": text,
            "normalized_text": norm,
            "normalized_tokens_json": canonical_json(
                [
                    {"token": item.text, "start": item.start, "end": item.end}
                    for item in tokens
                ]
            ),
            "normalized_sha256": sha256_bytes(
                "\x1f".join(item.text for item in tokens).encode("utf-8")
            ),
            "token_count": len(tokens),
        }
    )


def _calame_category_sets(
    source_dir: Path,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    handwritten = _read_jsonl(source_dir / "calamept_handwritten_only.jsonl")
    generated = _read_jsonl(source_dir / "calamept_gen_only.jsonl")
    hand_set = {(row["sentence"], row["last_word"]) for row in handwritten}
    generated_set = {(row["sentence"], row["last_word"]) for row in generated}
    if len(hand_set) != len(handwritten) or len(generated_set) != len(generated):
        raise ValueError("CALAME category files contain duplicate text/target pairs")
    if hand_set & generated_set:
        raise ValueError("CALAME handwritten and generated categories overlap")
    return hand_set, generated_set


def _calame_records(
    source_dir: Path, checksums: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    rows = _read_jsonl(source_dir / "calamept_all.jsonl")
    hand_set, generated_set = _calame_category_sets(source_dir)
    counts = Counter()
    registry: list[dict[str, Any]] = []
    fields: list[dict[str, Any]] = []
    row_ids: set[str] = set()
    for ordinal, row in enumerate(rows):
        if set(row) != {"id", "sentence", "last_word"}:
            raise ValueError(f"Unexpected CALAME row fields: {sorted(row)}")
        text_pair = (row["sentence"], row["last_word"])
        if text_pair in hand_set:
            category = "handwritten"
        elif text_pair in generated_set:
            category = "generated"
        else:
            raise ValueError(f"CALAME row {ordinal} has no source category")
        row_id = str(row["id"])
        if row_id in row_ids:
            raise ValueError(f"Duplicate CALAME source row id: {row_id}")
        row_ids.add(row_id)
        counts[category] += 1
        row_sha = sha256_bytes(canonical_json(row).encode("utf-8"))
        example_id = _example_id(
            "calame_pt", CALAME_REVISION, "all", "all", row_id, row_sha
        )
        source_sha = checksums["calamept_all.jsonl"]
        registry.append(
            {
                "benchmark_name": "calame_pt",
                "repository": "NOVA-vision-language/calame-pt",
                "revision": CALAME_REVISION,
                "configuration": "all",
                "split": "all_evaluation_only",
                "source_configuration": "default",
                "source_split": "train",
                "source_row_id": row_id,
                "example_id": example_id,
                "row_ordinal": ordinal,
                "source_file": "calamept_all.jsonl",
                "source_file_sha256": source_sha,
                "row_sha256": row_sha,
                "source_category": category,
                "context": row["sentence"],
                "target_word": row["last_word"],
                "passage": None,
                "question": None,
                "options_json": None,
                "answer": row["last_word"],
                "answer_source_value": row["last_word"],
                "answer_encoding": "literal_target_word",
                "public_answer_available": True,
                "field_roles_json": canonical_json(
                    ["context", "target_word", "complete_item"]
                ),
                "license_declared": "MIT",
                "attribution": (
                    "NOVA-vision-language, CALAME-PT; cite Lopes et al., "
                    "PROPOR 2024, https://aclanthology.org/2024.propor-1.45/"
                ),
                "provenance_caveat": CALAME_PROVENANCE,
                "raw_row_json": canonical_json(row),
            }
        )
        _add_match_field(
            fields,
            benchmark="calame_pt",
            example_id=example_id,
            row_id=row_id,
            source_sha=source_sha,
            category=category,
            role="context",
            text=row["sentence"],
            matchable=True,
        )
        _add_match_field(
            fields,
            benchmark="calame_pt",
            example_id=example_id,
            row_id=row_id,
            source_sha=source_sha,
            category=category,
            role="target_word",
            text=row["last_word"],
            matchable=False,
        )
        _add_match_field(
            fields,
            benchmark="calame_pt",
            example_id=example_id,
            row_id=row_id,
            source_sha=source_sha,
            category=category,
            role="complete_item",
            text=f"{row['sentence']} {row['last_word']}",
            # Keep malformed/blank target rows in the evaluation snapshot, but
            # do not treat their context alone as an exact complete-item hit.
            matchable=bool(tokenize_with_offsets(row["last_word"])),
        )

    if len(rows) != len(hand_set) + len(generated_set):
        raise ValueError("CALAME aggregate count differs from category source files")
    return registry, fields, {"all": len(rows), **dict(counts)}


def _belebele_records(
    source_dir: Path, checksums: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    rows = _read_jsonl(source_dir / "data" / "por_Latn.jsonl")
    registry: list[dict[str, Any]] = []
    fields: list[dict[str, Any]] = []
    source_sha = checksums["data/por_Latn.jsonl"]
    row_ids: set[str] = set()
    passage_ids: set[str] = set()
    for ordinal, row in enumerate(rows):
        expected = {
            "link",
            "question_number",
            "flores_passage",
            "question",
            "mc_answer1",
            "mc_answer2",
            "mc_answer3",
            "mc_answer4",
            "correct_answer_num",
            "dialect",
            "ds",
        }
        if set(row) != expected:
            raise ValueError(f"Unexpected Belebele row fields: {sorted(row)}")
        if row["dialect"] != "por_Latn":
            raise ValueError(f"Unexpected Belebele dialect: {row['dialect']!r}")
        qnum = str(row["question_number"])
        if qnum not in {"1", "2"}:
            raise ValueError(f"Unexpected Belebele question number: {qnum!r}")
        row_id = f"{row['link']}#q{qnum}"
        if row_id in row_ids:
            raise ValueError(f"Duplicate Belebele source row id: {row_id}")
        row_ids.add(row_id)
        passage_ids.add(str(row["link"]))
        options = [row[f"mc_answer{index}"] for index in range(1, 5)]
        answer_value = str(row["correct_answer_num"])
        if answer_value not in {"1", "2", "3", "4"}:
            raise ValueError(f"Invalid one-indexed Belebele answer: {answer_value}")
        answer = options[int(answer_value) - 1]
        row_sha = sha256_bytes(canonical_json(row).encode("utf-8"))
        example_id = _example_id(
            "belebele_por_latn", BELEBELE_REVISION, "por_Latn", "test", row_id, row_sha
        )
        composite = "\n".join(
            [
                row["flores_passage"],
                row["question"],
                *[
                    f"{chr(64 + index)}. {option}"
                    for index, option in enumerate(options, 1)
                ],
            ]
        )
        registry.append(
            {
                "benchmark_name": "belebele_por_latn",
                "repository": "facebook/belebele",
                "revision": BELEBELE_REVISION,
                "configuration": "por_Latn",
                "split": "test",
                "source_configuration": "default",
                "source_split": "por_Latn",
                "source_row_id": row_id,
                "example_id": example_id,
                "row_ordinal": ordinal,
                "source_file": "data/por_Latn.jsonl",
                "source_file_sha256": source_sha,
                "row_sha256": row_sha,
                "source_category": "por_Latn",
                "context": None,
                "target_word": None,
                "passage": row["flores_passage"],
                "question": row["question"],
                "options_json": canonical_json(options),
                "answer": answer,
                "answer_source_value": answer_value,
                "answer_encoding": "one_indexed_option_number_string",
                "public_answer_available": True,
                "field_roles_json": canonical_json(
                    [
                        "passage",
                        "question",
                        "option_1",
                        "option_2",
                        "option_3",
                        "option_4",
                        "correct_answer",
                        "question_plus_answer",
                        "complete_item",
                    ]
                ),
                "license_declared": "CC-BY-SA-4.0",
                "attribution": (
                    "Meta AI, The Belebele Benchmark; cite Bandarkar et al., "
                    "ACL 2024, https://aclanthology.org/2024.acl-long.44/"
                ),
                "provenance_caveat": BELEBELE_PROVENANCE,
                "raw_row_json": canonical_json(row),
            }
        )
        common = {
            "benchmark": "belebele_por_latn",
            "example_id": example_id,
            "row_id": row_id,
            "source_sha": source_sha,
            "category": "por_Latn",
        }
        _add_match_field(
            fields, **common, role="passage", text=row["flores_passage"], matchable=True
        )
        _add_match_field(
            fields, **common, role="question", text=row["question"], matchable=True
        )
        for index, option in enumerate(options, 1):
            _add_match_field(
                fields,
                **common,
                role=f"option_{index}",
                text=option,
                matchable=False,
            )
        _add_match_field(
            fields,
            **common,
            role="correct_answer",
            text=answer,
            matchable=False,
        )
        _add_match_field(
            fields,
            **common,
            role="question_plus_answer",
            text=f"{row['question']} {answer}",
            matchable=True,
        )
        _add_match_field(
            fields,
            **common,
            role="complete_item",
            text=composite,
            matchable=True,
        )
    return registry, fields, {"test": len(rows), "distinct_passages": len(passage_ids)}


def _write_table(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd", version="2.6", use_dictionary=True)


def _json_write(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _source_files() -> tuple[tuple[str, str, str], ...]:
    return (
        (
            "NOVA-vision-language/calame-pt",
            CALAME_REVISION,
            "README.md",
        ),
        (
            "NOVA-vision-language/calame-pt",
            CALAME_REVISION,
            "calamept_all.jsonl",
        ),
        (
            "NOVA-vision-language/calame-pt",
            CALAME_REVISION,
            "calamept_handwritten_only.jsonl",
        ),
        (
            "NOVA-vision-language/calame-pt",
            CALAME_REVISION,
            "calamept_gen_only.jsonl",
        ),
        ("facebook/belebele", BELEBELE_REVISION, "README.md"),
        ("facebook/belebele", BELEBELE_REVISION, "data/por_Latn.jsonl"),
    )


def _copy_source_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.part")
    shutil.copyfile(source, temp)
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, destination)


def build_snapshot(
    output_dir: Path | str = DEFAULT_SNAPSHOT_ROOT,
    cache_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Download only the pinned benchmark files and atomically publish a snapshot."""
    output = Path(output_dir).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        return verify_snapshot(output)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".incomplete", dir=output.parent
        )
    )
    source_root = stage / "source"
    cached: dict[tuple[str, str, str], Path] = {}
    try:
        api = HfApi()
        revisions = {
            "NOVA-vision-language/calame-pt": CALAME_REVISION,
            "facebook/belebele": BELEBELE_REVISION,
        }
        repo_info = {}
        for repo_id, revision in revisions.items():
            info = api.dataset_info(repo_id, revision=revision, files_metadata=True)
            if info.sha != revision:
                raise ValueError(f"Hub resolved {repo_id}@{revision} to {info.sha}")
            repo_info[repo_id] = info
        allowed_files = {
            ("NOVA-vision-language/calame-pt", "README.md"),
            ("NOVA-vision-language/calame-pt", "calamept_all.jsonl"),
            ("NOVA-vision-language/calame-pt", "calamept_handwritten_only.jsonl"),
            ("NOVA-vision-language/calame-pt", "calamept_gen_only.jsonl"),
            ("facebook/belebele", "README.md"),
            ("facebook/belebele", "data/por_Latn.jsonl"),
        }
        for repo_id, revision, filename in _source_files():
            siblings = {item.rfilename for item in repo_info[repo_id].siblings}
            if filename not in siblings:
                raise ValueError(
                    f"Pinned snapshot lacks required file {repo_id}@{revision}/{filename}"
                )
            if (repo_id, filename) not in allowed_files:
                raise ValueError(
                    f"Unapproved benchmark source file requested: {filename}"
                )
            cache_args = {"cache_dir": str(cache_dir)} if cache_dir else {}
            cached[(repo_id, revision, filename)] = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=filename,
                    revision=revision,
                    **cache_args,
                )
            )
        for repo_id, revision, filename in _source_files():
            relative = (
                Path("source/calame_pt")
                if repo_id.startswith("NOVA-")
                else Path("source/belebele")
            ) / filename
            _copy_source_file(cached[(repo_id, revision, filename)], stage / relative)

        calame_root = source_root / "calame_pt"
        belebele_root = source_root / "belebele"
        source_checksums = {
            str(path.relative_to(stage)): sha256_file(path)
            for path in sorted(source_root.rglob("*"))
            if path.is_file()
        }
        calame_sums = {
            filename: source_checksums[f"source/calame_pt/{filename}"]
            for filename in (
                "calamept_all.jsonl",
                "calamept_handwritten_only.jsonl",
                "calamept_gen_only.jsonl",
            )
        }
        bele_sums = {
            "data/por_Latn.jsonl": source_checksums[
                "source/belebele/data/por_Latn.jsonl"
            ]
        }
        calame_registry, calame_fields, calame_counts = _calame_records(
            calame_root, calame_sums
        )
        bele_registry, bele_fields, bele_counts = _belebele_records(
            belebele_root, bele_sums
        )
        registry = calame_registry + bele_registry
        match_fields = calame_fields + bele_fields
        registry.sort(key=lambda row: (row["benchmark_name"], row["row_ordinal"]))
        match_fields.sort(key=lambda row: (row["example_id"], row["field_id"]))

        _write_table(stage / "benchmark_registry.parquet", registry, REGISTRY_SCHEMA)
        _write_table(stage / "match_fields.parquet", match_fields, MATCH_FIELD_SCHEMA)
        _json_write(stage / "source_checksums.json", {"files": source_checksums})
        artifact_checksums = {
            filename: sha256_file(stage / filename)
            for filename in (
                "benchmark_registry.parquet",
                "match_fields.parquet",
                "source_checksums.json",
            )
        }
        calame_card = repo_info["NOVA-vision-language/calame-pt"].card_data.to_dict()
        bele_card = repo_info["facebook/belebele"].card_data.to_dict()
        manifest = {
            "snapshot_version": SNAPSHOT_VERSION,
            "status": "BD2_SNAPSHOT_COMPLETE",
            "bd2_calibration": "NOT_RUN",
            "bd3_production": "NOT_RUN",
            "bd4_review_exclusions": "NOT_RUN",
            "benchmarks": [
                {
                    "canonical_name": "CALAME-PT",
                    "benchmark_name": "calame_pt",
                    "repository": "NOVA-vision-language/calame-pt",
                    "revision": CALAME_REVISION,
                    "revision_verified": repo_info[
                        "NOVA-vision-language/calame-pt"
                    ].sha,
                    "logical_configuration": "all",
                    "logical_split": "all_evaluation_only",
                    "source_configuration": "default",
                    "source_split": "train",
                    "source_files": [
                        "source/calame_pt/README.md",
                        "source/calame_pt/calamept_all.jsonl",
                        "source/calame_pt/calamept_handwritten_only.jsonl",
                        "source/calame_pt/calamept_gen_only.jsonl",
                    ],
                    "counts": calame_counts,
                    "license_declared": "MIT",
                    "license_metadata": calame_card,
                    "field_roles": ["context", "target_word", "complete_item"],
                    "public_answers": True,
                    "provenance_caveat": CALAME_PROVENANCE,
                    "metadata_mapping_note": (
                        "The pinned README maps config default to split train for "
                        "calamept_all.jsonl and split test for "
                        "calamept_handwritten_only.jsonl. Its validation path is "
                        "calamept_gen_only_.jsonl, with a trailing-underscore typo; "
                        "the pinned tree contains calamept_gen_only.jsonl instead. "
                        "The snapshot inventories the aggregate and both actual "
                        "category files, then maps every row to a logical "
                        "all_evaluation_only set. The card's train label grants no "
                        "Cambacica pretraining use."
                    ),
                    "evaluation_only": True,
                },
                {
                    "canonical_name": "Belebele Portuguese",
                    "benchmark_name": "belebele_por_latn",
                    "repository": "facebook/belebele",
                    "revision": BELEBELE_REVISION,
                    "revision_verified": repo_info["facebook/belebele"].sha,
                    "logical_configuration": "por_Latn",
                    "logical_split": "test",
                    "source_configuration": "default",
                    "source_split": "por_Latn",
                    "source_files": [
                        "source/belebele/README.md",
                        "source/belebele/data/por_Latn.jsonl",
                    ],
                    "counts": bele_counts,
                    "license_declared": "CC-BY-SA-4.0",
                    "license_metadata": bele_card,
                    "field_roles": [
                        "passage",
                        "question",
                        "options",
                        "correct_answer",
                        "question_plus_answer",
                        "complete_item",
                    ],
                    "public_answers": True,
                    "provenance_caveat": BELEBELE_PROVENANCE,
                    "metadata_mapping_note": (
                        "At this pinned Hub revision, the auto-generated dataset "
                        "card uses config default and names the Portuguese shard "
                        "split por_Latn. This single Portuguese shard has the "
                        "benchmark's 900-question test content; the registry records "
                        "the logical variant/split as por_Latn/test and preserves "
                        "the physical Hub names separately."
                    ),
                    "evaluation_only": True,
                },
            ],
            "outputs": {
                filename: {"sha256": digest, "bytes": (stage / filename).stat().st_size}
                for filename, digest in artifact_checksums.items()
            },
            "row_counts": {
                "benchmark_registry.parquet": len(registry),
                "match_fields.parquet": len(match_fields),
            },
        }
        _json_write(stage / "benchmark_manifest.json", manifest)
        verify_snapshot(stage)
        os.replace(stage, output)
        return verify_snapshot(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_snapshot(snapshot_dir: Path | str) -> dict[str, Any]:
    """Verify all source files, row inventories, and snapshot checksums."""
    root = Path(snapshot_dir)
    manifest_path = root / "benchmark_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing benchmark snapshot manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("snapshot_version") != SNAPSHOT_VERSION:
        raise ValueError("Unsupported benchmark snapshot version")
    if manifest.get("status") != "BD2_SNAPSHOT_COMPLETE":
        raise ValueError("Benchmark snapshot is not marked complete")
    checksums_path = root / "source_checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))["files"]
    for relative, expected in checksums.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Snapshot source checksum mismatch: {relative}")
    artifact_names = tuple(manifest["outputs"])
    for relative in artifact_names:
        path = root / relative
        item = manifest["outputs"][relative]
        if not path.is_file():
            raise ValueError(f"Snapshot artifact is missing: {relative}")
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Snapshot artifact checksum mismatch: {relative}")
    registry_path = root / "benchmark_registry.parquet"
    fields_path = root / "match_fields.parquet"
    registry = pq.ParquetFile(registry_path)
    match_fields = pq.ParquetFile(fields_path)
    if not registry.schema_arrow.equals(REGISTRY_SCHEMA):
        raise ValueError("Benchmark registry Parquet schema mismatch")
    if not match_fields.schema_arrow.equals(MATCH_FIELD_SCHEMA):
        raise ValueError("Match field Parquet schema mismatch")
    expected_counts = manifest["row_counts"]
    if registry.metadata.num_rows != expected_counts["benchmark_registry.parquet"]:
        raise ValueError("Benchmark registry row count mismatch")
    if match_fields.metadata.num_rows != expected_counts["match_fields.parquet"]:
        raise ValueError("Match field row count mismatch")
    example_ids: set[str] = set()
    counts: Counter[str] = Counter()
    source_sha_values = set(checksums.values())
    for batch in registry.iter_batches(batch_size=128):
        for row in batch.to_pylist():
            example_id = row["example_id"]
            if example_id in example_ids:
                raise ValueError("Benchmark example IDs are not unique")
            example_ids.add(example_id)
            counts[row["benchmark_name"]] += 1
            raw = json.loads(row["raw_row_json"])
            row_sha = sha256_bytes(canonical_json(raw).encode("utf-8"))
            if row_sha != row["row_sha256"]:
                raise ValueError(f"Canonical row checksum mismatch: {example_id}")
            expected_id = _example_id(
                row["benchmark_name"],
                row["revision"],
                row["configuration"],
                "all" if row["benchmark_name"] == "calame_pt" else row["split"],
                row["source_row_id"],
                row_sha,
            )
            if example_id != expected_id:
                raise ValueError(f"Stable benchmark example ID mismatch: {example_id}")
            if row["source_file_sha256"] not in source_sha_values:
                raise ValueError(f"Unknown source file checksum: {example_id}")
    for batch in match_fields.iter_batches(batch_size=128):
        for row in batch.to_pylist():
            if row["example_id"] not in example_ids:
                raise ValueError(f"Orphan match field: {row['field_id']}")
            token_items = json.loads(row["normalized_tokens_json"])
            if len(token_items) != row["token_count"]:
                raise ValueError(f"Match token count mismatch: {row['field_id']}")
            if any(
                item["start"] < 0
                or item["end"] > len(row["original_text"])
                or item["start"] >= item["end"]
                for item in token_items
            ):
                raise ValueError(
                    f"Match token offsets are out of bounds: {row['field_id']}"
                )
            normalized_tokens = [
                item.text for item in tokenize_with_offsets(row["original_text"])
            ]
            stored_tokens = [item["token"] for item in token_items]
            if normalized_tokens != stored_tokens:
                raise ValueError(f"Match normalization mismatch: {row['field_id']}")
            if row["normalized_text"] != normalize_match_text(row["original_text"]):
                raise ValueError(f"Match normalized text mismatch: {row['field_id']}")
    if counts != Counter({"calame_pt": 2076, "belebele_por_latn": 900}):
        raise ValueError(
            f"Unexpected pinned benchmark inventory counts: {dict(counts)}"
        )
    return {
        "status": manifest["status"],
        "benchmark_rows": registry.metadata.num_rows,
        "match_fields": match_fields.metadata.num_rows,
        "source_files": len(checksums),
        "manifest_sha256": sha256_file(manifest_path),
    }
