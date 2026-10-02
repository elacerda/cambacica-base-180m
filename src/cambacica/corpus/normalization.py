"""Frozen, restartable raw-source normalization for Gate C1.

This module intentionally stops before deduplication, decontamination, corpus
mixture construction, train/validation splitting, and tokenization.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
from typing import Any, Iterator, Mapping
import unicodedata
from urllib.parse import quote
import xml.etree.ElementTree as ET

import pyarrow as pa
import pyarrow.parquet as pq

from cambacica.corpus.manifest import compute_file_sha256
from cambacica.corpus.schema import compute_content_sha256, normalize_text


NORMALIZATION_VERSION = "1.0.0"
DEFAULT_SHARD_TEXT_BYTES = 256 * 1024 * 1024
DEFAULT_RAW_ROOT = Path("/mnt/data/cambacica-base-180m/raw")
DEFAULT_NORMALIZED_ROOT = Path("/mnt/data/cambacica-base-180m/normalized")
GIGA_PROVENANCE_FIELDS = (
    "_gv2_upstream_shard",
    "_gv2_upstream_row_group",
    "_gv2_upstream_commit",
)

SOURCE_CONFIG: dict[str, dict[str, str]] = {
    "gutenberg_pt": {
        "raw_dir": "gutenberg",
        "output_dir": "gutenberg",
        "manifest_source": "gutenberg_pt",
    },
    "parlamento_pt": {
        "raw_dir": "parlamento",
        "output_dir": "parlamento",
        "manifest_source": "parlamento_pt",
    },
    "wikipedia_pt": {
        "raw_dir": "wikipedia",
        "output_dir": "wikipedia",
        "manifest_source": "wikipedia_pt",
    },
    "carolina": {
        "raw_dir": "carolina",
        "output_dir": "carolina",
        "manifest_source": "carolina",
    },
    "gigaverbo_v2": {
        "raw_dir": "gigaverbo",
        "output_dir": "gigaverbo",
        "manifest_source": "gigaverbo_v2",
    },
}
FROZEN_RAW_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "gutenberg_pt": {
        "pinned_revision": "snapshot_2026-10-01",
        "total_files": 655,
    },
    "parlamento_pt": {
        "pinned_commit_sha": "08f13e7e63ab9bfbd8c0b40955defe3bb7f68c2b",
    },
    "wikipedia_pt": {
        "snapshot_identifier": "20231101.pt",
        "pinned_commit_sha": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    },
    "carolina": {
        "pinned_commit_sha": "55e63a519393c70a48dcfa14a558499c6bb0583b",
    },
    "gigaverbo_v2": {
        "pinned_commit_sha": "7058ccf19eaeaf4505a96fc7e5305a01fc441fd8",
        "eligible_records": 15_756_679,
    },
}


NORMALIZED_SCHEMA = pa.schema(
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

_GUTENBERG_START = re.compile(
    r"\*{3}\s*START OF TH(?:IS|E) PROJECT GUTENBERG EBOOK[^*]*\*{3}",
    re.IGNORECASE,
)
_GUTENBERG_END = re.compile(
    r"\*{3}\s*END OF TH(?:IS|E) PROJECT GUTENBERG EBOOK",
    re.IGNORECASE,
)
_XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
_TEI_BODY_BLOCKS = {
    "ab",
    "argument",
    "byline",
    "cell",
    "closer",
    "dateline",
    "head",
    "item",
    "l",
    "lg",
    "note",
    "opener",
    "p",
    "quote",
    "salute",
    "signed",
    "sp",
    "speaker",
    "stage",
    "trailer",
}


@dataclass(frozen=True)
class RawRecord:
    """One logical record at its raw source boundary."""

    partition_key: str
    order_key: tuple[int, int, int]
    cursor: dict[str, Any]
    fields: dict[str, Any]


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def canonicalize_newlines(text: str) -> str:
    """Map CRLF and bare CR to LF without changing other characters."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_document_text(text: str) -> str:
    """Apply the frozen text rule shared with accepted C1 sample metrics."""
    return normalize_text(canonicalize_newlines(text))


def count_normalized_words(text: str) -> int:
    """Count words using the canonical C1 whitespace-splitting definition."""
    return len(text.split())


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove only standard Gutenberg wrapper sections when markers are valid."""
    start_match = _GUTENBERG_START.search(text)
    end_match = (
        _GUTENBERG_END.search(text, start_match.end())
        if start_match
        else _GUTENBERG_END.search(text)
    )
    if not start_match or not end_match or start_match.start() >= end_match.start():
        return text
    return text[start_match.end() : end_match.start()]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _element_text(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    value = "".join(element.itertext()).strip()
    return value or None


def extract_carolina_tei(
    root: ET.Element, taxonomy: str, fallback_id: str | None = None
) -> dict[str, Any]:
    """Extract body text and the small set of useful TEI header fields."""
    body = next((node for node in root.iter() if _local_name(node.tag) == "body"), None)
    blocks: list[str] = []
    inline_parts: list[str] = []

    def flush_inline() -> None:
        value = "".join(inline_parts).strip()
        if value:
            blocks.append(value)
        inline_parts.clear()

    def collect_text(node: ET.Element) -> None:
        if node.text:
            inline_parts.append(node.text)
        for child in list(node):
            if _local_name(child.tag) in _TEI_BODY_BLOCKS:
                flush_inline()
                value = _element_text(child)
                if value:
                    blocks.append(value)
            else:
                collect_text(child)
            if child.tail:
                inline_parts.append(child.tail)

    if body is not None:
        collect_text(body)
        flush_inline()
    text = "\n\n".join(blocks)

    header = next(
        (node for node in root.iter() if _local_name(node.tag) == "teiHeader"), None
    )
    title = None
    publication_date = None
    license_value = None
    original_url = None
    upstream_metadata: dict[str, Any] = {}
    tei_identifier = root.attrib.get(_XML_ID) or root.attrib.get("id")
    if header is not None:
        title_statement = next(
            (node for node in header.iter() if _local_name(node.tag) == "titleStmt"),
            None,
        )
        title_scope = title_statement if title_statement is not None else header
        title = next(
            (
                _element_text(node)
                for node in title_scope.iter()
                if _local_name(node.tag) == "title" and _element_text(node)
            ),
            None,
        )
        for role in ("author", "editor", "respStmt"):
            values = [
                value
                for node in header.iter()
                if _local_name(node.tag) == role
                if (value := _element_text(node))
            ]
            if values:
                upstream_metadata[role] = values
        publishers = [
            value
            for node in header.iter()
            if _local_name(node.tag) == "publisher"
            if (value := _element_text(node))
        ]
        if publishers:
            upstream_metadata["publisher"] = publishers
        identifiers = [
            {
                "type": node.attrib.get("type"),
                "value": _element_text(node),
            }
            for node in header.iter()
            if _local_name(node.tag) == "idno" and _element_text(node)
        ]
        if identifiers:
            upstream_metadata["identifiers"] = identifiers
        date_nodes = [node for node in header.iter() if _local_name(node.tag) == "date"]
        publication_date = next(
            (_element_text(node) for node in date_nodes if _element_text(node)), None
        )
        license_nodes = [
            node
            for node in header.iter()
            if _local_name(node.tag) in {"licence", "license"}
        ]
        license_value = next(
            (_element_text(node) for node in license_nodes if _element_text(node)),
            None,
        )
        for node in header.iter():
            if _local_name(node.tag) == "ref" and node.attrib.get("target"):
                original_url = node.attrib["target"].strip() or None
                break
    return {
        "text": text,
        "source": "carolina",
        "source_revision": None,
        "subset": taxonomy,
        "original_id": tei_identifier or fallback_id,
        "original_url": original_url,
        "license": license_value,
        "language": "pt-BR",
        "language_score": None,
        "variety": "pt-BR",
        "quality_score": None,
        "publication_date": publication_date,
        "domain_category": taxonomy,
        "title": title,
        "upstream_metadata_json": json.dumps(
            upstream_metadata,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if upstream_metadata
        else None,
    }


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe relative source path: {value!r}")
    return path.as_posix()


def _load_and_verify_raw_manifest(
    source: str, raw_source_root: Path, verify_raw: bool
) -> tuple[dict[str, Any], str, Path]:
    from cambacica.corpus.materialize import MaterializationManifest

    manifest_path = raw_source_root / "manifest.json"
    manifest_sha = compute_file_sha256(manifest_path)
    raw_manifest = MaterializationManifest.load(manifest_path).to_dict()
    if raw_manifest.get("status") != "COMPLETE":
        raise ValueError(
            f"Raw manifest must be COMPLETE, found {raw_manifest.get('status')!r}."
        )
    if raw_manifest.get("source") != SOURCE_CONFIG[source]["manifest_source"]:
        raise ValueError(
            f"Raw manifest source {raw_manifest.get('source')!r} does not match {source!r}."
        )
    if verify_raw:
        expected = FROZEN_RAW_EXPECTATIONS[source]
        source_metadata = raw_manifest.get("source_metadata") or {}
        mismatches = []
        for field, expected_value in expected.items():
            actual = (
                source_metadata.get(field)
                if source == "gigaverbo_v2" and field == "eligible_records"
                else raw_manifest.get(field)
            )
            if actual != expected_value:
                mismatches.append(f"{field}={actual!r} (expected {expected_value!r})")
        if mismatches:
            raise ValueError(
                "Raw manifest does not match the frozen C1 snapshot: "
                + "; ".join(mismatches)
            )
        manifest_obj = MaterializationManifest.load(manifest_path)
        valid, errors = manifest_obj.verify(
            raw_source_root,
            check_partial_files=False,
            check_runtime_state=False,
            check_ids_metadata=True,
        )
        if not valid:
            raise ValueError(
                "Raw materialization verification failed: " + "; ".join(errors[:8])
            )
    return raw_manifest, manifest_sha, manifest_path


def _source_entries(raw_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for raw_entry in raw_manifest.get("files", []):
        rel_path = _safe_relative_path(str(raw_entry.get("relative_path", "")))
        if rel_path.lower().endswith((".txt", ".xml.gz", ".parquet")):
            entries.append({**raw_entry, "relative_path": rel_path})
    return sorted(entries, key=lambda entry: entry["relative_path"])


def _raw_source_revision(raw_manifest: Mapping[str, Any]) -> str:
    return str(
        raw_manifest.get("pinned_commit_sha")
        or raw_manifest.get("snapshot_identifier")
        or raw_manifest.get("pinned_revision")
        or "unknown"
    )


def _is_after_cursor(record: RawRecord, cursors: Mapping[str, Any]) -> bool:
    cursor = cursors.get(record.partition_key)
    if not cursor:
        return True
    return tuple(record.order_key) > tuple(cursor["order_key"])


def _cursor_for_record(
    relative_path: str,
    raw_record_identifier: str,
    file_index: int,
    row_group: int = 0,
    row_index: int = 0,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "raw_source_file": relative_path,
        "raw_record_identifier": raw_record_identifier,
        "order_key": [file_index, row_group, row_index],
        **extra,
    }


def _should_skip_file(
    file_index: int, partition_key: str, cursors: Mapping[str, Any]
) -> bool:
    cursor = cursors.get(partition_key)
    return bool(cursor and file_index < int(cursor["order_key"][0]))


def _iter_gutenberg(
    raw_root: Path,
    entries: list[dict[str, Any]],
    revision: str,
    cursors: Mapping[str, Any],
) -> Iterator[RawRecord]:
    partition = "source"
    for file_index, entry in enumerate(entries):
        if _should_skip_file(file_index, partition, cursors):
            continue
        relative_path = entry["relative_path"]
        path = raw_root / relative_path
        raw_bytes = path.read_bytes()
        raw_text = raw_bytes.decode("utf-8-sig", errors="strict")
        body = strip_gutenberg_boilerplate(raw_text)
        match = re.fullmatch(r"pg(\d+)\.txt", Path(relative_path).name)
        ebook_id = (
            match.group(1) if match else str(entry.get("upstream_identifier", ""))
        )
        cursor = _cursor_for_record(relative_path, ebook_id, file_index)
        record = RawRecord(
            partition_key=partition,
            order_key=(file_index, 0, 0),
            cursor=cursor,
            fields={
                "text": body,
                "source": "gutenberg_pt",
                "source_revision": revision,
                "subset": "literature",
                "original_id": ebook_id,
                "original_url": f"https://www.gutenberg.org/ebooks/{ebook_id}",
                "license": "Project Gutenberg License / US Public Domain (jurisdiction-dependent)",
                "language": "pt",
                "language_score": None,
                "variety": None,
                "quality_score": None,
                "publication_date": None,
                "domain_category": "literature",
                "title": None,
            },
        )
        if _is_after_cursor(record, cursors):
            yield record


def _iter_parlamento(
    raw_root: Path,
    entries: list[dict[str, Any]],
    revision: str,
    cursors: Mapping[str, Any],
    failures: list[dict[str, Any]],
) -> Iterator[RawRecord]:
    partition = "source"
    for file_index, entry in enumerate(entries):
        if _should_skip_file(file_index, partition, cursors):
            continue
        relative_path = entry["relative_path"]
        path = raw_root / relative_path
        cursor = cursors.get(partition)
        start_line = 0
        start_offset = 0
        if cursor and cursor["raw_source_file"] == relative_path:
            start_line = int(cursor["raw_record_identifier"])
            start_offset = int(cursor.get("byte_offset_end", 0))
        with path.open("rb") as stream:
            if start_offset:
                stream.seek(start_offset)
            line_number = start_line
            while True:
                line_start = stream.tell()
                raw_line = stream.readline()
                if not raw_line:
                    break
                line_number += 1
                line_end = stream.tell()
                if raw_line.endswith(b"\n"):
                    payload = raw_line[:-1]
                    if payload.endswith(b"\r"):
                        payload = payload[:-1]
                else:
                    payload = raw_line
                try:
                    text = payload.decode(
                        "utf-8-sig" if line_number == 1 else "utf-8", errors="strict"
                    )
                except UnicodeDecodeError as exc:
                    failures.append(
                        {
                            "kind": "record",
                            "raw_source_file": relative_path,
                            "raw_record_identifier": str(line_number),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                doc_id = f"parl_pt_{line_number}"
                record = RawRecord(
                    partition_key=partition,
                    order_key=(file_index, 0, line_number),
                    cursor=_cursor_for_record(
                        relative_path,
                        str(line_number),
                        file_index,
                        row_index=line_number,
                        byte_offset_start=line_start,
                        byte_offset_end=line_end,
                    ),
                    fields={
                        "text": text,
                        "source": "parlamento_pt",
                        "source_revision": revision,
                        "subset": "debates",
                        "original_id": doc_id,
                        "original_url": "https://www.parlamento.pt/Cidadania/Paginas/DadosAbertos.aspx",
                        "license": "Open Government Data (Portuguese Parliament)",
                        "language": "pt-PT",
                        "language_score": None,
                        "variety": "pt-PT",
                        "quality_score": None,
                        "publication_date": None,
                        "domain_category": "parliamentary_records",
                        "title": None,
                    },
                )
                if _is_after_cursor(record, cursors):
                    yield record


def _iter_wikipedia(
    raw_root: Path,
    entries: list[dict[str, Any]],
    revision: str,
    cursors: Mapping[str, Any],
    failures: list[dict[str, Any]],
) -> Iterator[RawRecord]:
    partition = "source"
    for file_index, entry in enumerate(entries):
        if _should_skip_file(file_index, partition, cursors):
            continue
        relative_path = entry["relative_path"]
        parquet_file = pq.ParquetFile(raw_root / relative_path)
        cursor = cursors.get(partition)
        for row_group in range(parquet_file.num_row_groups):
            if cursor and file_index == int(cursor["order_key"][0]):
                if row_group < int(cursor["order_key"][1]):
                    continue
            row_index = 0
            for batch in parquet_file.iter_batches(
                batch_size=2048,
                row_groups=[row_group],
                columns=["id", "url", "title", "text"],
            ):
                for row in batch.to_pylist():
                    page_id = row.get("id")
                    raw_id = (
                        str(page_id)
                        if page_id is not None
                        else f"row-{file_index}-{row_group}-{row_index}"
                    )
                    title = row.get("title") or ""
                    text = row.get("text") or ""
                    original_url = row.get("url")
                    if (
                        not isinstance(text, str)
                        or not isinstance(title, str)
                        or (
                            original_url is not None
                            and not isinstance(original_url, str)
                        )
                    ):
                        failures.append(
                            {
                                "kind": "record",
                                "raw_source_file": relative_path,
                                "raw_record_identifier": raw_id,
                                "error": "Wikipedia id/url/title/text fields have invalid types.",
                            }
                        )
                        row_index += 1
                        continue
                    combined = (
                        f"{title}\n\n{text}"
                        if title and not text.startswith(title)
                        else text
                    )
                    record = RawRecord(
                        partition_key=partition,
                        order_key=(file_index, row_group, row_index),
                        cursor=_cursor_for_record(
                            relative_path,
                            raw_id,
                            file_index,
                            row_group=row_group,
                            row_index=row_index,
                        ),
                        fields={
                            "text": combined,
                            "source": "wikipedia_pt",
                            "source_revision": revision,
                            "subset": "articles",
                            "original_id": raw_id,
                            "original_url": original_url,
                            "license": "CC BY-SA 3.0/4.0 & GFDL",
                            "language": "pt",
                            "language_score": None,
                            "variety": None,
                            "quality_score": None,
                            "publication_date": None,
                            "domain_category": "encyclopedic",
                            "title": title or None,
                        },
                    )
                    if _is_after_cursor(record, cursors):
                        yield record
                    row_index += 1


def _carolina_taxonomy(relative_path: str) -> str:
    path = PurePosixPath(relative_path)
    components = path.parts
    if len(components) < 3 or not relative_path.endswith(".xml.gz"):
        raise ValueError(f"Cannot determine Carolina taxonomy from {relative_path!r}.")
    taxonomy_map = {
        "datasets_and_other_corpora": "dat",
        "judicial_branch": "jud",
        "legislative_branch": "leg",
        "public_domain_works": "pub",
        "social_media": "soc",
        "university_domains": "uni",
        "wikis": "wik",
    }
    for component in components:
        if component in taxonomy_map:
            return taxonomy_map[component]
    raise ValueError(f"Unrecognized Carolina taxonomy path: {relative_path!r}.")


def _iter_carolina(
    raw_root: Path,
    entries: list[dict[str, Any]],
    revision: str,
    cursors: Mapping[str, Any],
) -> Iterator[RawRecord]:
    partition = "source"
    for file_index, entry in enumerate(entries):
        if _should_skip_file(file_index, partition, cursors):
            continue
        relative_path = entry["relative_path"]
        taxonomy = _carolina_taxonomy(relative_path)
        cursor = cursors.get(partition)
        start_ordinal = (
            int(cursor["raw_record_identifier"].split("-")[-1])
            if cursor and cursor["raw_source_file"] == relative_path
            else 0
        )
        ordinal = 0
        with gzip.open(raw_root / relative_path, "rb") as compressed:
            for _, element in ET.iterparse(compressed, events=("end",)):
                if _local_name(element.tag) != "TEI":
                    continue
                ordinal += 1
                if ordinal <= start_ordinal:
                    element.clear()
                    continue
                record_id = f"tei-{ordinal:09d}"
                fields = extract_carolina_tei(
                    element,
                    taxonomy,
                    fallback_id=f"{relative_path}#{record_id}",
                )
                fields["source_revision"] = revision
                record = RawRecord(
                    partition_key=partition,
                    order_key=(file_index, 0, ordinal),
                    cursor=_cursor_for_record(
                        relative_path,
                        record_id,
                        file_index,
                        row_index=ordinal,
                    ),
                    fields=fields,
                )
                element.clear()
                if _is_after_cursor(record, cursors):
                    yield record


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _iter_gigaverbo(
    raw_root: Path,
    entries: list[dict[str, Any]],
    revision: str,
    cursors: Mapping[str, Any],
    failures: list[dict[str, Any]],
) -> Iterator[RawRecord]:
    selected_columns: list[str] | None = None
    for file_index, entry in enumerate(entries):
        relative_path = entry["relative_path"]
        expected_subset = entry.get("upstream_subset")
        if expected_subset is None:
            path_match = re.match(r"subset=([^/]+)/", relative_path)
            expected_subset = path_match.group(1) if path_match else None
        if not expected_subset:
            raise ValueError(
                f"GigaVerbo raw file lacks subset provenance: {relative_path}"
            )
        partition = f"subset:{expected_subset}"
        if _should_skip_file(file_index, partition, cursors):
            continue
        parquet_file = pq.ParquetFile(raw_root / relative_path)
        columns = parquet_file.schema_arrow.names
        if selected_columns is None:
            selected_columns = list(columns)
            required = {"text", "id", "source", "subset", *GIGA_PROVENANCE_FIELDS}
            missing = required - set(columns)
            if missing:
                raise ValueError(
                    f"GigaVerbo raw schema lacks required fields: {sorted(missing)}"
                )
        cursor = cursors.get(partition)
        for row_group in range(parquet_file.num_row_groups):
            if cursor and file_index == int(cursor["order_key"][0]):
                if row_group < int(cursor["order_key"][1]):
                    continue
            row_index = 0
            for batch in parquet_file.iter_batches(
                batch_size=2048,
                row_groups=[row_group],
                columns=selected_columns,
            ):
                for row in batch.to_pylist():
                    subset = row.get("subset")
                    if subset is None:
                        raise ValueError("GigaVerbo row has no subset value.")
                    subset = str(subset)
                    if subset != str(expected_subset):
                        raise ValueError(
                            f"GigaVerbo file subset {expected_subset!r} contains row subset {subset!r}."
                        )
                    raw_id_value = row.get("id")
                    raw_id = (
                        str(raw_id_value)
                        if raw_id_value is not None
                        else f"row-{file_index}-{row_group}-{row_index}"
                    )
                    text = row.get("text")
                    if text is None:
                        text = ""
                    if not isinstance(text, str):
                        failures.append(
                            {
                                "kind": "record",
                                "raw_source_file": relative_path,
                                "raw_record_identifier": raw_id,
                                "error": "GigaVerbo text field is not a string.",
                            }
                        )
                        row_index += 1
                        continue
                    source_url = row.get("source")
                    metadata = {
                        key: _json_safe(value)
                        for key, value in row.items()
                        if key
                        not in {
                            "text",
                            "id",
                            "source",
                            "subset",
                            *GIGA_PROVENANCE_FIELDS,
                        }
                    }
                    metadata_json = json.dumps(
                        metadata,
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    record = RawRecord(
                        partition_key=partition,
                        order_key=(file_index, row_group, row_index),
                        cursor=_cursor_for_record(
                            relative_path,
                            raw_id,
                            file_index,
                            row_group=row_group,
                            row_index=row_index,
                        ),
                        fields={
                            "text": text,
                            "source": "gigaverbo_v2",
                            "source_revision": revision,
                            "subset": subset,
                            "original_id": raw_id,
                            "original_url": source_url,
                            "license": None,
                            "language": "pt",
                            "language_score": None,
                            "variety": None,
                            "quality_score": _safe_float(row.get("edu_score")),
                            "publication_date": None,
                            "domain_category": subset,
                            "title": None,
                            "_gv2_upstream_shard": row.get("_gv2_upstream_shard"),
                            "_gv2_upstream_row_group": row.get(
                                "_gv2_upstream_row_group"
                            ),
                            "_gv2_upstream_commit": row.get("_gv2_upstream_commit"),
                            "upstream_metadata_json": metadata_json or None,
                        },
                    )
                    if _is_after_cursor(record, cursors):
                        yield record
                    row_index += 1


def _normalize_record(record: RawRecord, source: str) -> dict[str, Any]:
    fields = record.fields
    text_value = fields.get("text")
    if text_value is None:
        text_value = ""
    if not isinstance(text_value, str):
        raise TypeError(f"{source} record text must be a string or null.")
    text = normalize_document_text(text_value)
    normalized = {field.name: None for field in NORMALIZED_SCHEMA}
    normalized.update(fields)
    normalized.update(
        {
            "text": text,
            "source": source,
            "content_sha256": compute_content_sha256(text),
            "raw_source_file": record.cursor["raw_source_file"],
            "raw_record_identifier": record.cursor["raw_record_identifier"],
            "normalization_version": NORMALIZATION_VERSION,
            "_gv2_upstream_shard": fields.get("_gv2_upstream_shard"),
            "_gv2_upstream_row_group": fields.get("_gv2_upstream_row_group"),
            "_gv2_upstream_commit": fields.get("_gv2_upstream_commit"),
            "upstream_metadata_json": fields.get("upstream_metadata_json"),
        }
    )
    return normalized


def _source_expected_records(
    source: str,
    raw_root: Path,
    entries: list[dict[str, Any]],
    raw_manifest: Mapping[str, Any],
) -> int | None:
    if source == "gutenberg_pt":
        return len(entries)
    if source == "parlamento_pt":
        line_count = raw_manifest.get("line_count")
        if line_count is None or not entries:
            return None
        file_path = raw_root / entries[0]["relative_path"]
        if file_path.stat().st_size == 0:
            return 0
        with file_path.open("rb") as stream:
            stream.seek(-1, os.SEEK_END)
            ends_newline = stream.read(1) == b"\n"
        return int(line_count) if ends_newline else int(line_count) + 1
    if source == "wikipedia_pt":
        return sum(
            pq.ParquetFile(raw_root / entry["relative_path"]).metadata.num_rows
            for entry in entries
        )
    if source == "gigaverbo_v2":
        records = raw_manifest.get("source_metadata", {}).get("eligible_records")
        if records is not None:
            return int(records)
        return sum(int(entry.get("records", 0)) for entry in entries)
    return None


def _source_records(
    source: str,
    raw_root: Path,
    raw_manifest: Mapping[str, Any],
    entries: list[dict[str, Any]],
    cursors: Mapping[str, Any],
    failures: list[dict[str, Any]],
) -> Iterator[RawRecord]:
    revision = _raw_source_revision(raw_manifest)
    for entry_index, entry in enumerate(entries):
        relative_path = entry["relative_path"]
        if source == "gigaverbo_v2":
            entry_subset = entry.get("upstream_subset")
            if entry_subset is None:
                path_match = re.match(r"subset=([^/]+)/", relative_path)
                entry_subset = path_match.group(1) if path_match else None
            if not entry_subset:
                failures.append(
                    {
                        "kind": "source_file",
                        "raw_source_file": relative_path,
                        "error": "GigaVerbo raw file lacks subset provenance.",
                    }
                )
                continue
            partition = f"subset:{entry_subset}"
        else:
            partition = "source"
        if _should_skip_file(entry_index, partition, cursors):
            continue
        local_cursors: dict[str, Any] = {}
        cursor = cursors.get(partition)
        if cursor and int(cursor["order_key"][0]) == entry_index:
            local_cursor = dict(cursor)
            local_cursor["order_key"] = [0, *cursor["order_key"][1:]]
            local_cursors[partition] = local_cursor
        try:
            if source == "gutenberg_pt":
                iterator = _iter_gutenberg(raw_root, [entry], revision, local_cursors)
                for record in iterator:
                    key = (entry_index, 0, 0)
                    yield RawRecord(
                        record.partition_key,
                        key,
                        _cursor_for_record(
                            relative_path,
                            record.cursor["raw_record_identifier"],
                            entry_index,
                        ),
                        record.fields,
                    )
            elif source == "parlamento_pt":
                # The pinned materialization contains one newline-delimited train.txt.
                if entry_index != 0:
                    failures.append(
                        {
                            "kind": "source_file",
                            "raw_source_file": relative_path,
                            "error": "Unexpected extra ParlamentoPT text file.",
                        }
                    )
                    continue
                yield from _iter_parlamento(
                    raw_root, [entry], revision, local_cursors, failures
                )
            elif source == "wikipedia_pt":
                iterator = _iter_wikipedia(
                    raw_root, [entry], revision, local_cursors, failures
                )
                for record in iterator:
                    key = (entry_index, *record.order_key[1:])
                    yield RawRecord(
                        record.partition_key,
                        key,
                        _cursor_for_record(
                            relative_path,
                            record.cursor["raw_record_identifier"],
                            entry_index,
                            row_group=key[1],
                            row_index=key[2],
                        ),
                        record.fields,
                    )
            elif source == "carolina":
                iterator = _iter_carolina(raw_root, [entry], revision, local_cursors)
                for record in iterator:
                    key = (entry_index, 0, record.order_key[2])
                    yield RawRecord(
                        record.partition_key,
                        key,
                        _cursor_for_record(
                            relative_path,
                            record.cursor["raw_record_identifier"],
                            entry_index,
                            row_index=key[2],
                        ),
                        record.fields,
                    )
            else:
                iterator = _iter_gigaverbo(
                    raw_root, [entry], revision, local_cursors, failures
                )
                for record in iterator:
                    key = (entry_index, *record.order_key[1:])
                    yield RawRecord(
                        record.partition_key,
                        key,
                        _cursor_for_record(
                            relative_path,
                            record.cursor["raw_record_identifier"],
                            entry_index,
                            row_group=key[1],
                            row_index=key[2],
                        ),
                        record.fields,
                    )
        except Exception as exc:
            failures.append(
                {
                    "kind": "source_file",
                    "raw_source_file": relative_path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


def _atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_write_parquet(table: pa.Table, destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.partial")
    temporary.unlink(missing_ok=True)
    pq.write_table(
        table,
        temporary,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
        row_group_size=65_536,
    )
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return destination.stat().st_size, compute_file_sha256(destination)


def _get_tool_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _empty_manifest(
    source: str,
    raw_manifest_path: Path,
    raw_manifest_sha256: str | None,
    tool_commit: str | None,
    shard_text_bytes: int,
) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": 1,
        "normalization_schema_version": NORMALIZATION_VERSION,
        "runtime_versions": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "unicode_database": unicodedata.unidata_version,
            "pyarrow": pa.__version__,
        },
        "source": source,
        "status": "PARTIAL",
        "raw_source_manifest": str(raw_manifest_path),
        "raw_source_manifest_sha256": raw_manifest_sha256,
        "normalization_tool_git_commit": tool_commit,
        "normalized_schema_fields": NORMALIZED_SCHEMA.names,
        "shard_target_normalized_text_bytes": shard_text_bytes,
        "started_at_utc": now,
        "completed_at_utc": None,
        "source_document_count": None,
        "output_document_count": 0,
        "normalization_failure_count": 0,
        "failures": [],
        "normalized_files": [],
        "cursors": {},
        "total_normalized_bytes": 0,
        "total_normalized_characters": 0,
        "total_normalized_words": 0,
    }


def _manifest_totals(manifest: Mapping[str, Any]) -> dict[str, int]:
    fields = {
        "output_document_count": "documents",
        "total_normalized_bytes": "normalized_bytes",
        "total_normalized_characters": "normalized_characters",
        "total_normalized_words": "normalized_words",
    }
    return {
        key: sum(
            int(file_record.get(field, 0))
            for file_record in manifest["normalized_files"]
        )
        for key, field in fields.items()
    }


def _normalized_row_errors(row: Mapping[str, Any], source: str) -> list[str]:
    errors = []
    text = row.get("text")
    if not isinstance(text, str):
        return ["normalized text is not a string"]
    if row.get("content_sha256") != compute_content_sha256(text):
        errors.append("content_sha256 does not match normalized text")
    if row.get("normalization_version") != NORMALIZATION_VERSION:
        errors.append("normalization_version does not match the frozen version")
    if row.get("source") != source:
        errors.append("source does not match the source manifest")
    for field in (
        "source_revision",
        "subset",
        "original_id",
        "raw_source_file",
        "raw_record_identifier",
    ):
        value = row.get(field)
        if value is None or value == "":
            errors.append(f"critical provenance field {field} is missing")
    if source == "gigaverbo_v2":
        for field in GIGA_PROVENANCE_FIELDS:
            if row.get(field) is None or row.get(field) == "":
                errors.append(f"GigaVerbo provenance field {field} is missing")
    return errors


def _verify_normalized_manifest(
    output_dir: Path,
    manifest: Mapping[str, Any],
    raw_manifest_path: Path | None = None,
    *,
    hash_files: bool = True,
    allow_partial: bool = False,
    allow_in_progress: bool = False,
    allow_unlisted_shards: bool = False,
    verify_rows: bool = True,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if manifest.get("status") != "COMPLETE" and not allow_partial:
        errors.append(
            f"Normalized manifest status is {manifest.get('status')!r}, expected COMPLETE."
        )
    if manifest.get("normalization_schema_version") != NORMALIZATION_VERSION:
        errors.append("Normalization version mismatch.")
    runtime_versions = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "unicode_database": unicodedata.unidata_version,
        "pyarrow": pa.__version__,
    }
    if manifest.get("runtime_versions") != runtime_versions:
        errors.append("Normalization runtime versions differ from the manifest.")
    if (
        manifest.get("status") == "COMPLETE"
        and not allow_in_progress
        and (output_dir / "manifest.in_progress.json").exists()
    ):
        errors.append("A COMPLETE normalized source has an in-progress manifest.")
    if raw_manifest_path is not None:
        if not raw_manifest_path.is_file():
            errors.append(f"Raw source manifest is missing: {raw_manifest_path}")
        elif compute_file_sha256(raw_manifest_path) != manifest.get(
            "raw_source_manifest_sha256"
        ):
            errors.append("Raw source manifest SHA-256 changed after normalization.")
    seen: set[str] = set()
    root = output_dir.resolve()
    for file_record in manifest.get("normalized_files", []):
        relative_path = file_record.get("relative_path")
        if not isinstance(relative_path, str) or relative_path in seen:
            errors.append(f"Invalid or duplicate normalized path: {relative_path!r}")
            continue
        seen.add(relative_path)
        path = (output_dir / relative_path).resolve()
        if root not in path.parents or not path.is_file():
            errors.append(
                f"Normalized file missing or outside source root: {relative_path}"
            )
            continue
        if path.stat().st_size != file_record.get("bytes"):
            errors.append(f"Normalized file size mismatch: {relative_path}")
        if hash_files and compute_file_sha256(path) != file_record.get("sha256"):
            errors.append(f"Normalized file SHA-256 mismatch: {relative_path}")
        try:
            parquet_file = pq.ParquetFile(path)
            if parquet_file.schema_arrow.remove_metadata() != NORMALIZED_SCHEMA:
                errors.append(f"Normalized Parquet schema mismatch: {relative_path}")
            if parquet_file.metadata.num_rows != file_record.get("documents"):
                errors.append(f"Normalized row count mismatch: {relative_path}")
            if verify_rows:
                observed = {
                    "normalized_bytes": 0,
                    "normalized_characters": 0,
                    "normalized_words": 0,
                    "documents": 0,
                }
                for batch in parquet_file.iter_batches(batch_size=2048):
                    for row in batch.to_pylist():
                        text = row["text"]
                        for error in _normalized_row_errors(
                            row, str(manifest["source"])
                        ):
                            errors.append(
                                f"{relative_path}: {row.get('raw_source_file')}:{row.get('raw_record_identifier')}: {error}"
                            )
                        if not isinstance(text, str):
                            continue
                        observed["documents"] += 1
                        observed["normalized_bytes"] += len(text.encode("utf-8"))
                        observed["normalized_characters"] += len(text)
                        observed["normalized_words"] += count_normalized_words(text)
                for field, actual in observed.items():
                    if actual != int(file_record.get(field, -1)):
                        errors.append(
                            f"Normalized per-file {field} mismatch for {relative_path}: "
                            f"{actual} != {file_record.get(field)}"
                        )
        except Exception as exc:
            errors.append(f"Cannot read normalized file {relative_path}: {exc}")
    actual_shards = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("part-*.parquet")
        if path.is_file()
    }
    if (actual_shards != seen) and not (
        allow_unlisted_shards and seen.issubset(actual_shards)
    ):
        errors.append(
            "Normalized shard inventory mismatch: "
            f"unlisted={sorted(actual_shards - seen)[:5]}, "
            f"missing={sorted(seen - actual_shards)[:5]}"
        )
    if not allow_in_progress:
        partial_files = list(output_dir.rglob("*.partial"))
        if partial_files:
            errors.append(f"Incomplete normalized payload remains: {partial_files[0]}")
    totals = _manifest_totals(manifest)
    for key, value in totals.items():
        if manifest.get(key) != value:
            errors.append(
                f"Manifest aggregate {key} mismatch: {manifest.get(key)} != {value}"
            )
    if manifest.get("normalization_failure_count") != len(manifest.get("failures", [])):
        errors.append("Manifest failure count does not match failure inventory.")
    if manifest.get("status") == "COMPLETE":
        if manifest.get("normalization_failure_count") != 0:
            errors.append("A COMPLETE normalized source has normalization failures.")
        if manifest.get("source_document_count") != manifest.get(
            "output_document_count"
        ):
            errors.append("A COMPLETE normalized source has a document-count mismatch.")
    return not errors, errors


def verify_normalized_source(
    source: str,
    *,
    output_root: Path | str = DEFAULT_NORMALIZED_ROOT,
    raw_root: Path | str = DEFAULT_RAW_ROOT,
    hash_files: bool = True,
    allow_partial: bool = False,
    verify_rows: bool = True,
) -> tuple[bool, list[str]]:
    """Verify normalized output inventory and source-manifest identity."""
    if source not in SOURCE_CONFIG:
        raise ValueError(
            f"Unknown source {source!r}; expected one of {sorted(SOURCE_CONFIG)}"
        )
    config = SOURCE_CONFIG[source]
    output_dir = Path(output_root) / config["output_dir"]
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return False, [f"Normalized source manifest is missing: {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_manifest_path = Path(raw_root) / config["raw_dir"] / "manifest.json"
    return _verify_normalized_manifest(
        output_dir,
        manifest,
        raw_manifest_path,
        hash_files=hash_files,
        allow_partial=allow_partial,
        verify_rows=verify_rows,
    )


def _prepare_resume(
    output_dir: Path,
    source: str,
    raw_manifest_sha: str,
    tool_commit: str | None,
    shard_text_bytes: int,
    resume: bool,
    raw_manifest_path: Path,
) -> dict[str, Any]:
    final_path = output_dir / "manifest.json"
    progress_path = output_dir / "manifest.in_progress.json"
    if final_path.is_file():
        previous = json.loads(final_path.read_text(encoding="utf-8"))
        if previous.get("status") == "COMPLETE":
            valid, errors = _verify_normalized_manifest(
                output_dir,
                previous,
                raw_manifest_path,
                hash_files=True,
                allow_in_progress=True,
            )
            if valid:
                (output_dir / "manifest.in_progress.json").unlink(missing_ok=True)
                for partial_path in output_dir.rglob("*.partial"):
                    partial_path.unlink(missing_ok=True)
                return previous
            raise ValueError(
                "Existing COMPLETE normalized source failed verification: "
                + "; ".join(errors[:8])
            )
        raise ValueError(
            "A finished PARTIAL/FAILED normalized source already exists. Preserve it for audit and use a new output root after correcting the cause."
        )
    if progress_path.is_file():
        if not resume:
            raise ValueError(
                f"Interrupted normalization exists at {progress_path}; rerun with --resume."
            )
        manifest = json.loads(progress_path.read_text(encoding="utf-8"))
        expected = {
            "source": source,
            "raw_source_manifest_sha256": raw_manifest_sha,
            "normalization_tool_git_commit": tool_commit,
            "normalization_schema_version": NORMALIZATION_VERSION,
            "runtime_versions": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "unicode_database": unicodedata.unidata_version,
                "pyarrow": pa.__version__,
            },
            "shard_target_normalized_text_bytes": shard_text_bytes,
        }
        mismatches = [
            key for key, value in expected.items() if manifest.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                "Cannot resume because frozen inputs/options changed: "
                + ", ".join(mismatches)
            )
        valid, errors = _verify_normalized_manifest(
            output_dir,
            manifest,
            hash_files=True,
            allow_partial=True,
            allow_in_progress=True,
            allow_unlisted_shards=True,
        )
        if not valid:
            raise ValueError(
                "Checkpoint output verification failed: " + "; ".join(errors[:8])
            )
        return manifest
    existing = list(output_dir.rglob("part-*.parquet")) if output_dir.exists() else []
    if existing:
        raise ValueError(
            "Output shards exist without a manifest; refusing to overwrite them."
        )
    return _empty_manifest(
        source,
        raw_manifest_path,
        raw_manifest_sha,
        tool_commit,
        shard_text_bytes,
    )


def _remove_uncheckpointed_shards(
    output_dir: Path, manifest: Mapping[str, Any]
) -> None:
    committed = {
        str(item["relative_path"]) for item in manifest.get("normalized_files", [])
    }
    for path in output_dir.rglob("*.partial"):
        path.unlink(missing_ok=True)
    for path in output_dir.rglob("part-*.parquet"):
        relative = path.relative_to(output_dir).as_posix()
        if relative not in committed:
            path.unlink(missing_ok=True)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def normalize_source(
    source: str,
    *,
    raw_root: Path | str = DEFAULT_RAW_ROOT,
    output_root: Path | str = DEFAULT_NORMALIZED_ROOT,
    shard_text_bytes: int = DEFAULT_SHARD_TEXT_BYTES,
    resume: bool = True,
    verify_raw: bool = True,
    tool_commit: str | None = None,
) -> dict[str, Any]:
    """Normalize one fully materialized source into deterministic Parquet.

    Raw payload hashes are verified before processing when ``verify_raw`` is
    true. Production CLI invocations keep this enabled.
    """
    if source not in SOURCE_CONFIG:
        raise ValueError(
            f"Unknown source {source!r}; expected one of {sorted(SOURCE_CONFIG)}"
        )
    if shard_text_bytes <= 0:
        raise ValueError("shard_text_bytes must be a positive integer.")
    config = SOURCE_CONFIG[source]
    raw_source_root = Path(raw_root) / config["raw_dir"]
    output_dir = Path(output_root) / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    tool_commit = tool_commit if tool_commit is not None else _get_tool_commit()

    raw_manifest_path = raw_source_root / "manifest.json"
    try:
        raw_manifest, raw_manifest_sha, raw_manifest_path = (
            _load_and_verify_raw_manifest(source, raw_source_root, verify_raw)
        )
    except Exception as exc:
        failure_manifest = _empty_manifest(
            source,
            raw_manifest_path,
            compute_file_sha256(raw_manifest_path)
            if raw_manifest_path.is_file()
            else None,
            tool_commit,
            shard_text_bytes,
        )
        failure_manifest.update(
            {
                "status": "FAILED",
                "completed_at_utc": utc_now(),
                "normalization_failure_count": 1,
                "failures": [
                    {"kind": "raw_manifest", "error": f"{type(exc).__name__}: {exc}"}
                ],
            }
        )
        _atomic_json(output_dir / "manifest.json", failure_manifest)
        raise

    try:
        entries = _source_entries(raw_manifest)
        if not entries:
            raise ValueError(
                f"Raw source manifest contains no source payloads for {source}."
            )
    except Exception as exc:
        failure_manifest = _empty_manifest(
            source,
            raw_manifest_path,
            raw_manifest_sha,
            tool_commit,
            shard_text_bytes,
        )
        failure_manifest.update(
            {
                "status": "FAILED",
                "completed_at_utc": utc_now(),
                "normalization_failure_count": 1,
                "failures": [
                    {
                        "kind": "raw_manifest_inventory",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ],
            }
        )
        _atomic_json(output_dir / "manifest.json", failure_manifest)
        raise
    manifest = _prepare_resume(
        output_dir,
        source,
        raw_manifest_sha,
        tool_commit,
        shard_text_bytes,
        resume,
        raw_manifest_path,
    )
    manifest["raw_source_manifest"] = str(raw_manifest_path)
    if manifest.get("status") == "COMPLETE":
        return manifest
    _remove_uncheckpointed_shards(output_dir, manifest)
    progress_path = output_dir / "manifest.in_progress.json"
    _atomic_json(progress_path, manifest)

    cursors: dict[str, Any] = dict(manifest.get("cursors", {}))
    failures: list[dict[str, Any]] = list(manifest.get("failures", []))
    output_files: list[dict[str, Any]] = list(manifest.get("normalized_files", []))
    sequence_by_partition: dict[str, int] = {}
    for file_record in output_files:
        part_key = file_record.get("partition_key", "source")
        sequence_by_partition[part_key] = sequence_by_partition.get(part_key, 0) + 1
    buffer: list[dict[str, Any]] = []
    buffer_bytes = 0
    buffer_chars = 0
    buffer_words = 0
    buffer_partition: str | None = None
    first_cursor: dict[str, Any] | None = None
    last_cursor: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal buffer, buffer_bytes, buffer_chars, buffer_words
        nonlocal buffer_partition, first_cursor, last_cursor, manifest
        if not buffer:
            return
        assert (
            buffer_partition is not None
            and first_cursor is not None
            and last_cursor is not None
        )
        table = pa.Table.from_pylist(buffer, schema=NORMALIZED_SCHEMA)
        sequence = sequence_by_partition.get(buffer_partition, 0)
        if source == "gigaverbo_v2":
            subset = buffer[0]["subset"]
            partition_value = quote(str(subset), safe="._-")
            relative_path = f"subset={partition_value}/part-{sequence:05d}.parquet"
        else:
            relative_path = f"part-{sequence:05d}.parquet"
        destination = output_dir / relative_path
        file_bytes, file_sha = _atomic_write_parquet(table, destination)
        output_files.append(
            {
                "relative_path": relative_path,
                "sha256": file_sha,
                "bytes": file_bytes,
                "documents": len(buffer),
                "normalized_bytes": buffer_bytes,
                "normalized_characters": buffer_chars,
                "normalized_words": buffer_words,
                "subset": buffer[0].get("subset") if source == "gigaverbo_v2" else None,
                "partition_key": buffer_partition,
                "raw_start": first_cursor,
                "raw_end": last_cursor,
            }
        )
        sequence_by_partition[buffer_partition] = sequence + 1
        cursors[buffer_partition] = last_cursor
        manifest["normalized_files"] = output_files
        manifest["cursors"] = cursors
        manifest["failures"] = failures
        manifest["normalization_failure_count"] = len(failures)
        manifest.update(_manifest_totals(manifest))
        _atomic_json(progress_path, manifest)
        buffer = []
        buffer_bytes = buffer_chars = buffer_words = 0
        buffer_partition = None
        first_cursor = last_cursor = None

    for raw_record in _source_records(
        source, raw_source_root, raw_manifest, entries, cursors, failures
    ):
        if not _is_after_cursor(raw_record, cursors):
            continue
        try:
            normalized = _normalize_record(raw_record, source)
        except Exception as exc:
            failures.append(
                {
                    "kind": "record",
                    "raw_source_file": raw_record.cursor["raw_source_file"],
                    "raw_record_identifier": raw_record.cursor["raw_record_identifier"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        text = normalized["text"]
        text_bytes = len(text.encode("utf-8"))
        text_chars = len(text)
        text_words = count_normalized_words(text)
        if buffer and (
            raw_record.partition_key != buffer_partition
            or buffer_bytes + text_bytes > shard_text_bytes
        ):
            flush()
        if buffer_partition is None:
            buffer_partition = raw_record.partition_key
            first_cursor = raw_record.cursor
        buffer.append(normalized)
        buffer_bytes += text_bytes
        buffer_chars += text_chars
        buffer_words += text_words
        last_cursor = raw_record.cursor
        if buffer_bytes >= shard_text_bytes:
            flush()
    flush()

    # A source-file parser may have failed after yielding earlier records.
    expected_count = _source_expected_records(
        source, raw_source_root, entries, raw_manifest
    )
    output_count = sum(int(item["documents"]) for item in output_files)
    record_failure_count = sum(1 for item in failures if item.get("kind") == "record")
    parsed_count = output_count + record_failure_count
    if expected_count is not None and parsed_count != expected_count:
        failures.append(
            {
                "kind": "source_count_mismatch",
                "expected_source_document_count": expected_count,
                "parsed_source_document_count": parsed_count,
            }
        )
    source_count = expected_count if expected_count is not None else parsed_count
    has_unparsed_source_file = any(
        item.get("kind") == "source_file" for item in failures
    )
    if has_unparsed_source_file and expected_count is None:
        source_count = None
    manifest.update(_manifest_totals(manifest))
    manifest.update(
        {
            "source_document_count": source_count,
            "output_document_count": output_count,
            "normalization_failure_count": len(failures),
            "failures": failures,
            "completed_at_utc": utc_now(),
            "status": "PARTIAL" if failures else "COMPLETE",
        }
    )
    # Detect changes to the raw manifest while a long normalization was running.
    if compute_file_sha256(raw_manifest_path) != raw_manifest_sha:
        manifest["status"] = "FAILED"
        manifest["failures"].append(
            {
                "kind": "raw_manifest",
                "error": "Raw source manifest changed during normalization.",
            }
        )
        manifest["normalization_failure_count"] = len(manifest["failures"])
    _atomic_json(output_dir / "manifest.json", manifest)
    progress_path.unlink(missing_ok=True)
    if manifest["status"] in {"COMPLETE", "PARTIAL"}:
        valid, errors = _verify_normalized_manifest(
            output_dir,
            manifest,
            raw_manifest_path,
            hash_files=True,
            allow_partial=True,
        )
        if not valid:
            manifest["status"] = "FAILED"
            manifest["completed_at_utc"] = utc_now()
            manifest["failures"] = [
                {"kind": "normalized_output_verification", "error": error}
                for error in errors
            ]
            manifest["normalization_failure_count"] = len(errors)
            _atomic_json(output_dir / "manifest.json", manifest)
            raise ValueError(
                "Normalized output verification failed: " + "; ".join(errors[:8])
            )
    return manifest


__all__ = [
    "DEFAULT_NORMALIZED_ROOT",
    "DEFAULT_RAW_ROOT",
    "DEFAULT_SHARD_TEXT_BYTES",
    "NORMALIZATION_VERSION",
    "NORMALIZED_SCHEMA",
    "FROZEN_RAW_EXPECTATIONS",
    "SOURCE_CONFIG",
    "canonicalize_newlines",
    "count_normalized_words",
    "extract_carolina_tei",
    "normalize_document_text",
    "normalize_source",
    "strip_gutenberg_boilerplate",
    "verify_normalized_source",
]
