"""Normalized document schema and Parquet I/O for Cambacica corpus samples.

This module defines the canonical 14-field schema used across all Gate C1
sampling and inspection pipelines, ensuring strict type consistency, local
content SHA-256 computation, and safe null-handling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import unicodedata

import pyarrow as pa
import pyarrow.parquet as pq


DOCUMENT_FIELDS: List[str] = [
    "text",
    "source",
    "source_revision",
    "subset",
    "original_id",
    "original_url",
    "license",
    "language",
    "language_score",
    "variety",
    "quality_score",
    "publication_date",
    "domain_category",
    "content_sha256",
]

PARQUET_SCHEMA = pa.schema(
    [
        pa.field("text", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("source_revision", pa.string(), nullable=True),
        pa.field("subset", pa.string(), nullable=True),
        pa.field("original_id", pa.string(), nullable=True),
        pa.field("original_url", pa.string(), nullable=True),
        pa.field("license", pa.string(), nullable=True),
        pa.field("language", pa.string(), nullable=True),
        pa.field("language_score", pa.float32(), nullable=True),
        pa.field("variety", pa.string(), nullable=True),
        pa.field("quality_score", pa.float32(), nullable=True),
        pa.field("publication_date", pa.string(), nullable=True),
        pa.field("domain_category", pa.string(), nullable=True),
        pa.field("content_sha256", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class NormalizedDocument:
    """Normalized document representation for Gate C1 corpus samples.

    Parameters
    ----------
    text : str
        Cleaned document text in UTF-8. Must not be empty.
    source : str
        Canonical source identifier (e.g. 'carolina', 'gigaverbo_v2').
    source_revision : str or None, optional
        Upstream revision, commit SHA, or dump date if available.
    subset : str or None, optional
        Upstream configuration or crawl partition (e.g. 'edu_high', 'mc4_pt').
    original_id : str or None, optional
        Original document identifier from upstream.
    original_url : str or None, optional
        Canonical URL where document was harvested, if available.
    license : str or None, optional
        Document or collection license tag.
    language : str or None, optional
        Documented language identifier (e.g. 'pt', 'pt-BR', 'pt-PT').
    language_score : float or None, optional
        Upstream language identification confidence score.
    variety : str or None, optional
        Documented or verified variety ('pt-BR', 'pt-PT', 'palop').
    quality_score : float or None, optional
        Upstream educational or quality score.
    publication_date : str or None, optional
        Publication date or year in ISO format if available.
    domain_category : str or None, optional
        Domain typology or category label.
    content_sha256 : str
        Locally computed hex SHA-256 of the normalized text.
    """

    text: str
    source: str
    source_revision: Optional[str] = None
    subset: Optional[str] = None
    original_id: Optional[str] = None
    original_url: Optional[str] = None
    license: Optional[str] = None
    language: Optional[str] = None
    language_score: Optional[float] = None
    variety: Optional[str] = None
    quality_score: Optional[float] = None
    publication_date: Optional[str] = None
    domain_category: Optional[str] = None
    content_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the document to a dictionary matching the Parquet schema.

        Returns
        -------
        dict
            Dictionary containing all fields defined in DOCUMENT_FIELDS.
        """
        return asdict(self)


def normalize_text(text: str) -> str:
    """Normalize text into clean NFC unicode with trimmed whitespace.

    Parameters
    ----------
    text : str
        Input raw text.

    Returns
    -------
    str
        NFC normalized text without leading or trailing whitespace.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFC", text)
    return normalized.strip()


def compute_content_sha256(text: str) -> str:
    """Compute the SHA-256 checksum of UTF-8 encoded text.

    Parameters
    ----------
    text : str
        Text string to hash.

    Returns
    -------
    str
        Hexadecimal representation of the SHA-256 hash.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_and_normalize(doc: Dict[str, Any]) -> NormalizedDocument:
    """Validate input mapping and construct a NormalizedDocument.

    Parameters
    ----------
    doc : dict
        Raw document mapping containing source text and metadata.

    Returns
    -------
    NormalizedDocument
        Validated and normalized document instance.

    Raises
    ------
    ValueError
        If 'text' is missing/empty or if 'source' is not provided.
    """
    raw_text = doc.get("text")
    if not raw_text or not isinstance(raw_text, str):
        raise ValueError("Document 'text' must be a non-empty string.")

    cleaned_text = normalize_text(raw_text)
    if not cleaned_text:
        raise ValueError("Document 'text' must contain non-whitespace characters.")

    source = doc.get("source")
    if not source or not isinstance(source, str):
        raise ValueError("Document 'source' must be a non-empty string.")

    content_sha = compute_content_sha256(cleaned_text)

    # Optional float conversions
    lang_score = doc.get("language_score")
    if lang_score is not None:
        try:
            lang_score = float(lang_score)
        except (ValueError, TypeError):
            lang_score = None

    qual_score = doc.get("quality_score")
    if qual_score is not None:
        try:
            qual_score = float(qual_score)
        except (ValueError, TypeError):
            qual_score = None

    def _str_or_none(val: Any) -> Optional[str]:
        if val is None or val == "" or str(val).lower() == "none":
            return None
        return str(val).strip()

    return NormalizedDocument(
        text=cleaned_text,
        source=str(source).strip(),
        source_revision=_str_or_none(doc.get("source_revision")),
        subset=_str_or_none(doc.get("subset")),
        original_id=_str_or_none(doc.get("original_id")),
        original_url=_str_or_none(doc.get("original_url")),
        license=_str_or_none(doc.get("license")),
        language=_str_or_none(doc.get("language")),
        language_score=lang_score,
        variety=_str_or_none(doc.get("variety")),
        quality_score=qual_score,
        publication_date=_str_or_none(doc.get("publication_date")),
        domain_category=_str_or_none(doc.get("domain_category")),
        content_sha256=content_sha,
    )


def save_sample_parquet(
    documents: Sequence[NormalizedDocument],
    output_path: Path | str,
) -> Path:
    """Save normalized documents into a compressed Parquet file.

    Parameters
    ----------
    documents : sequence of NormalizedDocument
        Collection of normalized documents to persist.
    output_path : Path or str
        Destination path for the Parquet file.

    Returns
    -------
    Path
        Resolved output path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns: Dict[str, List[Any]] = {field.name: [] for field in PARQUET_SCHEMA}
    for doc in documents:
        d = doc.to_dict()
        for field in PARQUET_SCHEMA:
            columns[field.name].append(d.get(field.name))

    table = pa.Table.from_pydict(columns, schema=PARQUET_SCHEMA)
    pq.write_table(table, path, compression="zstd", compression_level=3)
    return path


def load_sample_parquet(input_path: Path | str) -> pa.Table:
    """Load a sample Parquet file and validate its schema.

    Parameters
    ----------
    input_path : Path or str
        Path to the Parquet sample file.

    Returns
    -------
    pyarrow.Table
        Loaded PyArrow table.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required schema fields are missing.
    """
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    table = pq.read_table(path)
    table_field_names = set(table.schema.names)
    for field in PARQUET_SCHEMA:
        if field.name not in table_field_names:
            raise ValueError(f"Missing required field '{field.name}' in {path.name}.")
    return table
