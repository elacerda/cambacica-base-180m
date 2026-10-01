"""Unit tests for corpus document schema, normalization, and Parquet persistence."""

from pathlib import Path
import pytest

from cambacica.corpus.schema import (
    DOCUMENT_FIELDS,
    PARQUET_SCHEMA,
    NormalizedDocument,
    compute_content_sha256,
    load_sample_parquet,
    normalize_text,
    save_sample_parquet,
    validate_and_normalize,
)


def test_normalize_text():
    """Verify Unicode NFC normalization and whitespace trimming."""
    raw = "   Texto em português com acentuação e espaços extras. \n\t  "
    norm = normalize_text(raw)
    assert norm == "Texto em português com acentuação e espaços extras."
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_compute_content_sha256():
    """Verify content SHA-256 calculation."""
    text = "Olá, mundo!"
    sha = compute_content_sha256(text)
    assert len(sha) == 64
    # Recomputing produces identical hash
    assert compute_content_sha256(text) == sha
    # Different text produces different hash
    assert compute_content_sha256("Outro texto") != sha


def test_validate_and_normalize_valid():
    """Verify validation of a valid document dict."""
    doc_dict = {
        "text": "Este é um documento de teste.",
        "source": "carolina",
        "source_revision": "v2.0.1",
        "subset": "jud",
        "original_id": "doc_123",
        "original_url": "https://example.org/doc/123",
        "license": "CC-BY-4.0",
        "language": "pt-BR",
        "language_score": 0.99,
        "variety": "pt-BR",
        "quality_score": 4.5,
        "publication_date": "2023-01-01",
        "domain_category": "judicial",
    }
    doc = validate_and_normalize(doc_dict)
    assert isinstance(doc, NormalizedDocument)
    assert doc.text == "Este é um documento de teste."
    assert doc.source == "carolina"
    assert doc.language_score == 0.99
    assert doc.quality_score == 4.5
    assert len(doc.content_sha256) == 64


def test_validate_and_normalize_empty_text():
    """Verify that empty or whitespace-only text raises ValueError."""
    with pytest.raises(ValueError, match="non-empty string"):
        validate_and_normalize({"text": "", "source": "test"})
    with pytest.raises(ValueError, match="non-whitespace"):
        validate_and_normalize({"text": "   \n\t  ", "source": "test"})


def test_validate_and_normalize_missing_source():
    """Verify that missing source raises ValueError."""
    with pytest.raises(ValueError, match="source"):
        validate_and_normalize({"text": "Texto válido"})


def test_null_handling():
    """Verify explicit null handling for optional fields."""
    doc_dict = {
        "text": "Texto simples",
        "source": "wikipedia_pt",
        "source_revision": None,
        "subset": "",
        "original_id": "None",
        "language_score": "invalid_float",
    }
    doc = validate_and_normalize(doc_dict)
    assert doc.source_revision is None
    assert doc.subset is None
    assert doc.original_id is None
    assert doc.language_score is None


def test_parquet_save_and_load(tmp_path: Path):
    """Verify roundtrip persistence to Parquet with schema enforcement."""
    docs = [
        validate_and_normalize({
            "text": f"Documento número {i}",
            "source": "test_source",
            "original_id": f"id_{i}",
            "quality_score": float(i) / 10.0,
        })
        for i in range(10)
    ]
    out_file = tmp_path / "test_sample.parquet"
    saved_path = save_sample_parquet(docs, out_file)
    assert saved_path.is_file()

    table = load_sample_parquet(saved_path)
    assert len(table) == 10
    assert table.column_names == DOCUMENT_FIELDS
    assert table["source"].to_pylist()[0] == "test_source"
    assert table["text"].to_pylist()[0] == "Documento número 0"
