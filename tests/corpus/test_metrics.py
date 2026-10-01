"""Unit tests for sample metrics calculation and diagnostics reporting."""

from pathlib import Path
import pyarrow as pa
import pytest

from cambacica.corpus.metrics import (
    compute_sample_metrics,
    format_report_text,
    inspect_sample_file,
)
from cambacica.corpus.schema import (
    PARQUET_SCHEMA,
    save_sample_parquet,
    validate_and_normalize,
)


def test_compute_sample_metrics_realistic():
    """Verify metrics calculation on a multi-document table."""
    docs = [
        validate_and_normalize(
            {
                "text": "Primeiro documento em português para teste estatístico. " * 3,
                "source": "carolina",
                "subset": "jud",
                "variety": "pt-BR",
                "quality_score": 4.2,
                "original_id": "doc_1",
            }
        ),
        validate_and_normalize(
            {
                "text": "Segundo documento com texto diferente e palavras adicionais.",
                "source": "carolina",
                "subset": "leg",
                "variety": "pt-BR",
                "quality_score": 3.8,
                "original_id": "doc_2",
            }
        ),
        validate_and_normalize(
            {
                "text": "Documento curto.",
                "source": "carolina",
                "subset": "jud",
                "variety": "pt-BR",
                "quality_score": None,
                "original_id": "doc_3",
            }
        ),
        # Exact duplicate of doc_1
        validate_and_normalize(
            {
                "text": "Primeiro documento em português para teste estatístico. " * 3,
                "source": "carolina",
                "subset": "jud",
                "variety": "pt-BR",
                "quality_score": 4.2,
                "original_id": "doc_4",
            }
        ),
    ]

    columns = {
        field.name: [doc.to_dict()[field.name] for doc in docs]
        for field in PARQUET_SCHEMA
    }
    table = pa.Table.from_pydict(columns, schema=PARQUET_SCHEMA)

    report = compute_sample_metrics(table, file_path="sample.parquet")
    assert report.document_count == 4
    assert report.exact_duplicate_count == 1
    assert report.exact_duplicate_rate == 0.25
    assert report.empty_or_near_empty_count >= 1  # doc_3 is short
    assert report.alphabetic_ratio > 0.70
    assert report.subset_distribution["jud"] == 3
    assert report.subset_distribution["leg"] == 1
    assert report.variety_distribution["pt-BR"] == 4
    assert report.quality_score_stats is not None
    assert report.quality_score_stats["count"] == 3
    assert report.quality_score_stats["max"] == pytest.approx(4.2, rel=1e-3)

    # Verify format_report_text renders without error
    rendered = format_report_text(report)
    assert "Documents: 4" in rendered
    assert "pt-BR: 4 (100.0%)" in rendered


def test_inspect_sample_file(tmp_path: Path):
    """Verify inspection from Parquet file path."""
    doc = validate_and_normalize(
        {
            "text": "Conteúdo de exemplo para teste de persistência e leitura.",
            "source": "wikipedia_pt",
            "original_id": "wiki_42",
        }
    )
    p_path = tmp_path / "wiki.parquet"
    save_sample_parquet([doc], p_path)

    report = inspect_sample_file(p_path)
    assert report.document_count == 1
    assert report.source_distribution["wikipedia_pt"] == 1
    assert report.exact_duplicate_count == 0
