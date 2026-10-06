"""Network-independent tests for targeted D2b near-dedup calibration."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cambacica.corpus.dedup import near_calibration as calibration
from cambacica.corpus.dedup import near_pilot as near
from cambacica.corpus.normalization import NORMALIZED_SCHEMA


def _record(
    record_id: str,
    *,
    source: str = "gigaverbo_v2",
    subset: str = "common_crawl",
    title: str = "Uma notícia portuguesa sobre ciência",
    url: str = "https://www.example.pt/artigo",
    words: int = 100,
) -> dict[str, object]:
    return {
        "pilot_occurrence_id": record_id,
        "source": source,
        "subset": subset,
        "title": title,
        "original_url": url,
        "normalized_words": words,
        "data_relative_path": f"{source}/part-00000.parquet",
        "data_row_ordinal": 0,
        "content_sha256": record_id,
        "_gv2_upstream_shard": "crawl-1",
        "_gv2_upstream_row_group": 2,
    }


def _signature() -> dict[str, list[int]]:
    return {
        "word3_128": [0] * 128,
        "word5_256": [0] * 256,
        "word7_128": [0] * 128,
    }


def test_exhaustive_gold_lsh_recall_counts_exact_positives() -> None:
    gold = {
        ("a", "b"): {"family-1"},
        ("a", "c"): {"family-1"},
        ("b", "c"): {"family-1"},
    }
    scores = {
        ("a", "b"): {
            "exact_jaccard_word3": 0.91,
            "exact_jaccard_word5": 0.91,
            "exact_jaccard_word7": 0.91,
        },
        ("a", "c"): {
            "exact_jaccard_word3": 0.82,
            "exact_jaccard_word5": 0.82,
            "exact_jaccard_word7": 0.82,
        },
        ("b", "c"): {
            "exact_jaccard_word3": 0.40,
            "exact_jaccard_word5": 0.40,
            "exact_jaccard_word7": 0.40,
        },
    }
    lsh = {
        ("a", "b"): {"word5_128_32x4"},
    }
    rows = calibration._threshold_rows(gold, scores, lsh)
    union = next(
        row
        for row in rows
        if row["lsh_configuration"] == "word5_256_union" and row["threshold"] == 0.80
    )
    assert union["exhaustive_gold_positive_pairs"] == 2
    assert union["positive_pairs_recalled"] == 1
    assert union["recall"] == 0.5
    assert union["candidate_precision_within_gold_families"] == 1.0


def test_directional_containment_and_length_ratio_are_reported() -> None:
    short = " ".join(f"token{index}" for index in range(150))
    long = " ".join(f"prefix{index}" for index in range(600)) + " " + short
    metrics = calibration._directional_pair_metrics(short, long, set())
    assert metrics["containment_a_in_b_word5"] == 1.0
    assert metrics["containment_b_in_a_word5"] < 0.30
    metadata = {
        "a": {
            **_record("a", source="gutenberg_pt", subset="", words=150),
            "normalized_words": 150,
        },
        "b": {
            **_record("b", source="gigaverbo_v2", words=750),
            "normalized_words": 750,
        },
    }
    score = calibration._score_pair(
        "a",
        "b",
        metadata,
        {"a": short, "b": long},
        {"a": _signature(), "b": _signature()},
        set(),
    )
    assert score["length_ratio_min_max"] == 0.2
    assert score["containment_review_flag"] is True
    assert score["short_document_action"] == "normal_candidate_generation"


def test_boilerplate_document_frequency_reduces_unrelated_body_similarity() -> None:
    template = " ".join(f"template{index:04d}" for index in range(600))
    texts = {
        f"doc-{index}": f"{template} corpo{index} "
        + " ".join(f"exclusivo{index}_{word}" for word in range(30))
        for index in range(4)
    }
    metadata = {
        record_id: {
            "normalized_words": len(text.split()),
            "calibration_families": ["same_url_domain"],
        }
        for record_id, text in texts.items()
    }
    common, metrics = calibration._document_frequency_reference(metadata, texts, cap=10)
    pair = calibration._directional_pair_metrics(texts["doc-0"], texts["doc-1"], common)
    assert metrics["document_frequency_cutoff"] == 3
    assert pair["exact_jaccard_word5"] > 0.80
    assert pair["distinctive_jaccard_word5"] < 0.10
    assert pair["common_fraction_of_shared_word5"] > 0.90


def test_short_document_guard_bands_and_parlamento_preservation() -> None:
    assert calibration._short_document_action(8, 50, "carolina", "gigaverbo_v2") == (
        "exact_only_or_diagnostic"
    )
    assert calibration._short_document_action(20, 99, "carolina", "gigaverbo_v2") == (
        "review_only"
    )
    assert calibration._short_document_action(100, 900, "carolina", "gigaverbo_v2") == (
        "normal_candidate_generation"
    )
    assert (
        calibration._short_document_action(14, 900, "parlamento_pt", "gigaverbo_v2")
        == "preserve_all_diagnostic"
    )


def test_candidate_family_selection_is_deterministic_and_order_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(calibration, "TITLE_KEY_SAMPLE_MODULUS", 1)
    monkeypatch.setattr(calibration, "URL_KEY_SAMPLE_MODULUS", 1)
    monkeypatch.setattr(calibration, "DOMAIN_KEY_SAMPLE_MODULUS", 1)
    monkeypatch.setattr(
        calibration,
        "GOLD_FAMILY_LIMITS",
        {
            "same_title": 1,
            "same_url_variant": 1,
            "same_url_domain": 1,
            "neighboring_crawl_records": 1,
        },
    )
    records = [
        _record(
            f"{index:064x}",
            url=f"https://example.pt/page/{index % 2}?utm_campaign={index}",
        )
        for index in range(4)
    ]
    records.extend(
        _record(
            f"{index:064x}",
            title="Outra reportagem distinta sobre arqueologia",
            url=f"https://another.pt/report/{index}",
        )
        for index in range(4, 6)
    )
    forward = calibration._FamilyCollector(seed=17)
    reverse = calibration._FamilyCollector(seed=17)
    for row in records:
        forward.add(row)
    for row in reversed(records):
        reverse.add(row)
    for left, right in zip(records[:4], records[1:4]):
        forward.add_neighbor(left, right)
    for left, right in reversed(list(zip(records[:4], records[1:4]))):
        reverse.add_neighbor(right, left)
    forward_result = forward.finalize()
    reverse_result = reverse.finalize()

    assert (
        forward_result["candidate_pairs"].keys()
        == reverse_result["candidate_pairs"].keys()
    )
    assert forward_result["gold_pairs"] == reverse_result["gold_pairs"]
    assert (
        forward_result["target_records"].keys()
        == reverse_result["target_records"].keys()
    )


def test_url_variants_collapse_query_fragment_and_trailing_slash() -> None:
    assert (
        calibration._canonical_url(
            "https://www.example.pt/noticia/?utm_source=feed#secao"
        )
        == "example.pt/noticia"
    )
    assert calibration._canonical_url("https://example.pt/noticia") == (
        "example.pt/noticia"
    )


def test_parlamento_scored_pair_never_gets_removal_disposition() -> None:
    score = calibration._score_pair(
        "p",
        "w",
        {
            "p": {
                **_record(
                    "p",
                    source="parlamento_pt",
                    subset="",
                    words=10,
                    title="Fórmula parlamentar",
                    url="",
                ),
            },
            "w": _record("w", source="gigaverbo_v2", words=300),
        },
        {
            "p": "O Senhor Presidente declarou aberta a sessão.",
            "w": " ".join(f"palavra{index}" for index in range(300)),
        },
        {"p": _signature(), "w": _signature()},
        set(),
    )
    assert score["short_document_action"] == "preserve_all_diagnostic"
    assert score["would_remove"] is False


def test_parlamento_is_excluded_from_policy_acceptance_but_counted_diagnostically() -> (
    None
):
    parliament = {
        "source_a": "parlamento_pt",
        "source_b": "parlamento_pt",
        "words_a": 8,
        "words_b": 8,
        "exact_jaccard_word5": 1.0,
        "distinctive_jaccard_word5": 1.0,
        "containment_review_flag": False,
        "provisional_classification": "ambiguous",
    }
    policies = calibration._policy_rows([parliament], [parliament], {}, {})
    assert all(row["review_pairs_accepted"] == 0 for row in policies)
    assert all(
        row["parlamento_diagnostic_pairs_at_or_above_0_80"] == 1 for row in policies
    )


def test_materializer_adds_targeted_text_without_mutating_exact_input(
    tmp_path: Path,
) -> None:
    exact_data = tmp_path / "exact" / "data" / "gutenberg_pt"
    exact_data.mkdir(parents=True)
    relative_path = "gutenberg_pt/part-00000.parquet"
    exact_path = exact_data / "part-00000.parquet"
    texts = [
        "A primeira obra literária tem palavras portuguesas.",
        "A segunda obra literária acrescenta outro texto.",
    ]
    normalized_rows = []
    for index, text in enumerate(texts):
        row = {field.name: None for field in NORMALIZED_SCHEMA}
        row.update(
            {
                "text": text,
                "source": "gutenberg_pt",
                "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "raw_source_file": "book.txt",
                "raw_record_identifier": f"book-{index}",
                "normalization_version": "1.0.0",
            }
        )
        normalized_rows.append(row)
    pq.write_table(
        pa.Table.from_pylist(normalized_rows, schema=NORMALIZED_SCHEMA), exact_path
    )
    original_digest = near.compute_file_sha256(exact_path)

    base_fields = [
        pa.field("pilot_occurrence_id", pa.string(), nullable=False),
        pa.field("normalized_words", pa.int64(), nullable=False),
        pa.field("length_stratum", pa.string(), nullable=False),
        pa.field("selection_role", pa.string(), nullable=False),
        pa.field("sample_tags", pa.list_(pa.string()), nullable=False),
        pa.field("data_relative_path", pa.string(), nullable=False),
        pa.field("data_row_ordinal", pa.int64(), nullable=False),
        pa.field("sampling_frame_population", pa.int64()),
        pa.field("sampling_frame_sample_n", pa.int64()),
        pa.field("base_sampling_weight", pa.float64()),
        pa.field("pilot_file_row_ordinal", pa.int64(), nullable=False),
    ]
    base_schema = NORMALIZED_SCHEMA
    for field in base_fields:
        base_schema = base_schema.append(field)
    base_row = {
        **normalized_rows[0],
        "pilot_occurrence_id": near.occurrence_id(relative_path, 0),
        "normalized_words": len(texts[0].split()),
        "length_stratum": "short",
        "selection_role": "stratified_base",
        "sample_tags": ["stratified_base"],
        "data_relative_path": relative_path,
        "data_row_ordinal": 0,
        "sampling_frame_population": 1,
        "sampling_frame_sample_n": 1,
        "base_sampling_weight": 1.0,
        "pilot_file_row_ordinal": 0,
    }
    base_path = tmp_path / "base" / "pilot_records.parquet"
    base_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([base_row], schema=base_schema), base_path)

    target_id = near.occurrence_id(relative_path, 1)
    target_descriptor = near._candidate_metadata(
        normalized_rows[1], relative_path, 1, 0
    )
    output_path = tmp_path / "calibration_records.parquet"
    metrics = calibration._materialize_calibration_records(
        files=[exact_path],
        base_records_path=base_path,
        target_records={target_id: target_descriptor},
        target_tags={target_id: {"same_title", "same_url_domain"}},
        output_path=output_path,
    )

    table = pq.read_table(output_path)
    assert metrics["calibration_record_count"] == 2
    assert table["pilot_occurrence_id"].to_pylist() == [
        near.occurrence_id(relative_path, 0),
        target_id,
    ]
    assert table["text"].to_pylist() == texts
    assert table["calibration_families"][1].as_py() == [
        "same_title",
        "same_url_domain",
    ]
    assert near.compute_file_sha256(exact_path) == original_digest
