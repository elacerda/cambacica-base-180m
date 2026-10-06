"""Network-independent tests for the non-destructive near-dedup pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tracemalloc

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cambacica.corpus.dedup import near_pilot as near
from cambacica.corpus.dedup.exact_pipeline import RESOLUTION_SCHEMA
from cambacica.corpus.normalization import NORMALIZED_SCHEMA


def _exact_root(tmp_path: Path) -> Path:
    root = tmp_path / "exact"
    data = root / "data"
    (data / "carolina").mkdir(parents=True)
    (data / "parlamento_pt").mkdir(parents=True)
    texts = [
        (
            "carolina",
            "dat",
            "A ciência aberta melhora a pesquisa pública e amplia o acesso ao conhecimento. "
            * 24,
            "https://example.pt/artigo-a",
            "Pesquisa científica",
        ),
        (
            "carolina",
            "dat",
            "Página institucional. A ciência aberta melhora a pesquisa pública e amplia o acesso ao conhecimento. "
            * 24,
            "https://example.pt/artigo-b",
            "Pesquisa científica",
        ),
        (
            "parlamento_pt",
            "",
            "O Senhor Presidente declarou aberta a sessão da Assembleia da República.",
            None,
            "Abertura da sessão",
        ),
        (
            "parlamento_pt",
            "",
            "O Senhor Presidente declarou aberta a sessão da Assembleia da República.",
            None,
            "Abertura da sessão",
        ),
    ]
    resolution_rows = []
    for source in ("carolina", "parlamento_pt"):
        source_rows = []
        for index, (row_source, subset, text, url, title) in enumerate(texts):
            if row_source != source:
                continue
            row = {field.name: None for field in NORMALIZED_SCHEMA}
            row.update(
                {
                    "text": text,
                    "source": row_source,
                    "subset": subset or None,
                    "original_id": f"{source}-{index}",
                    "original_url": url,
                    "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "raw_source_file": f"{source}.jsonl",
                    "raw_record_identifier": f"row-{index}",
                    "normalization_version": "1.0.0",
                    "title": title,
                }
            )
            source_rows.append(row)
            record_id = hashlib.sha256(f"{source}-{index}".encode()).hexdigest()
            resolution_rows.append(
                {
                    "record_id": record_id,
                    "cluster_id": None,
                    "content_sha256": row["content_sha256"],
                    "source": source,
                    "subset": subset or "",
                    "normalized_words": len(text.split()),
                    "normalized_shard": f"{source}/part-00000.parquet",
                    "raw_source_file": row["raw_source_file"],
                    "raw_record_identifier": row["raw_record_identifier"],
                    "representative_record_id": record_id,
                    "disposition": (
                        "preserved_diagnostic"
                        if source == "parlamento_pt"
                        else "retained"
                    ),
                    "ownership_rule": (
                        "parlamento_preserve_all"
                        if source == "parlamento_pt"
                        else "exact_unique"
                    ),
                    "selection_role": None,
                    "source_revision": row["source_revision"],
                    "original_id": row["original_id"],
                    "_gv2_upstream_shard": None,
                    "_gv2_upstream_row_group": None,
                    "_gv2_upstream_commit": None,
                    "normalized_row_ordinal": len(source_rows) - 1,
                }
            )
        pq.write_table(
            pa.Table.from_pylist(source_rows, schema=NORMALIZED_SCHEMA),
            data / source / "part-00000.parquet",
        )
    pq.write_table(
        pa.Table.from_pylist(resolution_rows, schema=RESOLUTION_SCHEMA),
        root / "record_resolution.parquet",
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "run_type": "production",
                "exact_dedup_version": "1.0.1",
                "retained_record_count": len(texts),
            }
        ),
        encoding="utf-8",
    )
    return root


def _metadata(
    source: str, subset: str, words: int, record_id: str
) -> dict[str, object]:
    return {
        "source": source,
        "subset": subset,
        "normalized_words": words,
        "pilot_occurrence_id": record_id,
    }


def test_length_strata_and_bounded_word_count_match_whitespace_definition() -> None:
    assert [
        near.length_stratum(n) for n in (0, 19, 20, 999, 1_000, 99_999, 100_000)
    ] == [
        "short",
        "short",
        "medium",
        "medium",
        "long",
        "long",
        "giant",
    ]
    text = ("a  b\nç\t" * 160_000).strip()
    assert near.count_normalized_words_bounded(text) == len(text.split())


def test_word_shingling_and_minhash_are_deterministic() -> None:
    text = "Lisboa, cidade de sete colinas; Portugal tem história."
    assert list(near.iter_word_shingle_hashes(text, 3, seed=17)) == list(
        near.iter_word_shingle_hashes(text, 3, seed=17)
    )
    config = near.MinHashConfig("word5-test", 5, 64, seed=17)
    assert near.compute_minhash_signature(
        text, config
    ) == near.compute_minhash_signature(text, config)


def test_signature_is_invariant_to_shingle_batch_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = " ".join(f"palavra{index % 311}" for index in range(12_000))
    config = near.MinHashConfig("batch-test", 5, 64, seed=5)
    monkeypatch.setattr(near, "SIGNATURE_BATCH_SHINGLES", 11)
    small_batches = near.compute_minhash_signature(text, config)
    monkeypatch.setattr(near, "SIGNATURE_BATCH_SHINGLES", 4096)
    large_batches = near.compute_minhash_signature(text, config)
    assert small_batches == large_batches


def test_short_document_rule_and_empty_text_handling() -> None:
    config = near.MinHashConfig("short-test", 5, 128, seed=8)
    assert near.compute_minhash_signature(
        "a frase curta termina", config
    ) == near.compute_minhash_signature("A frase curta termina", config)
    assert near.compute_minhash_signature(
        "a frase curta termina", config
    ) != near.compute_minhash_signature("a frase curta começa", config)
    assert len(near.compute_minhash_signature("", config)) == 128
    assert (
        near.estimate_jaccard_signature(
            near.compute_minhash_signature("", config),
            near.compute_minhash_signature("", config),
        )
        == 1.0
    )


def test_giant_document_shingling_streams_without_materializing_all_shingles() -> None:
    text = "palavra " * 110_000
    tracemalloc.start()
    generated = near.iter_word_shingle_hashes(text, 5, seed=9)
    assert iter(generated) is generated
    assert sum(1 for _ in generated) == 109_996
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 8 * 1024 * 1024


def test_bottom_k_selection_is_invariant_to_traversal_order() -> None:
    entries = [(f"id-{index:03d}", {"value": index}) for index in range(500)]
    left = near._BottomK(37, seed=31)
    right = near._BottomK(37, seed=31)
    for key, value in entries:
        left.add(key, value)
    for key, value in reversed(entries):
        right.add(key, value)
    assert [item["value"] for item in left.values()] == [
        item["value"] for item in right.values()
    ]


def test_lsh_candidate_generation_is_reproducible_under_input_order() -> None:
    config = near.MinHashConfig("word5_256", 5, 256, seed=41)
    texts = {
        "a": " ".join(f"token{index}" for index in range(80)),
        "b": "cabeçalho " + " ".join(f"token{index}" for index in range(80)),
        "c": " ".join(f"different{index}" for index in range(80)),
    }
    signatures_a = {
        key: {"word5_256": near.compute_minhash_signature(value, config)}
        for key, value in texts.items()
    }
    signatures_b = dict(reversed(list(signatures_a.items())))
    lsh = next(item for item in near.LSH_CONFIGS if item.name == "word5_128_16x8")
    assert near._synthetic_candidate_set(
        signatures_a, lsh
    ) == near._synthetic_candidate_set(signatures_b, lsh)


def test_exact_and_estimated_similarity_scoring() -> None:
    text = "A pesquisa científica em português precisa de dados públicos e diversos."
    exact = near.exact_jaccard_from_text(text, text, ngram_size=5)
    signature = near.compute_minhash_signature(
        text, near.MinHashConfig("word5", 5, 128, seed=21)
    )
    assert exact["exact_jaccard"] == 1.0
    assert near.estimate_jaccard_signature(signature, signature) == 1.0
    changed = near.exact_jaccard_from_text(
        "A sessão foi aberta.", "A sessão foi iniciada."
    )
    assert changed["exact_jaccard"] == 0.0


def test_threshold_sweep_counts_pairs_and_constructs_clusters() -> None:
    connection = near._create_lsh_database(Path(":memory:"))
    left = "a" * 64
    right = "b" * 64
    config_name = "word5_256_16x16"
    connection.execute(
        "INSERT INTO candidate_by_config VALUES (?, ?, ?)", (config_name, left, right)
    )
    connection.execute(
        "INSERT INTO pair_scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (left, right, config_name, "", 0.0, 0.0, 0.0, 0.88, 0.0, 1),
    )
    connection.execute(
        "CREATE TABLE candidate_union (id_a TEXT, id_b TEXT, generators TEXT)"
    )
    connection.execute(
        "INSERT INTO candidate_union VALUES (?, ?, ?)", (left, right, config_name)
    )
    metadata = {
        left: _metadata("carolina", "dat", 100, left),
        right: _metadata("gigaverbo_v2", "blogset", 90, right),
    }
    rows, _source_rows, clusters = near._summary_rows(
        connection, metadata, thresholds=(0.80, 0.90)
    )
    target = [row for row in rows if row["candidate_setup"] == config_name]
    assert target[0]["candidate_pairs"] == 1
    assert target[0]["accepted_near_duplicate_pairs"] == 1
    assert target[0]["clusters"] == 1
    assert target[1]["accepted_near_duplicate_pairs"] == 0
    assert target[1]["clusters"] == 0
    assert len(clusters[config_name][0.80]) == 1
    connection.close()


def test_hypothetical_ownership_keeps_every_parlamento_occurrence() -> None:
    parliament_a = "p" * 64
    parliament_b = "q" * 64
    web = "w" * 64
    metadata = {
        parliament_a: _metadata("parlamento_pt", "", 14, parliament_a),
        parliament_b: _metadata("parlamento_pt", "", 14, parliament_b),
        web: _metadata("gigaverbo_v2", "hplt2_pt", 500, web),
    }
    _clusters, removals = near._cluster_metrics(
        [(parliament_a, parliament_b), (parliament_b, web)], metadata
    )
    assert sum(removals["native_then_aggregator"].values()) == 0
    assert sum(removals["longest_non_parlamento"].values()) == 0


def test_enrichment_keeps_domain_parliament_and_neighbor_diagnostics() -> None:
    collector = near._EnrichmentCollector(seed=59)
    first = {
        **_metadata("gigaverbo_v2", "common_crawl", 400, "a" * 64),
        "original_url": "https://www.example.pt/a",
        "title": "Notícia A",
        "content_sha256": "hash-a",
        "_gv2_upstream_shard": "crawl-1",
        "_gv2_upstream_row_group": 3,
    }
    second = {
        **_metadata("gigaverbo_v2", "common_crawl", 380, "b" * 64),
        "original_url": "https://example.pt/b",
        "title": "Notícia B",
        "content_sha256": "hash-b",
        "_gv2_upstream_shard": "crawl-1",
        "_gv2_upstream_row_group": 3,
    }
    parliament_a = {
        **_metadata("parlamento_pt", "", 14, "c" * 64),
        "content_sha256": "repeated-phrase",
    }
    parliament_b = {
        **_metadata("parlamento_pt", "", 13, "d" * 64),
        "content_sha256": "repeated-phrase",
    }
    collector.add(first)
    collector.add(second)
    collector.add(parliament_a)
    collector.add(parliament_b)
    collector.add_neighbor(first, second)
    reasons = {item["reason"] for item in collector.pairs()}
    assert "same_url_domain" in reasons
    assert "parlamento_exact_hash_relative" in reasons
    assert "neighboring_crawl_records" in reasons


def test_synthetic_cases_expose_boilerplate_false_positive_and_containment() -> None:
    cases = {item["case_id"]: item for item in near._synthetic_cases()}
    template = cases["same_template_unrelated_body"]
    template_score = near.exact_jaccard_from_text(
        template["text_a"], template["text_b"]
    )
    assert template_score["exact_jaccard"] > 0.90
    contained = cases["document_contained_in_larger_document"]
    containment_score = near.exact_jaccard_from_text(
        contained["text_a"], contained["text_b"]
    )
    assert containment_score["containment"] == 1.0
    assert containment_score["exact_jaccard"] < 0.80


def test_pilot_manifest_integrity_and_parlamento_preservation(tmp_path: Path) -> None:
    exact = _exact_root(tmp_path)
    source_file = exact / "data" / "parlamento_pt" / "part-00000.parquet"
    original_sha = near.compute_file_sha256(source_file)
    output = tmp_path / "pilot"
    manifest = near.run_near_dedup_pilot(
        input_root=exact,
        output_root=output,
        seed=23,
        quotas={"short": 8, "medium": 8, "long": 8, "giant": 4},
        require_full_coverage=False,
        expected_manifest_sha256=None,
    )
    assert manifest["status"] == "COMPLETE"
    assert manifest["parlamento_pt"]["rows_marked_removed"] == 0
    assert manifest["parlamento_pt"]["sampled_rows"] == 2
    assert near.verify_near_pilot_manifest(output) == []
    assert near.compute_file_sha256(source_file) == original_sha
    target_file = output / manifest["artifacts"][0]["path"]
    target_file.write_bytes(target_file.read_bytes() + b"tamper")
    assert any(
        "size mismatch" in error for error in near.verify_near_pilot_manifest(output)
    )


def test_interrupted_pilot_cleans_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exact = _exact_root(tmp_path)
    output = tmp_path / "interrupted"

    def interrupt(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(near, "_scan_and_select", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        near.run_near_dedup_pilot(
            input_root=exact,
            output_root=output,
            require_full_coverage=False,
            expected_manifest_sha256=None,
        )
    assert not output.exists()
    assert list(tmp_path.glob(".interrupted.partial-*")) == []
