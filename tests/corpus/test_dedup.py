"""Unit tests for exact duplicate detection and MinHash diagnostics."""

from pathlib import Path
import pytest

from cambacica.corpus.dedup.exact import find_cross_source_exact_duplicates
from cambacica.corpus.dedup.minhash import (
    MinHashConfig,
    compute_minhash_signature,
    estimate_jaccard_similarity,
    find_minhash_near_duplicates,
    get_word_shingles,
)
from cambacica.corpus.schema import save_sample_parquet, validate_and_normalize


def test_cross_source_exact_duplicates(tmp_path: Path):
    """Verify exact duplicate detection across multiple sample files."""
    text_shared = "Este documento existe em duas fontes diferentes para teste de colisão exata."
    text_unique1 = "Texto exclusivo da primeira fonte de dados para validação."
    text_unique2 = "Texto exclusivo da segunda fonte de dados para validação."

    docs1 = [
        validate_and_normalize({"text": text_shared, "source": "src_alpha", "original_id": "a1"}),
        validate_and_normalize({"text": text_unique1, "source": "src_alpha", "original_id": "a2"}),
    ]
    docs2 = [
        validate_and_normalize({"text": text_shared, "source": "src_beta", "original_id": "b1"}),
        validate_and_normalize({"text": text_unique2, "source": "src_beta", "original_id": "b2"}),
    ]

    p1 = tmp_path / "src1.parquet"
    p2 = tmp_path / "src2.parquet"
    save_sample_parquet(docs1, p1)
    save_sample_parquet(docs2, p2)

    report = find_cross_source_exact_duplicates([p1, p2])
    assert report.total_duplicate_hashes == 1
    assert report.total_duplicate_documents == 1
    # Legacy aliases must still work
    assert "src_alpha <-> src_beta" in report.inter_source_pair_counts
    assert report.inter_source_pair_counts["src_alpha <-> src_beta"] == 1
    # New classification: cross-source pair
    assert "src_alpha <-> src_beta" in report.cross_source_pair_counts
    assert report.cross_source_pair_counts["src_alpha <-> src_beta"] == 1


def test_within_file_duplicates(tmp_path: Path):
    """Verify category A: within-file duplicates are counted separately."""
    text_dup = "A mesma frase aparece duas vezes no mesmo arquivo de amostra para teste."
    docs = [
        validate_and_normalize({"text": text_dup, "source": "carolina", "original_id": "r1"}),
        validate_and_normalize({"text": text_dup, "source": "carolina", "original_id": "r2"}),
        validate_and_normalize({"text": "Texto único aqui.", "source": "carolina", "original_id": "r3"}),
    ]
    p = tmp_path / "representative.parquet"
    save_sample_parquet(docs, p)

    # Need a second file (any) for compare to run
    docs2 = [
        validate_and_normalize({"text": "Outro texto completamente diferente aqui.", "source": "gigaverbo_v2", "original_id": "g1"}),
    ]
    p2 = tmp_path / "audit.parquet"
    save_sample_parquet(docs2, p2)

    report = find_cross_source_exact_duplicates([p, p2])
    assert report.within_file_counts.get("carolina", 0) == 1


def test_same_source_cross_mode_overlap(tmp_path: Path):
    """Verify category B: same upstream source, different files/modes."""
    text_overlap = "Documento que aparece tanto em representative quanto em diagnostic."
    text_only_rep = "Texto exclusivo do modo representative."
    text_only_diag = "Texto exclusivo do modo diagnostic."

    # Two files with the same source name but different modes
    docs_rep = [
        validate_and_normalize({"text": text_overlap, "source": "carolina", "original_id": "r1"}),
        validate_and_normalize({"text": text_only_rep, "source": "carolina", "original_id": "r2"}),
    ]
    docs_diag = [
        validate_and_normalize({"text": text_overlap, "source": "carolina", "original_id": "d1"}),
        validate_and_normalize({"text": text_only_diag, "source": "carolina", "original_id": "d2"}),
    ]

    p_rep = tmp_path / "representative.parquet"
    p_diag = tmp_path / "diagnostic.parquet"
    save_sample_parquet(docs_rep, p_rep)
    save_sample_parquet(docs_diag, p_diag)

    report = find_cross_source_exact_duplicates([p_rep, p_diag])

    # The shared document must land in SAME_SOURCE_CROSS_MODE_OVERLAP (category B)
    assert "carolina" in report.same_source_cross_mode_overlap, (
        f"Expected 'carolina' in same_source_cross_mode_overlap, "
        f"got: {report.same_source_cross_mode_overlap}"
    )
    assert report.same_source_cross_mode_overlap["carolina"] >= 1
    # Must NOT appear as a cross-source (category C) collision
    assert not any("carolina" in k for k in report.cross_source_pair_counts), (
        "Same-source cross-mode overlap must not be reported as cross-source"
    )


def test_no_false_cross_source_for_same_source(tmp_path: Path):
    """Verify that within-mode or cross-mode same-source overlaps are not
    reported as cross-source contamination."""
    shared = "Frase compartilhada entre dois modos do mesmo corpus Carolina."
    for fname in ("representative.parquet", "diagnostic.parquet"):
        docs = [
            validate_and_normalize({"text": shared, "source": "carolina", "original_id": f"id_{fname[:3]}"}),
        ]
        save_sample_parquet(docs, tmp_path / fname)

    report = find_cross_source_exact_duplicates(
        [tmp_path / "representative.parquet", tmp_path / "diagnostic.parquet"]
    )
    # cross_source_pair_counts must be empty
    assert len(report.cross_source_pair_counts) == 0, (
        "Same-source overlap must not appear in cross_source_pair_counts"
    )


def test_minhash_near_duplicates():
    """Verify MinHash signature calculation and candidate pair detection."""
    base_para = (
        "O rápido cachorro marrom pula sobre a cerca de madeira no jardim florido. "
        "O dia amanheceu ensolarado na pequena cidade do interior paulista, com pássaros cantando "
        "nas árvores frondosas da praça central e comerciantes abrindo suas lojas calmamente. "
        "Crianças caminhavam alegres rumo à escola municipal enquanto o padeiro preparava pães frescos."
    )
    text1 = base_para + " Tudo parecia tranquilo na vizinhança pacata."
    text2 = base_para + " Tudo parecia sereno na vizinhança pacata."
    text3 = "A astronomia moderna estuda a evolução de galáxias distantes usando telescópios orbitais de alta resolução."

    config = MinHashConfig(num_permutations=64, ngram_size=5, seed=42)

    sig1 = compute_minhash_signature(text1, config)
    sig2 = compute_minhash_signature(text2, config)
    sig3 = compute_minhash_signature(text3, config)

    sim_near = estimate_jaccard_similarity(sig1, sig2)
    sim_diff = estimate_jaccard_similarity(sig1, sig3)

    assert sim_near >= 0.70
    assert sim_diff < 0.10

    docs = [
        {"original_id": "1", "source": "s1", "text": text1},
        {"original_id": "2", "source": "s2", "text": text2},
        {"original_id": "3", "source": "s3", "text": text3},
    ]

    candidates = find_minhash_near_duplicates(docs, config, threshold=0.70)
    assert len(candidates) == 1
    assert candidates[0].doc_id_1 == "1"
    assert candidates[0].doc_id_2 == "2"
