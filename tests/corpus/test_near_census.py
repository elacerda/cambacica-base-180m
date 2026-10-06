"""Network-independent tests for the read-only D2c census stages."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import struct

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cambacica.corpus.dedup import near_census as census
from cambacica.corpus.manifest import compute_file_sha256


def _fixture_text(prefix: str = "") -> str:
    return prefix + " ".join(f"palavra{index}" for index in range(40))


def _write_exact_fixture(
    root: Path, *, reverse_file_creation: bool = False
) -> tuple[Path, str]:
    exact_root = root / "exact"
    data_root = exact_root / "data"
    rows_by_file = {
        "carolina/a.parquet": [
            {
                "text": _fixture_text(),
                "source": "carolina",
                "subset": "legislative",
                "original_id": "carolina-a",
                "original_url": "https://carolina.test/doc/a",
                "title": "Documento inicial",
                "domain_category": "legal",
            }
        ],
        "gigaverbo_v2/hplt1_pt/b.parquet": [
            {
                "text": _fixture_text("Cópia: "),
                "source": "gigaverbo_v2",
                "subset": "hplt1_pt",
                "original_id": "web-b",
                "original_url": "https://example.test/page/b",
                "title": "Outro título",
                "domain_category": "web",
            }
        ],
        "parlamento_pt/c.parquet": [
            {
                "text": "sim senhor obrigado",
                "source": "parlamento_pt",
                "subset": None,
                "original_id": "parlamento-c",
                "original_url": None,
                "title": "Fórmula curta",
                "domain_category": "parliament",
            },
            {
                "text": "sim senhor obrigado",
                "source": "parlamento_pt",
                "subset": None,
                "original_id": "parlamento-d",
                "original_url": None,
                "title": "Fórmula curta",
                "domain_category": "parliament",
            },
        ],
        "wikipedia_pt/e.parquet": [
            {
                "text": "Este texto tem conteúdo próprio e palavras distintas "
                "sobre uma espécie de peixe endêmica de Portugal.",
                "source": "wikipedia_pt",
                "subset": None,
                "original_id": "wiki-e",
                "original_url": "https://pt.wikipedia.org/wiki/Peixe",
                "title": "Peixe português",
                "domain_category": "encyclopedia",
            }
        ],
    }
    file_rows = list(rows_by_file.items())
    if reverse_file_creation:
        file_rows.reverse()
    for relative, rows in file_rows:
        path = data_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    count = sum(len(rows) for rows in rows_by_file.values())
    manifest = {
        "status": "COMPLETE",
        "run_type": "production",
        "exact_dedup_version": "fixture",
        "retained_record_count": count,
    }
    manifest_path = exact_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return exact_root, compute_file_sha256(manifest_path)


def test_fingerprints_are_deterministic_and_length_bands_are_complete() -> None:
    text = "Árvore, árvore! folhas verdes e água limpa no vale."
    engine = census._FingerprintEngine()
    first, has_words = engine.signature(text)
    second, second_has_words = engine.signature(text)
    assert first == second
    assert has_words is second_has_words is True
    assert len(first) == census.SIGNATURE_BYTES
    assert struct.unpack("<128Q", first) == tuple(
        census.near.compute_minhash_signature(
            text,
            census.near.MinHashConfig("word5_256", 5, 256, seed=census.SEED),
        )[:128]
    )
    assert [
        census.length_band(value)
        for value in (0, 19, 20, 99, 100, 999, 1_000, 99_999, 100_000)
    ] == [
        "lt20",
        "lt20",
        "20_99",
        "20_99",
        "100_999",
        "100_999",
        "1000_99999",
        "1000_99999",
        "ge100000",
    ]


def test_occurrence_ids_and_file_traversal_are_order_independent(
    tmp_path: Path,
) -> None:
    first = census.occurrence_id_hex("a/file.parquet", 17)
    second = census.occurrence_id_hex("b/file.parquet", 17)
    assert first == census.occurrence_id_hex("a/file.parquet", 17)
    assert first != second
    data = tmp_path / "data"
    for relative in ("z/last.parquet", "a/first.parquet", "m/middle.parquet"):
        (data / relative).parent.mkdir(parents=True, exist_ok=True)
        (data / relative).touch()
    files = census._iter_data_files(data)
    assert [path.relative_to(data).as_posix() for path in files] == [
        "a/first.parquet",
        "m/middle.parquet",
        "z/last.parquet",
    ]


def test_deterministic_reservoir_is_invariant_to_traversal_order() -> None:
    values = [(f"key-{index}", (f"a-{index}", f"b-{index}")) for index in range(50)]
    left = census._DeterministicReservoir(9, seed=71)
    right = census._DeterministicReservoir(9, seed=71)
    for key, pair in values:
        left.add(key, pair)
    for key, pair in reversed(values):
        right.add(key, pair)
    assert left.values() == right.values()


def test_similarity_bands_and_containment_flags() -> None:
    assert [
        census.similarity_band(value)
        for value in (0.69, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 1.0)
    ] == [
        "lt0.70",
        "0.70_0.80",
        "0.80_0.85",
        "0.85_0.90",
        "0.90_0.92",
        "0.92_0.95",
        "ge0.95",
        "ge0.95",
    ]
    assert census.containment_flags(
        jaccard=0.25,
        containment_a_in_b=0.99,
        containment_b_in_a=0.25,
        length_ratio=0.10,
        shared_shingles=150,
    ) == [
        "containment_ge0.90_jaccard_lt0.80",
        "containment_ge0.95_length_ratio_lt0.50",
        "shared_ge100_jaccard_lt0.80",
    ]


def test_bucket_member_cap_counts_without_unbounded_collection() -> None:
    members, size = census._bounded_bucket_members(
        (index.to_bytes(16, "big") for index in range(100_000)), cap=11
    )
    assert size == 100_000
    assert len(members) == 11
    assert members == [index.to_bytes(16, "big") for index in range(11)]


def test_external_exact_scoring_matches_in_memory_result(tmp_path: Path) -> None:
    text_a = " ".join(f"termo{index}" for index in range(1_200))
    text_b = text_a + " termo_extra"
    expected = census._exact_pair_metrics(
        text_a,
        text_b,
        tmp_path,
        expected_words_a=1_200,
        expected_words_b=1_201,
    )
    external = census._exact_pair_metrics(
        text_a,
        text_b,
        tmp_path,
        expected_words_a=300_001,
        expected_words_b=300_002,
    )
    assert external["exact_scoring_method"] == "external_sort"
    assert external["exact_jaccard"] == expected["exact_jaccard"]
    assert external["shared_shingles"] == expected["shared_shingles"]
    assert external["containment_a_in_b"] == expected["containment_a_in_b"]


def test_external_sort_bounds_merge_fan_in(tmp_path: Path) -> None:
    runs = []
    for run_index in range(65):
        path = tmp_path / f"run-{run_index:03d}.u64"
        values = [run_index, run_index + 100]
        path.write_bytes(b"".join(struct.pack("<Q", value) for value in values))
        runs.append(path)
    target = tmp_path / "merged.u64"
    unique_count = census._merge_runs_with_bounded_fan_in(
        runs, target, tmp_path, prefix="test", fan_in=8
    )
    values = list(census._iter_u64_file(target))
    assert unique_count == 130
    assert values == sorted(set(values))
    assert len(values) == 130


def test_parlamento_candidates_are_diagnostic_only() -> None:
    row = census._unpack_candidate_row(
        (
            b"a" * 16,
            b"b" * 16,
            1,
            3,
            4,
            0.99,
            "parlamento_pt",
            None,
            4,
            "lt20",
            "parlamento/a.parquet",
            0,
            0,
            0,
            "p1",
            None,
            "",
            "Fórmula",
            "parliament",
            "parlamento_pt",
            None,
            4,
            "lt20",
            "parlamento/b.parquet",
            1,
            0,
            0,
            "p2",
            None,
            "",
            "Fórmula",
            "parliament",
        )
    )
    assert row["parlamento_diagnostic_only"] is True
    assert row["removal_eligible"] is False
    assert row["short_pair"] is True


def test_exact_sample_strata_cover_candidate_dimensions() -> None:
    row = {
        "estimated_similarity_band": "0.85_0.90",
        "source_pair": "carolina <> gigaverbo_v2/hplt1_pt",
        "source_a": "carolina",
        "source_b": "gigaverbo_v2",
        "domain_a": "carolina.test",
        "domain_b": "example.test",
        "pair_relation": "cross_source",
        "length_ratio_band": "0.90_1.00",
        "length_band_a": "long",
        "length_band_b": "long",
        "candidate_configs": ["word5_128_32x4"],
        "max_bucket_size": 32,
        "bucket_hits": 4,
        "bucket_size_band": "21_64",
    }
    strata = census._candidate_sample_strata(row)
    assert {name for name, _value, _limit in strata} >= {
        "estimated_similarity",
        "source_subset_pair",
        "relationship",
        "length_ratio",
        "length_class_pair",
        "bucket_size",
        "configuration",
        "boilerplate_bucket_proxy",
    }


def test_small_census_smoke_all_stages_and_restart(tmp_path: Path, monkeypatch) -> None:
    exact_root, exact_sha = _write_exact_fixture(tmp_path)
    output_root = tmp_path / "census"

    first = census.run_fingerprints(
        input_root=exact_root,
        output_root=output_root,
        expected_manifest_sha256=None,
        expected_record_count=None,
        batch_size=2,
    )
    first_fingerprint_sha = compute_file_sha256(
        output_root / "signatures" / "fingerprints.parquet"
    )

    def should_not_refingerprint(*_args, **_kwargs):
        raise AssertionError("completed fingerprint stage was not reused")

    monkeypatch.setattr(
        census._FingerprintEngine, "signature", should_not_refingerprint
    )
    second = census.run_fingerprints(
        input_root=exact_root,
        output_root=output_root,
        expected_manifest_sha256=None,
        expected_record_count=None,
        batch_size=2,
    )
    monkeypatch.undo()
    assert first["metrics"]["records"] == second["metrics"]["records"] == 5
    assert (
        compute_file_sha256(output_root / "signatures" / "fingerprints.parquet")
        == first_fingerprint_sha
    )

    census.run_lsh_index(
        input_root=exact_root,
        output_root=output_root,
        expected_manifest_sha256=None,
        expected_record_count=None,
    )
    candidate_manifest = census.run_candidates(
        input_root=exact_root,
        output_root=output_root,
        expected_manifest_sha256=None,
    )
    assert candidate_manifest["metrics"]["unique_union_candidate_pairs"] >= 2
    candidate_db = sqlite3.connect(
        output_root / "lsh" / "candidates" / "candidate_index.sqlite3"
    )
    pair_rows = candidate_db.execute(
        "SELECT id_a,id_b,config_mask,bucket_hits FROM pairs ORDER BY id_a,id_b"
    ).fetchall()
    candidate_db.close()
    assert len({(row[0], row[1]) for row in pair_rows}) == len(pair_rows)
    assert any(row[2] == 3 for row in pair_rows)
    assert any(row[3] > 1 for row in pair_rows)
    selected_first, _ = census._select_exact_sample(
        output_root / "lsh" / "candidates" / "candidate_index.sqlite3"
    )
    selected_second, _ = census._select_exact_sample(
        output_root / "lsh" / "candidates" / "candidate_index.sqlite3"
    )
    assert selected_first == selected_second

    summary = census.run_summarize(
        input_root=exact_root,
        output_root=output_root,
        expected_manifest_sha256=None,
    )
    assert summary["candidate_pairs_without_parlamento_pt"] >= 1
    assert summary["parlamento_pt_within_source_candidate_pairs"] >= 1
    assert (
        output_root / "candidate_summary" / "candidate_counts_by_pair.csv"
    ).is_file()

    exact_sample = census.run_exact_sample(
        input_root=exact_root,
        output_root=output_root,
        expected_manifest_sha256=None,
    )
    assert exact_sample["exact_scored_pair_count"] >= 2
    scored = pq.read_table(
        output_root / "candidate_samples" / "exact_scored_sample.parquet"
    ).to_pylist()
    parliament_rows = [row for row in scored if row["parlamento_diagnostic_only"]]
    assert parliament_rows
    assert all(row["removal_eligible"] is False for row in parliament_rows)
    assert all(
        "words_a" in row and "words_b" in row and "shared_shingles" in row
        for row in scored
    )
    assert (output_root / "candidate_samples" / "review_pairs.parquet").is_file()
    assert (output_root / "candidate_samples" / "exact_sample_strata.csv").is_file()
    assert (output_root / "candidate_samples" / "containment_summary.csv").is_file()
    assert (
        census.verify_census(
            input_root=exact_root,
            output_root=output_root,
            expected_manifest_sha256=None,
        )
        == []
    )
    assert (
        census.main(
            [
                "verify",
                "--input-root",
                str(exact_root),
                "--output-root",
                str(output_root),
                "--expected-manifest-sha256",
                exact_sha,
            ]
        )
        == 0
    )
    assert compute_file_sha256(exact_root / "manifest.json") == exact_sha


def test_interrupted_partial_stage_is_cleaned_on_resume(tmp_path: Path) -> None:
    exact_root, _exact_sha = _write_exact_fixture(tmp_path)
    output_root = tmp_path / "resume-census"
    output_root.mkdir()
    stale = output_root / ".signatures.partial-interrupted"
    stale.mkdir(parents=True)
    (stale / "partial.parquet").write_text("incomplete", encoding="utf-8")
    census.run_fingerprints(
        input_root=exact_root,
        output_root=output_root,
        expected_manifest_sha256=None,
        expected_record_count=None,
        batch_size=2,
    )
    assert not stale.exists()
    assert (output_root / "signatures" / "manifest.json").is_file()


def test_stage_manifest_verification_detects_tampering(tmp_path: Path) -> None:
    exact_root, _exact_sha = _write_exact_fixture(tmp_path)
    output_root = tmp_path / "verify-census"
    census.run_fingerprints(
        input_root=exact_root,
        output_root=output_root,
        expected_manifest_sha256=None,
        expected_record_count=None,
        batch_size=4,
    )
    resource_path = output_root / "signatures" / "resource_report.json"
    resource_path.write_text("tampered\n", encoding="utf-8")
    errors = census.verify_census(
        input_root=exact_root,
        output_root=output_root,
        expected_manifest_sha256=None,
        require_complete=False,
    )
    assert any(
        "SHA-256 mismatch" in error or "size mismatch" in error for error in errors
    )


def test_fingerprint_stage_rejects_unexpected_exact_manifest(tmp_path: Path) -> None:
    exact_root, _exact_sha = _write_exact_fixture(tmp_path)
    output_root = tmp_path / "identity-census"
    with pytest.raises(ValueError, match="manifest SHA-256"):
        census.run_fingerprints(
            input_root=exact_root,
            output_root=output_root,
            expected_manifest_sha256="0" * 64,
            expected_record_count=None,
        )
    assert not output_root.exists()


def test_lsh_configuration_brackets_pilot_tradeoff() -> None:
    assert [(item.bands, item.rows_per_band) for item in census.LSH_CONFIGS] == [
        (32, 4),
        (8, 16),
    ]
    assert all(item.num_permutations == 128 for item in census.LSH_CONFIGS)


def test_candidate_generation_is_invariant_to_file_creation_order(
    tmp_path: Path,
) -> None:
    exact_a, _sha_a = _write_exact_fixture(tmp_path / "input-a")
    exact_b, _sha_b = _write_exact_fixture(
        tmp_path / "input-b", reverse_file_creation=True
    )
    outputs = [tmp_path / "out-a", tmp_path / "out-b"]
    for exact_root, output_root in zip((exact_a, exact_b), outputs):
        census.run_fingerprints(
            input_root=exact_root,
            output_root=output_root,
            expected_manifest_sha256=None,
            expected_record_count=None,
            batch_size=2,
        )
        census.run_lsh_index(
            input_root=exact_root,
            output_root=output_root,
            expected_manifest_sha256=None,
            expected_record_count=None,
        )
        census.run_candidates(
            input_root=exact_root,
            output_root=output_root,
            expected_manifest_sha256=None,
        )
    query = (
        "SELECT id_a,id_b,config_mask,bucket_hits,max_bucket_size,estimated_similarity "
        "FROM pairs ORDER BY id_a,id_b"
    )
    pair_sets = []
    for output_root in outputs:
        connection = sqlite3.connect(
            output_root / "lsh" / "candidates" / "candidate_index.sqlite3"
        )
        pair_sets.append(connection.execute(query).fetchall())
        connection.close()
    assert pair_sets[0] == pair_sets[1]
