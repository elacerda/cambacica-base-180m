"""Synthetic exact-dedup pipeline tests; no network or production pools used."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import cambacica.corpus.dedup.exact_pipeline as exact
from cambacica.corpus.normalization import (
    NORMALIZATION_VERSION,
    NORMALIZED_SCHEMA,
    SOURCE_CONFIG,
    count_normalized_words,
)
from cambacica.corpus.schema import compute_content_sha256


def _row(source: str, subset: str, text: str, record: str) -> dict:
    row = {
        "text": text,
        "source": source,
        "source_revision": "fixture-revision",
        "subset": subset,
        "original_id": record,
        "original_url": None,
        "license": "fixture",
        "language": "pt",
        "language_score": None,
        "variety": None,
        "quality_score": None,
        "publication_date": None,
        "domain_category": None,
        "content_sha256": compute_content_sha256(text),
        "title": None,
        "raw_source_file": f"{source}/{subset}.fixture",
        "raw_record_identifier": record,
        "normalization_version": NORMALIZATION_VERSION,
        "_gv2_upstream_shard": None,
        "_gv2_upstream_row_group": None,
        "_gv2_upstream_commit": None,
        "upstream_metadata_json": None,
    }
    if source == "gigaverbo_v2":
        row.update(
            {
                "_gv2_upstream_shard": "edu_high/fixture.parquet",
                "_gv2_upstream_row_group": 0,
                "_gv2_upstream_commit": "fixture-commit",
            }
        )
    return row


def _fixture_documents(*, include_length_bands: bool = False) -> dict[str, list[dict]]:
    docs: dict[str, list[dict]] = {source: [] for source in SOURCE_CONFIG}
    docs["gutenberg_pt"].extend(
        [
            _row(
                "gutenberg_pt",
                "literature",
                "A obra completa espelha um livro.",
                "book-a",
            ),
            _row(
                "gutenberg_pt",
                "literature",
                "A obra completa espelha um livro.",
                "book-b",
            ),
        ]
    )
    docs["carolina"].extend(
        [
            _row(
                "carolina", "leg", "Texto nativo de uma lei preservada.", "law-native"
            ),
            _row("carolina", "wik", "Empate nativo determinístico.", "tie-a"),
            _row("carolina", "wik", "Empate nativo determinístico.", "tie-b"),
        ]
    )
    docs["wikipedia_pt"].append(
        _row("wikipedia_pt", "wikipedia", "Artigo completo da Wikipédia.", "wiki-1")
    )
    docs["parlamento_pt"].extend(
        [
            _row("parlamento_pt", "debates", "Artigo completo da Wikipédia.", "line-1"),
            _row("parlamento_pt", "debates", "Fórmula repetida em plenário.", "line-2"),
            _row("parlamento_pt", "debates", "Fórmula repetida em plenário.", "line-3"),
        ]
    )
    docs["gigaverbo_v2"].extend(
        [
            _row(
                "gigaverbo_v2",
                "finepdfs_por_Latn",
                "Artigo completo da Wikipédia.",
                "gv-wiki-copy",
            ),
            _row(
                "gigaverbo_v2",
                "blogset",
                "Texto nativo de uma lei preservada.",
                "gv-law-copy",
            ),
            _row(
                "gigaverbo_v2",
                "finepdfs_por_Latn",
                "A obra completa espelha um livro.",
                "gv-book-copy",
            ),
            _row(
                "gigaverbo_v2",
                "common_crawl",
                "A obra completa espelha um livro.",
                "gv-book-mirror",
            ),
            _row(
                "gigaverbo_v2",
                "finepdfs_por_Latn",
                "Empate de subset GigaVerbo.",
                "gv-tier-native",
            ),
            _row(
                "gigaverbo_v2",
                "common_crawl",
                "Empate de subset GigaVerbo.",
                "gv-tier-legacy",
            ),
            _row(
                "gigaverbo_v2",
                "quati",
                "Texto curado repetido entre subsets.",
                "gv-curated-quati",
            ),
            _row(
                "gigaverbo_v2",
                "blogset",
                "Texto curado repetido entre subsets.",
                "gv-curated-blog",
            ),
        ]
    )
    if include_length_bands:
        docs["gutenberg_pt"].append(
            _row("gutenberg_pt", "literature", "livro " * 100_000, "book-long")
        )
        docs["carolina"].extend(
            [
                _row("carolina", "wik", "curto", "short-carolina"),
                _row("carolina", "wik", "médio " * 25, "medium-carolina"),
                _row("carolina", "leg", "lei " * 100_000, "long-carolina"),
            ]
        )
        docs["wikipedia_pt"].extend(
            [
                _row("wikipedia_pt", "wikipedia", "curto", "short-wiki"),
                _row("wikipedia_pt", "wikipedia", "artigo médio " * 25, "medium-wiki"),
            ]
        )
        docs["parlamento_pt"].extend(
            [
                _row("parlamento_pt", "debates", "curto", "short-parlamento"),
                _row(
                    "parlamento_pt",
                    "debates",
                    "intervenção parlamentar " * 15,
                    "medium-parlamento",
                ),
            ]
        )
        for subset in exact.GV_SUBSET_TIERS:
            docs["gigaverbo_v2"].extend(
                [
                    _row(
                        "gigaverbo_v2",
                        subset,
                        f"Texto exclusivo {subset}.",
                        f"unique-{subset}",
                    ),
                    _row("gigaverbo_v2", subset, f"curto {subset}", f"short-{subset}"),
                    _row(
                        "gigaverbo_v2",
                        subset,
                        (f"documento médio {subset} " * 10),
                        f"medium-{subset}",
                    ),
                    _row(
                        "gigaverbo_v2",
                        subset,
                        f"longo {subset} " * 100_000,
                        f"long-{subset}",
                    ),
                ]
            )
    else:
        # Cover every documented residual provenance tier in the pilot fixture.
        for subset in exact.GV_SUBSET_TIERS:
            if not any(row["subset"] == subset for row in docs["gigaverbo_v2"]):
                docs["gigaverbo_v2"].append(
                    _row(
                        "gigaverbo_v2",
                        subset,
                        f"Texto exclusivo {subset}.",
                        f"unique-{subset}",
                    )
                )
    return docs


def _write_normalized_root(root: Path, *, include_length_bands: bool = False) -> None:
    documents = _fixture_documents(include_length_bands=include_length_bands)
    for source, source_config in SOURCE_CONFIG.items():
        source_dir = root / source_config["output_dir"]
        source_dir.mkdir(parents=True, exist_ok=True)
        rows = documents[source]
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            relative = (
                f"subset={row['subset']}/part-00000.parquet"
                if source == "gigaverbo_v2"
                else "part-00000.parquet"
            )
            grouped.setdefault(relative, []).append(row)
        entries = []
        for relative, file_rows in sorted(grouped.items()):
            path = source_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pylist(file_rows, schema=NORMALIZED_SCHEMA)
            pq.write_table(table, path, compression="zstd", version="2.6")
            texts = [row["text"] for row in file_rows]
            entries.append(
                {
                    "relative_path": relative,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                    "documents": len(file_rows),
                    "normalized_bytes": sum(
                        len(text.encode("utf-8")) for text in texts
                    ),
                    "normalized_characters": sum(len(text) for text in texts),
                    "normalized_words": sum(len(text.split()) for text in texts),
                    "subset": file_rows[0]["subset"]
                    if source == "gigaverbo_v2"
                    else None,
                }
            )
        manifest = {
            "source": source,
            "status": "COMPLETE",
            "normalization_schema_version": NORMALIZATION_VERSION,
            "normalization_failure_count": 0,
            "output_document_count": len(rows),
            "total_normalized_words": sum(len(row["text"].split()) for row in rows),
            "total_normalized_bytes": sum(
                len(row["text"].encode("utf-8")) for row in rows
            ),
            "normalized_files": entries,
        }
        (source_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )


def _read_resolution(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def test_bounded_word_counter_matches_frozen_normalization_rule():
    for text in (
        "",
        "  uma\tduas\ntrês  ",
        "!!! \u001c ...",
        "espaços\u2003Unicode\u00a0preservados",
    ):
        assert exact._count_words_bounded(text) == count_normalized_words(text)


def test_exact_dedup_ownership_accounting_and_verifier(tmp_path: Path):
    normalized_root = tmp_path / "normalized"
    _write_normalized_root(normalized_root)
    output_root = tmp_path / "exact"

    manifest = exact.build_exact_dedup(
        normalized_root=normalized_root, output_root=output_root
    )
    resolution = _read_resolution(output_root / "record_resolution.parquet")
    by_original = {row["raw_record_identifier"]: row for row in resolution}

    assert by_original["gv-wiki-copy"]["disposition"] == "dropped"
    assert (
        by_original["gv-wiki-copy"]["representative_record_id"]
        == by_original["wiki-1"]["record_id"]
    )
    assert by_original["gv-law-copy"]["disposition"] == "dropped"
    assert (
        by_original["gv-law-copy"]["representative_record_id"]
        == by_original["law-native"]["record_id"]
    )
    assert by_original["gv-tier-legacy"]["disposition"] == "dropped"
    assert (
        by_original["gv-tier-legacy"]["representative_record_id"]
        == by_original["gv-tier-native"]["record_id"]
    )
    assert by_original["gv-curated-quati"]["disposition"] == "dropped"
    assert (
        by_original["gv-curated-quati"]["representative_record_id"]
        == by_original["gv-curated-blog"]["record_id"]
    )

    tie_ids = [by_original[name]["record_id"] for name in ("tie-a", "tie-b")]
    tie_winner = min(tie_ids)
    assert [
        by_original[name]["record_id"]
        for name in ("tie-a", "tie-b")
        if by_original[name]["disposition"] == "retained"
    ] == [tie_winner]
    book_ids = [by_original[name]["record_id"] for name in ("book-a", "book-b")]
    book_winner = min(book_ids)
    assert [
        by_original[name]["record_id"]
        for name in ("book-a", "book-b")
        if by_original[name]["disposition"] == "retained"
    ] == [book_winner]
    assert by_original["gv-book-mirror"]["representative_record_id"] == book_winner

    parliament = [row for row in resolution if row["source"] == "parlamento_pt"]
    assert len(parliament) == 3
    assert all(
        row["representative_record_id"] == row["record_id"] for row in parliament
    )
    assert all(row["disposition"] == "preserved_diagnostic" for row in parliament)
    assert manifest["parlamento_diagnostic"]["duplicate_hash_groups"] == 1
    assert manifest["parlamento_diagnostic"]["surplus_repeated_records"] == 1

    edges = pq.read_table(output_root / "duplicate_edges.parquet").to_pylist()
    assert all(edge["dropped_source"] != "parlamento_pt" for edge in edges)
    assert all(edge["retained_source"] != "parlamento_pt" for edge in edges)
    assert manifest["input_record_count"] == len(resolution)
    assert manifest["retained_record_count"] == len(resolution) - len(edges)
    assert (
        exact.verify_exact_dedup(
            normalized_root=normalized_root, output_root=output_root
        )
        == []
    )


def test_output_artifacts_are_independent_of_input_file_traversal_order(tmp_path: Path):
    normalized_root = tmp_path / "normalized"
    _write_normalized_root(normalized_root)
    manifests, input_files = exact._input_catalog(normalized_root)
    first = exact._stage_output(
        normalized_root=normalized_root,
        output_root=tmp_path / "first",
        input_manifests=manifests,
        input_files=input_files,
        run_type="production",
    )
    second = exact._stage_output(
        normalized_root=normalized_root,
        output_root=tmp_path / "second",
        input_manifests=manifests,
        input_files=list(reversed(input_files)),
        run_type="production",
    )
    assert first["output_files"] == second["output_files"]


def test_verifier_catches_corrupt_representative_even_with_updated_file_hash(
    tmp_path: Path,
):
    normalized_root = tmp_path / "normalized"
    _write_normalized_root(normalized_root)
    output_root = tmp_path / "exact"
    exact.build_exact_dedup(normalized_root=normalized_root, output_root=output_root)

    resolution_path = output_root / "record_resolution.parquet"
    table = pq.read_table(resolution_path)
    rows = table.to_pylist()
    dropped = next(row for row in rows if row["disposition"] == "dropped")
    dropped["representative_record_id"] = "missing-representative"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), resolution_path)
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_record = manifest["output_files"]["record_resolution.parquet"]
    file_record["sha256"] = hashlib.sha256(resolution_path.read_bytes()).hexdigest()
    file_record["bytes"] = resolution_path.stat().st_size
    file_record["rows"] = pq.ParquetFile(resolution_path).metadata.num_rows
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    errors = exact.verify_exact_dedup(
        normalized_root=normalized_root, output_root=output_root
    )
    assert any("missing representatives" in error for error in errors)


def test_interrupted_build_never_publishes_partial_output(tmp_path: Path, monkeypatch):
    normalized_root = tmp_path / "normalized"
    _write_normalized_root(normalized_root)
    output_root = tmp_path / "exact"

    def fail_materialization(*_args, **_kwargs):
        raise OSError("simulated interruption")

    monkeypatch.setattr(exact, "_materialize_and_count", fail_materialization)
    with pytest.raises(OSError, match="simulated interruption"):
        exact.build_exact_dedup(
            normalized_root=normalized_root, output_root=output_root
        )
    assert not output_root.exists()
    assert not list(tmp_path.glob(".exact.partial-*"))


def test_pilot_is_deterministic_stratified_and_verifyable(tmp_path: Path):
    normalized_root = tmp_path / "normalized"
    _write_normalized_root(normalized_root, include_length_bands=True)
    first_root = tmp_path / "pilot-a"
    second_root = tmp_path / "pilot-b"
    first = exact.run_exact_dedup_pilot(
        normalized_root=normalized_root,
        output_root=first_root,
        size=60,
        seed=417,
    )
    second = exact.run_exact_dedup_pilot(
        normalized_root=normalized_root,
        output_root=second_root,
        size=60,
        seed=417,
    )
    assert first["pilot"]["required_strata"] == second["pilot"]["required_strata"]
    assert (
        first["pilot"]["selected_row_groups"] == second["pilot"]["selected_row_groups"]
    )
    assert first["output_files"] == second["output_files"]
    assert first["pilot"]["selected_record_count"] >= 60
    assert first["pilot"]["stratified_base_record_count"] > 0
    assert set(first["pilot"]["sample_length_band_counts"]) == {
        "long_at_least_100000_words",
        "medium",
        "short_under_20_words",
    }
    assert first["eligible_duplicate_hash_groups"] > 0
    assert (
        "gigaverbo_v2/blogset"
        in first["pilot"]["stratified_accounting_by_source_subset"]
    )
    assert (
        first["pilot"]["stratified_accounting_by_source_subset"][
            "parlamento_pt/debates"
        ]["documents_removed"]
        == 0
    )
    assert (
        exact.verify_exact_dedup(
            normalized_root=normalized_root, output_root=first_root
        )
        == []
    )


def test_rejects_existing_output_without_mutating_it(tmp_path: Path):
    normalized_root = tmp_path / "normalized"
    _write_normalized_root(normalized_root)
    output_root = tmp_path / "existing"
    output_root.mkdir()
    sentinel = output_root / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        exact.build_exact_dedup(
            normalized_root=normalized_root, output_root=output_root
        )
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_interrupted_pilot_never_publishes_a_sample(tmp_path: Path, monkeypatch):
    normalized_root = tmp_path / "normalized"
    _write_normalized_root(normalized_root)
    output_root = tmp_path / "pilot"

    def fail_scan(*_args, **_kwargs):
        raise OSError("simulated pilot interruption")

    monkeypatch.setattr(exact, "_scan_pilot_row_groups", fail_scan)
    with pytest.raises(OSError, match="simulated pilot interruption"):
        exact.run_exact_dedup_pilot(
            normalized_root=normalized_root,
            output_root=output_root,
            size=60,
            seed=42,
        )
    assert not output_root.exists()
    assert not list(tmp_path.glob(".pilot.select-*"))
