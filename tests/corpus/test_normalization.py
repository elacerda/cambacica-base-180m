"""Small, network-independent tests for frozen C1 normalization."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import cambacica.corpus.normalization as normalization
from cambacica.corpus.characterize import _mix_capacities, characterize_sources
from cambacica.corpus.manifest import compute_file_sha256
from cambacica.corpus.mix import validate_mix_files
from cambacica.corpus.normalization import (
    NORMALIZED_SCHEMA,
    NORMALIZATION_VERSION,
    count_normalized_words,
    normalize_document_text,
    normalize_source,
    strip_gutenberg_boilerplate,
    verify_normalized_source,
)


def _write_raw_manifest(
    raw_source_dir: Path,
    source: str,
    file_paths: list[Path],
    *,
    line_count: int | None = None,
    gigaverbo_records: int | None = None,
) -> Path:
    records = []
    for path in file_paths:
        relative = path.relative_to(raw_source_dir).as_posix()
        record = {
            "relative_path": relative,
            "upstream_identifier": path.name,
            "url": f"https://fixture.invalid/{relative}",
            "bytes": path.stat().st_size,
            "sha256": compute_file_sha256(path),
        }
        if source == "gigaverbo_v2":
            record.update(
                {
                    "records": pq.ParquetFile(path).metadata.num_rows,
                    "upstream_subset": path.parent.name.removeprefix("subset="),
                    "upstream_shard": "edu_high/train-fixture.parquet",
                    "upstream_row_group": 0,
                }
            )
        records.append(record)
    payload = {
        "source": source,
        "upstream_repository": f"fixture/{source}",
        "pinned_revision": "fixture-revision",
        "pinned_commit_sha": "fixture-commit",
        "schema_version": 1,
        "status": "COMPLETE",
        "total_files": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "files": records,
    }
    if line_count is not None:
        payload["line_count"] = line_count
    if gigaverbo_records is not None:
        payload["source_metadata"] = {"eligible_records": gigaverbo_records}
    manifest_path = raw_source_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _fixture_raw_tree(root: Path) -> Path:
    raw_root = root / "raw"

    gutenberg = raw_root / "gutenberg"
    gutenberg.mkdir(parents=True)
    book_one = (
        "Administrative header\r\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK EXEMPLO ***\r\n"
        "Prefácio mantido.\r\n\r\nCapítulo I. Nota de rodapé.\r\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK EXEMPLO\r\n"
        "Administrative footer"
    )
    book_two = "  cafe\u0301, sem marcadores.  \n"
    (gutenberg / "pg101.txt").write_bytes(book_one.encode("utf-8"))
    (gutenberg / "pg102.txt").write_bytes(book_two.encode("utf-8"))
    _write_raw_manifest(gutenberg, "gutenberg_pt", sorted(gutenberg.glob("*.txt")))

    parlamento = raw_root / "parlamento"
    parlamento.mkdir(parents=True)
    parlamento_data = (
        "Linha um.\nLinha dois.\n\nFórmula repetida.\nFórmula repetida.\n".encode()
    )
    (parlamento / "train.txt").write_bytes(parlamento_data)
    _write_raw_manifest(
        parlamento,
        "parlamento_pt",
        [parlamento / "train.txt"],
        line_count=parlamento_data.count(b"\n"),
    )

    wikipedia = raw_root / "wikipedia"
    wiki_part = wikipedia / "20231101.pt"
    wiki_part.mkdir(parents=True)
    wiki_path = wiki_part / "train-00000-of-00006.parquet"
    wiki_table = pa.table(
        {
            "id": ["10", "11"],
            "url": ["https://pt.wikipedia.org/wiki/Exemplo", None],
            "title": ["Astronomia", "Título"],
            "text": ["Texto do corpo.", "Título já presente no texto."],
        }
    )
    pq.write_table(wiki_table, wiki_path, compression="zstd")
    _write_raw_manifest(wikipedia, "wikipedia_pt", [wiki_path])

    carolina = raw_root / "carolina"
    carolina_dir = carolina / "corpus" / "legislative_branch"
    carolina_dir.mkdir(parents=True)
    carolina_xml = (
        """<?xml version="1.0" encoding="UTF-8"?>
<teiCorpus xmlns="http://www.tei-c.org/ns/1.0">
  <TEI xml:id="lei-1">
    <teiHeader><fileDesc><titleStmt><title>Lei de exemplo</title><author>Ana Exemplo</author></titleStmt></fileDesc>
      <publicationStmt><date>2024</date><availability><licence>CC-BY-4.0</licence></availability></publicationStmt>
      <sourceDesc><p><ref target="https://example.org/lei-1">fonte</ref></p></sourceDesc>
    </teiHeader>
    <text><body><head>Capítulo primeiro</head><p>Uma <hi>palavra</hi> preservada.</p><note>Nota de rodapé preservada.</note></body></text>
  </TEI>
  <TEI><teiHeader><fileDesc><titleStmt><title>Lei longa</title></titleStmt></fileDesc></teiHeader>
    <text><body><p>"""
        + "lei " * 160
        + """integral.</p></body></text>
  </TEI>
</teiCorpus>"""
    )
    carolina_path = carolina_dir / "LEGfixture.xml.gz"
    carolina_path.write_bytes(gzip.compress(carolina_xml.encode("utf-8"), mtime=0))
    _write_raw_manifest(carolina, "carolina", [carolina_path])

    gigaverbo = raw_root / "gigaverbo"
    giga_subset = gigaverbo / "subset=blogset"
    giga_subset.mkdir(parents=True)
    giga_path = giga_subset / "train-fixture.parquet"
    giga_rows = [
        {
            "text": "curto",
            "id": "gv-id-1",
            "source": "https://example.org/gv/1",
            "subset": "blogset",
            "token_count": 1,
            "edu_score": 3.8,
            "edu_int_score": 4,
            "toxic_score": 0.01,
            "toxic_int_score": 0,
            "_gv2_upstream_shard": "edu_high/train-fixture.parquet",
            "_gv2_upstream_row_group": 3,
            "_gv2_upstream_commit": "fixture-commit",
        },
        {
            "text": "Texto com pontuação!",
            "id": "gv-id-2",
            "source": "https://example.org/gv/2",
            "subset": "blogset",
            "token_count": 4,
            "edu_score": 4.1,
            "edu_int_score": 4,
            "toxic_score": None,
            "toxic_int_score": None,
            "_gv2_upstream_shard": "edu_high/train-fixture.parquet",
            "_gv2_upstream_row_group": 3,
            "_gv2_upstream_commit": "fixture-commit",
        },
    ]
    pq.write_table(pa.Table.from_pylist(giga_rows), giga_path, compression="zstd")
    _write_raw_manifest(
        gigaverbo,
        "gigaverbo_v2",
        [giga_path],
        gigaverbo_records=2,
    )
    return raw_root


def _read_rows(root: Path, source_dir: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted((root / source_dir).rglob("part-*.parquet")):
        rows.extend(pq.ParquetFile(path).read().to_pylist())
    return rows


def test_word_rule_is_existing_c1_whitespace_definition():
    text = normalize_document_text("  Ca\u0301fe\r\n— Olá!\t  ")
    assert text == "Cáfe\n— Olá!"
    assert count_normalized_words(text) == len(text.split()) == 3
    assert count_normalized_words("!!! ...") == 2


def test_gutenberg_marker_rule_is_deterministic_and_conservative():
    raw = (
        "legal wrapper\n*** START OF THE PROJECT GUTENBERG EBOOK Livro ***\n"
        "Prefácio.\nCapítulo I. Nota.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK Livro\nfooter"
    )
    body = strip_gutenberg_boilerplate(raw)
    assert body == "\nPrefácio.\nCapítulo I. Nota.\n"
    assert strip_gutenberg_boilerplate(body) == body
    assert strip_gutenberg_boilerplate("Sem marcadores.") == "Sem marcadores."
    single_start = "wrapper\n*** START OF THE PROJECT GUTENBERG EBOOK Livro ***\nbody"
    single_end = "body\n*** END OF THE PROJECT GUTENBERG EBOOK Livro\nfooter"
    reversed_markers = (
        "*** END OF THE PROJECT GUTENBERG EBOOK Livro\nbody\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK Livro ***"
    )
    assert strip_gutenberg_boilerplate(single_start) == single_start
    assert strip_gutenberg_boilerplate(single_end) == single_end
    assert strip_gutenberg_boilerplate(reversed_markers) == reversed_markers


def test_all_source_adapters_preserve_records_metadata_and_boundaries(tmp_path: Path):
    raw_root = _fixture_raw_tree(tmp_path)
    normalized_root = tmp_path / "normalized"
    for source in (
        "gutenberg_pt",
        "parlamento_pt",
        "wikipedia_pt",
        "carolina",
        "gigaverbo_v2",
    ):
        manifest = normalize_source(
            source,
            raw_root=raw_root,
            output_root=normalized_root,
            shard_text_bytes=64,
            verify_raw=False,
            tool_commit="fixture-tool-commit",
        )
        assert manifest["status"] == "COMPLETE"
        assert manifest["normalization_schema_version"] == NORMALIZATION_VERSION
        valid, errors = verify_normalized_source(
            source,
            output_root=normalized_root,
            raw_root=raw_root,
        )
        assert valid, errors
        for file_record in manifest["normalized_files"]:
            assert len(file_record["sha256"]) == 64

    gutenberg_rows = _read_rows(normalized_root, "gutenberg")
    assert len(gutenberg_rows) == 2
    assert "Prefácio mantido." in gutenberg_rows[0]["text"]
    assert "Capítulo I. Nota de rodapé." in gutenberg_rows[0]["text"]
    assert "Administrative header" not in gutenberg_rows[0]["text"]
    assert "café" in gutenberg_rows[1]["text"]

    parlamento_rows = _read_rows(normalized_root, "parlamento")
    assert len(parlamento_rows) == 5
    assert [row["raw_record_identifier"] for row in parlamento_rows] == [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]
    assert [row["text"] for row in parlamento_rows] == [
        "Linha um.",
        "Linha dois.",
        "",
        "Fórmula repetida.",
        "Fórmula repetida.",
    ]
    assert parlamento_rows[3]["original_id"] == "parl_pt_4"
    assert parlamento_rows[3]["domain_category"] == "parliamentary_records"

    wikipedia_rows = _read_rows(normalized_root, "wikipedia")
    assert len(wikipedia_rows) == 2
    assert wikipedia_rows[0]["text"] == "Astronomia\n\nTexto do corpo."
    assert wikipedia_rows[1]["text"] == "Título já presente no texto."
    assert wikipedia_rows[0]["original_id"] == "10"
    assert wikipedia_rows[0]["original_url"] == "https://pt.wikipedia.org/wiki/Exemplo"

    carolina_rows = _read_rows(normalized_root, "carolina")
    assert len(carolina_rows) == 2
    assert carolina_rows[0]["subset"] == "leg"
    assert "Capítulo primeiro" in carolina_rows[0]["text"]
    assert "palavra preservada" in carolina_rows[0]["text"]
    assert carolina_rows[0]["original_id"] == "lei-1"
    assert carolina_rows[0]["title"] == "Lei de exemplo"
    assert carolina_rows[0]["publication_date"] == "2024"
    assert carolina_rows[0]["license"] == "CC-BY-4.0"
    assert "Nota de rodapé preservada." in carolina_rows[0]["text"]
    assert carolina_rows[0]["original_id"] == "lei-1"
    assert json.loads(carolina_rows[0]["upstream_metadata_json"])["author"] == [
        "Ana Exemplo"
    ]
    assert carolina_rows[1]["original_id"].endswith("#tei-000000002")
    assert len(carolina_rows[1]["text"]) > 500
    assert "lei " in carolina_rows[1]["text"]

    gigaverbo_rows = _read_rows(normalized_root, "gigaverbo")
    assert len(gigaverbo_rows) == 2
    assert {row["subset"] for row in gigaverbo_rows} == {"blogset"}
    assert {row["original_id"] for row in gigaverbo_rows} == {"gv-id-1", "gv-id-2"}
    assert {row["_gv2_upstream_shard"] for row in gigaverbo_rows} == {
        "edu_high/train-fixture.parquet"
    }
    assert {row["_gv2_upstream_row_group"] for row in gigaverbo_rows} == {3}
    assert {row["_gv2_upstream_commit"] for row in gigaverbo_rows} == {"fixture-commit"}
    metadata = json.loads(gigaverbo_rows[0]["upstream_metadata_json"])
    assert metadata["token_count"] == 1
    assert metadata["toxic_score"] == 0.01
    assert gigaverbo_rows[0]["raw_record_identifier"] == "gv-id-1"
    assert gigaverbo_rows[0]["original_url"] == "https://example.org/gv/1"

    for source in (
        "gutenberg_pt",
        "parlamento_pt",
        "wikipedia_pt",
        "carolina",
        "gigaverbo_v2",
    ):
        for row in _read_rows(
            normalized_root,
            {
                "gutenberg_pt": "gutenberg",
                "parlamento_pt": "parlamento",
                "wikipedia_pt": "wikipedia",
                "carolina": "carolina",
                "gigaverbo_v2": "gigaverbo",
            }[source],
        ):
            assert (
                row["content_sha256"]
                == hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            )
            assert row["normalization_version"] == NORMALIZATION_VERSION
            assert row["raw_source_file"]


def test_output_shards_are_deterministic_and_manifest_hashes_verify(tmp_path: Path):
    raw_root = _fixture_raw_tree(tmp_path)
    out_a = tmp_path / "normalized-a"
    out_b = tmp_path / "normalized-b"
    first = normalize_source(
        "gutenberg_pt",
        raw_root=raw_root,
        output_root=out_a,
        shard_text_bytes=64,
        verify_raw=False,
        tool_commit="fixed-commit",
    )
    second = normalize_source(
        "gutenberg_pt",
        raw_root=raw_root,
        output_root=out_b,
        shard_text_bytes=64,
        verify_raw=False,
        tool_commit="fixed-commit",
    )
    assert [item["sha256"] for item in first["normalized_files"]] == [
        item["sha256"] for item in second["normalized_files"]
    ]
    valid, errors = verify_normalized_source(
        "gutenberg_pt", output_root=out_a, raw_root=raw_root
    )
    assert valid, errors
    output_file = out_a / "gutenberg" / first["normalized_files"][0]["relative_path"]
    output_file.write_bytes(output_file.read_bytes() + b"corruption")
    valid, errors = verify_normalized_source(
        "gutenberg_pt", output_root=out_a, raw_root=raw_root
    )
    assert not valid
    assert any("SHA-256 mismatch" in error for error in errors)


def test_interrupted_normalization_resumes_from_committed_boundary(
    tmp_path: Path, monkeypatch
):
    raw_root = _fixture_raw_tree(tmp_path)
    interrupted_root = tmp_path / "normalized-interrupted"
    baseline_root = tmp_path / "normalized-baseline"
    original_writer = normalization._atomic_write_parquet
    calls = 0

    def fail_second_write(table, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated interruption")
        return original_writer(table, destination)

    monkeypatch.setattr(normalization, "_atomic_write_parquet", fail_second_write)
    with pytest.raises(OSError, match="simulated interruption"):
        normalize_source(
            "gutenberg_pt",
            raw_root=raw_root,
            output_root=interrupted_root,
            shard_text_bytes=1,
            verify_raw=False,
            tool_commit="fixed-commit",
        )
    progress = interrupted_root / "gutenberg" / "manifest.in_progress.json"
    assert progress.is_file()
    checkpoint = json.loads(progress.read_text(encoding="utf-8"))
    assert len(checkpoint["normalized_files"]) == 1

    monkeypatch.setattr(normalization, "_atomic_write_parquet", original_writer)
    resumed = normalize_source(
        "gutenberg_pt",
        raw_root=raw_root,
        output_root=interrupted_root,
        shard_text_bytes=1,
        resume=True,
        verify_raw=False,
        tool_commit="fixed-commit",
    )
    baseline = normalize_source(
        "gutenberg_pt",
        raw_root=raw_root,
        output_root=baseline_root,
        shard_text_bytes=1,
        verify_raw=False,
        tool_commit="fixed-commit",
    )
    assert resumed["status"] == "COMPLETE"
    assert [item["sha256"] for item in resumed["normalized_files"]] == [
        item["sha256"] for item in baseline["normalized_files"]
    ]
    assert len(_read_rows(interrupted_root, "gutenberg")) == 2
    assert not progress.exists()


def test_malformed_source_payload_is_visible_as_partial(tmp_path: Path):
    raw_root = tmp_path / "raw"
    source_dir = raw_root / "gutenberg"
    source_dir.mkdir(parents=True)
    malformed = source_dir / "pg999.txt"
    malformed.write_bytes(b"texto invalido: \xff")
    _write_raw_manifest(source_dir, "gutenberg_pt", [malformed])

    manifest = normalize_source(
        "gutenberg_pt",
        raw_root=raw_root,
        output_root=tmp_path / "normalized",
        verify_raw=False,
        tool_commit="fixture-commit",
    )
    assert manifest["status"] == "PARTIAL"
    assert manifest["source_document_count"] == 1
    assert manifest["output_document_count"] == 0
    assert manifest["normalization_failure_count"] >= 1
    assert any(failure["kind"] == "source_file" for failure in manifest["failures"])


def test_production_raw_manifest_must_match_frozen_identity(tmp_path: Path):
    raw_root = _fixture_raw_tree(tmp_path)
    with pytest.raises(ValueError, match="frozen C1 snapshot"):
        normalization._load_and_verify_raw_manifest(
            "gutenberg_pt", raw_root / "gutenberg", verify_raw=True
        )


def test_verifier_recomputes_normalized_content_sha256(tmp_path: Path):
    raw_root = _fixture_raw_tree(tmp_path)
    output_root = tmp_path / "normalized"
    manifest = normalize_source(
        "gutenberg_pt",
        raw_root=raw_root,
        output_root=output_root,
        verify_raw=False,
        tool_commit="fixture-tool-commit",
    )
    output_dir = output_root / "gutenberg"
    file_record = manifest["normalized_files"][0]
    parquet_path = output_dir / file_record["relative_path"]
    rows = pq.ParquetFile(parquet_path).read().to_pylist()
    rows[0]["content_sha256"] = "0" * 64
    pq.write_table(
        pa.Table.from_pylist(rows, schema=NORMALIZED_SCHEMA),
        parquet_path,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
        row_group_size=65_536,
    )
    file_record["bytes"] = parquet_path.stat().st_size
    file_record["sha256"] = compute_file_sha256(parquet_path)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    valid, errors = verify_normalized_source(
        "gutenberg_pt", output_root=output_root, raw_root=raw_root
    )
    assert not valid
    assert any("content_sha256 does not match" in error for error in errors)


def test_carolina_malformed_xml_file_is_not_silently_dropped(tmp_path: Path):
    raw_root = tmp_path / "raw"
    carolina = raw_root / "carolina"
    good_dir = carolina / "corpus" / "legislative_branch"
    bad_dir = carolina / "corpus" / "social_media"
    good_dir.mkdir(parents=True)
    bad_dir.mkdir(parents=True)
    good_path = good_dir / "LEGa.xml.gz"
    good_path.write_bytes(
        gzip.compress(
            b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><p>texto valido</p></body></text></TEI>',
            mtime=0,
        )
    )
    bad_path = bad_dir / "SOCa.xml.gz"
    bad_path.write_bytes(gzip.compress(b"<TEI><text>", mtime=0))
    _write_raw_manifest(carolina, "carolina", [good_path, bad_path])

    manifest = normalize_source(
        "carolina",
        raw_root=raw_root,
        output_root=tmp_path / "normalized",
        verify_raw=False,
        tool_commit="fixture-commit",
    )
    assert manifest["status"] == "PARTIAL"
    assert manifest["output_document_count"] == 1
    assert any(
        failure.get("raw_source_file", "").endswith("SOCa.xml.gz")
        for failure in manifest["failures"]
    )


def test_malformed_parlamento_record_is_reported_and_later_records_continue(
    tmp_path: Path,
):
    raw_root = tmp_path / "raw"
    source_dir = raw_root / "parlamento"
    source_dir.mkdir(parents=True)
    raw_text = b"record one\n\xff\nrecord three\n"
    source_path = source_dir / "train.txt"
    source_path.write_bytes(raw_text)
    _write_raw_manifest(
        source_dir,
        "parlamento_pt",
        [source_path],
        line_count=raw_text.count(b"\n"),
    )

    manifest = normalize_source(
        "parlamento_pt",
        raw_root=raw_root,
        output_root=tmp_path / "normalized",
        verify_raw=False,
        tool_commit="fixture-tool-commit",
    )
    assert manifest["status"] == "PARTIAL"
    assert manifest["source_document_count"] == 3
    assert manifest["output_document_count"] == 2
    assert [
        row["raw_record_identifier"]
        for row in _read_rows(tmp_path / "normalized", "parlamento")
    ] == [
        "1",
        "3",
    ]
    assert any(
        failure.get("raw_record_identifier") == "2" for failure in manifest["failures"]
    )


def test_characterization_word_totals_percentiles_and_mix_capacities(tmp_path: Path):
    raw_root = _fixture_raw_tree(tmp_path)
    normalized_root = tmp_path / "normalized"
    for source in (
        "gutenberg_pt",
        "parlamento_pt",
        "wikipedia_pt",
        "carolina",
        "gigaverbo_v2",
    ):
        normalize_source(
            source,
            raw_root=raw_root,
            output_root=normalized_root,
            shard_text_bytes=64,
            verify_raw=False,
            tool_commit="fixed-commit",
        )
    project_root = Path(__file__).resolve().parents[2]
    result = characterize_sources(
        normalized_root=normalized_root,
        verify_hashes=False,
        mix_paths=[
            project_root / f"configs/corpus_mix_{letter}.yaml" for letter in "abc"
        ],
    )
    assert result["total_normalized_words"] > 0
    assert result["source_reports"]["parlamento_pt"]["metrics"]["document_count"] == 5
    parl_stats = result["source_reports"]["parlamento_pt"]["metrics"]
    assert parl_stats["document_length_counts"]["empty_normalized_documents"] == 1
    assert parl_stats["word_length_distribution"]["median"] == 2
    assert parl_stats["word_length_distribution"]["p95"] == 2
    assert (
        result["source_reports"]["gigaverbo_v2"]["subsets"]["blogset"]["document_count"]
        == 2
    )
    assert (
        result["candidate_mix_feasibility"]["corpus_mix_a"][
            "max_non_oversampled_total_normalized_words"
        ]
        == 0
    )
    account = normalized_root / "characterization" / "normalized_word_accounting.csv"
    assert account.is_file()
    assert "gigaverbo_v2,blogset,2," in account.read_text(encoding="utf-8")


def test_mix_capacity_uses_configured_source_and_gigaverbo_subset_shares():
    mix_path = Path(__file__).resolve().parents[2] / "configs/corpus_mix_a.yaml"
    config = validate_mix_files([mix_path])[0]
    subset_names = config["sources"]["gigaverbo_v2_residual"]["subsets"]
    reports = {
        "carolina": {"metrics": {"normalized_words": 1_000}},
        "wikipedia_pt": {"metrics": {"normalized_words": 100_000}},
        "parlamento_pt": {"metrics": {"normalized_words": 100_000}},
        "gutenberg_pt": {"metrics": {"normalized_words": 100_000}},
        "gigaverbo_v2": {
            "metrics": {"normalized_words": 10_000_000},
            "subsets": {
                subset: {"normalized_words": 10_000_000} for subset in subset_names
            },
        },
    }

    capacity = _mix_capacities(reports, target_words=2_000, mix_paths=[mix_path])[
        "corpus_mix_a"
    ]
    assert capacity["max_non_oversampled_total_normalized_words"] == 2_500
    assert capacity["limiting_component"] == "carolina"
    assert capacity["oversampling_required_at_target"] is False
    oversubscribed = _mix_capacities(reports, target_words=3_000, mix_paths=[mix_path])[
        "corpus_mix_a"
    ]
    assert oversubscribed["oversampling_required_at_target"] is True


def test_mix_capacity_uses_exact_decimal_floors_and_preserves_gross_limits():
    config_root = Path(__file__).resolve().parents[2] / "configs"
    mix_paths = [
        config_root / "corpus_mix_a.yaml",
        config_root / "corpus_mix_b.yaml",
        config_root / "corpus_mix_c.yaml",
    ]
    subset_words = {
        "blogset": 15_892_975,
        "common_crawl": 584_947_339,
        "crawlPT_dedup": 791_426_699,
        "culturax": 16_942_627,
        "finepdfs_por_Latn": 4_242_157_200,
        "fineweb_2_pt": 5_692_906_792,
        "hplt1_pt": 904_738_680,
        "hplt2_pt": 4_451_301_378,
        "mc4_pt": 2_493_372_382,
        "oscar": 155_105_693,
        "quati": 33_854_779,
    }
    reports = {
        "carolina": {"metrics": {"normalized_words": 1_296_641_822}},
        "wikipedia_pt": {"metrics": {"normalized_words": 413_147_060}},
        "parlamento_pt": {"metrics": {"normalized_words": 396_693_229}},
        "gutenberg_pt": {"metrics": {"normalized_words": 21_054_903}},
        "gigaverbo_v2": {
            "metrics": {"normalized_words": 19_382_646_544},
            "subsets": {
                subset: {"normalized_words": words}
                for subset, words in subset_words.items()
            },
        },
    }
    capacities = _mix_capacities(reports, target_words=None, mix_paths=mix_paths)
    assert (
        capacities["corpus_mix_a"]["max_non_oversampled_total_normalized_words"]
        == 263_186_287
    )
    assert (
        capacities["corpus_mix_b"]["max_non_oversampled_total_normalized_words"]
        == 526_372_575
    )
    assert (
        capacities["corpus_mix_c"]["max_non_oversampled_total_normalized_words"]
        == 423_812_666
    )

    # Binary float multiplication previously floored each of these one word low.
    mix_b_subsets = capacities["corpus_mix_b"]["components"]["gigaverbo_v2"]["subsets"]
    assert (
        mix_b_subsets["crawlPT_dedup"]["subset_capacity_as_total_mix_words"]
        == 19_785_667_475
    )
    assert mix_b_subsets["quati"]["subset_capacity_as_total_mix_words"] == 1_692_738_950
    mix_c_subsets = capacities["corpus_mix_c"]["components"]["gigaverbo_v2"]["subsets"]
    assert (
        mix_c_subsets["hplt2_pt"]["subset_capacity_as_total_mix_words"]
        == 118_701_370_080
    )

    exact_fit_reports = {
        source: {"metrics": {"normalized_words": 1_000_000}}
        for source in ("carolina", "wikipedia_pt", "parlamento_pt", "gutenberg_pt")
    }
    exact_fit_reports["gigaverbo_v2"] = {
        "metrics": {"normalized_words": 1_000_000},
        "subsets": {
            subset: {"normalized_words": 1 if subset == "quati" else 1_000}
            for subset in subset_words
        },
    }
    exact_fit = _mix_capacities(
        exact_fit_reports,
        target_words=50,
        mix_paths=[config_root / "corpus_mix_b.yaml"],
    )["corpus_mix_b"]
    assert exact_fit["oversampling_required_at_target"] is False
    assert (
        exact_fit["components"]["gigaverbo_v2"]["subsets"]["quati"][
            "oversampling_required"
        ]
        is False
    )


def test_characterization_reports_partial_source_failures_without_claiming_capacity(
    tmp_path: Path,
):
    raw_root = _fixture_raw_tree(tmp_path)
    gutenberg_dir = raw_root / "gutenberg"
    malformed_path = gutenberg_dir / "pg103.txt"
    malformed_path.write_bytes(b"invalid utf-8: \xff")
    _write_raw_manifest(
        gutenberg_dir,
        "gutenberg_pt",
        sorted(gutenberg_dir.glob("*.txt")),
    )
    normalized_root = tmp_path / "normalized"
    for source in (
        "gutenberg_pt",
        "parlamento_pt",
        "wikipedia_pt",
        "carolina",
        "gigaverbo_v2",
    ):
        normalize_source(
            source,
            raw_root=raw_root,
            output_root=normalized_root,
            shard_text_bytes=64,
            verify_raw=False,
            tool_commit="fixed-commit",
        )

    result = characterize_sources(normalized_root=normalized_root, verify_hashes=False)
    gutenberg = result["source_reports"]["gutenberg_pt"]
    assert gutenberg["normalization_status"] == "PARTIAL"
    assert gutenberg["metrics"]["normalization_failures"] == 2
    assert result["incomplete_sources"] == ["gutenberg_pt"]
    assert all(
        mix["feasibility_status"] == "not_assessable_incomplete_source_pool"
        and mix["oversampling_required_at_target"] is None
        for mix in result["candidate_mix_feasibility"].values()
    )
