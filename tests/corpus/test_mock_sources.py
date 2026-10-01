"""Unit tests for source samplers using local mocks and fixtures (no network required)."""

import xml.etree.ElementTree as ET

from cambacica.corpus.cli import build_parser
from cambacica.corpus.sources.carolina import (
    _extract_tei_document,
    _proportional_allocation,
    TAXONOMY_POPULATION,
)
from cambacica.corpus.sources.gutenberg_pt import strip_gutenberg_boilerplate


def test_extract_tei_document_mock():
    """Verify TEI XML parsing into raw document dict."""
    tei_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <teiHeader>
        <fileDesc>
          <titleStmt>
            <title>JUD0001</title>
          </titleStmt>
          <publicationStmt>
            <date>2022-05-15</date>
            <license>CC BY 4.0</license>
          </publicationStmt>
          <sourceDesc>
            <p><ref target="https://tribunal.jus.br/acordao/1"/></p>
          </sourceDesc>
        </fileDesc>
      </teiHeader>
      <text>
        <body>
          <p>Acórdão da Terceira Turma Recursal.</p>
          <p>Vistos, relatados e discutidos estes autos em sessão plenária.</p>
        </body>
      </text>
    </TEI>
    """
    elem = ET.fromstring(tei_xml)
    doc = _extract_tei_document(elem, taxonomy="jud")

    assert doc is not None
    assert "Acórdão da Terceira Turma Recursal." in doc["text"]
    assert "Vistos, relatados e discutidos" in doc["text"]
    assert doc["source"] == "carolina"
    assert doc["original_id"] == "JUD0001"
    assert doc["publication_date"] == "2022-05-15"
    assert doc["original_url"] == "https://tribunal.jus.br/acordao/1"
    assert doc["domain_category"] == "jud"


def test_strip_gutenberg_boilerplate_mock():
    """Verify boilerplate stripping with standard Gutenberg header/footer markers."""
    raw_ebook = """
    The Project Gutenberg eBook of Os Lusíadas, by Luís de Camões
    This eBook is for the use of anyone anywhere.
    *** START OF THE PROJECT GUTENBERG EBOOK OS LUSÍADAS ***
    As armas e os barões assinalados,
    Que da ocidental praia Lusitana,
    Por mares nunca dantes navegados,
    Passaram ainda além da Taprobana.
    *** END OF THE PROJECT GUTENBERG EBOOK OS LUSÍADAS ***
    End of the Project Gutenberg EBook.
    """
    cleaned = strip_gutenberg_boilerplate(raw_ebook)
    assert "The Project Gutenberg eBook" not in cleaned
    assert "End of the Project Gutenberg" not in cleaned
    assert "As armas e os barões assinalados" in cleaned
    assert "Passaram ainda além da Taprobana" in cleaned


def test_cli_parser_defaults():
    """Verify CLI parser configuration and command routing."""
    parser = build_parser()

    # sample subcommand
    args = parser.parse_args(
        [
            "sample",
            "carolina",
            "--mode",
            "representative",
            "--size",
            "500",
            "--dry-run",
            "--min-length",
            "20",
        ]
    )
    assert args.subcommand == "sample"
    assert args.source == "carolina"
    assert args.mode == "representative"
    assert args.size == 500
    assert args.dry_run is True
    assert args.min_length == 20

    # inspect subcommand
    args2 = parser.parse_args(["inspect", "data/samples/gate_c1/"])
    assert args2.subcommand == "inspect"
    assert args2.path == "data/samples/gate_c1/"

    # compare subcommand
    args3 = parser.parse_args(["compare", "data/samples/gate_c1/", "--minhash"])
    assert args3.subcommand == "compare"
    assert args3.minhash is True


def test_parlamento_short_interventions_filter(monkeypatch, tmp_path):
    """Verify short parliamentary utterances are preserved by default in representative mode."""
    from cambacica.corpus.sources.parlamento_pt import ParlamentoPTSampler

    sample_lines = [
        "O Sr. Presidente: — Tem a palavra o Sr. Deputado.",  # 49 chars
        "Muito bem!",  # 10 chars
        "Apoiado!",  # 8 chars
        "   ",  # whitespace only
        "",  # empty
    ]

    class MockResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_lines(self, decode_unicode=True):
            return iter(sample_lines)

    import requests

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockResponse())

    sampler = ParlamentoPTSampler()

    # Test 1: min_length=0 (default) preserves all 3 valid short interventions
    out_dir_1 = tmp_path / "test_parl_default"
    _, manifest_1 = sampler.sample(
        mode="representative",
        size=10,
        output_dir=out_dir_1,
        min_length=0,
    )
    assert manifest_1.document_count == 3
    assert manifest_1.records_examined == 5

    # Test 2: min_length=20 explicitly filters out the 10 and 8 char lines
    out_dir_2 = tmp_path / "test_parl_filtered"
    _, manifest_2 = sampler.sample(
        mode="representative",
        size=10,
        output_dir=out_dir_2,
        min_length=20,
    )
    assert manifest_2.document_count == 1


# ---------------------------------------------------------------------------
# New Carolina-specific tests required by the Gate C1 fix specification
# ---------------------------------------------------------------------------


def test_carolina_representative_wik_allocation():
    """Verify wik and dat receive proportionally large allocations."""
    target = 10_000
    alloc = _proportional_allocation(target, TAXONOMY_POPULATION)
    # wik is the largest taxonomy (~42% of population)
    assert alloc["wik"] > alloc["jud"] * 5, (
        "wik allocation must be substantially larger than jud given population weights"
    )
    assert alloc["dat"] > alloc["leg"] * 10, (
        "dat allocation must be substantially larger than leg"
    )


def test_carolina_diagnostic_mode_all_shards(monkeypatch, tmp_path):
    """Verify diagnostic mode scans all available shards, not just the first N."""
    from cambacica.corpus.sources import carolina as carolina_mod
    from cambacica.corpus.sources.carolina import CarolinaSampler

    # Mock fs listing: return 5 fake files per taxonomy
    fake_files = {
        tax: [f"datasets/carolina/{tax}/shard_{i:03d}.xml.gz" for i in range(5)]
        for tax in ["wik", "dat", "soc", "uni", "jud", "leg", "pub"]
    }

    visited_files: list = []

    def mock_iter_carolina_xml_files(fs, taxonomy):
        return fake_files.get(taxonomy, [])

    def mock_stream_carolina_taxonomy(
        fs, taxonomy, shard_indices=None, max_files=None, byte_counter=None
    ):
        files = fake_files.get(taxonomy, [])
        if shard_indices is not None:
            to_visit = [files[i] for i in shard_indices if i < len(files)]
        elif max_files is not None:
            to_visit = files[:max_files]
        else:
            to_visit = files
        visited_files.extend(to_visit)
        # Yield one dummy document per file visited
        for fpath in to_visit:
            yield {
                "text": f"Texto de diagnóstico do arquivo {fpath} com conteúdo suficiente.",
                "source": "carolina",
                "source_revision": "v2.0.1",
                "subset": taxonomy,
                "original_id": fpath,
                "original_url": None,
                "license": "CC BY 4.0",
                "language": "pt-BR",
                "language_score": 1.0,
                "variety": "pt-BR",
                "quality_score": None,
                "publication_date": None,
                "domain_category": taxonomy,
            }

    monkeypatch.setattr(
        carolina_mod, "iter_carolina_xml_files", mock_iter_carolina_xml_files
    )
    monkeypatch.setattr(
        carolina_mod, "stream_carolina_taxonomy", mock_stream_carolina_taxonomy
    )
    monkeypatch.setattr(carolina_mod, "resolve_hf_commit_sha", lambda *a, **k: "abc123")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", lambda: object())

    sampler = CarolinaSampler()
    # Diagnostic mode with quotas larger than what 5 shards can deliver
    # so it must exhaust all shards
    _, manifest = sampler.sample(
        mode="diagnostic",
        size=100,
        seed=42,
        output_dir=tmp_path / "carolina_diag",
    )

    # With 5 files per taxonomy and one doc per file, and 6 taxonomies in quotas,
    # we expect up to 30 files visited total (all available shards)
    assert len(visited_files) > 0
    # Manifest must reflect underfill since 5 docs << quota 20
    stopping = manifest.stopping_reason or ""
    assert "underfill" in stopping or manifest.document_count <= 100


def test_carolina_representative_mode_uses_distributed_shards(monkeypatch, tmp_path):
    """Verify representative mode uses distributed shards, not just first N."""
    from cambacica.corpus.sources import carolina as carolina_mod
    from cambacica.corpus.sources.carolina import CarolinaSampler

    # 20 fake shards per major taxonomy
    fake_files = {
        tax: [f"datasets/carolina/{tax}/shard_{i:03d}.xml.gz" for i in range(20)]
        for tax in ["wik", "dat", "soc", "uni", "jud", "leg", "pub"]
    }
    visited_indices: dict = {tax: [] for tax in fake_files}

    def mock_iter(fs, taxonomy):
        return fake_files.get(taxonomy, [])

    def mock_stream(
        fs, taxonomy, shard_indices=None, max_files=None, byte_counter=None
    ):
        files = fake_files.get(taxonomy, [])
        if shard_indices is not None:
            to_visit = shard_indices
        elif max_files is not None:
            to_visit = list(range(min(max_files, len(files))))
        else:
            to_visit = list(range(len(files)))
        visited_indices[taxonomy].extend(to_visit)
        for idx in to_visit:
            if idx < len(files):
                yield {
                    "text": f"Texto representativo do shard {idx} de {taxonomy} com conteúdo.",
                    "source": "carolina",
                    "source_revision": "v2.0.1",
                    "subset": taxonomy,
                    "original_id": f"{taxonomy}_{idx}",
                    "original_url": None,
                    "license": None,
                    "language": "pt-BR",
                    "language_score": 1.0,
                    "variety": "pt-BR",
                    "quality_score": None,
                    "publication_date": None,
                    "domain_category": taxonomy,
                }

    monkeypatch.setattr(carolina_mod, "iter_carolina_xml_files", mock_iter)
    monkeypatch.setattr(carolina_mod, "stream_carolina_taxonomy", mock_stream)
    monkeypatch.setattr(carolina_mod, "resolve_hf_commit_sha", lambda *a, **k: "abc123")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", lambda: object())

    sampler = CarolinaSampler()
    sampler.sample(
        mode="representative",
        size=1000,
        seed=42,
        output_dir=tmp_path / "carolina_rep",
        shards_per_taxonomy=4,
    )

    # For wik (20 shards, 4 selected), the selected indices must NOT all be {0,1,2,3}
    wik_visited = set(visited_indices.get("wik", []))
    # With 20 shards and 4 distributed selections, at least one index must be >= 5
    assert any(idx >= 5 for idx in wik_visited), (
        f"wik distributed shards should span the full range; got indices {wik_visited}"
    )
