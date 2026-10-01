"""Optional network smoke test for live source streaming.

Disabled by default. Set CAMBACICA_NETWORK_TESTS=1 to run against live endpoints.
Collects at most 2-3 items from one source to verify live connectivity safely.
"""

import os
from pathlib import Path
import pytest

from cambacica.corpus.sources.carolina import CarolinaSampler
from cambacica.corpus.sources.gigaverbo_v2 import GigaVerboSampler
from cambacica.corpus.sources.gutenberg_pt import GutenbergPTSampler
from cambacica.corpus.sources.parlamento_pt import ParlamentoPTSampler
from cambacica.corpus.sources.wikipedia import WikipediaSampler


@pytest.mark.skipif(
    os.environ.get("CAMBACICA_NETWORK_TESTS") != "1",
    reason="Network smoke tests disabled by default. Set CAMBACICA_NETWORK_TESTS=1 to enable.",
)
def test_wikipedia_streaming_smoke(tmp_path: Path):
    """Smoke test streaming 3 articles from Portuguese Wikipedia live."""
    sampler = WikipediaSampler()
    p_path, manifest = sampler.sample(
        mode="representative",
        size=3,
        seed=42,
        output_dir=tmp_path / "smoke_wiki",
        max_stream_items=10,
    )

    assert p_path.is_file()
    assert manifest.document_count == 3
    assert manifest.source == "wikipedia_pt"
    assert manifest.upstream_commit_sha is not None


@pytest.mark.skipif(
    os.environ.get("CAMBACICA_NETWORK_TESTS") != "1",
    reason="Network smoke tests disabled by default. Set CAMBACICA_NETWORK_TESTS=1 to enable.",
)
def test_parlamento_streaming_smoke(tmp_path: Path):
    """Smoke test streaming 3 debate lines from ParlamentoPT live."""
    sampler = ParlamentoPTSampler()
    p_path, manifest = sampler.sample(
        mode="representative",
        size=3,
        seed=42,
        output_dir=tmp_path / "smoke_parlamento",
        max_stream_lines=15,
        min_length=0,
    )

    assert p_path.is_file()
    assert manifest.document_count == 3
    assert manifest.source == "parlamento_pt"
    assert manifest.upstream_commit_sha is not None


@pytest.mark.skipif(
    os.environ.get("CAMBACICA_NETWORK_TESTS") != "1",
    reason="Network smoke tests disabled by default. Set CAMBACICA_NETWORK_TESTS=1 to enable.",
)
def test_carolina_streaming_smoke(tmp_path: Path):
    """Smoke test streaming documents from Corpus Carolina live (proportional mode)."""
    sampler = CarolinaSampler()
    p_path, manifest = sampler.sample(
        mode="representative",
        size=5,
        seed=42,
        output_dir=tmp_path / "smoke_carolina",
        shards_per_taxonomy=1,
    )

    assert p_path.is_file()
    assert manifest.document_count >= 1
    assert manifest.source == "carolina"
    assert manifest.upstream_commit_sha is not None
    # Manifest must record taxonomy allocation
    assert "taxonomy_allocation" in manifest.stats


@pytest.mark.skipif(
    os.environ.get("CAMBACICA_NETWORK_TESTS") != "1",
    reason="Network smoke tests disabled by default. Set CAMBACICA_NETWORK_TESTS=1 to enable.",
)
def test_gigaverbo_streaming_smoke(tmp_path: Path):
    """Smoke test streaming 2 documents from GigaVerbo-v2 live."""
    sampler = GigaVerboSampler()
    p_path, manifest = sampler.sample(
        mode="candidate",
        size=2,
        seed=42,
        output_dir=tmp_path / "smoke_gv",
        num_shards_to_visit=1,
    )

    assert p_path.is_file()
    assert manifest.document_count == 2
    assert manifest.source == "gigaverbo_v2"
    assert manifest.upstream_commit_sha is not None


@pytest.mark.skipif(
    os.environ.get("CAMBACICA_NETWORK_TESTS") != "1",
    reason="Network smoke tests disabled by default. Set CAMBACICA_NETWORK_TESTS=1 to enable.",
)
def test_gutenberg_streaming_smoke(tmp_path: Path):
    """Smoke test downloading 2 Portuguese books from Project Gutenberg live."""
    sampler = GutenbergPTSampler()
    p_path, manifest = sampler.sample(
        mode="representative",
        size=2,
        seed=42,
        output_dir=tmp_path / "smoke_gutenberg",
        max_attempts=3,
    )

    assert p_path.is_file()
    assert manifest.document_count == 2
    assert manifest.source == "gutenberg_pt"
    assert manifest.stats.get("selected_ebook_ids") is not None
