"""Unit tests for GigaVerbo-v2 exclusion configuration, subset matching,
and audit vs candidate sampling frame design."""

from pathlib import Path

import pytest

from cambacica.corpus.sources.gigaverbo_v2 import (
    is_subset_excluded,
    load_gigaverbo_exclusions,
)


def test_load_gigaverbo_exclusions():
    """Verify loading blocked subsets from real YAML config."""
    config_path = Path("configs/gigaverbo_exclusions.yaml")
    exact, patterns, config_sha = load_gigaverbo_exclusions(config_path)

    assert len(exact) > 0
    assert len(patterns) > 0
    assert len(config_sha) == 64

    # Verify key blocked subsets are detected
    assert "dolly-15k-libretranslate-pt" in exact
    assert "bactrian-x" in exact
    assert "ultrachatbr" in exact
    assert "xlsum" in exact
    assert "corpus-carolina" in exact
    assert "wikipedia" in exact
    assert "bdtd" in exact


def test_is_subset_excluded():
    """Verify exact and wildcard matching for exclusions."""
    exact = {"dolly_15k_libretranslate_pt", "xlsum", "corpus_carolina"}
    patterns = ["*bactrian*", "*ultrachat*"]

    # Exact matches
    assert is_subset_excluded("dolly_15k_libretranslate_pt", exact, patterns)
    assert is_subset_excluded("XLSUM", exact, patterns)  # Case insensitive
    assert is_subset_excluded("corpus_carolina", exact, patterns)

    # Wildcard matches
    assert is_subset_excluded("MBZUAI/Bactrian-X", exact, patterns)
    assert is_subset_excluded("recogna-nlp/UltrachatBR", exact, patterns)

    # Allowed subsets (web residual)
    assert not is_subset_excluded("fineweb_2_pt", exact, patterns)
    assert not is_subset_excluded("mc4_pt", exact, patterns)
    assert not is_subset_excluded("CC-MAIN-2025-30", exact, patterns)
    assert not is_subset_excluded(None, exact, patterns)


def test_gigaverbo_audit_visits_all_shards(monkeypatch, tmp_path):
    """Verify that audit mode visits ALL available shards, not just a subset."""
    from cambacica.corpus.sources import gigaverbo_v2 as gv_mod
    from cambacica.corpus.sources.gigaverbo_v2 import GigaVerboSampler

    shards_visited: list = []

    class MockFS:
        def ls(self, path, detail=False):
            return [f"{path}/shard-{i:05d}-of-00056.parquet" for i in range(56)]

        def open(self, path, mode="rb"):
            return MockParquetContextManager(path, shards_visited)

    class MockParquetContextManager:
        def __init__(self, path, tracker):
            self.path = path
            self.tracker = tracker

        def __enter__(self):
            self.tracker.append(self.path)
            return self

        def __exit__(self, *args):
            pass

    import pyarrow.parquet as pq

    def mock_parquet_file(f):
        class _MockPF:
            def iter_batches(self, batch_size=None, columns=None):
                # Return one batch with one valid record
                import pyarrow as pa
                texts = ["Este é um texto de exemplo para o GigaVerbo com comprimento adequado."]
                ids = ["id_1"]
                sources = ["fineweb_2_pt"]
                subsets = ["fineweb_2_pt"]
                scores = [3.5]
                batch = pa.RecordBatch.from_pydict({
                    "text": texts, "id": ids, "source": sources,
                    "subset": subsets, "edu_score": scores,
                })
                return iter([batch])

        return _MockPF()

    monkeypatch.setattr(gv_mod, "resolve_hf_commit_sha", lambda *a, **k: "testsha")
    monkeypatch.setattr(pq, "ParquetFile", mock_parquet_file)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfFileSystem", lambda: MockFS())

    sampler = GigaVerboSampler()
    _, manifest = sampler.sample(
        mode="audit",
        size=100,
        seed=42,
        output_dir=tmp_path / "gv_audit",
    )

    # Audit must have attempted all 56 shards
    assert manifest.stats.get("shards_visited", 0) == 56, (
        f"Audit mode must visit all 56 shards, got {manifest.stats.get('shards_visited')}"
    )


def test_gigaverbo_candidate_applies_exclusions_before_sampling(monkeypatch, tmp_path):
    """Verify that candidate mode applies exclusion policy before accepting docs."""
    from cambacica.corpus.sources import gigaverbo_v2 as gv_mod
    from cambacica.corpus.sources.gigaverbo_v2 import GigaVerboSampler

    class MockFS:
        def ls(self, path, detail=False):
            return [f"{path}/shard-{i:05d}-of-00056.parquet" for i in range(56)]

        def open(self, path, mode="rb"):
            return MockParquetCtx(path)

    class MockParquetCtx:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    import pyarrow.parquet as pq
    import pyarrow as pa

    def mock_parquet_file(f):
        class _MockPF:
            def iter_batches(self, batch_size=None, columns=None):
                # Mix: one excluded subset, one allowed
                batch = pa.RecordBatch.from_pydict({
                    "text": [
                        "Texto excluído do dataset dolly libretranslate para teste.",
                        "Texto permitido do dataset FineWeb-2 para treino em português.",
                    ],
                    "id": ["excl_1", "allowed_1"],
                    "source": ["dolly_15k_libretranslate_pt", "fineweb_2_pt"],
                    "subset": ["dolly_15k_libretranslate_pt", "fineweb_2_pt"],
                    "edu_score": [1.0, 3.5],
                })
                return iter([batch])

        return _MockPF()

    monkeypatch.setattr(gv_mod, "resolve_hf_commit_sha", lambda *a, **k: "testsha")
    monkeypatch.setattr(pq, "ParquetFile", mock_parquet_file)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfFileSystem", lambda: MockFS())

    sampler = GigaVerboSampler()
    _, manifest = sampler.sample(
        mode="candidate",
        size=100,
        seed=42,
        output_dir=tmp_path / "gv_candidate",
    )

    # The excluded subset must be counted in exclusion_counts_by_subset
    excl_counts = manifest.stats.get("exclusion_counts_by_subset", {})
    assert any("dolly" in k.lower() for k in excl_counts), (
        f"Expected dolly to appear in exclusion_counts, got: {excl_counts}"
    )

    # The allowed subset must appear in subsets_after_exclusion
    after = manifest.stats.get("subsets_after_exclusion", {})
    assert any("fineweb" in k.lower() for k in after), (
        f"Expected fineweb_2_pt to appear in subsets_after_exclusion, got: {after}"
    )

    # Subsets before exclusion must include the excluded one too
    before = manifest.stats.get("subsets_before_exclusion", {})
    assert any("dolly" in k.lower() for k in before), (
        f"Expected dolly to appear in subsets_before_exclusion, got: {before}"
    )
