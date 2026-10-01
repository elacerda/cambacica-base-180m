"""Unit tests for GigaVerbo-v2 exclusion configuration, subset matching,
and audit vs candidate sampling frame design."""

from pathlib import Path

from cambacica.corpus.sources.gigaverbo_v2 import (
    is_subset_excluded,
    load_gigaverbo_exclusions,
    select_distributed_row_groups,
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

                texts = [
                    "Este é um texto de exemplo para o GigaVerbo com comprimento adequado."
                ]
                ids = ["id_1"]
                sources = ["fineweb_2_pt"]
                subsets = ["fineweb_2_pt"]
                scores = [3.5]
                batch = pa.RecordBatch.from_pydict(
                    {
                        "text": texts,
                        "id": ids,
                        "source": sources,
                        "subset": subsets,
                        "edu_score": scores,
                    }
                )
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
                batch = pa.RecordBatch.from_pydict(
                    {
                        "text": [
                            "Texto excluído do dataset dolly libretranslate para teste.",
                            "Texto permitido do dataset FineWeb-2 para treino em português.",
                        ],
                        "id": ["excl_1", "allowed_1"],
                        "source": ["dolly_15k_libretranslate_pt", "fineweb_2_pt"],
                        "subset": ["dolly_15k_libretranslate_pt", "fineweb_2_pt"],
                        "edu_score": [1.0, 3.5],
                    }
                )
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


def test_select_distributed_row_groups_determinism():
    """Verify that row-group selection is deterministic across repeated runs."""
    res1 = select_distributed_row_groups(
        num_row_groups=22, n_groups=4, seed=42, shard_name="shard_0"
    )
    res2 = select_distributed_row_groups(
        num_row_groups=22, n_groups=4, seed=42, shard_name="shard_0"
    )
    assert res1 == res2
    assert len(res1) == 4


def test_select_distributed_row_groups_does_not_always_choose_zero():
    """Verify that selection does not degenerate to picking row group 0."""
    res = select_distributed_row_groups(
        num_row_groups=22, n_groups=4, seed=42, shard_name="train-00000"
    )
    assert 0 not in res or any(idx >= 10 for idx in res)
    assert res != [0, 1, 2, 3]


def test_select_distributed_row_groups_spans_shard():
    """Verify that selected row groups span early, middle, and late physical regions."""
    num_rg = 30
    n_groups = 5
    res = select_distributed_row_groups(
        num_row_groups=num_rg, n_groups=n_groups, seed=42, shard_name="shard_x"
    )
    assert len(res) == n_groups
    assert min(res) < num_rg // 3, "Expected at least one early row group"
    assert any((num_rg // 3) <= x < (2 * num_rg // 3) for x in res), (
        "Expected at least one middle row group"
    )
    assert max(res) >= (2 * num_rg // 3), "Expected at least one late row group"


def test_candidate_exclusions_before_reservoir_and_stats_recorded(
    monkeypatch, tmp_path
):
    """Verify exclusions happen before reservoir insertion and stats are recorded."""
    from cambacica.corpus.sources import gigaverbo_v2 as gv_mod
    from cambacica.corpus.sources.gigaverbo_v2 import GigaVerboSampler
    import pyarrow as pa
    import pyarrow.parquet as pq

    class MockFS:
        def ls(self, path, detail=False):
            return [f"{path}/shard-00000-of-00056.parquet"]

        def open(self, path, mode="rb"):
            return MockParquetCtx(path)

    class MockParquetCtx:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def mock_parquet_file(f):
        class _MockPF:
            num_row_groups = 2

            def iter_batches(self, batch_size=None, row_groups=None, columns=None):
                batch = pa.RecordBatch.from_pydict(
                    {
                        "text": [
                            "Texto em português do dataset bactrianx que deve ser excluído pelo filtro.",
                            "Texto em português do FineWeb-2 que deve ser aceito normalmente no treino.",
                        ],
                        "id": ["bactrian_1", "fw2_1"],
                        "source": ["bactrianx", "fineweb_2_pt"],
                        "subset": ["bactrianx", "fineweb_2_pt"],
                        "edu_score": [1.0, 4.0],
                    }
                )
                return iter([batch])

        return _MockPF()

    monkeypatch.setattr(gv_mod, "resolve_hf_commit_sha", lambda *a, **k: "testsha")
    monkeypatch.setattr(pq, "ParquetFile", mock_parquet_file)
    monkeypatch.setattr(gv_mod, "HfFileSystem", lambda: MockFS())

    sampler = GigaVerboSampler()
    p_path, manifest = sampler.sample(
        mode="candidate",
        size=10,
        seed=42,
        output_dir=tmp_path / "gv_cand_test",
        row_groups_per_shard=2,
    )

    tbl = pq.read_table(p_path)
    subsets_in_sample = set(tbl["subset"].to_pylist())
    assert "bactrianx" not in subsets_in_sample
    assert "fineweb_2_pt" in subsets_in_sample

    stats = manifest.stats
    assert stats["total_excluded"] >= 1
    assert "bactrianx" in stats["exclusion_counts_by_subset"]
    assert "bactrianx" in stats["exclusion_rules_matched"]
    assert "fineweb_2_pt" in stats["subsets_after_exclusion"]


def test_audit_mode_does_not_apply_exclusions(monkeypatch, tmp_path):
    """Verify that audit mode accepts all subsets without applying exclusions."""
    from cambacica.corpus.sources import gigaverbo_v2 as gv_mod
    from cambacica.corpus.sources.gigaverbo_v2 import GigaVerboSampler
    import pyarrow as pa
    import pyarrow.parquet as pq

    class MockFS:
        def ls(self, path, detail=False):
            return [f"{path}/shard-00000-of-00056.parquet"]

        def open(self, path, mode="rb"):
            return MockParquetCtx(path)

    class MockParquetCtx:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def mock_parquet_file(f):
        class _MockPF:
            num_row_groups = 2

            def iter_batches(self, batch_size=None, row_groups=None, columns=None):
                rg = row_groups[0] if row_groups else 0
                batch = pa.RecordBatch.from_pydict(
                    {
                        "text": [
                            f"Texto em português do dataset bactrianx que em audit NÃO deve ser excluído rg {rg}.",
                            f"Texto em português do FineWeb-2 que deve ser aceito normalmente no teste rg {rg}.",
                        ],
                        "id": [f"bactrian_{rg}", f"fw2_{rg}"],
                        "source": ["bactrianx", "fineweb_2_pt"],
                        "subset": ["bactrianx", "fineweb_2_pt"],
                        "edu_score": [1.0, 4.0],
                    }
                )
                return iter([batch])

        return _MockPF()

    monkeypatch.setattr(gv_mod, "resolve_hf_commit_sha", lambda *a, **k: "testsha")
    monkeypatch.setattr(pq, "ParquetFile", mock_parquet_file)
    monkeypatch.setattr(gv_mod, "HfFileSystem", lambda: MockFS())

    sampler = GigaVerboSampler()
    p_path, manifest = sampler.sample(
        mode="audit",
        size=10,
        seed=42,
        output_dir=tmp_path / "gv_audit_test",
        row_groups_per_shard=2,
    )

    tbl = pq.read_table(p_path)
    subsets_in_sample = set(tbl["subset"].to_pylist())
    assert "bactrianx" in subsets_in_sample
    assert "fineweb_2_pt" in subsets_in_sample
    assert manifest.stats.get("total_excluded", 0) == 0
