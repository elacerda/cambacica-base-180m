"""Unit tests for provenance manifest generation and serialization."""

from pathlib import Path

from cambacica.corpus.manifest import (
    ProvenanceManifest,
    compute_file_sha256,
)


def test_manifest_roundtrip(tmp_path: Path):
    """Verify JSON serialization and deserialization of ProvenanceManifest."""
    dummy_file = tmp_path / "dummy.parquet"
    dummy_file.write_bytes(b"PAR1dummy_parquet_bytes")

    file_sha = compute_file_sha256(dummy_file)
    manifest = ProvenanceManifest(
        source="carolina",
        mode="representative",
        target_size=10000,
        document_count=9998,
        seed=42,
        upstream_identifier="carolina-c4ai/corpus-carolina",
        upstream_revision="v2.0.1",
        upstream_commit_sha="55e63a519393c70a48dcfa14a558499c6bb0583b",
        upstream_url="https://huggingface.co/datasets/carolina-c4ai/corpus-carolina",
        upstream_configuration="corpus",
        population_scope="823 xml.gz shards",
        sampling_frame="deterministic_shard_partition",
        records_examined=12000,
        bytes_read=15000000,
        stopping_reason="target_size_reached",
        output_parquet="representative.parquet",
        parquet_sha256=file_sha,
        parquet_bytes=dummy_file.stat().st_size,
        stats={"total_chars": 500000, "total_words": 80000},
    )

    manifest_file = tmp_path / "manifest.json"
    manifest.save(manifest_file)
    assert manifest_file.is_file()

    loaded = ProvenanceManifest.load(manifest_file)
    assert loaded.source == "carolina"
    assert loaded.mode == "representative"
    assert loaded.document_count == 9998
    assert loaded.parquet_sha256 == file_sha
    assert loaded.upstream_commit_sha == "55e63a519393c70a48dcfa14a558499c6bb0583b"
    assert loaded.population_scope == "823 xml.gz shards"
    assert loaded.sampling_frame == "deterministic_shard_partition"
    assert loaded.records_examined == 12000
    assert loaded.bytes_read == 15000000
    assert loaded.stopping_reason == "target_size_reached"
    assert loaded.stats["total_chars"] == 500000
