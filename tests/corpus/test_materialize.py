"""Unit tests for corpus materialization infrastructure.

Tests the materialization contract, manifest serialization, atomic writes,
SHA-256 computation, safety boundaries, resume behavior, and verification.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cambacica.corpus.cli import main
from cambacica.corpus.manifest import compute_file_sha256
from cambacica.corpus.materialize import (
    BaseMaterializer,
    GenericStubMaterializer,
    GutenbergMaterializer,
    MaterializationManifest,
    MaterializedFileRecord,
)


def test_materialized_file_record():
    """Test MaterializedFileRecord serialization."""
    rec = MaterializedFileRecord(
        relative_path="pg1234.txt",
        upstream_identifier="1234",
        url="https://www.gutenberg.org/cache/epub/1234/pg1234.txt",
        bytes=1024,
        sha256="abc123def456",
        checksum_source="local_sha256",
    )
    d = rec.to_dict()
    assert d["relative_path"] == "pg1234.txt"
    assert d["bytes"] == 1024
    assert d["sha256"] == "abc123def456"
    assert d["checksum_source"] == "local_sha256"


def test_materialization_manifest_lifecycle(tmp_path: Path):
    """Test saving, loading, and verifying a MaterializationManifest."""
    manifest_file = tmp_path / "manifest.json"
    dummy_file = tmp_path / "pg100.txt"
    dummy_file.write_text("Hello Portuguese literature\n", encoding="utf-8")

    dummy_sha = compute_file_sha256(dummy_file)
    dummy_bytes = dummy_file.stat().st_size

    manifest = MaterializationManifest(
        source="gutenberg_pt",
        upstream_repository="project_gutenberg_pt",
        pinned_revision="snapshot_2026-10-01",
        snapshot_date="2026-10-01",
        catalog_snapshot_date="2026-10-01",
        catalog_query="browse/languages/pt",
        catalog_size_ebooks=1,
        status="COMPLETE",
        total_files=1,
        total_bytes=dummy_bytes,
        ebook_ids=[100],
        files=[
            {
                "relative_path": "pg100.txt",
                "upstream_identifier": "100",
                "url": "https://www.gutenberg.org/cache/epub/100/pg100.txt",
                "bytes": dummy_bytes,
                "sha256": dummy_sha,
                "checksum_source": "local_sha256",
            }
        ],
    )
    manifest.save(manifest_file)
    assert manifest_file.is_file()

    loaded = MaterializationManifest.load(manifest_file)
    assert loaded.source == "gutenberg_pt"
    assert loaded.status == "COMPLETE"
    assert loaded.total_files == 1
    assert loaded.total_bytes == dummy_bytes

    is_valid, errors = loaded.verify(tmp_path)
    assert is_valid
    assert len(errors) == 0


def test_materialization_manifest_verification_failures(tmp_path: Path):
    """Test detection of missing files, checksum mismatches, and orphan partial files."""
    manifest_file = tmp_path / "manifest.json"
    dummy_file = tmp_path / "pg200.txt"
    dummy_file.write_text("Corpus content\n", encoding="utf-8")

    dummy_sha = compute_file_sha256(dummy_file)
    dummy_bytes = dummy_file.stat().st_size

    manifest = MaterializationManifest(
        source="gutenberg_pt",
        upstream_repository="project_gutenberg_pt",
        pinned_revision="snapshot_2026-10-01",
        status="COMPLETE",
        total_files=2,
        total_bytes=dummy_bytes + 500,
        files=[
            {
                "relative_path": "pg200.txt",
                "upstream_identifier": "200",
                "url": "https://example.com/200",
                "bytes": dummy_bytes,
                "sha256": dummy_sha,
            },
            {
                "relative_path": "pg201.txt",
                "upstream_identifier": "201",
                "url": "https://example.com/201",
                "bytes": 500,
                "sha256": "deadbeef" * 8,
            },
        ],
    )
    manifest.save(manifest_file)

    # Missing file pg201.txt should cause failure
    is_valid, errors = manifest.verify(tmp_path)
    assert not is_valid
    assert any("Missing file on disk: pg201.txt" in e for e in errors)

    # Corrupting pg200.txt sha/bytes
    dummy_file.write_text("Modified corrupted content\n", encoding="utf-8")
    is_valid, errors = manifest.verify(tmp_path)
    assert not is_valid
    assert any("SHA-256 mismatch for pg200.txt" in e for e in errors)

    # Orphan .partial file detection
    orphan_partial = tmp_path / "pg999.txt.partial"
    orphan_partial.write_text("partial unfinished content\n", encoding="utf-8")
    is_valid, errors = manifest.verify(tmp_path)
    assert not is_valid
    assert any("Orphaned partial file detected" in e for e in errors)


def test_base_materializer_safety_boundary():
    """Test that materializer rejects destinations outside the raw storage root."""
    with pytest.raises(ValueError, match="violates safety boundary"):
        BaseMaterializer(
            source_name="carolina",
            destination_override="/tmp/unauthorized_destination",
            allow_custom_destination=False,
        )


def test_generic_stub_materializer_refuses_materialize(tmp_path: Path):
    """Test that non-pilot sources refuse live materialization."""
    mat = GenericStubMaterializer(
        source_name="carolina",
        destination_override=tmp_path / "carolina",
        allow_custom_destination=True,
    )
    plan = mat.plan()
    assert plan["source"] == "carolina"
    assert plan["repository"] == "carolina-c4ai/corpus-carolina"
    assert plan["pinned_revision"] == "v2.0.1"

    with pytest.raises(NotImplementedError, match="restricted to Gutenberg"):
        mat.materialize()


def test_gutenberg_materializer_plan(tmp_path: Path):
    """Test GutenbergMaterializer dry-run plan."""
    mat = GutenbergMaterializer(
        destination_override=tmp_path / "gutenberg",
        allow_custom_destination=True,
    )
    plan = mat.plan()
    assert plan["source"] == "gutenberg_pt"
    assert plan["pinned_revision"] == "snapshot_2026-10-01"
    assert plan["snapshot_date"] == "2026-10-01"
    assert plan["expected_catalog_size"] == 655
    assert plan["destination"] == str(tmp_path / "gutenberg")


def test_gutenberg_provenance_mismatch_detection(tmp_path: Path):
    """Test that unexpected catalog IDs raise a provenance mismatch error."""
    mat = GutenbergMaterializer(
        destination_override=tmp_path / "gutenberg",
        allow_custom_destination=True,
    )
    with patch(
        "cambacica.corpus.sources.gutenberg_pt.discover_gutenberg_pt_ids",
        return_value=[1, 2, 3],  # Wrong count: 3 instead of 655
    ):
        with pytest.raises(RuntimeError, match="Provenance mismatch"):
            mat.resolve_ebook_ids(verify_live=True)


def test_gutenberg_materialize_mocked(tmp_path: Path):
    """Test end-to-end Gutenberg materialization with mocked downloads."""
    dest_dir = tmp_path / "gutenberg"
    mat = GutenbergMaterializer(
        destination_override=dest_dir,
        allow_custom_destination=True,
    )

    test_ids = [2837, 3333, 7384]

    mock_text_content = {
        2837: b"*** START OF THE PROJECT GUTENBERG EBOOK TEST1 ***\nRaw text 1\n",
        3333: b"*** START OF THE PROJECT GUTENBERG EBOOK TEST2 ***\nRaw text 2\n",
        7384: b"*** START OF THE PROJECT GUTENBERG EBOOK TEST3 ***\nRaw text 3\n",
    }

    def mock_get(url, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        for bid, text_bytes in mock_text_content.items():
            if f"/{bid}/" in url:
                mock_resp.iter_content.return_value = [text_bytes]
                return mock_resp
        mock_resp.status_code = 404
        return mock_resp

    with patch("requests.Session.get", side_effect=mock_get):
        manifest = mat.materialize(
            concurrency=2,
            timeout=5,
            max_retries=1,
            ebook_ids=test_ids,
            verify_live=False,
        )

    assert manifest.status == "COMPLETE"
    assert manifest.total_files == 3
    assert len(manifest.files) == 3
    assert len(manifest.failed_ids) == 0

    # Verify all 3 files exist and are RAW (containing boilerplate)
    for bid in test_ids:
        p = dest_dir / f"pg{bid}.txt"
        assert p.is_file()
        content = p.read_bytes()
        assert content == mock_text_content[bid]
        assert b"START OF THE PROJECT GUTENBERG EBOOK" in content

    # Verify manifest and metadata files on disk
    manifest_path = dest_dir / "manifest.json"
    ebook_ids_path = dest_dir / "ebook_ids.json"
    assert manifest_path.is_file()
    assert ebook_ids_path.is_file()

    with ebook_ids_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["ebook_ids"] == test_ids

    # Verification passes
    is_valid, errors = mat.verify()
    assert is_valid
    assert len(errors) == 0

    # Idempotency check: re-running materialization performs zero downloads
    with patch("requests.Session.get") as mock_get_second:
        second_manifest = mat.materialize(
            concurrency=2,
            timeout=5,
            max_retries=1,
            ebook_ids=test_ids,
            verify_live=False,
        )
        assert mock_get_second.call_count == 0
        assert second_manifest.status == "COMPLETE"
        assert second_manifest.total_files == 3


def test_cli_materialize_commands(tmp_path: Path):
    """Test CLI subcommands for materialize."""
    # Dry-run
    ret = main(["materialize", "gutenberg", "--dry-run"])
    assert ret == 0

    # Non-pilot source
    ret = main(["materialize", "carolina"])
    assert ret == 1

    # Verify-only on empty directory
    empty_dir = tmp_path / "empty_gutenberg"
    ret = main(
        [
            "materialize",
            "gutenberg",
            "--verify-only",
            "--destination",
            str(empty_dir),
            "--allow-custom-destination",
        ]
    )
    assert ret == 1
