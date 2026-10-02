"""Unit tests for corpus materialization infrastructure.

Tests the materialization contract, manifest serialization, atomic writes,
SHA-256 computation, safety boundaries, resume behavior, and verification.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
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
    _atomic_write_json,
    _get_clean_tool_git_commit,
    ParlamentoMaterializer,
    WikipediaMaterializer,
    CarolinaMaterializer,
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


def test_materialization_metadata_writes_are_atomic(tmp_path: Path):
    """A failed replace leaves the last valid JSON metadata file untouched."""
    manifest_path = tmp_path / "manifest.json"
    manifest = MaterializationManifest(
        source="gutenberg_pt",
        upstream_repository="project_gutenberg_pt",
        pinned_revision="snapshot_2026-10-01",
        status="COMPLETE",
    )
    manifest.save(manifest_path)
    original_manifest = manifest_path.read_bytes()

    manifest.status = "FAILED"
    with patch(
        "cambacica.corpus.materialize.os.replace", side_effect=OSError("simulated kill")
    ):
        with pytest.raises(OSError, match="simulated kill"):
            manifest.save(manifest_path)
    assert manifest_path.read_bytes() == original_manifest
    assert (
        json.loads((tmp_path / "manifest.json.partial").read_text())["status"]
        == "FAILED"
    )

    ids_path = tmp_path / "ebook_ids.json"
    _atomic_write_json({"ebook_ids": [1]}, ids_path)
    original_ids = ids_path.read_bytes()
    with patch(
        "cambacica.corpus.materialize.os.replace", side_effect=OSError("simulated kill")
    ):
        with pytest.raises(OSError, match="simulated kill"):
            _atomic_write_json({"ebook_ids": [2]}, ids_path)
    assert ids_path.read_bytes() == original_ids


def test_clean_git_provenance_refuses_dirty_tree():
    """Production provenance must identify a clean committed materializer."""
    with patch(
        "cambacica.corpus.materialize.subprocess.run",
        side_effect=[
            SimpleNamespace(returncode=0, stdout="a" * 40 + "\n", stderr=""),
            SimpleNamespace(
                returncode=0,
                stdout=" M src/cambacica/corpus/materialize.py\n",
                stderr="",
            ),
        ],
    ):
        with pytest.raises(RuntimeError, match="dirty working tree"):
            _get_clean_tool_git_commit()


def test_truncated_manifest_is_reported_and_preserved(tmp_path: Path):
    """A malformed canonical manifest is neither trusted nor overwritten."""
    destination = tmp_path / "gutenberg"
    destination.mkdir()
    manifest_path = destination / "manifest.json"
    manifest_path.write_text('{"status":', encoding="utf-8")
    mat = GutenbergMaterializer(
        destination_override=destination,
        allow_custom_destination=True,
    )

    valid, errors = mat.verify()
    assert not valid
    assert any("valid JSON manifest" in error for error in errors)

    with patch(
        "cambacica.corpus.materialize._get_clean_tool_git_commit",
        return_value="test-commit",
    ):
        with pytest.raises(RuntimeError, match="invalid canonical manifest"):
            mat.materialize(ebook_ids=[100], verify_live=False)
    assert manifest_path.read_text(encoding="utf-8") == '{"status":'


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
    """Live catalog changes cannot replace the frozen snapshot ID set."""
    mat = GutenbergMaterializer(
        destination_override=tmp_path / "gutenberg",
        allow_custom_destination=True,
    )
    frozen_ids = mat.resolve_ebook_ids(verify_live=False)
    changed_live_ids = frozen_ids.copy()
    changed_live_ids[0] = max(frozen_ids) + 1
    changed_live_ids.sort()
    with patch(
        "cambacica.corpus.sources.gutenberg_pt.discover_gutenberg_pt_ids",
        return_value=changed_live_ids,
    ):
        with pytest.raises(RuntimeError, match="Provenance mismatch"):
            mat.resolve_ebook_ids(verify_live=True)
    assert mat.resolve_ebook_ids(verify_live=False) == frozen_ids


def test_gutenberg_snapshot_is_required_even_when_live_catalog_is_available(
    tmp_path: Path,
):
    """The live catalog cannot become the accepted ID source without a snapshot."""
    mat = GutenbergMaterializer(
        destination_override=tmp_path / "gutenberg",
        allow_custom_destination=True,
    )
    mat.snapshot_config_path = "configs/gutenberg_pt_snapshot_missing.json"
    with patch(
        "cambacica.corpus.sources.gutenberg_pt.discover_gutenberg_pt_ids",
        return_value=list(range(1, 656)),
    ) as live_catalog:
        with pytest.raises(RuntimeError, match="not tracked by Git"):
            mat.resolve_ebook_ids(verify_live=True)
        live_catalog.assert_not_called()


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
        mock_resp.url = url
        for bid, text_bytes in mock_text_content.items():
            if f"/{bid}/" in url:
                if bid == 2837 and "cache/epub" in url:
                    mock_resp.status_code = 404
                    return mock_resp
                mock_resp.iter_content.return_value = [text_bytes]
                return mock_resp
        mock_resp.status_code = 404
        return mock_resp

    with (
        patch("requests.Session.get", side_effect=mock_get),
        patch(
            "cambacica.corpus.materialize._get_clean_tool_git_commit",
            return_value="test-commit",
        ),
    ):
        manifest = mat.materialize(
            concurrency=2,
            timeout=5,
            max_retries=1,
            ebook_ids=test_ids,
            verify_live=False,
        )

    assert manifest.status == "COMPLETE"
    assert manifest.tool_git_commit == "test-commit"
    assert manifest.total_files == 3
    assert len(manifest.files) == 3
    assert len(manifest.failed_ids) == 0
    url_2837 = next(
        record["url"]
        for record in manifest.files
        if record["upstream_identifier"] == "2837"
    )
    assert url_2837 == "https://www.gutenberg.org/files/2837/2837-0.txt"

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
    with (
        patch("requests.Session.get") as mock_get_second,
        patch(
            "cambacica.corpus.materialize._get_clean_tool_git_commit",
            return_value="test-commit",
        ),
    ):
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
        assert (
            next(
                record["url"]
                for record in second_manifest.files
                if record["upstream_identifier"] == "2837"
            )
            == url_2837
        )


def test_corrupt_existing_payload_is_never_blessed(tmp_path: Path):
    """Verify-only and normal reruns both reject mutated cached bytes."""
    destination = tmp_path / "gutenberg"
    mat = GutenbergMaterializer(
        destination_override=destination,
        allow_custom_destination=True,
    )
    payload = b"trusted raw text\n"

    def successful_get(url, **kwargs):
        response = MagicMock()
        response.status_code = 200
        response.url = url
        response.iter_content.return_value = [payload]
        return response

    with (
        patch("requests.Session.get", side_effect=successful_get),
        patch(
            "cambacica.corpus.materialize._get_clean_tool_git_commit",
            return_value="test-commit",
        ),
    ):
        first = mat.materialize(
            ebook_ids=[100],
            verify_live=False,
            max_retries=1,
        )
    assert first.status == "COMPLETE"
    manifest_path = destination / "manifest.json"
    original_manifest = manifest_path.read_bytes()

    payload_path = destination / "pg100.txt"
    mutated = bytearray(payload_path.read_bytes())
    mutated[0] ^= 1
    payload_path.write_bytes(mutated)

    valid, errors = mat.verify()
    assert not valid
    assert any("SHA-256 mismatch for pg100.txt" in error for error in errors)
    verify_ret = main(
        [
            "materialize",
            "gutenberg",
            "--verify-only",
            "--destination",
            str(destination),
            "--allow-custom-destination",
        ]
    )
    assert verify_ret == 1

    with (
        patch("requests.Session.get") as mock_get,
        patch(
            "cambacica.corpus.materialize._get_clean_tool_git_commit",
            return_value="test-commit",
        ),
    ):
        with pytest.raises(RuntimeError, match="failed integrity verification"):
            mat.materialize(
                ebook_ids=[100],
                verify_live=False,
                max_retries=1,
            )
        assert mock_get.call_count == 0
    assert manifest_path.read_bytes() == original_manifest


def test_failed_materialization_resumes_from_trusted_partial(tmp_path: Path):
    """A failed run keeps runtime records separately and safely resumes them."""
    destination = tmp_path / "gutenberg"
    mat = GutenbergMaterializer(
        destination_override=destination,
        allow_custom_destination=True,
    )
    content = {100: b"first payload\n", 101: b"second payload\n"}

    def first_run_get(url, **kwargs):
        response = MagicMock()
        response.url = url
        ebook_id = 100 if "/100/" in url else 101
        if ebook_id == 101:
            response.status_code = 404
        else:
            response.status_code = 200
            response.iter_content.return_value = [content[ebook_id]]
        return response

    with (
        patch("requests.Session.get", side_effect=first_run_get),
        patch("cambacica.corpus.materialize.time.sleep"),
        patch(
            "cambacica.corpus.materialize._get_clean_tool_git_commit",
            return_value="test-commit",
        ),
    ):
        failed = mat.materialize(
            ebook_ids=[100, 101],
            verify_live=False,
            max_retries=1,
            concurrency=2,
        )
    assert failed.status == "FAILED"
    assert not (destination / "manifest.json").exists()
    runtime_manifest = destination / "manifest.in_progress.json"
    assert runtime_manifest.is_file()
    assert MaterializationManifest.load(runtime_manifest).status == "FAILED"

    def second_run_get(url, **kwargs):
        response = MagicMock()
        response.status_code = 200
        response.url = url
        ebook_id = 100 if "/100/" in url else 101
        response.iter_content.return_value = [content[ebook_id]]
        return response

    with (
        patch("requests.Session.get", side_effect=second_run_get) as mock_get,
        patch(
            "cambacica.corpus.materialize._get_clean_tool_git_commit",
            return_value="test-commit",
        ),
    ):
        recovered = mat.materialize(
            ebook_ids=[100, 101],
            verify_live=False,
            max_retries=1,
            concurrency=2,
        )
    assert recovered.status == "COMPLETE"
    assert all("/100/" not in call.args[0] for call in mock_get.call_args_list)
    assert (destination / "pg100.txt").read_bytes() == content[100]
    assert (destination / "pg101.txt").read_bytes() == content[101]
    assert not runtime_manifest.exists()
    valid, errors = mat.verify()
    assert valid, errors


def test_interrupted_partial_manifest_recovers_verified_records(tmp_path: Path):
    """An atomic PARTIAL checkpoint is reusable only after hash verification."""
    destination = tmp_path / "gutenberg"
    destination.mkdir()
    mat = GutenbergMaterializer(
        destination_override=destination,
        allow_custom_destination=True,
    )
    first_payload = b"checkpointed payload\n"
    first_path = destination / "pg100.txt"
    first_path.write_bytes(first_payload)
    partial = MaterializationManifest(
        source="gutenberg_pt",
        upstream_repository="project_gutenberg_pt",
        pinned_revision="snapshot_2026-10-01",
        snapshot_date="2026-10-01",
        catalog_query="browse/languages/pt",
        catalog_size_ebooks=2,
        status="PARTIAL",
        tool_git_commit="test-commit",
        ebook_ids=[100, 101],
        total_files=1,
        total_bytes=len(first_payload),
        files=[
            {
                "relative_path": "pg100.txt",
                "upstream_identifier": "100",
                "url": "https://www.gutenberg.org/files/100/100-0.txt",
                "bytes": len(first_payload),
                "sha256": compute_file_sha256(first_path),
            }
        ],
    )
    partial.save(destination / "manifest.in_progress.json")

    def mock_get(url, **kwargs):
        response = MagicMock()
        response.status_code = 200
        response.url = url
        response.iter_content.return_value = [b"new payload\n"]
        return response

    with (
        patch("requests.Session.get", side_effect=mock_get) as mock_session_get,
        patch(
            "cambacica.corpus.materialize._get_clean_tool_git_commit",
            return_value="test-commit",
        ),
    ):
        recovered = mat.materialize(
            ebook_ids=[100, 101],
            verify_live=False,
            max_retries=1,
        )
    assert recovered.status == "COMPLETE"
    assert all("/100/" not in call.args[0] for call in mock_session_get.call_args_list)
    assert (
        next(
            record["url"]
            for record in recovered.files
            if record["upstream_identifier"] == "100"
        )
        == "https://www.gutenberg.org/files/100/100-0.txt"
    )


def test_cli_materialize_commands(tmp_path: Path):
    """Test CLI subcommands for materialize."""
    # Dry-run
    ret = main(["materialize", "gutenberg", "--dry-run"])
    assert ret == 0

    # Non-pilot stub source
    ret = main(["materialize", "gigaverbo"])
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


def test_parlamento_materializer_plan(tmp_path):
    mat = ParlamentoMaterializer(
        destination_override=tmp_path / "parlamento", allow_custom_destination=True
    )
    plan = mat.plan()
    assert plan["source"] == "parlamento_pt"
    assert "artifact_url" in plan


def test_parlamento_materializer_dry_run(tmp_path):
    mat = ParlamentoMaterializer(
        destination_override=tmp_path / "parlamento", allow_custom_destination=True
    )
    plan = mat.plan()
    assert "08f13e7e63ab9bfbd8c0b40955defe3bb7f68c2b" in plan["artifact_url"]
    assert plan["pinned_commit_sha"] == "08f13e7e63ab9bfbd8c0b40955defe3bb7f68c2b"


def test_parlamento_materializer_mocked(tmp_path, monkeypatch):
    mat = ParlamentoMaterializer(
        destination_override=tmp_path / "parlamento", allow_custom_destination=True
    )

    class MockResp:
        def __init__(self):
            self.status_code = 200

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"line1\nline2\n"

    def mock_get(*args, **kwargs):
        url = args[1] if len(args) > 1 else kwargs.get("url", "")
        assert "08f13e7e63ab9bfbd8c0b40955defe3bb7f68c2b" in url
        return MockResp()

    monkeypatch.setattr("requests.Session.get", mock_get)
    monkeypatch.setattr(
        "cambacica.corpus.materialize._get_clean_tool_git_commit", lambda: "fake"
    )

    manifest = mat.materialize(max_retries=1)
    assert manifest.status == "COMPLETE"
    assert manifest.line_count == 2
    assert manifest.upstream_blob_oid == "d01100ee7525d918539d8a2c2cea2836c7948191"


def test_parlamento_corruption_rejected(tmp_path, monkeypatch):
    mat = ParlamentoMaterializer(
        destination_override=tmp_path / "parlamento", allow_custom_destination=True
    )

    class MockResp:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"data\n"

    monkeypatch.setattr("requests.Session.get", lambda *a, **k: MockResp())
    monkeypatch.setattr(
        "cambacica.corpus.materialize._get_clean_tool_git_commit", lambda: "fake"
    )

    mat.materialize()

    # Corrupt
    (tmp_path / "parlamento" / "train.txt").write_text("bad data")

    import pytest

    with pytest.raises(RuntimeError):
        mat.materialize()


def test_parlamento_trusted_manifest_reuse(tmp_path, monkeypatch):
    mat = ParlamentoMaterializer(
        destination_override=tmp_path / "parlamento", allow_custom_destination=True
    )

    class MockResp:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"data\n"

    monkeypatch.setattr("requests.Session.get", lambda *a, **k: MockResp())
    monkeypatch.setattr(
        "cambacica.corpus.materialize._get_clean_tool_git_commit", lambda: "fake"
    )

    mat.materialize()

    # Second run
    monkeypatch.setattr(
        "requests.Session.get", lambda *a, **k: 1 / 0
    )  # Should not be called
    mat.materialize()


def test_parlamento_verify_only(tmp_path, monkeypatch):
    mat = ParlamentoMaterializer(
        destination_override=tmp_path / "parlamento", allow_custom_destination=True
    )

    class MockResp:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"data\n"

    monkeypatch.setattr("requests.Session.get", lambda *a, **k: MockResp())
    monkeypatch.setattr(
        "cambacica.corpus.materialize._get_clean_tool_git_commit", lambda: "fake"
    )
    mat.materialize()
    assert mat.verify()[0]


def test_parlamento_verify_only_on_empty(tmp_path):
    mat = ParlamentoMaterializer(
        destination_override=tmp_path / "parlamento", allow_custom_destination=True
    )
    assert mat.verify()[0] is False


def test_parlamento_incomplete_download_recovers(tmp_path, monkeypatch):
    mat = ParlamentoMaterializer(
        destination_override=tmp_path / "parlamento", allow_custom_destination=True
    )

    calls = 0

    class MockResp:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise Exception("Network Error")
            yield b"data\n"

    monkeypatch.setattr("requests.Session.get", lambda *a, **k: MockResp())
    monkeypatch.setattr(
        "cambacica.corpus.materialize._get_clean_tool_git_commit", lambda: "fake"
    )
    monkeypatch.setattr("time.sleep", lambda x: None)

    mat.materialize()
    assert (tmp_path / "parlamento" / "train.txt").exists()


# WIKIPEDIA
def test_wikipedia_materializer_plan(tmp_path):
    mat = WikipediaMaterializer(
        destination_override=tmp_path / "wikipedia", allow_custom_destination=True
    )
    plan = mat.plan()
    assert plan["shard_count"] == 6
    assert len(plan["shards"]) == 6


def test_wikipedia_pinned_commit_used(tmp_path):
    mat = WikipediaMaterializer(
        destination_override=tmp_path / "wikipedia", allow_custom_destination=True
    )
    assert "b04c8d1ceb2f5cd4588862100d08de323dccfbaa" in mat.base_url


def test_wikipedia_mocked_6_shards(tmp_path, monkeypatch):
    mat = WikipediaMaterializer(
        destination_override=tmp_path / "wikipedia", allow_custom_destination=True
    )

    class MockResp:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"data"

    monkeypatch.setattr("requests.Session.get", lambda *a, **k: MockResp())
    monkeypatch.setattr(
        "cambacica.corpus.materialize._get_clean_tool_git_commit", lambda: "fake"
    )

    manifest = mat.materialize()
    assert manifest.status == "COMPLETE"
    assert manifest.total_files == 6
    assert manifest.dataset_config == "20231101.pt"


def test_wikipedia_trusted_reuse(tmp_path, monkeypatch):
    mat = WikipediaMaterializer(
        destination_override=tmp_path / "wikipedia", allow_custom_destination=True
    )

    class MockResp:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"data"

    monkeypatch.setattr("requests.Session.get", lambda *a, **k: MockResp())
    monkeypatch.setattr(
        "cambacica.corpus.materialize._get_clean_tool_git_commit", lambda: "fake"
    )

    mat.materialize()
    monkeypatch.setattr("requests.Session.get", lambda *a, **k: 1 / 0)
    mat.materialize()


def test_wikipedia_corruption_rejected(tmp_path, monkeypatch):
    mat = WikipediaMaterializer(
        destination_override=tmp_path / "wikipedia", allow_custom_destination=True
    )

    class MockResp:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"data"

    monkeypatch.setattr("requests.Session.get", lambda *a, **k: MockResp())
    monkeypatch.setattr(
        "cambacica.corpus.materialize._get_clean_tool_git_commit", lambda: "fake"
    )

    mat.materialize()
    (
        tmp_path / "wikipedia" / "20231101.pt" / "train-00000-of-00006.parquet"
    ).write_text("bad")
    import pytest

    with pytest.raises(RuntimeError):
        mat.materialize()


def test_wikipedia_partial_recovery(tmp_path, monkeypatch):
    mat = WikipediaMaterializer(
        destination_override=tmp_path / "wikipedia", allow_custom_destination=True
    )

    calls = []

    class MockResp:
        def __init__(self, url):
            self.url = url

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"data"

    def mock_get(self_obj, url, *a, **k):
        calls.append(url)
        if len(calls) == 3:
            raise Exception("Fail on 3rd")
        return MockResp(url)

    monkeypatch.setattr("requests.Session.get", mock_get)
    monkeypatch.setattr(
        "cambacica.corpus.materialize._get_clean_tool_git_commit", lambda: "fake"
    )
    monkeypatch.setattr("time.sleep", lambda x: None)

    manifest = mat.materialize(max_retries=1, concurrency=1)
    assert manifest.status == "FAILED"

    calls.clear()

    def mock_get2(self_obj, url, *a, **k):
        calls.append(url)
        return MockResp(url)

    monkeypatch.setattr("requests.Session.get", mock_get2)

    manifest = mat.materialize(max_retries=1, concurrency=1)
    assert manifest.status == "COMPLETE"
    assert len(calls) < 6


def test_wikipedia_manifest_shard_oids(tmp_path, monkeypatch):
    mat = WikipediaMaterializer(
        destination_override=tmp_path / "wikipedia", allow_custom_destination=True
    )

    class MockResp:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"data"

    monkeypatch.setattr("requests.Session.get", lambda *a, **k: MockResp())
    monkeypatch.setattr(
        "cambacica.corpus.materialize._get_clean_tool_git_commit", lambda: "fake"
    )

    manifest = mat.materialize()
    assert manifest.upstream_shard_oids
    assert "train-00000-of-00006.parquet" in manifest.upstream_shard_oids


# CAROLINA
def test_carolina_materializer_plan(tmp_path):
    mat = CarolinaMaterializer(
        destination_override=tmp_path / "carolina", allow_custom_destination=True
    )
    plan = mat.plan()
    assert "dat" in plan["taxonomies"]
    assert plan["pinned_commit_sha"] == "55e63a519393c70a48dcfa14a558499c6bb0583b"


def test_carolina_mocked_discovery_and_download(tmp_path, monkeypatch):
    mat = CarolinaMaterializer(
        destination_override=tmp_path / "carolina", allow_custom_destination=True
    )

    def mock_list_files(self, path):
        return [f"{path}/checksum.sha256", f"{path}/f0.xml.gz"]

    monkeypatch.setattr(CarolinaMaterializer, "_list_files", mock_list_files)

    class MockResp:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"data"

    monkeypatch.setattr("requests.Session.get", lambda *a, **k: MockResp())
    monkeypatch.setattr(
        "cambacica.corpus.materialize._get_clean_tool_git_commit", lambda: "fake"
    )

    manifest = mat.materialize(concurrency=10)
    assert manifest.status == "COMPLETE"
    assert manifest.taxonomy_file_counts


def test_carolina_checksum_files_acquired(tmp_path, monkeypatch):
    mat = CarolinaMaterializer(
        destination_override=tmp_path / "carolina", allow_custom_destination=True
    )

    def mock_list_files(self, path):
        return [f"{path}/checksum.sha256", f"{path}/1.xml.gz"]

    monkeypatch.setattr(CarolinaMaterializer, "_list_files", mock_list_files)

    class MockResp:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"data"

    monkeypatch.setattr("requests.Session.get", lambda *a, **k: MockResp())
    monkeypatch.setattr(
        "cambacica.corpus.materialize._get_clean_tool_git_commit", lambda: "fake"
    )
    manifest = mat.materialize()
    assert manifest.status == "COMPLETE"
    assert "dat" in manifest.taxonomy_checksum_files


def test_carolina_trusted_reuse(tmp_path, monkeypatch):
    mat = CarolinaMaterializer(
        destination_override=tmp_path / "carolina", allow_custom_destination=True
    )

    def mock_list_files(self, path):
        return [f"{path}/1.xml.gz"]

    monkeypatch.setattr(CarolinaMaterializer, "_list_files", mock_list_files)

    class MockResp:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"data"

    monkeypatch.setattr("requests.Session.get", lambda *a, **k: MockResp())
    monkeypatch.setattr(
        "cambacica.corpus.materialize._get_clean_tool_git_commit", lambda: "fake"
    )
    mat.materialize()
    monkeypatch.setattr("requests.Session.get", lambda *a, **k: 1 / 0)
    mat.materialize()


def test_carolina_corruption_rejected(tmp_path, monkeypatch):
    mat = CarolinaMaterializer(
        destination_override=tmp_path / "carolina", allow_custom_destination=True
    )

    def mock_list_files(self, path):
        return [f"{path}/1.xml.gz"]

    monkeypatch.setattr(CarolinaMaterializer, "_list_files", mock_list_files)

    class MockResp:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"data"

    monkeypatch.setattr("requests.Session.get", lambda *a, **k: MockResp())
    monkeypatch.setattr(
        "cambacica.corpus.materialize._get_clean_tool_git_commit", lambda: "fake"
    )
    mat.materialize()

    (tmp_path / "carolina" / "corpus" / "judicial_branch" / "1.xml.gz").write_text(
        "bad"
    )
    import pytest

    with pytest.raises(RuntimeError):
        mat.materialize()


def test_carolina_taxonomy_coverage(tmp_path, monkeypatch):
    mat = CarolinaMaterializer(
        destination_override=tmp_path / "carolina", allow_custom_destination=True
    )

    def mock_list_files(self, path):
        return [f"{path}/1.xml.gz"]

    monkeypatch.setattr(CarolinaMaterializer, "_list_files", mock_list_files)

    class MockResp:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"data"

    monkeypatch.setattr("requests.Session.get", lambda *a, **k: MockResp())
    monkeypatch.setattr(
        "cambacica.corpus.materialize._get_clean_tool_git_commit", lambda: "fake"
    )
    manifest = mat.materialize()

    assert len(manifest.taxonomy_file_counts) == 7


def test_carolina_pinned_commit_url(tmp_path, monkeypatch):
    mat = CarolinaMaterializer(
        destination_override=tmp_path / "carolina", allow_custom_destination=True
    )

    def mock_list_files(self, path):
        return [f"{path}/1.xml.gz"]

    monkeypatch.setattr(CarolinaMaterializer, "_list_files", mock_list_files)

    urls = []

    class MockResp:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"data"

    def mock_get(self_obj, url, *a, **k):
        urls.append(url)
        return MockResp()

    monkeypatch.setattr("requests.Session.get", mock_get)
    monkeypatch.setattr(
        "cambacica.corpus.materialize._get_clean_tool_git_commit", lambda: "fake"
    )
    mat.materialize()

    assert all("55e63a519393c70a48dcfa14a558499c6bb0583b" in u for u in urls)


def test_carolina_list_files_filters_non_target_artifacts(tmp_path, monkeypatch):
    """Verify that _list_files discards stray/temp artifacts (e.g. .goutputstream)."""
    mat = CarolinaMaterializer(
        destination_override=tmp_path / "carolina", allow_custom_destination=True
    )

    fake_tree_response = [
        {"type": "file", "path": "corpus/wikis/pt/WIK01.xml.gz"},
        {"type": "file", "path": "corpus/wikis/pt/checksum.sha256"},
        {"type": "file", "path": "corpus/wikis/pt/.goutputstream-AQRB22"},
        {"type": "file", "path": "corpus/wikis/pt/README.txt"},
    ]

    class MockResp:
        def raise_for_status(self):
            pass

        def json(self):
            return fake_tree_response

    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: MockResp())
    files = mat._list_files("corpus/wikis/pt")

    assert "corpus/wikis/pt/WIK01.xml.gz" in files
    assert "corpus/wikis/pt/checksum.sha256" in files
    assert "corpus/wikis/pt/.goutputstream-AQRB22" not in files
    assert "corpus/wikis/pt/README.txt" not in files
    assert len(files) == 2
