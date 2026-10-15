from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cambacica.benchmark import artifacts


def test_bundle_has_identity_and_valid_sha256_ledger(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    hashes = artifacts.write_artifact_bundle(
        stage,
        run_id="run-test",
        configuration={"architecture": "A", "context_length": 2048},
        result={"status": "failed", "failure_kind": "synthetic-test"},
        implementation_digest="a" * 64,
    )
    ledger = {}
    for line in (stage / "checksums.sha256").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        ledger[name] = digest
    assert ledger == hashes
    for name, digest in ledger.items():
        assert hashlib.sha256((stage / name).read_bytes()).hexdigest() == digest
    result = json.loads((stage / "result.json").read_text(encoding="utf-8"))
    assert result["configuration_sha256"] == ledger["configuration.json"]
    assert result["artifact_checksums"] == {
        "configuration.json": ledger["configuration.json"]
    }
    manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "run-test"
    assert manifest["artifacts"]["result.json"] == ledger["result.json"]


def test_publish_is_candidate_specific_atomic_and_never_overwrites(tmp_path: Path) -> None:
    stage = tmp_path / "scratch" / "run-1"
    stage.mkdir(parents=True)
    (stage / "result.json").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "nfs"
    published = artifacts.publish_artifact_bundle(
        stage,
        output_root=output,
        architecture="B",
        context_length=4096,
        run_id="run-1",
    )
    assert published == output / "architecture-B" / "context-4096" / "run-1"
    assert (published / "result.json").read_text(encoding="utf-8") == "{}\n"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        artifacts.publish_artifact_bundle(
            stage,
            output_root=output,
            architecture="B",
            context_length=4096,
            run_id="run-1",
        )


def test_storage_layout_requires_local_scratch_and_nfs_output(monkeypatch, tmp_path: Path) -> None:
    types = {tmp_path / "scratch": "ext4", tmp_path / "output": "nfs4"}
    monkeypatch.setattr(artifacts, "_filesystem_type", lambda path: types[path])
    artifacts.require_storage_layout(tmp_path / "scratch", tmp_path / "output")
    types[tmp_path / "output"] = "ext4"
    with pytest.raises(ValueError, match="must use NFS"):
        artifacts.require_storage_layout(tmp_path / "scratch", tmp_path / "output")
