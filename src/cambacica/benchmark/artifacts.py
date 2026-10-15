"""Immutable local staging and atomic publication for C3 result bundles."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_sha256(repo_root: Path = REPO_ROOT) -> str:
    """Hash the C3 implementation sources and their declared dependencies."""

    paths = [repo_root / "pyproject.toml"]
    for package in ("model", "benchmark"):
        paths.extend(sorted((repo_root / "src/cambacica" / package).glob("*.py")))
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(repo_root).as_posix()):
        relative = path.relative_to(repo_root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_provenance(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}
    return {"commit": commit, "dirty": bool(status.strip())}


def runtime_identity(torch_module: Any) -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch_module.__version__,
        "cuda_runtime_version": getattr(torch_module.version, "cuda", None),
    }


def new_run_id(seed: int) -> str:
    from datetime import datetime, timezone
    import uuid

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{stamp}-seed{seed}-{uuid.uuid4().hex[:12]}"


def write_artifact_bundle(
    staging_dir: Path,
    *,
    run_id: str,
    configuration: dict[str, object],
    result: dict[str, object],
    implementation_digest: str,
) -> dict[str, str]:
    """Write checksummed config, result, manifest, and checksum ledger files."""

    staging_dir.mkdir(parents=True, exist_ok=True)
    config_path = staging_dir / "configuration.json"
    result_path = staging_dir / "result.json"
    manifest_path = staging_dir / "manifest.json"
    checksum_path = staging_dir / "checksums.sha256"

    config_bytes = canonical_json(configuration)
    config_path.write_bytes(config_bytes)
    result_payload = dict(result)
    configuration_digest = hashlib.sha256(config_bytes).hexdigest()
    result_payload["configuration_sha256"] = configuration_digest
    result_payload["artifact_checksums"] = {"configuration.json": configuration_digest}
    result_bytes = canonical_json(result_payload)
    result_path.write_bytes(result_bytes)
    file_hashes = {
        "configuration.json": hashlib.sha256(config_bytes).hexdigest(),
        "result.json": hashlib.sha256(result_bytes).hexdigest(),
    }
    manifest = {
        "schema_version": "c3-benchmark-artifact-v1",
        "run_id": run_id,
        "status": result_payload.get("status"),
        "implementation_sha256": implementation_digest,
        "artifacts": file_hashes,
    }
    manifest_bytes = canonical_json(manifest)
    manifest_path.write_bytes(manifest_bytes)
    file_hashes["manifest.json"] = hashlib.sha256(manifest_bytes).hexdigest()
    checksum_text = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(file_hashes.items())
    )
    checksum_path.write_text(checksum_text, encoding="ascii")
    for path in (config_path, result_path, manifest_path, checksum_path):
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    return file_hashes


def _filesystem_type(path: Path) -> str:
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    result = subprocess.run(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(target)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def require_storage_layout(scratch_root: Path, output_root: Path) -> None:
    scratch_type = _filesystem_type(scratch_root)
    output_type = _filesystem_type(output_root)
    if scratch_type.startswith("nfs"):
        raise ValueError(f"runtime scratch must be local storage, found {scratch_type}")
    if not output_type.startswith("nfs"):
        raise ValueError(f"final benchmark artifacts must use NFS, found {output_type}")
    scratch = scratch_root.resolve()
    output = output_root.resolve()
    if scratch == output or scratch in output.parents or output in scratch.parents:
        raise ValueError("runtime scratch and finalized output roots must be separate")


def publish_artifact_bundle(
    staging_dir: Path,
    *,
    output_root: Path,
    architecture: str,
    context_length: int,
    run_id: str,
) -> Path:
    """Copy to NFS staging then atomically rename into an immutable run path."""

    parent = output_root / f"architecture-{architecture}" / f"context-{context_length}"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / run_id
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite completed benchmark: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.partial-", dir=parent))
    try:
        for source in sorted(staging_dir.iterdir()):
            if not source.is_file():
                continue
            target = temporary / source.name
            shutil.copyfile(source, target)
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite completed benchmark: {destination}"
            )
        os.rename(temporary, destination)
        parent_fd = os.open(parent, os.O_RDONLY)
        try:
            try:
                os.fsync(parent_fd)
            except OSError:
                # The directory rename already published a complete bundle.
                pass
        finally:
            os.close(parent_fd)
        return destination
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
