"""Corpus materialization infrastructure for Gate C1.

Provides reproducible, verifiable raw source materialization with deterministic
snapshot pinning, resume safety, atomic temporary writes, local SHA-256 hashing,
and machine-readable provenance manifests.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import requests
import yaml

from cambacica.corpus.manifest import compute_file_sha256

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/corpus_materialization.yaml")
MATERIALIZER_REPO_ROOT = Path(__file__).resolve().parents[3]
GUTENBERG_CATALOG_URL = "https://www.gutenberg.org/browse/languages/pt"
GUTENBERG_URL_CANDIDATES = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]


def _atomic_write_json(data: dict, path: Path | str) -> Path:
    """Write JSON through a same-directory temporary file and atomic replace."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = out_path.with_name(f"{out_path.name}.partial")
    try:
        with temporary_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, out_path)
    except Exception:
        # A completed prior destination remains intact; the temporary file is
        # deliberately retained for inspection/recovery after a failed write.
        raise
    return out_path


def _get_clean_tool_git_commit() -> str:
    """Return HEAD only when the materializer repository is clean."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=MATERIALIZER_REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=MATERIALIZER_REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not establish materializer Git provenance: {exc}"
        ) from exc

    if head.returncode != 0 or status.returncode != 0 or not head.stdout.strip():
        raise RuntimeError("Could not establish materializer Git provenance.")
    if status.stdout.strip():
        raise RuntimeError(
            "Refusing production materialization from a dirty working tree; "
            "commit or discard all tracked and untracked changes first."
        )
    return head.stdout.strip()


def load_yaml_config(config_path: Path | str) -> dict:
    """Load configuration dictionary from a YAML file.

    Parameters
    ----------
    config_path : Path or str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary, or empty dict if not found.
    """
    path = Path(config_path)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class MaterializedFileRecord:
    """Record describing a single materialized raw file payload.

    Parameters
    ----------
    relative_path : str
        Relative path of the payload under the source raw destination directory.
    upstream_identifier : str
        Upstream identifier or name for this payload item (e.g. eBook ID).
    url : str
        URL or upstream resource identifier from which payload was fetched.
    bytes : int
        Size in bytes of the materialized payload on local disk.
    sha256 : str
        Locally computed hexadecimal SHA-256 checksum of payload bytes.
    checksum_source : str, default 'local_sha256'
        Provenance of the checksum ('local_sha256' or 'upstream_provided').
    """

    relative_path: str
    upstream_identifier: str
    url: str
    bytes: int
    sha256: str
    checksum_source: str = "local_sha256"

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a serializable dictionary.

        Returns
        -------
        dict
            Dictionary representation of the file record.
        """
        return asdict(self)


@dataclass
class MaterializationManifest:
    """Machine-readable source materialization manifest.

    Parameters
    ----------
    schema_version : int, default 1
        Schema version of this manifest structure.
    source : str
        Canonical source identifier (e.g. 'gutenberg_pt', 'carolina').
    upstream_repository : str
        Canonical upstream repository, project, or organization identifier.
    pinned_revision : str
        Configured human-readable revision identifier or snapshot tag.
    pinned_commit_sha : str or None, optional
        Immutable upstream Git commit SHA when available.
    snapshot_date : str or None, optional
        Catalog or repository snapshot date.
    acquisition_started_at : str, optional
        ISO 8601 UTC timestamp marking when materialization began.
    acquisition_completed_at : str or None, optional
        ISO 8601 UTC timestamp marking when materialization finished.
    status : str, default 'PARTIAL'
        Materialization status ('COMPLETE', 'PARTIAL', 'FAILED').
    total_files : int, default 0
        Total number of materialized payload files recorded.
    total_bytes : int, default 0
        Total byte size of all materialized payload files recorded.
    tool_git_commit : str or None, optional
        Clean Git commit hash of the cambacica codebase at materialization time.
    checksum_provenance : str, default 'local_payload_sha256'
        Explanation of checksum origin.
    ebook_ids : list of int or None, optional
        Sorted deterministic list of catalog eBook IDs for Gutenberg.
    catalog_snapshot_date : str or None, optional
        Snapshot date for catalog-based sources.
    catalog_query : str or None, optional
        Catalog query string used for discovery.
    catalog_size_ebooks : int or None, optional
        Expected count of catalog items.
    failed_ids : list of int, optional
        List of item identifiers that failed acquisition.
    failure_reasons : dict, optional
        Dictionary mapping item identifiers to failure reason strings.
    files : list of dict, optional
        List of serialized MaterializedFileRecord dictionaries.
    """

    source: str
    upstream_repository: str
    pinned_revision: str
    schema_version: int = 1
    pinned_commit_sha: Optional[str] = None
    snapshot_date: Optional[str] = None
    acquisition_started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    acquisition_completed_at: Optional[str] = None
    status: str = "PARTIAL"
    total_files: int = 0
    total_bytes: int = 0
    tool_git_commit: Optional[str] = None
    checksum_provenance: str = "local_payload_sha256"
    ebook_ids: Optional[List[int]] = None
    catalog_snapshot_date: Optional[str] = None
    catalog_query: Optional[str] = None
    catalog_size_ebooks: Optional[int] = None
    failed_ids: List[int] = field(default_factory=list)
    failure_reasons: Dict[str, str] = field(default_factory=dict)
    files: List[Dict[str, Any]] = field(default_factory=list)
    taxonomy_file_counts: Optional[Dict[str, int]] = None
    taxonomy_checksum_files: Optional[Dict[str, str]] = None
    dataset_config: Optional[str] = None
    snapshot_identifier: Optional[str] = None
    upstream_shard_oids: Optional[Dict[str, str]] = None
    raw_artifact_name: Optional[str] = None
    line_count: Optional[int] = None
    upstream_blob_oid: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to a serializable dictionary.

        Returns
        -------
        dict
            Serialized manifest dictionary.
        """
        return asdict(self)

    def save(self, path: Path | str) -> Path:
        """Serialize manifest to a JSON file on disk.

        Parameters
        ----------
        path : Path or str
            Destination file path.

        Returns
        -------
        Path
            Resolved destination file path.
        """
        return _atomic_write_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: Path | str) -> MaterializationManifest:
        """Load manifest from a JSON file on disk.

        Parameters
        ----------
        path : Path or str
            Path to the JSON manifest.

        Returns
        -------
        MaterializationManifest
            Deserialized manifest instance.

        Raises
        ------
        FileNotFoundError
            If manifest file does not exist.
        """
        in_path = Path(path)
        try:
            with in_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not read valid JSON manifest {in_path}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise ValueError(f"Manifest {in_path} must contain a JSON object.")
        if not isinstance(data.get("files", []), list):
            raise ValueError(f"Manifest {in_path} field 'files' must be a list.")
        if not isinstance(data.get("failed_ids", []), list):
            raise ValueError(f"Manifest {in_path} field 'failed_ids' must be a list.")
        if not isinstance(data.get("failure_reasons", {}), dict):
            raise ValueError(
                f"Manifest {in_path} field 'failure_reasons' must be an object."
            )
        if data.get("ebook_ids") is not None and not isinstance(
            data["ebook_ids"], list
        ):
            raise ValueError(f"Manifest {in_path} field 'ebook_ids' must be a list.")
        if data.get("ebook_ids") is not None and any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in data["ebook_ids"]
        ):
            raise ValueError(
                f"Manifest {in_path} field 'ebook_ids' must contain positive integers."
            )
        for index, record in enumerate(data.get("files", [])):
            if not isinstance(record, dict):
                raise ValueError(
                    f"Manifest {in_path} file record {index} must be an object."
                )
            required_record_fields = {
                "relative_path",
                "upstream_identifier",
                "url",
                "bytes",
                "sha256",
            }
            if not required_record_fields.issubset(record):
                raise ValueError(
                    f"Manifest {in_path} file record {index} is missing required fields."
                )
            if not isinstance(record["relative_path"], str) or not isinstance(
                record["url"], str
            ):
                raise ValueError(
                    f"Manifest {in_path} file record {index} has invalid path or URL."
                )
            if not record["url"]:
                raise ValueError(
                    f"Manifest {in_path} file record {index} has an empty URL."
                )
            if (
                not isinstance(record["bytes"], int)
                or isinstance(record["bytes"], bool)
                or record["bytes"] < 0
            ):
                raise ValueError(
                    f"Manifest {in_path} file record {index} has invalid byte count."
                )
            digest = record["sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in digest)
            ):
                raise ValueError(
                    f"Manifest {in_path} file record {index} has invalid SHA-256."
                )
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        try:
            manifest = cls(**filtered_data)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Manifest {in_path} has invalid fields: {exc}") from exc
        if manifest.status not in {"COMPLETE", "PARTIAL", "FAILED"}:
            raise ValueError(
                f"Manifest {in_path} has unsupported status {manifest.status!r}."
            )
        return manifest

    def verify(
        self,
        destination: Path | str,
        *,
        check_partial_files: bool = True,
        check_runtime_state: bool = True,
        check_ids_metadata: bool = True,
    ) -> Tuple[bool, List[str]]:
        """Verify all recorded files against files present on disk.

        Parameters
        ----------
        destination : Path or str
            Root directory of materialized payloads.

        Returns
        -------
        tuple of (bool, list of str)
            True if all files verify perfectly with no discrepancies, else False,
            and a list of human-readable error messages.
        """
        root = Path(destination)
        errors: List[str] = []

        if not root.is_dir():
            errors.append(f"Destination directory does not exist: {root}")
            return False, errors

        # Verify no unfinished payload or metadata writes remain.
        if check_partial_files:
            partial_files = list(root.glob("*.partial*"))
            partial_files.extend(root.glob(".*.partial*"))
            for pf in sorted(set(partial_files)):
                errors.append(f"Orphaned partial file detected: {pf.name}")
        if check_runtime_state and (root / "manifest.in_progress.json").exists():
            errors.append(
                "An interrupted or unfinished manifest.in_progress.json exists."
            )

        # Check manifest status
        if self.status != "COMPLETE":
            errors.append(f"Manifest status is '{self.status}', expected 'COMPLETE'.")

        # Check source-specific expected counts
        if (
            self.catalog_size_ebooks is not None
            and len(self.files) != self.catalog_size_ebooks
        ):
            errors.append(
                f"File count mismatch: recorded {len(self.files)} files, "
                f"expected catalog size {self.catalog_size_ebooks}."
            )

        if self.failed_ids:
            errors.append(
                f"Manifest records {len(self.failed_ids)} failed IDs: {self.failed_ids}"
            )

        if self.source == "gutenberg_pt" and self.ebook_ids is not None:
            if len(self.ebook_ids) != len(set(self.ebook_ids)):
                errors.append("Manifest ebook_ids contains duplicate identifiers.")
            if self.ebook_ids != sorted(self.ebook_ids):
                errors.append("Manifest ebook_ids is not sorted deterministically.")
            try:
                expected_ids = {str(int(value)) for value in self.ebook_ids}
                record_ids = {
                    str(int(record.get("upstream_identifier"))) for record in self.files
                }
            except (TypeError, ValueError):
                errors.append("Manifest contains a non-numeric Gutenberg eBook ID.")
            else:
                if record_ids != expected_ids:
                    errors.append(
                        "Manifest file identifiers do not match the frozen ebook_ids."
                    )
                for record in self.files:
                    try:
                        item_id = int(record.get("upstream_identifier"))
                    except (TypeError, ValueError):
                        continue
                    if record.get("relative_path") != f"pg{item_id}.txt":
                        errors.append(
                            f"File path does not match Gutenberg ID {item_id}: "
                            f"{record.get('relative_path')}"
                        )
        elif self.source == "gutenberg_pt" and self.status == "COMPLETE":
            errors.append("COMPLETE Gutenberg manifest is missing frozen ebook_ids.")

        if self.source == "carolina" and self.status == "COMPLETE":
            if not self.taxonomy_file_counts:
                errors.append(
                    "COMPLETE Carolina manifest is missing taxonomy_file_counts."
                )
            else:
                expected_keys = {"dat", "jud", "leg", "pub", "soc", "uni", "wik"}
                actual_keys = set(self.taxonomy_file_counts.keys())
                if actual_keys != expected_keys:
                    errors.append(
                        f"Carolina taxonomy keys mismatch: {actual_keys} vs {expected_keys}"
                    )
                total_tax_files = sum(self.taxonomy_file_counts.values())
                total_checksums = (
                    len(self.taxonomy_checksum_files)
                    if self.taxonomy_checksum_files
                    else 0
                )
                if total_tax_files + total_checksums != self.total_files:
                    errors.append(
                        f"Carolina taxonomy total files ({total_tax_files} + {total_checksums}) != total_files ({self.total_files})"
                    )

        if self.source == "wikipedia_pt" and self.status == "COMPLETE":
            if self.total_files != 6:
                errors.append(
                    f"COMPLETE Wikipedia manifest has {self.total_files} files, expected 6."
                )

        # Verify files list integrity
        computed_total_bytes = 0
        seen_paths: Set[str] = set()

        for record in self.files:
            rel_path = record.get("relative_path")
            expected_bytes = record.get("bytes", 0)
            expected_sha = record.get("sha256", "")

            if not rel_path:
                errors.append(f"File record missing relative_path: {record}")
                continue

            if rel_path in seen_paths:
                errors.append(f"Duplicate file entry in manifest: {rel_path}")
            seen_paths.add(rel_path)

            file_path = (root / rel_path).resolve()
            if root.resolve() not in file_path.parents:
                errors.append(f"File record escapes the destination: {rel_path}")
                continue
            if not file_path.is_file():
                errors.append(f"Missing file on disk: {rel_path}")
                continue

            actual_bytes = file_path.stat().st_size
            if actual_bytes != expected_bytes:
                errors.append(
                    f"Byte size mismatch for {rel_path}: recorded {expected_bytes}, "
                    f"actual on disk {actual_bytes}."
                )

            actual_sha = compute_file_sha256(file_path)
            if actual_sha.lower() != expected_sha.lower():
                errors.append(
                    f"SHA-256 mismatch for {rel_path}: recorded {expected_sha}, "
                    f"computed {actual_sha}."
                )

            computed_total_bytes += actual_bytes

        ids_metadata_path = root / "ebook_ids.json"
        if (
            check_ids_metadata
            and ids_metadata_path.is_file()
            and self.ebook_ids is not None
        ):
            try:
                with ids_metadata_path.open("r", encoding="utf-8") as f:
                    ids_metadata = json.load(f)
                metadata_ids = (
                    ids_metadata.get("ebook_ids")
                    if isinstance(ids_metadata, dict)
                    else None
                )
                if metadata_ids != self.ebook_ids:
                    errors.append("ebook_ids.json does not match manifest ebook_ids.")
            except (OSError, json.JSONDecodeError):
                errors.append("ebook_ids.json is malformed or unreadable.")

        if self.total_files != len(self.files):
            errors.append(
                f"Manifest total_files ({self.total_files}) does not match "
                f"len(files) ({len(self.files)})."
            )

        if self.total_bytes != computed_total_bytes:
            errors.append(
                f"Manifest total_bytes ({self.total_bytes}) does not match "
                f"computed sum of file bytes ({computed_total_bytes})."
            )

        return len(errors) == 0, errors


class BaseMaterializer:
    """Base materializer defining common contract and safety constraints.

    Parameters
    ----------
    source_name : str
        Canonical source identifier.
    config_path : Path or str, default 'configs/corpus_materialization.yaml'
        Path to materialization configuration YAML.
    destination_override : Path or str or None, optional
        Optional path overriding configured destination.
    allow_custom_destination : bool, default False
        If True, permits destinations outside the configured raw storage root
        (primarily intended for unit and local tests).
    """

    def __init__(
        self,
        source_name: str,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
        destination_override: Optional[Path | str] = None,
        allow_custom_destination: bool = False,
    ) -> None:
        self.source_name = source_name
        self.config_path = Path(config_path)
        self.config = load_yaml_config(self.config_path)

        source_cfg = self.config.get("sources", {}).get(source_name)
        if not source_cfg:
            # Check for alias
            for k, v in self.config.get("sources", {}).items():
                if v.get("canonical_name") == source_name:
                    source_cfg = v
                    break
        if not source_cfg:
            raise ValueError(
                f"Source '{source_name}' not defined in {self.config_path}."
            )
        self.source_config = source_cfg

        storage_cfg = self.config.get("storage", {})
        self.storage_config = storage_cfg
        self.raw_storage_root = Path(
            storage_cfg.get("paths", {}).get("raw", "/mnt/data/cambacica-base-180m/raw")
        ).resolve()

        if destination_override:
            dest = Path(destination_override).resolve()
        else:
            dest = Path(
                source_cfg.get("destination", self.raw_storage_root / source_name)
            ).resolve()

        if not allow_custom_destination:
            # Enforce writing only below the configured raw storage path
            if not (
                dest == self.raw_storage_root or self.raw_storage_root in dest.parents
            ):
                raise ValueError(
                    f"Destination '{dest}' violates safety boundary: "
                    f"must be located within configured raw path '{self.raw_storage_root}'."
                )

        self.destination = dest
        self.allow_custom_destination = allow_custom_destination

    def plan(self) -> Dict[str, Any]:
        """Produce a dry-run execution plan.

        Returns
        -------
        dict
            Structured planning dictionary describing configured revisions,
            targets, and safety parameters.
        """
        raise NotImplementedError

    def materialize(
        self,
        concurrency: int = 4,
        timeout: int = 25,
        max_retries: int = 3,
        **kwargs,
    ) -> MaterializationManifest:
        """Execute materialization of source data.

        Parameters
        ----------
        concurrency : int, default 4
            Number of concurrent download worker threads.
        timeout : int, default 25
            HTTP request timeout in seconds.
        max_retries : int, default 3
            Maximum retries per item.
        **kwargs : any
            Additional source-specific options.

        Returns
        -------
        MaterializationManifest
            Final materialization manifest.
        """
        raise NotImplementedError

    def verify(self) -> Tuple[bool, List[str]]:
        """Verify previously materialized files and manifest.

        Returns
        -------
        tuple of (bool, list of str)
            Verification success flag and error messages.
        """
        manifest_file = self.destination / "manifest.json"
        if not manifest_file.is_file():
            return False, [f"Manifest not found: {manifest_file}"]
        try:
            manifest = MaterializationManifest.load(manifest_file)
        except ValueError as exc:
            return False, [str(exc)]
        return manifest.verify(self.destination)


class GutenbergMaterializer(BaseMaterializer):
    """Materializer for Project Gutenberg Portuguese literary works.

    Acquires full plain-text source payloads for the 655 cataloged Portuguese
    eBooks matching the 2026-10-01 accepted snapshot.
    Preserves raw text payloads without stripping headers/footers or normalizing.
    """

    def __init__(
        self,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
        destination_override: Optional[Path | str] = None,
        allow_custom_destination: bool = False,
    ) -> None:
        super().__init__(
            source_name="gutenberg_pt",
            config_path=config_path,
            destination_override=destination_override,
            allow_custom_destination=allow_custom_destination,
        )
        self.snapshot_date = self.source_config.get("snapshot_date", "2026-10-01")
        self.catalog_query = self.source_config.get(
            "catalog_query", "browse/languages/pt"
        )
        self.expected_size = self.source_config.get("catalog_size_ebooks", 655)
        self.snapshot_config_path = self.source_config.get("snapshot_config")

    def plan(self) -> Dict[str, Any]:
        """Produce a dry-run execution plan for Gutenberg materialization.

        Returns
        -------
        dict
            Dry-run plan details.
        """
        existing_manifest = self.destination / "manifest.json"
        existing_files = (
            list(self.destination.glob("*.txt")) if self.destination.exists() else []
        )

        return {
            "source": self.source_name,
            "canonical_name": self.source_config.get("canonical_name", "gutenberg_pt"),
            "repository": self.source_config.get("repository", "project_gutenberg_pt"),
            "pinned_revision": self.source_config.get(
                "pinned_revision", "snapshot_2026-10-01"
            ),
            "pinned_commit_sha": None,
            "snapshot_date": self.snapshot_date,
            "catalog_query": self.catalog_query,
            "expected_catalog_size": self.expected_size,
            "destination": str(self.destination),
            "estimated_raw_size": self.source_config.get(
                "estimated_raw_size", "0.4 GB"
            ),
            "destination_exists": self.destination.exists(),
            "existing_text_files": len(existing_files),
            "has_manifest": existing_manifest.is_file(),
            "safety_mechanisms": {
                "atomic_writes": "Payload and JSON metadata are replaced atomically after complete writes",
                "idempotent_resume": "Reuses only payloads matching trusted prior manifest records",
                "checksum_hashing": "Locally computes SHA-256 for each payload",
                "tool_provenance": "Requires a clean Git working tree and records HEAD",
                "raw_preservation": "Plain-text unaltered; no boilerplate stripping or normalization",
            },
        }

    def resolve_ebook_ids(self, verify_live: bool = True) -> List[int]:
        """Resolve the deterministic list of 655 eBook IDs.

        Loads the checked-in snapshot as the only accepted ID source. When
        requested, the live catalog is checked for exact consistency.

        Parameters
        ----------
        verify_live : bool, default True
            If True and network is accessible, queries live Gutenberg catalog
            and validates exact equality with the accepted snapshot.

        Returns
        -------
        list of int
            Sorted list of 655 deterministic eBook IDs.

        Raises
        ------
        RuntimeError
            If live catalog count or content does not match the accepted snapshot.
        """
        if not self.snapshot_config_path:
            raise RuntimeError("No frozen Gutenberg snapshot is configured.")
        snapshot_path = Path(self.snapshot_config_path)
        if not snapshot_path.is_absolute():
            snapshot_path = MATERIALIZER_REPO_ROOT / snapshot_path
        snapshot_path = snapshot_path.resolve()
        try:
            snapshot_relative_path = snapshot_path.relative_to(
                MATERIALIZER_REPO_ROOT.resolve()
            )
        except ValueError as exc:
            raise RuntimeError(
                "The authoritative Gutenberg snapshot must be a checked-in file "
                "inside this repository."
            ) from exc
        tracked_snapshot = subprocess.run(
            [
                "git",
                "ls-files",
                "--error-unmatch",
                "--",
                snapshot_relative_path.as_posix(),
            ],
            cwd=MATERIALIZER_REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )
        if tracked_snapshot.returncode != 0:
            raise RuntimeError(
                f"Gutenberg snapshot {snapshot_relative_path} is not tracked by Git."
            )
        try:
            with snapshot_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            raw_ids = data["ebook_ids"]
            if not isinstance(raw_ids, list):
                raise ValueError("ebook_ids must be a JSON list")
            if data.get("schema_version") != 1:
                raise ValueError("unsupported snapshot schema version")
            if data.get("source") not in {None, self.source_name}:
                raise ValueError("snapshot source does not match gutenberg_pt")
            if data.get("snapshot_date") != self.snapshot_date:
                raise ValueError(
                    f"snapshot_date {data.get('snapshot_date')!r} does not match "
                    f"configured date {self.snapshot_date!r}"
                )
            if data.get("catalog_query") != self.catalog_query:
                raise ValueError("catalog_query does not match the configured query")
            if data.get("catalog_size_ebooks") != self.expected_size:
                raise ValueError(
                    "catalog_size_ebooks does not match the configured expected size"
                )
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in raw_ids
            ):
                raise ValueError("ebook_ids must contain JSON integers")
            snapshot_ids = list(raw_ids)
            if any(value <= 0 for value in snapshot_ids):
                raise ValueError("ebook_ids must contain positive integer IDs")
            if len(snapshot_ids) != len(set(snapshot_ids)):
                raise ValueError("ebook_ids contains duplicate IDs")
            snapshot_ids = sorted(snapshot_ids)
            if len(snapshot_ids) != self.expected_size:
                raise ValueError(
                    f"Frozen snapshot contains {len(snapshot_ids)} IDs; "
                    f"expected {self.expected_size}"
                )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Could not load authoritative Gutenberg snapshot {snapshot_path}: {exc}"
            ) from exc

        # The live catalog can only confirm the checked-in snapshot; it never
        # supplies IDs or changes the accepted set.
        if verify_live:
            try:
                from cambacica.corpus.sources.gutenberg_pt import (
                    discover_gutenberg_pt_ids,
                )

                live_ids = sorted(
                    {int(value) for value in discover_gutenberg_pt_ids(timeout=20)}
                )
                if len(live_ids) != self.expected_size:
                    raise RuntimeError(
                        f"Provenance mismatch: Live Gutenberg catalog returned {len(live_ids)} "
                        f"eBooks, expected exactly {self.expected_size}."
                    )
                if live_ids != snapshot_ids:
                    raise RuntimeError(
                        "Provenance mismatch: Live Gutenberg catalog IDs differ from "
                        "the accepted 2026-10-01 snapshot."
                    )
            except Exception as err:
                if isinstance(err, RuntimeError) and "Provenance mismatch" in str(err):
                    raise
                logger.warning(
                    f"Live Gutenberg catalog discovery failed ({err}); "
                    "continuing with the authoritative frozen snapshot."
                )
        return snapshot_ids

    def _download_single(
        self,
        ebook_id: int,
        session: requests.Session,
        timeout: int,
        max_retries: int,
        trusted_record: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, int, str, str, int, Optional[str], Optional[str], int, bool]:
        """Download or verify a single eBook payload.

        Parameters
        ----------
        ebook_id : int
            eBook numeric ID.
        session : requests.Session
            Reusable HTTP session.
        timeout : int
            HTTP timeout in seconds.
        max_retries : int
            Max retry attempts.

        Returns
        -------
        tuple
            (success: bool, ebook_id: int, relative_path: str, url: str,
             bytes: int, sha256: str | None, error: str | None, retries: int, cached: bool)
        """
        filename = f"pg{ebook_id}.txt"
        final_path = self.destination / filename
        partial_path = self.destination / f"{filename}.partial"

        # A final payload is reusable only when a prior manifest supplies its
        # trusted size, digest, and original successful URL.
        if final_path.is_file() and trusted_record is not None:
            size = final_path.stat().st_size
            sha = compute_file_sha256(final_path)
            expected_size = int(trusted_record["bytes"])
            expected_sha = str(trusted_record["sha256"]).lower()
            if size != expected_size or sha.lower() != expected_sha:
                raise RuntimeError(
                    f"Existing payload {filename} does not match its trusted manifest "
                    f"record (expected {expected_size} bytes / {expected_sha}, "
                    f"found {size} bytes / {sha}). Refusing to bless local changes."
                )
            return (
                True,
                ebook_id,
                filename,
                str(trusted_record["url"]),
                size,
                sha,
                None,
                0,
                True,
            )

        candidate_urls = [c.format(id=ebook_id) for c in GUTENBERG_URL_CANDIDATES]
        if trusted_record is not None and trusted_record.get("url"):
            previous_url = str(trusted_record["url"])
            candidate_urls = [previous_url] + [
                candidate for candidate in candidate_urls if candidate != previous_url
            ]
        retries_used = 0
        last_error = None

        for attempt in range(max_retries):
            for url in candidate_urls:
                resp = None
                try:
                    resp = session.get(url, timeout=timeout, stream=True)
                    if resp.status_code != 200:
                        last_error = f"HTTP {resp.status_code} for {url}"
                        resp.close()
                        continue

                    hasher = hashlib.sha256()
                    bytes_count = 0
                    with open(partial_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                                hasher.update(chunk)
                                bytes_count += len(chunk)
                        f.flush()
                        os.fsync(f.fileno())

                    if bytes_count == 0:
                        if partial_path.exists():
                            partial_path.unlink()
                        resp.close()
                        continue

                    successful_url = getattr(resp, "url", None)
                    if not isinstance(successful_url, str) or not successful_url:
                        successful_url = url
                    resp.close()
                    # Atomic rename on complete, flushed payload write.
                    os.replace(partial_path, final_path)
                    sha = hasher.hexdigest()
                    return (
                        True,
                        ebook_id,
                        filename,
                        successful_url,
                        bytes_count,
                        sha,
                        None,
                        retries_used,
                        False,
                    )
                except Exception as e:
                    last_error = str(e)
                    if resp is not None:
                        try:
                            resp.close()
                        except Exception:
                            pass
                    if partial_path.exists():
                        try:
                            partial_path.unlink()
                        except Exception:
                            pass

            retries_used += 1
            time.sleep(0.5 * (2**attempt))

        return (
            False,
            ebook_id,
            filename,
            candidate_urls[0],
            0,
            None,
            last_error or "Download failed across all candidate URLs",
            retries_used,
            False,
        )

    def materialize(
        self,
        concurrency: int = 4,
        timeout: int = 25,
        max_retries: int = 3,
        ebook_ids: Optional[List[int]] = None,
        verify_live: bool = True,
        **kwargs,
    ) -> MaterializationManifest:
        """Acquire raw Gutenberg Portuguese text payloads reproducibly.

        Parameters
        ----------
        concurrency : int, default 4
            Worker thread count for HTTP downloads.
        timeout : int, default 25
            HTTP timeout per request in seconds.
        max_retries : int, default 3
            Maximum retries per book.
        ebook_ids : list of int or None, optional
            Optional override for eBook IDs list (used in testing).
        verify_live : bool, default True
            Whether to verify IDs against live catalog.
        **kwargs : any
            Additional options.

        Returns
        -------
        MaterializationManifest
            Populated and saved source manifest.
        """
        tool_git_commit = _get_clean_tool_git_commit()
        frozen_ids = self.resolve_ebook_ids(verify_live=verify_live)
        if ebook_ids is None:
            resolved_ids = frozen_ids
        else:
            requested_ids = [int(value) for value in ebook_ids]
            if len(requested_ids) != len(set(requested_ids)):
                raise ValueError("ebook_ids override contains duplicate IDs")
            if any(value <= 0 for value in requested_ids):
                raise ValueError("ebook_ids override must contain positive IDs")
            resolved_ids = sorted(requested_ids)
            if not self.allow_custom_destination and resolved_ids != frozen_ids:
                raise ValueError(
                    "Production materialization must use the authoritative frozen "
                    "Gutenberg snapshot ID set."
                )
        if not resolved_ids:
            raise ValueError("At least one Gutenberg eBook ID is required.")

        self.destination.mkdir(parents=True, exist_ok=True)
        manifest_path = self.destination / "manifest.json"
        in_progress_path = self.destination / "manifest.in_progress.json"
        ebook_ids_path = self.destination / "ebook_ids.json"
        started_at = datetime.now(timezone.utc).isoformat()

        def load_existing(path: Path, label: str) -> Optional[MaterializationManifest]:
            if not path.exists():
                return None
            try:
                return MaterializationManifest.load(path)
            except ValueError as exc:
                raise RuntimeError(
                    f"Cannot safely recover from invalid {label} {path}: {exc}"
                ) from exc

        trusted_records: Dict[int, Dict[str, Any]] = {}

        def add_trusted_records(prior: MaterializationManifest) -> None:
            if prior.source != self.source_name:
                raise RuntimeError(
                    f"Existing manifest source {prior.source!r} does not match "
                    f"{self.source_name!r}."
                )
            if (
                prior.ebook_ids != resolved_ids
                or prior.snapshot_date != self.snapshot_date
                or prior.pinned_revision
                != self.source_config.get("pinned_revision", "snapshot_2026-10-01")
                or prior.catalog_query != self.catalog_query
            ):
                return
            for record in prior.files:
                try:
                    item_id = int(record["upstream_identifier"])
                    relative_path = str(record["relative_path"])
                    expected_size = int(record["bytes"])
                    expected_sha = str(record["sha256"]).lower()
                    url = str(record["url"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Existing manifest contains an invalid file record: {record!r}"
                    ) from exc
                if not url or url == "None":
                    raise RuntimeError(
                        f"Existing manifest has no successful URL for {record!r}."
                    )
                if item_id not in resolved_ids:
                    continue
                expected_path = f"pg{item_id}.txt"
                if relative_path != expected_path:
                    raise RuntimeError(
                        f"Existing manifest maps Gutenberg ID {item_id} to unexpected "
                        f"path {relative_path!r}."
                    )
                if item_id in trusted_records:
                    existing = trusted_records[item_id]
                    if (
                        int(existing["bytes"]) != expected_size
                        or str(existing["sha256"]).lower() != expected_sha
                    ):
                        raise RuntimeError(
                            f"Existing manifests disagree about trusted payload {relative_path}."
                        )
                    continue
                file_path = self.destination / relative_path
                if file_path.exists():
                    actual_size = file_path.stat().st_size
                    actual_sha = compute_file_sha256(file_path).lower()
                    if actual_size != expected_size or actual_sha != expected_sha:
                        raise RuntimeError(
                            f"Existing payload {relative_path} does not match its trusted "
                            "manifest record. Refusing to bless local changes."
                        )
                trusted_records[item_id] = dict(record, url=url)

        previous_manifest = load_existing(manifest_path, "canonical manifest")
        if previous_manifest is not None and previous_manifest.status == "COMPLETE":
            valid, errors = previous_manifest.verify(
                self.destination,
                check_partial_files=False,
                check_runtime_state=False,
                check_ids_metadata=False,
            )
            if not valid:
                raise RuntimeError(
                    "Existing COMPLETE manifest failed integrity verification; "
                    "refusing to overwrite it: " + "; ".join(errors[:5])
                )
            add_trusted_records(previous_manifest)

        in_progress_manifest = load_existing(in_progress_path, "in-progress manifest")
        if (
            in_progress_manifest is not None
            and in_progress_manifest.status in {"PARTIAL", "FAILED"}
            and in_progress_manifest.tool_git_commit == tool_git_commit
            and in_progress_manifest.ebook_ids == resolved_ids
        ):
            add_trusted_records(in_progress_manifest)

        completed_records: Dict[str, MaterializedFileRecord] = {
            f"pg{item_id}.txt": MaterializedFileRecord(
                relative_path=f"pg{item_id}.txt",
                upstream_identifier=str(item_id),
                url=str(record["url"]),
                bytes=int(record["bytes"]),
                sha256=str(record["sha256"]),
                checksum_source=str(record.get("checksum_source", "local_sha256")),
            )
            for item_id, record in trusted_records.items()
            if (self.destination / f"pg{item_id}.txt").is_file()
        }

        manifest = MaterializationManifest(
            schema_version=1,
            source=self.source_name,
            upstream_repository=self.source_config.get(
                "repository", "project_gutenberg_pt"
            ),
            pinned_revision=self.source_config.get(
                "pinned_revision", "snapshot_2026-10-01"
            ),
            pinned_commit_sha=None,
            snapshot_date=self.snapshot_date,
            acquisition_started_at=started_at,
            status="PARTIAL",
            tool_git_commit=tool_git_commit,
            checksum_provenance="local_payload_sha256",
            ebook_ids=resolved_ids,
            catalog_snapshot_date=self.snapshot_date,
            catalog_query=self.catalog_query,
            catalog_size_ebooks=len(resolved_ids),
        )

        def checkpoint(status: str = "PARTIAL") -> None:
            files = [
                completed_records[key].to_dict() for key in sorted(completed_records)
            ]
            manifest.status = status
            manifest.files = files
            manifest.total_files = len(files)
            manifest.total_bytes = sum(record["bytes"] for record in files)
            manifest.save(in_progress_path)

        checkpoint()
        failed_ids: List[int] = []
        failure_reasons: Dict[str, str] = {}
        total_retries = 0
        newly_downloaded_bytes = 0
        cached_count = 0
        successful_since_checkpoint = 0
        concurrency = max(1, concurrency)
        worker_state = threading.local()
        session_lock = threading.Lock()
        worker_sessions: List[requests.Session] = []

        def worker_session() -> requests.Session:
            session = getattr(worker_state, "session", None)
            if session is None:
                session = requests.Session()
                session.headers.update(
                    {
                        "User-Agent": "cambacica-corpus-materialization/0.1.0 "
                        "(reproducible research pretraining)"
                    }
                )
                worker_state.session = session
                with session_lock:
                    worker_sessions.append(session)
            return session

        # Incomplete temp files are never trusted. The last atomically written
        # runtime checkpoint, if present, was loaded above.
        for stale_path in self.destination.glob("*.txt.partial"):
            stale_path.unlink(missing_ok=True)
        (self.destination / "manifest.in_progress.json.partial").unlink(missing_ok=True)

        def download_item(item_id: int):
            return self._download_single(
                item_id,
                worker_session(),
                timeout,
                max_retries,
                trusted_records.get(item_id),
            )

        try:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(download_item, item_id): item_id
                    for item_id in resolved_ids
                }
                for future in as_completed(futures):
                    item_id = futures[future]
                    try:
                        (
                            success,
                            result_id,
                            relative_path,
                            url,
                            byte_count,
                            sha,
                            error,
                            retries,
                            cached,
                        ) = future.result()
                        total_retries += retries
                        if success and sha is not None:
                            completed_records[relative_path] = MaterializedFileRecord(
                                relative_path=relative_path,
                                upstream_identifier=str(result_id),
                                url=url,
                                bytes=byte_count,
                                sha256=sha,
                                checksum_source="local_sha256",
                            )
                            if cached:
                                cached_count += 1
                            else:
                                newly_downloaded_bytes += byte_count
                            successful_since_checkpoint += 1
                            if successful_since_checkpoint >= 25:
                                checkpoint()
                                successful_since_checkpoint = 0
                        else:
                            failed_ids.append(result_id)
                            failure_reasons[str(result_id)] = error or "Unknown error"
                    except Exception as exc:
                        failed_ids.append(item_id)
                        failure_reasons[str(item_id)] = str(exc)
        finally:
            for session in worker_sessions:
                session.close()

        sorted_files = [
            completed_records[key].to_dict() for key in sorted(completed_records)
        ]
        record_ids = {int(record["upstream_identifier"]) for record in sorted_files}
        expected_ids = set(resolved_ids)
        is_complete = (
            record_ids == expected_ids
            and len(sorted_files) == len(expected_ids)
            and not failed_ids
        )

        manifest.acquisition_completed_at = datetime.now(timezone.utc).isoformat()
        manifest.status = "COMPLETE" if is_complete else "FAILED"
        manifest.files = sorted_files
        manifest.total_files = len(sorted_files)
        manifest.total_bytes = sum(record["bytes"] for record in sorted_files)
        manifest.failed_ids = sorted(failed_ids)
        manifest.failure_reasons = failure_reasons

        if is_complete:
            valid, errors = manifest.verify(
                self.destination,
                check_partial_files=False,
                check_runtime_state=False,
                check_ids_metadata=False,
            )
            if not valid:
                raise RuntimeError(
                    "Refusing to publish COMPLETE manifest because payload verification "
                    "failed: " + "; ".join(errors[:5])
                )
            manifest.save(manifest_path)
            _atomic_write_json(
                {
                    "schema_version": 1,
                    "source": self.source_name,
                    "snapshot_date": self.snapshot_date,
                    "catalog_query": self.catalog_query,
                    "catalog_size_ebooks": len(resolved_ids),
                    "ebook_ids": resolved_ids,
                },
                ebook_ids_path,
            )
            in_progress_path.unlink(missing_ok=True)
        else:
            checkpoint(status="FAILED")

        logger.info(
            f"Materialization finished: {manifest.status}. "
            f"Files: {manifest.total_files}/{len(resolved_ids)}, "
            f"Bytes: {manifest.total_bytes:,}, Cached: {cached_count}, "
            f"Downloaded: {newly_downloaded_bytes:,}, Retries: {total_retries}"
        )
        return manifest


class GenericStubMaterializer(BaseMaterializer):
    """Stub materializer for non-pilot sources during Gate C1.

    Enforces that sources other than the accepted pilot cannot perform live
    downloads before prior gate milestones complete.
    """

    def plan(self) -> Dict[str, Any]:
        """Produce dry-run execution plan from materialization YAML.

        Returns
        -------
        dict
            Configured source parameters, revisions, and destination.
        """
        return {
            "source": self.source_name,
            "canonical_name": self.source_config.get(
                "canonical_name", self.source_name
            ),
            "repository": self.source_config.get("repository"),
            "pinned_revision": self.source_config.get("pinned_revision"),
            "pinned_commit_sha": self.source_config.get("pinned_commit_sha"),
            "acquisition_mode": self.source_config.get("acquisition_mode"),
            "destination": str(self.destination),
            "estimated_raw_size": self.source_config.get("estimated_raw_size"),
            "checksum_provenance_requirements": self.source_config.get(
                "checksum_provenance_requirements", ""
            ).strip(),
            "status": "staged_pending_pilot_validation",
        }

    def materialize(self, **kwargs) -> MaterializationManifest:
        """Refuse live download execution during the Gutenberg pilot.

        Parameters
        ----------
        **kwargs : any
            Keyword arguments.

        Raises
        ------
        NotImplementedError
            Always raised to prevent unauthorized materialization.
        """
        raise NotImplementedError(
            f"Source '{self.source_name}' materialization is staged for subsequent execution. "
            "Gate C1 pilot materialization is strictly restricted to Gutenberg."
        )


class ParlamentoMaterializer(BaseMaterializer):
    def __init__(
        self,
        config_path="configs/corpus_materialization.yaml",
        destination_override=None,
        allow_custom_destination=False,
    ) -> None:
        super().__init__(
            source_name="parlamento_pt",
            config_path=config_path,
            destination_override=destination_override,
            allow_custom_destination=allow_custom_destination,
        )
        self.pinned_commit = self.source_config.get(
            "pinned_commit_sha", "08f13e7e63ab9bfbd8c0b40955defe3bb7f68c2b"
        )
        self.filename = "train.txt"
        self.url = f"https://huggingface.co/datasets/PORTULAN/parlamento-pt/resolve/{self.pinned_commit}/{self.filename}"
        self.expected_size = 2709043913
        self.upstream_blob_oid = "d01100ee7525d918539d8a2c2cea2836c7948191"

    def plan(self):
        """Generate a dry-run materialization plan.

        Returns
        -------
        dict
            Plan dictionary with all fields expected by the CLI dry-run display.
        """
        return {
            "source": self.source_name,
            "canonical_name": "PORTULAN/parlamento-pt",
            "repository": "PORTULAN/parlamento-pt",
            "pinned_revision": self.source_config.get("pinned_revision", "main"),
            "pinned_commit_sha": self.pinned_commit,
            "acquisition_mode": "single_file_download",
            "destination": str(self.destination),
            "artifact_name": self.filename,
            "artifact_url": self.url,
            "upstream_blob_oid": self.upstream_blob_oid,
            "estimated_raw_size": "~2.52 GiB (2,709,043,913 bytes)",
            "checksum_provenance_requirements": (
                "local SHA-256; upstream LFS blob OID recorded"
            ),
        }

    def materialize(self, concurrency=4, timeout=25, max_retries=3, **kwargs):
        tool_git_commit = _get_clean_tool_git_commit()
        self.destination.mkdir(parents=True, exist_ok=True)
        manifest_path = self.destination / "manifest.json"
        in_progress_path = self.destination / "manifest.in_progress.json"

        # Check existing complete manifest
        if manifest_path.is_file():
            prev = MaterializationManifest.load(manifest_path)
            if prev.status == "COMPLETE":
                valid, _ = prev.verify(self.destination)
                if valid:
                    return prev
                else:
                    raise RuntimeError("Existing COMPLETE manifest failed verify.")

        manifest = MaterializationManifest(
            source=self.source_name,
            upstream_repository=self.source_config.get(
                "repository", "PORTULAN/parlamento-pt"
            ),
            pinned_revision=self.source_config.get("pinned_revision", "main"),
            pinned_commit_sha=self.pinned_commit,
            tool_git_commit=tool_git_commit,
            status="PARTIAL",
            raw_artifact_name=self.filename,
            upstream_blob_oid=self.upstream_blob_oid,
        )

        final_path = self.destination / self.filename
        partial_path = self.destination / f"{self.filename}.partial"

        if final_path.exists():
            raise RuntimeError("File exists but manifest not complete/valid.")

        manifest.save(in_progress_path)

        import requests
        import time

        session = requests.Session()
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = session.get(self.url, timeout=timeout, stream=True)
                resp.raise_for_status()
                hasher = hashlib.sha256()
                bytes_count = 0
                lines_count = 0
                with open(partial_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1048576):
                        if chunk:
                            f.write(chunk)
                            hasher.update(chunk)
                            bytes_count += len(chunk)
                            lines_count += chunk.count(b"\n")
                    f.flush()
                    os.fsync(f.fileno())

                os.replace(partial_path, final_path)
                sha = hasher.hexdigest()
                manifest.line_count = lines_count

                record = MaterializedFileRecord(
                    relative_path=self.filename,
                    upstream_identifier=self.filename,
                    url=self.url,
                    bytes=bytes_count,
                    sha256=sha,
                    checksum_source="local_sha256",
                )
                manifest.files = [record.to_dict()]
                manifest.total_files = 1
                manifest.total_bytes = bytes_count
                manifest.status = "COMPLETE"
                manifest.acquisition_completed_at = datetime.now(
                    timezone.utc
                ).isoformat()

                valid, errors = manifest.verify(
                    self.destination,
                    check_partial_files=False,
                    check_runtime_state=False,
                )
                if not valid:
                    raise RuntimeError(f"Verify failed: {errors}")

                manifest.save(manifest_path)
                in_progress_path.unlink(missing_ok=True)
                return manifest

            except Exception as e:
                last_error = e
                if partial_path.exists():
                    partial_path.unlink()
                time.sleep(1)

        manifest.status = "FAILED"
        manifest.failure_reasons = {self.filename: str(last_error)}
        manifest.save(in_progress_path)
        return manifest


WIKIPEDIA_PT_SHARDS = [
    {
        "name": "train-00000-of-00006.parquet",
        "upstream_oid": "eebdb1cbfe35d9ab278616f374a2f3071a4e5678",
        "expected_bytes": 425972594,
    },
    {
        "name": "train-00001-of-00006.parquet",
        "upstream_oid": "37f1eb91948e151a4e1eb789517533a6f3ab307e",
        "expected_bytes": 212472795,
    },
    {
        "name": "train-00002-of-00006.parquet",
        "upstream_oid": "c2f09e8490e502bbd20f5d3e9413c68f9d902985",
        "expected_bytes": 202884385,
    },
    {
        "name": "train-00003-of-00006.parquet",
        "upstream_oid": "3ab581d3ad28a06a81d65923c5f6bfe2d2af1ae3",
        "expected_bytes": 218920129,
    },
    {
        "name": "train-00004-of-00006.parquet",
        "upstream_oid": "f3d7b2f3f9b581daeb94c814fbc087c5c56ae645",
        "expected_bytes": 227282628,
    },
    {
        "name": "train-00005-of-00006.parquet",
        "upstream_oid": "964738db11fc45435a7eafd04c79a7d96e95a923",
        "expected_bytes": 292108528,
    },
]


class WikipediaMaterializer(BaseMaterializer):
    def __init__(
        self,
        config_path="configs/corpus_materialization.yaml",
        destination_override=None,
        allow_custom_destination=False,
    ) -> None:
        super().__init__(
            source_name="wikipedia_pt",
            config_path=config_path,
            destination_override=destination_override,
            allow_custom_destination=allow_custom_destination,
        )
        self.pinned_commit = self.source_config.get(
            "pinned_commit_sha", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
        )
        self.config_dir = "20231101.pt"
        self.base_url = f"https://huggingface.co/datasets/wikimedia/wikipedia/resolve/{self.pinned_commit}/{self.config_dir}/"

    def plan(self):
        """Generate a dry-run materialization plan.

        Returns
        -------
        dict
            Plan dictionary with all fields expected by the CLI dry-run display.
        """
        total_bytes = sum(s["expected_bytes"] for s in WIKIPEDIA_PT_SHARDS)
        return {
            "source": self.source_name,
            "canonical_name": "wikimedia/wikipedia",
            "repository": "wikimedia/wikipedia",
            "pinned_revision": self.config_dir,
            "pinned_commit_sha": self.pinned_commit,
            "acquisition_mode": "hf_parquet_snapshot",
            "destination": str(self.destination),
            "dataset_config": self.config_dir,
            "snapshot_identifier": self.config_dir,
            "shard_count": len(WIKIPEDIA_PT_SHARDS),
            "shards": [s["name"] for s in WIKIPEDIA_PT_SHARDS],
            "upstream_shard_oids": {
                s["name"]: s["upstream_oid"] for s in WIKIPEDIA_PT_SHARDS
            },
            "estimated_raw_size": (
                f"~{total_bytes / (1024**3):.2f} GiB ({total_bytes:,} bytes)"
            ),
            "checksum_provenance_requirements": (
                "local SHA-256; upstream LFS OID recorded per shard"
            ),
        }

    def materialize(self, concurrency=4, timeout=25, max_retries=3, **kwargs):
        tool_git_commit = _get_clean_tool_git_commit()
        self.destination.mkdir(parents=True, exist_ok=True)
        (self.destination / self.config_dir).mkdir(parents=True, exist_ok=True)
        manifest_path = self.destination / "manifest.json"
        in_progress_path = self.destination / "manifest.in_progress.json"

        trusted_records = {}

        # Check existing complete manifest
        if manifest_path.is_file():
            prev = MaterializationManifest.load(manifest_path)
            if prev.status == "COMPLETE":
                valid, _ = prev.verify(
                    self.destination,
                    check_partial_files=False,
                    check_runtime_state=False,
                )
                if valid:
                    return prev
                else:
                    raise RuntimeError("Existing COMPLETE manifest failed verify.")

        if in_progress_path.is_file():
            prev = MaterializationManifest.load(in_progress_path)
            if prev.tool_git_commit == tool_git_commit:
                for f in prev.files:
                    trusted_records[f["upstream_identifier"]] = f

        upstream_oids = {s["name"]: s["upstream_oid"] for s in WIKIPEDIA_PT_SHARDS}
        manifest = MaterializationManifest(
            source=self.source_name,
            upstream_repository=self.source_config.get(
                "repository", "wikimedia/wikipedia"
            ),
            pinned_revision=self.source_config.get("pinned_revision", "20231101.pt"),
            pinned_commit_sha=self.pinned_commit,
            tool_git_commit=tool_git_commit,
            status="PARTIAL",
            dataset_config=self.config_dir,
            snapshot_identifier=self.config_dir,
            upstream_shard_oids=upstream_oids,
        )

        completed_files = []
        failure_reasons = {}

        def download_shard(shard_info):
            name = shard_info["name"]
            url = self.base_url + name
            rel_path = f"{self.config_dir}/{name}"
            final_path = self.destination / rel_path
            partial_path = self.destination / f"{rel_path}.partial"

            if final_path.is_file() and name in trusted_records:
                r = trusted_records[name]
                if (
                    final_path.stat().st_size == r["bytes"]
                    and compute_file_sha256(final_path) == r["sha256"]
                ):
                    return True, name, r
                else:
                    raise RuntimeError("Changed/corrupted payload")

            import requests
            import time

            session = requests.Session()
            last_err = None
            for attempt in range(max_retries):
                try:
                    resp = session.get(url, timeout=timeout, stream=True)
                    resp.raise_for_status()
                    hasher = hashlib.sha256()
                    bytes_count = 0
                    with open(partial_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1048576):
                            if chunk:
                                f.write(chunk)
                                hasher.update(chunk)
                                bytes_count += len(chunk)
                        f.flush()
                        os.fsync(f.fileno())

                    os.replace(partial_path, final_path)
                    sha = hasher.hexdigest()

                    record = MaterializedFileRecord(
                        relative_path=rel_path,
                        upstream_identifier=name,
                        url=url,
                        bytes=bytes_count,
                        sha256=sha,
                        checksum_source="local_sha256",
                    )
                    return True, name, record.to_dict()
                except Exception as e:
                    last_err = e
                    if partial_path.exists():
                        partial_path.unlink()
                    time.sleep(1)

            return False, name, str(last_err)

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(download_shard, s) for s in WIKIPEDIA_PT_SHARDS]
            for fut in concurrent.futures.as_completed(futures):
                success, name, res = fut.result()
                if success:
                    completed_files.append(res)
                else:
                    # Wikipedia shard names are not integer IDs; track failures only in
                    # failure_reasons, not failed_ids (which is reserved for integer keys).
                    failure_reasons[name] = res

        manifest.files = sorted(completed_files, key=lambda r: r["relative_path"])
        manifest.total_files = len(completed_files)
        manifest.total_bytes = sum(f["bytes"] for f in completed_files)
        manifest.failure_reasons = failure_reasons

        if len(completed_files) == len(WIKIPEDIA_PT_SHARDS) and not failure_reasons:
            manifest.status = "COMPLETE"
            manifest.acquisition_completed_at = datetime.now(timezone.utc).isoformat()
            # Verify BEFORE saving as COMPLETE — a failed verify must not persist.
            valid, errs = manifest.verify(
                self.destination, check_partial_files=False, check_runtime_state=False
            )
            if not valid:
                manifest.status = "FAILED"
                manifest.save(in_progress_path)
                raise RuntimeError(
                    "Refusing to publish COMPLETE Wikipedia manifest because payload "
                    "verification failed: " + "; ".join(errs[:5])
                )
            manifest.save(manifest_path)
            in_progress_path.unlink(missing_ok=True)
        else:
            manifest.status = "FAILED"
            manifest.save(in_progress_path)

        return manifest


CAROLINA_TAXONOMY_DIRS = {
    "dat": "corpus/datasets_and_other_corpora",
    "jud": "corpus/judicial_branch",
    "leg": "corpus/legislative_branch",
    "pub": "corpus/public_domain_works",
    "soc": "corpus/social_media",
    "uni": "corpus/university_domains",
    "wik": "corpus/wikis",
}


class CarolinaMaterializer(BaseMaterializer):
    def __init__(
        self,
        config_path="configs/corpus_materialization.yaml",
        destination_override=None,
        allow_custom_destination=False,
    ) -> None:
        super().__init__(
            source_name="carolina",
            config_path=config_path,
            destination_override=destination_override,
            allow_custom_destination=allow_custom_destination,
        )
        self.pinned_commit = self.source_config.get(
            "pinned_commit_sha", "55e63a519393c70a48dcfa14a558499c6bb0583b"
        )
        self.api_base = (
            "https://huggingface.co/api/datasets/carolina-c4ai/corpus-carolina/tree"
        )
        self.resolve_base = (
            "https://huggingface.co/datasets/carolina-c4ai/corpus-carolina/resolve"
        )

    def plan(self):
        """Generate a dry-run materialization plan.

        Returns
        -------
        dict
            Plan dictionary with all fields expected by the CLI dry-run display.
        """
        # Expected .xml.gz file counts per taxonomy from upstream audit (v2.0.1).
        expected_file_counts = {
            "dat": 153,
            "jud": 37,
            "leg": 161,
            "pub": 1,
            "soc": 2,
            "uni": 7,
            "wik": 193,
        }
        total_xml = sum(expected_file_counts.values())
        total_artifacts = total_xml + len(CAROLINA_TAXONOMY_DIRS)  # + checksum files
        return {
            "source": self.source_name,
            "canonical_name": "carolina-c4ai/corpus-carolina",
            "repository": "carolina-c4ai/corpus-carolina",
            "pinned_revision": self.source_config.get("pinned_revision", "v2.0.1"),
            "pinned_commit_sha": self.pinned_commit,
            "acquisition_mode": "full_corpus_acquisition",
            "destination": str(self.destination),
            "taxonomies": list(CAROLINA_TAXONOMY_DIRS.keys()),
            "expected_xml_gz_files": total_xml,
            "expected_checksum_files": len(CAROLINA_TAXONOMY_DIRS),
            "expected_total_artifacts": total_artifacts,
            "expected_taxonomy_file_counts": expected_file_counts,
            "estimated_raw_size": "~3.10 GiB (3,106,333,225 bytes compressed)",
            "checksum_provenance_requirements": (
                "local SHA-256; upstream checksum.sha256 files per taxonomy acquired"
            ),
        }

    def _list_files(self, path):
        import requests

        url = f"{self.api_base}/{self.pinned_commit}/{path}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        files = []
        for item in r.json():
            if item["type"] == "directory":
                files.extend(self._list_files(item["path"]))
            elif item["type"] == "file":
                files.append(item["path"])
        return files

    def materialize(self, concurrency=4, timeout=25, max_retries=3, **kwargs):
        tool_git_commit = _get_clean_tool_git_commit()
        self.destination.mkdir(parents=True, exist_ok=True)
        manifest_path = self.destination / "manifest.json"
        in_progress_path = self.destination / "manifest.in_progress.json"

        trusted_records = {}
        file_list_to_download = []

        if manifest_path.is_file():
            prev = MaterializationManifest.load(manifest_path)
            if prev.status == "COMPLETE":
                valid, _ = prev.verify(
                    self.destination,
                    check_partial_files=False,
                    check_runtime_state=False,
                )
                if valid:
                    return prev
                else:
                    raise RuntimeError("Existing COMPLETE manifest failed verify.")

        if in_progress_path.is_file():
            prev = MaterializationManifest.load(in_progress_path)
            if prev.tool_git_commit == tool_git_commit:
                for f in prev.files:
                    trusted_records[f["relative_path"]] = f

        if not file_list_to_download:
            for k, dirpath in CAROLINA_TAXONOMY_DIRS.items():
                file_list_to_download.extend(self._list_files(dirpath))

        manifest = MaterializationManifest(
            source=self.source_name,
            upstream_repository=self.source_config.get(
                "repository", "carolina-c4ai/corpus-carolina"
            ),
            pinned_revision=self.source_config.get("pinned_revision", "v2.0.1"),
            pinned_commit_sha=self.pinned_commit,
            tool_git_commit=tool_git_commit,
            status="PARTIAL",
            taxonomy_file_counts={},
            taxonomy_checksum_files={},
        )

        completed_files = []
        failure_reasons = {}

        def download_file(hf_path):
            url = f"{self.resolve_base}/{self.pinned_commit}/{hf_path}"
            rel_path = hf_path
            final_path = self.destination / rel_path
            final_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path = self.destination / f"{rel_path}.partial"

            if final_path.is_file() and rel_path in trusted_records:
                r = trusted_records[rel_path]
                if (
                    final_path.stat().st_size == r["bytes"]
                    and compute_file_sha256(final_path) == r["sha256"]
                ):
                    return True, rel_path, r
                else:
                    raise RuntimeError("Changed/corrupted payload")

            import requests
            import time

            session = requests.Session()
            last_err = None
            for attempt in range(max_retries):
                try:
                    resp = session.get(url, timeout=timeout, stream=True)
                    resp.raise_for_status()
                    hasher = hashlib.sha256()
                    bytes_count = 0
                    with open(partial_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1048576):
                            if chunk:
                                f.write(chunk)
                                hasher.update(chunk)
                                bytes_count += len(chunk)
                        f.flush()
                        os.fsync(f.fileno())

                    os.replace(partial_path, final_path)
                    sha = hasher.hexdigest()

                    record = MaterializedFileRecord(
                        relative_path=rel_path,
                        upstream_identifier=rel_path,
                        url=url,
                        bytes=bytes_count,
                        sha256=sha,
                        checksum_source="local_sha256",
                    )
                    return True, rel_path, record.to_dict()
                except Exception as e:
                    last_err = e
                    if partial_path.exists():
                        partial_path.unlink()
                    time.sleep(1)

            return False, rel_path, str(last_err)

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(download_file, p) for p in file_list_to_download]
            for fut in concurrent.futures.as_completed(futures):
                success, rp, res = fut.result()
                if success:
                    completed_files.append(res)
                else:
                    # Carolina paths are strings; track failures only in failure_reasons.
                    failure_reasons[rp] = res

        # Sort deterministically before writing manifest.
        manifest.files = sorted(completed_files, key=lambda r: r["relative_path"])
        manifest.total_files = len(completed_files)
        manifest.total_bytes = sum(f["bytes"] for f in completed_files)
        manifest.failure_reasons = failure_reasons

        # Compute taxonomy counts from acquired files.
        for f in completed_files:
            rp = f["relative_path"]
            if rp.endswith(".xml.gz"):
                for tax, prefix in CAROLINA_TAXONOMY_DIRS.items():
                    if rp.startswith(prefix):
                        manifest.taxonomy_file_counts[tax] = (
                            manifest.taxonomy_file_counts.get(tax, 0) + 1
                        )
            elif rp.endswith("checksum.sha256"):
                for tax, prefix in CAROLINA_TAXONOMY_DIRS.items():
                    if rp.startswith(prefix):
                        manifest.taxonomy_checksum_files[tax] = rp

        if len(completed_files) == len(file_list_to_download) and not failure_reasons:
            manifest.status = "COMPLETE"
            manifest.acquisition_completed_at = datetime.now(timezone.utc).isoformat()
            # Verify BEFORE saving as COMPLETE — a failed verify must not persist.
            valid, errs = manifest.verify(
                self.destination, check_partial_files=False, check_runtime_state=False
            )
            if not valid:
                manifest.status = "FAILED"
                manifest.save(in_progress_path)
                raise RuntimeError(
                    "Refusing to publish COMPLETE Carolina manifest because payload "
                    "verification failed: " + "; ".join(errs[:5])
                )
            manifest.save(manifest_path)
            in_progress_path.unlink(missing_ok=True)
        else:
            manifest.status = "FAILED"
            manifest.save(in_progress_path)

        return manifest


MATERIALIZER_REGISTRY = {
    "gutenberg_pt": GutenbergMaterializer,
    "gutenberg": GutenbergMaterializer,
    "carolina": CarolinaMaterializer,
    "wikipedia_pt": WikipediaMaterializer,
    "wikipedia": WikipediaMaterializer,
    "parlamento_pt": ParlamentoMaterializer,
    "parlamento": ParlamentoMaterializer,
    "gigaverbo_v2": GenericStubMaterializer,
    "gigaverbo": GenericStubMaterializer,
}


def get_materializer(
    source: str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    destination: Optional[Path | str] = None,
    allow_custom_destination: bool = False,
) -> BaseMaterializer:
    key = source.lower().replace("-", "_")
    cls = MATERIALIZER_REGISTRY.get(key)
    if not cls:
        raise ValueError(
            f"Unknown source '{source}'. Supported: {sorted(MATERIALIZER_REGISTRY.keys())}"
        )
    return cls(
        config_path=config_path,
        destination_override=destination,
        allow_custom_destination=allow_custom_destination,
    )
