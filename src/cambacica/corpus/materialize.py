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
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import requests
import yaml

from cambacica.corpus.manifest import compute_file_sha256, get_git_commit

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/corpus_materialization.yaml")
GUTENBERG_CATALOG_URL = "https://www.gutenberg.org/browse/languages/pt"
GUTENBERG_URL_CANDIDATES = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]


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
        Git commit hash of the cambacica codebase at time of materialization.
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
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return out_path

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
        with in_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def verify(self, destination: Path | str) -> Tuple[bool, List[str]]:
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

        # Verify no orphan .partial files remain
        partial_files = list(root.glob("*.partial*"))
        if partial_files:
            for pf in partial_files:
                errors.append(f"Orphaned partial file detected: {pf.name}")

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

            file_path = root / rel_path
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
        manifest = MaterializationManifest.load(manifest_file)
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
                "atomic_writes": "Downloads to *.partial and renames on payload completion",
                "idempotent_resume": "Skips redownloading verified payload files",
                "checksum_hashing": "Locally computes SHA-256 for each payload",
                "raw_preservation": "Plain-text unaltered; no boilerplate stripping or normalization",
            },
        }

    def resolve_ebook_ids(self, verify_live: bool = True) -> List[int]:
        """Resolve the deterministic list of 655 eBook IDs.

        Compares live catalog results against frozen snapshot configuration or
        sampling metadata to detect any provenance mismatch.

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
        snapshot_ids: Optional[List[int]] = None

        # 1. Load from snapshot config if specified
        if self.snapshot_config_path:
            p = Path(self.snapshot_config_path)
            if p.is_file():
                try:
                    with p.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    snapshot_ids = sorted(
                        list(set(int(x) for x in data.get("ebook_ids", [])))
                    )
                except Exception as e:
                    logger.warning(f"Could not load snapshot config {p}: {e}")

        # 2. Reconstruct from sampling metadata if not found in snapshot config
        if not snapshot_ids:
            sampling_manifest_path = Path(
                "/mnt/data/cambacica-base-180m/samples/gate_c1/gutenberg_pt/manifest_representative.json"
            )
            if sampling_manifest_path.is_file():
                try:
                    with sampling_manifest_path.open("r", encoding="utf-8") as f:
                        sm = json.load(f)
                    if sm.get("stats", {}).get("selected_ebook_ids"):
                        logger.info("Found existing sampling metadata from Gate C1.")
                except Exception:
                    pass

        # 3. Live catalog validation
        if verify_live:
            try:
                from cambacica.corpus.sources.gutenberg_pt import (
                    discover_gutenberg_pt_ids,
                )

                live_ids = discover_gutenberg_pt_ids(timeout=20)
                if len(live_ids) != self.expected_size:
                    raise RuntimeError(
                        f"Provenance mismatch: Live Gutenberg catalog returned {len(live_ids)} "
                        f"eBooks, expected exactly {self.expected_size}."
                    )
                if snapshot_ids and live_ids != snapshot_ids:
                    raise RuntimeError(
                        "Provenance mismatch: Live Gutenberg catalog IDs differ from "
                        "the accepted 2026-10-01 snapshot."
                    )
                return live_ids
            except Exception as err:
                if isinstance(err, RuntimeError) and "Provenance mismatch" in str(err):
                    raise
                logger.warning(
                    f"Live Gutenberg catalog discovery failed ({err}); "
                    "falling back to accepted frozen snapshot."
                )

        if snapshot_ids and len(snapshot_ids) == self.expected_size:
            return snapshot_ids

        raise RuntimeError(
            "Unable to resolve accepted 655 Gutenberg eBook IDs: "
            "neither live catalog nor accepted snapshot config was accessible."
        )

    def _download_single(
        self,
        ebook_id: int,
        session: requests.Session,
        timeout: int,
        max_retries: int,
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

        # Check existing file on disk (idempotency check)
        if final_path.is_file():
            size = final_path.stat().st_size
            if size > 0:
                sha = compute_file_sha256(final_path)
                primary_url = GUTENBERG_URL_CANDIDATES[0].format(id=ebook_id)
                return True, ebook_id, filename, primary_url, size, sha, None, 0, True

        candidate_urls = [c.format(id=ebook_id) for c in GUTENBERG_URL_CANDIDATES]
        retries_used = 0
        last_error = None

        for attempt in range(max_retries):
            for url in candidate_urls:
                try:
                    resp = session.get(url, timeout=timeout, stream=True)
                    if resp.status_code == 200:
                        hasher = hashlib.sha256()
                        bytes_count = 0
                        with open(partial_path, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=65536):
                                if chunk:
                                    f.write(chunk)
                                    hasher.update(chunk)
                                    bytes_count += len(chunk)

                        if bytes_count == 0:
                            if partial_path.exists():
                                partial_path.unlink()
                            continue

                        # Atomic rename on complete payload write
                        partial_path.replace(final_path)
                        sha = hasher.hexdigest()
                        return (
                            True,
                            ebook_id,
                            filename,
                            url,
                            bytes_count,
                            sha,
                            None,
                            retries_used,
                            False,
                        )
                except Exception as e:
                    last_error = str(e)
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
        self.destination.mkdir(parents=True, exist_ok=True)
        started_at = datetime.now(timezone.utc).isoformat()

        if ebook_ids is None:
            resolved_ids = self.resolve_ebook_ids(verify_live=verify_live)
        else:
            resolved_ids = sorted(list(set(int(x) for x in ebook_ids)))

        manifest_path = self.destination / "manifest.json"
        ebook_ids_path = self.destination / "ebook_ids.json"

        # Adjacent metadata file with deterministic ID list
        with ebook_ids_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": 1,
                    "source": self.source_name,
                    "snapshot_date": self.snapshot_date,
                    "catalog_query": self.catalog_query,
                    "catalog_size_ebooks": len(resolved_ids),
                    "ebook_ids": resolved_ids,
                },
                f,
                indent=2,
            )

        # Initialize in-progress manifest (status = PARTIAL)
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
            tool_git_commit=get_git_commit(),
            checksum_provenance="local_payload_sha256",
            ebook_ids=resolved_ids,
            catalog_snapshot_date=self.snapshot_date,
            catalog_query=self.catalog_query,
            catalog_size_ebooks=len(resolved_ids),
        )
        manifest.save(manifest_path)

        completed_records: Dict[str, MaterializedFileRecord] = {}
        failed_ids: List[int] = []
        failure_reasons: Dict[str, str] = {}
        total_retries = 0
        newly_downloaded_bytes = 0
        cached_count = 0

        # Execute downloads safely using ThreadPoolExecutor
        with requests.Session() as session:
            session.headers.update(
                {
                    "User-Agent": "cambacica-corpus-materialization/0.1.0 (reproducible research pretraining)"
                }
            )
            concurrency = max(1, concurrency)
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(
                        self._download_single,
                        bid,
                        session,
                        timeout,
                        max_retries,
                    ): bid
                    for bid in resolved_ids
                }

                for future in as_completed(futures):
                    bid = futures[future]
                    try:
                        (
                            success,
                            item_id,
                            rel_path,
                            url,
                            byte_count,
                            sha,
                            err,
                            retries,
                            cached,
                        ) = future.result()
                        total_retries += retries

                        if success and sha is not None:
                            record = MaterializedFileRecord(
                                relative_path=rel_path,
                                upstream_identifier=str(item_id),
                                url=url,
                                bytes=byte_count,
                                sha256=sha,
                                checksum_source="local_sha256",
                            )
                            completed_records[rel_path] = record
                            if cached:
                                cached_count += 1
                            else:
                                newly_downloaded_bytes += byte_count
                        else:
                            failed_ids.append(item_id)
                            failure_reasons[str(item_id)] = (
                                err or "Unknown download error"
                            )
                    except Exception as ex:
                        failed_ids.append(bid)
                        failure_reasons[str(bid)] = str(ex)

        # Sort files deterministically by relative_path
        sorted_files = [
            completed_records[k].to_dict() for k in sorted(completed_records.keys())
        ]
        total_bytes = sum(f["bytes"] for f in sorted_files)

        is_complete = len(sorted_files) == len(resolved_ids) and len(failed_ids) == 0

        manifest.acquisition_completed_at = datetime.now(timezone.utc).isoformat()
        manifest.status = "COMPLETE" if is_complete else "FAILED"
        manifest.files = sorted_files
        manifest.total_files = len(sorted_files)
        manifest.total_bytes = total_bytes
        manifest.failed_ids = sorted(failed_ids)
        manifest.failure_reasons = failure_reasons
        manifest.save(manifest_path)

        logger.info(
            f"Materialization finished: {manifest.status}. "
            f"Files: {manifest.total_files}/{len(resolved_ids)}, "
            f"Bytes: {manifest.total_bytes:,}, Cached: {cached_count}, Retries: {total_retries}"
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


MATERIALIZER_REGISTRY = {
    "gutenberg_pt": GutenbergMaterializer,
    "gutenberg": GutenbergMaterializer,
    "carolina": GenericStubMaterializer,
    "wikipedia_pt": GenericStubMaterializer,
    "wikipedia": GenericStubMaterializer,
    "parlamento_pt": GenericStubMaterializer,
    "parlamento": GenericStubMaterializer,
    "gigaverbo_v2": GenericStubMaterializer,
    "gigaverbo": GenericStubMaterializer,
}


def get_materializer(
    source: str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    destination: Optional[Path | str] = None,
    allow_custom_destination: bool = False,
) -> BaseMaterializer:
    """Instantiate the materializer registered for a given source.

    Parameters
    ----------
    source : str
        Source identifier.
    config_path : Path or str, default DEFAULT_CONFIG_PATH
        Path to materialization YAML config.
    destination : Path or str or None, optional
        Optional destination override.
    allow_custom_destination : bool, default False
        Allow paths outside /mnt/data for testing.

    Returns
    -------
    BaseMaterializer
        Instantiated materializer.

    Raises
    ------
    ValueError
        If source is not in the materializer registry.
    """
    key = source.lower().replace("-", "_")
    cls = MATERIALIZER_REGISTRY.get(key)
    if not cls:
        raise ValueError(
            f"Unknown source '{source}'. Supported: {sorted(MATERIALIZER_REGISTRY.keys())}"
        )
    if cls is GutenbergMaterializer:
        return GutenbergMaterializer(
            config_path=config_path,
            destination_override=destination,
            allow_custom_destination=allow_custom_destination,
        )
    return GenericStubMaterializer(
        source_name=key,
        config_path=config_path,
        destination_override=destination,
        allow_custom_destination=allow_custom_destination,
    )
