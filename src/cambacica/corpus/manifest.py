"""Provenance manifest generation and serialization for Gate C1 samples.

Ensures that every generated sample Parquet file is accompanied by a
detailed manifest documenting source revisions, git commits, seeds,
exact document counts, and file checksums.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, Optional


def get_git_commit() -> Optional[str]:
    """Attempt to resolve the current git commit SHA.

    Returns
    -------
    str or None
        Short commit SHA if inside a git repository, else None.
    """
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


def compute_file_sha256(file_path: Path | str) -> str:
    """Compute the SHA-256 checksum of a file on disk.

    Parameters
    ----------
    file_path : Path or str
        Path to the file to hash.

    Returns
    -------
    str
        Hexadecimal representation of the SHA-256 hash.
    """
    path = Path(file_path)
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class ProvenanceManifest:
    """Provenance metadata container for a Gate C1 sample.

    Parameters
    ----------
    source : str
        Canonical source identifier (e.g. 'carolina', 'gigaverbo_v2').
    mode : str
        Sampling mode ('representative', 'diagnostic', 'audit', 'candidate').
    target_size : int
        Requested target document count.
    document_count : int
        Actual number of documents included in the sample.
    seed : int
        Deterministic random seed used for sampling.
    upstream_identifier : str
        Canonical repository, dataset ID, or collection name.
    upstream_revision : str or None, optional
        Human-readable version tag, config name, or dump snapshot date.
    upstream_commit_sha : str or None, optional
        Immutable 40-character Git commit hash when available.
    upstream_url : str or None, optional
        Official repository or download URL.
    upstream_configuration : str or None, optional
        Sub-configuration or split name if applicable.
    population_scope : str or None, optional
        Description of total upstream population / corpus volume.
    sampling_frame : str or None, optional
        Explicit sampling frame used (e.g. partition, reservoir, bounded scan).
    records_examined : int or None, optional
        Total number of raw upstream records evaluated during sampling.
    bytes_read : int or None, optional
        Approximate network/disk bytes transferred during sampling.
    stopping_reason : str or None, optional
        Stopping condition triggered during sampling.
    exclusion_config_hash : str or None, optional
        SHA-256 hash of the exclusions config file (e.g. for GigaVerbo).
    output_parquet : str
        Filename of the associated Parquet output file.
    parquet_sha256 : str
        SHA-256 checksum of the output Parquet file.
    parquet_bytes : int
        File size in bytes of the output Parquet file.
    script_version : str, default "0.1.0"
        Version of the sampling script/library.
    git_commit : str or None, optional
        Git commit hash at time of sample generation.
    created_at_utc : str, optional
        ISO 8601 UTC timestamp of creation.
    stats : dict, optional
        Summary metrics of the sample (total chars, total words, etc.).
    """

    source: str
    mode: str
    target_size: int
    document_count: int
    seed: int
    upstream_identifier: str
    output_parquet: str
    parquet_sha256: str
    parquet_bytes: int
    upstream_revision: Optional[str] = None
    upstream_commit_sha: Optional[str] = None
    upstream_url: Optional[str] = None
    upstream_configuration: Optional[str] = None
    population_scope: Optional[str] = None
    sampling_frame: Optional[str] = None
    records_examined: Optional[int] = None
    bytes_read: Optional[int] = None
    stopping_reason: Optional[str] = None
    exclusion_config_hash: Optional[str] = None
    script_version: str = "0.1.0"
    git_commit: Optional[str] = None
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to a serializable dictionary.

        Returns
        -------
        dict
            Dictionary representation of the manifest.
        """
        return asdict(self)

    def save(self, output_path: Path | str) -> Path:
        """Write manifest to a formatted JSON file.

        Parameters
        ----------
        output_path : Path or str
            Destination JSON file path.

        Returns
        -------
        Path
            Resolved output file path.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, input_path: Path | str) -> ProvenanceManifest:
        """Load manifest from a JSON file.

        Parameters
        ----------
        input_path : Path or str
            Path to the JSON manifest.

        Returns
        -------
        ProvenanceManifest
            Deserialized manifest object.
        """
        path = Path(input_path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
