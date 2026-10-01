"""Base class and common utilities for Gate C1 source samplers.

Defines the common interface, safety limits, cache directory configuration,
and output persistence patterns for all source-specific samplers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import List, Optional, Tuple

from cambacica.corpus.manifest import (
    ProvenanceManifest,
    compute_file_sha256,
    get_git_commit,
)
from cambacica.corpus.schema import (
    NormalizedDocument,
    save_sample_parquet,
)


def ensure_user_hf_cache() -> None:
    """Ensure Hugging Face cache directories are user-writable.

    Sets HF_HOME and HF_DATASETS_CACHE to user-owned directories to prevent
    permission errors when system cache directories are owned by root.
    """
    user_home = Path.home() / ".cache" / "huggingface_user"
    user_datasets = user_home / "datasets"
    user_datasets.mkdir(parents=True, exist_ok=True)

    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(user_home)
    if "HF_DATASETS_CACHE" not in os.environ:
        os.environ["HF_DATASETS_CACHE"] = str(user_datasets)

    try:
        import datasets.config
        datasets.config.HF_DATASETS_CACHE = user_datasets
    except Exception:
        pass


def resolve_hf_commit_sha(
    repo_id: str,
    revision: Optional[str] = None,
) -> Optional[str]:
    """Resolve the immutable commit SHA of a Hugging Face dataset.

    Parameters
    ----------
    repo_id : str
        Hugging Face dataset identifier (e.g. 'carolina-c4ai/corpus-carolina').
    revision : str or None, optional
        Revision, branch, or tag name.

    Returns
    -------
    str or None
        Immutable 40-character commit SHA, or None if resolution fails.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.dataset_info(repo_id, revision=revision, timeout=10)
        return getattr(info, "sha", None)
    except Exception:
        return None


class BaseSourceSampler(ABC):
    """Abstract base class for Gate C1 corpus source samplers.

    Parameters
    ----------
    source_name : str
        Canonical source identifier (e.g. 'carolina', 'wikipedia_pt').
    config : dict or None, optional
        Source-specific configuration dictionary from corpus_sources.yaml.
    """

    def __init__(
        self,
        source_name: str,
        config: Optional[dict] = None,
    ) -> None:
        self.source_name: str = source_name
        self.config: dict = config or {}
        ensure_user_hf_cache()

    @abstractmethod
    def sample(
        self,
        mode: str,
        size: int,
        seed: int = 42,
        output_dir: Optional[Path | str] = None,
        **kwargs,
    ) -> Tuple[Path, ProvenanceManifest]:
        """Execute deterministic sampling for the specified mode and size.

        Parameters
        ----------
        mode : str
            Sampling mode (e.g. 'representative', 'diagnostic', 'audit', 'candidate').
        size : int
            Target number of documents to collect.
        seed : int, default 42
            Deterministic random seed.
        output_dir : Path or str or None, optional
            Destination directory for Parquet and manifest outputs.
        **kwargs : any
            Additional source-specific arguments.

        Returns
        -------
        tuple of (Path, ProvenanceManifest)
            Path to the generated Parquet file and the associated manifest.
        """
        pass

    @abstractmethod
    def plan(
        self,
        mode: str,
        size: int,
        **kwargs,
    ) -> dict:
        """Generate a dry-run execution plan without performing transfers.

        Parameters
        ----------
        mode : str
            Sampling mode.
        size : int
            Target number of documents.
        **kwargs : any
            Source-specific options.

        Returns
        -------
        dict
            Dry-run execution plan with source metadata, safety limits, and frames.
        """
        pass

    def _persist_sample(
        self,
        documents: List[NormalizedDocument],
        mode: str,
        target_size: int,
        seed: int,
        output_dir: Path | str,
        upstream_identifier: str,
        upstream_revision: Optional[str] = None,
        upstream_commit_sha: Optional[str] = None,
        upstream_url: Optional[str] = None,
        upstream_configuration: Optional[str] = None,
        population_scope: Optional[str] = None,
        sampling_frame: Optional[str] = None,
        records_examined: Optional[int] = None,
        bytes_read: Optional[int] = None,
        stopping_reason: Optional[str] = None,
        exclusion_config_hash: Optional[str] = None,
    ) -> Tuple[Path, ProvenanceManifest]:
        """Persist normalized documents and write the provenance manifest.

        Parameters
        ----------
        documents : list of NormalizedDocument
            Collected normalized documents.
        mode : str
            Sampling mode.
        target_size : int
            Requested sample size.
        seed : int
            Sampling seed.
        output_dir : Path or str
            Destination directory.
        upstream_identifier : str
            Canonical upstream identifier.
        upstream_revision : str or None, optional
            Upstream human-readable version or branch name.
        upstream_commit_sha : str or None, optional
            Immutable upstream git commit SHA.
        upstream_url : str or None, optional
            Upstream URL.
        upstream_configuration : str or None, optional
            Upstream configuration / subset.
        population_scope : str or None, optional
            Upstream population size or scope description.
        sampling_frame : str or None, optional
            Sampling frame description.
        records_examined : int or None, optional
            Number of upstream records examined.
        bytes_read : int or None, optional
            Estimated or tracked network/disk bytes read.
        stopping_reason : str or None, optional
            Stopping condition encountered.
        exclusion_config_hash : str or None, optional
            Hash of exclusions file if applicable.

        Returns
        -------
        tuple of (Path, ProvenanceManifest)
            Output Parquet path and manifest.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        parquet_filename = f"{mode}.parquet"
        parquet_path = out_dir / parquet_filename
        manifest_path = out_dir / f"manifest_{mode}.json"

        # Save Parquet
        save_sample_parquet(documents, parquet_path)

        # File metrics
        file_sha = compute_file_sha256(parquet_path)
        file_bytes = parquet_path.stat().st_size

        total_chars = sum(len(d.text) for d in documents)
        total_words = sum(len(d.text.split()) for d in documents)

        manifest = ProvenanceManifest(
            source=self.source_name,
            mode=mode,
            target_size=target_size,
            document_count=len(documents),
            seed=seed,
            upstream_identifier=upstream_identifier,
            upstream_revision=upstream_revision,
            upstream_commit_sha=upstream_commit_sha,
            upstream_url=upstream_url,
            upstream_configuration=upstream_configuration,
            population_scope=population_scope,
            sampling_frame=sampling_frame,
            records_examined=records_examined,
            bytes_read=bytes_read,
            stopping_reason=stopping_reason,
            exclusion_config_hash=exclusion_config_hash,
            output_parquet=parquet_filename,
            parquet_sha256=file_sha,
            parquet_bytes=file_bytes,
            git_commit=get_git_commit(),
            stats={
                "total_chars": total_chars,
                "total_words": total_words,
            },
        )
        manifest.save(manifest_path)

        return parquet_path, manifest
