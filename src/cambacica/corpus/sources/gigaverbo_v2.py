"""GigaVerbo-v2 sampler for Gate C1.

Streams documents from Polygl0t/gigaverbo-v2 (edu_high) across multiple Parquet
shards, providing distinct 'audit' and 'candidate' modes with versioned exclusion
filtering.

Audit vs candidate design
--------------------------
*audit* spans ALL 56 available shards with a small, deterministic per-shard
quota so that every source/subset region of edu_high is observable.  No
exclusion filtering is applied.

*candidate* first applies the exclusion config defined in
``configs/gigaverbo_exclusions.yaml`` to each record **before** adding it to
the reservoir.  It also uses a distributed shard selection (not identical to
audit) to keep the two modes comparable without forcing artificial disjointness.

Both modes report:
- subsets encountered before any exclusions;
- exclusion counts by subset (candidate only);
- subsets remaining after exclusions (candidate only);
- records examined and bytes read;
- overlap with the other mode is reported by the ``compare`` command separately.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml

from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq

from cambacica.corpus.manifest import ProvenanceManifest
from cambacica.corpus.sampling import (
    DeterministicReservoirSampler,
)
from cambacica.corpus.schema import (
    NormalizedDocument,
    validate_and_normalize,
)
from cambacica.corpus.sources.base import (
    BaseSourceSampler,
    resolve_hf_commit_sha,
)


logger = logging.getLogger(__name__)


class _CountingSeekableReader:
    """Seekable reader wrapper that counts bytes read."""

    def __init__(self, raw: Any, counter: List[int]) -> None:
        self.raw = raw
        self.counter = counter

    def read(self, size: int = -1) -> bytes:
        """Read bytes and update counter.

        Parameters
        ----------
        size : int, default -1
            Number of bytes to read.

        Returns
        -------
        bytes
            Read data.
        """
        chunk = self.raw.read(size)
        if chunk:
            self.counter[0] += len(chunk)
        return chunk

    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek to position.

        Parameters
        ----------
        offset : int
            Byte offset.
        whence : int, default 0
            Seek mode.

        Returns
        -------
        int
            New position.
        """
        return self.raw.seek(offset, whence)

    def tell(self) -> int:
        """Return current position.

        Returns
        -------
        int
            Current byte position.
        """
        return self.raw.tell()

    def seekable(self) -> bool:
        """Return True (always seekable).

        Returns
        -------
        bool
        """
        return True

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw, name)


def load_gigaverbo_exclusions(
    config_path: Path | str,
) -> Tuple[Set[str], List[str], str]:
    """Load blocked subsets, glob patterns, and config hash from exclusions YAML.

    Parameters
    ----------
    config_path : Path or str
        Path to gigaverbo_exclusions.yaml.

    Returns
    -------
    tuple of (set of str, list of str, str)
        Tuple containing:
        - normalized set of exact blocked subset names
        - list of wildcard match patterns
        - SHA-256 hash of the exclusions config file
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Exclusions config not found at: {path}")

    content = path.read_text(encoding="utf-8")
    config_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    data = yaml.safe_load(content)

    exact_blocked: Set[str] = set()
    for item in data.get("blocked_subsets", []):
        exact_blocked.add(str(item).strip().lower())

    patterns: List[str] = []
    exclusions = data.get("exclusions", {})
    for group_name, items in exclusions.items():
        if isinstance(items, list):
            for entry in items:
                if isinstance(entry, dict):
                    for pat in entry.get("match_patterns", []):
                        patterns.append(pat.lower())

    return exact_blocked, patterns, config_sha


def is_subset_excluded(
    subset_name: Optional[str],
    exact_blocked: Set[str],
    patterns: List[str],
) -> bool:
    """Check if a subset identifier matches any exclusion rule.

    Parameters
    ----------
    subset_name : str or None
        Raw subset name from GigaVerbo metadata.
    exact_blocked : set of str
        Set of normalized blocked subset strings.
    patterns : list of str
        Wildcard patterns.

    Returns
    -------
    bool
        True if the subset must be excluded, False otherwise.
    """
    if not subset_name:
        return False
    norm_name = str(subset_name).strip().lower()

    if norm_name in exact_blocked:
        return True

    for pat in patterns:
        if fnmatch.fnmatch(norm_name, pat):
            return True

    return False


class GigaVerboSampler(BaseSourceSampler):
    """Sampler for Polygl0t/gigaverbo-v2 (edu_high).

    Supports:
    - 'audit': Samples documents across ALL shards without subset filtering
      to inspect the full unfiltered upstream distribution.  Each shard
      contributes at most ``quota_per_shard`` documents.
    - 'candidate': Applies the exclusion policy before sampling.  Uses a
      deterministic shard selection distributed across the full 56-shard
      collection.  Reports subsets encountered before exclusions, exclusion
      counts by subset, and subsets remaining.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(source_name="gigaverbo_v2", config=config)
        self.canonical_id: str = "Polygl0t/gigaverbo-v2"
        self.revision: str = "2026-09-27"
        self.split: str = "edu_high"

    def plan(
        self,
        mode: str = "candidate",
        size: int = 25000,
        num_shards_to_visit: int = 8,
        exclusions_path: Optional[Path | str] = None,
        **kwargs,
    ) -> dict:
        """Generate a dry-run execution plan without performing transfers.

        Parameters
        ----------
        mode : str, default 'candidate'
            Sampling mode ('audit' or 'candidate').
        size : int, default 25000
            Target number of documents.
        num_shards_to_visit : int, default 8
            For 'candidate' mode: number of evenly-distributed shards.
            For 'audit' mode: all 56 shards are visited regardless.
        exclusions_path : Path or str or None, optional
            Path to exclusions config file.
        **kwargs : any
            Additional arguments (ignored).

        Returns
        -------
        dict
            Dry-run execution plan.
        """
        if exclusions_path is None:
            exclusions_path = Path("configs/gigaverbo_exclusions.yaml")
        config_sha = None
        if mode == "candidate" and Path(exclusions_path).is_file():
            _, _, config_sha = load_gigaverbo_exclusions(exclusions_path)

        commit_sha = resolve_hf_commit_sha(self.canonical_id, revision="main")

        total_shards = 56  # known upstream count
        if mode == "audit":
            frame = (
                f"audit_full_shard_scan (all {total_shards} shards, "
                f"small deterministic quota per shard, no exclusions applied)"
            )
            quota_per_shard = max(1, size // total_shards)
            visit_count = total_shards
        else:
            visit_count = num_shards_to_visit
            frame = (
                f"candidate_distributed_shard_subsample ({visit_count} of "
                f"{total_shards} shards evenly distributed; exclusion policy applied)"
            )
            quota_per_shard = max(1, size // visit_count)

        return {
            "source": self.source_name,
            "mode": mode,
            "target_size": size,
            "upstream_identifier": self.canonical_id,
            "upstream_revision": self.revision,
            "upstream_commit_sha": commit_sha,
            "upstream_configuration": self.split,
            "population_scope": (
                f"{total_shards} parquet shards (~72.8 GB, ~28M documents in edu_high split)"
            ),
            "sampling_frame": frame,
            "selected_partitions": [
                f"datasets/{self.canonical_id}/{self.split}: "
                f"{visit_count} shards visited"
            ],
            "safety_limits": {
                "shards_visited": visit_count,
                "quota_per_shard": quota_per_shard,
                "batch_size": 2048,
            },
            "exclusion_config_hash": config_sha,
            "estimated_transfer": (
                f"~{visit_count * 10}–{visit_count * 25} MB "
                f"(column-pruned Parquet range requests)"
            ),
        }

    def sample(
        self,
        mode: str = "candidate",
        size: int = 25000,
        seed: int = 42,
        output_dir: Optional[Path | str] = None,
        exclusions_path: Optional[Path | str] = None,
        num_shards_to_visit: int = 8,
        batch_size: int = 2048,
        **kwargs,
    ) -> Tuple[Path, ProvenanceManifest]:
        """Execute deterministic sampling from GigaVerbo-v2 edu_high.

        Parameters
        ----------
        mode : str, default 'candidate'
            Sampling mode ('audit' or 'candidate').
        size : int, default 25000
            Target document count.
        seed : int, default 42
            Deterministic random seed.
        output_dir : Path or str or None, optional
            Output destination directory.
        exclusions_path : Path or str or None, optional
            Path to gigaverbo_exclusions.yaml (used only in 'candidate' mode).
        num_shards_to_visit : int, default 8
            For 'candidate' mode: number of evenly-distributed shards to visit.
            For 'audit' mode: all available shards are visited regardless.
        batch_size : int, default 2048
            RecordBatch read size.
        **kwargs : any
            Additional arguments (ignored).

        Returns
        -------
        tuple of (Path, ProvenanceManifest)
            Path to output Parquet file and metadata manifest.
        """
        if output_dir is None:
            output_dir = Path("data/samples/gate_c1/gigaverbo_v2")
        else:
            output_dir = Path(output_dir)

        if exclusions_path is None:
            exclusions_path = Path("configs/gigaverbo_exclusions.yaml")

        exact_blocked: Set[str] = set()
        patterns: List[str] = []
        config_sha: Optional[str] = None

        if mode == "candidate":
            exact_blocked, patterns, config_sha = load_gigaverbo_exclusions(
                exclusions_path
            )

        commit_sha = resolve_hf_commit_sha(self.canonical_id, revision="main")
        fs = HfFileSystem()
        shards_dir = f"datasets/{self.canonical_id}/{self.split}"
        all_shard_files = sorted(
            [f for f in fs.ls(shards_dir, detail=False) if f.endswith(".parquet")]
        )
        if not all_shard_files:
            raise RuntimeError(f"No parquet shards found in {shards_dir}")

        total_shards = len(all_shard_files)

        if mode == "audit":
            # Audit: visit EVERY shard with a small per-shard quota
            selected_shards = all_shard_files
            quota_per_shard = max(1, size // total_shards)
            sampling_frame = (
                f"audit_full_shard_scan (all {total_shards} shards, "
                f"quota_per_shard={quota_per_shard}, no exclusions applied)"
            )
        else:
            # Candidate: select num_shards_to_visit evenly distributed shards
            step = max(1, total_shards // num_shards_to_visit)
            selected_shards = [
                all_shard_files[i]
                for i in range(0, total_shards, step)
            ][:num_shards_to_visit]
            quota_per_shard = max(1, size // len(selected_shards))
            sampling_frame = (
                f"candidate_distributed_shard_subsample "
                f"({len(selected_shards)} of {total_shards} shards evenly distributed; "
                f"exclusion policy applied before sampling)"
            )

        reservoir = DeterministicReservoirSampler(capacity=size, seed=seed)

        target_cols = ["id", "source", "subset", "edu_score", "text"]
        # Tracking for audit/exclusion reporting
        subsets_before_exclusion: Dict[str, int] = {}
        exclusion_counts_by_subset: Dict[str, int] = {}
        subsets_after_exclusion: Dict[str, int] = {}

        records_examined = 0
        byte_counter = [0]
        stopping_reason = "shards_exhausted"

        for shard_idx, shard_path in enumerate(selected_shards):
            shard_accepted = 0
            try:
                with fs.open(shard_path, "rb") as f:
                    wrapped_f = _CountingSeekableReader(f, byte_counter)
                    pf = pq.ParquetFile(wrapped_f)
                    for batch in pf.iter_batches(
                        batch_size=batch_size, columns=target_cols
                    ):
                        b_dict = batch.to_pydict()
                        texts = b_dict.get("text", [])
                        ids = b_dict.get("id", [])
                        sources = b_dict.get("source", [])
                        subsets = b_dict.get("subset", [])
                        scores = b_dict.get("edu_score", [])

                        for i in range(len(texts)):
                            records_examined += 1
                            text = texts[i]
                            if not text or len(text.strip()) < 50:
                                continue

                            subset = subsets[i] if i < len(subsets) else None
                            doc_id = str(ids[i]) if i < len(ids) else f"{shard_idx}_{i}"
                            src_url = sources[i] if i < len(sources) else None
                            edu_sc = scores[i] if i < len(scores) else None

                            # Track subsets before any exclusion
                            subset_key = str(subset) if subset else "(null)"
                            subsets_before_exclusion[subset_key] = (
                                subsets_before_exclusion.get(subset_key, 0) + 1
                            )

                            if mode == "candidate" and is_subset_excluded(
                                subset, exact_blocked, patterns
                            ):
                                exclusion_counts_by_subset[subset_key] = (
                                    exclusion_counts_by_subset.get(subset_key, 0) + 1
                                )
                                continue

                            raw_doc = {
                                "text": text,
                                "source": "gigaverbo_v2",
                                "source_revision": self.revision,
                                "subset": subset,
                                "original_id": doc_id,
                                "original_url": src_url,
                                "license": None,  # Do not fabricate per-row license
                                "language": "pt",
                                "language_score": 1.0,
                                "variety": None,
                                "quality_score": float(edu_sc) if edu_sc is not None else None,
                                "publication_date": None,
                                "domain_category": subset,
                            }

                            norm_doc = validate_and_normalize(raw_doc)
                            if reservoir.add(doc_id, norm_doc):
                                shard_accepted += 1
                                subsets_after_exclusion[subset_key] = (
                                    subsets_after_exclusion.get(subset_key, 0) + 1
                                )

                            if shard_accepted >= quota_per_shard * 2:
                                break

                        if shard_accepted >= quota_per_shard * 2:
                            break
            except Exception as e:
                logger.warning(f"Error streaming shard {shard_path}: {e}")
                continue

            if len(reservoir) >= size:
                stopping_reason = "target_size_reached"
                break

        documents = reservoir.get_sample()
        if len(documents) >= size:
            stopping_reason = "target_size_reached"

        parquet_path, manifest = self._persist_sample(
            documents=documents,
            mode=mode,
            target_size=size,
            seed=seed,
            output_dir=output_dir,
            upstream_identifier=self.canonical_id,
            upstream_revision=self.revision,
            upstream_commit_sha=commit_sha,
            upstream_url=f"https://huggingface.co/datasets/{self.canonical_id}",
            upstream_configuration=self.split,
            population_scope=(
                f"{total_shards} parquet shards (~72.8 GB, ~28M documents in edu_high split)"
            ),
            sampling_frame=sampling_frame,
            records_examined=records_examined,
            bytes_read=byte_counter[0],
            stopping_reason=stopping_reason,
            exclusion_config_hash=config_sha,
        )

        # Enrich manifest with exclusion diagnostics
        manifest.stats["subsets_before_exclusion"] = subsets_before_exclusion
        if mode == "candidate":
            manifest.stats["exclusion_counts_by_subset"] = exclusion_counts_by_subset
            manifest.stats["subsets_after_exclusion"] = subsets_after_exclusion
            manifest.stats["total_excluded"] = sum(exclusion_counts_by_subset.values())
        manifest.stats["shards_visited"] = len(selected_shards)
        manifest_path = output_dir / f"manifest_{mode}.json"
        manifest.save(manifest_path)
        return parquet_path, manifest
