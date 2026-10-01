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
the reservoir.  Like audit, candidate spans all 56 shards using distributed
intra-shard row-group selection to approximate the residual edu_high pool.

Both modes report unambiguous statistics satisfying:
- sum(records_encountered_per_subset) == records_examined
- eligible_records == records_examined - total_excluded
- sum(eligible_records_per_subset) == eligible_records
- sum(retained_sample_per_subset) == final sample size
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
    stable_hash64,
)
from cambacica.corpus.schema import validate_and_normalize
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


def get_exclusion_rules_map(config_path: Path | str) -> Dict[str, str]:
    """Build a mapping from normalized subset patterns to rule explanations.

    Parameters
    ----------
    config_path : Path or str
        Path to gigaverbo_exclusions.yaml.

    Returns
    -------
    dict of str to str
        Mapping of pattern strings to descriptive rule strings.
    """
    path = Path(config_path)
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rules: Dict[str, str] = {}
    exclusions = data.get("exclusions", {})
    for group_name, items in exclusions.items():
        if isinstance(items, list):
            for entry in items:
                if isinstance(entry, dict):
                    name = entry.get("name", group_name)
                    reason = entry.get("reason", "")
                    rule_label = f"{group_name}/{name}: {reason}".strip()
                    for pat in entry.get("match_patterns", []):
                        rules[pat.lower()] = rule_label
    return rules


def match_subset_exclusion(
    subset_name: Optional[str],
    exact_blocked: Set[str],
    patterns: List[str],
    rules_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Check if a subset is excluded and return the matched rule description.

    Parameters
    ----------
    subset_name : str or None
        Raw subset name from GigaVerbo metadata.
    exact_blocked : set of str
        Set of normalized blocked subset strings.
    patterns : list of str
        Wildcard patterns.
    rules_map : dict of str to str, optional
        Mapping from pattern/string to rule explanation.

    Returns
    -------
    str or None
        Rule description if excluded, None otherwise.
    """
    if not subset_name:
        return None
    norm_name = str(subset_name).strip().lower()
    if norm_name in exact_blocked:
        if rules_map and norm_name in rules_map:
            return rules_map[norm_name]
        for pat, desc in (rules_map or {}).items():
            if fnmatch.fnmatch(norm_name, pat):
                return desc
        return f"blocked_subsets:{norm_name}"

    for pat in patterns:
        if fnmatch.fnmatch(norm_name, pat):
            if rules_map and pat in rules_map:
                return rules_map[pat]
            return f"pattern:{pat}"

    return None


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
    return match_subset_exclusion(subset_name, exact_blocked, patterns) is not None


def select_distributed_row_groups(
    num_row_groups: int,
    n_groups: int,
    seed: int = 42,
    shard_name: str = "",
) -> List[int]:
    """Select deterministic row-group indices distributed across a shard.

    Guarantees that row groups span the physical file (early, middle, late)
    rather than always picking row group 0.

    Parameters
    ----------
    num_row_groups : int
        Total number of row groups in the Parquet shard.
    n_groups : int
        Target number of row groups to select.
    seed : int, default 42
        Deterministic random seed.
    shard_name : str, default ""
        Shard identifier used for deterministic hash perturbation.

    Returns
    -------
    list of int
        Sorted list of selected row-group indices (0-indexed).
    """
    if num_row_groups <= 0:
        return []
    if num_row_groups <= n_groups or n_groups <= 0:
        return list(range(num_row_groups))

    step = num_row_groups / n_groups
    selected: List[int] = []
    seen: Set[int] = set()

    for i in range(n_groups):
        center = int(i * step + step / 2)
        center = min(center, num_row_groups - 1)
        candidates = [max(0, center - 1), center, min(num_row_groups - 1, center + 1)]
        best = min(
            candidates,
            key=lambda idx: stable_hash64(f"{shard_name}:rg:{idx}", seed=seed),
        )
        if best not in seen:
            seen.add(best)
            selected.append(best)

    if len(selected) < n_groups:
        for idx in range(num_row_groups):
            if idx not in seen:
                seen.add(idx)
                selected.append(idx)
                if len(selected) >= n_groups:
                    break

    return sorted(selected)


class GigaVerboSampler(BaseSourceSampler):
    """Sampler for Polygl0t/gigaverbo-v2 (edu_high).

    Supports:
    - 'audit': Samples documents across ALL shards visiting distributed
      row groups per shard without subset filtering to inspect the full
      unfiltered upstream distribution.
    - 'candidate': Applies the exclusion policy before sampling.  Uses a
      deterministic shard and row-group selection distributed across the full
      collection.  Reports subsets encountered before exclusions, exclusion
      counts by subset, rules matched, and subsets remaining.
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
        num_shards_to_visit: Optional[int] = None,
        row_groups_per_shard: int = 4,
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
        num_shards_to_visit : int or None, optional
            Number of shards to visit. Defaults to None (visiting all 56 shards
            for both 'audit' and 'candidate' modes).
        row_groups_per_shard : int, default 4
            Number of distributed row groups to visit per shard.
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
        if num_shards_to_visit is None or num_shards_to_visit >= total_shards:
            visit_count = total_shards
        else:
            visit_count = num_shards_to_visit

        if mode == "audit":
            frame = (
                f"distributed_rowgroup_audit (all {total_shards} shards, "
                f"{row_groups_per_shard} distributed row groups per shard, "
                f"no exclusions applied)"
            )
        elif visit_count == total_shards:
            frame = (
                f"distributed_rowgroup_candidate (all {total_shards} shards, "
                f"{row_groups_per_shard} distributed row groups per shard; "
                f"exclusion policy applied before sampling)"
            )
        else:
            frame = (
                f"distributed_rowgroup_candidate ({visit_count} of "
                f"{total_shards} shards evenly distributed, "
                f"{row_groups_per_shard} distributed row groups per shard; "
                f"exclusion policy applied before sampling)"
            )

        total_rg = visit_count * row_groups_per_shard
        quota_per_rg = max(1, size // total_rg) if total_rg else size

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
                f"{visit_count} shards visited ({row_groups_per_shard} row groups/shard)"
            ],
            "safety_limits": {
                "shards_visited": visit_count,
                "row_groups_per_shard": row_groups_per_shard,
                "quota_per_row_group": quota_per_rg,
                "batch_size": 2048,
            },
            "exclusion_config_hash": config_sha,
            "estimated_transfer": (
                f"~{visit_count * row_groups_per_shard * 2}–{visit_count * row_groups_per_shard * 5} MB "
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
        num_shards_to_visit: Optional[int] = None,
        row_groups_per_shard: int = 4,
        batch_size: int = 2048,
        **kwargs,
    ) -> Tuple[Path, ProvenanceManifest]:
        """Execute deterministic distributed row-group sampling from GigaVerbo-v2.

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
        num_shards_to_visit : int or None, optional
            Number of shards to visit. Defaults to None (visiting all available
            shards for both 'audit' and 'candidate' modes).
        row_groups_per_shard : int, default 4
            Number of distributed row groups to visit per shard.
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
        rules_map: Dict[str, str] = {}

        if mode == "candidate":
            exact_blocked, patterns, config_sha = load_gigaverbo_exclusions(
                exclusions_path
            )
            rules_map = get_exclusion_rules_map(exclusions_path)

        commit_sha = resolve_hf_commit_sha(self.canonical_id, revision="main")
        fs = HfFileSystem()
        shards_dir = f"datasets/{self.canonical_id}/{self.split}"
        all_shard_files = sorted(
            [f for f in fs.ls(shards_dir, detail=False) if f.endswith(".parquet")]
        )
        if not all_shard_files:
            raise RuntimeError(f"No parquet shards found in {shards_dir}")

        total_shards = len(all_shard_files)

        if (
            mode == "audit"
            or num_shards_to_visit is None
            or num_shards_to_visit >= total_shards
        ):
            selected_shards = all_shard_files
            if mode == "audit":
                sampling_frame = (
                    f"distributed_rowgroup_audit (all {total_shards} shards, "
                    f"{row_groups_per_shard} distributed row groups per shard, "
                    f"no exclusions applied)"
                )
            else:
                sampling_frame = (
                    f"distributed_rowgroup_candidate (all {total_shards} shards, "
                    f"{row_groups_per_shard} distributed row groups per shard; "
                    f"exclusion policy applied before sampling)"
                )
        else:
            step = max(1, total_shards // num_shards_to_visit)
            selected_shards = [
                all_shard_files[i] for i in range(0, total_shards, step)
            ][:num_shards_to_visit]
            sampling_frame = (
                f"distributed_rowgroup_candidate "
                f"({len(selected_shards)} of {total_shards} shards evenly distributed, "
                f"{row_groups_per_shard} distributed row groups per shard; "
                f"exclusion policy applied before sampling)"
            )

        total_rg_targets = len(selected_shards) * row_groups_per_shard
        quota_per_rg = max(1, size // total_rg_targets) if total_rg_targets else size

        reservoir = DeterministicReservoirSampler(capacity=size, seed=seed)
        target_cols = ["id", "source", "subset", "edu_score", "text"]

        records_encountered_per_subset: Dict[str, int] = {}
        exclusion_counts_by_subset: Dict[str, int] = {}
        exclusion_rules_matched: Dict[str, str] = {}
        eligible_records_per_subset: Dict[str, int] = {}
        shard_row_groups_map: Dict[str, List[int]] = {}

        total_row_groups_visited = 0
        records_examined = 0
        byte_counter = [0]
        stopping_reason = "shards_and_row_groups_exhausted"

        for shard_idx, shard_path in enumerate(selected_shards):
            shard_name = Path(shard_path).name
            try:
                with fs.open(shard_path, "rb") as f:
                    wrapped_f = _CountingSeekableReader(f, byte_counter)
                    pf = pq.ParquetFile(wrapped_f)
                    num_rgs = getattr(pf, "num_row_groups", 1) or 1
                    chosen_rgs = select_distributed_row_groups(
                        num_row_groups=num_rgs,
                        n_groups=row_groups_per_shard,
                        seed=seed + shard_idx,
                        shard_name=shard_name,
                    )
                    shard_row_groups_map[shard_name] = chosen_rgs

                    for rg_idx in chosen_rgs:
                        total_row_groups_visited += 1
                        rg_accepted = 0

                        try:
                            batch_iter = pf.iter_batches(
                                batch_size=batch_size,
                                row_groups=[rg_idx],
                                columns=target_cols,
                            )
                        except TypeError:
                            batch_iter = pf.iter_batches(
                                batch_size=batch_size,
                                columns=target_cols,
                            )

                        for batch in batch_iter:
                            b_dict = batch.to_pydict()
                            texts = b_dict.get("text", [])
                            ids = b_dict.get("id", [])
                            sources = b_dict.get("source", [])
                            subsets = b_dict.get("subset", [])
                            scores = b_dict.get("edu_score", [])

                            for i in range(len(texts)):
                                text = texts[i]
                                if not text or len(text.strip()) < 50:
                                    continue

                                records_examined += 1
                                subset = subsets[i] if i < len(subsets) else None
                                doc_id = (
                                    str(ids[i])
                                    if i < len(ids)
                                    else f"{shard_idx}_{rg_idx}_{i}"
                                )
                                src_url = sources[i] if i < len(sources) else None
                                edu_sc = scores[i] if i < len(scores) else None

                                subset_key = str(subset) if subset else "(null)"
                                records_encountered_per_subset[subset_key] = (
                                    records_encountered_per_subset.get(subset_key, 0)
                                    + 1
                                )

                                if mode == "candidate":
                                    matched_rule = match_subset_exclusion(
                                        subset, exact_blocked, patterns, rules_map
                                    )
                                    if matched_rule:
                                        exclusion_counts_by_subset[subset_key] = (
                                            exclusion_counts_by_subset.get(
                                                subset_key, 0
                                            )
                                            + 1
                                        )
                                        if subset_key not in exclusion_rules_matched:
                                            exclusion_rules_matched[subset_key] = (
                                                matched_rule
                                            )
                                        continue

                                eligible_records_per_subset[subset_key] = (
                                    eligible_records_per_subset.get(subset_key, 0) + 1
                                )

                                raw_doc = {
                                    "text": text,
                                    "source": "gigaverbo_v2",
                                    "source_revision": self.revision,
                                    "subset": subset,
                                    "original_id": doc_id,
                                    "original_url": src_url,
                                    "license": None,
                                    "language": "pt",
                                    "language_score": 1.0,
                                    "variety": None,
                                    "quality_score": float(edu_sc)
                                    if edu_sc is not None
                                    else None,
                                    "publication_date": None,
                                    "domain_category": subset,
                                }

                                norm_doc = validate_and_normalize(raw_doc)
                                reservoir.add(doc_id, norm_doc)
                                rg_accepted += 1
                                if rg_accepted >= max(10, quota_per_rg * 2):
                                    break

                            if rg_accepted >= max(10, quota_per_rg * 2):
                                break
            except Exception as e:
                logger.warning(f"Error streaming shard {shard_path}: {e}")
                continue

        documents = reservoir.get_sample()
        if len(documents) >= size:
            stopping_reason = "target_size_reached"

        retained_sample_per_subset: Dict[str, int] = {}
        for doc in documents:
            s_key = str(doc.subset) if doc.subset else "(null)"
            retained_sample_per_subset[s_key] = (
                retained_sample_per_subset.get(s_key, 0) + 1
            )

        total_excluded = sum(exclusion_counts_by_subset.values())
        eligible_records = records_examined - total_excluded

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

        manifest.stats["records_encountered_per_subset"] = (
            records_encountered_per_subset
        )
        manifest.stats["subsets_before_exclusion"] = records_encountered_per_subset
        manifest.stats["exclusion_counts_by_subset"] = exclusion_counts_by_subset
        manifest.stats["exclusion_rules_matched"] = exclusion_rules_matched
        manifest.stats["total_excluded"] = total_excluded
        manifest.stats["eligible_records"] = eligible_records
        manifest.stats["eligible_records_per_subset"] = eligible_records_per_subset
        manifest.stats["subsets_after_exclusion"] = eligible_records_per_subset
        manifest.stats["retained_sample_per_subset"] = retained_sample_per_subset
        manifest.stats["shards_visited"] = len(selected_shards)
        manifest.stats["row_groups_visited"] = total_row_groups_visited
        manifest.stats["selected_shards"] = [Path(s).name for s in selected_shards]
        manifest.stats["selected_row_groups"] = shard_row_groups_map
        manifest_path = output_dir / f"manifest_{mode}.json"
        manifest.save(manifest_path)
        return parquet_path, manifest
