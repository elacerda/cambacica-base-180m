"""Exact duplicate detection across Gate C1 sample files using content_sha256.

Scans one or more Parquet sample files to identify identical documents shared
across different sources or duplicated within a single sample, while
distinguishing between four semantically distinct relationship types.

Duplicate relationship types
-----------------------------
WITHIN_FILE_DUPLICATES
    Exact duplicate text occurring multiple times within a single Parquet file.

SAME_SOURCE_CROSS_MODE_OVERLAP
    Same content appearing in multiple sampling modes or files for the same
    upstream source (e.g. carolina/representative vs carolina/diagnostic, or
    gigaverbo_v2/audit vs gigaverbo_v2/candidate).  This is expected behaviour
    and should NOT be conflated with corpus contamination.

CROSS_SOURCE_DUPLICATES
    Same content appearing in genuinely different upstream sources
    (e.g. carolina appearing in gigaverbo_v2).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import pyarrow.parquet as pq


@dataclass
class DuplicateOccurrence:
    """Location record for a duplicate document occurrence."""

    source: str
    file_path: str
    mode: Optional[str]
    row_index: int
    original_id: Optional[str]


@dataclass
class CrossSourceDuplicateReport:
    """Report detailing exact duplicates detected across samples.

    Each category (A–C below) is mutually exclusive for a given hash:

    A. within_file_duplicates
       Duplicate text inside a single Parquet file.
    B. same_source_cross_mode_overlap
       Same text in multiple files sharing the same upstream source name
       (e.g. different sampling modes).  Overlap is expected and benign.
    C. cross_source_duplicates
       Same text appearing in files with genuinely different source names.
    """

    total_unique_hashes: int
    total_duplicate_hashes: int
    total_duplicate_documents: int

    # A: within a single file
    within_file_counts: Dict[str, int]

    # B: same source, different files/modes
    same_source_cross_mode_overlap: Dict[str, int]

    # C: different upstream sources
    cross_source_pair_counts: Dict[str, int]

    # Legacy alias kept for backwards compatibility with existing tests
    inter_source_pair_counts: Dict[str, int]
    intra_source_duplicate_counts: Dict[str, int]

    sample_collisions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary.

        Returns
        -------
        dict
            Report mapping.
        """
        return asdict(self)


def _source_from_file(path: Path, table_source: Optional[str]) -> str:
    """Derive the upstream source name for a file.

    Parameters
    ----------
    path : Path
        File path.
    table_source : str or None
        Source value read from the first row, if available.

    Returns
    -------
    str
        Source identifier.
    """
    return table_source or path.stem


def _mode_from_path(path: Path) -> Optional[str]:
    """Infer sample mode from the file stem (e.g. 'representative').

    Parameters
    ----------
    path : Path
        Parquet file path.

    Returns
    -------
    str or None
        Mode name if recognisable, otherwise None.
    """
    stem = path.stem.lower()
    for m in ("representative", "diagnostic", "audit", "candidate"):
        if m in stem:
            return m
    return stem or None


def find_cross_source_exact_duplicates(
    parquet_files: Sequence[Path | str],
    max_example_collisions: int = 10,
) -> CrossSourceDuplicateReport:
    """Detect exact duplicates across a collection of Parquet sample files.

    Classifies each duplicated hash into one of three relationship types:

    A. WITHIN_FILE_DUPLICATES — same hash appears more than once in one file.
    B. SAME_SOURCE_CROSS_MODE_OVERLAP — same upstream source, different files.
    C. CROSS_SOURCE_DUPLICATES — genuinely different upstream sources.

    Parameters
    ----------
    parquet_files : sequence of Path or str
        List of Parquet files to inspect.
    max_example_collisions : int, default 10
        Maximum number of collision examples to retain in the report.

    Returns
    -------
    CrossSourceDuplicateReport
        Diagnostic report of exact duplicate occurrences.
    """
    # (hash, file_path) -> occurrences within that file
    # hash -> list of (source, file_path, mode, row_index, original_id)
    hash_to_occurrences: Dict[str, List[DuplicateOccurrence]] = defaultdict(list)

    for p in parquet_files:
        path = Path(p)
        if not path.is_file():
            continue

        try:
            table = pq.read_table(
                path, columns=["source", "original_id", "content_sha256"]
            )
            raw_sources = table["source"].to_pylist()
            ids = table["original_id"].to_pylist()
            hashes = table["content_sha256"].to_pylist()

            first_source = raw_sources[0] if raw_sources else None
            mode = _mode_from_path(path)

            for idx, (src, orig_id, sha) in enumerate(zip(raw_sources, ids, hashes)):
                if sha:
                    hash_to_occurrences[sha].append(
                        DuplicateOccurrence(
                            source=src or path.parent.name,
                            file_path=str(path),
                            mode=mode,
                            row_index=idx,
                            original_id=orig_id,
                        )
                    )
        except Exception:
            continue

    total_unique = len(hash_to_occurrences)
    dup_hashes = 0
    total_dup_docs = 0

    # Category A: within-file (same hash, same file path)
    within_file: Dict[str, int] = defaultdict(int)
    # Category B: same source, different file paths
    same_src_cross_mode: Dict[str, int] = defaultdict(int)
    # Category C: different source names
    cross_src_pairs: Dict[str, int] = defaultdict(int)

    example_collisions: List[Dict[str, Any]] = []

    for sha, occurrences in hash_to_occurrences.items():
        if len(occurrences) <= 1:
            continue

        dup_hashes += 1
        total_dup_docs += len(occurrences) - 1

        # Group by file path
        file_groups: Dict[str, List[DuplicateOccurrence]] = defaultdict(list)
        for occ in occurrences:
            file_groups[occ.file_path].append(occ)

        # A: within-file duplicates (multiple rows same hash in one file)
        for fp, occs in file_groups.items():
            if len(occs) > 1:
                src_name = occs[0].source
                within_file[src_name] += len(occs) - 1

        # Collect unique (source, file_path) pairs
        unique_source_files: Set[Tuple[str, str]] = {
            (occ.source, occ.file_path) for occ in occurrences
        }
        unique_sources: Set[str] = {src for src, _ in unique_source_files}
        unique_files: Set[str] = {fp for _, fp in unique_source_files}

        if len(unique_files) > 1:
            if len(unique_sources) == 1:
                # B: same source, different files (cross-mode overlap)
                src_name = next(iter(unique_sources))
                same_src_cross_mode[src_name] += 1
            else:
                # C: different upstream sources
                sorted_srcs = sorted(unique_sources)
                for i in range(len(sorted_srcs)):
                    for j in range(i + 1, len(sorted_srcs)):
                        pair_key = f"{sorted_srcs[i]} <-> {sorted_srcs[j]}"
                        cross_src_pairs[pair_key] += 1

        if len(example_collisions) < max_example_collisions:
            example_collisions.append({
                "content_sha256": sha,
                "count": len(occurrences),
                "relationship": (
                    "within_file" if len(unique_files) == 1
                    else ("same_source_cross_mode" if len(unique_sources) == 1
                          else "cross_source")
                ),
                "occurrences": [
                    {
                        "source": o.source,
                        "file": Path(o.file_path).name,
                        "mode": o.mode,
                        "original_id": o.original_id,
                    }
                    for o in occurrences
                ],
            })

    return CrossSourceDuplicateReport(
        total_unique_hashes=total_unique,
        total_duplicate_hashes=dup_hashes,
        total_duplicate_documents=total_dup_docs,
        within_file_counts=dict(within_file),
        same_source_cross_mode_overlap=dict(same_src_cross_mode),
        cross_source_pair_counts=dict(cross_src_pairs),
        # Legacy aliases
        inter_source_pair_counts=dict(cross_src_pairs),
        intra_source_duplicate_counts=dict(within_file),
        sample_collisions=example_collisions,
    )
