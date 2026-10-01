"""Deduplication and cross-source overlap diagnostics for Gate C1.

Provides exact duplicate detection across sources via content_sha256 and
a configurable MinHash diagnostic comparator for near-duplicate inspection.
"""

from cambacica.corpus.dedup.exact import (
    CrossSourceDuplicateReport,
    find_cross_source_exact_duplicates,
)
from cambacica.corpus.dedup.minhash import (
    CandidateDuplicatePair,
    MinHashConfig,
    find_minhash_near_duplicates,
)

__all__ = [
    "CrossSourceDuplicateReport",
    "find_cross_source_exact_duplicates",
    "MinHashConfig",
    "CandidateDuplicatePair",
    "find_minhash_near_duplicates",
]
