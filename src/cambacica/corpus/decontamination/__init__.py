"""Pinned benchmark snapshots and read-only contamination matching."""

from .matcher import (
    CandidatePolicy,
    CorpusDocument,
    MatchField,
    MatchResult,
    MatcherRun,
    BenchmarkMatcher,
    normalize_match_text,
    tokenize_with_offsets,
)

__all__ = [
    "BenchmarkMatcher",
    "CandidatePolicy",
    "CorpusDocument",
    "MatchField",
    "MatchResult",
    "MatcherRun",
    "normalize_match_text",
    "tokenize_with_offsets",
]
