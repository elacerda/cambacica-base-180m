"""MinHash near-duplicate diagnostics for Gate C1 corpus exploration.

Provides lightweight MinHash signature generation and candidate pair identification
for empirical near-duplicate inspection across samples. Does not automatically
delete data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import xxhash


@dataclass(frozen=True)
class MinHashConfig:
    """Configuration for MinHash shingling and hashing.

    Parameters
    ----------
    num_permutations : int, default 64
        Number of hash permutations (signature length).
    ngram_size : int, default 5
        Word n-gram size for shingling.
    seed : int, default 42
        Deterministic random seed.
    """

    num_permutations: int = 64
    ngram_size: int = 5
    seed: int = 42


@dataclass
class CandidateDuplicatePair:
    """Pair of documents identified as potential near-duplicates."""

    doc_id_1: str
    source_1: str
    doc_id_2: str
    source_2: str
    estimated_jaccard: float
    snippet_1: str
    snippet_2: str
    file_1: Optional[str] = None
    file_2: Optional[str] = None
    relationship: str = "CROSS_SOURCE"

    def to_dict(self) -> Dict[str, Any]:
        """Convert candidate pair to dictionary.

        Returns
        -------
        dict
            Mapping representing the candidate pair.
        """
        return asdict(self)


def get_word_shingles(text: str, n: int = 5) -> Set[str]:
    """Generate normalized word n-gram shingles from text.

    Parameters
    ----------
    text : str
        Input text.
    n : int, default 5
        N-gram size.

    Returns
    -------
    set of str
        Set of whitespace-separated n-gram strings.
    """
    words = [w.lower() for w in re.findall(r"\w+", text)]
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def compute_minhash_signature(
    text: str,
    config: MinHashConfig,
) -> List[int]:
    """Compute the MinHash signature of a text document.

    Parameters
    ----------
    text : str
        Document text.
    config : MinHashConfig
        MinHash configuration parameters.

    Returns
    -------
    list of int
        Array of length `config.num_permutations` containing minimum hash values.
    """
    shingles = get_word_shingles(text, n=config.ngram_size)
    if not shingles:
        return [0] * config.num_permutations

    signature: List[int] = [0xFFFFFFFFFFFFFFFF] * config.num_permutations
    shingle_bytes = [s.encode("utf-8") for s in shingles]

    for p in range(config.num_permutations):
        perm_seed = (config.seed + p * 10007) & 0xFFFFFFFF
        min_val = min(
            xxhash.xxh64(sb, seed=perm_seed).intdigest() for sb in shingle_bytes
        )
        signature[p] = min_val

    return signature


def estimate_jaccard_similarity(sig1: Sequence[int], sig2: Sequence[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures.

    Parameters
    ----------
    sig1 : sequence of int
        First signature.
    sig2 : sequence of int
        Second signature of equal length.

    Returns
    -------
    float
        Estimated Jaccard similarity in [0.0, 1.0].
    """
    if not sig1 or not sig2 or len(sig1) != len(sig2):
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return round(matches / len(sig1), 4)


def find_minhash_near_duplicates(
    documents: Sequence[dict],
    config: MinHashConfig,
    threshold: float = 0.80,
    max_candidates: int = 50,
) -> List[CandidateDuplicatePair]:
    """Identify candidate near-duplicate document pairs using MinHash.

    Distinguishes candidates by relationship:
    - 'WITHIN_FILE': Documents originating from the same file.
    - 'SAME_SOURCE_CROSS_MODE': Documents from the same upstream source across different files/modes.
    - 'CROSS_SOURCE': Documents from genuinely different upstream sources.

    Maintains distinct candidate budgets so that exact or near-identical
    cross-mode overlap pairs do not crowd out genuine cross-source candidates.

    Parameters
    ----------
    documents : sequence of dict
        List of document dicts with keys: 'text', 'source', 'original_id', and optional 'file_path'.
    config : MinHashConfig
        MinHash configuration.
    threshold : float, default 0.80
        Jaccard similarity threshold for flagging a candidate pair.
    max_candidates : int, default 50
        Maximum candidate pairs per relationship category to retain.

    Returns
    -------
    list of CandidateDuplicatePair
        Candidate near-duplicate pairs with similarity >= threshold.
    """
    signatures: List[Tuple[dict, List[int]]] = []
    for doc in documents:
        sig = compute_minhash_signature(doc.get("text", ""), config)
        signatures.append((doc, sig))

    candidates_by_rel: Dict[str, List[CandidateDuplicatePair]] = {
        "CROSS_SOURCE": [],
        "SAME_SOURCE_CROSS_MODE": [],
        "WITHIN_FILE": [],
    }

    n_docs = len(signatures)
    for i in range(n_docs):
        doc1, sig1 = signatures[i]
        src1 = str(doc1.get("source") or "src1")
        file1 = doc1.get("file_path")
        for j in range(i + 1, n_docs):
            doc2, sig2 = signatures[j]
            src2 = str(doc2.get("source") or "src2")
            file2 = doc2.get("file_path")

            if file1 and file2 and file1 == file2:
                rel = "WITHIN_FILE"
            elif src1 == src2:
                rel = "SAME_SOURCE_CROSS_MODE"
            else:
                rel = "CROSS_SOURCE"

            # Skip checking if category quota is already filled
            if len(candidates_by_rel[rel]) >= max_candidates:
                continue

            sim = estimate_jaccard_similarity(sig1, sig2)
            if sim >= threshold:
                candidates_by_rel[rel].append(
                    CandidateDuplicatePair(
                        doc_id_1=str(doc1.get("original_id") or i),
                        source_1=src1,
                        doc_id_2=str(doc2.get("original_id") or j),
                        source_2=src2,
                        estimated_jaccard=sim,
                        snippet_1=(doc1.get("text", "")[:120]).replace("\n", " "),
                        snippet_2=(doc2.get("text", "")[:120]).replace("\n", " "),
                        file_1=str(file1) if file1 else None,
                        file_2=str(file2) if file2 else None,
                        relationship=rel,
                    )
                )

        # Early exit if all categories reached capacity
        if all(len(c) >= max_candidates for c in candidates_by_rel.values()):
            break

    # Return CROSS_SOURCE first, then SAME_SOURCE_CROSS_MODE, then WITHIN_FILE
    result = (
        candidates_by_rel["CROSS_SOURCE"]
        + candidates_by_rel["SAME_SOURCE_CROSS_MODE"]
        + candidates_by_rel["WITHIN_FILE"]
    )
    return result
