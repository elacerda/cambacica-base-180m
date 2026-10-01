"""Deterministic sampling algorithms for Gate C1 corpus exploration.

This module provides stable hash-based sampling functions and reservoir
samplers to ensure experiments are reproducible across runs and independent of
physical row order or partition boundaries.
"""

from __future__ import annotations

import heapq
from typing import Any, Dict, List, Tuple
import xxhash


def stable_hash64(key: str | bytes, seed: int = 0) -> int:
    """Compute a deterministic 64-bit unsigned integer hash.

    Uses xxHash64 for high-throughput, uniform bit distribution.

    Parameters
    ----------
    key : str or bytes
        Input identifier or content string.
    seed : int, default 0
        Deterministic integer seed.

    Returns
    -------
    int
        Unsigned 64-bit integer in the range [0, 2**64 - 1].
    """
    if isinstance(key, str):
        key_bytes = key.encode("utf-8")
    else:
        key_bytes = key

    # Combine seed into xxhash64
    hasher = xxhash.xxh64(key_bytes, seed=seed & 0xFFFFFFFF)
    return hasher.intdigest()


def stable_uniform(key: str | bytes, seed: int = 0) -> float:
    """Map a key to a deterministic uniform float in [0.0, 1.0).

    Parameters
    ----------
    key : str or bytes
        Input identifier.
    seed : int, default 0
        Deterministic seed.

    Returns
    -------
    float
        Floating-point number uniformly distributed in [0.0, 1.0).
    """
    h = stable_hash64(key, seed=seed)
    return h / 18446744073709551616.0  # 2**64


class DeterministicReservoirSampler:
    """Bounded deterministic reservoir sampler.

    Retains the items with the smallest hash values `stable_hash64(id, seed)`.
    This guarantees that the exact same sample of size `capacity` is chosen
    regardless of physical iteration order or chunk sizes.

    Parameters
    ----------
    capacity : int
        Maximum number of items to retain. Must be greater than 0.
    seed : int, default 42
        Deterministic random seed.

    Raises
    ------
    ValueError
        If capacity is less than 1.
    """

    def __init__(self, capacity: int, seed: int = 42) -> None:
        if capacity < 1:
            raise ValueError(f"Capacity must be >= 1, got {capacity}.")
        self.capacity: int = capacity
        self.seed: int = seed
        # Max-heap storing (-hash, tiebreaker, item)
        self._heap: List[Tuple[int, int, Any]] = []
        self._count: int = 0

    def add(self, key: str | bytes, item: Any) -> bool:
        """Consider adding an item to the reservoir based on its key hash.

        Parameters
        ----------
        key : str or bytes
            Unique identifier used for deterministic hashing.
        item : any
            Payload object associated with the key.

        Returns
        -------
        bool
            True if the item was added or replaced an existing item in the
            reservoir, False if discarded.
        """
        self._count += 1
        h = stable_hash64(key, seed=self.seed)

        if len(self._heap) < self.capacity:
            # Heap not full; push
            heapq.heappush(self._heap, (-h, self._count, item))
            return True

        # Heap full; compare with current max hash (root of max-heap)
        current_max_neg_hash = self._heap[0][0]
        current_max_hash = -current_max_neg_hash

        if h < current_max_hash:
            # Replace the maximum hash with the new smaller hash
            heapq.heapreplace(self._heap, (-h, self._count, item))
            return True

        return False

    def get_sample(self) -> List[Any]:
        """Retrieve collected items sorted deterministically by their hash.

        Returns
        -------
        list
            Items currently held in the reservoir, sorted by ascending hash.
        """
        # Sort items by hash (ascending)
        sorted_heap = sorted(self._heap, key=lambda entry: (-entry[0], entry[1]))
        return [entry[2] for entry in sorted_heap]

    def __len__(self) -> int:
        """Return the current number of items in the reservoir."""
        return len(self._heap)

    @property
    def total_seen(self) -> int:
        """Return the total number of items considered so far."""
        return self._count


class StratifiedSampler:
    """Maintains independent deterministic reservoirs per stratum/category.

    Parameters
    ----------
    quotas : dict of str to int
        Mapping from stratum/category name to target sample capacity.
    seed : int, default 42
        Deterministic random seed.

    Raises
    ------
    ValueError
        If any quota is less than 1.
    """

    def __init__(self, quotas: Dict[str, int], seed: int = 42) -> None:
        self.quotas: Dict[str, int] = dict(quotas)
        self.seed: int = seed
        self._reservoirs: Dict[str, DeterministicReservoirSampler] = {}

        for category, capacity in self.quotas.items():
            if capacity < 1:
                raise ValueError(
                    f"Quota for '{category}' must be >= 1, got {capacity}."
                )
            # Derive distinct seed per stratum to ensure statistical independence
            stratum_seed = stable_hash64(category, seed=seed) & 0xFFFFFFFF
            self._reservoirs[category] = DeterministicReservoirSampler(
                capacity=capacity,
                seed=stratum_seed,
            )

    def add(self, category: str, key: str | bytes, item: Any) -> bool:
        """Add an item to the specified stratum's reservoir.

        Parameters
        ----------
        category : str
            Category name matching one of the configured quotas.
        key : str or bytes
            Unique identifier for deterministic hashing.
        item : any
            Payload object.

        Returns
        -------
        bool
            True if the item was accepted into the stratum reservoir, False
            otherwise (e.g. category not tracked or discarded).
        """
        reservoir = self._reservoirs.get(category)
        if reservoir is None:
            return False
        return reservoir.add(key, item)

    def is_full(self) -> bool:
        """Check whether all stratum reservoirs have reached their quota.

        Returns
        -------
        bool
            True if every configured stratum has reached its capacity.
        """
        return all(len(res) >= res.capacity for res in self._reservoirs.values())

    def get_sample_by_category(self) -> Dict[str, List[Any]]:
        """Retrieve samples grouped by category.

        Returns
        -------
        dict
            Mapping from category to list of sampled items.
        """
        return {cat: res.get_sample() for cat, res in self._reservoirs.items()}

    def get_all_samples(self) -> List[Any]:
        """Retrieve all sampled items across all strata.

        Returns
        -------
        list
            Combined list of all sampled items.
        """
        all_items: List[Any] = []
        for res in self._reservoirs.values():
            all_items.extend(res.get_sample())
        return all_items

    def get_counts(self) -> Dict[str, int]:
        """Get the current item counts for each stratum.

        Returns
        -------
        dict
            Mapping from category name to current sample count.
        """
        return {cat: len(res) for cat, res in self._reservoirs.items()}
