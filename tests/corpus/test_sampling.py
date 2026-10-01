"""Unit tests for deterministic sampling algorithms."""

import random
import pytest

from cambacica.corpus.sampling import (
    DeterministicReservoirSampler,
    StratifiedSampler,
    stable_hash64,
    stable_uniform,
)
from cambacica.corpus.sources.carolina import (
    _proportional_allocation,
    _select_distributed_shards,
    TAXONOMY_POPULATION,
)


def test_stable_hash64_determinism():
    """Verify that stable_hash64 is strictly deterministic across calls."""
    key = "doc_identity_987654"
    h1 = stable_hash64(key, seed=42)
    h2 = stable_hash64(key, seed=42)
    assert h1 == h2
    assert isinstance(h1, int)
    assert 0 <= h1 < 2**64

    # Seed changes result
    h3 = stable_hash64(key, seed=99)
    assert h1 != h3


def test_stable_uniform_range():
    """Verify stable_uniform maps to [0.0, 1.0)."""
    for i in range(100):
        val = stable_uniform(f"key_{i}", seed=123)
        assert 0.0 <= val < 1.0


def test_reservoir_sampler_exact_capacity():
    """Verify reservoir capacity enforcement and total_seen tracking."""
    capacity = 15
    sampler = DeterministicReservoirSampler(capacity=capacity, seed=42)

    for i in range(100):
        sampler.add(f"id_{i}", f"payload_{i}")

    assert len(sampler) == capacity
    assert sampler.total_seen == 100
    sample = sampler.get_sample()
    assert len(sample) == capacity


def test_reservoir_sampler_order_invariance():
    """Verify that sample selection is identical regardless of input stream order."""
    capacity = 10
    seed = 42

    items = [(f"doc_{i}", f"content_{i}") for i in range(100)]

    # Run 1: original order
    sampler1 = DeterministicReservoirSampler(capacity=capacity, seed=seed)
    for k, v in items:
        sampler1.add(k, v)
    sample1 = sampler1.get_sample()

    # Run 2: shuffled order
    shuffled_items = list(items)
    random.Random(1337).shuffle(shuffled_items)

    sampler2 = DeterministicReservoirSampler(capacity=capacity, seed=seed)
    for k, v in shuffled_items:
        sampler2.add(k, v)
    sample2 = sampler2.get_sample()

    # Exact deterministic identity
    assert sample1 == sample2


def test_stratified_sampler_quotas():
    """Verify that StratifiedSampler respects quotas for each stratum."""
    quotas = {
        "wik": 5,
        "jud": 10,
        "soc": 3,
    }
    sampler = StratifiedSampler(quotas=quotas, seed=42)

    for i in range(50):
        sampler.add("wik", f"wik_{i}", f"data_wik_{i}")
        sampler.add("jud", f"jud_{i}", f"data_jud_{i}")
        sampler.add("soc", f"soc_{i}", f"data_soc_{i}")
        # Add item to non-existent category
        sampler.add("unknown", f"unk_{i}", f"data_unk_{i}")

    assert sampler.is_full()
    counts = sampler.get_counts()
    assert counts == quotas

    all_samples = sampler.get_all_samples()
    assert len(all_samples) == sum(quotas.values())


# ---------------------------------------------------------------------------
# New tests required by the Gate C1 fix specification
# ---------------------------------------------------------------------------


def test_proportional_allocation_sums_to_target():
    """Verify proportional allocation sums exactly to target using largest-remainder."""
    target = 10_000
    allocation = _proportional_allocation(target, TAXONOMY_POPULATION)

    # Every taxonomy gets >= 0 documents (rare categories may be 0 in representative mode)
    for k, v in allocation.items():
        assert v >= 0, f"taxonomy {k!r} got negative allocation {v}"

    # Total must be exactly target with largest-remainder
    assert sum(allocation.values()) == target


def test_proportional_allocation_documented_counts():
    """Verify largest-remainder allocation matches documented counts for N=10,000."""
    target = 10_000
    allocation = _proportional_allocation(target, TAXONOMY_POPULATION)
    expected = {
        "dat": 5093,
        "wik": 4540,
        "jud": 181,
        "uni": 125,
        "soc": 42,
        "leg": 19,
        "pub": 0,
    }
    assert allocation == expected
    assert sum(allocation.values()) == target


def test_proportional_allocation_dominant_categories():
    """Verify dat and wik receive the largest allocations."""
    target = 10_000
    allocation = _proportional_allocation(target, TAXONOMY_POPULATION)

    # dat and wik are the two largest populations — they must dominate
    assert allocation["dat"] > allocation["wik"], (
        "dat allocation must exceed wik (dat is largest in Carolina v2.0.1)"
    )
    assert allocation["wik"] > allocation["jud"], (
        "wik allocation must exceed jud"
    )
    assert allocation["dat"] > allocation["jud"], (
        "dat allocation must exceed jud"
    )
    assert allocation["wik"] > allocation["leg"], (
        "wik allocation must exceed leg"
    )

    # dat should receive over 50% and wik over 45%
    dat_frac = allocation["dat"] / target
    wik_frac = allocation["wik"] / target
    assert dat_frac > 0.50, f"dat fraction {dat_frac:.2%} unexpectedly low"
    assert wik_frac > 0.40, f"wik fraction {wik_frac:.2%} unexpectedly low"


def test_select_distributed_shards_count():
    """Verify distributed shard selection returns the requested count."""
    files = [f"shard_{i:04d}.xml.gz" for i in range(50)]

    selected = _select_distributed_shards(files, n_shards=4, seed=42)
    assert len(selected) <= 4
    assert len(selected) > 0
    # All selected must come from the original list
    for s in selected:
        assert s in files


def test_select_distributed_shards_spans_range():
    """Verify distributed selection spans early, middle, and late shards."""
    files = [f"shard_{i:04d}.xml.gz" for i in range(60)]
    n = len(files)
    selected = _select_distributed_shards(files, n_shards=6, seed=42)

    indices = [files.index(s) for s in selected]
    # Min index should be in first third, max index in last third
    assert min(indices) < n // 3, "No shard selected from first third of collection"
    assert max(indices) >= (2 * n) // 3, "No shard selected from last third of collection"


def test_select_distributed_shards_determinism():
    """Verify shard selection is identical across repeated calls with same seed."""
    files = [f"shard_{i:04d}.xml.gz" for i in range(56)]
    sel1 = _select_distributed_shards(files, n_shards=5, seed=42)
    sel2 = _select_distributed_shards(files, n_shards=5, seed=42)
    assert sel1 == sel2


def test_stratified_sampler_underfill_reporting():
    """Verify StratifiedSampler counts correctly reflect partial fills."""
    quotas = {"wik": 100, "jud": 100}
    sampler = StratifiedSampler(quotas=quotas, seed=42)

    # Only supply 30 items to wik, 100 items to jud
    for i in range(30):
        sampler.add("wik", f"wik_{i}", f"data_wik_{i}")
    for i in range(100):
        sampler.add("jud", f"jud_{i}", f"data_jud_{i}")

    counts = sampler.get_counts()
    assert counts["wik"] == 30, "wik should be underfilled"
    assert counts["jud"] == 100, "jud should be full"

    # Sampler must NOT be full because wik is underfilled
    assert not sampler.is_full(), "Sampler should not report full when wik is underfilled"
