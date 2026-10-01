"""Source-specific samplers for Gate C1.

Provides samplers for Corpus Carolina, Portuguese Wikipedia, ParlamentoPT,
GigaVerbo-v2, and Project Gutenberg Portuguese.
"""

from cambacica.corpus.sources.base import BaseSourceSampler
from cambacica.corpus.sources.carolina import CarolinaSampler
from cambacica.corpus.sources.gigaverbo_v2 import (
    GigaVerboSampler,
    is_subset_excluded,
    load_gigaverbo_exclusions,
)
from cambacica.corpus.sources.gutenberg_pt import GutenbergPTSampler
from cambacica.corpus.sources.parlamento_pt import ParlamentoPTSampler
from cambacica.corpus.sources.wikipedia import WikipediaSampler

SAMPLER_REGISTRY = {
    "carolina": CarolinaSampler,
    "wikipedia_pt": WikipediaSampler,
    "parlamento_pt": ParlamentoPTSampler,
    "gigaverbo_v2": GigaVerboSampler,
    "gutenberg_pt": GutenbergPTSampler,
}

__all__ = [
    "BaseSourceSampler",
    "CarolinaSampler",
    "WikipediaSampler",
    "ParlamentoPTSampler",
    "GigaVerboSampler",
    "GutenbergPTSampler",
    "SAMPLER_REGISTRY",
    "load_gigaverbo_exclusions",
    "is_subset_excluded",
]
