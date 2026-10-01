"""Cambacica corpus module for Gate C1 (CORPUS).

Provides unified schemas, sampling strategies, inspection metrics,
and source-specific stream samplers.
"""

from cambacica.corpus.schema import (
    DOCUMENT_FIELDS,
    PARQUET_SCHEMA,
    NormalizedDocument,
    load_sample_parquet,
    save_sample_parquet,
    validate_and_normalize,
)

__all__ = [
    "DOCUMENT_FIELDS",
    "PARQUET_SCHEMA",
    "NormalizedDocument",
    "validate_and_normalize",
    "save_sample_parquet",
    "load_sample_parquet",
]
