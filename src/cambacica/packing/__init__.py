"""Synthetic-only reference packing implementations."""

from cambacica.packing.p1_reference import (
    BOS_ID,
    CONTEXT_LENGTH,
    EOS_ID,
    IGNORE_INDEX,
    PAD_INPUT_ID,
    SPLITS,
    VOCAB_SIZE,
    Document,
    DocumentSpan,
    PackedSplit,
    PackingPolicy,
    SplitAccounting,
    TrainingWindow,
    WindowProvenance,
    pack_documents_by_split,
    validate_packed_split,
)

__all__ = [
    "BOS_ID",
    "CONTEXT_LENGTH",
    "EOS_ID",
    "IGNORE_INDEX",
    "PAD_INPUT_ID",
    "SPLITS",
    "VOCAB_SIZE",
    "Document",
    "DocumentSpan",
    "PackedSplit",
    "PackingPolicy",
    "SplitAccounting",
    "TrainingWindow",
    "WindowProvenance",
    "pack_documents_by_split",
    "validate_packed_split",
]
