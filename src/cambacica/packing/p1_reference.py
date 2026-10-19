"""Pure-Python, synthetic-only reference for C4-P1 contiguous EOS packing.

This module accepts only caller-supplied token IDs and provenance metadata.
It performs no file, tokenizer, model, CUDA, NVML, or network operations.
The implementation is a correctness reference, not a production data path.

BOS insertion and scoring of inserted EOS targets remain provisional policy
choices. The defaults here omit BOS and score inserted EOS targets.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Iterable, Iterator

CONTEXT_LENGTH = 4096
VOCAB_SIZE = 24576
BOS_ID = 1
EOS_ID = 2
PAD_INPUT_ID = EOS_ID
IGNORE_INDEX = -100
SPLITS = ("train", "validation", "test")
SCHEMA_VERSION = "cambacica-c4-p1-synthetic-reference-v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _metadata_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty, trimmed string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _token_id(name: str, value: object) -> int:
    if type(value) is not int or not 0 <= value < VOCAB_SIZE:
        raise ValueError(f"{name} must be an integer token ID in [0, {VOCAB_SIZE})")
    return value


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PackingPolicy:
    """Explicit provisional BOS/EOS-loss and safe-input-padding choices."""

    insert_bos_at_split_start: bool = False
    score_eos_targets: bool = True
    safe_input_padding_id: int = PAD_INPUT_ID

    def __post_init__(self) -> None:
        if type(self.insert_bos_at_split_start) is not bool:
            raise ValueError("insert_bos_at_split_start must be a bool")
        if type(self.score_eos_targets) is not bool:
            raise ValueError("score_eos_targets must be a bool")
        _token_id("safe_input_padding_id", self.safe_input_padding_id)


@dataclass(frozen=True)
class Document:
    """One artificial tokenized document and its artificial provenance."""

    document_id: str
    group_id: str
    split: str
    source: str
    subset: str
    content_hash: str
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _metadata_string("document_id", self.document_id)
        _metadata_string("group_id", self.group_id)
        _metadata_string("split", self.split)
        _metadata_string("source", self.source)
        _metadata_string("subset", self.subset)
        if not isinstance(self.content_hash, str) or not _SHA256_RE.fullmatch(
            self.content_hash
        ):
            raise ValueError("content_hash must be a lowercase SHA-256 hex digest")
        try:
            token_ids = tuple(self.token_ids)
        except TypeError as error:
            raise ValueError(
                "token_ids must be an iterable of integer token IDs"
            ) from error
        for index, token_id in enumerate(token_ids):
            _token_id(f"token_ids[{index}]", token_id)
        object.__setattr__(self, "token_ids", token_ids)


@dataclass(frozen=True)
class DocumentSpan:
    """Payload range and separate inserted EOS location for one document."""

    document_id: str
    group_id: str
    split: str
    source: str
    subset: str
    content_hash: str
    stream_start: int
    stream_end: int
    eos_offset: int
    payload_digest: str

    @property
    def payload_token_count(self) -> int:
        return self.stream_end - self.stream_start


@dataclass(frozen=True)
class WindowProvenance:
    """Document-relative payload and EOS coordinates intersecting a window."""

    document_id: str
    payload_stream_start: int | None
    payload_stream_end: int | None
    document_token_start: int | None
    document_token_end: int | None
    eos_stream_offset: int | None


@dataclass(frozen=True)
class TrainingWindow:
    """One fixed-size input/label window with disjoint target ownership."""

    window_index: int
    input_start: int
    backing_stream_start: int
    backing_stream_end: int
    valid_target_count: int
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    source_spans: tuple[WindowProvenance, ...]

    @property
    def owned_target_indices(self) -> range:
        """Global stream indices of targets owned by this window."""
        return range(
            self.input_start + 1, self.input_start + 1 + self.valid_target_count
        )

    @property
    def scored_target_count(self) -> int:
        return sum(label != IGNORE_INDEX for label in self.labels)

    def scored_pairs(self) -> Iterator[tuple[int, int]]:
        """Yield input/target pairs that a loss consumer should score."""
        return (
            (input_id, label)
            for input_id, label in zip(self.input_ids, self.labels, strict=True)
            if label != IGNORE_INDEX
        )


@dataclass(frozen=True)
class SplitAccounting:
    document_count: int
    empty_document_count: int
    payload_tokens: int
    eos_tokens: int
    bos_tokens: int
    real_stream_tokens: int
    candidate_target_count: int
    unique_scored_targets: int
    first_stream_token_unscored: bool
    window_count: int
    input_pad_slots: int
    ignored_label_slots: int
    eos_masked_label_count: int
    scored_label_fraction: float
    source_token_counts: tuple[tuple[str, int], ...]

    @property
    def source_token_shares(self) -> tuple[tuple[str, float], ...]:
        if self.payload_tokens == 0:
            return tuple((source, 0.0) for source, _ in self.source_token_counts)
        return tuple(
            (source, count / self.payload_tokens)
            for source, count in self.source_token_counts
        )


@dataclass(frozen=True)
class PackedSplit:
    split: str
    tokens: tuple[int, ...]
    document_spans: tuple[DocumentSpan, ...]
    windows: tuple[TrainingWindow, ...]
    policy: PackingPolicy
    bos_offset: int | None
    accounting: SplitAccounting
    stream_digest: str
    ordered_span_digest: str
    identity_digest: str


def _span_digest_payload(spans: tuple[DocumentSpan, ...]) -> list[dict[str, object]]:
    return [
        {
            "content_hash": span.content_hash,
            "document_id": span.document_id,
            "eos_offset": span.eos_offset,
            "group_id": span.group_id,
            "payload_digest": span.payload_digest,
            "source": span.source,
            "split": span.split,
            "stream_end": span.stream_end,
            "stream_start": span.stream_start,
            "subset": span.subset,
        }
        for span in spans
    ]


def _window_provenance(
    spans: tuple[DocumentSpan, ...], backing_start: int, backing_end: int
) -> tuple[WindowProvenance, ...]:
    provenance: list[WindowProvenance] = []
    for span in spans:
        overlap_start = max(span.stream_start, backing_start)
        overlap_end = min(span.stream_end, backing_end)
        has_payload = overlap_start < overlap_end
        eos_in_window = backing_start <= span.eos_offset < backing_end
        if not has_payload and not eos_in_window:
            continue
        provenance.append(
            WindowProvenance(
                document_id=span.document_id,
                payload_stream_start=overlap_start if has_payload else None,
                payload_stream_end=overlap_end if has_payload else None,
                document_token_start=(
                    overlap_start - span.stream_start if has_payload else None
                ),
                document_token_end=(
                    overlap_end - span.stream_start if has_payload else None
                ),
                eos_stream_offset=span.eos_offset if eos_in_window else None,
            )
        )
    return tuple(provenance)


def _make_windows(
    tokens: tuple[int, ...],
    spans: tuple[DocumentSpan, ...],
    policy: PackingPolicy,
) -> tuple[TrainingWindow, ...]:
    candidate_target_count = max(len(tokens) - 1, 0)
    eos_offsets = {span.eos_offset for span in spans}
    windows: list[TrainingWindow] = []
    for window_index, input_start in enumerate(
        range(0, candidate_target_count, CONTEXT_LENGTH)
    ):
        valid_target_count = min(CONTEXT_LENGTH, candidate_target_count - input_start)
        backing_end = input_start + valid_target_count + 1
        inputs = tokens[input_start : input_start + valid_target_count]
        targets = list(tokens[input_start + 1 : backing_end])
        if not policy.score_eos_targets:
            for local_index, target_index in enumerate(
                range(input_start + 1, backing_end)
            ):
                if target_index in eos_offsets:
                    targets[local_index] = IGNORE_INDEX
        pad_count = CONTEXT_LENGTH - valid_target_count
        windows.append(
            TrainingWindow(
                window_index=window_index,
                input_start=input_start,
                backing_stream_start=input_start,
                backing_stream_end=backing_end,
                valid_target_count=valid_target_count,
                input_ids=inputs + (policy.safe_input_padding_id,) * pad_count,
                labels=tuple(targets) + (IGNORE_INDEX,) * pad_count,
                source_spans=_window_provenance(spans, input_start, backing_end),
            )
        )
    return tuple(windows)


def _make_accounting(
    tokens: tuple[int, ...],
    spans: tuple[DocumentSpan, ...],
    windows: tuple[TrainingWindow, ...],
    policy: PackingPolicy,
    bos_offset: int | None,
) -> SplitAccounting:
    source_counts: dict[str, int] = {}
    for span in spans:
        source_counts[span.source] = source_counts.get(span.source, 0) + (
            span.stream_end - span.stream_start
        )
    source_token_counts = tuple(sorted(source_counts.items()))
    payload_tokens = sum(span.payload_token_count for span in spans)
    candidate_target_count = max(len(tokens) - 1, 0)
    unique_scored_targets = sum(window.scored_target_count for window in windows)
    ignored_label_slots = sum(
        label == IGNORE_INDEX for window in windows for label in window.labels
    )
    input_pad_slots = sum(
        CONTEXT_LENGTH - window.valid_target_count for window in windows
    )
    eos_masked_label_count = candidate_target_count - unique_scored_targets
    return SplitAccounting(
        document_count=len(spans),
        empty_document_count=sum(span.payload_token_count == 0 for span in spans),
        payload_tokens=payload_tokens,
        eos_tokens=len(spans),
        bos_tokens=int(bos_offset is not None),
        real_stream_tokens=len(tokens),
        candidate_target_count=candidate_target_count,
        unique_scored_targets=unique_scored_targets,
        first_stream_token_unscored=bool(tokens),
        window_count=len(windows),
        input_pad_slots=input_pad_slots,
        ignored_label_slots=ignored_label_slots,
        eos_masked_label_count=eos_masked_label_count,
        scored_label_fraction=(
            unique_scored_targets / candidate_target_count
            if candidate_target_count
            else 0.0
        ),
        source_token_counts=source_token_counts,
    )


def _identity_digest(packed: PackedSplit) -> str:
    return _canonical_digest(
        {
            "context_length": CONTEXT_LENGTH,
            "ignore_index": IGNORE_INDEX,
            "ordered_span_digest": packed.ordered_span_digest,
            "policy": {
                "insert_bos_at_split_start": packed.policy.insert_bos_at_split_start,
                "safe_input_padding_id": packed.policy.safe_input_padding_id,
                "score_eos_targets": packed.policy.score_eos_targets,
            },
            "schema_version": SCHEMA_VERSION,
            "split": packed.split,
            "stream_digest": packed.stream_digest,
            "vocab_size": VOCAB_SIZE,
        }
    )


def pack_documents_by_split(
    documents: Iterable[Document], policy: PackingPolicy = PackingPolicy()
) -> dict[str, PackedSplit]:
    """Build independent train/validation/test streams from artificial records.

    Input order is preserved within each split. The function never reads or
    writes files and never discovers documents; it only uses the iterable
    supplied by the caller.
    """
    if not isinstance(policy, PackingPolicy):
        raise TypeError("policy must be a PackingPolicy")
    try:
        document_rows = tuple(documents)
    except TypeError as error:
        raise ValueError("documents must be an iterable of Document values") from error

    seen_document_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    by_split: dict[str, list[Document]] = {split: [] for split in SPLITS}
    for index, document in enumerate(document_rows):
        if not isinstance(document, Document):
            raise TypeError(f"documents[{index}] must be a Document")
        if document.split not in SPLITS:
            raise ValueError(f"unknown split {document.split!r}")
        if document.document_id in seen_document_ids:
            raise ValueError(f"duplicate document_id {document.document_id!r}")
        seen_document_ids.add(document.document_id)
        previous_split = group_splits.setdefault(document.group_id, document.split)
        if previous_split != document.split:
            raise ValueError(
                f"group_id {document.group_id!r} appears in multiple splits"
            )
        by_split[document.split].append(document)

    packed_by_split: dict[str, PackedSplit] = {}
    for split in SPLITS:
        rows = by_split[split]
        tokens_list: list[int] = []
        spans_list: list[DocumentSpan] = []
        bos_offset = None
        if rows and policy.insert_bos_at_split_start:
            bos_offset = 0
            tokens_list.append(BOS_ID)
        for document in rows:
            stream_start = len(tokens_list)
            tokens_list.extend(document.token_ids)
            stream_end = len(tokens_list)
            eos_offset = len(tokens_list)
            tokens_list.append(EOS_ID)
            spans_list.append(
                DocumentSpan(
                    document_id=document.document_id,
                    group_id=document.group_id,
                    split=document.split,
                    source=document.source,
                    subset=document.subset,
                    content_hash=document.content_hash,
                    stream_start=stream_start,
                    stream_end=stream_end,
                    eos_offset=eos_offset,
                    payload_digest=_canonical_digest(list(document.token_ids)),
                )
            )
        tokens = tuple(tokens_list)
        spans = tuple(spans_list)
        windows = _make_windows(tokens, spans, policy)
        accounting = _make_accounting(tokens, spans, windows, policy, bos_offset)
        stream_digest = _canonical_digest(
            {"schema_version": SCHEMA_VERSION, "tokens": list(tokens)}
        )
        ordered_span_digest = _canonical_digest(_span_digest_payload(spans))
        packed = PackedSplit(
            split=split,
            tokens=tokens,
            document_spans=spans,
            windows=windows,
            policy=policy,
            bos_offset=bos_offset,
            accounting=accounting,
            stream_digest=stream_digest,
            ordered_span_digest=ordered_span_digest,
            identity_digest="",
        )
        packed = replace(packed, identity_digest=_identity_digest(packed))
        validate_packed_split(packed)
        packed_by_split[split] = packed
    return packed_by_split


def validate_packed_split(packed: PackedSplit) -> None:
    """Fail closed if stream, spans, windows, totals, or hashes disagree."""
    if not isinstance(packed, PackedSplit):
        raise TypeError("packed must be a PackedSplit")
    if packed.split not in SPLITS:
        raise ValueError(f"unknown split {packed.split!r}")
    if not isinstance(packed.policy, PackingPolicy):
        raise ValueError("packed policy is malformed")
    if len(packed.tokens) and packed.bos_offset is not None:
        if packed.bos_offset != 0 or packed.tokens[0] != BOS_ID:
            raise ValueError("BOS provenance does not match the stream")
    expected_bos_offset = (
        0 if packed.document_spans and packed.policy.insert_bos_at_split_start else None
    )
    if packed.bos_offset != expected_bos_offset:
        raise ValueError("BOS policy and BOS provenance do not match")
    for index, token_id in enumerate(packed.tokens):
        _token_id(f"tokens[{index}]", token_id)

    cursor = int(packed.bos_offset is not None)
    seen_document_ids: set[str] = set()
    for span_index, span in enumerate(packed.document_spans):
        for name in ("document_id", "group_id", "source", "subset"):
            _metadata_string(name, getattr(span, name))
        if span.split != packed.split:
            raise ValueError("document span split does not match packed split")
        if span.document_id in seen_document_ids:
            raise ValueError("duplicate document_id in ordered spans")
        seen_document_ids.add(span.document_id)
        if not isinstance(span.content_hash, str) or not _SHA256_RE.fullmatch(
            span.content_hash
        ):
            raise ValueError("document span content_hash is malformed")
        for name in ("stream_start", "stream_end", "eos_offset"):
            value = getattr(span, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"document span {span_index} {name} is malformed")
        if span.stream_start != cursor or not cursor <= span.stream_end <= len(
            packed.tokens
        ):
            raise ValueError(f"document span {span_index} has unmatched stream totals")
        if span.eos_offset != span.stream_end or span.eos_offset >= len(packed.tokens):
            raise ValueError(f"document span {span_index} has an invalid EOS offset")
        if not isinstance(span.payload_digest, str) or not _SHA256_RE.fullmatch(
            span.payload_digest
        ):
            raise ValueError(f"document span {span_index} payload digest is malformed")
        if packed.tokens[span.eos_offset] != EOS_ID:
            raise ValueError(f"document span {span_index} EOS ID does not match")
        payload = list(packed.tokens[span.stream_start : span.stream_end])
        if span.payload_digest != _canonical_digest(payload):
            raise ValueError(
                f"document span {span_index} payload digest does not match"
            )
        cursor = span.eos_offset + 1
    if cursor != len(packed.tokens):
        raise ValueError("stream contains tokens without document or BOS provenance")

    expected_stream_digest = _canonical_digest(
        {"schema_version": SCHEMA_VERSION, "tokens": list(packed.tokens)}
    )
    if packed.stream_digest != expected_stream_digest:
        raise ValueError("stream digest does not match tokens")
    expected_span_digest = _canonical_digest(
        _span_digest_payload(packed.document_spans)
    )
    if packed.ordered_span_digest != expected_span_digest:
        raise ValueError("ordered span digest does not match provenance")

    expected_windows = _make_windows(
        packed.tokens, packed.document_spans, packed.policy
    )
    if packed.windows != expected_windows:
        raise ValueError("window alignment, padding, ownership, or provenance mismatch")
    owned_targets = tuple(
        target_index
        for window in packed.windows
        for target_index in window.owned_target_indices
    )
    if owned_targets != tuple(range(1, len(packed.tokens))):
        raise ValueError("next-token target positions are missing or multiply owned")
    expected_accounting = _make_accounting(
        packed.tokens,
        packed.document_spans,
        packed.windows,
        packed.policy,
        packed.bos_offset,
    )
    if packed.accounting != expected_accounting:
        raise ValueError("accounting totals do not match stream, spans, and windows")
    if packed.identity_digest != _identity_digest(packed):
        raise ValueError("canonical identity digest does not match packed split")
