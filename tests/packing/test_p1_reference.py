from __future__ import annotations

import builtins
import hashlib
import os
from dataclasses import replace
from pathlib import Path

import pytest

from cambacica.packing.p1_reference import (
    CONTEXT_LENGTH,
    EOS_ID,
    IGNORE_INDEX,
    PAD_INPUT_ID,
    VOCAB_SIZE,
    Document,
    PackingPolicy,
    pack_documents_by_split,
    validate_packed_split,
)


def synthetic_document(
    document_id: str,
    token_ids: tuple[int, ...] | list[int],
    *,
    split: str = "train",
    group_id: str | None = None,
    source: str = "synthetic-source",
    subset: str = "synthetic-subset",
) -> Document:
    payload = ",".join(str(token_id) for token_id in token_ids)
    content_hash = hashlib.sha256(
        f"synthetic-fixture:{document_id}:{payload}".encode("utf-8")
    ).hexdigest()
    return Document(
        document_id=document_id,
        group_id=group_id or f"synthetic-group:{document_id}",
        split=split,
        source=source,
        subset=subset,
        content_hash=content_hash,
        token_ids=tuple(token_ids),
    )


def train_split(documents: list[Document], policy: PackingPolicy | None = None):
    packed = pack_documents_by_split(documents, policy or PackingPolicy())
    return packed["train"]


def test_hand_worked_eos_stream_spans_and_cross_document_transition() -> None:
    packed = train_split(
        [
            synthetic_document("synthetic:alpha", [10, 11]),
            synthetic_document("synthetic:beta", [12]),
            synthetic_document("synthetic:empty", []),
        ]
    )

    assert packed.tokens == (10, 11, EOS_ID, 12, EOS_ID, EOS_ID)
    assert [
        (span.document_id, span.stream_start, span.stream_end, span.eos_offset)
        for span in packed.document_spans
    ] == [
        ("synthetic:alpha", 0, 2, 2),
        ("synthetic:beta", 3, 4, 4),
        ("synthetic:empty", 5, 5, 5),
    ]
    window = packed.windows[0]
    assert tuple(window.owned_target_indices) == (1, 2, 3, 4, 5)
    assert window.labels[:5] == (11, EOS_ID, 12, EOS_ID, EOS_ID)
    assert (EOS_ID, 12) in tuple(window.scored_pairs())
    assert packed.accounting.payload_tokens == 3
    assert packed.accounting.eos_tokens == 3
    assert packed.accounting.candidate_target_count == 5
    assert packed.accounting.unique_scored_targets == 5
    assert packed.accounting.source_token_counts == (("synthetic-source", 3),)
    assert packed.accounting.source_token_shares == (("synthetic-source", 1.0),)


def test_interleaved_splits_build_independent_streams_without_token_leakage() -> None:
    packed = pack_documents_by_split(
        [
            synthetic_document("synthetic:train-a", [77], split="train"),
            synthetic_document("synthetic:valid-a", [77], split="validation"),
            synthetic_document("synthetic:test-a", [88], split="test"),
            synthetic_document("synthetic:train-b", [88], split="train"),
        ]
    )

    assert packed["train"].tokens == (77, EOS_ID, 88, EOS_ID)
    assert packed["validation"].tokens == (77, EOS_ID)
    assert packed["test"].tokens == (88, EOS_ID)
    assert [span.document_id for span in packed["train"].document_spans] == [
        "synthetic:train-a",
        "synthetic:train-b",
    ]
    assert all(
        span.split == split
        for split, item in packed.items()
        for span in item.document_spans
    )


def test_two_full_windows_and_partial_tail_align_and_own_each_target_once() -> None:
    payload = tuple(index % 20_000 for index in range(2 * CONTEXT_LENGTH + 5))
    packed = train_split([synthetic_document("synthetic:long", payload)])
    expected_targets = tuple(range(1, len(packed.tokens)))

    assert [window.valid_target_count for window in packed.windows] == [
        CONTEXT_LENGTH,
        CONTEXT_LENGTH,
        5,
    ]
    assert (
        tuple(
            target
            for window in packed.windows
            for target in window.owned_target_indices
        )
        == expected_targets
    )
    for window in packed.windows:
        count = window.valid_target_count
        start = window.input_start
        assert window.input_ids[:count] == packed.tokens[start : start + count]
        assert window.labels[:count] == packed.tokens[start + 1 : start + count + 1]
        assert window.input_ids[count:] == (PAD_INPUT_ID,) * (CONTEXT_LENGTH - count)
        assert window.labels[count:] == (IGNORE_INDEX,) * (CONTEXT_LENGTH - count)
    assert packed.accounting.candidate_target_count == len(packed.tokens) - 1
    assert packed.accounting.unique_scored_targets == len(packed.tokens) - 1
    assert packed.accounting.input_pad_slots == CONTEXT_LENGTH - 5


@pytest.mark.parametrize(
    ("payload_length", "expected_stream_length", "expected_window_counts"),
    [
        (CONTEXT_LENGTH - 1, CONTEXT_LENGTH, [CONTEXT_LENGTH - 1]),
        (CONTEXT_LENGTH, CONTEXT_LENGTH + 1, [CONTEXT_LENGTH]),
        (CONTEXT_LENGTH + 1, CONTEXT_LENGTH + 2, [CONTEXT_LENGTH, 1]),
    ],
)
def test_4096_4097_backing_window_boundary(
    payload_length: int,
    expected_stream_length: int,
    expected_window_counts: list[int],
) -> None:
    payload = tuple(index % (VOCAB_SIZE - 3) for index in range(payload_length))
    packed = train_split([synthetic_document("synthetic:boundary", payload)])

    assert len(packed.tokens) == expected_stream_length
    assert [
        window.valid_target_count for window in packed.windows
    ] == expected_window_counts
    assert packed.accounting.candidate_target_count == expected_stream_length - 1
    assert packed.accounting.unique_scored_targets == expected_stream_length - 1
    assert tuple(
        target for window in packed.windows for target in window.owned_target_indices
    ) == tuple(range(1, expected_stream_length))
    if payload_length == CONTEXT_LENGTH + 1:
        assert packed.windows[0].labels[-1] == packed.tokens[CONTEXT_LENGTH]
        assert packed.windows[1].input_ids[0] == packed.tokens[CONTEXT_LENGTH]
        assert packed.windows[1].labels[0] == packed.tokens[CONTEXT_LENGTH + 1]


def test_empty_only_and_one_token_documents_remain_in_accounting() -> None:
    one_empty = train_split([synthetic_document("synthetic:empty-one", [])])
    two_empty = train_split(
        [
            synthetic_document("synthetic:empty-a", []),
            synthetic_document("synthetic:empty-b", []),
        ]
    )
    one_token = train_split([synthetic_document("synthetic:one-token", [31])])

    assert one_empty.tokens == (EOS_ID,)
    assert one_empty.windows == ()
    assert one_empty.accounting.document_count == 1
    assert one_empty.accounting.empty_document_count == 1
    assert one_empty.accounting.candidate_target_count == 0
    assert one_empty.accounting.window_count == 0
    assert two_empty.tokens == (EOS_ID, EOS_ID)
    assert two_empty.accounting.empty_document_count == 2
    assert two_empty.accounting.unique_scored_targets == 1
    assert two_empty.accounting.input_pad_slots == CONTEXT_LENGTH - 1
    assert one_token.tokens == (31, EOS_ID)
    assert one_token.accounting.payload_tokens == 1
    assert one_token.accounting.eos_tokens == 1
    assert one_token.accounting.unique_scored_targets == 1


def test_long_document_continuation_keeps_document_relative_window_offsets() -> None:
    payload = tuple(index % 20_000 for index in range(8200))
    packed = train_split([synthetic_document("synthetic:continuing", payload)])

    assert [window.valid_target_count for window in packed.windows] == [4096, 4096, 8]
    provenance = [window.source_spans[0] for window in packed.windows]
    assert [
        (span.document_token_start, span.document_token_end) for span in provenance
    ] == [(0, 4097), (4096, 8193), (8192, 8200)]
    assert [span.eos_stream_offset for span in provenance] == [None, None, 8200]
    assert packed.document_spans[0].eos_offset == 8200


def test_safe_padding_and_ignored_labels_never_use_pad_minus_one() -> None:
    packed = train_split([synthetic_document("synthetic:short", [41])])
    window = packed.windows[0]

    assert PAD_INPUT_ID == EOS_ID
    assert PAD_INPUT_ID != -1
    assert all(0 <= token_id < VOCAB_SIZE for token_id in window.input_ids)
    assert window.labels[0] == EOS_ID
    assert set(window.labels[1:]) == {IGNORE_INDEX}
    assert len(tuple(window.scored_pairs())) == 1
    assert window.scored_target_count == 1
    assert packed.accounting.ignored_label_slots == CONTEXT_LENGTH - 1
    with pytest.raises(ValueError, match="safe_input_padding_id"):
        PackingPolicy(safe_input_padding_id=-1)


def test_bos_and_eos_scoring_remain_explicit_provisional_policy_choices() -> None:
    documents = [synthetic_document("synthetic:policy", [51])]
    no_eos_score = train_split(
        documents,
        PackingPolicy(insert_bos_at_split_start=True, score_eos_targets=False),
    )

    assert no_eos_score.tokens == (1, 51, EOS_ID)
    assert no_eos_score.bos_offset == 0
    assert no_eos_score.document_spans[0].stream_start == 1
    assert no_eos_score.document_spans[0].eos_offset == 2
    assert no_eos_score.windows[0].labels[:2] == (51, IGNORE_INDEX)
    assert no_eos_score.accounting.candidate_target_count == 2
    assert no_eos_score.accounting.unique_scored_targets == 1
    assert no_eos_score.accounting.eos_masked_label_count == 1
    assert no_eos_score.accounting.first_stream_token_unscored


@pytest.mark.parametrize("bad_token", [-1, VOCAB_SIZE, 1.5, True, "7"])
def test_document_rejects_invalid_payload_token_ids(bad_token: object) -> None:
    with pytest.raises(ValueError, match="token ID"):
        synthetic_document("synthetic:invalid-token", [bad_token])  # type: ignore[list-item]


def test_rejects_unknown_splits_duplicate_ids_and_group_contamination() -> None:
    with pytest.raises(ValueError, match="unknown split"):
        pack_documents_by_split(
            [synthetic_document("synthetic:unknown", [1], split="holdout")]
        )
    with pytest.raises(ValueError, match="duplicate document_id"):
        pack_documents_by_split(
            [
                synthetic_document("synthetic:duplicate", [1]),
                synthetic_document("synthetic:duplicate", [2]),
            ]
        )
    with pytest.raises(ValueError, match="multiple splits"):
        pack_documents_by_split(
            [
                synthetic_document(
                    "synthetic:group-train", [1], group_id="synthetic:g"
                ),
                synthetic_document(
                    "synthetic:group-valid",
                    [2],
                    split="validation",
                    group_id="synthetic:g",
                ),
            ]
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("document_id", "", "document_id"),
        ("group_id", "synthetic:group\ninvalid", "group_id"),
        ("source", " ", "source"),
        ("subset", "synthetic\x00subset", "subset"),
        ("content_hash", "not-a-hash", "content_hash"),
    ],
)
def test_rejects_malformed_document_provenance(
    field: str, value: str, message: str
) -> None:
    fields = {
        "document_id": "synthetic:provenance",
        "group_id": "synthetic:group",
        "split": "train",
        "source": "synthetic-source",
        "subset": "synthetic-subset",
        "content_hash": "a" * 64,
        "token_ids": (4,),
    }
    fields[field] = value
    with pytest.raises(ValueError, match=message):
        Document(**fields)  # type: ignore[arg-type]


def test_canonical_reproduction_and_identity_changes_for_document_order_or_split() -> (
    None
):
    alpha = synthetic_document("synthetic:alpha", [1, 2])
    beta = synthetic_document("synthetic:beta", [3])
    first = train_split([alpha, beta])
    reproduced = train_split([alpha, beta])
    changed_document = train_split(
        [synthetic_document("synthetic:alpha", [1, 9]), beta]
    )
    reordered = train_split([beta, alpha])
    moved = pack_documents_by_split(
        [synthetic_document("synthetic:alpha", [1, 2], split="validation"), beta]
    )["validation"]

    assert first == reproduced
    assert first.stream_digest == reproduced.stream_digest
    assert first.ordered_span_digest == reproduced.ordered_span_digest
    assert first.identity_digest == reproduced.identity_digest
    assert first.identity_digest != changed_document.identity_digest
    assert first.identity_digest != reordered.identity_digest
    assert first.identity_digest != moved.identity_digest
    assert first.ordered_span_digest != reordered.ordered_span_digest
    assert first.split == "train"
    assert moved.split == "validation"


def test_validator_rejects_unmatched_accounting_totals() -> None:
    packed = train_split([synthetic_document("synthetic:accounting", [5, 6])])
    corrupted_accounting = replace(
        packed.accounting,
        payload_tokens=packed.accounting.payload_tokens + 1,
    )

    with pytest.raises(ValueError, match="accounting totals"):
        validate_packed_split(replace(packed, accounting=corrupted_accounting))


def test_reference_is_in_memory_and_attempts_no_file_writes(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def deny_write(*_args, **_kwargs):
        raise AssertionError("the in-memory reference attempted filesystem access")

    monkeypatch.setattr(builtins, "open", deny_write)
    monkeypatch.setattr(os, "open", deny_write)
    monkeypatch.setattr(Path, "open", deny_write)
    monkeypatch.setattr(Path, "write_text", deny_write)
    monkeypatch.setattr(Path, "write_bytes", deny_write)

    packed = train_split([synthetic_document("synthetic:memory", [9, 10])])

    assert packed.tokens == (9, 10, EOS_ID)
    assert tuple(tmp_path.iterdir()) == ()
