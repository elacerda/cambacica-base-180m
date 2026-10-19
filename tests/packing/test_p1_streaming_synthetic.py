"""C4-P1.3a bounded synthetic CPU tests; no real corpus or GPU access."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from cambacica.packing.p1_reference import (
    CONTEXT_LENGTH,
    IGNORE_INDEX,
    Document,
    pack_documents_by_split,
)
from cambacica.packing.p1_streaming_synthetic import (
    SyntheticDiskIndex,
    SyntheticWindowConsumer,
)


def doc(
    name: str,
    tokens: tuple[int, ...] | list[int],
    split: str = "train",
    group: str | None = None,
) -> Document:
    t = tuple(tokens)
    return Document(
        document_id=f"artificial:{name}",
        group_id=group or f"artificial-group:{name}",
        split=split,
        source="artificial-source",
        subset="fixture",
        content_hash=hashlib.sha256(repr((name, t)).encode()).hexdigest(),
        token_ids=t,
    )


def make(tmp_path, docs, *, seed="fixture-1", epoch=0, name="index"):
    return SyntheticDiskIndex.build(
        iter(docs), tmp_path / name, view="fixture-view", seed=seed, epoch=epoch
    )


def collect(
    index, split, tmp_path, *, fragment_size=97, cursor="state.json", interrupt=None
):
    consumer = SyntheticWindowConsumer(
        index, split, tmp_path / cursor, fragment_size=fragment_size
    )
    windows = []
    while (window := consumer.next_window()) is not None:
        windows.append(window)
        if interrupt is not None and len(windows) == interrupt:
            return windows, consumer
        consumer.acknowledge(window)
    return windows, consumer


def test_reference_equivalence_and_splits(tmp_path):
    docs = [
        doc("a", [10, 11]),
        doc("e", []),
        doc("b", [12]),
        doc("v", [15], "validation"),
        doc("t", [], "test"),
    ]
    index = make(tmp_path, docs)
    for split in ("train", "validation", "test"):
        ordered = list(index.ordered_document_ids(split))
        by_id = {d.document_id: d for d in docs}
        baseline = pack_documents_by_split([by_id[x] for x in ordered])[split]
        observed, _ = collect(
            index, split, tmp_path, cursor=f"{split}.json", fragment_size=1
        )
        assert tuple(observed) == baseline.windows
        assert index.manifest["split_lengths"][split] == len(baseline.tokens)
        assert index.reference_digests(split) == (
            baseline.stream_digest,
            baseline.ordered_span_digest,
        )


@pytest.mark.parametrize("length", [0, 1, 4095, 4096, 4097, 8192, 8193, 8299])
def test_boundary_geometry_matches_reference(tmp_path, length):
    documents = [doc("long", [i % 203 for i in range(length)])]
    index = make(tmp_path, documents)
    baseline = pack_documents_by_split(documents)["train"]
    observed, _ = collect(index, "train", tmp_path, fragment_size=31)
    assert tuple(observed) == baseline.windows
    assert sum(w.valid_target_count for w in observed) == max(0, length)
    assert [
        i
        for w in observed
        for i in range(w.input_start + 1, w.input_start + w.valid_target_count + 1)
    ] == list(range(1, length + 1))


@pytest.mark.parametrize("fragment", [1, 2, 7, 97, 4096])
def test_fragment_boundaries_not_pack_boundaries(tmp_path, fragment):
    documents = [
        doc("x", [7] * 4000),
        doc("empty", []),
        doc("y", [9] * 4100),
        doc("z", [12] * 3),
    ]
    index = make(tmp_path, documents)
    baseline = pack_documents_by_split(
        [
            next(d for d in documents if d.document_id == name)
            for name in index.ordered_document_ids("train")
        ]
    )["train"]
    observed, _ = collect(index, "train", tmp_path, fragment_size=fragment)
    assert tuple(observed) == baseline.windows


def test_order_invariant_to_input_arrival_order(tmp_path):
    docs = [
        doc(f"doc-{n:03d}", [n + 5, n + 6], "validation" if n % 7 == 0 else "train")
        for n in range(50)
    ]
    first = make(tmp_path, docs, name="a")
    second = make(tmp_path, list(reversed(docs)), name="b")
    assert first.identity.view_digest == second.identity.view_digest
    assert first.identity.order_digest == second.identity.order_digest
    assert {
        s: list(first.ordered_document_ids(s)) for s in ("train", "validation", "test")
    } == {
        s: list(second.ordered_document_ids(s)) for s in ("train", "validation", "test")
    }
    for split in ("train", "validation", "test"):
        wa, _ = collect(first, split, tmp_path, cursor=f"a-{split}.json")
        wb, _ = collect(second, split, tmp_path, cursor=f"b-{split}.json")
        assert wa == wb


def test_heldout_independent_of_training_epoch_and_seed(tmp_path):
    docs = [doc(f"t{i}", [i + 5]) for i in range(16)] + [
        doc(f"v{i}", [i + 5], "validation") for i in range(6)
    ]
    a = make(tmp_path, docs, seed="seed-one", epoch=0, name="a")
    b = make(tmp_path, docs, seed="seed-two", epoch=9, name="b")
    assert list(a.ordered_document_ids("validation")) == list(
        b.ordered_document_ids("validation")
    )
    assert list(a.ordered_document_ids("train")) != list(
        b.ordered_document_ids("train")
    )


def test_acknowledged_cursor_replays_unacked_window(tmp_path):
    docs = [doc("big", [i % 101 + 10 for i in range(8900)])]
    index = make(tmp_path, docs)
    full, _ = collect(index, "train", tmp_path, cursor="uninterrupted.json")
    consumer = SyntheticWindowConsumer(index, "train", tmp_path / "resume.json")
    first = consumer.next_window()
    consumer.acknowledge(first)
    delivered_not_acked = consumer.next_window()
    assert consumer.next_window() == delivered_not_acked
    resumed = SyntheticWindowConsumer(index, "train", tmp_path / "resume.json")
    assert resumed.acked == 0
    assert resumed.next_window() == delivered_not_acked
    resumed.acknowledge(delivered_not_acked)
    suffix = []
    while (item := resumed.next_window()) is not None:
        suffix.append(item)
        resumed.acknowledge(item)
    assert [first, delivered_not_acked, *suffix] == full
    assert resumed.acked == len(full) - 1


def test_ack_fails_out_of_order_or_modified(tmp_path):
    index = make(tmp_path, [doc("some", [5, 6, 7])])
    consumer = SyntheticWindowConsumer(index, "train", tmp_path / "state.json")
    win = consumer.next_window()
    with pytest.raises(ValueError, match="exact next"):
        consumer.acknowledge(replace(win, window_index=1))
    with pytest.raises(ValueError, match="exact next"):
        consumer.acknowledge(replace(win, labels=(IGNORE_INDEX,) * CONTEXT_LENGTH))
    consumer.acknowledge(win)
    with pytest.raises(ValueError, match="exact next"):
        consumer.acknowledge(win)


def test_reject_checksum_corruption_and_identity_drift(tmp_path):
    index = make(tmp_path, [doc("data", [5, 6, 7])])
    state = tmp_path / "state.json"
    consumer = SyntheticWindowConsumer(index, "train", state)
    consumer.acknowledge(consumer.next_window())
    raw = json.loads(state.read_text())
    raw["payload"]["acked"] = 555
    state.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="checksum"):
        SyntheticWindowConsumer(index, "train", state)
    raw["payload"]["acked"] = 0
    raw["payload"]["identity"]["seed"] = "wrong"
    import cambacica.packing.p1_streaming_synthetic as mod

    raw["sha256"] = mod._digest(raw["payload"])
    state.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="identity"):
        SyntheticWindowConsumer(index, "train", state)


def test_build_rejects_split_group_collision_and_duplicate_doc(tmp_path):
    with pytest.raises(ValueError, match="multiple splits"):
        make(
            tmp_path,
            [doc("a", [1], "train", "shared"), doc("b", [1], "test", "shared")],
            name="groups",
        )
    assert not (tmp_path / "groups").exists()
    with pytest.raises(ValueError, match="duplicate"):
        make(tmp_path, [doc("a", [1]), doc("a", [1])], name="duplicate")


def test_input_token_corruption_detected(tmp_path):
    idx = make(tmp_path, [doc("a", [10])])
    with (idx.path / "payload.u16").open("ab") as output:
        output.write(b"x")
    with pytest.raises(ValueError, match="integrity"):
        SyntheticDiskIndex(idx.path)


def test_invalid_fragment_size_and_seed(tmp_path):
    with pytest.raises(ValueError, match="seed"):
        make(tmp_path, [doc("a", [1])], seed="")
    idx = make(tmp_path, [doc("a", [1])], name="valid")
    with pytest.raises(ValueError, match="fragment size"):
        SyntheticWindowConsumer(idx, "train", tmp_path / "ack", fragment_size=0)


def test_no_window_if_empty_split(tmp_path):
    idx = make(tmp_path, [])
    assert all(
        idx.manifest["split_lengths"][s] == 0 for s in ("train", "validation", "test")
    )
    for split in ("train", "validation", "test"):
        assert (
            SyntheticWindowConsumer(
                idx, split, tmp_path / f"{split}.json"
            ).next_window()
            is None
        )


def test_resume_carry_matches_reference_and_is_tamper_evident(tmp_path):
    idx = make(tmp_path, [doc("long", [5] * 4100)])
    cursor = tmp_path / "carry.json"
    consumer = SyntheticWindowConsumer(idx, "train", cursor)
    w = consumer.next_window()
    consumer.acknowledge(w)
    content = json.loads(cursor.read_text())
    assert content["payload"]["resume"]["next_input_stream_offset"] == 4096
    assert content["payload"]["resume"]["next_carry"]["token"] == 5
    assert content["payload"]["resume"]["next_window_ordinal"] == 1
    import cambacica.packing.p1_streaming_synthetic as mod

    content["payload"]["resume"]["next_carry"]["token"] = 6
    content["sha256"] = mod._digest(content["payload"])
    cursor.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="cursor carry"):
        SyntheticWindowConsumer(idx, "train", cursor)


def test_atomic_snapshot_write_failure_never_acknowledges(tmp_path, monkeypatch):
    idx = make(tmp_path, [doc("data", [5, 6, 7])])
    path = tmp_path / "ack.json"
    consumer = SyntheticWindowConsumer(idx, "train", path)
    win = consumer.next_window()
    import cambacica.packing.p1_streaming_synthetic as mod

    real_replace = mod.os.replace

    def refuse_replace(*args, **kwargs):
        raise OSError("synthetic failure before rename")

    monkeypatch.setattr(mod.os, "replace", refuse_replace)
    with pytest.raises(OSError, match="synthetic failure"):
        consumer.acknowledge(win)
    assert consumer.acked == -1
    assert not path.exists()
    assert consumer.next_window() == win
    monkeypatch.setattr(mod.os, "replace", real_replace)
    consumer.acknowledge(win)
    assert consumer.acked == 0
