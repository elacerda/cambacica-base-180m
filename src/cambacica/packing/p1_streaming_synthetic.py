"""C4-P1.3a: bounded, disk-indexed, synthetic-only P1 streaming reference.

No corpus readers, tokenizer, PyTorch, CUDA, NVML, model or network imports.
The input is an explicit iterable of artificial ``Document`` instances.
This is a CPU prototype, not a production dataset/sampler/checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from cambacica.packing.p1_reference import (
    CONTEXT_LENGTH,
    EOS_ID,
    IGNORE_INDEX,
    SCHEMA_VERSION as P1_REFERENCE_SCHEMA,
    SPLITS,
    VOCAB_SIZE,
    Document,
    TrainingWindow,
    WindowProvenance,
)

SCHEMA = "c4-p1.3a-synthetic-v1"
ORDER_DOMAIN = b"cambacica-c4-p1.3a-synthetic-O1-v1\0"
SHA_LEN = 64


def _canonical(obj: object) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _digest(obj: object) -> str:
    return hashlib.sha256(_canonical(obj)).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _order_key(
    *, view: str, split: str, seed: str, epoch: int, document_id: str
) -> str:
    # Fixture-only seed and schema: not a production sampling policy.
    effective_epoch = epoch if split == "train" else 0
    h = hashlib.sha256()
    h.update(ORDER_DOMAIN)
    h.update(
        _canonical(
            [
                view,
                split,
                effective_epoch,
                seed if split == "train" else "fixed-heldout",
                document_id,
            ]
        )
    )
    return h.hexdigest()


@dataclass(frozen=True)
class StreamIdentity:
    view: str
    seed: str
    epoch: int
    view_digest: str
    order_digest: str
    data_sha256: str
    index_sha256: str


@dataclass(frozen=True)
class _TokenEvent:
    token: int
    document_id: str
    stream_offset: int
    document_offset: int | None  # None = inserted EOS


class SyntheticDiskIndex:
    """Immutable on-disk O1 index of artificial pretokenized documents."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.db_path = self.path / "order.sqlite"
        self.data_path = self.path / "payload.u16"
        self.manifest_path = self.path / "manifest.json"
        if not all(
            p.is_file() for p in (self.db_path, self.data_path, self.manifest_path)
        ):
            raise ValueError("synthetic O1 index missing a required file")
        try:
            m = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("invalid index manifest") from exc
        if m.get("schema") != SCHEMA or set(m["split_lengths"]) != set(SPLITS):
            raise ValueError("incompatible synthetic index manifest")
        for key, path in (
            ("data_sha256", self.data_path),
            ("index_sha256", self.db_path),
        ):
            if _sha_file(path) != m.get(key):
                raise ValueError(f"synthetic index integrity failure: {key}")
        self.manifest = m
        self.identity = StreamIdentity(
            **{k: m[k] for k in StreamIdentity.__dataclass_fields__}
        )

    @classmethod
    def build(
        cls,
        documents: Iterable[Document],
        path: Path,
        *,
        view: str,
        seed: str,
        epoch: int,
    ) -> SyntheticDiskIndex:
        """Spool an explicit artificial iterable to local disk, never all in RAM."""
        if (
            not isinstance(view, str)
            or not view
            or not isinstance(seed, str)
            or not seed
        ):
            raise ValueError("fixture view and seed must be nonempty strings")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("fixture epoch must be a nonnegative integer")
        path = Path(path)
        if path.exists():
            raise FileExistsError("synthetic index output must not exist")
        if not path.parent.is_dir():
            raise FileNotFoundError("synthetic index parent must exist")
        staging = Path(tempfile.mkdtemp(prefix=".c4-p13a-", dir=path.parent))
        conn = None
        try:
            conn = sqlite3.connect(staging / "order.sqlite")
            conn.execute(
                "CREATE TABLE docs (doc_id TEXT PRIMARY KEY, group_id TEXT NOT NULL, "
                "split TEXT NOT NULL, source TEXT NOT NULL, subset TEXT NOT NULL, "
                "content_hash TEXT NOT NULL, payload_hash TEXT NOT NULL, "
                "byte_pos INTEGER NOT NULL, length INTEGER NOT NULL, sort_key TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE groups (group_id TEXT PRIMARY KEY, split TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE ordered (split TEXT NOT NULL, ordinal INTEGER NOT NULL, "
                "doc_id TEXT NOT NULL UNIQUE, stream_start INTEGER NOT NULL, "
                "stream_end INTEGER NOT NULL, PRIMARY KEY(split, ordinal))"
            )
            conn.execute(
                "CREATE INDEX ix_ordered_offset ON ordered (split, stream_end)"
            )
            offset_bytes = 0
            with (staging / "payload.u16").open("wb") as output:
                for doc in documents:
                    if not isinstance(doc, Document) or doc.split not in SPLITS:
                        raise ValueError(
                            "expected artificial Document with a known split"
                        )
                    row = conn.execute(
                        "SELECT split FROM groups WHERE group_id=?", (doc.group_id,)
                    ).fetchone()
                    if row and row[0] != doc.split:
                        raise ValueError("group spans multiple splits")
                    conn.execute(
                        "INSERT OR IGNORE INTO groups VALUES (?,?)",
                        (doc.group_id, doc.split),
                    )
                    payload_hasher = hashlib.sha256()
                    payload_hasher.update(b"[")
                    for i in range(0, len(doc.token_ids), 4096):
                        block = doc.token_ids[i : i + 4096]
                        if any(
                            type(t) is not int or t < 0 or t >= VOCAB_SIZE
                            for t in block
                        ):
                            raise ValueError("invalid artificial token ID")
                        output.write(struct.pack("<" + "H" * len(block), *block))
                        for j, token in enumerate(block):
                            if i + j:
                                payload_hasher.update(b",")
                            payload_hasher.update(str(token).encode("ascii"))
                    payload_hasher.update(b"]")
                    try:
                        conn.execute(
                            "INSERT INTO docs VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (
                                doc.document_id,
                                doc.group_id,
                                doc.split,
                                doc.source,
                                doc.subset,
                                doc.content_hash,
                                payload_hasher.hexdigest(),
                                offset_bytes,
                                len(doc.token_ids),
                                _order_key(
                                    view=view,
                                    split=doc.split,
                                    seed=seed,
                                    epoch=epoch,
                                    document_id=doc.document_id,
                                ),
                            ),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ValueError("duplicate document identity") from exc
                    offset_bytes += 2 * len(doc.token_ids)
                output.flush()
                os.fsync(output.fileno())
            # Canonical view checksum independent of arrival order and IO layout.
            view_h = hashlib.sha256()
            order_h = hashlib.sha256()
            for row in conn.execute(
                "SELECT doc_id,group_id,split,source,subset,content_hash,payload_hash,length "
                "FROM docs ORDER BY doc_id"
            ):
                view_h.update(_canonical(row) + b"\n")
            split_lengths: dict[str, int] = {}
            split_counts: dict[str, int] = {}
            for split in SPLITS:
                pos = 0
                ordinal = 0
                for key, doc_id, length in conn.execute(
                    "SELECT sort_key,doc_id,length FROM docs WHERE split=? ORDER BY sort_key,doc_id",
                    (split,),
                ):
                    conn.execute(
                        "INSERT INTO ordered VALUES (?,?,?,?,?)",
                        (
                            split,
                            ordinal,
                            doc_id,
                            pos,
                            pos + length + 1,
                        ),
                    )
                    order_h.update(_canonical([split, ordinal, key, doc_id]) + b"\n")
                    pos += length + 1
                    ordinal += 1
                split_lengths[split] = pos
                split_counts[split] = ordinal
            conn.commit()
            conn.close()
            conn = None
            db = staging / "order.sqlite"
            with db.open("rb") as f:
                os.fsync(f.fileno())
            manifest = {
                "schema": SCHEMA,
                "view": view,
                "seed": seed,
                "epoch": epoch,
                "view_digest": view_h.hexdigest(),
                "order_digest": order_h.hexdigest(),
                "data_sha256": _sha_file(staging / "payload.u16"),
                "index_sha256": _sha_file(db),
                "split_lengths": split_lengths,
                "split_counts": split_counts,
            }
            with (staging / "manifest.json").open("w", encoding="utf-8") as output:
                output.write(_canonical(manifest).decode("ascii") + "\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(staging, path)
            dirfd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
        finally:
            if conn is not None:
                conn.close()
            if staging.exists():
                shutil.rmtree(staging)
        return cls(path)

    def ordered_document_ids(self, split: str) -> Iterator[str]:
        if split not in SPLITS:
            raise ValueError("unknown split")
        with sqlite3.connect(self.db_path) as conn:
            for (doc_id,) in conn.execute(
                "SELECT doc_id FROM ordered WHERE split=? ORDER BY ordinal", (split,)
            ):
                yield doc_id

    def reference_digests(self, split: str) -> tuple[str, str]:
        """Bounded-memory stream and span SHA-256 matching original P1 JSON.

        The expensive full-stream checksum is only an offline synthetic oracle.
        It is never recomputed for individual windows or production input.
        """
        if split not in SPLITS:
            raise ValueError("unknown split")
        tokens_h = hashlib.sha256()
        tokens_h.update(
            b'{"schema_version":' + _canonical(P1_REFERENCE_SCHEMA) + b',"tokens":['
        )
        spans_h = hashlib.sha256()
        spans_h.update(b"[")
        token_first = True
        span_first = True
        with sqlite3.connect(self.db_path) as conn, self.data_path.open("rb") as data:
            rows = conn.execute(
                "SELECT docs.doc_id,docs.group_id,docs.split,docs.source,docs.subset,"
                "docs.content_hash,docs.payload_hash,docs.byte_pos,docs.length,"
                "ordered.stream_start "
                "FROM ordered JOIN docs ON docs.doc_id=ordered.doc_id "
                "WHERE ordered.split=? ORDER BY ordered.ordinal",
                (split,),
            )
            for (
                doc_id,
                group_id,
                split_name,
                source,
                subset,
                content_hash,
                payload_hash,
                byte_pos,
                length,
                start,
            ) in rows:
                for off in range(0, length, 4096):
                    take = min(4096, length - off)
                    data.seek(byte_pos + 2 * off)
                    raw = data.read(take * 2)
                    if len(raw) != 2 * take:
                        raise ValueError("truncated synthetic payload")
                    for token in struct.unpack("<" + "H" * take, raw):
                        if not token_first:
                            tokens_h.update(b",")
                        tokens_h.update(str(token).encode("ascii"))
                        token_first = False
                if not token_first:
                    tokens_h.update(b",")
                tokens_h.update(str(EOS_ID).encode("ascii"))
                token_first = False
                span = {
                    "content_hash": content_hash,
                    "document_id": doc_id,
                    "eos_offset": start + length,
                    "group_id": group_id,
                    "payload_digest": payload_hash,
                    "source": source,
                    "split": split_name,
                    "stream_end": start + length,
                    "stream_start": start,
                    "subset": subset,
                }
                if not span_first:
                    spans_h.update(b",")
                spans_h.update(_canonical(span))
                span_first = False
        tokens_h.update(b"]}")
        spans_h.update(b"]")
        return tokens_h.hexdigest(), spans_h.hexdigest()

    def tokens_from(
        self, split: str, start: int, limit: int, *, fragment_size: int = 97
    ) -> Iterator[_TokenEvent]:
        """Read only a bounded number of artificial token events from the index."""
        if split not in SPLITS or type(start) is not int or start < 0:
            raise ValueError("invalid split/offset")
        if not (type(limit) is int and 0 <= limit <= CONTEXT_LENGTH + 1):
            raise ValueError("invalid bounded token read limit")
        if not (
            type(fragment_size) is int and 1 <= fragment_size <= CONTEXT_LENGTH + 1
        ):
            raise ValueError("invalid fragment size")
        end = self.manifest["split_lengths"][split]
        if start > end:
            raise ValueError("offset beyond split")
        remaining = limit
        with sqlite3.connect(self.db_path) as conn, self.data_path.open("rb") as data:
            for doc_id, doc_start, length, byte_pos in conn.execute(
                "SELECT docs.doc_id,ordered.stream_start,docs.length,docs.byte_pos "
                "FROM ordered JOIN docs ON docs.doc_id=ordered.doc_id "
                "WHERE ordered.split=? AND ordered.stream_end>? ORDER BY ordered.ordinal",
                (split, start),
            ):
                if not remaining:
                    break
                relative = max(0, start - doc_start)
                if relative < length:
                    position = relative
                    while position < length and remaining:
                        n = min(fragment_size, length - position, remaining)
                        data.seek(byte_pos + 2 * position)
                        raw = data.read(n * 2)
                        if len(raw) != 2 * n:
                            raise ValueError("truncated synthetic token payload")
                        block = struct.unpack("<" + "H" * n, raw)
                        for i, token in enumerate(block):
                            yield _TokenEvent(
                                token, doc_id, doc_start + position + i, position + i
                            )
                        remaining -= n
                        position += n
                if remaining and start <= doc_start + length:
                    yield _TokenEvent(EOS_ID, doc_id, doc_start + length, None)
                    remaining -= 1


def _provenance(events: list[_TokenEvent]) -> tuple[WindowProvenance, ...]:
    # O(context) time, independent of the number of documents in the index.
    by_doc: dict[str, dict[str, int | None]] = {}
    for event in events:
        found = by_doc.setdefault(
            event.document_id,
            {
                "payload_stream_start": None,
                "payload_stream_end": None,
                "document_token_start": None,
                "document_token_end": None,
                "eos_stream_offset": None,
            },
        )
        if event.document_offset is None:
            found["eos_stream_offset"] = event.stream_offset
        else:
            if found["payload_stream_start"] is None:
                found["payload_stream_start"] = event.stream_offset
                found["document_token_start"] = event.document_offset
            found["payload_stream_end"] = event.stream_offset + 1
            found["document_token_end"] = event.document_offset + 1
    return tuple(
        WindowProvenance(document_id=doc, **fields) for doc, fields in by_doc.items()
    )


def _window_digest(win: TrainingWindow) -> str:
    return _digest(
        {
            "window_index": win.window_index,
            "input_start": win.input_start,
            "input_ids": win.input_ids,
            "labels": win.labels,
            "provenance": [vars(x) for x in win.source_spans],
        }
    )


class SyntheticWindowConsumer:
    """One outstanding window at most, durable acknowledgments, replay on restart.

    This is a synthetic cursor only: it cannot commit external training effects.
    """

    def __init__(
        self,
        index: SyntheticDiskIndex,
        split: str,
        cursor_path: Path,
        *,
        fragment_size: int = 97,
    ):
        if split not in SPLITS:
            raise ValueError("unknown split")
        if (
            type(fragment_size) is not int
            or not 1 <= fragment_size <= CONTEXT_LENGTH + 1
        ):
            raise ValueError("invalid fragment size")
        self.index = index
        self.split = split
        self.cursor_path = Path(cursor_path)
        if not self.cursor_path.parent.is_dir():
            raise FileNotFoundError("cursor parent does not exist")
        self.fragment_size = fragment_size
        self.acked = -1
        self.chain = hashlib.sha256(b"C4-P1.3a-empty-ack-chain").hexdigest()
        self.last_digest: str | None = None
        self._pending: TrainingWindow | None = None
        if self.cursor_path.exists():
            self._load_snapshot()

    def _resume_fields(self, acked: int) -> dict[str, object]:
        total = self.index.manifest["split_lengths"][self.split]
        next_start = min((acked + 1) * CONTEXT_LENGTH, total)
        # The next input token is exactly the carry from the preceding
        # window. If no scored target remains, no new window is emitted.
        remaining = max(0, total - next_start - 1)
        carry = None
        if remaining:
            event = next(self.index.tokens_from(self.split, next_start, 1))
            carry = {
                "token": event.token,
                "document_id": event.document_id,
                "stream_offset": event.stream_offset,
                "document_token_offset": event.document_offset,
                "is_eos": event.document_offset is None,
            }
        return {
            "next_window_ordinal": acked + 1,
            "next_input_stream_offset": next_start,
            "next_carry": carry,
        }

    def _identity(self) -> dict[str, str | int]:
        return {
            "view_digest": self.index.identity.view_digest,
            "order_digest": self.index.identity.order_digest,
            "data_sha256": self.index.identity.data_sha256,
            "index_sha256": self.index.identity.index_sha256,
            "view": self.index.identity.view,
            "seed": self.index.identity.seed,
            "epoch": self.index.identity.epoch,
            "split": self.split,
            "schema": SCHEMA,
        }

    def _make_window(self, ordinal: int) -> TrainingWindow | None:
        stream_len = self.index.manifest["split_lengths"][self.split]
        start = ordinal * CONTEXT_LENGTH
        valid = min(CONTEXT_LENGTH, max(0, stream_len - 1 - start))
        if not valid:
            return None
        events = list(
            self.index.tokens_from(
                self.split, start, valid + 1, fragment_size=self.fragment_size
            )
        )
        if len(events) != valid + 1 or [x.stream_offset for x in events] != list(
            range(start, start + valid + 1)
        ):
            raise ValueError(
                "missing, duplicated or out-of-order synthetic stream token"
            )
        pad = CONTEXT_LENGTH - valid
        return TrainingWindow(
            window_index=ordinal,
            input_start=start,
            backing_stream_start=start,
            backing_stream_end=start + valid + 1,
            valid_target_count=valid,
            input_ids=tuple(x.token for x in events[:-1]) + (EOS_ID,) * pad,
            labels=tuple(x.token for x in events[1:]) + (IGNORE_INDEX,) * pad,
            source_spans=_provenance(events),
        )

    def next_window(self) -> TrainingWindow | None:
        """Deliver one window; repeated calls until ack return the same window."""
        if self._pending is None:
            self._pending = self._make_window(self.acked + 1)
        return self._pending

    def acknowledge(self, window: TrainingWindow) -> None:
        """Atomically persist the contiguous consumer ack (not external effects)."""
        expected = self.next_window()
        if expected is None or expected != window:
            raise ValueError("ack requires the exact next contiguous delivered window")
        new_acked = self.acked + 1
        last_digest = _window_digest(window)
        chain = _digest([self.chain, new_acked, last_digest])
        payload = {
            "identity": self._identity(),
            "acked": new_acked,
            "chain": chain,
            "last_digest": last_digest,
            "committed_target_count": min(
                (new_acked + 1) * CONTEXT_LENGTH,
                max(self.index.manifest["split_lengths"][self.split] - 1, 0),
            ),
            "resume": self._resume_fields(new_acked),
        }
        document = {"payload": payload, "sha256": _digest(payload)}
        fd, temporary = tempfile.mkstemp(
            prefix=".c4-ack-", suffix=".json", dir=self.cursor_path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(_canonical(document).decode("ascii") + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.cursor_path)
            dirfd = os.open(self.cursor_path.parent, os.O_RDONLY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.acked = new_acked
        self.last_digest = last_digest
        self.chain = chain
        self._pending = None

    def _load_snapshot(self) -> None:
        try:
            snapshot = json.loads(self.cursor_path.read_text(encoding="utf-8"))
            payload = snapshot["payload"]
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid synthetic cursor snapshot") from exc
        if snapshot.get("sha256") != _digest(payload):
            raise ValueError("corrupted synthetic cursor snapshot checksum")
        if payload.get("identity") != self._identity():
            raise ValueError(
                "cursor view/order/split/seed/epoch/index identity mismatch"
            )
        acked = payload.get("acked")
        if type(acked) is not int or acked < 0:
            raise ValueError("cursor acknowledged ordinal malformed")
        if payload.get("committed_target_count") != min(
            (acked + 1) * CONTEXT_LENGTH,
            max(self.index.manifest["split_lengths"][self.split] - 1, 0),
        ):
            raise ValueError("cursor committed target count mismatch")
        if payload.get("resume") != self._resume_fields(acked):
            raise ValueError("cursor carry or stream offset mismatch")
        prior_window = self._make_window(acked)
        if prior_window is None or _window_digest(prior_window) != payload.get(
            "last_digest"
        ):
            raise ValueError("cursor last committed window digest mismatch")
        if (
            not isinstance(payload.get("chain"), str)
            or len(payload["chain"]) != SHA_LEN
        ):
            raise ValueError("cursor chain malformed")
        self.acked = acked
        self.last_digest = payload["last_digest"]
        self.chain = payload["chain"]
