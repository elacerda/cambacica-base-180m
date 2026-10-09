"""A small inverted-anchor matcher for benchmark exposure evidence.

This is independent of the D2c whole-document MinHash/Jaccard implementation.
Corpus-wide counts and match evidence spill to SQLite; Python memory holds the
pinned benchmark index and one corpus document at a time.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import unicodedata
from typing import Any, Iterable, Iterator


MATCHER_VERSION = "c1-bd2-token-anchor-v2"
NORMALIZATION_VERSION = "nfc-casefold-unicode-word-offsets-v1"
_ANCHOR_SEPARATOR = "\x1f"


@dataclass(frozen=True)
class Token:
    """One normalized word token and its source character offsets."""

    text: str
    start: int
    end: int


def normalize_match_text(text: str) -> str:
    """Apply the matcher-only Unicode normalization contract."""
    return unicodedata.normalize("NFC", text).casefold()


def iter_tokens_with_offsets(text: str) -> Iterator[Token]:
    """Yield Unicode word tokens with source character offsets in text order.

    Combining marks remain attached to a preceding letter or number. Offsets
    are Python Unicode character offsets into the exact input string.
    """
    start: int | None = None
    previous_was_word = False
    for index, char in enumerate(text):
        category = unicodedata.category(char)
        is_letter_or_number = category[0] in {"L", "N"}
        is_mark = category[0] == "M" and previous_was_word
        if is_letter_or_number or is_mark:
            if start is None:
                start = index
            previous_was_word = True
            continue
        if start is not None:
            raw = text[start:index]
            normalized = unicodedata.normalize("NFC", raw).casefold()
            if normalized:
                yield Token(normalized, start, index)
        start = None
        previous_was_word = False
    if start is not None:
        raw = text[start:]
        normalized = unicodedata.normalize("NFC", raw).casefold()
        if normalized:
            yield Token(normalized, start, len(text))


def tokenize_with_offsets(text: str) -> list[Token]:
    """Return all normalized tokens for bounded benchmark fields and tests."""
    return list(iter_tokens_with_offsets(text))


def _iter_token_chunks(
    text: str, chunk_tokens: int, overlap_tokens: int
) -> Iterator[tuple[int, list[Token]]]:
    """Yield overlapping token windows while retaining at most one window."""
    overlap_tokens = min(overlap_tokens, chunk_tokens - 1)
    window: list[Token] = []
    window_start = 0
    for token in iter_tokens_with_offsets(text):
        window.append(token)
        if len(window) == chunk_tokens:
            yield window_start, window
            window_start += len(window) - overlap_tokens
            window = window[-overlap_tokens:]
    if window:
        yield window_start, window


@dataclass(frozen=True)
class MatchField:
    """A benchmark field and its field-level identity."""

    benchmark_name: str
    example_id: str
    source_row_id: str
    source_file_sha256: str
    source_category: str | None
    field_id: str
    field_role: str
    original_text: str
    matchable: bool = True


@dataclass(frozen=True)
class CorpusDocument:
    """One immutable corpus occurrence supplied to a matcher run."""

    doc_id: str
    text: str
    source_shard: str = "<memory>"
    source: str | None = None
    source_row_ordinal: int | None = None
    input_manifest_sha256: str | None = None


@dataclass(frozen=True)
class CandidatePolicy:
    """Versioned candidate thresholds; values remain provisional until BD3."""

    policy_version: str = "c1-bd2-policy-candidate-v1"
    normalization_version: str = NORMALIZATION_VERSION
    matcher_version: str = MATCHER_VERSION
    anchor_ngram_tokens: int = 13
    minimum_contiguous_tokens: int = 50
    distinctive_coverage: float = 0.80
    minimum_rare_anchors: int = 2
    max_benchmark_item_df: int = 2
    max_ngram_df: int = 100
    minimum_question_tokens: int = 6
    max_exact_question_document_frequency: int = 100
    question_answer_max_token_gap: int = 128
    exact_seed_tokens: int = 8
    document_chunk_tokens: int = 8192
    commit_every_documents: int = 256

    def __post_init__(self) -> None:
        if self.anchor_ngram_tokens < 2:
            raise ValueError("anchor_ngram_tokens must be at least 2")
        if self.minimum_contiguous_tokens < 1:
            raise ValueError("minimum_contiguous_tokens must be positive")
        if not 0.0 < self.distinctive_coverage <= 1.0:
            raise ValueError("distinctive_coverage must be in (0, 1]")
        if (
            min(
                self.minimum_rare_anchors,
                self.max_benchmark_item_df,
                self.max_ngram_df,
                self.minimum_question_tokens,
                self.question_answer_max_token_gap,
                self.exact_seed_tokens,
                self.document_chunk_tokens,
                self.commit_every_documents,
            )
            < 1
        ):
            raise ValueError("integer policy limits must be positive")
        if self.max_exact_question_document_frequency < 1:
            raise ValueError("exact question document-frequency limit must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: Path | str) -> CandidatePolicy:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class MatchResult:
    """One deterministic, review-only benchmark exposure candidate."""

    doc_id: str
    source_shard: str
    source: str | None
    source_row_ordinal: int | None
    input_manifest_sha256: str | None
    document_text_sha256: str
    benchmark_name: str
    example_id: str
    source_row_id: str
    source_file_sha256: str
    source_category: str | None
    field_id: str
    field_role: str
    decision_rule: str
    exact_match: bool
    matched_tokens: int
    contiguous_tokens: int
    distinctive_anchor_count: int
    distinctive_token_coverage: float
    matched_anchor_document_frequency_min: int | None
    matched_anchor_document_frequency_max: int | None
    corpus_token_start: int
    corpus_token_end: int
    corpus_char_start: int
    corpus_char_end: int
    benchmark_token_start: int
    benchmark_token_end: int
    benchmark_char_start: int
    benchmark_char_end: int


@dataclass(frozen=True)
class AnchorEvidence:
    """One retained rare-anchor match with offsets in both source fields."""

    doc_id: str
    benchmark_name: str
    example_id: str
    source_row_id: str
    source_file_sha256: str
    source_category: str | None
    field_id: str
    field_role: str
    anchor_sha256: str
    anchor_text: str
    anchor_document_frequency: int
    corpus_token_start: int
    corpus_token_end: int
    corpus_char_start: int
    corpus_char_end: int
    benchmark_token_start: int
    benchmark_token_end: int
    benchmark_char_start: int
    benchmark_char_end: int


@dataclass(frozen=True)
class ShardAccounting:
    source_shard: str
    document_count: int
    record_id_sha256: str


@dataclass(frozen=True)
class ScanAccounting:
    documents_seen: int
    candidate_fields: int
    shard_accounting: tuple[ShardAccounting, ...]


def _anchor_digest(tokens: tuple[str, ...]) -> str:
    raw = _ANCHOR_SEPARATOR.join(tokens).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class BenchmarkMatcher:
    """Build a compact benchmark index and run bounded, read-only scans."""

    _MATCH_ROLES = frozenset(
        {"complete_item", "context", "passage", "question", "question_plus_answer"}
    )
    # Complete items are retained in the exact-prefix index, while long
    # passage/question spans supply approximate anchors. Indexing anchors for
    # composite items would duplicate every option and passage n-gram.
    _ANCHOR_ROLES = frozenset(
        {"context", "passage", "question", "question_plus_answer"}
    )
    _INDEX_ROLES = _MATCH_ROLES | {"correct_answer", "target_word"}

    def __init__(
        self, fields: Iterable[MatchField], policy: CandidatePolicy | None = None
    ) -> None:
        self.policy = policy or CandidatePolicy()
        self.fields: dict[str, MatchField] = {}
        self.field_tokens: dict[str, list[Token]] = {}
        self.field_token_values: dict[str, tuple[str, ...]] = {}
        # The in-memory lookup key uses Python's tuple hash for speed. Each
        # bucket retains stable anchor IDs and postings; document windows are
        # compared with representative benchmark tokens to guard collisions.
        self._anchor_lookup: dict[int, dict[str, list[tuple[str, int]]]] = {}
        self._anchor_item_df: dict[str, int] = {}
        self._question_item_df: dict[tuple[str, ...], int] = {}
        self._exact_seed_lookup: dict[tuple[str, ...], list[str]] = {}
        self._short_exact_seed_lookup: dict[str, dict[tuple[str, ...], list[str]]] = {}
        self._field_distinctive_positions: dict[str, set[int]] = {}
        self._question_answer_links: dict[
            str, tuple[str, tuple[str, ...], str | None]
        ] = {}
        self._build_index(fields)
        max_field_tokens = max(
            (len(values) for values in self.field_token_values.values()), default=0
        )
        max_question_answer_span = max(
            (
                len(self.field_token_values[field_id])
                + self.policy.question_answer_max_token_gap
                + len(answer_values)
                for field_id, (
                    _answer_id,
                    answer_values,
                    _combined_id,
                ) in self._question_answer_links.items()
            ),
            default=0,
        )
        self._document_chunk_overlap_tokens = max(
            self.policy.anchor_ngram_tokens - 1,
            max_field_tokens - 1,
            max_question_answer_span,
        )

    def _build_index(self, fields: Iterable[MatchField]) -> None:
        anchor_fields: dict[str, set[str]] = {}
        question_fields: dict[tuple[str, ...], set[str]] = {}
        for field in fields:
            # Options and other non-matchable fields cannot provide evidence.
            # Keep only answer fields needed to test question-plus-answer
            # co-occurrence; this keeps the pinned benchmark index compact.
            if field.field_role not in self._INDEX_ROLES:
                continue
            if field.field_id in self.fields:
                raise ValueError(
                    f"Duplicate benchmark field identity: {field.field_id}"
                )
            self.fields[field.field_id] = field
            self.field_tokens[field.field_id] = tokenize_with_offsets(
                field.original_text
            )
            self.field_token_values[field.field_id] = tuple(
                item.text for item in self.field_tokens[field.field_id]
            )
            if not field.matchable or field.field_role not in self._MATCH_ROLES:
                continue
            token_values = self.field_token_values[field.field_id]
            if (
                field.field_role == "question"
                and len(token_values) < self.policy.minimum_question_tokens
            ):
                continue
            if field.field_role == "question":
                question_fields.setdefault(token_values, set()).add(field.example_id)
            seed_length = min(self.policy.exact_seed_tokens, len(token_values))
            if seed_length:
                seed = token_values[:seed_length]
                if seed_length == self.policy.exact_seed_tokens:
                    self._exact_seed_lookup.setdefault(seed, []).append(field.field_id)
                else:
                    self._short_exact_seed_lookup.setdefault(seed[0], {}).setdefault(
                        seed, []
                    ).append(field.field_id)
            if field.field_role not in self._ANCHOR_ROLES:
                continue
            ngram_size = self.policy.anchor_ngram_tokens
            if len(token_values) < ngram_size:
                continue
            for position in range(len(token_values) - ngram_size + 1):
                gram = token_values[position : position + ngram_size]
                digest = _anchor_digest(gram)
                anchor_fields.setdefault(digest, set()).add(field.example_id)
                bucket = self._anchor_lookup.setdefault(hash(gram), {})
                bucket.setdefault(digest, []).append((field.field_id, position))
        self._anchor_item_df = {
            digest: len(example_ids) for digest, example_ids in anchor_fields.items()
        }
        self._question_item_df = {
            tokens: len(example_ids) for tokens, example_ids in question_fields.items()
        }
        ngram_size = self.policy.anchor_ngram_tokens
        for field_id, token_values in self.field_token_values.items():
            field = self.fields[field_id]
            if not field.matchable or field.field_role not in self._ANCHOR_ROLES:
                continue
            if len(token_values) < ngram_size:
                continue
            distinctive_positions: set[int] = set()
            for position in range(len(token_values) - ngram_size + 1):
                digest = _anchor_digest(token_values[position : position + ngram_size])
                if self._anchor_item_df[digest] <= self.policy.max_benchmark_item_df:
                    distinctive_positions.update(range(position, position + ngram_size))
            self._field_distinctive_positions[field_id] = distinctive_positions
        for field_ids in self._exact_seed_lookup.values():
            field_ids.sort()
        for seed_map in self._short_exact_seed_lookup.values():
            for field_ids in seed_map.values():
                field_ids.sort()
        for bucket in self._anchor_lookup.values():
            for positions in bucket.values():
                positions.sort()
        fields_by_example: dict[str, dict[str, MatchField]] = {}
        for field in self.fields.values():
            fields_by_example.setdefault(field.example_id, {})[field.field_role] = field
        for field in self.fields.values():
            if field.field_role != "question":
                continue
            row_fields = fields_by_example[field.example_id]
            answer_field = row_fields.get("correct_answer") or row_fields.get(
                "target_word"
            )
            if answer_field is None:
                continue
            answer_tokens = tuple(
                item.text for item in tokenize_with_offsets(answer_field.original_text)
            )
            if not answer_tokens:
                continue
            question_plus_answer = row_fields.get("question_plus_answer")
            self._question_answer_links[field.field_id] = (
                answer_field.field_id,
                answer_tokens,
                question_plus_answer.field_id if question_plus_answer else None,
            )

    def start_run(
        self,
        scratch_dir: Path | str | None = None,
        keep_scratch: bool = False,
        profile: bool = False,
    ) -> MatcherRun:
        return MatcherRun(
            self,
            scratch_dir=scratch_dir,
            keep_scratch=keep_scratch,
            profile=profile,
        )


class MatcherRun:
    """One spill-backed corpus pass and deterministic result iterator."""

    def __init__(
        self,
        matcher: BenchmarkMatcher,
        scratch_dir: Path | str | None = None,
        keep_scratch: bool = False,
        profile: bool = False,
    ) -> None:
        self.matcher = matcher
        self.policy = matcher.policy
        self.keep_scratch = keep_scratch
        self.profile = profile
        self._profile_timings = {
            "sqlite_accounting_seconds": 0.0,
            "tokenization_seconds": 0.0,
            "anchor_lookup_and_evidence_write_seconds": 0.0,
            "exact_lookup_and_evidence_write_seconds": 0.0,
            "sqlite_frequency_update_seconds": 0.0,
            "sqlite_commit_seconds": 0.0,
            "finalization_seconds": 0.0,
        }
        if scratch_dir is None:
            scratch = Path(tempfile.gettempdir())
        else:
            scratch = Path(scratch_dir).expanduser()
            scratch.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix="cambacica-bd2-match-", suffix=".sqlite3", dir=scratch
        )
        os.close(descriptor)
        self.db_path = Path(name)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA cache_size=-16384")
        self.connection.executescript(
            """
            CREATE TABLE seen_docs (
                doc_id TEXT PRIMARY KEY
            ) WITHOUT ROWID;
            CREATE TABLE anchor_frequency (
                anchor TEXT PRIMARY KEY,
                document_frequency INTEGER NOT NULL,
                frequent INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE anchor_evidence (
                anchor TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                field_id TEXT NOT NULL,
                benchmark_start INTEGER NOT NULL,
                corpus_start INTEGER NOT NULL,
                corpus_char_start INTEGER NOT NULL,
                corpus_char_end INTEGER NOT NULL,
                PRIMARY KEY (
                    anchor, doc_id, field_id, benchmark_start, corpus_start
                )
            ) WITHOUT ROWID;
            CREATE INDEX anchor_evidence_doc_field
                ON anchor_evidence(doc_id, field_id, benchmark_start, corpus_start);
            CREATE TABLE pending_anchor_evidence (
                anchor TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                field_id TEXT NOT NULL,
                benchmark_start INTEGER NOT NULL,
                corpus_start INTEGER NOT NULL,
                corpus_char_start INTEGER NOT NULL,
                corpus_char_end INTEGER NOT NULL,
                PRIMARY KEY (
                    anchor, doc_id, field_id, benchmark_start, corpus_start
                )
            ) WITHOUT ROWID;
            CREATE TABLE exact_evidence (
                doc_id TEXT NOT NULL,
                field_id TEXT NOT NULL,
                corpus_start INTEGER NOT NULL,
                corpus_end INTEGER NOT NULL,
                corpus_char_start INTEGER NOT NULL,
                corpus_char_end INTEGER NOT NULL,
                PRIMARY KEY(doc_id, field_id, corpus_start, corpus_end)
            ) WITHOUT ROWID;
            CREATE TABLE doc_metadata (
                doc_id TEXT PRIMARY KEY,
                source_shard TEXT NOT NULL,
                source TEXT,
                source_row_ordinal INTEGER,
                input_manifest_sha256 TEXT,
                document_text_sha256 TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE final_hits (
                sort_doc TEXT NOT NULL,
                sort_example TEXT NOT NULL,
                sort_field TEXT NOT NULL,
                sort_rule TEXT NOT NULL,
                start_token INTEGER NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (
                    sort_doc, sort_example, sort_field, sort_rule, start_token
                )
            ) WITHOUT ROWID;
            """
        )
        self.connection.commit()
        self._documents_seen = 0
        self._normalized_words_seen = 0
        self._last_document_tokens = 0
        self._candidate_fields = 0
        self._largest_document_tokens = 0
        self._shards: dict[str, tuple[int, Any]] = {}
        self._finished = False
        self._closed = False

    def __enter__(self) -> MatcherRun:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def add_document(self, document: CorpusDocument) -> None:
        """Index one row using a bounded token window and SQLite spill."""
        if self._finished or self._closed:
            raise RuntimeError("Matcher run is already finished or closed")
        if not document.doc_id:
            raise ValueError("Corpus document requires a stable record ID")
        if not isinstance(document.text, str):
            raise TypeError("Corpus document text must be a string")
        accounting_started = time.perf_counter() if self.profile else 0.0
        try:
            self.connection.execute(
                "INSERT INTO seen_docs(doc_id) VALUES (?)", (document.doc_id,)
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Duplicate corpus record ID: {document.doc_id}") from exc
        self._documents_seen += 1
        count, digest = self._shards.get(document.source_shard, (0, hashlib.sha256()))
        id_bytes = document.doc_id.encode("utf-8")
        digest.update(len(id_bytes).to_bytes(8, "big"))
        digest.update(id_bytes)
        self._shards[document.source_shard] = (count + 1, digest)
        if self.profile:
            self._profile_timings["sqlite_accounting_seconds"] += (
                time.perf_counter() - accounting_started
            )

        # Pending evidence is local to this document until its distinct anchor
        # frequencies are known. If an anchor crosses the configured cap here,
        # evidence from earlier documents and this document are both discarded.
        found: set[str] = set()
        ngram_size = self.policy.anchor_ngram_tokens
        chunk_tokens = max(
            self.policy.document_chunk_tokens,
            self.matcher._document_chunk_overlap_tokens + 1,
        )
        rare_evidence_written = False
        document_tokens = 0
        token_chunks = iter(
            _iter_token_chunks(
                document.text,
                chunk_tokens,
                self.matcher._document_chunk_overlap_tokens,
            )
        )
        while True:
            phase_started = time.perf_counter() if self.profile else 0.0
            try:
                global_start, tokens = next(token_chunks)
            except StopIteration:
                break
            if self.profile:
                self._profile_timings["tokenization_seconds"] += (
                    time.perf_counter() - phase_started
                )
            token_values = tuple(token.text for token in tokens)
            self._largest_document_tokens = max(
                self._largest_document_tokens, global_start + len(tokens)
            )
            document_tokens = max(document_tokens, global_start + len(tokens))

            phase_started = time.perf_counter() if self.profile else 0.0
            for position in range(max(0, len(token_values) - ngram_size + 1)):
                gram = token_values[position : position + ngram_size]
                bucket = self.matcher._anchor_lookup.get(hash(gram))
                if bucket is None:
                    continue
                char_start = tokens[position].start
                char_end = tokens[position + ngram_size - 1].end
                for anchor, field_positions in bucket.items():
                    if (
                        self.matcher._anchor_item_df[anchor]
                        > self.policy.max_benchmark_item_df
                    ):
                        continue
                    representative_field, representative_position = field_positions[0]
                    if (
                        self.matcher.field_token_values[representative_field][
                            representative_position : representative_position
                            + ngram_size
                        ]
                        != gram
                    ):
                        continue
                    if anchor not in found:
                        frequency = self.connection.execute(
                            "SELECT frequent FROM anchor_frequency WHERE anchor=?",
                            (anchor,),
                        ).fetchone()
                        if frequency is not None and frequency[0]:
                            continue
                        found.add(anchor)
                    matched_field_positions = [
                        (field_id, benchmark_start)
                        for field_id, benchmark_start in field_positions
                        if self.matcher.field_token_values[field_id][
                            benchmark_start : benchmark_start + ngram_size
                        ]
                        == gram
                    ]
                    self.connection.executemany(
                        """INSERT OR IGNORE INTO pending_anchor_evidence
                           (anchor, doc_id, field_id, benchmark_start, corpus_start,
                            corpus_char_start, corpus_char_end)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        [
                            (
                                anchor,
                                document.doc_id,
                                field_id,
                                benchmark_start,
                                global_start + position,
                                char_start,
                                char_end,
                            )
                            for field_id, benchmark_start in matched_field_positions
                        ],
                    )
            if self.profile:
                self._profile_timings["anchor_lookup_and_evidence_write_seconds"] += (
                    time.perf_counter() - phase_started
                )

            # The overlap covers every complete benchmark field and any allowed
            # question-answer gap, including matches that cross a chunk boundary.
            phase_started = time.perf_counter() if self.profile else 0.0
            for position in range(len(token_values)):
                remaining_tokens = len(token_values) - position
                if remaining_tokens >= self.policy.exact_seed_tokens:
                    seed = token_values[
                        position : position + self.policy.exact_seed_tokens
                    ]
                    candidate_ids = self.matcher._exact_seed_lookup.get(seed, ())
                    for field_id in candidate_ids:
                        field = self.matcher.fields[field_id]
                        field_values = self.matcher.field_token_values[field_id]
                        if len(field_values) > remaining_tokens:
                            continue
                        if (
                            token_values[position : position + len(field_values)]
                            != field_values
                        ):
                            continue
                        end = position + len(field_values)
                        global_position = global_start + position
                        self.connection.execute(
                            """INSERT OR IGNORE INTO exact_evidence
                               (doc_id, field_id, corpus_start, corpus_end,
                                corpus_char_start, corpus_char_end)
                               VALUES (?, ?, ?, ?, ?, ?)""",
                            (
                                document.doc_id,
                                field_id,
                                global_position,
                                global_position + len(field_values),
                                tokens[position].start,
                                tokens[end - 1].end,
                            ),
                        )
                        if field.field_role == "question":
                            self._record_nearby_answer(
                                document.doc_id,
                                field_id,
                                position,
                                end,
                                tokens,
                                token_values,
                                global_start,
                            )

                short_seed_map = self.matcher._short_exact_seed_lookup.get(
                    token_values[position]
                )
                if not short_seed_map:
                    continue
                for short_seed, candidate_ids in short_seed_map.items():
                    seed_length = len(short_seed)
                    if seed_length > remaining_tokens:
                        continue
                    if token_values[position : position + seed_length] != short_seed:
                        continue
                    for field_id in candidate_ids:
                        field = self.matcher.fields[field_id]
                        field_values = self.matcher.field_token_values[field_id]
                        if len(field_values) > remaining_tokens:
                            continue
                        if (
                            token_values[position : position + len(field_values)]
                            != field_values
                        ):
                            continue
                        end = position + len(field_values)
                        global_position = global_start + position
                        self.connection.execute(
                            """INSERT OR IGNORE INTO exact_evidence
                               (doc_id, field_id, corpus_start, corpus_end,
                                corpus_char_start, corpus_char_end)
                               VALUES (?, ?, ?, ?, ?, ?)""",
                            (
                                document.doc_id,
                                field_id,
                                global_position,
                                global_position + len(field_values),
                                tokens[position].start,
                                tokens[end - 1].end,
                            ),
                        )
                        if field.field_role == "question":
                            self._record_nearby_answer(
                                document.doc_id,
                                field_id,
                                position,
                                end,
                                tokens,
                                token_values,
                                global_start,
                            )
            if self.profile:
                self._profile_timings["exact_lookup_and_evidence_write_seconds"] += (
                    time.perf_counter() - phase_started
                )

        self._last_document_tokens = document_tokens
        self._normalized_words_seen += document_tokens

        phase_started = time.perf_counter() if self.profile else 0.0
        for anchor in sorted(found):
            row = self.connection.execute(
                "SELECT document_frequency, frequent FROM anchor_frequency WHERE anchor=?",
                (anchor,),
            ).fetchone()
            if row is None:
                self.connection.execute(
                    "INSERT INTO anchor_frequency VALUES (?, 1, 0)", (anchor,)
                )
                frequent = False
            else:
                document_frequency, already_frequent = row
                document_frequency += 1
                became_frequent = (
                    not already_frequent
                    and document_frequency > self.policy.max_ngram_df
                )
                frequent = bool(already_frequent or became_frequent)
                self.connection.execute(
                    "UPDATE anchor_frequency SET document_frequency=?, frequent=? WHERE anchor=?",
                    (document_frequency, int(frequent), anchor),
                )
                if became_frequent:
                    self.connection.execute(
                        "DELETE FROM anchor_evidence WHERE anchor=?", (anchor,)
                    )
            if not frequent:
                cursor = self.connection.execute(
                    """INSERT OR IGNORE INTO anchor_evidence
                       SELECT anchor, doc_id, field_id, benchmark_start, corpus_start,
                              corpus_char_start, corpus_char_end
                       FROM pending_anchor_evidence
                       WHERE anchor=? AND doc_id=?""",
                    (anchor, document.doc_id),
                )
                rare_evidence_written = rare_evidence_written or cursor.rowcount > 0
            self.connection.execute(
                "DELETE FROM pending_anchor_evidence WHERE anchor=? AND doc_id=?",
                (anchor, document.doc_id),
            )
        if self.profile:
            self._profile_timings["sqlite_frequency_update_seconds"] += (
                time.perf_counter() - phase_started
            )

        phase_started = time.perf_counter() if self.profile else 0.0
        if (
            rare_evidence_written
            or self.connection.execute(
                "SELECT 1 FROM exact_evidence WHERE doc_id=? LIMIT 1",
                (document.doc_id,),
            ).fetchone()
        ):
            self.connection.execute(
                """INSERT INTO doc_metadata
                   (doc_id, source_shard, source, source_row_ordinal,
                    input_manifest_sha256, document_text_sha256)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    document.doc_id,
                    document.source_shard,
                    document.source,
                    document.source_row_ordinal,
                    document.input_manifest_sha256,
                    _text_digest(document.text),
                ),
            )
        if self.profile:
            self._profile_timings["sqlite_accounting_seconds"] += (
                time.perf_counter() - phase_started
            )
        if self._documents_seen % self.policy.commit_every_documents == 0:
            phase_started = time.perf_counter() if self.profile else 0.0
            self.connection.commit()
            if self.profile:
                self._profile_timings["sqlite_commit_seconds"] += (
                    time.perf_counter() - phase_started
                )

    def _record_nearby_answer(
        self,
        doc_id: str,
        question_field_id: str,
        question_start: int,
        question_end: int,
        corpus_tokens: list[Token],
        token_values: tuple[str, ...],
        global_token_start: int,
    ) -> None:
        """Record a question plus answer co-occurrence without answer-only hits."""
        relation = self.matcher._question_answer_links.get(question_field_id)
        if relation is None:
            return
        _answer_field_id, answer_values, combined_field_id = relation
        if not combined_field_id:
            return
        gap = self.policy.question_answer_max_token_gap
        search_start = max(0, question_start - gap - len(answer_values))
        search_end = min(len(token_values), question_end + gap + len(answer_values))
        answer_span: tuple[int, int] | None = None
        for position in range(
            search_start, max(search_start, search_end - len(answer_values) + 1)
        ):
            if token_values[position] != answer_values[0]:
                continue
            end = position + len(answer_values)
            if end > search_end or token_values[position:end] != answer_values:
                continue
            token_gap = max(0, max(question_start - end, position - question_end))
            if token_gap <= gap:
                answer_span = (position, end)
                break
        if answer_span is None:
            return
        answer_start, answer_end = answer_span
        combined_start = min(question_start, answer_start)
        combined_end = max(question_end, answer_end)
        self.connection.execute(
            """INSERT OR IGNORE INTO exact_evidence
               (doc_id, field_id, corpus_start, corpus_end,
                corpus_char_start, corpus_char_end)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                doc_id,
                combined_field_id,
                global_token_start + combined_start,
                global_token_start + combined_end,
                corpus_tokens[combined_start].start,
                corpus_tokens[combined_end - 1].end,
            ),
        )

    def scan(self, documents: Iterable[CorpusDocument]) -> ScanAccounting:
        """Consume the supplied iterator exactly once and finalize candidate rows."""
        if self._documents_seen or self._finished:
            raise RuntimeError("scan() requires a fresh matcher run")
        try:
            for document in documents:
                self.add_document(document)
            return self.finish()
        except Exception:
            self.connection.rollback()
            raise

    def finish(self) -> ScanAccounting:
        """Finalize results after ``add_document`` or a bounded manual pass."""
        if self._finished or self._closed:
            raise RuntimeError("Matcher run is already finished or closed")
        try:
            phase_started = time.perf_counter() if self.profile else 0.0
            self.connection.commit()
            self._finalize_hits()
            self.connection.commit()
            if self.profile:
                self._profile_timings["finalization_seconds"] += (
                    time.perf_counter() - phase_started
                )
            self._finished = True
            accounting = tuple(
                ShardAccounting(shard, count, digest.hexdigest())
                for shard, (count, digest) in sorted(self._shards.items())
            )
            return ScanAccounting(
                documents_seen=self._documents_seen,
                candidate_fields=self._candidate_fields,
                shard_accounting=accounting,
            )
        except Exception:
            self.connection.rollback()
            raise

    def _insert_result(self, result: MatchResult) -> None:
        payload = json.dumps(
            asdict(result), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        self.connection.execute(
            """INSERT OR IGNORE INTO final_hits
               (sort_doc, sort_example, sort_field, sort_rule, start_token, payload)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                result.doc_id,
                result.example_id,
                result.field_id,
                result.decision_rule,
                result.corpus_token_start,
                payload,
            ),
        )

    def _base_result(
        self,
        *,
        doc_id: str,
        field_id: str,
        decision_rule: str,
        exact: bool,
        matched_tokens: int,
        contiguous_tokens: int,
        anchor_count: int,
        coverage: float,
        anchor_df_min: int | None = None,
        anchor_df_max: int | None = None,
        corpus_start: int,
        corpus_end: int,
        corpus_char_start: int,
        corpus_char_end: int,
        benchmark_start: int,
        benchmark_end: int,
    ) -> MatchResult:
        field = self.matcher.fields[field_id]
        field_tokens = self.matcher.field_tokens[field_id]
        metadata = self.connection.execute(
            """SELECT source_shard, source, source_row_ordinal,
                      input_manifest_sha256, document_text_sha256
               FROM doc_metadata WHERE doc_id=?""",
            (doc_id,),
        ).fetchone()
        if metadata is None:
            raise ValueError(f"Missing source locator for matching row {doc_id}")
        benchmark_start = max(0, min(benchmark_start, len(field_tokens)))
        benchmark_end = max(benchmark_start, min(benchmark_end, len(field_tokens)))
        if benchmark_start < benchmark_end:
            benchmark_char_start = field_tokens[benchmark_start].start
            benchmark_char_end = field_tokens[benchmark_end - 1].end
        else:
            benchmark_char_start = 0
            benchmark_char_end = 0
        return MatchResult(
            doc_id=doc_id,
            source_shard=metadata[0],
            source=metadata[1],
            source_row_ordinal=metadata[2],
            input_manifest_sha256=metadata[3],
            document_text_sha256=metadata[4],
            benchmark_name=field.benchmark_name,
            example_id=field.example_id,
            source_row_id=field.source_row_id,
            source_file_sha256=field.source_file_sha256,
            source_category=field.source_category,
            field_id=field.field_id,
            field_role=field.field_role,
            decision_rule=decision_rule,
            exact_match=exact,
            matched_tokens=matched_tokens,
            contiguous_tokens=contiguous_tokens,
            distinctive_anchor_count=anchor_count,
            distinctive_token_coverage=coverage,
            matched_anchor_document_frequency_min=anchor_df_min,
            matched_anchor_document_frequency_max=anchor_df_max,
            corpus_token_start=corpus_start,
            corpus_token_end=corpus_end,
            corpus_char_start=corpus_char_start,
            corpus_char_end=corpus_char_end,
            benchmark_token_start=benchmark_start,
            benchmark_token_end=benchmark_end,
            benchmark_char_start=benchmark_char_start,
            benchmark_char_end=benchmark_char_end,
        )

    def _best_coherent_alignment(
        self, doc_id: str, field_id: str
    ) -> tuple[int, int, float, int | None, int | None, tuple[int, ...] | None]:
        """Return the strongest local, order-consistent anchor alignment.

        The offset band is relative to the benchmark field length. Its width
        follows the existing coverage threshold: a field requiring 80% coverage
        can tolerate up to 25% offset drift across the field. This is not an
        absolute document-distance limit; it prevents distant corpus fragments
        from being combined into one coverage score.
        """
        field_values = self.matcher.field_token_values[field_id]
        distinctive = self.matcher._field_distinctive_positions.get(field_id, set())
        if not field_values or not distinctive:
            return 0, 0, 0.0, None, None, None

        drift_limit = math.ceil(
            (1.0 / self.policy.distinctive_coverage - 1.0) * len(field_values)
        )
        evidence_cursor = self.connection.execute(
            """SELECT e.anchor, e.benchmark_start, e.corpus_start,
                      e.corpus_char_start, e.corpus_char_end,
                      f.document_frequency, e.corpus_start-e.benchmark_start
               FROM anchor_evidence e
               JOIN anchor_frequency f ON f.anchor=e.anchor
               WHERE e.doc_id=? AND e.field_id=? AND f.frequent=0
               ORDER BY e.corpus_start-e.benchmark_start,
                        e.benchmark_start, e.corpus_start, e.anchor""",
            (doc_id, field_id),
        )

        window: deque[tuple[int, int, int, int, int, str, int]] = deque()
        position_counts = [0] * len(field_values)
        anchor_counts: Counter[str] = Counter()
        frequency_counts: Counter[int] = Counter()
        covered_positions = 0
        best_score = (0, 0)
        best_delta: int | None = None
        ngram_size = self.policy.anchor_ngram_tokens

        def update_positions(start: int, amount: int) -> None:
            nonlocal covered_positions
            for token_position in range(start, start + ngram_size):
                if token_position not in distinctive:
                    continue
                previous = position_counts[token_position]
                position_counts[token_position] += amount
                if previous == 0 and amount > 0:
                    covered_positions += 1
                elif previous == 1 and amount < 0:
                    covered_positions -= 1

        for row in evidence_cursor:
            anchor, benchmark_start, corpus_start = row[:3]
            char_start, char_end, document_frequency, delta = row[3:]
            item = (
                int(delta),
                int(benchmark_start),
                int(corpus_start),
                int(char_start),
                int(char_end),
                str(anchor),
                int(document_frequency),
            )
            while window and item[0] - window[0][0] > drift_limit:
                expired = window.popleft()
                update_positions(expired[1], -1)
                anchor_counts[expired[5]] -= 1
                if not anchor_counts[expired[5]]:
                    del anchor_counts[expired[5]]
                frequency_counts[expired[6]] -= 1
                if not frequency_counts[expired[6]]:
                    del frequency_counts[expired[6]]

            window.append(item)
            update_positions(item[1], 1)
            anchor_counts[item[5]] += 1
            frequency_counts[item[6]] += 1
            score = (covered_positions, len(anchor_counts))
            if score > best_score:
                best_score = score
                best_delta = item[0]

        if best_delta is None:
            return 0, 0, 0.0, None, None, None

        # Reconstruct a monotone benchmark-to-corpus chain inside the winning
        # offset band. At each benchmark position, the earliest corpus
        # occurrence after the last selected anchor leaves the most room for
        # later anchors and avoids counting reordered fragments.
        alignment_cursor = self.connection.execute(
            """SELECT e.anchor, e.benchmark_start, e.corpus_start,
                      e.corpus_char_start, e.corpus_char_end,
                      f.document_frequency
               FROM anchor_evidence e
               JOIN anchor_frequency f ON f.anchor=e.anchor
               WHERE e.doc_id=? AND e.field_id=? AND f.frequent=0
                 AND e.corpus_start-e.benchmark_start BETWEEN ? AND ?
               ORDER BY e.benchmark_start, e.corpus_start, e.anchor""",
            (
                doc_id,
                field_id,
                best_delta - drift_limit,
                best_delta,
            ),
        )
        aligned_positions: set[int] = set()
        aligned_anchors: set[str] = set()
        aligned_frequencies: list[int] = []
        representative: tuple[int, ...] | None = None
        previous_corpus_start = -1
        current_benchmark_start: int | None = None
        selected_for_benchmark: tuple[Any, ...] | None = None

        def select_current() -> None:
            nonlocal previous_corpus_start, representative
            if selected_for_benchmark is None:
                return
            anchor, benchmark_start, corpus_start = selected_for_benchmark[:3]
            if int(corpus_start) <= previous_corpus_start:
                return
            previous_corpus_start = int(corpus_start)
            if representative is None:
                representative = tuple(
                    int(value) for value in selected_for_benchmark[1:5]
                )
            aligned_anchors.add(str(anchor))
            aligned_frequencies.append(int(selected_for_benchmark[5]))
            aligned_positions.update(
                token_position
                for token_position in range(
                    int(benchmark_start), int(benchmark_start) + ngram_size
                )
                if token_position in distinctive
            )

        for row in alignment_cursor:
            anchor, benchmark_start, corpus_start = row[:3]
            if (
                current_benchmark_start is not None
                and int(benchmark_start) != current_benchmark_start
            ):
                select_current()
                selected_for_benchmark = None
            current_benchmark_start = int(benchmark_start)
            if int(corpus_start) > previous_corpus_start and (
                selected_for_benchmark is None
                or int(corpus_start) < int(selected_for_benchmark[2])
            ):
                selected_for_benchmark = row
        select_current()

        coverage = len(aligned_positions) / len(distinctive)
        df_min = min(aligned_frequencies) if aligned_frequencies else None
        df_max = max(aligned_frequencies) if aligned_frequencies else None
        return (
            len(aligned_positions),
            len(aligned_anchors),
            coverage,
            df_min,
            df_max,
            representative,
        )

    def _finalize_hits(self) -> None:
        policy = self.policy
        exact_rows = self.connection.execute(
            """SELECT doc_id, field_id, corpus_start, corpus_end,
                      corpus_char_start, corpus_char_end
               FROM exact_evidence ORDER BY doc_id, field_id, corpus_start"""
        )
        question_df: dict[str, int] = {}
        for row in self.connection.execute(
            """SELECT field_id, count(DISTINCT doc_id)
               FROM exact_evidence WHERE field_id IN
                 (SELECT field_id FROM exact_evidence)
               GROUP BY field_id"""
        ):
            question_df[row[0]] = int(row[1])
        for (
            doc_id,
            field_id,
            corpus_start,
            corpus_end,
            char_start,
            char_end,
        ) in exact_rows:
            field = self.matcher.fields[field_id]
            field_tokens = self.matcher.field_tokens[field_id]
            role = field.field_role
            if role == "complete_item":
                rule = "exact_complete_item"
            elif role in {"context", "passage"}:
                rule = "exact_context" if role == "context" else "exact_passage"
            elif role == "question_plus_answer":
                rule = "exact_question_plus_answer"
            elif role == "question":
                if len(field_tokens) < policy.minimum_question_tokens:
                    continue
                field_values = self.matcher.field_token_values[field_id]
                if (
                    self.matcher._question_item_df[field_values]
                    > policy.max_benchmark_item_df
                ):
                    continue
                if (
                    question_df.get(field_id, 0)
                    > policy.max_exact_question_document_frequency
                ):
                    continue
                rule = "exact_distinctive_question"
            else:
                continue
            self._insert_result(
                self._base_result(
                    doc_id=doc_id,
                    field_id=field_id,
                    decision_rule=rule,
                    exact=True,
                    matched_tokens=len(field_tokens),
                    contiguous_tokens=len(field_tokens),
                    anchor_count=0,
                    coverage=1.0,
                    corpus_start=corpus_start,
                    corpus_end=corpus_end,
                    corpus_char_start=char_start,
                    corpus_char_end=char_end,
                    benchmark_start=0,
                    benchmark_end=len(field_tokens),
                )
            )

        current_doc: str | None = None
        current_field: str | None = None
        anchor_ids: set[str] = set()
        active_span: tuple[int, int, int, int, int, int] | None = None
        best_span: tuple[int, int, int, int, int, int] | None = None

        def flush_active() -> None:
            nonlocal active_span, best_span
            if active_span is not None and (
                best_span is None
                or active_span[1] - active_span[0] > best_span[1] - best_span[0]
            ):
                best_span = active_span
            active_span = None

        def add_evidence(row: tuple[Any, ...]) -> None:
            nonlocal active_span
            anchor, b_start, d_start, char_start, char_end, _document_frequency = row
            anchor_ids.add(anchor)
            interval_end = b_start + policy.anchor_ngram_tokens
            span = (
                b_start,
                interval_end,
                d_start,
                d_start + policy.anchor_ngram_tokens,
                char_start,
                char_end,
            )
            if active_span is None:
                active_span = span
                return
            ab_start, ab_end, ad_start, ad_end, ac_start, ac_end = active_span
            same_alignment = d_start - b_start == ad_start - ab_start
            if same_alignment and b_start <= ab_end and d_start <= ad_end:
                active_span = (
                    min(ab_start, b_start),
                    max(ab_end, interval_end),
                    min(ad_start, d_start),
                    max(ad_end, d_start + policy.anchor_ngram_tokens),
                    min(ac_start, char_start),
                    max(ac_end, char_end),
                )
            else:
                flush_active()
                active_span = span

        def finish_group(group_doc: str, group_field: str) -> None:
            nonlocal anchor_ids
            nonlocal active_span, best_span
            if not anchor_ids:
                return
            flush_active()
            (
                aligned_tokens,
                anchor_count,
                coverage,
                anchor_df_min,
                anchor_df_max,
                aligned_representative,
            ) = self._best_coherent_alignment(group_doc, group_field)
            rule: str | None = None
            if (
                best_span
                and best_span[1] - best_span[0] >= policy.minimum_contiguous_tokens
            ):
                rule = "contiguous_token_run"
                selected_span = best_span
            elif (
                coverage >= policy.distinctive_coverage
                and anchor_count >= policy.minimum_rare_anchors
                and aligned_representative is not None
            ):
                rule = "distinctive_anchor_coverage"
                selected_span = (
                    aligned_representative[0],
                    aligned_representative[0] + policy.anchor_ngram_tokens,
                    aligned_representative[1],
                    aligned_representative[1] + policy.anchor_ngram_tokens,
                    aligned_representative[2],
                    aligned_representative[3],
                )
            else:
                selected_span = None
            if rule is not None and selected_span is not None:
                b_start, b_end, d_start, d_end, char_start, char_end = selected_span
                self._insert_result(
                    self._base_result(
                        doc_id=group_doc,
                        field_id=group_field,
                        decision_rule=rule,
                        exact=False,
                        matched_tokens=(
                            best_span[1] - best_span[0]
                            if rule == "contiguous_token_run" and best_span
                            else aligned_tokens
                        ),
                        contiguous_tokens=(best_span[1] - best_span[0])
                        if best_span
                        else 0,
                        anchor_count=anchor_count,
                        coverage=coverage,
                        anchor_df_min=anchor_df_min,
                        anchor_df_max=anchor_df_max,
                        corpus_start=d_start,
                        corpus_end=d_end,
                        corpus_char_start=char_start,
                        corpus_char_end=char_end,
                        benchmark_start=b_start,
                        benchmark_end=b_end,
                    )
                )
            anchor_ids = set()
            active_span = None
            best_span = None

        evidence_cursor = self.connection.execute(
            """SELECT e.doc_id, e.field_id, e.anchor, e.benchmark_start,
                      e.corpus_start, e.corpus_char_start, e.corpus_char_end,
                      f.document_frequency
               FROM anchor_evidence e
               JOIN anchor_frequency f ON f.anchor=e.anchor
               WHERE f.frequent=0
               ORDER BY e.doc_id, e.field_id, e.benchmark_start, e.corpus_start"""
        )
        for row in evidence_cursor:
            doc_id, field_id = row[0], row[1]
            if current_doc is not None and (doc_id, field_id) != (
                current_doc,
                current_field,
            ):
                finish_group(current_doc, current_field or "")
            current_doc, current_field = doc_id, field_id
            add_evidence(
                (
                    row[2],
                    int(row[3]),
                    int(row[4]),
                    int(row[5]),
                    int(row[6]),
                    int(row[7]),
                )
            )
        if current_doc is not None and current_field is not None:
            finish_group(current_doc, current_field)
        self._candidate_fields = int(
            self.connection.execute("SELECT count(*) FROM final_hits").fetchone()[0]
        )

    def iter_results(self) -> Iterator[MatchResult]:
        """Yield complete candidate evidence in deterministic sort order."""
        if not self._finished:
            raise RuntimeError("Matcher results are unavailable before scan completion")
        cursor = self.connection.execute(
            "SELECT payload FROM final_hits ORDER BY sort_doc, sort_example, sort_field, sort_rule, start_token"
        )
        for (payload,) in cursor:
            yield MatchResult(**json.loads(payload))

    def iter_anchor_evidence(self) -> Iterator[AnchorEvidence]:
        """Stream every retained anchor span supporting approximate hits."""
        if not self._finished:
            raise RuntimeError("Matcher evidence is unavailable before scan completion")
        cursor = self.connection.execute(
            """SELECT e.doc_id, e.field_id, e.anchor, e.benchmark_start,
                      e.corpus_start, e.corpus_char_start, e.corpus_char_end,
                      f.document_frequency
               FROM anchor_evidence e
               JOIN anchor_frequency f ON f.anchor=e.anchor
               WHERE f.frequent=0 AND EXISTS (
                   SELECT 1 FROM final_hits h
                   WHERE h.sort_doc=e.doc_id AND h.sort_field=e.field_id
                     AND h.sort_rule IN
                       ('contiguous_token_run', 'distinctive_anchor_coverage')
               )
               ORDER BY e.doc_id, e.field_id, e.benchmark_start,
                        e.corpus_start, e.anchor"""
        )
        ngram_size = self.policy.anchor_ngram_tokens
        for row in cursor:
            doc_id, field_id, anchor, benchmark_start, corpus_start = row[:5]
            corpus_char_start, corpus_char_end, document_frequency = row[5:]
            field = self.matcher.fields[field_id]
            field_tokens = self.matcher.field_tokens[field_id]
            benchmark_start = int(benchmark_start)
            benchmark_end = benchmark_start + ngram_size
            yield AnchorEvidence(
                doc_id=doc_id,
                benchmark_name=field.benchmark_name,
                example_id=field.example_id,
                source_row_id=field.source_row_id,
                source_file_sha256=field.source_file_sha256,
                source_category=field.source_category,
                field_id=field_id,
                field_role=field.field_role,
                anchor_sha256=anchor,
                anchor_text=" ".join(
                    self.matcher.field_token_values[field_id][
                        benchmark_start:benchmark_end
                    ]
                ),
                anchor_document_frequency=int(document_frequency),
                corpus_token_start=int(corpus_start),
                corpus_token_end=int(corpus_start) + ngram_size,
                corpus_char_start=int(corpus_char_start),
                corpus_char_end=int(corpus_char_end),
                benchmark_token_start=benchmark_start,
                benchmark_token_end=benchmark_end,
                benchmark_char_start=field_tokens[benchmark_start].start,
                benchmark_char_end=field_tokens[benchmark_end - 1].end,
            )

    @property
    def documents_seen(self) -> int:
        return self._documents_seen

    @property
    def largest_document_tokens(self) -> int:
        return self._largest_document_tokens

    @property
    def scratch_bytes(self) -> int:
        self.connection.commit()
        return self.db_path.stat().st_size if self.db_path.exists() else 0

    @property
    def normalized_words_seen(self) -> int:
        return self._normalized_words_seen

    @property
    def last_document_tokens(self) -> int:
        return self._last_document_tokens

    @property
    def profile_timings(self) -> dict[str, float]:
        return dict(self._profile_timings)

    @property
    def anchor_evidence_rows(self) -> int:
        return int(
            self.connection.execute("SELECT count(*) FROM anchor_evidence").fetchone()[
                0
            ]
        )

    def anchor_frequencies(self) -> dict[str, int]:
        """Return exact distinct-document counts for query anchors seen so far."""
        return {
            anchor: int(count)
            for anchor, count in self.connection.execute(
                "SELECT anchor, document_frequency FROM anchor_frequency ORDER BY anchor"
            )
        }

    def close(self) -> None:
        if self._closed:
            return
        self.connection.close()
        self._closed = True
        if not self.keep_scratch:
            self.db_path.unlink(missing_ok=True)


def fields_from_snapshot(path: Path | str) -> list[MatchField]:
    """Read field identities from a verified snapshot Parquet table."""
    import pyarrow.parquet as pq

    result: list[MatchField] = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        columns=[
            "benchmark_name",
            "example_id",
            "source_row_id",
            "source_file_sha256",
            "source_category",
            "field_id",
            "field_role",
            "matchable",
            "original_text",
        ],
        batch_size=256,
    ):
        for row in batch.to_pylist():
            if row["field_role"] not in BenchmarkMatcher._INDEX_ROLES:
                continue
            result.append(
                MatchField(
                    benchmark_name=row["benchmark_name"],
                    example_id=row["example_id"],
                    source_row_id=row["source_row_id"],
                    source_file_sha256=row["source_file_sha256"],
                    source_category=row["source_category"],
                    field_id=row["field_id"],
                    field_role=row["field_role"],
                    original_text=row["original_text"],
                    matchable=row["matchable"],
                )
            )
    return result
