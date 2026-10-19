"""C4-P1.3b: bounded, synthetic-only CPU scalability measurements.

This benchmark generates artificial token IDs, never opens production corpus
files, and never imports a tokenizer, torch, CUDA or NVML. Measurements are
host- and filesystem-specific, not training-throughput estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import tempfile
import time
from pathlib import Path
from typing import Iterator

from cambacica.packing.p1_reference import CONTEXT_LENGTH, VOCAB_SIZE, Document
from cambacica.packing.p1_streaming_synthetic import (
    SyntheticDiskIndex,
    SyntheticWindowConsumer,
)

SCHEMA = "c4-p1.3b-synthetic-scalability-v1"
MAX_DOCUMENTS = 50_000
MAX_DOCUMENT_TOKENS = 16_384
MAX_STREAM_TOKENS = 2_000_000


def _fixture_documents(count: int, tokens_per_document: int) -> Iterator[Document]:
    """Yield deterministic synthetic documents, never accumulating the view."""
    for ordinal in range(count):
        identifier = f"p13b-artificial:{ordinal:08d}"
        yield Document(
            document_id=identifier,
            group_id=f"p13b-artificial-group:{ordinal:08d}",
            split="train",
            source="p13b-synthetic-only",
            subset="synthetic-scalability",
            content_hash=hashlib.sha256(identifier.encode("ascii")).hexdigest(),
            token_ids=tuple(
                3 + ((ordinal * 131 + offset * 17) % (VOCAB_SIZE - 3))
                for offset in range(tokens_per_document)
            ),
        )


def _validate_parameters(
    documents: int, tokens_per_document: int, fragment_size: int, restart_after: int
) -> tuple[int, int]:
    if type(documents) is not int or not 1 <= documents <= MAX_DOCUMENTS:
        raise ValueError(f"documents must be an integer in [1, {MAX_DOCUMENTS}]")
    if (
        type(tokens_per_document) is not int
        or not 0 <= tokens_per_document <= MAX_DOCUMENT_TOKENS
    ):
        raise ValueError(
            f"tokens_per_document must be an integer in [0, {MAX_DOCUMENT_TOKENS}]"
        )
    if type(fragment_size) is not int or not 1 <= fragment_size <= CONTEXT_LENGTH + 1:
        raise ValueError("fragment_size must be an integer in [1, 4097]")
    total_tokens = documents * (tokens_per_document + 1)
    if total_tokens > MAX_STREAM_TOKENS:
        raise ValueError(f"synthetic stream exceeds {MAX_STREAM_TOKENS} tokens")
    window_count = (max(0, total_tokens - 1) + CONTEXT_LENGTH - 1) // CONTEXT_LENGTH
    if type(restart_after) is not int or restart_after < 0:
        raise ValueError("restart_after must be a nonnegative integer")
    if restart_after and not 0 < restart_after < window_count:
        raise ValueError("restart_after must precede an uncommitted window")
    return total_tokens, window_count


def benchmark_case(
    *,
    documents: int,
    tokens_per_document: int,
    fragment_size: int = 257,
    restart_after: int = 0,
    scratch_parent: Path | None = None,
) -> dict[str, object]:
    """Measure an isolated artificial index/stream/cursor lifecycle on CPU.

    A fresh process should run each case: ru_maxrss is a *process high-water*
    mark and cannot be reset within a process. This function performs no
    optimizer effects and simulates restart by reopening the consumer, not by
    killing the operating-system process.
    """
    total_tokens, expected_windows = _validate_parameters(
        documents, tokens_per_document, fragment_size, restart_after
    )
    if scratch_parent is not None:
        scratch_parent = Path(scratch_parent)
        if not scratch_parent.is_dir():
            raise ValueError("scratch_parent must already be an existing directory")
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="c4-p13b-synthetic-", dir=scratch_parent
    ) as temporary:
        base = Path(temporary)
        index_path = base / "index"
        build_start = time.perf_counter()
        index = SyntheticDiskIndex.build(
            _fixture_documents(documents, tokens_per_document),
            index_path,
            view="p13b-artificial-fixed-view-v1",
            seed="p13b-artificial-seed-not-for-production",
            epoch=0,
        )
        build_seconds = time.perf_counter() - build_start
        if index.manifest["split_counts"] != {
            "train": documents,
            "validation": 0,
            "test": 0,
        }:
            raise AssertionError("synthetic split counts changed")
        if index.manifest["split_lengths"]["train"] != total_tokens:
            raise AssertionError("synthetic stream size changed")

        index_bytes = (index_path / "order.sqlite").stat().st_size
        payload_bytes = (index_path / "payload.u16").stat().st_size
        manifest_bytes = (index_path / "manifest.json").stat().st_size
        if payload_bytes != 2 * documents * tokens_per_document:
            raise AssertionError("synthetic payload size changed")

        cursor_path = base / "cursor.json"
        consumer = SyntheticWindowConsumer(
            index, "train", cursor_path, fragment_size=fragment_size
        )
        read_seconds = 0.0
        ack_seconds = 0.0
        resume_seconds = 0.0
        windows = 0
        scored_targets = 0
        scored_eos = 0
        resume_checks = 0
        while True:
            read_start = time.perf_counter()
            window = consumer.next_window()
            read_seconds += time.perf_counter() - read_start
            if window is None:
                break
            if (
                window.window_index != windows
                or window.input_start != windows * CONTEXT_LENGTH
            ):
                raise AssertionError("noncontiguous synthetic window ordering")
            if window.valid_target_count != min(
                CONTEXT_LENGTH, total_tokens - 1 - windows * CONTEXT_LENGTH
            ):
                raise AssertionError("synthetic target count mismatch")
            scored_targets += window.valid_target_count
            scored_eos += sum(label == 2 for label in window.labels)

            ack_start = time.perf_counter()
            consumer.acknowledge(window)
            ack_seconds += time.perf_counter() - ack_start
            windows += 1
            if windows == restart_after:
                # Deliver unacknowledged work, then abandon in-memory state.
                before = consumer.next_window()
                if before is None:
                    raise AssertionError("no unacknowledged restart fixture")
                restart_start = time.perf_counter()
                consumer = SyntheticWindowConsumer(
                    index, "train", cursor_path, fragment_size=fragment_size
                )
                recovered = consumer.next_window()
                resume_seconds += time.perf_counter() - restart_start
                if recovered != before or consumer.acked != windows - 1:
                    raise AssertionError("synthetic checkpoint replay divergence")
                resume_checks += 1

        if windows != expected_windows or scored_targets != max(0, total_tokens - 1):
            raise AssertionError("synthetic window/target accounting mismatch")
        if scored_eos != documents - int(tokens_per_document == 0):
            # For empty documents the *first EOS* has no predecessor.
            raise AssertionError("synthetic EOS scoring mismatch")
        if consumer.acked != expected_windows - 1:
            raise AssertionError("cursor did not acknowledge all emitted windows")
        cursor_bytes = cursor_path.stat().st_size if cursor_path.exists() else 0
        chain = consumer.chain
        order_digest = index.identity.order_digest
        view_digest = index.identity.view_digest
        read_and_ack_seconds = read_seconds + ack_seconds
        total_seconds = time.perf_counter() - t0
        return {
            "schema": SCHEMA,
            "fixture_only": True,
            "documents": documents,
            "tokens_per_document": tokens_per_document,
            "fragment_size": fragment_size,
            "restart_after": restart_after,
            "stream_tokens": total_tokens,
            "scored_targets": scored_targets,
            "scored_eos": scored_eos,
            "windows": windows,
            "build_seconds": round(build_seconds, 6),
            "window_read_seconds": round(read_seconds, 6),
            "ack_fsync_seconds": round(ack_seconds, 6),
            "read_and_ack_seconds": round(read_and_ack_seconds, 6),
            "resume_verify_seconds": round(resume_seconds, 6),
            "resume_checks": resume_checks,
            "total_seconds": round(total_seconds, 6),
            "scored_targets_per_second_read_ack": round(
                scored_targets / read_and_ack_seconds, 3
            )
            if read_and_ack_seconds
            else None,
            "sqlite_bytes": index_bytes,
            "payload_bytes": payload_bytes,
            "manifest_bytes": manifest_bytes,
            "cursor_bytes": cursor_bytes,
            "total_index_bytes": index_bytes + payload_bytes + manifest_bytes,
            "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_rss_unit": "KiB (Linux ru_maxrss)",
            "order_digest": order_digest,
            "view_digest": view_digest,
            "ack_chain_digest": chain,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "scratch_parent": str(base.parent),
            "note": "CPU-only synthetic measurements; no GPU, real text or training",
        }


def main(argv: list[str] | None = None) -> int:
    """Run one isolated bounded benchmark case and write JSON to stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, required=True)
    parser.add_argument("--tokens-per-document", type=int, required=True)
    parser.add_argument("--fragment-size", type=int, default=257)
    parser.add_argument("--restart-after", type=int, default=0)
    parser.add_argument("--scratch-parent", type=Path)
    args = parser.parse_args(argv)
    try:
        result = benchmark_case(
            documents=args.documents,
            tokens_per_document=args.tokens_per_document,
            fragment_size=args.fragment_size,
            restart_after=args.restart_after,
            scratch_parent=args.scratch_parent,
        )
    except (ValueError, AssertionError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
