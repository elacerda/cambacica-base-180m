# C4-P1.3a — Synthetic deterministic disk-indexed P1 streaming prototype

**Status: IMPLEMENTED AS A PATCH / ORION VERIFICATION PENDING.**

The scientist authorized a **synthetic CPU-only** implementation and tests after approving O1 hash ordering with a disk-backed index, a single logical producer and a contiguous consumer-acknowledged cursor. This is an experimental fixture-specific implementation, **not an authorization to use any C1 corpus input or GPU**. No tokenizer/model/training code is called.

## Scope

- New `src/cambacica/packing/p1_streaming_synthetic.py`: explicit artificial P1 `Document` iterator -> SQLite disk-backed O1 sort index and a `payload.u16` local binary store, a single split-specific window producer, and checksum-protected snapshots.
- New `tests/packing/test_p1_streaming_synthetic.py`: small artificial fixtures compared against the existing `p1_reference` implementation (windows, provenance, stream/ordered-span SHA-256); deterministic order, split isolation, 4096/4097/8193 boundaries, irregular fragments, acknowledgment/replay, failure injection, tampering and corruption handling.
- No modification to frozen corpus, tokenizer, model, P1 reference, optimizer, recipes or GPU service.

## Design contracts for this synthetic version

The *scientifically approved architecture* O1 is implemented via SHA-256 keys under a **prototype-only** domain string, using explicitly supplied fixture `view`, `seed` and `epoch`. The seed and canonical-key encoding here are **not** approved as production sampling policy; no A/B/C mixture, epochs or reuse are selected. For validation/test, O1 ordering remains fixed under changes to the *training* seed/epoch.

The local index contains artificial token IDs, not source text. It requires an **explicit, newly created output directory** and spools an iterable to disk. Initial publication uses temporary directory, `fsync`, atomic directory rename, and a manifest with SHA-256 of its two stored files. Memory is bounded with respect to number of documents; each input document is an artificial `Document` whose token tuple is already supplied by the caller.

The producer seeks from `window_ordinal * 4096` using a B-tree index of split-specific ordinal and token offsets. Each window reads at most 4097 tokens, across arbitrarily sized synthetic token fragments; EOS is inserted **exactly once after each complete document**, never at a fragment/IO boundary. The reference's next-token alignment, cross-document causal targets, 4096/4097 overlap, EOS scoring, `EOS=2` safe input pad and label `-100` are retained.

A single `SyntheticWindowConsumer` holds **at most one unacknowledged window**. It writes an atomic JSON checksum snapshot on `acknowledge(window)`, storing the index identity, split, last contiguous acknowledged ordinal, count of committed targets, a rolling acknowledgment chain, next input offset, next ordinal and carry token/provenance. On restart, the index files and snapshot are verified, and the last committed window is independently reconstructed; the first unacknowledged window is replayed. **No external optimizer effect is committed atomically with the cursor**. Replaying work whose external effect was already performed but not acknowledged may duplicate that effect; this is *not* an exactly-once training guarantee.

## Validation evidence and limitations

- Offline container check on 2026-10-19: **25 focused synthetic tests passed** against a small *local test-only stand-in* implementing the frozen P1 reference interface. The stand-in is **not packaged**, and its success is **not a substitute for verification with the actual repository's P1 reference**.
- **Orion-focused and full-suite validation remain pending.** Pytest and Ruff results from Orion must be supplied before marking C4-P1.3a COMPLETE / PASS. Ruff lint and formatting must be rerun on Orion's two new Python files; this packaging environment does not have Ruff installed.
- The implementation and tests below were prepared locally, not pushed to `origin/main` in this step. Apply them to a **clean** checkout using the supplied archive and check `git diff` before committing.
- A full-corpus-size index, actual tokenizer segmentation equivalence, real-data RSS, disk bandwidth, production checkpoint format, C1 rights/mixture, batch sizing, concurrency and optimizer-state coupling are **not validated or authorized**.

## Validation commands (Orion, no GPUs)

```bash
cd ~/dev/cambacica-base-180m
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /srv/cambacica-c3/venv/bin/python -m pytest -q tests/packing/test_p1_streaming_synthetic.py tests/packing/test_p1_reference.py
ruff check src/cambacica/packing/p1_streaming_synthetic.py tests/packing/test_p1_streaming_synthetic.py
ruff format src/cambacica/packing/p1_streaming_synthetic.py tests/packing/test_p1_streaming_synthetic.py
ruff format --check src/cambacica/packing/p1_streaming_synthetic.py tests/packing/test_p1_streaming_synthetic.py
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /srv/cambacica-c3/venv/bin/python -m pytest -q

git diff --check
```

Do not stage/push until the tests and diff have been reviewed.

**Gate:** C4-P1.3 architecture APPROVED; C4-P1.3a implementation prepared and locally tested, awaiting Orion integration verification; C4 overall **IN PROGRESS**. C1 still IN PROGRESS. C3.7, real C4 pipeline, C5/C6 and all GPU work remain **NOT AUTHORIZED**.
