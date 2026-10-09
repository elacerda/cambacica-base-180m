# C1-BD2.5 Independent Calibration and Production Performance Preflight

**Date:** 2026-10-09
**Status:** preflight evidence prepared; scientist review pending
**Recommendation:** **OPTIMIZE** before a new production gate; C1-BD3 remains **NOT RUN / NOT APPROVED**

## Scope and inputs

This bounded preflight prepared a blinded 40-case review packet, checked a suspected approximate-match aggregation defect, and profiled the actual matcher against a deterministic sample of the pinned post-D1 corpus. It did not run a full scan or make training-data decisions.

The input identities were:

- Post-D1 corpus manifest: `57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428` (21,603,689 records; 21,470,091,017 normalized words; 46,567,510,543 compressed bytes).
- BD2 snapshot manifest: `d9b5bfdec543af37fdb8733239b0f8b0255339454b5ebca8880371bb86f77cda`.
- BD2 calibration manifest: `904fd5dbe8175b7b2c216af69e4a592a5cca20e9314ac21c694b10471e2ea28e`.
- CALAME-PT revision: `353671bc95cc3d94d488201f67d41b640eb80c55`; Belebele revision: `d4c91dedc9de484dbea7b7d940f898f59fd135e9` (`por_Latn/test`).

The measured run used the full default `CandidatePolicy`: `c1-bd2-policy-candidate-v1`, 13-token anchors, 50-token contiguous threshold, 80% distinctive coverage, two rare anchors, benchmark item DF ≤2, corpus anchor DF ≤100, six-token minimum question, 128-token question/answer gap, eight-token exact seed, 8,192-token windows, and 256-document SQLite commit interval. The full policy JSON and SHA-256 `602f9d851cdc038b433ee252011cdddf48834f3ebd1ca1d625396baab7128267` are in `performance_results.json`; BD3 policy freeze remains false.

The complete preflight packet, performance outputs, checksums, and sample identities are under `/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2-preflight/`. They are outside Git; no corpus or benchmark bulk data was committed.

## Independent 40-case panel

The panel has ten expected positive controls and ten expected hard negatives for each benchmark. Its 40 benchmark example IDs are disjoint from the BD2 fixture examples; Belebele examples also use distinct passages. Positive constructions cover full items and passages/contexts, question plus answer, embedded copies, punctuation/case changes, small insertions and deletions, partial contiguous overlap, and short distinctive fields. Negative constructions cover answer-only text, generic Portuguese, common multiple-choice instructions, topic-adjacent different facts, bibliographies, shared proverbs, distinct entities with a shared template, generic repeated boilerplate, and benchmark fragments separated by long gaps.

All 40 are controlled synthetic documents made from pinned benchmark text or explicitly synthetic negative text. The 12-million-word corpus sample produced no candidate hits, and no independently verified real positive was found in this bounded work. The panel therefore makes no claim of two real source families per benchmark. Every expected class remains unverified by a scientist. The blind CSV has an empty scientist-review label and omits expected class and construction rationale; the separate answer key records those fields and synthetic provenance. This is a review packet, not scientific sign-off or a population precision/recall estimate.

Against construction labels, the matcher detected the intended benchmark example in **20/20 positive controls** and returned no candidate in **20/20 negative controls**. These are controlled fixtures, not human-verified accuracy results. The candidate policy still has `freeze_for_bd3=false`.

All 2,076 CALAME rows remain in the immutable snapshot. Row 718 has a whitespace-only target and is excluded from the future accuracy denominator by the metric convention: 2,075 evaluable rows. No model evaluation was implemented or run. Benchmark source-rights and provenance caveats remain open.

## Matcher correctness

The audit demonstrated a defect in the previous approximate coverage logic. It unioned benchmark-coordinate anchor positions across a document without requiring a localized, order-consistent corpus alignment. Distant fragments could therefore combine into high aggregate coverage.

The matcher now finds a coherent relative-offset band and reconstructs a monotone benchmark-to-corpus anchor chain before calculating distinctive coverage and anchor count. Its drift allowance is derived from the existing 80% coverage candidate value (up to 25% of benchmark-field length); it is a provisional safeguard, not a scientist-approved threshold or an absolute document-distance limit. Full exact evidence remains on its separate path and is preserved regardless of document length.

Focused tests cover a contiguous embedded passage, fragments separated by thousands of tokens, reversed fragments, distant repeated boilerplate, two anchors without substantial overlap, edited copies with insertions/deletions, exact full items in multi-window documents, offsets, high-frequency anchors, and bounded Python memory on a long document. The focused suite passed: **27 passed**. The full required validation is recorded at handoff. Heavily edited copies whose coordinate drift exceeds the provisional allowance may be missed; scientists must review this trade-off before policy freeze.

A measured hot path was also reduced: the exact lookup used to allocate and probe several seed tuples at each token. It now probes one full-length seed and a first-token bucket for shorter fields. On the identical deterministic profile sample, exact lookup/evidence time changed from 14.483s to 8.807s (39.2% lower); end-to-end profile time changed from 55.857s to 52.587s (5.9% lower). The latter comparison includes normal run-to-run variation. Exact and short-question fixtures pass after the change.

## Performance measurement

The actual profile command was:

```bash
PYTHONPATH=src python3 -X faulthandler scripts/profile_decontamination_sample.py \
  --input-root /mnt/data/cambacica-base-180m/deduplicated/exact \
  --snapshot-dir /mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2/snapshot \
  --output-dir /tmp/c1-bd25-profile-optimized \
  --scratch-dir /tmp/cambacica-bd2-preflight-scratch-optimized \
  --target-words 12000000 --max-seconds 1800 --stress-documents 64
```

The SHA-256 identity of the selected row-group set is `099c8ac5e29fc07e79d23a133e0a9219c4854587d8631ecacf3cd2474e024b52`. It processed 13,927 records and 12,055,769 normalized words across all five source families in 52.587s elapsed profile wall time and 52.424 CPU seconds. Average end-to-end throughput was 229,252 words/s; scan-loop throughput was 238,087 words/s. Peak RSS was 712,757,248 bytes. SQLite was 1,146,880 bytes with 13,927 changes and no candidate evidence. Candidate hit and anchor-evidence outputs contain zero rows. The SQLite size reflects the no-hit sample and does not estimate match-heavy storage.

The selected-column compressed row-group metadata upper bound was 82,844,772 bytes. `/proc/self/io` reported 141,242,210 `rchar` bytes but only 4,096 `read_bytes`, indicating most reads were served from cache; exact physical compressed bytes read were not measurable. Input read/decompression and Python conversion plus content hashing took 0.393s and 0.339s respectively in this cached sample. These values do not characterize uncached Orion NFS behavior. Matcher index construction took a separate 4.080s; profile elapsed excludes input inventory validation and row-group planning.

| Source family | Documents | Words | End-to-end words/s | Long docs (≥8,192 words) | Short docs (<32 words) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Carolina | 695 | 1,011,750 | 148,726 | 23 | 398 |
| GigaVerbo v2 | 5,987 | 9,439,241 | 251,265 | 169 | 0 |
| Gutenberg PT | 21 | 427,426 | 245,294 | 19 | 0 |
| Parlamento PT | 5,720 | 584,796 | 275,908 | 0 | 3,337 |
| Wikipedia PT | 1,504 | 592,556 | 277,305 | 1 | 64 |

The profile used 400,000-word minimums for small families, then allocated the remainder by pinned word share. It is stratified rather than a probability sample. GigaVerbo covered `fineweb_2_pt`, `hplt2_pt`, and `finepdfs_por_Latn`; two documents above the 1,000,000-character profiling cap were skipped. The longest included document had 136,862 normalized words. Three GigaVerbo subsets and these skipped records limit generalization to all source subfamilies and extreme document lengths.

A separate 2-second bounded smoke run reached its internal wall-time limit, emitted `TIME_LIMIT_REACHED_BEFORE_FINALIZATION` with partial counters, skipped finalization/stress work, and exited without leaving a long-running process. The full 12-million-word run stayed under its 30-minute cap.

Tokenization accounted for 34.355s of timed matcher stages; exact lookup/evidence 8.807s; anchor lookup/evidence 4.899s; SQLite accounting 0.178s, frequency updates 0.011s, and commits 0.158s. Finalization took 0.002s with no hits. Output serialization took 1.229s, including synthetic stress output. The main measured bottleneck is Python tokenization, followed by exact seed lookup. Input reading was a small part of this cached sample.

The separate synthetic match-heavy run processed 15,104 words in 64 wrappers, creating 256 candidate rows and 28,160 anchor-evidence rows. It used a 13,131,776-byte SQLite database, took 1.919s total, 0.379s finalization, and 1.228s output serialization. Its throughput (7,871 words/s) is synthetic-only and is not corpus throughput. This run demonstrates the cost of dense evidence, not likely corpus match density.

## Runtime and storage projection

The projection weights each source family's measured end-to-end rate by that family's full post-D1 word total. The raw measured-rate projection is 24.64 hours. Scenarios apply explicit throughput factors to that estimate:

| Scenario | Assumed rate vs measured family rates | Approx. wall time | Approx. CPU demand if CPU-bound |
| --- | ---: | ---: | ---: |
| Optimistic | 1.10× | 22.4 hours | 22 CPU-hours |
| Central | 0.80× | 30.8 hours | 31 CPU-hours |
| Conservative | 0.55× | 44.8 hours | 45 CPU-hours |

These are extrapolations from 12.06 million sampled words, not Orion measurements. The profile was nearly single-core CPU bound. The corpus mix weights GigaVerbo heavily, where the sample rate was 251k words/s; Carolina was slower at 149k words/s. A 46.57 GB sequential input read alone would take about 39 minutes at 20 MB/s, 16 minutes at 50 MB/s, and 8 minutes at 100 MB/s. Metadata latency, cache state, decompression, shared CPU, and candidate density can change these results; the profile's cached read counters do not validate Orion NFS throughput.

Observed peak RSS was 0.71 GB. A planning range of roughly 0.8–2 GB is reasonable only if the matcher index and per-document lengths remain similar; larger or pathological records can raise it. No corpus-wide candidate evidence density was observed. For a transparent scratch sensitivity, the zero-hit sample implies about 82 bytes per seen document (approximately 1.8 GB for 21.6 million IDs). The synthetic stress database used about 466 bytes per anchor-evidence row and about 110 evidence rows per candidate row. Applying those synthetic expansion factors:

- At 0.01% of documents with one hit row each, scratch is about 1.9 GB and compressed output about 1–2 MB.
- At 0.1% with one hit row each, scratch is about 2.9 GB and compressed output about 14 MB.
- At 1% with four hit rows per matching document, scratch is about 46 GB and compressed output about 0.5 GB.

These storage values are scenario calculations, not estimates of prevalence. The output range can grow several-fold when evidence is less repetitive than the synthetic passage. A high hit density could require hundreds of GB of scratch. Production must measure free local scratch and reserve headroom after scientists review a bounded Orion run; current zero-hit sample storage cannot justify a fixed storage allocation.

## Restart safety

The BD3 scan is not resumable. `MatcherRun` uses a temporary SQLite database and deletes it on close; production also removes its incomplete staged output on failure. Although SQLite commits every 256 documents, no durable progress manifest or index state is tied to those commits. A failure therefore requires rescanning from the beginning.

The simplest safe design is deterministic shard and row order with a durable SQLite database whose transaction includes document evidence, distinct-document anchor-frequency updates, exactly-once record accounting, and a committed progress marker. The checkpoint identity must bind the exact input manifest, ordered shard inventory, snapshot manifest, matcher version, and complete policy digest. Resume must reject any identity mismatch, skip only rows recorded in the same committed transaction, and retain corpus-wide anchor-frequency state. Candidate finalization and output writing must happen only after all shards complete; independently filtered shard outputs cannot be concatenated because document frequency is global. Keep the existing staged-directory publication atomic, add an independently verified restart-versus-uninterrupted equivalence test, and publish the final manifest only after checksum and row-count verification.

No checkpoint mechanism was implemented or tested in this preflight. Resumability remains a BD3 readiness requirement. Multi-day exposure is material: the current projected run is about one to two days, and a late failure could lose nearly all elapsed work.

## Recommended gate and remaining decisions

**C1-BD2.5: PREFLIGHT COMPLETE** for the supported deliverables, pending scientist review of the panel. **C1-BD3 recommendation: OPTIMIZE; NOT RUN / NOT APPROVED.** Before a new BD3 decision, implement and test restart-safe persistence, review the approximate alignment drift safeguard and calibration labels, confirm Orion throughput and scratch capacity, and document scientist approval of the final policy. Do not freeze the policy based on the synthetic controls.

Required gate statuses remain:

- D1: **COMPLETE / PASS**; D2–D2d: **COMPLETE**.
- Near-deduplication: **APPROVED / CLOSED**, zero removals.
- C1-BD1: **APPROVED**; C1-BD2: **COMPLETE**; C1-BD2.5: **PREFLIGHT COMPLETE**.
- C1-BD3: **NOT RUN / NOT APPROVED**; C1-BD4: **NOT RUN**.
- C1: **IN PROGRESS**; C2: **PENDING**.

No exclusions, train/validation/test splits, tokenizer training, model training, A/B/C mix changes, or full-corpus scan occurred.

## Optional bounded Orion profile command

The local measurement does not establish Orion NFS or CPU performance. If the scientist wants a same-sized Orion confirmation before revisiting the gate, run one 12-million-word profile in a dedicated `tmux` session. Set `ORION_REPO` to the checkout and `ORION_SCRATCH_ROOT` to local scratch with sufficient free space:

```bash
PROFILE_ID="$(date -u +%Y%m%dT%H%M%SZ)"
ORION_REPO="$HOME/dev/cambacica-base-180m"
ORION_SCRATCH_ROOT="/local_nvme/cambacica-bd25-$PROFILE_ID"
ORION_OUTPUT="/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2-preflight/orion-$PROFILE_ID"
ORION_LOG="/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2-preflight/logs/profile-$PROFILE_ID.log"
mkdir -p "$ORION_SCRATCH_ROOT" "$(dirname "$ORION_LOG")"
test ! -e "$ORION_OUTPUT"
tmux new-session -d -s "c1-bd25-$PROFILE_ID" \
  "cd '$ORION_REPO' && /usr/bin/time -v timeout --signal=INT --kill-after=10s 1800s env PYTHONPATH=src python3 -X faulthandler scripts/profile_decontamination_sample.py --input-root /mnt/data/cambacica-base-180m/deduplicated/exact --snapshot-dir /mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2/snapshot --output-dir '$ORION_OUTPUT' --scratch-dir '$ORION_SCRATCH_ROOT' --target-words 12000000 --max-seconds 1740 --stress-documents 64 > '$ORION_LOG' 2>&1"
```

Check progress with `tmux attach -t c1-bd25-$PROFILE_ID`, `tail -f "$ORION_LOG"`, and `ps -o pid,etime,%cpu,%mem,rss,cmd -C python3`. The script's internal timer requests a partial report at 29 minutes; the outer timeout sends an interrupt at 30 minutes and allows 10 seconds for SQLite cleanup. For early safe termination, use `tmux send-keys -t "c1-bd25-$PROFILE_ID" C-c`, then confirm the process exited before inspecting the unique scratch directory. Preserve the log and partial outputs; remove only that run's uniquely named scratch directory after confirming no process uses it.
