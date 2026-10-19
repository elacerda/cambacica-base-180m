# C4-P1.3 — Deterministic streaming, sampling and resumable cursor design v1

**Status update (2026-10-19): SCIENTIST APPROVED INITIAL C4-P1.3
DESIGN BASELINE — O1 + SINGLE LOGICAL PRODUCER + ACKNOWLEDGED CURSOR.**

**Subsequent implementation and verification (2026-10-19):**
Following this initial design decision, the scientist *separately*
authorized the synthetic CPU-only C4-P1.3a prototype. It was
implemented and pushed as
`0c30fffcfa17c76767f9516283410c7fbd514683`, and
the Orion evidence now supports **C4-P1.3a COMPLETE / PASS**:
49 focused tests passed; the full available CPU suite reported
450 passed / 5 skipped; Ruff lint and post-format checks passed.
See [C4_P1_3A_SYNTHETIC_CPU_IMPLEMENTATION_V1.md](C4_P1_3A_SYNTHETIC_CPU_IMPLEMENTATION_V1.md).
Statements below about CPU implementation "not authorized" describe
the **original design-only approval**, not the later separately
authorized synthetic P1.3a task. No C1 production processing, GPU,
optimizer coupling or training has been approved.

The scientist approved the proposed *architecture baseline* after the
design-only proposal had been published. The approved choices are
(1) O1: an external hash-ordered, disk-backed index for a fixed immutable
reference view and split; (2) a **single logical producer**, with
bounded buffers and deterministic ordered emission; and (3) a cursor
anchored to the highest contiguous **consumer-acknowledged/committed
window**, not the last produced or prefetched item.

The approval also establishes the intended **synthetic crash/replay
verification objective** as a design requirement, not an executed test.

**Approval scope:** architecture and scientific design only. This
does **not** select a production sampler, concrete random seed,
corpus mixture, epoch/reuse policy, token budget, batch, checkpoint
format/cadence, memory/disk budget or implementation technology.
It authorizes no CPU implementation/tests, corpus read, tokenizer run,
NFS materialization, GPU operation, optimizer step, C3.7 pilot,
C5 smoke training or C6 pretraining. Separate scientist authorization
is required before even a synthetic CPU streaming prototype is built
or executed.

## 1. Previously approved contracts (not reopened)

- C1 post-D1 population: **21,603,689** retained documents. Exact
  deduplication, near-dedup decision and BD4 contamination policy do
  not change. C1 remains **IN PROGRESS** on source rights/usage and
  final A/B/C mixture.
- C1-S1 split manifest SHA-256:
  `8b937f90d40487c643fe48baa6b5c2950713fe2e0d2d534f3e2a3c08e74dbe16`.
  Assignment Parquet SHA-256:
  `ef52e3cf3033934df13e3d16c207e1783c509dff6fb91954fb8c25ab719ece1e`.
  The split is **group-consistent**: all occurrences and segments inherit
  their C1-S1 document/group assignment. C1 `occurrence-id-v2`
  references source-relative shard and original absolute row ordinal.
  Never reassign or recalculate split membership based on a later order.
- C2: SentencePiece Unigram v24576, model SHA-256
  `c8982a23d3c5ecc0a61d300edc1e354e38f1358aa3db5420d5bf19f7b95cbf23`;
  EOS=2, BOS=1, `pad_id=-1` (invalid embedding input).
- C3: Architecture C, 176,180,224 tied parameters, context 4096.
  C3.6 synthetic throughput/memory/restart results remain engineering
  evidence only; no final training batch or optimizer is selected.
- C4-P1: independent per-split contiguous token streams separated by
  one EOS **after every document**, including empty documents. Cross-
  document causal attention within a window is allowed. P1.1 initial
  policy scores inserted EOS and valid cross-document next-token
  targets; inserts **no automatic per-document BOS** and initially
  inserts no split-start BOS; target labels `-100` are ignored.
- C4-P1.2: **COMPLETE / PASS** in a bounded synthetic CPU loss
  integration with the actual C3 model class; format-only commit
  `29e966a283b2fc7f1343fefacceeb3c47ad9287f`. This does not
  validate real tokenization, scalable streaming or dataloader resume.

The pure-Python reference
`src/cambacica/packing/p1_reference.py` materializes `tuple(documents)`,
all split streams, windows and provenance in memory; its
`_window_provenance` scans document spans for each window. It is a
**correctness oracle on small synthetic fixtures**, not a scalable
implementation for 21.6M documents. A streaming implementation must
be a separate bounded-memory producer and be compared against the
reference for manageable synthetic inputs.

## 2. Proposed logical interfaces (no code yet)

`C1-S1 split-preserving reference view -> deterministic document order
-> token-fragment source -> P1 EOS stream -> causal target-window builder
-> split-specific consumer -> acknowledged cursor -> atomic snapshot`

### A. Split-filtered immutable document references

A reference includes: dataset/view ID and digest, C1-S1 manifest and
assignment digests, source/subset/revision, source-relative shard,
absolute original row ordinal, `occurrence-id-v2` document ID,
C1-S1 group ID, split, normalized content hash, and any stable native
record ID. Validate the **already-published** C1-S1 assignment; do
not derive a new holdout from the sampler seed.

Train, validation and test must have **disjoint document/group
identities and entirely separate stream/window/cursor namespaces**.
Shared numeric token IDs do not indicate leakage: provenance does.

### B. Deterministic document ordering

Two alternatives are scientifically distinguishable. **O1 is now
APPROVED for the initial design baseline; O2 remains an unselected
alternative**. The precise ordering-key serialization/namespace,
seed/epoch values and treatment of selected/repeated documents
must be versioned before implementation:

- **O1 — external canonical hash order (recommended for reproducibility):**
  over a fixed approved reference view and one independently assigned
  split, compute a versioned, domain-separated SHA-256 ordering key
  over `(view_digest, split, epoch, sampling_seed, document_id)`,
  sort by `(key, document_id)` using a **disk-backed index/external
  sort** and emit consecutive ordinals. The ordering seed is NOT the
  C1-S1 split seed, and the order must not depend on Parquet shard
  layout, read batch size or worker scheduling. This is a stable
  permutation of whatever view has been selected; it does **not**
  choose A/B/C, quotas or the number of document repeats.
- **O2 — seeded bounded-buffer shuffle:** fixed input order plus
  pinned PRNG algorithm/state and explicitly versioned buffer size.
  A bounded shuffle is **not** a uniform global permutation, and
  changes to buffering/parallelism can change the result unless
  ordering/PRNG consumption are fully specified. A deterministic
  replay design is possible but requires a stronger cursor and
  experimental bias review.

**Scientist decision: O1 approved as the initial ordering
architecture**, not a complete sampling policy. Hash-ordering
documents is distinct from sampling sources to satisfy mixture
targets. Holdout validation/test references should remain fixed
and free of training-epoch reshuffling. No actual ordering seed or
train-view mixture has been selected.

The sampling protocol still must decide, in a separate C1/C4 review:
final A/B/C mixture, source/subset quotas, **declared measurement unit**
(normalized words vs model tokens), replacement/repeats, complete
versus partial document exposures, source exhaustion, long-document
weighting, epoch boundaries and the handling of tiny strata. A
normalized-word share is **not automatically** a model-token share.
Do not treat the provisional `corpus_mix_b.yaml` as selected.

### C. Tokenized-document fragment interface

A token fragment carries immutable document ID, split, fragment
ordinal, document-relative token offsets `[start,end)`, tokenizer
identity, payload token IDs and a continuity/final-fragment marker.
The consumer must verify strictly consecutive offsets, one final
marker, exact reconstruction of token order, and that exactly
**one EOS is appended after the final fragment**. Never append EOS
at a row-group, shard, buffer, batch, queue or intermediate fragment
boundary. Preserve empty documents (they produce EOS only).

**Important unresolved tokenizer issue:** SentencePiece Unigram
tokenization is not generally additive over arbitrarily sliced text.
A future real-data adapter must demonstrate token-equivalence to
the **whole normalized document** or use a scientifically reviewed,
lossless segmented-tokenization strategy. Do not assume that joining
independently tokenized byte/text chunks reconstructs the same IDs.
The C4-P1.3 design consumes only hypothetical/pre-tokenized artificial
fragments; it does not authorize reading or tokenizing C1 text.

### D. P1 token-to-window state machine

With context `C=4096`, given an independently constructed split stream
`s[0:L]`, emit pairs `(s[i],s[i+1])` for target indices
`i+1 in [1,L)`. Window `k` begins at `k*C` and owns target
indices `[k*C+1, min((k+1)*C,L-1)]` exactly once.
A full window consumes `C+1=4097` backing tokens and advances
target ownership by **4096**; one boundary token is retained as
the next window's initial input. The boundary carry is not a new
corpus token and must not be counted as a second scored target.

Streaming state needs **at most the fixed window, one-token carry,
a bounded fragment queue and bounded provenance spans**, rather than
the full corpus. All memory ceilings are *future configuration values
that must be measured and approved*; no byte budget is inferred here
from the C3.6 GPU process peak.

A final partial window may use EOS ID 2 as a safe input pad and label
`-100` for unscored slots. Do not count padding as corpus tokens,
new EOS, or targets. Aggregate eventual CE by valid label count,
not arithmetic mean of differently filled windows. Zero-target
split streams retain document/accounting records and yield no
loss-bearing window. The P1 policy provides no special attenuation
at EOS/document boundaries.

Shards, read batches and prefetch buffers are **I/O partitions, not
scientific sample or attention boundaries**. In particular, never
flush and pad a partial window merely because a shard or queue
ends; doing so would change target accounting and context.

## 3. Deterministic execution versus parallel workers

Every document and output window receives a stable logical ordinal
before any dispatch. Workers may asynchronously fetch/produce data,
but final emission occurs by global ordinal with a **bounded reorder
buffer**. Worker count, scheduling and read batch size must not
change token IDs, EOS placement, target ownership, window provenance
or the manifest's logical ordered-output digest.

A **single logical producer is approved as the initial architecture
baseline**; multi-worker dispatch is an unapproved future
optimization and must pass byte-identical/digest-identical synthetic
output checks against the single-worker oracle. Runtime scheduling order itself
is not the sample order. Fail closed on missing, duplicate or
out-of-order fragments, lost EOS, split transition and identity drift.

If an ordering plan requires a full document index, its disk footprint
and build cost must be budgeted explicitly. A disk-backed index can
use bounded resident memory while still using O(number of references)
**local disk space**; this is not a zero-cost global shuffle.

## 4. Cursor/restart and acknowledgment protocol

Separate **produced**, **prefetched**, **delivered** and **durably
acknowledged** positions. The scientifically meaningful resume point
is the highest contiguous window whose effects are committed by the
consumer, **not** the worker queue tail.

**Proposed minimal cursor schema (versioned; fields/serialization
not yet frozen):**

- run/view manifest ID and SHA-256; C1-S1 manifest and assignments
  SHA-256; C2 tokenizer identity; C4-P1.1 policy digest;
  ordered-document-index digest and sampling policy/seed/epoch ID;
  source/shard identifiers and document/fragment logical ordinals;
- split-specific global stream token offset, current document-relative
  payload offset, EOS-emitted flag, carry token and its provenance;
- next window ordinal, last produced/delivered ordinal, last
  **acknowledged** ordinal, cumulative payload/EOS/scored-target
  counters and audit/provenance digest chain;
- reader/index version, PRNG state **only when a PRNG-based ordering
  mechanism is selected**, and unresolved in-flight prefetch identities
  sufficient to discard/replay deterministically;
- crash-consistent snapshot schema/version, checksum, source code
  commit, and operational configuration digest.

Cursor invariants: when resuming, validate all frozen identities
**before** reading further tokens; restore at the committed position,
reconstruct or discard unacknowledged prefetch data and re-emit
uncommitted outputs deterministically. An appended EOS is committed
exactly once per document *in the logical stream*, even if work is
replayed physically after a crash.

**Critical atomicity boundary:** the approved acknowledgment
architecture does not mean an implementation or a verified
recovery mechanism exists. A data cursor checkpoint alone cannot
guarantee exactly-once **optimizer updates**. A future
training-state snapshot must atomically bind model, optimizer,
scheduler, RNG, data index/cursor, stream identity, update counter
and scored-token count; absent that coupling, replay can change
training results. C3.6's synthetic model-restart PASS does not
prove dataloader restart. Proposed semantics are **no loss or double
counting of *committed* windows**, with replay allowed for
unacknowledged work; do not promise globally exactly-once external
effects without a transactional consumer contract.

## 5. Crash safety and storage planning (future execution only)

- Use local ext4 for active indices, write-ahead cursor state,
  staging files and checkpoints; NFS may hold immutable published
  manifests/reports after verification. Production corpus remains
  immutable in its authorized storage.
- An eventual implementation should write a new generation to a
  temporary file, `fsync` file, atomically rename within ext4 and
  `fsync` parent directory. Never overwrite immutable published
  identities. On resume, verify checksum and reject damaged or
  incompatible generations rather than silently starting elsewhere.
- Bound resident token buffers, pending reference records, queue
  items, worker count, window provenance metadata and disk index
  bytes through **separately reviewed configurations**. Measure
  peak RSS, read amplification and performance in a separately
  authorized future experiment. Do not reuse C3.6's **GPU**
  process-memory accounting as a CPU/disk budget.
- Potential failure cases: process kill before/after EOS, before/
  after window delivery, during snapshot temporary write, after
  atomic rename, at document/shard/epoch boundary, at window
  boundary, with in-flight queue and with corrupted state.

## 6. Proposed synthetic-only CPU acceptance matrix (NOT EXECUTED)

A future implementation can use tiny **artificial token IDs and
artificial document metadata**, with *no* C1 input paths and no
PyTorch/CUDA dependency for the streaming core:

1. Match the P1 reference exactly (tokens, window inputs/labels,
   EOS offsets, spans, accounting, ordered hashes) on bounded fixtures
   including split interleaving and empty/one-token documents.
2. Match at 4096/4097/8193 boundaries; carry the preceding target
   token as the next window's first input and assign every target
   exactly once.
3. Long artificial documents split into irregular fragments;
   fragmentation, shard and batch boundaries must not emit extra
   EOS, flush partial windows or change the output digest.
4. Cross-source/split/group contamination and wrong C1-S1 identity
   must fail closed. No test may read C1 production shards.
5. Test zero-target single-EOS stream, EOS scoring, padding and
   masked labels per C4-P1.1; verify total scored targets exactly
   `max(L-1,0)` for approved default policy.
6. Change fragment size, buffer size, worker scheduling or input
   dispatch order without changing the pinned logical order:
   ordered outputs and canonical digests must be identical.
7. For each crash-injection point, compare uninterrupted versus
   snapshot/resume **committed window sequences and counters**.
   Unacknowledged windows may replay; no committed target may
   disappear or be committed twice.
8. Reject corrupted checkpoint, altered seed/view/split/tokenizer
   digest, changed ordering/index version and invalid cursor
   checksum; do not silently reset from the beginning.
9. Assert memory remains bounded by configured queue/window limits
   while artificial input count grows; collect actual resource
   evidence, not an unverified complexity assertion.
10. Explicitly exercise one-worker vs synthetic multi-worker
    emission equivalence if multi-worker mode is proposed; no
    benefit may be claimed until measured.
11. Verify stable heldout streams when a **training** order/epoch
    changes; holdout views must not inherit training shuffle.
12. Check that the restart contract includes the acknowledged
    consumer boundary and all sample-order provenance rather
    than only a Python generator index.

Tests of SentencePiece equivalence on real normalized text,
real-data I/O/occupancy, actual NFS/SSD cost, optimizer coupling
and GPU throughput are **separate later gates** requiring explicit
approvals.

## 7. Scientific decisions needed before implementation approval

| Question | Recommended first design | Decision status |
| --- | --- | --- |
| Ordering O1 vs O2 | **O1** external hash-ordered disk-backed index over fixed immutable reference view | **O1 DESIGN APPROVED**; key serialization, seed and selected view still OPEN |
| Initial execution topology | One logical stream/window producer; bounded buffers and deterministic emission | **DESIGN APPROVED**; memory bounds/implementation still OPEN |
| Seed and epoch rule | Versioned independent sampling seed, no hidden reuse; explicit epoch boundaries | OPEN |
| C1 mixture and quota unit | Keep **blocked** until C1 rights and mix; do not equate normalized words to tokens | BLOCKED BY C1 |
| End-of-epoch / incomplete stream policy | Treat epoch boundary as explicit event, not implicit BOS/EOS or shard flush; settle context/reuse rules separately | OPEN |
| Partial-window policy at final split/epoch | Mask remaining label slots; never treat shard/buffer end as a final split | P1.1 principle APPROVED; epoch rule OPEN |
| In-flight prefetch and ack | Resume at highest contiguous consumer-acknowledged/committed window; replay unacknowledged work | **ARCHITECTURE APPROVED**; atomic consumer binding/format still OPEN |
| Cursor/checkpoint format and cadence | Checksum + atomic local ext4 snapshot tied to future consumer state | OPEN |
| Bounded memory/disk budgets | Explicit values from measured synthetic/approved later tests, not guessed | OPEN |
| CPU synthetic prototype C4-P1.3a | Separate authorization granted; synthetic O1/index/cursor implementation verified on Orion CPU | **COMPLETE / PASS** (synthetic only) |
| Real C1 adapter/tokenization and any GPU | Separate rights/usage review and execution authorization | NOT AUTHORIZED |

**Scientist approval now recorded:** the initial design baseline
selects O1, a single logical producer, stable split-specific ordered
window IDs, and a cursor at the last contiguous consumer-acknowledged
window. Crash/replay on artificial data is the required *future
verification approach*, not a test already run or permission
to implement the test harness.

**Subsequent gate resolution:** the bounded CPU-only synthetic
streaming prototype was separately authorized and verified in
C4-P1.3a; see the linked implementation report. This design approval
alone still cannot be used to justify any **new** implementation,
production dataset access or training workload.

## 8. Status and stop line

**C4-P1.3: INITIAL DESIGN BASELINE SCIENTIST APPROVED;
SYNTHETIC CPU FOLLOW-UP C4-P1.3a COMPLETE / PASS.**
**C4: IN PROGRESS. C1: IN PROGRESS. C3.7, C4 production processing,
C5 and C6: NOT AUTHORIZED.**

The original design-only decision performed no implementation.
The subsequent separately authorized P1.3a implemented and verified
a synthetic CPU prototype; it did not read/tokenize production
C1 text, run a real dataloader, use GPU or update a model/optimizer.
