# C4-P2 — Scientific design for the future production data pipeline (v1)

**Date:** 2026-10-19.
**Status: DESIGN PROPOSAL / SCIENTIST REVIEW PENDING.**
**Authorization received:** C4-P2 **documentation and scientific planning only**.

This document proposes interfaces, alternatives, invariants, approval
dependencies and test criteria for a *future* Cambacica-180M data/training
pipeline. It does **not** approve or select production sampling,
a corpus mixture A/B/C, source licenses, training epochs, a token budget,
batch size, optimizer, seed, checkpoint cadence or a hardware execution
plan. There has been **no production implementation, runtime benchmark,
corpus access, tokenizer execution, GPU use or model update** in C4-P2.

This contract is a proposal, not an implementation specification already
frozen by the scientist. Where recommendations appear, they are explicitly
**not approved** until a separate decision record is signed.

## 1. Dependencies and scientific identities that cannot change

| Existing gate | Frozen/previously approved consequence |
| --- | --- |
| C1 post-D1 | 21,603,689 retained documents; no new eligibility filtering, silent deletions or deduplication changes |
| C1-S1 | Split manifest SHA-256 `8b937f90d40487c643fe48baa6b5c2950713fe2e0d2d534f3e2a3c08e74dbe16`; assignment Parquet SHA-256 `ef52e3cf3033934df13e3d16c207e1783c509dff6fb91954fb8c25ab719ece1e` |
| C1 rights | **IN PROGRESS**; source usage, training permissions, obligations, redistribution and release require separate resolution |
| C1 candidate mixtures | `configs/corpus_mix_{a,b,c}.yaml` remain **provisional**, denominated in *normalized words*, never selected by C4-P2 |
| C2 | SentencePiece Unigram, vocab 24,576; model SHA-256 `c8982a23d3c5ecc0a61d300edc1e354e38f1358aa3db5420d5bf19f7b95cbf23`; UNK 0, BOS 1, EOS 2, tokenizer PAD -1 |
| C3 | Architecture C: 176,180,224 tied parameters, 4,096 model positions; no architecture change |
| C4-P1.1 | P1 contiguous *within one split*, EOS 2 appended to every document including empty, EOS and cross-document next-token targets scored, no automatic BOS per document or initial split, causal cross-document attention; labels use `-100` on unavailable slots |
| C4-P1.2 | Bounded synthetic CPU integration with actual C3 model **COMPLETE / PASS**; production loss implementation must still make `ignore_index=-100` explicit |
| C4-P1.3 | O1 hash ordering over an immutable view using disk-backed index, single logical producer, bounded orderly emission, acknowledged-window cursor — **architecture approved**, not full production sampling policy |
| C4-P1.3a | Synthetic O1/SQLite packing and cursor **COMPLETE / PASS**, not an actual production dataloader |
| C4-P1.3b | Synthetic-only CPU study **COMPLETE / PASS**; full Orion CPU suite 470 PASS / 5 SKIP; 128, 1,024 and 8,192 artificial document cases on ext4; not a measured real-corpus rate |

The frozen split belongs to document/group identities, not a stream
position. All fragments of each document inherit their original split,
group, source and version. A group must never be reassigned because a
mixture, order, reader, epoch or batch changes.

The C1-S1 split policy is not proof of eliminating every cross-source
near-duplicate or semantic overlap; do not assert stronger decontamination
than C1 actually established.

## 2. Proposed future pipeline boundary

```text
Immutable approved C1-S1 reference manifests (no payload in control plane)
  -> rights/eligibility allowlist (separately approved)
  -> fixed train/validation/test views with signed identities
  -> versioned, split-specific source sampling/ordering schedule
  -> controlled whole-document tokenizer adapter (C2 exact model)
  -> auditable document token fragments with offsets
  -> P1 contiguous EOS packing (window input/labels, 4096/4097)
  -> bounded train/eval batch builder and explicit masked-loss consumer
  -> transactional training update / acknowledged data-state boundary
  -> atomic local checkpoint + independently verified provenance ledger
```

**No arrow above constitutes permission to execute it.** In particular,
the P1.3a `payload.u16` store contains synthetic IDs, not approved real
token shards. The production path must be separately designed, budgeted,
reviewed and authorized, and must not treat the synthetic fixture index
as a production-file interface automatically.

### Data-plane and control-plane separation (proposed)

- **Control plane:** immutable references, checksum pins, rights decisions,
  split assignment identity, ordering schedule, source quotas, offsets,
  group/document identity, token statistics, cursor, manifest digests.
- **Data plane:** actual normalized document payloads, exact tokenizer
  output and stored token fragments; entirely **blocked until C1 rights
  and a separate C4-D execution gate**.
- **Audit plane:** append-only and checksummed source/split exposure,
  ordered document/segment spans, EOS offsets, target ownership,
  rejected attempts, checkpoint generations, stop reasons and
  reconciliation reports. Audit records must not expose copyrighted
  text by accident.

Every transformation is identified by schema version, code commit,
exact input manifest/view digest, configuration digest and output digest.
Fail closed on missing/duplicate document IDs, split/group conflicts,
unexpected source revision, rights approval mismatch or changed content
hash. Do not re-normalize or silently skip documents.

## 3. Mixture A/B/C and quotas: scientifically open

The three C1 candidate configurations have distinct aims:
**A curated-heavy**, **B balanced**, and **C residual-heavy control**.
These descriptions come from their provisional YAML metadata; C4-P2
does **not** endorse B just because the frozen C2 tokenizer's training
sample was B-conditioned.

Their proportions are in **normalized words** (defined at C1 as
`len(text.split())`), not SentencePiece tokens, not EOS-inclusive
stream tokens and not scored next-token labels. They cannot be equated
by multiplying a fixed universal tokens/word factor.

**Proposed two-stage review, not an approved sampling algorithm:**

1. **C1 authorization and reference view:** resolve each source's
   applicable rights and obligations for the intended use, select the
   approved sources/mixture policy, and pin the immutable train-only
   reference view and its subset/stratum membership. Disallowed
   sources must fail closed rather than silently be reassigned a
   numerical weight by the sampler.
2. **C4 declared optimization unit:** separately choose whether quotas
   control (Q-W) *normalized-word exposure*, (Q-T) *C2 payload-token
   exposure*, or another explicitly defined scientific measure.
   **Q-W and Q-T are alternatives, not interchangeable**. Report
   payload model tokens, inserted EOS, total stream tokens, unique
   scored targets and pad/ignored labels even if Q-W is selected.
3. **Document granularity:** whole-document no-reuse prefix admission
   within an approved fixed view is a **candidate** policy (Q-D1).
   If the next whole document would overshoot a quota, stop and
   report the shortfall and unexposed documents; never silently cut
   long documents or move their fragments to another split. More
   sophisticated exact-token scheduling, partial-document views or
   replacement require additional decisions.
4. **Audit actual shares:** for each source/subset, report requested
   shares, eligible capacity, exposed document/group count, normalized
   words, C2 payload tokens, EOS count and scored target count.
   Explicitly report drift from desired shares, headroom, overshoot
   or undershoot and sources that exhaust first.
5. **Comparison fairness:** A/B/C comparisons should share the same
   frozen tokenizer, architecture, held-out references and **declared
   exposure budget**. Record whether the budget is unique payload
   tokens, repeated payload tokens, inserted EOS, scored labels or
   compute/update count. Equality in one metric does not establish
   equality in the others.

The earlier C1 no-reuse capacity ceilings are in normalized words;
they do **not** establish that any given token budget is achievable.
Token censuses, real-dataset packing occupancy and quota feasibility
from C2 output require explicit, rights-cleared access authorization.

### O1 global order does not itself solve source interleaving

A deterministic SHA-256 permutation can be defined for each immutable
split and view, but **weighted interleaving** of selected sources is a
separate policy. Possible future schedulers include:

- **S1: global ordered eligible-document view** with composition
  already frozen upstream. Simple deterministic order; no intrinsic
  quota tracking or weight control.
- **S2: per-source/subset O1 ordered queues plus a deterministic,
  quota-aware merge schedule**. Better explicit exposure reporting,
  but the unit, tie-breaks, source-exhaustion behavior and long
  documents must all be specified and audited.

**No scheduler is selected.** Whichever is approved later must define
a total order independent of file enumeration, physical shard boundaries,
worker timing and batch size. Train uses training-only references;
validation/test use fixed, independently identified evaluation orders
without inheriting the training mixture or shuffle.

## 4. Epochs, reuse and end-of-stream boundary (OPEN)

C4-P1.1 defines normal *document* boundaries and P1 window scoring.
It does **not** approve BOS/reset semantics when an epoch starts, when
a selected view exhausts, or when a source reaches a quota.

Alternatives requiring scientific selection:

- **E1 — independent epoch streams (recommended as an auditable
  baseline):** end the logical epoch stream, close a final partial
  window with masked labels and begin a new, separately identified
  stream next epoch. The initial P1 no-auto-BOS choice remains unless
  independently revised; never invent a target between epoch streams.
- **E2 — continuous multi-epoch stream:** allow causal next-token pairs
  across epoch boundaries. This changes the scored-target set and
  context continuity; it must be compared explicitly with E1.
- **R0/R1 — no reuse versus approved repeat exposure:** decide whether
  documents may appear again, how many times, in what order, under
  what stopping rule, and which counter charges each exposure.
  **No reuse or epoch count is selected by this proposal.**

Under E1, each stream `j` of length `L_j` has
`max(L_j - 1, 0)` candidate next-token targets under the initial
P1.1 loss policy; total candidates are the sum over **independent**
streams, not necessarily `max(sum(L_j)-1,0)`. Padding, BOS and EOS
accounting must be reported independently. For E2, a new scientific
boundary contract would have to specify targets spanning epochs.

A fixed training horizon is a separate decision: unique data coverage,
scored-label target budget, token exposures, updates and optimizer
schedule are **different counters**. Stop/restart/early-exhaustion
behavior must be deterministic and versioned; do not infer a number
of epochs by dividing a desired total by C1 normalized-word counts.

## 5. Tokenizer and P1 sequence interface (design contract)

The future adapter may receive a normalized document only after C1
rights clearance. It must verify the exact C2 model hash and tokenizer
options; disable/detect automatic special tokens before appending
**one EOS=2 per document**, including empty ones. `pad_id=-1`
must never enter an embedding input. A real text chunk is not
guaranteed to tokenize equivalently to a whole document with
SentencePiece Unigram; a long-document strategy needs a correctness
proof/test before being used on real text.

Proposed token-fragment metadata: view and source identities, split,
group/document IDs, document content SHA-256, tokenizer model hash,
fragment ordinal, contiguous document-relative token offsets,
last-fragment flag, per-fragment checksum, source-shard/original
row coordinate and any authorized usage policy tag.

P1 emits 4096 aligned input/next-token label positions based on
a 4097-token backing slice with stride 4096. Exactly one next-token
target owner per global stream index after the first, including
eligible inserted EOS and targets after EOS. Use EOS as a **safe
input pad**, `-100` as ignored labels, and a masked loss with
`ignore_index=-100` explicitly configured. Aggregate batch/window
negative log likelihood by the **number of valid labels**, not
the unweighted mean of per-window losses; handle zero-target streams
without a mean CE over all-ignored labels.

Source-specific loss/mixture statistics need a clear **target owner
rule**, since a cross-document transition can use an input token
from source X to predict the next token from source Y. Recommend
reporting the target document/source as the primary exposure
attribution, with explicit cross-source transition counts; this
is proposed and **not yet approved**.

## 6. Batches, model state and crash-consistent restart

### Batching questions to resolve

- Final microbatch, gradient accumulation, rank/device layout,
  mixed precision, variable-tail batching and sample assignment.
  C3.6's 4×1 is an **engineering candidate**, not a C4 training
  decision.
- Decide whether all-ignored batches are skipped or handled with
  another specified policy; never claim a finite mean CE from an
  empty scored-label population.
- Per-update counters must include **scored targets, payload tokens,
  EOS tokens, real stream tokens, padded/ignored slots, repeated
  exposure and optimizer updates**.
- Validation/test batches must be fixed and isolated from train;
  none of their targets may update training parameters.

### Cursor and checkpoint coupling

The P1.3a synthetic consumer acknowledges one window with a
checksummed local snapshot. This prevents silent changes to the
*synthetic delivered sequence* but does **not** make optimizer
updates exactly-once.

A proposed future **transactional training snapshot** must bind
the following in one crash-consistent logical generation:

- model weights; optimizer, LR scheduler and scaler if used;
- completed optimizer update and gradient-accumulation boundary;
- Python/NumPy/Torch CPU and applicable CUDA RNG states, sampler
  state/seed and ordered-index identity;
- train/validation/test logical cursor identities as relevant,
  highest contiguous **committed** train window, any partially
  consumed gradient-accumulation state, and outstanding/in-flight
  prefetch identities;
- cumulative counters, document/group/source exposure ledger,
  ordered span/EOS/target provenance digests, C1-S1 assignments
  and C2/C3/C4 policy checksums;
- run configuration, code commit/runtime manifest and future
  fail-closed hardware allocation identity.

**Commit boundary must be specified before implementation**: the
data cursor cannot be durably advanced separately from the
model/optimizer state that consumed those targets. If checkpoint
publication happens only every N optimizer updates, a restart may
replay **all uncheckpointed updates**, not merely the last prefetched
window; exact replay requires restoring corresponding model and
RNG state. An independently persisted data-only ack at each
window can race the training-state commit and skip or repeat learning.
The final implementation must resolve this, not reuse P1.3a's
per-window ack as a production optimizer transaction.

Proposed persistence envelope: local ext4 staging/checkpoints,
checksum-verified immutable generation manifests, write temporary
files, `fsync`, atomic same-filesystem rename and parent-directory
`fsync`, then publish verified immutable results outside active
storage if approved. Handle interrupted writes, mismatched hashes,
corruption, source drift and repeated/partial accumulation fail
closed. Declare rollback and retention separately.

**Observation, not a chosen checkpoint policy:** P1.3b's single
largest synthetic Orion run (8,192 artificial documents, 1,056,768
stream tokens) measured 23,960 KiB peak process RSS, 6,943,291
bytes of local index and 6.924444 s total time. The ack+`fsync`
component took 2.415402 s out of 5.546933 s read+ack time.
That is evidence of per-window cursor persistence overhead in the
prototype, not evidence for selecting batched acknowledgments or
a training checkpoint interval.

## 7. Storage, memory, monitoring and hardware policies

All **resource budgets are OPEN**. A later review must estimate and
measure, separately:

1. active corpus/reference index (document counts, index overhead,
   payload storage, metadata, SHA-256 manifests and materialization
   temp space), avoiding extrapolation from 128-token artificial docs;
2. peak CPU RSS for tokenizer, decompression, span provenance, bounded
   reordering queues and per-worker prefetch with a fixed cap;
3. throughput of whole-document tokenization, 4096/4097 window
   occupancy, storage read amplification, IOPS and local-ext4
   checkpoint `fsync`/atomic-rename overhead;
4. model/optimizer/gradient/scaler checkpoint size, retention
   generations, temp duplication margin and rollback time;
5. failure alerts and stop conditions: inconsistent identities,
   rights status, split contamination, non-finite losses,
   all-ignored targets, token counts, disk-space margin,
   data checksum drift, GPU allocation and process isolation.

Local ext4 is the approved *active-work-path principle*; NFS is
not an active checkpoint or index write path. Real storage capacity,
resource reservation and exact retention numbers must be rechecked
under a future authorized operational gate.

**GPU constraints inherited, not newly approved:** any future
GPU proposal must independently revalidate physical GPU0 mapping,
lock/VRAM thresholds and isolation from GPU1 (the live Qwen/vLLM
service). **There is NO GPU authorization in C4-P2.**

## 8. Required verification matrix (for future separately authorized gates)

| Test class | Acceptance criterion / falsification |
| --- | --- |
| Split/rights | Reject any document missing rights approval or contradicting frozen C1-S1 source/group assignments; no cross-split batches or stream windows |
| Sampling determinism | Same frozen view/config/seed -> same ordered reference and quota ledger across shard permutations, buffer choices and interrupted schedules |
| Quota and whole-doc behavior | Reconcile intended vs actual counts in the declared unit; expose shortfalls/overshoot and unused records; no silent truncation or resampling |
| Long/empty documents | Contiguous offsets, exactly one EOS even if empty, no phantom EOS at shard or fragment boundaries; equivalence with whole-document C2 tokenization |
| P1 geometry | Every scored target owned once, 4096/4097 overlap correct, padding/ignore_index safe, per-split window provenance complete |
| Epoch boundaries | Independently validate E1 versus E2 expected target counts/attention behavior before selecting one |
| Validation/test | Same stable holdout views and references independent of train sampler, epoch, seed and A/B/C experiment |
| Cursor failure replay | Kill/restart at document, window, ack, shard, temp/checkpoint rename and source-exhaustion points; committed positions monotonic |
| Training-state rollback | Resume model/optimizer/RNG/data together at same update boundary; identical synthetic update provenance, compared under fixed supported runtime; never assume cross-kernel bitwise equivalence |
| Resource ceilings | Measured peak RSS, local disk occupancy, throughput, queue caps and checkpoint growth on authorized fixture; fail closed on capacity/identity breach |
| Observability | Independently validate token/EOS/target/window/update/exposure counters and final checksum chain versus manifests |

No row is marked PASS by this planning document. Real-data tests,
rights decisions, optimizer operations and GPU experiments each need
independent scientific authorization.

## 9. Decision register and proposed staged follow-ups

| Issue | Status | Review needed |
| --- | --- | --- |
| C1 source rights/use | **BLOCKED / C1 IN PROGRESS** | Clearance or explicit disallow rule per source and usage |
| Mixture A/B/C | **OPEN / C1-dependent** | Scientifically choose mixture or experiment design after rights review |
| Quota unit Q-W vs Q-T | **OPEN** | Decide objective, exact counters, weighting and tolerances |
| Document admission Q-D1 vs other | **OPEN** | Whole-document quota remainder/exhaustion and long docs |
| Source merge S1 vs S2 | **OPEN** | O1-compatible deterministic interleaving and tie-breaks |
| O1 initial ordering architecture | **APPROVED IN P1.3** | Production namespace, seed, serialization and view identities still OPEN |
| Epoch policy E1 vs E2 | **OPEN** | Cross-epoch context, BOS/reset, partial windows and target counts |
| Reuse policy R0 vs R1 | **OPEN** | Repeats, stop conditions, source exhaustion and training-token budget |
| Tokenizer long-doc equivalence | **OPEN; C1-ACCESS BLOCKED** | Independent valid method/test; do not tokenize C1 in this gate |
| Loss/target attribution | **OPEN** | Define counts by target document/source with cross-source transitions |
| Batch/optimizer/schedule | **OPEN** | Scientific training recipe, no GPU permission |
| Ack/checkpoint cadence | **OPEN** | Atomically bind sampler cursor to future training state |
| RSS/disk/retention budgets | **OPEN** | Measurements, margins and rollback budget; P1.3b only synthetic |
| Production loader implementation | **NOT AUTHORIZED** | New scientist decision after C1 rights and contract review |
| Any production data, GPU or training | **NOT AUTHORIZED** | Distinct gates C4-D/C4-T, C3.7, C5 and C6 |

**Proposed next design review (not approved automatically):**
C4-P2.1 — settle the *interface-level* sampling/exposure ledger,
the quota/epoch option trade-offs and the atomic checkpoint-ack
contract, initially on paper. C1 rights and mixture choice may proceed
in parallel under their own scientist-governed gate. Later synthetic
implementation/tests, real-data processing and training approvals
must remain independent decisions.

## 10. Gate and stop conditions

**C4-P2: DESIGN PROPOSED / SCIENTIST REVIEW PENDING.**
**C4 overall: IN PROGRESS. C1 rights and mix: IN PROGRESS.**
**C3.7, C4 production data reads/materialization, C5, C6,
GPU and training: NOT AUTHORIZED.**

Scientific approval of this design in the future would **not**
itself permit implementing or running a real-corpus pipeline.
No real corpus text was opened, read, sampled or tokenized; no
training code, CUDA, NVML or GPU activity was performed under
this C4-P2 planning authorization.
