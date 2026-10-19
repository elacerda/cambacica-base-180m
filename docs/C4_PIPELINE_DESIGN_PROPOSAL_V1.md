# C4 — Data pipeline and training-system design proposal v1

**Status: DESIGN IN PROGRESS / P1.2 AND P1.3a SYNTHETIC CPU PASS / P1.3 O1 ARCHITECTURE APPROVED (2026-10-19).**

This document is a **planning artifact**, not a frozen C4 policy,
authorization to process real corpus text, or authorization to train.
The scientist authorized C4 planning after C3 completed and subsequently
approved **P1 (contiguous EOS packing)** as the initial approach, with a
bounded synthetic-only CPU reference and tests. The detailed decision and
acceptance contract are in [C4_P1_PACKING_DECISION_V1.md](C4_P1_PACKING_DECISION_V1.md).
All production data processing, local GPU execution, real-data training,
C3.7, C5 and C6 require their own explicit gates/approvals.

## 1. Inputs and inherited decisions

| Source | Pinned or previously approved identity | C4 consequence |
| --- | --- | --- |
| C1 normalized, exact-deduplicated corpus | 21,603,689 post-D1 retained documents; no further near-dedup or BD4 deletions | Never redefine the eligible population or silent filtering |
| C1-S1 split | Immutable document/group assignments; split manifest SHA-256 `8b937f90d40487c643fe48baa6b5c2950713fe2e0d2d534f3e2a3c08e74dbe16`; assignments SHA-256 `ef52e3cf3033934df13e3d16c207e1783c509dff6fb91954fb8c25ab719ece1e` | Train/validation/test provenance must remain disjoint; all tokens of a document inherit its parent split |
| C1 mixtures A/B/C | Provisional YAML configurations based on **normalized-word shares**, not model-token shares | Do not silently select B, convert shares to tokens, or choose final mixture |
| C2 | Frozen SentencePiece Unigram v24576, model SHA-256 `c8982a23d3c5ecc0a61d300edc1e354e38f1358aa3db5420d5bf19f7b95cbf23`; special IDs unk 0, bos 1, eos 2; pad -1 | Tokenization must use exact model and explicitly define special-token and padding behavior |
| C3 | Architecture C, 9×1024 / 16 heads / SwiGLU4096 / 176,180,224 tied parameters; context 4096; BF16 reference | Model shape fixed, other training choices still unresolved |
| C3.6 | GPU0-only C3.6 engineering measurements and deterministic synthetic restart passed | 4×1 is a **provisional engineering candidate**, not the approved batch, optimizer or training plan |

C1 remains **IN PROGRESS** on usage/rights and final mixture. The C1
normalization contract is immutable, and its `normalized_words` metric
(`len(text.split())`) **is not** a tokenizer-token count. Long documents
are retained in their original parent split and must be segmented without
moving any segment into a different split. Work with source provenance,
document identifiers, original coordinates, and immutable checksum pins.

The C3 synthetic wall throughput cannot be used as a measured real-corpus
end-to-end rate or scientifically justified training duration.

## 2. Proposed C4 work packages and gates

### C4-P — Contract design and CPU-only synthetic prototypes

**Completed, separately authorized synthetic-only CPU work:** P1 reference
packing, P1.2 masked-loss integration and P1.3a deterministic O1
SQLite-indexed streaming/cursor prototype and their tests. These
authorizations are limited to artificial tokens and metadata and
do **not** permit C1 production corpus reads, real tokenization,
production token shards, GPU workloads, training or other production
C4-D components.

The pure-Python P1 reference is in `src/cambacica/packing/p1_reference.py`;
its 24 focused tests previously passed. **C4-P1.1 initial EOS/BOS/loss
policy is scientist-approved**, as recorded in
[C4_P1_1_SEQUENCE_LOSS_CONTRACT_V1.md](C4_P1_1_SEQUENCE_LOSS_CONTRACT_V1.md).
**C4-P1.2 synthetic CPU loss integration is COMPLETE / PASS**:
51 focused tests passed after formatting, with an earlier complete
CPU suite of 425 passed / five skipped, documented in
[C4_P1_2_CPU_LOSS_INTEGRATION_V1.md](C4_P1_2_CPU_LOSS_INTEGRATION_V1.md).
This remains a bounded synthetic probe, not production model training.

**C4-P1.3 initial deterministic streaming architecture is now
scientist-approved as a design baseline** in
[C4_P1_3_STREAMING_SAMPLING_DESIGN_V1.md](C4_P1_3_STREAMING_SAMPLING_DESIGN_V1.md).
The choices are **O1 external hash ordering with a disk-backed
index**, **one logical producer with bounded deterministic
emission**, and a **cursor at the last contiguous acknowledged
consumer window**. The bounded synthetic CPU implementation **C4-P1.3a is COMPLETE / PASS**,
with 49 focused Orion passes, 450 complete-suite passes and five skips,
and successful Ruff lint and post-format checks; see
[C4_P1_3A_SYNTHETIC_CPU_IMPLEMENTATION_V1.md](C4_P1_3A_SYNTHETIC_CPU_IMPLEMENTATION_V1.md).
The approved O1 baseline is **not** a completed production sampler.
Epoch/reset BOS, production seed/order-key serialization, source quotas,
C1 mixture, epoch/repeats, data-loader/optimizer checkpoint coupling
and final batch remain undecided.

1. Produce versioned **data interface contracts** for
   `document -> tokenized document -> sequence -> training batch`. Every
   emitted segment must trace to a document/group, split, source/subset,
   revision and hash; mixed-document sequences need auditable span-level
   provenance.
2. Fix a sequence/accounting policy for model `context_length=4096`:
   exactly 4,096 input IDs and 4,096 next-token targets, with a
   4,097-token underlying window **or an equivalent formally proven
   non-duplicating boundary/overlap policy**. Report exact unique
   consumed tokens, scored labels and special-token overhead separately.
   Never equate these counts to normalized words.
3. Describe the handling of text normalization (no renormalization),
   EOS/BOS, empty/one-token documents, short fragments, long documents,
   trailing partial windows, masked labels, attention and padding.
   The frozen tokenizer has `pad_id=-1`, which is **not** a valid embedding
   input ID and must never be fed to the model.
4. Produce deterministic sampling/shuffling contracts at the
   `(document/group, epoch, segment, source, split)` level.
   Frozen split isolation is a hard invariant. Sampling by source must
   be auditable and check quotas in the **declared** unit. Reuse/epochs,
   replacement, exhaustion and same-document adjacency require policy.
5. Plan local ext4 staging, bounded queues, manifests, checksum and
   restart pointers; avoid NFS as the active write path.
6. Specify CPU-testable correctness, contamination and rollback tests.

### C4-D — Data pipeline implementation

**NOT AUTHORIZED by C4 planning.** After scientist acceptance of C4-P,
implement independently reviewable tokenizer/packing/dataloader modules,
first on tiny **synthetic text** only. Any read of C1 production text,
generation of real token shards, or empirical real-corpus packing
occupancy requires its own explicit permission after rights review.
No train/validation/test merging or corpus policy mutation.

### C4-T — Training recipe and execution protocol

**NOT AUTHORIZED by C4 planning.** Prepare versioned alternatives for
optimizer, parameter/gradient/optimizer-state dtypes, LR and warmup,
weight decay, gradient clipping, global/effective batch, token budget,
checkpoint cadence/retention and activation checkpointing. Fix the
learning-rate schedule in **tokens seen**, or state precisely how update
counts map to tokens under the selected batch.

Candidate `microbatch=4, accumulation=1, effective tokens/update=16,384`
has measured synthetic throughput ~105,095 tokens/s, with ~18,204 MiB
sampled process peak. The higher 4×4 cell measured ~110,674 tokens/s
with ~19,740 MiB. Neither configuration is approved as the
final training batch. Compare the measured engineering tradeoff to the
future scientifically acceptable convergence behavior; do not silently
inherit C3.4's AdamW settings.

### C4-V — End-to-end reproducibility and exit review

**NOT AUTHORIZED by C4 planning.** Require a bounded real pipeline
validation plan covering data lineage, EOS/packing and loss correctness,
throughput including tokenization/I/O, deterministic sample order,
checkpoint/optimizer/scheduler/dataloader state restoration, operational
storage, failure injection, and independent verifier evidence. A separate
scientist authorization is required for any actual training/GPU run.

Only after all authorized C4 work packages and C1 prerequisites are
satisfied may C4 be proposed for `COMPLETE`.

## 3. Explicit model compatibility issues requiring a choice

As of C3 commit `9233cd506acafccd5cc491eb6f34adfe7febde5e`,
`src/cambacica/model/transformer.py`:

- calls SDPA with `attn_mask=None, is_causal=True`;
- generates RoPE positions from each tensor's sequence index;
- exposes `forward(input_ids, labels)` without packed-document attention
  masks, custom position offsets or a separate loss mask;
- calculates CE over the full supplied labels tensor (PyTorch default
  `ignore_index=-100`, not an explicitly validated packing contract).

Thus the pipeline cannot claim *document-isolated* attention or RoPE
position resets without an explicit architecture-compatible modification
and accompanying correctness/performance review. Possible **design
alternatives, not approvals**:

- **P1, contiguous EOS stream:** concatenate complete documents within
  a single split with EOS delimiters; permit causal attention across
  document boundaries; preserve span-level provenance. Efficient and
  compatible with C3 model as-is, but document contexts interact.
- **P2, boundary-isolated packing:** block cross-document attention and
  reset positions, requiring explicit attention-mask/position-ID
  support and confirmation that kernel choice and efficiency remain
  appropriate. Do not assume Flash SDPA continues unchanged.
- **P3, document-only windows:** never join documents, segment long
  ones, pad shorter sequences with safe valid input IDs and ignore
  masked target positions. Simpler leakage semantics but can waste
  compute and requires a tested loss-mask and padding contract.

The scientist selected **P1 as the initial baseline** on 2026-10-19.
P2/P3 remain documented alternatives for future controlled comparisons,
not selected production paths. The detailed P1 CPU-reference contract and
unresolved boundary/sequence decisions are in
[C4_P1_PACKING_DECISION_V1.md](C4_P1_PACKING_DECISION_V1.md).
For P1, an EOS token is a separator, not proof of attention isolation.
For P3, choose explicit masking and avoid inserting `pad_id=-1`
into embeddings. Empty retained documents must be represented in
accounting rather than silently removed from C1.

## 4. Exact data/provenance contracts to design

Define a versioned manifest with:

- source dataset/ref + source/subset + C1 revision and rights status;
- immutable C1 original document coordinates, group key, content hash,
  and C1-S1 assignment identity;
- immutable tokenizer ID, hash, vocabulary and special-token policy;
- source and split, sampling epoch, random seed, rank and dataset view
  checksum; explicit draw ordering and whether repeats are permitted;
- segment index and token range in parent document;
- packed sequence ID and ordered source spans, EOS/BOS offsets,
  pad/ignored-label counts, number of scored labels and utilization;
- output shard SHA-256, schema, producer code commit and clean/dirty
  state, and independent verification status.

Reject unknown source/split, hash mismatches, ID collisions, duplicate
segment emission when forbidden, dropped target tokens, out-of-range
token IDs, overlong sequences and malformed manifests. For held-out
splits require stable, deterministic evaluation references; do not
sample validation/test with training mixture policies by default.

## 5. Checkpoint and training-state protocol

Unlike C3.6's synthetic checkpoint, C4 must plan the production state
boundary: model, optimizer, scheduler, scaler (if any), RNG (Python,
NumPy, Torch CPU/CUDA), sampler and dataloader cursor, completed
optimizer update, tokens and scored-label counters, corpus/view/shard
hashes, code/runtime configuration and GPU identity. Define atomic
local save, integrity check before deserialize, immutable report
publication, resume verification and error recovery. Distinguish
data-stream determinism, numerical determinism and hardware/software
portability. Do not promise bitwise equality across different kernels
or data-loader implementations without measurement.

## 6. Training-run operational envelope (for future approval only)

- Target host `cenpesor`, physical GPU **0 only**,
  UUID `GPU-7562e34d-b0a2-a725-2ad5-ed1327be3cb0`,
  PCI `0:27:00.0`. Physical GPU 1 (Qwen service) strictly prohibited.
- Future GPU tools must preserve verified CUDA/NVML mapping, fail-closed
  GPU identity and process memory checks, local exclusive lock,
  foreground execution, with no fallback device/retry.
- C3.6 engineering controls: PyTorch allocator limit 35,840 MiB,
  sampled process initial peak target 36,864 MiB, hard budget
  40,960 MiB, minimum GPU free-memory margin 4,096 MiB.
  A future training protocol must explicitly reapprove or strengthen
  these; do not imply C3.6 approval extends to new GPU workloads.
- Local ext4 for active staging/checkpoints, NFS for immutable finalized
  manifests/results only. Plan byte/IO/retention budgets from measured
  data, not from extrapolated normalized words.

## 7. Decision register (all currently open)

| Decision | Current status | Evidence/requirement before approval |
| --- | --- | --- |
| Packing P1/P2/P3; attention across documents | **P1 APPROVED** | Contiguous EOS stream, causal cross-document attention allowed; low-level semantics remain to review |
| Initial BOS/EOS and loss/padding/mask semantics | **C4-P1.1 APPROVED; C4-P1.2 CPU PASS** | No per-document BOS; score EOS and cross-document targets; labels `-100` for pad; bounded synthetic model integration verified |
| 4,096 input/label alignment, tail/overlap | **P1.1 POLICY APPROVED; SYNTHETIC STREAMING PASS** | 4,097 backing / 4,096 stride; P1.3a synthetic SQLite streaming equivalence tested; production streaming not verified |
| Mixture A/B/C and source quotas | BLOCKED BY C1 | Final rights/use policy and mixture-selection gate |
| Mixture weights expressed in actual model tokens | OPEN | Distinguish from C1 normalized-word shares |
| Training microbatch/accumulation | OPEN | 4×1 engineering candidate only |
| Optimizer, LR, warmup and schedule | OPEN | Scientist-approved future recipe |
| Effective training-token budget and repeats | OPEN | C1 capacity, tokenizer-token census and science |
| Parameter/state precision and activation checkpoint | OPEN | Future training proposal and measured memory |
| Checkpoint cadence, persistence and retention | OPEN | Local ext4 capacity and recovery requirements |
| GPU training/pilot authorization | NOT GRANTED | Separate explicit scientist approval |
| C4-P P1 synthetic CPU reference/loss integration | **COMPLETE / PASS (P1.2)** | No corpus, CUDA or training; see P1.2 record |
| C4-P1.3 deterministic streaming design | **INITIAL ARCHITECTURE APPROVED** | O1 external disk index, single logical producer and acknowledged-window cursor; production sampling open |
| C4-P1.3a synthetic CPU streaming and replay | **COMPLETE / PASS** | 49 focused PASS; complete CPU suite 450 PASS / 5 SKIP; formatting/lint PASS; no real data, optimizer, CUDA or training |

## 8. Proposed C4-P exit criteria

A versioned scientist-reviewed design for packing and data isolation,
exact tokenizer and sequence contract, deterministic sampling,
training-state interface, full validation/failure matrix, resource
budget and explicit open decisions. Reviewer signoff is required
**before** implementing or using real corpus production paths.

**Gate status:** C4 **IN PROGRESS — P1.2 AND P1.3a SYNTHETIC CPU COMPLETE / PASS; PRODUCTION SAMPLING, C1 RIGHTS AND TRAINING RECIPE REMAIN OPEN**, not C4 PASS.
**Hard stop:** no production-token shards, real-corpus training, GPU run,
C3.7 pilot, C5 smoke run or C6 pretraining under this authorization.
