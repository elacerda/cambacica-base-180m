# C4-P1 — Scientist-approved contiguous EOS packing: design and CPU test contract v1

**Decision date:** 2026-10-19.

**Scientific decision: P1 APPROVED as the INITIAL PACKING BASELINE.**
Approval applies to **contiguous, EOS-delimited token packing within one
immutable C1-S1 split**, using causal attention that may cross document
boundaries. It does not select corpus mixture A/B/C, sampling weights,
optimizer, effective training batch or a production training pipeline.
P2 document-isolated packing and P3 document-only windows remain
possible comparison alternatives, not currently selected.

**Status update (2026-10-19):** P1 approach **APPROVED** and its initial
sequence/BOS/EOS/loss policy **C4-P1.1 SCIENTIST APPROVED**. The versioned
[C4-P1.1 sequence/loss contract](C4_P1_1_SEQUENCE_LOSS_CONTRACT_V1.md)
supersedes provisional statements about BOS and EOS-loss scoring below.
C4 overall remains **IN PROGRESS**.
The scientist's approval of the P1 approach in response to the request
to continue detailed design and CPU tests authorizes a bounded,
**synthetic-only CPU reference prototype and its tests** for C4-P.
No real corpus files, tokenization of real documents, GPU workloads,
C3.7/C5/C6 training or deployment are authorized.

## 1. Frozen inherited identities

- Frozen split: C1-S1, manifest SHA-256
  `8b937f90d40487c643fe48baa6b5c2950713fe2e0d2d534f3e2a3c08e74dbe16`;
  assignments SHA-256
  `ef52e3cf3033934df13e3d16c207e1783c509dff6fb91954fb8c25ab719ece1e`.
  All segments inherit their parent document's split and group. The same
  train/validation/test token stream is NEVER shared between splits.
- Frozen C2 tokenizer: `unigram-v24576`; model SHA-256
  `c8982a23d3c5ecc0a61d300edc1e354e38f1358aa3db5420d5bf19f7b95cbf23`;
  vocabulary size 24,576; BOS 1; EOS 2; UNK 0;
  pad ID **-1 (must never be an embedding input)**.
- Frozen C3 architecture C: tied 176,180,224 parameters, 4,096
  causal input positions, attention `attn_mask=None, is_causal=True`,
  RoPE positions starting at each sequence window.
- The `configs/corpus_mix_{a,b,c}.yaml` shares are expressed in
  **normalized words**, not tokenizer tokens. Mixture selection, token
  weights, repeat policy and final effective batch remain open.

## 2. P1 semantics: approved approach vs proposed implementation defaults

**Approved P1 behavior:** append a document boundary EOS between
successive documents of the **same split**, producing a continuous causal
stream. Cross-document attention is permitted and RoPE positions are
not reset at each document boundary. An EOS separator **does not**
implement attention isolation.

**Historical CPU reference proposal.** For initial P1 packing/sequence/loss
policy now approved on 2026-10-19, see
[C4-P1.1](C4_P1_1_SEQUENCE_LOSS_CONTRACT_V1.md); other research choices
remain open:

1. Input is a finite ordered list of **artificial tokenized documents**.
   Each has `document_id`, `group_id`, `split`, `source/subset`,
   `content_hash` (synthetic fixture metadata), and zero or more
   payload token IDs in `[0,24576)`. Explicitly reject `-1`,
   unknown splits, malformed IDs and duplicate document IDs within a
   candidate view. No automatic filesystem/corpus discovery.
2. Prototype EOS rule: emit the C2 EOS ID **2 after every document**,
   including empty documents. Retain empty documents and EOS in
   the accounting/provenance ledger, even if an empty-only split has
   too few tokens to produce a supervised example.
3. Initial C4-P1.1 BOS/EOS rule (now **APPROVED**): do **not**
   auto-insert BOS per document; default reference does not insert a
   split-start BOS. **Score inserted EOS** when it has a preceding
   stream token, and score valid cross-document transition targets.
   Epoch-boundary BOS, tokenizer-provided specials and resampling remain
   future integration decisions. The reference's configurable alternative
   BOS/EOS options remain experimental, not approved production policies.
4. Split streams are constructed **independently**, with no adjacency,
   carry, refill or position overlap between train, validation and test.
   Within a split, EOS-delimited documents may share context.
5. The prototype must retain span-level provenance mapping
   `[stream_start, stream_end)` to each document's payload and separate
   EOS offset; also record document count, empty count, payload tokens,
   EOS tokens, stream tokens and split-specific stream SHA-256.
   Non-empty and empty input records must be traceable; nothing should
   silently disappear.

## 3. Exact next-token window geometry

For each independent split-specific stream
`s[0], ..., s[L-1]` of **actual tokens (including inserted EOS)**,
the available next-token training pairs are exactly

`(s[i], s[i+1])`, for every `0 <= i < L-1`.

For target context length `C=4096`, prototype windows begin at
`k*C` (k = 0, 1, ...), and each window owns a **non-overlapping set
of target indices**:

- `input_ids[j] = s[k*C+j]`
- `labels[j] = s[k*C+j+1]`
- for valid `j` satisfying `k*C+j+1 < L`.

This is equivalent to slicing a **4,097-token backing window** for
each full `C=4096` input/target pair, moving forward by 4,096
**target** positions per window. The token at a window boundary is
reused as an input after having been a target in the previous window;
that is **intentional autoregressive continuity**, not double counting
as an owned target.

For a final partial window, input padding must use a **valid safe
embedding token ID** (e.g. EOS 2), while all trailing labels must be
`ignore_index=-100`; those padding positions must not count as
scored labels or new corpus tokens. No `pad_id=-1` may reach the
embedding input. A causal mask alone does not remove the need to
mask the trailing labels.

Exactly `max(L-1, 0)` training targets exist per split, unless a
separate *document-boundary loss masking* decision is approved later.
In this proposed prototype, EOS and cross-document transition targets
are eligible: the first payload token of a new document may be
predicted given a preceding EOS/document context. This is one reason
P1 and P2 have different semantics. The first stream token has no
within-split predecessor and is not a scored target under the no-BOS
proposal; record this explicitly.

**Proposed measurement fields** (separate from corpus normalized-word
counts): `payload_tokens`, `eos_tokens`,
`real_stream_tokens`, `unique_scored_targets`,
`first_stream_token_unscored`, `window_count`,
`input_pad_slots`, `ignored_label_slots`,
`scored_label_fraction`, `source_token_shares`,
`split_id`, `stream_digest`, `ordered_span_digest`.
If windows overlap in the underlying backing stream, deduplicate
*target ownership*, not the model's required context token.

## 4. Boundaries, provenance and sampling

- Long documents are segmented across windows without moving to
  another split, truncating or silently removing their tails. Store
  per-window source span offsets and document-relative token ranges.
- Empty documents produce an EOS in the proposed stream but never
  get filtered from source/document counts. Extremely short streams
  may yield zero supervised targets; this is an explicit accounting
  outcome, not hidden removal.
- Preserve `source/subset`, C1 `occurrence-id-v2` document identity,
  original coordinates, group and assignment provenance in any
  future adapter. For synthetic fixtures, use obviously artificial
  IDs and no C1 production paths.
- The ordered synthetic stream is deterministic for a fixed ordered
  fixture. **An input order is not a scientific sampling policy**:
  source mixture A/B/C, deterministic shuffle by seed, without/
  with-replacement exposure, sample quotas and train epoch definitions
  are undecided and require separate C4/C1 proposals.
- Held-out validation/test references must be independently stable,
  not resampled by whichever training mixture is later selected.
- No packer may use train and held-out segments within the same
  sequence. Fail closed on unknown/mixed split and on invalid token IDs.

## 5. CPU-only acceptance test matrix

The **pure Python reference implementation** is
`src/cambacica/packing/p1_reference.py`, without PyTorch,
CUDA, NVML, network calls, production Parquet access, tokenizer
instantiation, real tokenization or training. Its input consists only
of explicit fake token IDs and fake provenance identifiers; it should
be possible to unit test with `CUDA_VISIBLE_DEVICES=''`.

The module fixes the window context at 4,096 and the vocabulary at
24,576, inserts EOS after every input document, preserves per-document
payload ranges and separate EOS offsets, and constructs one stream for
each of train, validation and test. It owns each candidate target
position once, pads only inputs with the valid EOS ID, and writes
`-100` to padded labels. The default provisional policy omits BOS and
scores inserted EOS targets; both choices are explicit options. The
reference does not compute a model loss. Its `scored_pairs()` iterator
omits `-100` labels, so a loss consumer must use `ignore_index=-100` to
give those positions zero loss contribution.

The following CPU tests are required:

1. Hand-worked two/three-document EOS example; assert exact stream
   tokens, ordered spans, EOS offsets and cross-document **causal**
   transition target policy.
2. No split leakage for interleaved artificial train, validation and
   test documents, including deliberately repeated payload token IDs.
3. Exact next-token input/label alignment for two full windows plus
   one partial tail, with each target index owned **once**.
4. Nontrivial boundary at 4,096/4,097: no missed or duplicated
   targets, and exact `L-1` scored-label accounting.
5. Empty-only and one-token document cases, repeated EOS and
   trailing partial window accounting; preserve zero-target inputs
   in the document ledger.
6. Long-document continuity and provenance offsets across windows.
7. Reject `pad_id=-1`, negative/out-of-vocabulary/non-integer
   payload IDs, duplicate document IDs and unknown splits.
8. Safe padding: all input IDs in `[0,24576)`; padded labels
   exactly `-100`; zero loss contribution claimed for ignored labels.
9. Deterministic reproducibility: same artificial ordered input
   has exactly identical windows, ledger, canonical digests and
   counters; a changed document/order/split changes identity.
10. Explicit fail-closed behavior on malformed provenance and
    unmatched totals; verify no files are written outside the
    test's temporary directory (prefer an in-memory implementation).

Report the test inventory and observed pass/skip counts. Preserve
the existing C1/C2/C3 fixtures and all frozen artifacts. The initial sequence and loss semantics were explicitly approved as
**C4-P1.1** (see linked contract above). This does not validate integration
with the PyTorch model or authorize a production packing/data pipeline.

**Implementation and CPU validation record (2026-10-19):** all ten
acceptance areas above are covered by 24 focused tests. The focused
suite passed **24/24**. The full available suite passed **379 tests**
with **10 skips**: three optional C3/PyTorch modules, five network smoke
tests disabled by default, and two tokenizer tests skipped because
SentencePiece is unavailable. Tests ran with
`CUDA_VISIBLE_DEVICES=''`; no CUDA/NVML runtime, production corpus
payload, or training run was used. Ruff and formatting checks pass for
the changed Python files. Repository-wide Ruff still reports five
pre-existing E402 findings in two untouched optional-PyTorch test
files, and repository-wide formatting reports 13 untouched files that
would be reformatted.

## 6. Research questions still OPEN

- **C4-P1.1 decided:** no BOS automatically per document, score EOS
  and valid cross-document next-token targets, use explicit
  `ignore_index=-100` for padded labels. **Still open:** future
  epoch/reset BOS rules and prevention of tokenizer double specials;
  independently validate masked cross-entropy integration on CPU.
- Confirm full-window and trailing-partial policy for efficient
  production batching; evaluate padding efficiency on a rights-cleared
  view only after separate approval.
- Define source quotas in a declared unit after C1 and token-count
  measurement, rather than assuming words=tokens; decide C1 mixture.
- Determine ordered sampling, random seeds, exposure/repeats,
  long-document weighting and held-out stability.
- Determine final microbatch/gradient accumulation, optimizer,
  learning rate, schedule, token budget, gradient clipping, checkpoint
  cadence and retention, activation checkpointing.
- Re-test stateful dataloader resume separately from the **synthetic
  model-only** C3.6 restart test.

**Stop line:** C4-P1 approval is *not* C4 completion. No C1 source
rights have been resolved by this decision; C3.7, C4 real-corpus
execution, C5, C6, and any GPU work remain unauthorized.
