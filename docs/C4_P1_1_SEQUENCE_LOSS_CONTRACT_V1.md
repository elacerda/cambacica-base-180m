# C4-P1.1 — Initial sequence, BOS/EOS, and loss contract

**Scientist approval: 2026-10-19.** **Status: APPROVED as an INITIAL P1 sequence/loss policy**, not an approval of C4 as a whole, not a completed model integration test, and not a final training recipe.

**Basis:** scientist explicitly approved the C4-P1.1 choices presented
after review of synthetic P1 reference commit
`4e0fcf9f14729e9bf27bffadc3e994799078db65`.
This decision refines [C4 P1](C4_P1_PACKING_DECISION_V1.md)
without changing frozen C1/C2/C3 scientific identities.

## 1. Policy approved for the initial P1 contract

1. **One EOS per document:** concatenate tokenized documents assigned to
   the **same** C1-S1 split, append C2 `EOS=2` after each document
   (including empty documents), and record per-document provenance.
   Train, validation and test streams remain strictly independent.
2. **EOS is a valid prediction target:** score every *inserted* EOS that
   has a predecessor in its split stream. EOS is not automatically
   masked just because it marks a document boundary. The first stream
   token has no within-stream predecessor and is never a target.
3. **No automatic BOS per document:** do not prepend `BOS=1` at each
   document start. Initial P1 reference configuration uses
   `insert_bos_at_split_start=False`; **the scientist did not
   separately select** a policy for later epoch/reset boundaries,
   and no new BOS insertion is authorized here. Tokenizer adapters
   must explicitly control automatically inserted special tokens
   and reject/flag double insertion before production use.
4. **Causal continuity:** within a window, allow standard causal
   attention from a later document into earlier documents of the
   **same split**; EOS denotes the boundary but does not isolate
   attention. Do not reset RoPE at each EOS; the C3 model's position
   indices restart at each *window*, not each document.
5. **Window geometry:** exactly `C=4096` input positions and
   `C=4096` aligned next-token label positions. Full windows use
   a 4097-token backing slice, advance 4096 stream positions per
   window and own disjoint target indices. At a boundary, a token
   can be reused as next-window input without being counted twice
   as an owned target.
6. **Padding and masking:** on a final partial window, pad **input**
   positions with a valid C2 embedding ID (initial reference: EOS 2);
   pad unavailable **labels** with `ignore_index=-100`. Do not pass
   tokenizer `pad_id=-1` to embeddings. **No automatic loss masking
   is applied at normal document boundaries**: the first token
   after an EOS is a valid next-token prediction target when it has
   a predecessor. Ignored labels contribute neither CE loss nor
   scored-label accounting.
7. **Cross entropy contract:** the eventual consumer of P1 batches
   must request `ignore_index=-100` **explicitly** in PyTorch
   cross-entropy and compare numerical results to the reference
   scored-label set. `src/cambacica/model/transformer.py` currently
   inherits that value as PyTorch's default, but this implicit
   behavior is **not** the completed C4-P1.1 integration.
8. **Accounting:** for a split stream of `L` actual tokens
   including emitted EOS and any future separately approved BOS,
   `candidate_target_count=max(L-1,0)`. Under the approved initial
   `score_eos_targets=True` and no other target exclusions,
   `unique_scored_targets=candidate_target_count`. Report actual
   stream tokens, payload tokens, inserted EOS count, scored labels,
   pad slots and ignored label slots separately; never equate
   `normalized_words` with model tokens.
9. **Zero-target cases:** an empty stream or single-EOS-only stream
   has no supervised pairs and must remain in provenance/accounting,
   not be silently discarded from the document population.
   A future loss consumer must not feed an all-`-100` target batch
   into a mean-reduced cross-entropy and claim a finite training
   loss; skip or explicitly handle such windows under a tested
   policy.
10. **Split safeguards:** every emitted token and window must be
    attributable to document/group/split; source text and token
    streams from different C1-S1 splits must never share a packed
    sequence or attention context. The selected mixture and
    sampling/replacement cadence remain future decisions.

## 2. Relationship to the existing reference

The already implemented pure-Python reference uses policy defaults
`insert_bos_at_split_start=False`, `score_eos_targets=True` and
`safe_input_padding_id=EOS_ID`. These are consistent with the initial
approved policy. The reference's other configurable BOS/EOS branches
remain **experimental**, not additional scientist-approved production
policies.

The 24 focused P1 tests and reported available CPU suite
(379 passed, 10 skipped) cover **reference correctness only**.
The scientific approval in this document **does not certify**
integration with the PyTorch C3 model, an actual masked CE evaluation,
or long-running training correctness.

## 3. Follow-on C4-P1.2 CPU integration acceptance proposal

**Status update (2026-10-19): implementation explicitly authorized by the scientist, now committed; the actual CPU test execution remains PENDING.** See [C4-P1.2 integration record](C4_P1_2_CPU_LOSS_INTEGRATION_V1.md).

Build a bounded synthetic-only CPU test between the approved P1
reference and the existing C3 `CambacicaCausalLM.forward`:
- verify exact `input_ids`/shifted `labels` alignment, `torch.long`
  dtype and legal input IDs;
- verify EOS prediction and causal cross-document transition targets;
- compare CE with **explicit** `ignore_index=-100` against an independent
  manually masked reduction; ensure ignored slots change neither
  numerator nor denominator;
- assert the final partial window's valid target count and that
  unscored padding has zero contribution;
- verify no cross-split packed batches and deterministic identity;
- handle zero-valid-label batches by an explicit error/skip contract;
- keep reference sequence context at 4096 while designing tiny,
  bounded CPU-only model unit tests to avoid expensive full
  Architecture-C 4096×24576 logits; **never silently change or
  unfreeze Architecture C**;
- report PyTorch availability/skips, exact test results, Ruff/format,
  and whether any model code adjustment is needed.

No CUDA/NVML, no GPU0/GPU1, no production corpus, no tokenizer
materialization of C1 text and no training. The C3 benchmark's
deterministic restart does not replace this loss integration test.

## 4. Still open, and stop line

- Final C1 rights/research-use clearance and training-view mixture A/B/C.
- Per-source quota units, deterministic source sampling, epoch boundaries,
  token reuse and shuffle/ordering.
- Full training optimizer, scheduler, effective batch, token budget,
  precision/state-layout policy and checkpoint cadence.
- Production streaming/data-loader correctness and checkpoint/resume.
- C3.7 pilot, C4 production processing, C5/C6, and any GPU execution
  require explicit independent approvals.

**Gate C4 status: IN PROGRESS. C4-P1.1 POLICY APPROVED; P1.2 CPU
integration IMPLEMENTED / EXECUTION AND REVIEW PENDING.**
