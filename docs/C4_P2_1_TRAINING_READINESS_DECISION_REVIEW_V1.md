# C4-P2.1 — Scientific decision review and shortest safe path to first training

**Date:** 2026-10-19.
**Status: DOCUMENTED PROPOSAL / SCIENTIST DECISIONS PENDING.**
**Authorization:** scientist authorized C4-P2.1 **review and documentation only**.
**Goal:** separate the **minimum prerequisites for a short real-data
learning smoke run (C5)** from the **additional prerequisites for full
pretraining (C6)**, without reopening frozen C1/C2/C3/P1 contracts.

This document recommends a sequence of work and candidate scientific
defaults; it **does not approve** a mixture, rights clearance, seed,
quota unit, epoch/reuse policy, recipe, tokenizer access, code change,
GPU allocation or training. No corpus was opened, sampled, counted or
tokenized; no benchmark or training was run as part of C4-P2.1.

The design basis is
[C4_P2_PRODUCTION_PIPELINE_SCIENTIFIC_DESIGN_V1.md](C4_P2_PRODUCTION_PIPELINE_SCIENTIFIC_DESIGN_V1.md),
[C4_PIPELINE_DESIGN_PROPOSAL_V1.md](C4_PIPELINE_DESIGN_PROPOSAL_V1.md),
[C1_SPLIT_AND_MIXTURE_READINESS.md](C1_SPLIT_AND_MIXTURE_READINESS.md),
the C3 scientific selection and C4-P1.1/1.2/1.3a/1.3b records.

## 1. What is done and what actually prevents real training

| Dependency | Current factual status | Minimum evidence needed for real C5 |
| --- | --- | --- |
| C1 frozen post-D1 dataset and C1-S1 group split | 21,603,689 records, C1-S1 assignment published/verified | Do not mutate population, split, or source/group membership |
| C1 usage rights and allowed training view | **IN PROGRESS / BLOCKING** | Explicit source/subset rights/use decision for every used record category, obligations, exclusions/fail-closed rule; scientist-approved immutable train/validation/test view and provenance |
| C2 tokenizer | **COMPLETE / frozen** SentencePiece Unigram 24,576 | Exact model/version/options verified in approved tokenizer adapter; tokenization only after real-access approval |
| C3 model and GPU feasibility | **COMPLETE / PASS** for architecture C and synthetic C3.6 engineering | Revalidate isolated physical GPU0, runtime limits, lock and scientific authorization for the new workload; GPU1 Qwen strictly untouched |
| C4 P1 contracts, CPU integration and synthetic scaling | P1.1 approved, P1.2/P1.3a/P1.3b **PASS** | Transform tested synthetic interfaces into production C4-D under a **new implementation/access approval**; verify real-token long-document correctness |
| C4-T recipe | **NOT APPROVED** | Choose optimizer, precision, LR schedule/warmup, effective batch, clip/decay, scored-token/update budget, validation and stopping protocol |
| C4-V integrated restart | **NOT IMPLEMENTED/APPROVED** | Persist/model-optimizer-scheduler-RNG/cursor ledger in a single consistent snapshot and inject failures with approved fixture |
| C5 short learning smoke | **PENDING / NOT AUTHORIZED** | Independently authorize real-data short run, log learning and reproducibility checks, check data rights/isolation and bounded resource envelope |
| C6 main pretraining | **PENDING / NOT AUTHORIZED** | Freeze complete rights-approved data+mixture+exposure budget+recipe and C4 exit, review C5 evidence, separately authorize C6 |

A C3.7 **quality pilot** is separately blocked by C1 and requires
independent approval. It cannot be relabeled as C5 or run under C3.6
synthetic GPU permission.

### Important distinction: first learning run vs whole project

C5 does **not** logically require that every one of the 21.6 million
C1 documents be cleared if a scientist instead approves a **small,
explicitly rights-cleared training and holdout view** for a limited
experiment. It DOES require rights decisions for every source actually
consumed, correct inherited C1-S1 assignments, and a reproducible
recipe. A smaller view is a **new scientific choice**, not a quiet
modification of the frozen corpus or arbitrary deletion of records.

Full C6 needs the selected production population/mixture rights and
the final exposure/reuse policy in addition to the C5 evidence.
A small C5 validation is not proof of full-scale data throughput,
convergence or final-model performance.

## 2. Corpus/mixture bottleneck must be resolved before token budget

Current mixture configurations
`configs/corpus_mix_a.yaml`, `..._b.yaml`, `..._c.yaml`
express shares in **normalized words** (`len(text.split())`),
not frozen-C2 tokens, inserted EOS, scored targets or training updates.
The frozen C2 tokenizer sample was B-conditioned; this does NOT
approve mixture B or make it a mixture-neutral sample.

The existing C1 readiness analysis found **idealized no-reuse
word-based upper bounds after holdout**:

| Candidate | No-reuse normalized-word ceiling | Key limitation |
| --- | ---: | --- |
| A curated-heavy | At most 259,918,425 words | Gutenberg 8% allocation |
| B balanced | At most 519,836,850 words | Gutenberg 4% allocation |
| C residual-heavy | 415,336,413 **illustrative only** | `blogset` 3.75% of total, assuming exactly 98% word retention |

Those are **not** actually achieved whole-document views or
token-budget estimates. The C ceiling explicitly has no final
after-holdout calculation; the actual whole-document ceiling may be
smaller. C1 also documented that a 1-billion-normalized-word
exposure with strict no-reuse would exceed the limiting source
capacity by approximately 3.80× (A), 1.90× (B) or 2.36× (C)
even before holdout.

**Mandatory scientific fork before planning C6:**
(a) approve a smaller no-reuse budget;
(b) allow a documented repetition/oversampling policy and report
repeated-exposure effects; or (c) separately change/reapprove source
shares/quotas after rights review. None is selected here.
Reweighting or multiplying word ceilings by assumed tokens/word is
not a legitimate shortcut.

A rights-cleared token census would distinguish unique document tokens,
EOS overhead, stream tokens, scored targets and replayed exposure,
but that measurement **requires separate C1 access permission**.

## 3. C4-P2.1 candidate decisions — for scientist review, NOT selected

| ID | Candidate direction | Why / what still needs approval |
| --- | --- | --- |
| D1 — comparison metric | **Recommend** C2 payload-token exposure plus separate scored-label/compute counters as the training-budget objective; retain normalized words as C1 capacity accounting | Compare against Q-W; model-token census cannot be assumed; define whether EOS counts in quotas |
| D2 — initial single-view sampling | **Recommend** one immutable, explicitly approved train-only view with deterministic O1 order; optionally S1 for a small single-source smoke | Multi-source weighted quotas need separately approved deterministic S2 merge, key namespace, tie-break and source exhaustion |
| D3 — admission granularity | **Recommend** preserve complete documents and their C1-S1 group identities; report quota shortfall rather than truncate silently | Handling oversized documents, whole-doc tokenizer equivalence and quota tolerance still open |
| D4 — epoch boundary | **Recommend E1** independent epoch streams for auditable next-target ownership; no scored cross-epoch target by default | Requires explicit scientific approval; keep P1.1 no automatic per-document BOS; define start/reset specials and partial windows |
| D5 — reuse | **Recommend R0** no repeated exposure for an initial small rights-cleared smoke where capacity suffices | Main C6 may require R1 or quota change; define count, budget and duplicate effects before choosing |
| D6 — source/target attribution | **Recommend** track model target source/document, EOS and cross-source transitions, with full input span provenance | Must be scientifically reviewed; target source can differ from causal input source |
| D7 — ACK and checkpoint | **Recommend** acknowledging only data included in the **same durable model/optimizer/scheduler/RNG checkpoint generation**, at a declared optimizer-update boundary | Per-window P1.3a synthetic ack must NOT advance independently in production; resume and rollback protocol need tests |
| D8 — resource/recipe | Treat C3.6 microbatch 4 × accumulation 1 as a **benchmark comparison candidate only** | New recipe, physical GPU0 authorization, memory limits and real-data end-to-end rate must be reviewed |

None of D1–D8 is recorded as **SCIENTIST APPROVED** by this
proposal. This table is a decision agenda; numerical seeds,
token budget, update count and checkpoint cadence are deliberately
unfixed rather than guessed.

## 4. Minimal dependency path for a real C5 learning smoke

**Phase A — C1 rights/view decision (scientific + applicable usage review).**

1. Inventory license/rights and intended training use by candidate
   source and subset; document any attribution, contractual constraints,
   redistribution and model-release distinction. A license label
   alone does not imply permission.
2. Explicitly approve the finite allowed source set and a bounded
   train-only view with immutable C1-S1 validation/test references.
   Disallowed or unresolved source IDs must be rejected, not skipped
   invisibly. This *does not* change the frozen C1 population.
3. Pin hashes for the selected view, rights decision, split assignments,
   source revisions and corpus/text identity. No actual corpus access
   until a separate approved operation permits it.

**Phase B — C4-P2 choices and C4-D/C4-T engineering (new approvals).**

1. Scientist chooses quota/sampler, epoch/reuse, BOS/reset and
   token/scored-target exposure rules for the C5 bounded view.
2. Separately authorize C4-D implementation, first synthetic-only.
   Verify corpus/split provenance, document-segment tokenization
   equivalence (including long documents), no missing/repeated EOS,
   P1 4096/4097 overlap, masked loss, tail/empty-window handling.
3. After C1 rights and separate real-access permission, validate
   the adapter only on an explicitly approved limited real view.
   No materialization of the whole corpus by default.
4. Separately approve C4-T recipe and operational resource limits.
   Implement a single coherent checkpoint generation binding data
   cursors to model, optimizer, scheduler and all RNG states, with
   failure/restart and loss/effective-batch tests.

**Phase C — C4-V exit review and C5 GPU authorization.**

1. Run CPU synthetic and separately approved bounded integration
   tests, negative split/rights tests, invalid-order/checksum tests,
   epoch/quota accounting checks, controlled crash rollback tests.
2. Produce auditable hashes, run provenance, budget and stop conditions;
   re-verify physical GPU0, vLLM GPU1 isolation, free memory and
   exclusive lock without presuming C3.6 automatically approves C5.
3. Obtain an explicit scientist authorization for a small C5 run.
   Monitor finite losses, expected movement against a defined control,
   masked-label correctness, held-out isolation, throughput and
   checkpoint/restart consistency. Set caps *before* execution.
   A single downward training-loss trace does not establish
   generalization or final language quality.

**For C6:** add approved full mixture/view and training-token budget,
source exhaustion/reuse treatment, exact schedule, validation cadence,
checkpoint retention/storage accounting, scientifically accepted C5
outcome and a separate C6 execution authorization.

### C5 gates are not the same as engineering probes

The already passed C3.6 GPU engineering probes and C4-P1 synthetic
CPU tests demonstrate feasibility and correctness on artificial
inputs. Neither is a rights-cleared real-text C5 learning smoke.
A future synthetic-only optimizer test can be a useful engineering
milestone but must not be marked C5 COMPLETE solely because its loss
decreases.

## 5. Checkpoint atomicity: required minimum for first real run

A restart-safe commit generation should be a versioned manifest
referencing checksummed complete state including:

- model, optimizer, LR scheduler, gradient/scaler state if applicable,
  global update and accumulation boundary;
- CPU/GPU/Python/NumPy/Torch RNG and sampler order/configuration,
  exactly which stream windows and next-token targets were consumed;
- fixed view/split/right/use identity, ordered source-exposure ledger,
  tokenizer/model hashes and actual score/EOS/token counters;
- finalized last *committed* consumer position, in-flight workers
  replay policy and snapshot generation number.

Save temporary state on approved local ext4; verify checksums
before loading; publish an atomic manifest for a **complete**
generation. After crash, restore both data cursor and model state
from the last complete generation; replay only updates not
represented in that generation. Checkpoint cadence and retention
are **OPEN**, and real hardware failure consistency has not yet
been demonstrated. This is a **scientific target contract**,
not code or a guarantee of exactly-once updates.

## 6. Approval checklist (not a permission request to execute)

| Gate / work product | Needed for C5? | Needed for C6? | Present status |
| --- | --- | --- | --- |
| Authorized view with rights/usage and existing C1-S1 group holdouts | **YES** | **YES**, plus final production view | BLOCKED BY C1 |
| Frozen C2 model and C3 architecture | YES | YES | COMPLETE |
| P1 EOS/target packing and synthetic prototype | YES | YES | APPROVED / CPU PASS |
| Explicit sampling/quota/epoch/reuse decisions | YES for bounded view | YES for production | OPEN |
| Real tokenizer adapter / complete C4-D loader | YES for real smoke | YES | NOT AUTHORIZED / NOT IMPLEMENTED |
| Approved optimizer/step/batch/precision/schedule | YES | YES, frozen final recipe | OPEN |
| Atomic full training state and restart tests | YES | YES, including retention | NOT IMPLEMENTED |
| Rights/split isolation, checksum and end-to-end C4-V review | YES | YES, full scale | NOT AUTHORIZED |
| Fresh isolated GPU0 runtime permission | YES for GPU C5 | YES | NOT GRANTED FOR TRAINING |
| C5 short learning validation | Run it first | Must pass prior to C6 | PENDING |
| C6 full mixture/exposure/token budget/scaling evidence | No | **YES** | OPEN |
| Separate run authorization | C5-specific | C6-specific | NOT GRANTED |

C1 rights decisions can progress **in parallel** with the purely
documentary C4-P2 scientific choices and future separately authorized
synthetic-only engineering. Production corpus/tokenization and GPU
activities cannot be scheduled until their respective gates open.

## 7. Next scientist decisions, in the practical order

First decide whether the goal is:

- **M1 — fastest bounded real-text learning smoke:** scientist selects
  an explicitly rights-cleared minimal view and the shortest safe
  recipe, while leaving A/B/C and the full C6 budget undecided; or
- **M2 — freeze main pretraining directly:** scientist first clears
  full intended sources, chooses A/B/C or a reviewed replacement,
  solves quota capacity/reuse and sets the model-token budget.
  C5 is still required for execution validation.

**C4-P2.1 recommendation: prioritize M1, while advancing C1 rights
and production-mixture science in parallel.** This is a proposal,
not a scientist-approved scope change, and does not authorize C1
reads, C4-D/C4-T implementations, C3.7, C5 or C6.

Required next scientific review topics:
(1) M1 vs M2;
(2) permission and view for C1;
(3) D1–D8 candidate sampling/epoch/checkpoint interfaces;
(4) authorization request for limited synthetic C4-D prototype
separately from this document.

## 8. Gate boundary

**C4-P2.1: DESIGN REVIEW DOCUMENTED / SCIENTIST DECISION PENDING.**
**C4: IN PROGRESS. C1: IN PROGRESS.**
**C3.7, real-corpus C4-D, training/GPU C5/C6: NOT AUTHORIZED.**

No real text, tokenizer payload, CUDA/NVML/GPU, data materialization
or optimizer execution occurred under this documentation-only
authorization.
