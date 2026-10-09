# C1 Near-Deduplication: Final First-Training Decision

**Decision:** APPROVED / CLOSED FOR FIRST TRAINING
**Near-deduplication production:** NOT RUN BY DESIGN; **zero removals**
**Decision date:** 2026-10-09

## Decision

For the first Cambacica Base 180M training run, preserve every record in the post-exact corpus. Do not perform near-deduplication removals. The post-D1 exact corpus remains the authoritative input to benchmark decontamination.

This is a final operational decision for the first training scope. It does not say the corpus is near-duplicate-free, that remaining redundancy has no training effect, or that D2 sample simulations estimate production impact.

## Pinned input and completed evidence

- Corpus: `/mnt/data/cambacica-base-180m/deduplicated/exact`
- Manifest SHA-256: `57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428`
- Retained documents: **21,603,689**
- D1 exact removals: **45,736** documents
- Normalized word capacity after D1: **21,470,091,017** words
- D1: **COMPLETE / PASS**
- D2 pilot, D2b calibration, D2c full census, and D2d scientific review: **COMPLETE**
- D2c verifier: **[PASS] near-census input identity, stages and artifacts verified**

D2c artifacts are at `/mnt/data/cambacica-base-180m/dedup-census/near-v1/`; its scientific report is [C1_NEAR_DUP_CENSUS_D2C.md](C1_NEAR_DUP_CENSUS_D2C.md). D2d artifacts are at `/mnt/data/cambacica-base-180m/dedup-review/d2d-v1/`; its review is [C1_NEAR_DEDUP_SCIENTIFIC_REVIEW_D2D.md](C1_NEAR_DEDUP_SCIENTIFIC_REVIEW_D2D.md). The D2d packet and simulations are review aids, not production deletion authorization or population impact estimates.

## Why preserve the records

D2–D2d established that genuine near copies exist, alongside pairs that look almost identical under word-5 Jaccard but represent different material. Distinct Wikipedia taxonomic articles exceeded 98% Jaccard because they shared references and structure; legal records shared most of their text while referring to different proceedings. Gutenberg/web copies can differ in edition, transcription, or unique content, and Carolina/Wikipedia versions can preserve different facts.

A global Jaccard cutoff is therefore unsafe: it can delete distinct entities and records. A high score is a candidate signal, not proof of document identity or substantive equivalence.

The available candidate panels were enriched and do not estimate corpus-wide removable prevalence. The D2c LSH buckets were capped, so candidate generation does not establish complete recall. The sample-only P2 simulation is not a production-scale impact estimate. No selective production policy has been validated to the standard needed to authorize loss of source information.

The trade-off is explicit: preserving all post-D1 records retains unique facts, editions, and source-specific coverage, while also retaining genuine approximate redundancy. That redundancy may affect training frequency or loss; its effect has not been measured as a production intervention.

## Future direction and next C1 stage

If near-deduplication is reconsidered for a later training scope, the scientific direction remains **B — selective removal only with verified document identity, substantive equivalence, and a direct verified edge to the retained owner**. This is a future option, not a frozen or executable policy. No P2 policy will be implemented for the first run.

Near-deduplication is not required before benchmark decontamination. The two stages answer different questions: near-deduplication considers redundant corpus records; benchmark decontamination checks overlap with the material used to evaluate the model. Continue from the immutable post-D1 corpus into benchmark inventory and decontamination planning without running another near-duplication investigation.

## Required status

| Work item | Status |
| --- | --- |
| D1 exact deduplication | **COMPLETE / PASS** |
| D2–D2d investigation and review | **COMPLETE** |
| Near-dedup scientific decision | **APPROVED / CLOSED FOR FIRST TRAINING** |
| Near-dedup production | **NOT RUN BY DESIGN; zero removals** |
| Benchmark decontamination | **NEXT / NOT RUN** |
| Gate C1 | **IN PROGRESS** |
| Gate C2 | **PENDING** |

The first-run corpus limitation is that exact duplicates were removed, but some genuine near copies remain. There is no claim of zero contamination or zero redundancy, and no estimate of their training impact.
