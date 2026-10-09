# C1-BD1 Benchmark and Decontamination Approval

- **Decision:** C1-BD1 scientific scope **APPROVED**
- **Record date:** 2026-10-09
- **Approval basis:** Scientist-approved direction supplied for Gate C1-BD2. No individual signatory or separate meeting record was supplied.
**Production authorization:** C1-BD3 **NOT APPROVED / NOT RUN**

This record approves the benchmark direction and operational constraints. It
does not assert that dataset licenses settle every right in upstream source
material, approve final matching thresholds, authorize any training-record
exclusion, or approve a production scan.

## Scientist-approved scope

- Use **CALAME-PT** as a core Portuguese continuation benchmark.
- Use **Belebele `por_Latn/test`** as a core reading-comprehension benchmark.
- Defer ENEM and other benchmark families.
- Prefer LM Evaluation Harness for future benchmark evaluation.
- Run one read-only corpus pass for eventual production decontamination.
- Require human review before approving any training-record exclusion.
- Keep the post-D1 source corpus immutable and keep A/B/C mix specifications
  unchanged.
- Preserve all post-D1 records for the first training run under the approved
  near-deduplication decision: **zero near-deduplication removals**.
- Use deterministic document/exact-content-group disjointness for the future
  split. Do not claim near-duplicate-disjoint splits; residual near-copy risk
  remains documented.

## Candidate revisions recorded for BD2 verification

| Benchmark | Candidate repository and revision | BD1 status |
| --- | --- | --- |
| CALAME-PT | `NOVA-vision-language/calame-pt` @ `353671bc95cc3d94d488201f67d41b640eb80c55` | Candidate pin; revision, files, schema, counts, metadata, and checksums required BD2 verification. |
| Belebele Portuguese | `facebook/belebele` @ `d4c91dedc9de484dbea7b7d940f898f59fd135e9` | Candidate pin; `por_Latn/test` revision, files, schema, counts, metadata, and checksums required BD2 verification. |

Scientific selection approval did not itself validate access, rights, schema,
or the evaluation adaptation. BD2 reports those checks separately.

## Provisional matching hypotheses

The proposed starting policy is **at least 50 consecutive matching word tokens,
or at least 80% distinctive-token coverage with at least two rare 13-token
anchors**. A starting corpus anchor document-frequency ceiling of 100 is also
subject to calibration. These values are hypotheses for BD2 controls only; they
are not production thresholds and are not approved for BD3.

Exact full-item, passage/context, distinctive-question, and question-plus-answer
evidence may be reported. A standalone answer word or option letter is never a
candidate trigger. Any corpus match is review evidence, not an automatic
exclusion.

## Implementation decisions delegated to BD2

BD2 may choose deterministic stable row/field IDs and checksums, preserve raw
benchmark rows while storing separate match-normalized views, use NFC plus
casefolded Unicode word tokens with original offsets, use bounded SQLite spill
for corpus document-frequency evidence, select synthetic and narrowly selected
benchmark-derived controls, and publish artifacts atomically outside Git.
These implementation choices do not alter benchmark scope or authorize source
corpus changes.

## Decisions requiring final scientist approval before BD3

1. Accept the verified dataset revisions and provenance/rights caveats for the
   intended local evaluation use.
2. Resolve CALAME denominator handling: retain all 2,076 pinned rows in the
   snapshot and decide whether the whitespace-only target row is excluded from
   a future metric. The reported 2,075 evaluator variant is not an official
   CALAME split.
3. Approve the exact and approximate evidence rules, short-question handling,
   benchmark rarity rule, corpus document-frequency cutoff, coverage threshold,
   anchor count, and any source-specific adaptations.
4. Approve the scan policy and output protocol in writing, including who will
   review hits and how a long document containing only a copied passage will be
   handled.
5. Keep training-record exclusions and clean-evaluation denominator changes
   subject to separate evidence-backed review and explicit approval.

BD3 must remain a manually initiated, read-only scan over the pinned post-D1
manifest. This record does not approve it.
