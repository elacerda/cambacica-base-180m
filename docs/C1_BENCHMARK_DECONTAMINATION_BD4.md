# C1 Benchmark Decontamination — BD4 Candidate Review

- **Review date:** 2026-10-10
- **Status:** Agent-proposed review prepared; scientist classification and closure approval pending.
- **Exclusion authority:** None. No training document was excluded or modified.

## Scope and provenance

This review covers only the 20 rows in the independently verified BD3 run
`bd3-20261009T201945Z-cenpesor` and the 18 post-D1 documents named by those
rows. It does not rerun BD3, perform another corpus-wide scan, change the
approved matcher policy, or create a corpus view or training split.

The source corpus is the immutable post-D1 corpus at
`/mnt/data/cambacica-base-180m/deduplicated/exact`, manifest SHA-256
`57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428`. The
benchmark source is the pinned BD2 snapshot at
`/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2/snapshot`,
manifest SHA-256
`d9b5bfdec543af37fdb8733239b0f8b0255339454b5ebca8880371bb86f77cda`. The
approved BD3 policy hash recorded in the scan manifest is
`00d070939f2f7b472cbeaa707fd974bb02626a3724fd0867a28f6da455afc3c9`.

The independent verification log reports
`BD3_SCAN_VERIFIED_COMPLETE`, `input_record_id_digests_verified: true`,
21,603,689 records accounted, 532 input data files, 20 hit rows, zero anchor
evidence rows, and exit status 0. Its elapsed time was 1m35.45s. The original
scan log reports exit status 0 and elapsed time 13h21m39s. The verification
evidence is recorded in the internal review packet; the scan manifest itself
remains unchanged.

For every hit, the review joined the exact question field and registry row by
benchmark/example/field identity, checked the source-row and source-file
identifiers, read the located post-D1 document, recomputed its UTF-8 text
SHA-256, recomputed its occurrence ID, and checked the matched token sequence
and character offsets. All 20 joins, all 18 document hashes and occurrence
IDs, and all 20 spans passed. Candidate ordering is preserved from the
original Parquet order: `doc_id`, corpus character start/end, benchmark,
example, and field ID. The two documents with two rows each retain both hit
identities and both offsets.

## Counts and provisional classifications

All 20 rows are from `gigaverbo_v2`, match Belebele Portuguese question fields,
and use `exact_distinctive_question`. Each is an exact six- or seven-token
question match. BD3 produced zero approximate anchor-evidence rows and zero
CALAME-PT candidate hits.

| Belebele question | Hits | Documents | Generic lexical overlap | Possible benchmark exposure |
| --- | ---: | ---: | ---: | ---: |
| “Qual é a composição química da água?” | 15 | 13 | 11 | 4 |
| “O que os cientistas queriam descobrir?” | 3 | 3 | 3 | 0 |
| “Onde foi realizada a cerimônia fúnebre?” | 1 | 1 | 1 | 0 |
| “Qual era o propósito da Grande Pirâmide?” | 1 | 1 | 0 | 1 |
| **Total** | **20** | **18** | **15** | **5** |

The controlled vocabulary is `generic_lexical_overlap`,
`possible_benchmark_exposure`, `strong_benchmark_exposure`, and `inconclusive`.
The classifications below are agent-proposed assessments. No human review is
claimed; the scientist-final fields in the internal matrix are blank. No hit
is classified `strong_benchmark_exposure` or `inconclusive` at this stage.

| Review row | Example | Proposed assessment | Evidence-based assessment |
| --- | --- | --- | --- |
| BD4-H01 | Scientists | `generic_lexical_overlap` | Tuatara embryology article; the phrase is followed by a question about embryonic anatomy, unrelated to comet impacts or Earth formation. |
| BD4-H02 | Water composition | `possible_benchmark_exposure` | Exact question followed by an H₂O explanation that gives the benchmark’s correct fact. This may be generic science FAQ content; no passage or choices appear. |
| BD4-H03 | Water composition | `generic_lexical_overlap` | Philosophy discussion uses water composition as an example of empirical knowledge. |
| BD4-H04 | Scientists | `generic_lexical_overlap` | Hummingbird research article; the phrase refers to the mechanics of wing-generated sound. |
| BD4-H05 | Water composition | `generic_lexical_overlap` | Agronomy text discusses testing water chemistry for herbicide application. |
| BD4-H06 | Water composition | `generic_lexical_overlap` | Philosophy text gives a hypothetical about a priori knowledge. |
| BD4-H07 | Water composition | `generic_lexical_overlap` | This occurrence is a question-list entry without a local answer. It is a separate offset from BD4-H08; that other occurrence makes the document-level assessment possible exposure. |
| BD4-H08 | Water composition | `possible_benchmark_exposure` | Separate occurrence in the same document, with a direct H₂O explanation after the exact question; no passage or choices appear. |
| BD4-H09 | Water composition | `generic_lexical_overlap` | The matched phrase is only a prefix of a related-list question about hydrogen peroxide (`água oxigenada`). |
| BD4-H10 | Scientists | `generic_lexical_overlap` | Praying-mantis mating article; the discovery concerns mating behavior, not the benchmark’s comet passage. |
| BD4-H11 | Water composition | `generic_lexical_overlap` | Rhetorical example in a discussion of mathematics and everyday knowledge; no question-answer context. |
| BD4-H12 | Funeral location | `generic_lexical_overlap` | Narrative concerns a different funeral, in Belém; its church differs from the benchmark passage’s St Peter’s Square. |
| BD4-H13 | Water composition | `generic_lexical_overlap` | This occurrence is a question-list entry without a local answer. It is distinct from BD4-H14; that separate occurrence makes the document-level assessment possible exposure. |
| BD4-H14 | Water composition | `possible_benchmark_exposure` | Separate occurrence in the same document, followed by an H₂O explanation that matches the benchmark’s answer fact; no passage or choices appear. |
| BD4-H15 | Water composition | `generic_lexical_overlap` | Philosophy/logic text mentions water composition in a parenthetical example about bathing. |
| BD4-H16 | Water composition | `generic_lexical_overlap` | Entry in a broad list of water-science questions, without benchmark passage or choices. |
| BD4-H17 | Water composition | `generic_lexical_overlap` | The matched phrase is a prefix of a related-topic question about hydrogen peroxide, not water. |
| BD4-H18 | Water composition | `possible_benchmark_exposure` | The hit begins a longer textbook question about water, copper, and salt. Elsewhere in the same textbook, a 10-token passage fragment and the exact correct answer option occur. The textbook setting makes generic reuse plausible, but the combined overlap warrants scientist review. |
| BD4-H19 | Great Pyramid | `possible_benchmark_exposure` | The exact question appears in a pyramid article. A separate section about 12,300 characters later shares the benchmark passage’s first 13 word tokens; the rest of the sentence differs. No complete passage or options appear. |
| BD4-H20 | Water composition | `generic_lexical_overlap` | Entry in a broad water-science question list, followed by discussion of water uses; no benchmark passage or choices were found. |

BD4-H07 and BD4-H08 are separate occurrences in one document; BD4-H13 and
BD4-H14 are separate occurrences in another. In both documents the first
occurrence is a list entry and the second has a direct answer context. The
matrix preserves each source offset and does not collapse either pair.

## Passage and answer-choice review

No complete benchmark passage was found as an exact whitespace-normalized
match in any of the 18 reviewed documents. No complete four-choice answer set
was found. One exact answer option—the correct water-composition option—occurs
twice in the same chemistry textbook as BD4-H18. The same document contains
10 consecutive tokens from the benchmark passage at that location and a
second nine-token occurrence. These are common chemistry wording within a
textbook; they are separated from the matched question, and the full passage
and distractors are absent.

The Great Pyramid document for BD4-H19 contains a 13-token opening fragment of
the benchmark passage in a separate “Curiosities” section. The question hit
and fragment are not adjacent, the sentence then diverges, and none of the
four answer options is reproduced. This is possible exposure, not evidence of
a complete copied item.

Three additional direct water-question contexts (BD4-H02, H08, and H14) are
followed by explanations of the correct H₂O fact. Because the fact and
question are common in general science material and no benchmark passage or
choice set appears, these rows remain possible rather than strong exposure.

## Proposed action and closure conditions

**Agent-proposed exclusion recommendation: zero documents.** No reviewed
document currently has evidence strong enough to support an exclusion
recommendation. This is provisional: five rows in five documents remain
`possible_benchmark_exposure` and require scientist adjudication. This
recommendation does not authorize exclusions, does not represent the
scientist’s decision, and does not close BD4.

The post-D1 corpus remains unchanged and is to be preserved in full. No
modified corpus, exclusion manifest, or training split was created. BD4 can
be closed only after a scientist reviews and records final per-hit/document
classifications, explicitly decides whether any record-level exclusion is
warranted, and signs the decision. Any exclusion needs a separate
scientist-approved, evidence-backed record-level decision. If the scientist
approves zero exclusions, the complete post-D1 corpus remains the input for
the next approved C1 steps.

## Limitations

- The BD3 thresholds were fixed for this approved run; they are not validated
  universal contamination thresholds.
- This review assesses only candidates emitted by the approved detector. It
  does not estimate detector recall or establish corpus-wide absence of
  contamination.
- Exact question matches can arise in ordinary question lists, journalism,
  textbook prose, and Q&A pages. Conversely, paraphrases, translations,
  answer-only exposure, OCR, and other missed forms may not be detected.
- The reviewed documents are long and may combine topics. The assessment
  distinguishes local hit context from other evidence elsewhere in the same
  document; a short question match alone is not treated as proof that the
  entire document is contaminated.
- Benchmark passage text and source excerpts are retained in the internal
  review packet only. This repository report contains no bulk source text.

## Internal review packet

The checksummed, non-Git artifact directory is:

`/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd4/review-v1/`

It contains the 20-row candidate matrix, 18-document summary, four-example
summary, bounded evidence excerpts and provenance, full benchmark references
for scientist review, the provisional recommendation, validation results,
and `artifact_manifest.json` with checksums. The artifact-manifest SHA-256 is
recorded in `artifact_manifest.sha256`.

## Remaining C1 work before C2

1. Scientist approval of the five possible-exposure cases, the 15 generic
   assessments, and the proposed zero-exclusion action; record any changed
   classification or separately approved exclusion.
2. Complete BD5 verification and publish the residual-risk and C1 handoff
   record.
3. Finish C1 approvals for the final corpus composition and reproducible
   train/validation/test split, then produce and verify the immutable corpus
   manifests and statistics that define the C2 tokenizer input.

C1 remains **IN PROGRESS** and C2 remains **PENDING**. This report prepares
BD4 for scientist review; it does not declare BD4 scientifically approved.
