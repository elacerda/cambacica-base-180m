# C1-BD4 — Final Scientific Review and Closure

- **Decision date:** 2026-10-13
- **Decision reference:** `C1-BD4-2026-10-13`
- **Status:** COMPLETE / SCIENTIST APPROVED
- **First-training exclusion count:** **0**

The project scientist reviewed the 20 reported candidate contexts and approved
retaining all 18 affected post-D1 documents. This is a decision about the
available matching evidence. It is not proof that every possible form of
benchmark exposure is absent.

## Production scan and verification identities

BD3 scanned the immutable post-D1 corpus at
`/mnt/data/cambacica-base-180m/deduplicated/exact`, whose manifest SHA-256 is
`57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428`. The
manifest identifies 21,603,689 retained documents.

The production output is
`/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd3/scan-bd3-20261009T201945Z-cenpesor`.
Its original manifest SHA-256 is
`20e55353720f75d4883889f7fd9b7109b3f82e18f0ec4cf05384765d7bae3010`. The
published manifest remains `BD3_SCAN_COMPLETE_PENDING_INDEPENDENT_VERIFICATION`;
that historical field was not changed. The separate verification log records
`BD3_SCAN_VERIFIED_COMPLETE`, 21,603,689 accounted records, 532 verified input
shards, 20 candidate-hit rows, zero approximate-anchor evidence rows, and exit
status 0.

The BD3 scientific authorization is recorded in
[`C1_BD3_SCIENTIFIC_APPROVAL.md`](C1_BD3_SCIENT_APPROVAL.md). No new scan,
matching rule, threshold, or exclusion authority was introduced by BD4.

## Candidate matches and scientific classification

All 20 hits originate from GigaVerbo v2 and match question fields from the
Portuguese Belebele test snapshot. Every row has
`field_role=question`, `decision_rule=exact_distinctive_question`,
`exact_match=true`, and a six- or seven-token match.

| Belebele Portuguese question | Example ID | Hits |
| --- | --- | ---: |
| Qual é a composição química da água? | `ee87648fdbff389027ae5ce8f1baf943c736f428f947762a1474fadd1f8e3edf` | 15 |
| O que os cientistas queriam descobrir? | `69288de4b75c4484c848b1028e512cae95927371c3bbcf7061d8ad78b471ad90` | 3 |
| Onde foi realizada a cerimônia fúnebre? | `57aa505f32e9e28b8e22af8f3b87a5f9e5fd0fe5f8dd5478ade5262a2004539a` | 1 |
| Qual era o propósito da Grande Pirâmide? | `b4b75c6816f2e66550e71d261baf6de7db9c8a7d5a0172677e20d3ba668e8659` | 1 |
| **Total** | **4 examples** | **20** |

The reviewed excerpts place the overlaps in independent contexts. The water
composition question appears in educational, chemistry, water-treatment, and
philosophical material; some occurrences begin longer questions about
hydrogen peroxide rather than water. The scientists question occurs in
journalism about different scientific subjects. The funeral question appears
in a historical account of a ceremony in Belém do Pará. The Great Pyramid
question appears in an independently contextualized discussion of the
pyramids of Giza.

The approved classification for every hit is
**`SHORT_GENERIC_PHRASE_OVERLAP — NO_EXCLUSION`**. The available contexts do
not demonstrate copied benchmark passages or question-answer exposure. Earlier
agent-proposed labels for five possible-exposure cases were provisional and
are superseded by this scientist-approved decision; they were not blind
controls or independent adjudications.

## Document decisions and review artifacts

The deterministic document table contains the 18 unique post-D1 `doc_id`
values referenced by the 20 hits. Every document is marked `RETAIN`. The
versioned exclusion manifest explicitly contains
`"excluded_document_count": 0` and `"excluded_document_ids": []`. No training
record was removed or edited. The 21,603,689-record post-D1 corpus and its
manifest remain unchanged.

The reproducible exporter is
[`finalize_bd4_review.py`](../scripts/finalize_bd4_review.py). It reconciles
each hit to the pinned benchmark question and checks the 18 referenced source
records by their post-D1 shard, row ordinal, occurrence ID, and text checksum.
It does not scan the corpus. Review records preserve the original BD3 hit
fields and corpus coordinates.

The final artifact set is published at
`/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd4/review-v1/`.
A byte-identical staging copy is also held outside Git at
`/tmp/cambacica-base-180m/decontamination/benchmark-v1/bd4/review-v1/`. The
review manifest SHA-256 is
`9889c448fd67a3728bdd508d1eb41061413bb0b42f13a38569c33a6b9d944be3`.

| Artifact | SHA-256 |
| --- | --- |
| `reviewed_hits-v1.jsonl` | `debd56ada9266a85e96b020c90179f9e133906865343ce390bdb16b734da7f35` |
| `document_decisions-v1.csv` | `06679a37fb4f80af6d4c2bb31d425915e9386c30ebc690d14621b7f706728e62` |
| `exclusion_manifest-v1.json` | `dc16c54d5bdcd8873f7ee12544d82f116cc54168b47372440cf2c9b8e489ecfb` |
| `FINAL_REVIEW_README.md` | `12ea53b7840aa54b701422121d9e9bdcff72874676267596bee23e4f88c4b946` |

The output manifest also records the pinned D1, BD3, and benchmark-snapshot
identities, the separate BD3 verification result, all counts, source and
artifact checksums, the decision reference, and review limitations. Two
independent exporter runs produced byte-identical outputs.

## Interpretation and limitations

The scientist reviewed contextual excerpts, not an independent full-document
comparison against every benchmark passage. No long-passage or complete
question-answer exposure was demonstrated in the reported candidate contexts.
Lexical matching can miss paraphrases, translations, answer-only exposure,
and other forms that do not meet the approved policy. Conversely, an exact
short question match can occur in ordinary material. The result is no detected
strong contamination among the reported candidates, not proof of zero
contamination across the corpus.

The pinned Belebele snapshot is `facebook/belebele` at revision
`d4c91dedc9de484dbea7b7d940f898f59fd135e9`, configuration `por_Latn`, split
`test`, with a declared CC BY-SA 4.0 license. Its passages derive from
FLORES-200 and linked source material; the dataset label does not independently
resolve rights to every upstream passage. CALAME-PT is pinned at
`NOVA-vision-language/calame-pt` revision
`353671bc95cc3d94d488201f67d41b640eb80c55`, with a declared MIT license. Its
provenance describes handwritten and GPT-3.5-generated examples derived from
Portuguese Wikipedia, OSCAR, and Arquivo.pt, without per-example upstream
source IDs. The declared license does not by itself settle rights to embedded
third-party content. The applied BD3 policy reported no CALAME-PT candidates.

## Remaining C1 work before C2

- Resolve evaluation provenance and rights caveats for the pinned benchmark
  snapshots, the CALAME whitespace-only row 718 denominator (2,076 snapshot
  rows versus 2,075 proposed evaluable rows), and pin the evaluation code.
- Select corpus composition A/B/C and finalize source, language, quality, and
  license policies.
- Separately approve and create the reproducible train/validation/test split,
  then produce and verify the immutable manifests and statistics that define
  C2's input.

C1 remains **IN PROGRESS** and C2 remains **PENDING**. BD4 closes the
first-training benchmark-decontamination workstream with zero exclusions; it
does not mark C1 PASS.
