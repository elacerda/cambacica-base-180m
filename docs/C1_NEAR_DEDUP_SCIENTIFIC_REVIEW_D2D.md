# Gate C1 Near-Dedup Scientific Review (D2d)

**Review status:** COMPLETE; generated assessments remain PROVISIONAL  
**Decision proposal:** **B — selective near-deduplication**, limited to two curated/web relationship classes pending human verification
**Production policy:** not ready to freeze; limited human verification is required  
**Near-dedup production:** NOT RUN  
**Benchmark decontamination:** PENDING  
**Input:** post-exact corpus, 21,603,689 retained records  
**Exact manifest SHA-256:** `57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428`  
**D2c census:** `/mnt/data/cambacica-base-180m/dedup-census/near-v1/`  
**D2d review packet:** `/mnt/data/cambacica-base-180m/dedup-review/d2d-v1/`

## Scope and method

This review uses the completed D2c census and its existing 8,014-pair
exact-scored sample. It does not rerun fingerprints, LSH, candidate generation,
or exact scoring. The D2c verifier had already returned **PASS**; this review
checked the pinned exact-manifest hash, stage manifests, artifact checksums,
schemas, and sample row counts without rerunning the full verifier.

The deterministic review packet contains 148 pairs selected by fixed
source-family and diagnostic strata, SHA-256 ranking, and forced inclusion of
the named critical cases. Each row carries stable record IDs, source/subset,
available titles and provenance, normalized word counts, length ratio, exact
and estimated scores, directional containment, shared/union/unique shingle
counts, existing flags, six bounded excerpts, and a provisional assessment.
All generated labels are explicitly **PROVISIONAL**; `human_label` is blank.

The packet has 123 development pairs and 25 held-out pairs. Pairs sharing a
selected record, sampled book ID, taxonomy family, or connected selected
near-duplicate group stay in one split. The held-out groups were selected after
the predicates were fixed from the remaining groups and checked for failure
modes. The split is case-based, not blinded or probability-sampled: the task's
named examples, D2c aggregate strata, and development cases informed policy
design. No precision, recall, or confidence interval is claimed.

## Census evidence and source families

D2c generated **65,412,497 unique LSH candidate pairs** with **6,650,577
endpoints**. These are candidate relationships, not duplicated-document
counts. The exact-score sample includes 2,007 pairs at exact word-5 Jaccard
`>=0.80` and 474 at `>=0.95`; it is stratified and enriched, so these fractions
do not estimate corpus prevalence. The broader candidate catalog contains
MinHash estimates, not exact Jaccard scores for every pair. Its 16,059,211
estimated `>=0.80` pairs are not exact-score positives.

| Family | Capped census candidate pairs | Exact-scored sample | Sample J `>=0.80` | Sample J `>=0.95` |
| --- | ---: | ---: | ---: | ---: |
| Wikipedia PT ↔ Wikipedia PT | 23,001,648 | 1,096 | 330 | 83 |
| Carolina/wik ↔ Wikipedia PT | 6,533,095 | 676 | 92 | 18 |
| Carolina/wik ↔ Carolina/wik | 5,484,896 | 267 | 20 | 0 |
| Gutenberg PT ↔ GigaVerbo | 96 | 96 | 44 | 25 |
| GigaVerbo ↔ GigaVerbo | 5,338,890 | 2,279 | 92 | 24 |
| Other curated sources ↔ GigaVerbo | 308,560 | 1,391 | 281 | 37 |
| Any pair involving ParlamentoPT | 23,488,194 | 1,923 | 1,103 | 271 |

Candidate counts describe the bounded LSH output and do not count unique
documents or verified copies. The 96 Gutenberg/GigaVerbo candidates all occur
in the exact-scored sample; this covers the current detected candidate set,
not LSH recall or pairs omitted by bucket caps.

## Scientific findings

### Wikipedia entities and shared templates

The Argidia and Gnamptonychia articles are distinct moth genera. Their opening
sentences name different entities, while the rest shares references and
bibliographic structure: 120 words each, Jaccard **0.98347**, 119 shared
shingles out of 121 union shingles, and one unique shingle on each side. The
packet also contains distinct Alchemilla, Acer, and Acacia pairs. Held-out
Pseudometisa and Barandra are different genera at Jaccard **0.98077**. A global
`J>=0.95` rule would accept these distinct-entity pairs.

Title/entity identity can protect some of these cases only when checked against
the document. A Wikipedia URL and title identify the encyclopedia article, and
its opening text can expose a different taxon or factual value. Carolina/wik
often has an opaque `WIK...` identifier and no original article URL or title;
title inequality alone is therefore not a safe test. For a Carolina/Wikipedia
pair, use a reliable canonical title plus matching entity evidence in both
texts. If that evidence is unavailable or the body differs materially, retain
both records. Duplicate text, the same entity at another revision, and
different entities sharing a template are separate relationships.

### Carolina/wik and Wikipedia PT copies

The Nazz pair has 1,303/1,324 words, Jaccard **0.98408**, and the same article
opening; every sampled Carolina word-5 shingle is in the Wikipedia text. The
same-entity evidence is also strong for Osteoblastoma (**0.98381**), Otto
Wagner (**0.97318**), and Suisei (**0.97164**). The last two are flagged by
the existing boilerplate heuristic because their Carolina IDs are opaque and
their openings differ slightly. Those flags are not truth labels.

The held-out NGC 4013 pair has matching title/opening evidence but only
Jaccard **0.79518** on 74/75-word records. It may be another capture of the
same article, or only a shared lead. It falls below the proposed P2 screen and
must be preserved until reviewed. Same entity does not by itself make two
versions interchangeable; check for changed facts and unique content.

### Gutenberg literary works and web copies

Several long pairs strongly support copies of the same literary work:

- *Dom Casmurro*, Gutenberg eBook 55752 ↔ HPLT2: 65,707/66,216 words,
  Jaccard **0.99258**, and Gutenberg-to-web containment `1.0`.
- *A Fallencia*, Gutenberg eBook 69229 ↔ HPLT2: 72,455/72,578 words,
  Jaccard **0.99847**; the Gutenberg opening identifies the 1901 second
  edition, and HPLT includes a crawl header.
- *Os Lusíadas*, Gutenberg eBook 3333 ↔ HPLT2: 56,372/56,855 words,
  Jaccard **0.99171**.
- *Chronica d'el rei D. Diniz*, Gutenberg eBook 18167 ↔ FineWeb2:
  25,574/25,570 words, Jaccard **0.99889**, with matching volume and
  National Library of Portugal provenance in the excerpts.

The bounded excerpts show web wrappers and, for the HPLT2 examples, a generic
newsletter header and unrelated footer text. The full-record shingle scores
show near-equivalent body text; for *Dom Casmurro*, all unique Gutenberg
shingles occur in the HPLT2 text. That supports same-work identity and favors
the curated Gutenberg record, but it does not establish identical edition,
ordering, or absence of a small substantive difference. The available
GigaVerbo URLs are dataset-level rather than per-document origins. Inspect
complete candidate texts and confirm author/title, edition, and missing
material before suppressing a web record.

Held-out *Sala das Pérolas* (Gutenberg/HPLT2) has Jaccard **0.94589** and
matching book evidence. It is a likely copy below P2's proposed `0.98` screen.
D2b's *Os Filhos do Padre Anselmo* Gutenberg/HPLT1 anchor is another likely
copy at **0.8515**. These are deliberate conservative false-negative risks;
do not lower a threshold without reviewing edition and content differences.

### Templates, short text, and containment

The 306 `boilerplate_suspect` flags are a heuristic, not ground truth. It
flags some likely Carolina/Wikipedia copies, while high-Jaccard false
positives include taxonomic bibliographies and distinct legal cases. Carolina
judicial records ADI 4.578 and ADC 30 have Jaccard **0.99089** but different
dockets, parties, and legal questions. Very short Carolina social records can
also score `1.0` when only a few formulaic words and emoji differ. Neither a
global cutoff nor a minimum word count alone is safe: the 120-word taxonomic
false positive passes a plausible body-length floor, and same-length
different-entity documents have length ratio `1.0`.

D2c marks 3,874 exact-scored pairs with a containment diagnostic. The
`containment_summary.csv` reports overlapping conditions: 3,728 pairs with at
least 100 shared shingles and Jaccard below 0.80, 1,298 with smaller-document
containment at least 0.90 and Jaccard below 0.80, and 383 with containment at
least 0.95 and length ratio below 0.50. These are review signals, not
duplicate labels. D2b likewise found repeated legal front matter with many
shared shingles but little distinctive body overlap. No corpus-wide shingle
document-frequency or distinctive-overlap cutoff was computed. Shared and
unique shingle counts are preserved in the packet, but they do not identify
which shared shingles are boilerplate.

ParlamentoPT remains **preserve-all**. Its 23,483,287 within-source candidate
pairs and 1,923 exact-scored sample rows are diagnostic. Short procedural text
and recurring parliamentary language do not justify deleting utterances; all
policy simulations report zero ParlamentoPT removals.

## Bucket saturation and recall limits

The 256-member cap encountered 4,083 overflow buckets in `word5_128_32x4` and
571 in `word5_128_8x16`; the maximum bucket size was 37,256. The recorded
omitted-pair upper bounds are 28,764,758,904 and 7,016,124,562 pair
occurrences, respectively. They are not unique omitted relationships: a pair
can occur in multiple buckets/configurations, and another bucket may emit it.
The counts establish material bucket saturation, but cannot yield unique
missed pairs or population recall. LSH itself can miss pairs outside emitted
buckets. Do not claim complete population-scale recall, enumerate omitted
pairs, or infer duplicate prevalence from these measurements.

## Policy comparison and sample-only scenarios

All scenarios use only the 8,014 already exact-scored pairs. A relationship is
accepted only from its own exact-scored pair. The deterministic simulation
selects a retained owner by source preference and stable record ID, and skips
an edge if its proposed representative has already been removed. Every
hypothetical removal in `sample_policy_removal_edges.csv` names its direct pair
and a representative that remains retained. The semantic relationship is
still provisional; these counts are not deletion authorizations.

| Policy | Eligible relationships | Sample accepted pairs | Unique affected records | Direct pair-supported removals | Normalized words removed | Assessment |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| P0 — preserve all | None | 0 | 0 | 0 | 0 | Safe baseline; genuine copies remain. |
| P1 — global exact Jaccard `>=0.95` | All non-Parlamento sources | 203 | 401 | 200 | 2,538,875 | Unsafe: accepts distinct Wikipedia and legal records, and short/formulaic text. |
| P2 — curated/web screen | Verified Carolina/wik ↔ Wikipedia PT; Gutenberg ↔ GigaVerbo FineWeb2 or HPLT2 | 31 | 62 | 31 | 812,914 | Recommended direction, but human identity/content checks remain. |
| P3 — broader selective | P2 plus long GigaVerbo/GigaVerbo pairs passing stricter length, containment, and flag screens | 42 | 84 | 42 | 1,080,392 | Not validated; adds 11 web/web removals (267,478 words) in this sample. |

P2's 31 sample relations comprise 15 Carolina/wik→Wikipedia pairs and 16
Gutenberg→GigaVerbo pairs. Its hypothetical removals are 15 Carolina/wik
records (16,866 words), five FineWeb2 records (80,994 words), and 11 HPLT2
records (715,054 words). P2's book screen is exact Jaccard `>=0.98`, length
ratio `>=0.90`, smaller-document containment `>=0.99`, and at least 1,000
words. Its article screen is exact Jaccard `>=0.95`, length ratio `>=0.90`,
containment `>=0.95`, at least 100 words, and the Wikipedia title present in
both bounded texts. These values are review-screen predicates, not a frozen
production threshold; direct identity and content equivalence must still be
confirmed.

P0 changes no mix. P2 would reduce only the eligible web records in the
GigaVerbo residual pool while preserving the Gutenberg and Wikipedia PT
representatives; P1 could remove across curated and web roles, and P3 could
thin GigaVerbo subset/domain coverage. The A/B/C nominal source shares do not
change in this review. Because the pair sample is enriched, no corpus word
loss or mix availability change can be estimated from it.

## Decision and ownership rule

Recommend **B — selective near-deduplication** with exactly these potentially
eligible relationships:

1. **Gutenberg PT `literature` ↔ GigaVerbo `fineweb_2_pt` or `hplt2_pt`** for
   a verified copy of the same literary work, with the P2 screen above and a
   full-text check for edition differences or substantive omissions. Retain
   Gutenberg. Other GigaVerbo subsets, including HPLT1, FinePDFs, MC4,
   Common Crawl, crawlPT, Quati, and Blogset, remain excluded until a specific
   reviewed relationship supports them.
2. **Carolina `wik` ↔ Wikipedia PT `articles`** only when the same canonical
   article entity is established from reliable Wikipedia metadata and matching
   body text, the P2 screen is met, and a complete-text check finds no
   meaningful unique facts or sections. Retain Wikipedia PT. An opaque WIK ID
   or a high score alone is insufficient.

All other relationships remain preserve-all: Wikipedia↔Wikipedia,
Carolina/wik↔Carolina/wik, other Carolina classes, generic web↔web, legal or
form-like documents, short text, and all ParlamentoPT records. Do not delete by
connected-component membership. If A~B and B~C but A~C was not directly
verified, C must remain unless it has its own qualifying direct edge to a
retained representative. Within a genuine multi-copy set, choose a native
representative first, apply documented GigaVerbo provenance tiers only within
eligible web copies, and use stable occurrence ID only as the final tie-break.

The evidence rejects global score-only deletion, but supports a narrow
curated/web policy direction. It is **not sufficient to freeze production**:
the labels are provisional and the held-out cases show both false-positive
risk and likely copies below P2 screens. Complete the short case checklist in
`unresolved_cases.md`; then freeze or revise only these two source
relationships. No new corpus-scale experiment is proposed.

## Reproducibility and status

`manifest.json` records the pinned input identity, source artifact checksums,
deterministic selection/split rules, and output checksums. `generate_review.py`
rebuilds the packet, labels, policy matrix, sample simulation, direct-edge
removal list, and summary from the completed D2c Parquet/CSV/JSON artifacts.
The exact-dedup output was not modified. Near-dedup production was not run;
benchmark decontamination remains pending; C1 remains **IN PROGRESS** and C2
remains **PENDING**.
