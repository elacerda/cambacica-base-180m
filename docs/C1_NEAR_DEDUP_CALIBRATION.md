# Gate C1 Near-Dedup Calibration (D2b)

**Status:** calibration complete; production contract not ready to freeze
**Input:** immutable post-exact corpus, 21,603,689 retained rows
**Exact manifest SHA-256:** `57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428`
**Seed:** `20261006`
**Calibration output:** `/tmp/cambacica-base-180m/dedup-pilots/near-calibration-v1/`

D2b completed a targeted, read-only calibration. It did not write retained
corpus data, perform near-dedup deletion, or run benchmark decontamination.
The D2 start-commit reporting typo and its Git-history resolution are recorded
in [`C1_NEAR_DEDUP_PILOT.md`](C1_NEAR_DEDUP_PILOT.md).

## Calibration panel

The panel contains the 5,525-record D2 pilot plus 2,399 records selected from
deterministic title/URL families and neighboring crawl records: 7,924 records
and 2,133 unique scored pairs. It includes all five top-level sources and all
11 GigaVerbo subsets. The GigaVerbo counts are: `blogset` 252,
`common_crawl` 470, `crawlPT_dedup` 558, `culturax` 383,
`finepdfs_por_Latn` 582, `fineweb_2_pt` 1,264, `hplt1_pt` 475,
`hplt2_pt` 1,093, `mc4_pt` 762, `oscar` 419, and `quati` 264. The other
source totals are Carolina 971, Gutenberg 79, ParlamentoPT 176, and Wikipedia
PT 176.

The calibration found one probable real high-similarity pair: Gutenberg
`Os Filhos do Padre Anselmo` and GigaVerbo `hplt1_pt`, with 115,482 and
118,870 words, length ratio 0.9715, exact 5-gram Jaccard 0.8515, 112,529
shared shingles, and directional containment 0.9318 / 0.9082. Its 256-value
MinHash estimate was 0.8125. The excerpts and source URLs support a likely
Gutenberg-to-HPLT copy, but the review artifact leaves `human_label` blank;
this remains one probable example, not a human-confirmed truth set.

There were 193 exhaustive gold pairs: 192 pairs from selected small metadata
families plus this independently identified D2 positive anchor. All pairs
inside each selected family were scored exactly. The families did not yield
another exact word-5 pair at Jaccard 0.80 or above. This enriched set is for
behavioral calibration and must not be used to estimate corpus prevalence.

Metadata limits what the family search can establish. GigaVerbo `original_url`
values point to dataset pages on `huggingface.co`, and its
`upstream_metadata_json` has no per-document crawl URL. Carolina rows in this
panel also have no `original_url`. Consequently, D2b could not form valid
same-website page families for those records; the shared dataset host is not a
web-page domain. Same-title, URL-variant, and neighbor checks were still run
where the available fields permitted them.

## Similarity and candidate generation

At exact word-5 Jaccard thresholds 0.80 and 0.85, the exhaustive gold set has
one positive pair, and each tested word-5 LSH configuration recalled it (1/1).
The word-3 LSH configuration also recalled its one positive at both thresholds
(1/1). Word-7 recalled its one positive at 0.80 (1/1); it had no exact
word-7-positive at 0.85. At 0.90, 0.92, and 0.95 there are no positive gold
pairs, so recall is undefined, not 100%. The measured denominator is one, so
these results do not establish production candidate-generation recall.

All three pairs at exact word-5 Jaccard 0.80 or higher were in the 0.90–1.00
length-ratio band. Two were short ParlamentoPT formulas and remain preserved;
the other was the Gutenberg/HPLT1 pair. The only pair at 0.90 or higher was a
short ParlamentoPT formula. This small, enriched set supports checking length
ratio alongside Jaccard, but cannot calibrate a population-wide threshold.

| Representation / configuration | Mean absolute MinHash error | LSH candidates | Raw signature size per row |
| --- | ---: | ---: | ---: |
| 3-word shingles, 128 values, 32×4 | 0.00471 | 172 | 1 KiB |
| 5-word shingles, 64 values, 16×4 | 0.00505 | 53 | 512 B |
| 5-word shingles, 128 values, 32×4 | 0.00538 | 94 | 1 KiB |
| 5-word shingles, 256 values, 32×8 | 0.00342 | 15 | 2 KiB |
| 7-word shingles, 128 values, 32×4 | 0.00368 | 107 | 1 KiB |

The error values compare estimates on the 2,133 scored pairs, not independent
test data. Five-word shingles are the simplest provisional representation:
they retain the Gutenberg/HPLT1 example at 0.85, while 3-grams generated more
candidate pairs and 7-grams dropped that pair below 0.85. The 128-value
5-gram signature with 32×4 bands is a reasonable cost-conscious candidate
generator for further calibration, but the one-pair recall result is too weak
to freeze it. On 21.6 million rows, raw signatures alone would require about
20.6 GiB at 128 values or 41.2 GiB at 256 values, before index overhead and
compression.

## Length, containment, and boilerplate

Containment is useful as a review signal and unsafe as a deletion rule. One
Carolina legislative pair has 132 words embedded in a 306-word document:
containment 0.9752, but Jaccard 0.4604 and distinctive Jaccard 0.0. A second
has 322 words embedded in 5,823 words: containment 0.9869, Jaccard 0.0789,
and length ratio 0.0553. A shared-shingle floor alone would not protect the
first case: it shares 157 word-5 shingles that are recurring legislative
front matter. Six pairs were flagged for containment review; none of the gold
positive pairs was recovered by containment alone.

The same legislative boilerplate appears in a 132/192-word pair with exact
Jaccard 0.7854 and 161 shared word-5 shingles, but only four shared shingles
remain after the calibration's document-frequency filter (distinctive Jaccard
0.0833). The adversarial unrelated-body control reached Jaccard 0.9371 and
MinHash 0.9141; after excluding shingles with document frequency at least
three in its four-document reference cohort, distinctive Jaccard fell to
0.0. That result explains D2's high-Jaccard synthetic failure and shows why
plain Jaccard can accept a shared template. The real Gutenberg/HPLT1 pair
retains distinctive Jaccard 0.8515, so the mitigation is promising. Its
frequency cutoff came from a capped calibration reference and is not a
production cutoff.

Shorter-document behavior supports provisional bands, not frozen policy:

| Shorter document | Scored pairs | Exact word-5 J ≥ 0.80 | Mean absolute MinHash error (64 / 128 / 256) | Provisional handling |
| --- | ---: | ---: | ---: | --- |
| Under 20 words | 33 | 2, both ParlamentoPT | 0.0513 / 0.0405 / 0.0263 | Exact-only or diagnostic |
| 20–99 words | 19 | 0 | 0.0147 / 0.0073 / 0.0050 | Review-only |
| 100–999 words | 1,822 | 0 | 0.0043 / 0.0049 / 0.0030 | Normal candidate scoring plus containment diagnostics |
| 1,000–99,999 words | 258 | 0 | 0.0033 / 0.0039 / 0.0029 | Normal candidate scoring plus containment diagnostics |
| 100,000+ words | 1 | 1 | 0.0391 / 0.0156 / 0.0390 | Normal scoring plus containment diagnostics; more evidence needed |

The 100,000+ result is the Gutenberg/HPLT1 example alone. Short-text bands
remain provisional. All 176 sampled ParlamentoPT rows are preserved; two
short high-similarity formula pairs are diagnostic examples, not deletion
candidates.

## Decision matrix

The following precision/recall figures use the 60-pair review artifact's
evidence-based provisional labels, after applying the mandatory
ParlamentoPT-preserve-all rule. Each policy accepts only one review pair at
the reference threshold 0.85; it is the probable Gutenberg/HPLT1 pair. Thus
the displayed 1/1 values are not reliable estimates. Gold similarity recall
is also 1/1 at exact word-5 Jaccard 0.85 for every policy; its denominator is
one. No policy has a valid measured recall at 0.90 or above.

| Policy | Review precision / recall | Gold similarity recall at 0.85 | Failure modes and compute implications | Assessment |
| --- | --- | --- | --- | --- |
| P1: global Jaccard | 1/1 / 1/1 | 1/1 | Lowest cost; boilerplate can score highly, short edits and containment are poorly handled. Two ParlamentoPT pairs at J ≥ 0.80 remain diagnostic and are preserved. | Insufficient alone |
| P2: Jaccard + short guard | 1/1 / 1/1 | 1/1 | Same index cost as P1; short near-copies may be missed and boilerplate remains. | Safer short-text handling, still insufficient |
| P3: P2 + containment diagnostics | 1/1 / 1/1 | 1/1 | Adds exact directional overlap on candidates; six diagnostic flags here, with no gold-positive recall gain. | Useful review signal; never delete by containment alone |
| P4: P3 + distinctive overlap | 1/1 / 1/1 | 1/1 | Best observed boilerplate separation; highest cost because it needs document-frequency aggregation. Sample-local frequency can misclassify common content. | Promising, but needs corpus-scale DF calibration |

## Provisional proposal and remaining evidence

Do not freeze a production deletion threshold from D2b. For the next
calibration, use 5-word shingles with a 128-value MinHash candidate index and
exact word-5 scoring on candidates; compare 256 values if the extra signature
storage is acceptable. Use Jaccard as the main score, with a short-text guard
(under 20 words: exact-only/diagnostic; 20–99: review-only) and directional
containment as a review diagnostic for longer records. Treat distinctive
overlap as a boilerplate warning and review feature until its document-
frequency reference and minimum distinctive-content requirements are
validated on broader real pairs. A Jaccard threshold of 0.85 is only a
reference point for additional review sampling, not a production cutoff.

Near-duplicate ownership remains provisional: prefer a native or curated
source over a generic web copy, follow the documented GigaVerbo provenance
tiers, and use occurrence ID only as the final deterministic tie-break. For
clearly Wikipedia-derived Carolina `wik` material, Wikipedia PT is the
provisional canonical owner. The inspected Carolina/Wikipedia pair is not a
duplicate: the Wikipedia record is about *Arborimus pomo* (51 words), while
the Carolina row is about *Achondrostoma arcasii* (67 words); Jaccard is
0.0467 with five shared shingles. The exact-dedup decisions, including the 125
pairs retained under Carolina alphabetical ordering, remain unchanged.

The 60-row human-review artifact contains 52 ambiguous pairs, six containment
cases, one boilerplate false positive, and one probable duplicate. Its labels
are triage hints only; `human_label` is blank. More independently confirmed
near copies are needed across same-site pages, source pairs, length bands, and
threshold boundaries. GigaVerbo's current metadata cannot supply per-page
URLs for same-site sampling. This is why D2b is complete as a calibration
exercise but the near-dedup contract is **NOT READY TO FREEZE**.

## Validation and artifacts

The separate output root contains `manifest.json`, the 7,924 calibration
records, 2,133 scored/candidate pairs, 193 exhaustive gold pairs, the 60-pair
human-review artifact, threshold recall, length-ratio, containment,
boilerplate, short-document, configuration, policy, signature, and resource
reports. Its total size is about 114 MB. The manifest and every listed
artifact checksum verify successfully. The run scanned 21,603,689 metadata
rows in 393 seconds, used 2.65 GB peak RSS, and completed in 857 seconds wall
time (1,161 CPU seconds). These are D2b measurements, not production
near-dedup resource estimates.

Repository validation is recorded with the D2b commit. Gate C1 remains
**IN PROGRESS**, C2 remains **PENDING**, benchmark decontamination remains
**PENDING**, and production near dedup remains **NOT RUN**.
