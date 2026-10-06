# Gate C1 Near-Deduplication Pilot (D2)

**Status:** pilot complete; near-dedup production not run
**Pilot version:** 0.1.0
**Seed:** `20261006`
**Input:** `/mnt/data/cambacica-base-180m/deduplicated/exact/data`
**Exact manifest SHA-256:** `57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428`
**Output:** `/mnt/data/cambacica-base-180m/dedup-pilots/near-v1/`

The pilot reads only the exact-deduplicated corpus. It does not delete or
rewrite any production row. All 176 sampled ParlamentoPT records remain in the
pilot, and none is marked removable. The exact input manifest and every pilot
artifact checksum were verified after the run.

## Sample and representation

The sampling frame was every retained row in exact `record_resolution.parquet`.
Rows were stratified by source, subset, and normalized-word count: short
0–19, medium 20–999, long 1,000–99,999, and giant at least 100,000 words. A
fixed-seed circular sample over the exact SHA-256 occurrence IDs selected up
to 48 short, 80 medium, 48 long, and 12 giant records per nonempty cell. The
sample contained 2,779 base records: 247 short, 1,538 medium, 888 long, and
106 giant. All five top-level sources and all 11 GigaVerbo subsets are present.
Base counts by top-level source were Carolina 971, GigaVerbo 1,377, Gutenberg
79, ParlamentoPT 176, and Wikipedia PT 176. GigaVerbo subset counts are in the
manifest and `source_subset_summary.csv`.

To enrich likely crawl neighbors, the pilot also sampled adjacent exact-data
rows when they shared a GigaVerbo subset, upstream shard, and upstream row
group. This added 2,746 diagnostic records, for 5,525 total records. It added
699 reason-tagged diagnostic pairs: 694 neighboring crawl pairs, four
same-domain pairs, and one repeated ParlamentoPT exact-hash pair. Source and
subset totals retain their original provenance. These deliberate additions
and the per-cell quotas make the sample unsuitable for population prevalence
claims.

The baseline shingle unit is lowercased Unicode word tokens from Python
`re.\w+`, independent of the future Cambacica tokenizer. The pilot measured
3-, 5-, and 7-word shingles, with 128, 256, and 128 MinHash permutations
respectively. Shingles use stable xxHash64 values and deterministic 64-bit
affine permutations. Text shorter than the shingle size is represented as one
whole-document shingle; empty text receives a fixed sentinel signature.

Ten LSH configurations compared 64/128/256 prefixes for 5-grams across
different band sizes, plus 3-gram/128 and 7-gram/128 controls. SQLite-backed
buckets cap a bucket at 256 members. The largest observed bucket had seven
members; no bucket hit the cap and no pathological mega-bucket occurred.

## Candidate and threshold results

Across the 5,525 records, LSH generated 219 unique pairs across all tested
configurations. The 5-gram LSH union generated 92 pairs, including 84 pairs
whose two records both came from the stratified base sample. Diagnostic
enrichment contributed 697 additional scored pairs after overlap with LSH;
the candidate and scored artifacts contain 916 unique pairs total. At
threshold 0.80, 5-gram candidate endpoints were concentrated in Carolina
legal records (124 endpoints), Carolina judicial (14) and Carolina Wikipedia
(22); GigaVerbo HPLT1 (11), HPLT2 (2), and Oscar (4); Gutenberg (1),
ParlamentoPT (4), and Wikipedia PT (2). Most Carolina legal candidates did
not pass the threshold.

The table below shows the 5-gram LSH union sweep. Acceptance uses the 256-value
MinHash similarity estimate. Full per-configuration and source/subset counts
for all five thresholds are in `threshold_summary.csv` and
`source_subset_summary.csv`.

| Threshold | Candidates | Accepted | Clusters | Documents | Within / cross | Removals* |
|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 92 | 2 | 2 | 4 | 1 / 1 | 1 |
| 0.85 | 92 | 1 | 1 | 2 | 1 / 0 | 0 |
| 0.90 | 92 | 1 | 1 | 2 | 1 / 0 | 0 |
| 0.92 | 92 | 1 | 1 | 2 | 1 / 0 | 0 |
| 0.95 | 92 | 1 | 1 | 2 | 1 / 0 | 0 |

*Hypothetical removable documents under either tested ownership policy;
ParlamentoPT is preserved under both.

At 0.80 the cross-source pair is Gutenberg/HPLT1; its exact 5-gram Jaccard is
0.852, while its 256-permutation estimate is 0.812. It passes at 0.80 and is
rejected at 0.85. The other accepted pair is an identical short ParlamentoPT
utterance. At 0.85 and above that preserved ParlamentoPT pair is the only
accepted pair. The sweep therefore does not establish a useful production
threshold by itself.

Ten configurations produced these unique candidate counts; the five
following numbers are accepted pairs at thresholds 0.80, 0.85, 0.90, 0.92,
and 0.95, in that order:

| Configuration | Candidate pairs | Accepted pairs by threshold |
|---|---:|---|
| 3-gram, 128, 32 × 4 | 154 | 3, 2, 2, 1, 1 |
| 5-gram, 64, 16 × 4 | 52 | 3, 2, 1, 1, 1 |
| 5-gram, 64, 8 × 8 | 8 | 3, 2, 1, 1, 1 |
| 5-gram, 128, 32 × 4 | 91 | 3, 1, 1, 1, 1 |
| 5-gram, 128, 16 × 8 | 11 | 3, 1, 1, 1, 1 |
| 5-gram, 128, 8 × 16 | 4 | 3, 1, 1, 1, 1 |
| 5-gram, 256, 32 × 8 | 15 | 2, 1, 1, 1, 1 |
| 5-gram, 256, 16 × 16 | 4 | 2, 1, 1, 1, 1 |
| 5-gram, 256, 8 × 32 | 1 | 1, 1, 1, 1, 1 |
| 7-gram, 128, 32 × 4 | 106 | 2, 1, 1, 1, 1 |

## Review and adversarial validation

`review_pairs.parquet` contains 25 pairs with source/subset, occurrence IDs,
word counts, MinHash estimates, exact shingle Jaccard and containment,
overlap counts, short excerpts, and common/differing shingle examples. Its
estimate bands contain 21 pairs below 0.75, two in 0.75–0.80, one in
0.80–0.85, and one in 0.95–1.00; there were no observed pairs in
0.85–0.95. The review set includes 15 neighboring crawl pairs, three short
ParlamentoPT formula cases, one Wikipedia/Carolina pair, and one
Gutenberg/web pair. Exact-title enrichment found no same-title pair in the
base sample. A separate read-only title probe over its 1,147 nonempty titles
used rare-token postings (document frequency at most 80) to form 17 candidate
pairs. None passed the screen of character `SequenceMatcher` at least 0.72 or
title-token Jaccard at least 0.60, after a loose overlap prefilter. That probe
was not added to the pair artifacts, so fuzzy title matching remains outside
the core pilot. Most reviewed neighbor pairs had exact Jaccard 0.0, showing
that crawl adjacency is useful for review sampling but is not evidence of
duplication.

Nine known-answer synthetic pairs were evaluated over ten LSH configurations
and five thresholds (450 decisions). At threshold 0.80, the grid yielded 30
true-positive, 20 false-positive, 10 false-negative, and 30 true-negative
configuration/case decisions. At 0.90 the counts were 29, 15, 11, and 35;
at 0.95 they were 20, 12, 20, and 38. Across the full grid there were 77
false-positive and 63 false-negative decisions. These are adversarial test
counts, not prevalence estimates.

The template with unrelated bodies had exact Jaccard 0.937 and triggered all
ten LSH configurations at 0.80; eight still triggered at 0.85 and five at
0.90. Reordered paragraphs had exact Jaccard 0.978 and were detected at every
tested threshold. A four-word sentence edit had Jaccard 0.0 under the
short-document whole-shingle rule and was missed by all configurations. A
160-word document contained within a much larger document had Jaccard 0.057
but containment 1.0 and was also missed. This confirms that ordinary MinHash
Jaccard does not solve containment and that short text needs a separate
policy.

Two short ParlamentoPT phrases became LSH candidate pairs. The identical
repeated utterance scored 1.0; a procedural variant scored exact Jaccard 0.800
but estimated 0.781 and fell below the 0.80 estimate threshold. A third
short-formula example entered through same-domain diagnostics. All Parliament
rows remain preserved. Candidate similarity among procedural utterances must
not be interpreted as duplicate documents.

## Ownership and resources

Ownership remains provisional. At 0.80, the eligible Gutenberg/HPLT1 pair
would remove one row under either tested hypothesis, but the owner changes:
`native_then_aggregator` keeps Gutenberg, while `longest_non_parlamento` keeps
the longer HPLT1 record. At 0.85 and above, only the ParlamentoPT pair
remains, with zero hypothetical removals. No ownership hierarchy is frozen.

The complete run took 409.8 seconds wall time and 721.2 CPU seconds, with
1.78 GB (1.65 GiB) peak RSS and an 80.99 MB output footprint before the manifest. It
enumerated all 21.60M exact rows using 2.45 GB of compressed resolution
columns and 1.48 GB of compressed exact-data metadata columns. Three
signatures processed 28.87M normalized words each. Measured 5-gram/256
throughput was 130 signatures/second (42.4 CPU seconds for this sample). The
disk-backed LSH bucket index used 254.7 MB for 1.105M entries and completed in
11.2 seconds. These are measurements from one run on the current storage host.

For planning only, scaling the measured 5-gram throughput to 128 permutations
gives 4.6–7.4 CPU hours for the retained corpus, about 21.0 GB of compressed
5-gram/128 signatures, and a 60–120 GB 16-band SQLite index scenario. A linear
candidate-pair scenario of 0.65M, with a 10× sensitivity case of 6.53M, is
not a population estimate: the sample deliberately oversamples length/source
cells and includes diagnostic neighbors. Full production will read all text,
and needs a bounded, disk-backed implementation; these estimates do not
measure production runtime or I/O.

## Decision

Keep 0.80–0.90 as the next manual-calibration range, not a production policy.
The exact Jaccard of the Gutenberg/HPLT1 review pair exceeded 0.85 while its
MinHash estimate fell below 0.85; the synthetic template also creates false
positives at the lower end. Threshold, short-document handling, boilerplate
normalization, containment checks, LSH recall, and near-cluster ownership
still need review. This pilot is **not ready to freeze the near-dedup
production contract**. Near-dedup production was not run, benchmark
decontamination remains pending, C1 remains in progress, and C2 remains
pending.
