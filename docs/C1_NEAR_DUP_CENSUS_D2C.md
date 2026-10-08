# Gate C1 Full-Corpus Near-Duplication Census (D2c)

**Implementation:** complete
**Full-corpus census:** COMPLETE / PASS; stages and artifacts verified against the pinned exact input
**Near-dedup production:** NOT RUN
**Input:** `/mnt/data/cambacica-base-180m/deduplicated/exact/data`
**Exact manifest SHA-256:** `57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428`
**Output root:** `/mnt/data/cambacica-base-180m/dedup-census/near-v1/`

## Purpose and limits

D2 and D2b did not provide enough independent positive pairs to freeze a
global near-dedup rule. D2c measures candidate volume and its concentration
over the complete immutable post-exact corpus, then exact-scores a deterministic
stratified candidate sample for human characterization. It does not select
owners, remove records, write a retained corpus, freeze a threshold, or start
benchmark decontamination. ParlamentoPT candidates are diagnostic-only and
always have `removal_eligible=false`.

Every command checks the authoritative exact manifest SHA-256 before starting
work. A smoke fixture can use the Python API with an explicit `None` expected
digest; the CLI always defaults to the pinned production digest.

## Representation and LSH choices

Fingerprints reuse D2/D2b's stable representation: lowercased Unicode
`re.\w+` word tokens, 5-word shingles, xxHash64 shingle hashes, and the first
128 values from D2/D2b's deterministic `word5_256` affine MinHash family. Text
shorter than five words becomes one whole-text shingle. Empty records are
counted and kept in the fingerprint artifact, then omitted from LSH because
they have no shingles. All nonempty short records, including ParlamentoPT,
enter the diagnostic candidate index.

The census uses the union of two configurations over the same 128-value
signature:

| Configuration | LSH shape | Why it is included |
| --- | ---: | --- |
| `word5_128_32x4` | 32 bands × 4 rows | Broad candidate generation. D2 produced 91 pairs and D2b measured 94; this is the cost-conscious configuration D2b recommended for further population measurement. |
| `word5_128_8x16` | 8 bands × 16 rows | Strict volume bracket. D2 produced 4 pairs, versus 91 for 32×4, showing how sharply the pilot candidate population changes with banding. |

D2b's exhaustive-gold recall denominator was one positive pair. Both configs
recalled it, but that result is too small to establish production recall. The
union records which config generated each pair. It is not a production rule.

Buckets are indexed on disk in SQLite. A bucket with at most 256 members emits
all its pairs. For a larger bucket, the candidate generator uses the first 256
cryptographic occurrence IDs in stable order, reports the full bucket size,
and records an upper bound on omitted pair combinations. This keeps work per
bucket bounded and makes any candidate loss visible. Counts describe this
bounded candidate generator; overflow metrics must be reviewed before drawing
population conclusions.

## Stages and restart behavior

Each stage publishes a complete directory by atomic rename and writes a
manifest with input identity, row counts, resource metrics, and artifact
checksums. A rerun reuses a completed stage with the same input manifest.
Interrupted `.partial-*` directories for that stage are removed on its next
run. A completed stage with a mismatched manifest is left in place and the
command stops rather than overwriting it. Separate `flock` locks prevent two
copies of the same stage from running at once.

1. `signatures/`: stream exact Parquet files in sorted relative-path order;
   store 128-value signatures, stable occurrence IDs, source/subset, word
   counts and bands, original URL/title, and row-group locators.
2. `lsh-index`: build the disk-backed bucket postings for both configs.
3. `candidates`: enumerate capped buckets, suppress duplicate pairs,
   recover candidate endpoint metadata, and calculate the 128-value MinHash
   estimate for every generated pair. It does not exact-score the population.
4. `candidate_summary/`: write candidate counts by source/subset, source pair,
   document length band, length-ratio band, estimated-similarity band, and
   within/cross-source relationship.
5. `candidate_samples/`: choose deterministic bottom-k samples across those
   dimensions, include every candidate in source/subset pair strata of at most
   24 pairs and top-level source-pair strata of at most 64 pairs, retrieve only
   selected texts, exact-score those pairs, and publish a bounded review panel.

The sample also covers LSH config and bucket-size strata, repeated/wide-bucket
boilerplate proxies, same-domain candidates, and curated/web, web/web,
Carolina/Wikipedia, Gutenberg/web, short, and giant pairs when present. Exact
scores include both directional containments, length ratio, word counts,
unique/shared/union shingle counts, source and URL provenance, and bounded
start/middle/end excerpts. Candidate and exact-score samples are stratified,
not probability samples; their counts are evidence for review and do not
estimate corpus prevalence. No corpus-wide shingle document-frequency table
or frozen distinctive-overlap cutoff is built.

`near-census verify` rechecks the exact manifest, every stage manifest and all
published artifact hashes, then checks the fingerprint count against the exact
manifest. It is intended for the final manual verification and reads the large
signature/index artifacts to hash them.

## Measured full-census resources

The completed stage resource reports record these measurements:

| Stage | Wall seconds | Peak RSS bytes | Measured output / count |
| --- | ---: | ---: | --- |
| Fingerprints | 25,807.781 | 7,509,233,664 | 21,603,689 records; 22,857,453,805-byte signature Parquet |
| LSH index | 59,207.704 | 4,421,447,680 | 864,147,560 postings; 31,879,938,048-byte SQLite index |
| Candidate enumeration | 20,652.954 | 4,964,134,912 | 65,412,497 unique candidate pairs; 6,650,577 endpoints; 5,625,425,920-byte candidate database |
| Candidate summary | 958.279 | 77,086,720 | 65,412,497 candidate pairs summarized |
| Exact-scored sample | 4,042.118 | 771,731,456 | 8,014 pairs; 15,431 text endpoints |

The `full_corpus_planning` block in the sample resource report was written
before the full run and remains a planning note; use the measured stage values
above instead. D2 pilot projections are not census measurements or estimates
of duplicate prevalence.

## Full-census stage commands

The following commands document the reproducible stage interfaces. D2c is
already complete; this D2d review did not rerun any stage or invoke a deletion
command. Each completed stage is manifest-checked and reused on a matching
rerun.

```bash
cd ~/dev/cambacica-base-180m
PYTHONPATH=src python3 -m cambacica.corpus.dedup.near_census fingerprints \
  --input-root /mnt/data/cambacica-base-180m/deduplicated/exact \
  --output-root /mnt/data/cambacica-base-180m/dedup-census/near-v1 \
  --expected-manifest-sha256 57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428
PYTHONPATH=src python3 -m cambacica.corpus.dedup.near_census lsh-index \
  --input-root /mnt/data/cambacica-base-180m/deduplicated/exact \
  --output-root /mnt/data/cambacica-base-180m/dedup-census/near-v1 \
  --expected-manifest-sha256 57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428
PYTHONPATH=src python3 -m cambacica.corpus.dedup.near_census candidates \
  --input-root /mnt/data/cambacica-base-180m/deduplicated/exact \
  --output-root /mnt/data/cambacica-base-180m/dedup-census/near-v1 \
  --expected-manifest-sha256 57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428
PYTHONPATH=src python3 -m cambacica.corpus.dedup.near_census summarize \
  --input-root /mnt/data/cambacica-base-180m/deduplicated/exact \
  --output-root /mnt/data/cambacica-base-180m/dedup-census/near-v1 \
  --expected-manifest-sha256 57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428
PYTHONPATH=src python3 -m cambacica.corpus.dedup.near_census exact-sample \
  --input-root /mnt/data/cambacica-base-180m/deduplicated/exact \
  --output-root /mnt/data/cambacica-base-180m/dedup-census/near-v1 \
  --expected-manifest-sha256 57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428
PYTHONPATH=src python3 -m cambacica.corpus.dedup.near_census verify \
  --input-root /mnt/data/cambacica-base-180m/deduplicated/exact \
  --output-root /mnt/data/cambacica-base-180m/dedup-census/near-v1 \
  --expected-manifest-sha256 57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428
```

The `lsh` convenience command runs `lsh-index` and `candidates` in sequence;
the separate commands let an operator inspect or resume either heavy stage.
Candidate generation reuses the completed index after an interruption. Final compact artifacts
include `candidate_summary/candidate_counts_by_source_subset.csv`,
`candidate_counts_by_pair.csv`, `candidate_counts_by_similarity_band.csv`,
`candidate_samples/exact_scored_sample.parquet`,
`candidate_samples/exact_sample_strata.csv`,
`candidate_samples/exact_score_summary.csv`,
`candidate_samples/containment_summary.csv`,
`candidate_samples/review_pairs.parquet`,
`candidate_samples/census_summary.json`, and
`candidate_samples/resource_report.json`.

The census report is intended to support all three outcomes: broader
near-dedup calibration/production, a conservative policy limited to evidence-
supported source classes, or skipping global near-dedup in favor of benchmark
decontamination and split design. It makes no outcome decision itself.
