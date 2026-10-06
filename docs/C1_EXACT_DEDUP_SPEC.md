# Gate C1 Exact Deduplication Contract

**Version:** 1.0.1
**Status:** frozen
**Gate:** C1 — IN PROGRESS

This contract removes complete normalized records only when their verified
`content_sha256` values are identical. It does not match passages, substrings,
or semantic similarity. Near deduplication is a later, separately calibrated
stage.

## Identity and record units

These three identities serve different purposes:

- **Normalized occurrence identity:** `record_id` uses identity version
  `occurrence-id-v2`, a namespaced SHA-256 over `source`, `normalized_shard`,
  and the zero-based row ordinal within that immutable normalized Parquet
  shard. The ordinal is continuous across Parquet batches and row groups. It
  resets for each shard, so IDs do not depend on traversal order across files.
  The frozen normalized artifact and its manifest define the shard boundary;
  no extra manifest key is needed in the occurrence hash.
- **Upstream/source identity:** `original_id`, `raw_record_identifier`, raw
  source file, source revision, and available upstream shard/row-group/commit
  fields remain provenance. They are preserved in resolution mappings and do
  not identify a unique normalized occurrence. The GigaVerbo `common_crawl`
  row-group materialization `subset=common_crawl/train-00007-of-00056__row-group-00028.parquet`
  contains 4,192 rows, where `original_id` /
  `raw_record_identifier` `78c9db87c96264dc24a941dfeb0c8ed4` occurs 214 times.
  This is repeated upstream metadata, not a SHA-256 collision.
- **Content identity:** normalized `content_sha256`, defined by
  [`C1_NORMALIZATION_SPEC.md`](C1_NORMALIZATION_SPEC.md), determines exact
  duplicate membership. It is never used as occurrence identity.

Eligible dedup units are one complete Gutenberg PT book, one pinned-snapshot
Wikipedia PT article, one source-native Corpus Carolina TEI document, and one
GigaVerbo residual row. Exact membership is based only on equal
`content_sha256` values.

ParlamentoPT is diagnostic-only. Every row is retained and resolves to itself.
Repeated ParlamentoPT hashes are summarized in
`parlamento_duplicate_hashes.parquet`; ParlamentoPT is excluded from exact
ownership, cross-source clusters, and duplicate edges.

## Ownership

For an eligible hash shared by records, retain the first record under this
ordered policy:

1. Native records in `carolina`, `gutenberg_pt`, or `wikipedia_pt` precede
   GigaVerbo aggregator records. This gives Wikipedia ownership over exact
   article copies, Carolina ownership over exact complete-record copies, and
   Gutenberg ownership over exact complete-book copies.
2. If several native records share a hash, order by source name, subset, then
   occurrence `record_id`.
3. Within GigaVerbo, use the documented C1 provenance buckets in order:
   `finepdfs_por_Latn`; curated/native web (`crawlPT_dedup`, `quati`,
   `blogset`); modern general web (`fineweb_2_pt`); legacy web (`mc4_pt`,
   `hplt2_pt`, `hplt1_pt`, `common_crawl`, `oscar`, `culturax`). Then order by
   subset name and occurrence `record_id`. Unlisted future subsets sort after the
   documented buckets by subset name and identity.
4. Any other eligible source uses source name, subset, and occurrence `record_id`.

Retained rows keep their original source and subset. A dropped row maps to
exactly one retained record in `record_resolution.parquet` and has one
corresponding row in `duplicate_edges.parquet`. No row is relabelled as a
member of another source or subset.

## Bounded-memory implementation and outputs

The index pass reads projected identity, hash, source/subset and provenance
columns only. It writes the compact index to disk-backed SQLite with a bounded
cache, then resolves hash groups in a stable external sort order. A second
streaming pass reads normalized text to calculate exact per-record word
accounting and writes only retained records to `data/`. The normalized pools
are read-only. Output is written in a private staging directory and published
by an atomic directory rename only after every artifact is complete.

The output root is
`/mnt/data/cambacica-base-180m/deduplicated/exact/`. It contains:

- `data/`: retained normalized Parquet rows, preserving the normalized schema;
- `manifest.json`: frozen contract version, normalized manifest identities,
  output checksums, counts, and diagnostic summaries;
- `duplicate_edges.parquet`: one edge for every dropped eligible record;
- `record_resolution.parquet`: every input row mapped to its representative;
- `cluster_index.parquet`: eligible exact-hash cluster inventory;
- `parlamento_duplicate_hashes.parquet`: ParlamentoPT-only repeat diagnostics;
- `accounting_by_source_subset.csv`: before/after document and word counts,
  removals, document loss fraction, and word loss fraction at source and
  source/subset scopes.

The verifier checks input manifest identities, output checksums and schemas,
input-to-resolution coverage, representative and edge integrity, deterministic
ownership, retained eligible hash uniqueness, ParlamentoPT preservation,
cluster diagnostics, and reconciled source/subset accounting.

## Deterministic pilot

Pilot mode reads local normalized pools only. It selects a fixed-seed row group
for every top-level source and every GigaVerbo subset, then uses deterministic
hash-ranked reservoirs stratified into short (`<20` words), medium, and long
(`>=100,000` words) bands. It includes complete duplicate groups found in the
selected row groups when possible and labels added rows as
`duplicate_enrichment`. Representative rate summaries use only the base
stratified rows. Pilot fractions remain diagnostic and are not production
estimates. The retained pilot `data/` is a deterministic sample input for a
later MinHash/LSH pilot; this contract sets no Jaccard threshold.

Example small local pilot and verifier:

```sh
python3 -m cambacica.corpus exact-dedup pilot \
  --normalized-root /mnt/data/cambacica-base-180m/normalized \
  --output-root /tmp/cambacica-exact-pilot \
  --pilot-size 600 --seed 20261004
python3 -m cambacica.corpus exact-dedup verify \
  --normalized-root /mnt/data/cambacica-base-180m/normalized \
  --output-root /tmp/cambacica-exact-pilot
```

The separate full-corpus production command is:

```sh
python3 -m cambacica.corpus exact-dedup run \
  --normalized-root /mnt/data/cambacica-base-180m/normalized \
  --output-root /mnt/data/cambacica-base-180m/deduplicated/exact
python3 -m cambacica.corpus exact-dedup verify \
  --normalized-root /mnt/data/cambacica-base-180m/normalized \
  --output-root /mnt/data/cambacica-base-180m/deduplicated/exact
```

This command is reserved for the later production action. The first manual
attempt failed during compact indexing because upstream identity fields were
not occurrence-unique; it published no output and is not a completed production
run. No production exact-dedup run was made while validating this v1.0.1 fix.

### Production execution note

Production exact deduplication and verification completed successfully using
implementation version 1.0.1 (`occurrence-id-v2`). Verification passed all
invariants:
- Input records: 21,649,425
- Retained records: 21,603,689
- Dropped eligible records: 45,736 (33,748 duplicate hash groups)
- Normalized words before: 21,510,183,558; after: 21,470,091,017 (40,092,541 removed)
- Exact document loss: 0.2113%; word loss: 0.1864%
- ParlamentoPT: 2,670,846 records preserved diagnostic; 0 dropped
- Manifest SHA-256: `57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428`


## Deferred C1 order

The required downstream order is:

```text
normalized
→ exact dedup
→ near dedup
→ benchmark decontamination
→ split
→ final A/B/C construction
```

Benchmark decontamination has no frozen benchmark inventory or matching rule
in this contract. Near-dedup thresholds also remain unfrozen pending a
representative pilot. Production exact dedup is a separate next action and is
not run as part of implementation or pilot validation.
