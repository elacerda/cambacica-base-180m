# Gate C1 Exact Deduplication Contract

**Version:** 1.0.0
**Status:** frozen
**Gate:** C1 — IN PROGRESS

This contract removes complete normalized records only when their verified
`content_sha256` values are identical. It does not match passages, substrings,
or semantic similarity. Near deduplication is a later, separately calibrated
stage.

## Identity and record units

The identity key is the existing normalized `content_sha256`, defined by
[`C1_NORMALIZATION_SPEC.md`](C1_NORMALIZATION_SPEC.md). Eligible dedup units
are one complete Gutenberg PT book, one pinned-snapshot Wikipedia PT article,
one source-native Corpus Carolina TEI document, and one GigaVerbo residual row.
Each normalized input row receives a stable `record_id` from its source,
revision, subset, original ID, raw source file and raw record identifier, plus
GigaVerbo provenance where available. The SHA-256 of that canonical identity
is independent of Parquet traversal order.

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
   stable `record_id`.
3. Within GigaVerbo, use the documented C1 provenance buckets in order:
   `finepdfs_por_Latn`; curated/native web (`crawlPT_dedup`, `quati`,
   `blogset`); modern general web (`fineweb_2_pt`); legacy web (`mc4_pt`,
   `hplt2_pt`, `hplt1_pt`, `common_crawl`, `oscar`, `culturax`). Then order by
   subset name and stable `record_id`. Unlisted future subsets sort after the
   documented buckets by subset name and identity.
4. Any other eligible source uses source name, subset, and stable `record_id`.

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

This command is documented for the later production action; it was not run
while implementing or validating this contract.

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
