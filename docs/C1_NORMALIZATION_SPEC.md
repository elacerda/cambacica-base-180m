# Gate C1 Raw-Source Normalization Specification

**Version:** 1.0.0

**Status:** frozen; production run pending

**Gate:** C1 — IN PROGRESS

This stage makes source records inspectable and comparable in `normalized_words`.
It does not deduplicate, decontaminate, filter, weight, sample, split, truncate,
chunk, or tokenize the source pools.

## Canonical text and word accounting

The existing accepted C1 sample schema already applies Unicode NFC and trims
leading and trailing Unicode whitespace. The source samplers calculate words as
`len(text.split())`; `src/cambacica/corpus/metrics.py` repeats that definition.
The composition tables in `docs/C1_COMPOSITION_ANALYSIS.md` use those sample
metrics. Version 1 keeps that word-count definition and NFC/outer-whitespace
convention so `normalized_words` remains comparable to prior C1 measurements.

For normalized text, plain-text payloads use strict UTF-8 decoding, removing a
UTF-8 BOM only at the start of the payload; TEI uses its XML-declared encoding
and XML entity/newline rules; Parquet string fields use Arrow's UTF-8 decoding.
After source extraction, map CRLF and bare CR to LF; normalize Unicode to NFC;
trim leading and trailing Unicode whitespace; encode as UTF-8 for byte counts
and hashes. No other characters or internal whitespace are rewritten. Non-
whitespace controls are preserved where the source format permits them; control
code points that Python treats as whitespace are subject to the specified
outer trim or word split. Case, spelling, diacritics, punctuation, and document
boundaries are preserved.

`normalized_words` for one document is exactly Python's
`len(normalized_text.split())`: each maximal run separated by Python Unicode
whitespace semantics is one word; punctuation remains attached and
punctuation-only runs count. Corpus totals sum those document counts. This is a
whitespace word proxy, not a linguistic tokenizer or a model-token count.
Character counts are Python Unicode code-point counts after NFC. Byte counts
are UTF-8 payload bytes of text only, excluding Parquet overhead and record
separators. Percentiles use the existing C1 nearest-rank rule
`sorted_values[ceil(p*n)-1]`.

The pre-normalization C1 samples used the same word-count rule but not every
production source adapter followed the same extraction path. In particular,
Carolina's old sampler only read direct paragraph `.text`, which can omit inline
TEI content. Production totals therefore supersede sample extrapolations; the
word-count unit itself remains compatible.

## Source record contract

- **Frozen raw identities:** Gutenberg `snapshot_2026-10-01` with 655 files;
  ParlamentoPT commit `08f13e7e63ab9bfbd8c0b40955defe3bb7f68c2b`; Wikipedia PT
  snapshot `20231101.pt` at commit `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`;
  Carolina v2.0.1 at commit `55e63a519393c70a48dcfa14a558499c6bb0583b`; and
  GigaVerbo-v2 at commit `7058ccf19eaeaf4505a96fc7e5305a01fc441fd8` with
  15,756,679 materialized residual records. Production normalization rejects
  raw manifests that do not match these identities.
- **Gutenberg PT:** one frozen raw `pg<ID>.txt` file is one book document. Remove
  only text before the first standard `START OF THIS/THE PROJECT GUTENBERG EBOOK`
  marker and after the first following standard `END OF THIS/THE PROJECT
  GUTENBERG EBOOK` marker. If markers are absent or reversed, retain the text
  (apart from the global normalization above). Everything between markers,
  including prefaces, notes, chapter headings, dictionaries, and long works,
  remains. No minimum length is applied.
- **Corpus Carolina:** each source-native TEI `<TEI>` element is one document;
  each `.xml.gz` file is parsed independently. Text comes from body paragraph
  and other human-readable block elements in document order, using descendant
  text so inline TEI markup does not discard words. Structural whitespace at
  block boundaries is trimmed, and extracted blocks are joined by two LF
  characters. Taxonomy (`dat`, `wik`, `jud`, `uni`, `soc`, `leg`,
  `pub`), TEI identity/title/date/license/reference metadata, selected author,
  editor, publisher, and identifier fields, and source-file/record provenance
  are retained. A TEI element without an upstream ID receives a stable
  file-and-ordinal fallback ID. No document-length cap applies.
- **Wikipedia PT:** one row in the locally materialized, pinned `20231101.pt`
  Parquet snapshot is one article. Use only its `id`, `url`, `title`, and `text`
  fields. Preserve the existing C1 title rule: prepend `title + "\n\n"` only
  when non-empty title is not already a prefix of text. Do not contact current
  Wikimedia.
- **ParlamentoPT:** one physical line in the pinned `train.txt` is one raw
  text record, with its 1-based line number as record identity; the line is not
  asserted to be a speaker turn or an entire parliamentary sitting. This is the
  newline-delimited `text` record boundary exposed by the pinned dataset's
  `train` split and used by the existing sampler ([pinned dataset family
  card](https://huggingface.co/datasets/PORTULAN/parlamento-pt)). Preserve
  repeated records and all non-empty or empty line records; remove only the
  record delimiter. Do not join adjacent lines or infer speech boundaries
  absent from the source. The verified raw manifest records 2,670,846 LF
  delimiters; the earlier `~11.5M debate interventions` sampler description
  was unsupported and is withdrawn.
- **GigaVerbo-v2 residual:** every row persisted by the completed raw
  materialization is normalized once, retaining subset, id, source URL,
  quality/toxicity/token-count metadata, and `_gv2_upstream_shard`,
  `_gv2_upstream_row_group`, `_gv2_upstream_commit`. No exclusion rules,
  resampling, or provisional mixture weights are rerun.

Malformed UTF-8, XML/GZip, Parquet, or source records are recorded with their
raw location and error. Successfully parsed empty documents remain output rows.
Failures make a finished source `PARTIAL`; they do not silently remove
outliers or authorize a `COMPLETE` status.

## Unified normalized schema

The existing 14 C1 document concepts remain in their current order:

`text`, `source`, `source_revision`, `subset`, `original_id`, `original_url`,
`license`, `language`, `language_score`, `variety`, `quality_score`,
`publication_date`, `domain_category`, `content_sha256`.

The normalized Parquet schema adds only `title`, `raw_source_file`,
`raw_record_identifier`, `normalization_version`, the three per-row GigaVerbo
provenance fields, and `upstream_metadata_json` for useful upstream fields not
represented directly. `content_sha256` is SHA-256 of normalized UTF-8 text
exactly as stored; raw file provenance is separate. Optional scores such as
`language_score` remain null when a pinned source does not supply them; a
source-level language assignment is not reported as a measured confidence.

## Storage, restart, and manifests

Default root is `/mnt/data/cambacica-base-180m/normalized/`. Source directories
are `gutenberg/`, `parlamento/`, `wikipedia/`, `carolina/`, and `gigaverbo/`;
GigaVerbo files are Hive-partitioned as `subset=<source subset>/`.

Parquet uses Zstandard level 6, Parquet 2.6, fixed schema, stable input-file
and row order, and a 256 MiB normalized UTF-8 text-byte target per output shard.
Shards end only between documents. A single document larger than the target
gets its own larger shard. This target was checked against persisted C1 sample
record sizes: observed maxima were 17,467,300 bytes (Carolina diagnostic),
13,619,916 (Gutenberg), 1,722,439 (GigaVerbo), 146,007 (Wikipedia), and 9,361
(ParlamentoPT). The 256 MiB target is over 15 times the largest observed sample
record, while the raw GigaVerbo payload averages about 18.4 MB across 2,260
files; the target should combine small inputs into practical scan shards.

Each source has an atomically written `manifest.json`, or
`manifest.in_progress.json` during an interrupted run. Output shards are
written to same-directory `.partial` files and atomically renamed before the
checkpoint is advanced. Resume is permitted only when source, normalization
version, raw-manifest SHA-256, tool commit, Python/Unicode/PyArrow runtime, and
shard target still match.
Uncheckpointed partial shards are discarded and replayed from the last
checkpointed source record. A completed manifest inventories every normalized
file with SHA-256, byte/document/text totals, cursor range, raw-manifest SHA,
tool commit, failures, timestamps, and Python/Unicode/PyArrow runtime versions.
`COMPLETE` requires zero failures;
`PARTIAL` means records or files failed; `FAILED` means the run could not safely
start or validate. Verification checks the raw-manifest identity, every output
file hash/schema/row count, every row's normalized content hash and version,
per-file byte/character/word totals, and aggregate manifest invariants.

## Characterization and mix feasibility

Characterization is descriptive and retains outliers. It reports documents,
normalized bytes/characters/words; nearest-rank word-length mean, median, p90,
p95, p99, and max; empty, under-20-word, under-100-word, at-least-100,000-word,
and at-least-1,000,000-word counts; normalization failures; critical metadata
completeness; and field coverage. Reports are produced per source, per source
subset when present, and per GigaVerbo subset. PARTIAL sources retain
observational metrics but mix feasibility is marked not assessable; reported
capacity is only a lower bound from successfully normalized documents.

The canonical word-accounting CSV uses rows at source/subset granularity and
contains documents, normalized words, and fraction of total words. Candidate
mixes have no fixed total normalized-word target; the report therefore gives
each mix's maximum non-oversampled total, constrained by every source share and
each configured GigaVerbo subset share. If a target is later supplied, the
report states whether that target would require repeating a source pool. These
are gross pre-deduplication capacities; post-deduplication volume is not
estimated here. No candidate corpus is constructed here.
Capacity floors use exact rational forms of the configured decimal shares.

The downstream C1 order is `normalized → exact dedup → first-run near-dedup
preserve-all decision → benchmark decontamination → split → final A/B/C
construction`. D1 is complete. D2–D2d are complete, and near-dedup removals
are approved at zero for the first run; see
[`C1_NEAR_DEDUP_FINAL_DECISION.md`](C1_NEAR_DEDUP_FINAL_DECISION.md).
Benchmark decontamination is next; its proposed inventory and matching plan
are in
[`C1_BENCHMARK_DECONTAMINATION_PLAN.md`](C1_BENCHMARK_DECONTAMINATION_PLAN.md).
Split policy and final mix selection remain later C1 decisions.

## Production commands

After the implementation is committed and the worktree is clean, run each
source command separately. Each command verifies the pinned raw manifest and
payload before normalization and resumes an interrupted run by default.

```sh
python3 -m cambacica.corpus normalize gutenberg_pt
python3 -m cambacica.corpus normalize parlamento_pt
python3 -m cambacica.corpus normalize wikipedia_pt
python3 -m cambacica.corpus normalize carolina
python3 -m cambacica.corpus normalize gigaverbo_v2
python3 -m cambacica.corpus characterize
```

The characterization command requires all five source manifests to exist and
be verifiable. It writes source JSON reports, the canonical normalized-word
accounting CSVs, and the mix-capacity summary under
`/mnt/data/cambacica-base-180m/normalized/characterization/`.
