# C2 — Tokenizer experimental framework

**Implementation status: validated; C2 is in progress. No production tokenizer
has been trained, no full validation text set has been materialized, and no
candidate has been compared on the production evaluation sample.**

## Hypotheses and candidate matrix

C2 asks whether vocabulary size and the subword objective materially change
Portuguese text length, word fragmentation, and coverage at the planned model
scale. The hypotheses are that larger vocabularies reduce tokens per word and
that BPE and Unigram may trade compression against fragmentation differently.
These are questions for measurement, not expected outcomes.

The six candidates differ only in SentencePiece model type and requested
vocabulary size:

| ID | Algorithm | Requested vocabulary |
| --- | --- | ---: |
| `bpe-v16384` | BPE | 16,384 |
| `bpe-v24576` | BPE | 24,576 |
| `bpe-v32768` | BPE | 32,768 |
| `unigram-v16384` | Unigram | 16,384 |
| `unigram-v24576` | Unigram | 24,576 |
| `unigram-v32768` | Unigram | 32,768 |

All six use SentencePiece **0.2.2** and the same training data, character
coverage, special-token IDs, byte fallback, whitespace settings, sentence
length limit, and single-threaded execution. The version is pinned in the
optional `tokenizer` dependency. The exact configuration bytes have SHA-256
`2552ea368fd448dec29843d015210caf24f261b5495ab9292c373795801e1375`; the CLI
refuses a different configuration. PyArrow is pinned to **24.0.0** for reading
the corpus and split artifacts. Each artifact also records the Python and
platform runtime.

SentencePiece BPE learns pair merges. Unigram proposes candidate pieces and
prunes them against a probabilistic segmentation objective. The shared settings
make the input and common preprocessing comparable, but the algorithms do not
share an objective or identical optimization steps. SentencePiece's punctuation
segmentation is built into the library and has no corresponding TrainerSpec
switch in this pinned version; both families therefore use the same library
behavior. Unigram's seeded initialization is controlled through the pinned
SentencePiece random-generator API. The seed does not promise identical output
across different binaries, platforms, or future library versions.

## Frozen input and text treatment

Training reads only the published C1-S2 v1 sample at
`/mnt/data/cambacica-base-180m/tokenizer-sample/c1-s2-v1/sample`. The command
checks the sample-manifest SHA-256
`93c6c5400115c80aa5f5a63b4b593a5bc45bc46ecea429c78ffa5718f978e97c`, validates
the checksum manifest and all listed Parquet files, and confirms the 46,583
document and 19,999,380 normalized-word totals. It streams exact stored UTF-8
text into a deterministic temporary SentencePiece input file. It does not
normalize, clean, filter, or reweight the source text. The only added line feed
separates Parquet records in SentencePiece's line-oriented input; existing line
feeds remain sentence boundaries. The frozen C1 normalization is NFC plus
outer-whitespace trimming.

The run records the training seed **20261015**, sample-manifest identity,
configuration identity, implementation checksum, all tokenizer artifact
checksums, and runtime versions. The C1-S2 sample's own seed remains **20261014**.
Training uses all sample sentences (`input_sentence_size=0`) in sorted artifact
and row order, with input shuffling disabled and one thread. A run's intermediate
corpus and model files live under the chosen local scratch directory. Published
candidate directories are independent and immutable; interrupted publication
can resume from a matching staged artifact, while an existing candidate is
never overwritten.

The six requested vocabulary sizes **include** the three special tokens and
all 256 byte-fallback pieces. Thus 259 slots are reserved and 16,125, 24,317, or
32,509 slots remain for learned pieces. `hard_vocab_limit=true` requires the
final model to match the requested size exactly.

The common SentencePiece settings are `character_coverage=1.0`,
`byte_fallback=true`, `normalization_rule_name=identity`,
`add_dummy_prefix=false`, `remove_extra_whitespaces=false`, and
`escape_whitespaces=true`. They preserve accented Portuguese characters,
existing internal whitespace, punctuation, and digits in the stored text.
NFC text is not normalized a second time. `split_by_number=false` and
`split_digits=false` leave digit sequences available to both algorithms;
scientific notation is measured explicitly because its punctuation can still
split the sequence. Unknown UTF-8 characters fall back to their bytes, so the
expected unknown-token rate is zero for valid UTF-8. Byte fallback guarantees
coverage, but can fragment a rare character into multiple byte pieces.

The special-token contract is `<unk>` ID 0, `<s>` ID 1, `</s>` ID 2, and no
padding token (`pad_id=-1`). There are no user-defined control or chat tokens.
The tokenizer never inserts BOS or EOS during ordinary encoding. The future
causal-LM input pipeline must append exactly one `</s>` after each C1 document
and must keep document boundaries when packing. C2 measures raw-text tokens
separately from the additional per-document EOS cost.

## Fixed held-out evaluation

Evaluation uses only the published C1-S1 **validation** assignment and its
source records. The pinned split-manifest SHA-256 is
`8b937f90d40487c643fe48baa6b5c2950713fe2e0d2d534f3e2a3c08e74dbe16`; the
assignment payload SHA-256 is
`ef52e3cf3033934df13e3d16c207e1783c509dff6fb91954fb8c25ab719ece1e`. The test
split is never read for C2 evaluation.

The fixed evaluation seed is **20261016**, separate from both C1-S2 and
SentencePiece training. For each `(source, subset, domain_category)` stratum,
the selector ranks validation `doc_id` values by SHA-256 under the pinned
namespace and keeps the first 32, or every record when the stratum has fewer
than 32. This makes the sample independent of file traversal order and gives
Carolina taxonomy labels their own reported strata. The selector checkpoints
only selected references in SQLite on local scratch. It does not persist a
validation text sample. Evaluation verifies and scans the 2.48 GB assignment
metadata artifact once to make that selection, then reads only D1 Parquet row
groups containing selected documents. It verifies selected text hashes,
document coordinates, labels, and word counts against C1-S1 and D1. The index
is reused for all six candidates while that scratch directory remains.

The cap is applied within each source/subset/domain label. Aggregate metrics
over the selected sample are descriptive of this capped sample. The
source-balanced macro reports an unweighted arithmetic mean of source-level
metrics, giving each source one equal contribution; it does not combine unlike
metrics into a single score. Detailed results stay available by source and by
source/subset/domain. C1-S2's Carolina sample still has a substantial training
skew: `wik` and `leg` make up approximately **91.5% of Carolina words**. The
evaluation tables retain the Carolina taxonomies rather than hiding this
composition behind an overall average.

## Metrics and limits

Each evaluation artifact reports:

- Raw tokenizer IDs per `len(text.split())` normalized words, plus the ratio
  after adding one EOS token per document; IDs per Unicode code point are
  reported separately.
- Nearest-rank P50, P95, and P99 for the number of IDs from encoding each
  whitespace-delimited word by itself. These word-level counts are not an
  alignment of pieces emitted by whole-document encoding.
- Accented Portuguese word token counts, single-piece rate, and P50/P95;
  numeric-span and scientific-notation fragmentation; and punctuation
  single-piece rate.
- Rare-sample word-form fragmentation. “Rare” means a normalized lexical form
  seen at most twice in that reported sample grouping; it does not estimate
  full-corpus frequency. Technical vocabulary is a fixed 33-term Portuguese
  probe list in the pinned configuration. The output reports only observed
  terms and their piece/byte-fallback rates, so the list is a narrow probe, not
  a complete technical lexicon.
- Exact document round-trip rate, unknown-token and byte-fallback rates,
  document/character/token throughput, and final tokenizer artifact bytes.
- Embedding and output-head parameter counts at reference width 768, under
  tied and untied matrix assumptions.

These are descriptive tokenizer diagnostics. A small held-out sample cannot
establish broad Portuguese coverage, downstream language-model quality, or
performance on domains absent from the split. Throughput depends on the host
and storage path. The macro table prevents source volume from dominating its
summary but does not correct the provisional B training mixture.

## Vocabulary and parameter budget

At reference width 768, one embedding matrix costs `vocabulary × 768` learned
parameters. An untied output head adds the same amount. C3 may choose another
width or weight tying, so these are sensitivity figures rather than a final
architecture decision.

| Vocabulary | One embedding matrix | Untied input plus output matrices | Share of 180M if tied | Share if untied |
| ---: | ---: | ---: | ---: | ---: |
| 16,384 | 12,582,912 | 25,165,824 | 7.0% | 14.0% |
| 24,576 | 18,874,368 | 37,748,736 | 10.5% | 21.0% |
| 32,768 | 25,165,824 | 50,331,648 | 14.0% | 28.0% |

## Artifact layout and storage

Final candidate artifacts belong on the project NFS under
`/mnt/data/cambacica-base-180m/tokenizers/c2-v1/<candidate>/`. Evaluation JSON
files and comparison tables belong under
`/mnt/data/cambacica-base-180m/tokenizer-evaluations/c2-v1/`. Temporary
SentencePiece input, SQLite selection state, and Parquet processing belong on
local NVMe scratch. The CLI checks that scratch is not NFS and finalized output
is on NFS. Training output includes `tokenizer.model`, `tokenizer.vocab`,
`training_manifest.json`, and `checksums.sha256`; evaluation and comparison
artifacts also name their exact inputs and checksums.

Resource planning is not a benchmark. The training text is expected to be on
the order of 0.15–0.25 GB for 20 million Portuguese words; SentencePiece's
Unigram candidate generation can use substantially more memory than the input
file. Reserve **16 GB RAM**, **20 GB free local NVMe**, and **2 GB free NFS** for
the six artifacts. Training is CPU-only and deliberately single-threaded for
stable execution. Runtime on `cenpesor` has not been measured; allow several
hours per candidate until the first manually reviewed run yields an observed
rate. Evaluation's first candidate additionally streams and hashes the 2.48 GB
assignment file once, then reads only selected D1 row groups. The reported
compressed row-group bytes will give the actual read volume. No tmux session or
background process is started by this framework.

## Production command reference

Install the pinned optional dependencies, then choose scratch on a local NVMe
mount. Run each command separately from the repository root. The framework
does not start tmux for the operator.

```bash
python3 -m pip install -e '.[tokenizer]'
export C2_SAMPLE=/mnt/data/cambacica-base-180m/tokenizer-sample/c1-s2-v1/sample
export C2_SPLIT=/mnt/data/cambacica-base-180m/splits/c1-s1-v1
export C2_D1=/mnt/data/cambacica-base-180m/deduplicated/exact
export C2_TOKENIZERS=/mnt/data/cambacica-base-180m/tokenizers/c2-v1
export C2_EVALUATIONS=/mnt/data/cambacica-base-180m/tokenizer-evaluations/c2-v1
export C2_SCRATCH=/tmp/cambacica-c2-v1
mkdir -p "$C2_SCRATCH/train" "$C2_SCRATCH/evaluate" "$C2_SCRATCH/compare"
```

Train one candidate per command:

```bash
cambacica-tokenizer train --candidate bpe-v16384 --sample-root "$C2_SAMPLE" --output-root "$C2_TOKENIZERS" --scratch-dir "$C2_SCRATCH/train"
cambacica-tokenizer train --candidate bpe-v24576 --sample-root "$C2_SAMPLE" --output-root "$C2_TOKENIZERS" --scratch-dir "$C2_SCRATCH/train"
cambacica-tokenizer train --candidate bpe-v32768 --sample-root "$C2_SAMPLE" --output-root "$C2_TOKENIZERS" --scratch-dir "$C2_SCRATCH/train"
cambacica-tokenizer train --candidate unigram-v16384 --sample-root "$C2_SAMPLE" --output-root "$C2_TOKENIZERS" --scratch-dir "$C2_SCRATCH/train"
cambacica-tokenizer train --candidate unigram-v24576 --sample-root "$C2_SAMPLE" --output-root "$C2_TOKENIZERS" --scratch-dir "$C2_SCRATCH/train"
cambacica-tokenizer train --candidate unigram-v32768 --sample-root "$C2_SAMPLE" --output-root "$C2_TOKENIZERS" --scratch-dir "$C2_SCRATCH/train"
```

Evaluate the trained candidates separately. Keep the same `C2_SCRATCH` path so
the fixed validation-reference index is reused:

```bash
cambacica-tokenizer evaluate --candidate bpe-v16384 --tokenizer-root "$C2_TOKENIZERS" --split-root "$C2_SPLIT" --d1-root "$C2_D1" --evaluation-root "$C2_EVALUATIONS" --scratch-dir "$C2_SCRATCH/evaluate"
cambacica-tokenizer evaluate --candidate bpe-v24576 --tokenizer-root "$C2_TOKENIZERS" --split-root "$C2_SPLIT" --d1-root "$C2_D1" --evaluation-root "$C2_EVALUATIONS" --scratch-dir "$C2_SCRATCH/evaluate"
cambacica-tokenizer evaluate --candidate bpe-v32768 --tokenizer-root "$C2_TOKENIZERS" --split-root "$C2_SPLIT" --d1-root "$C2_D1" --evaluation-root "$C2_EVALUATIONS" --scratch-dir "$C2_SCRATCH/evaluate"
cambacica-tokenizer evaluate --candidate unigram-v16384 --tokenizer-root "$C2_TOKENIZERS" --split-root "$C2_SPLIT" --d1-root "$C2_D1" --evaluation-root "$C2_EVALUATIONS" --scratch-dir "$C2_SCRATCH/evaluate"
cambacica-tokenizer evaluate --candidate unigram-v24576 --tokenizer-root "$C2_TOKENIZERS" --split-root "$C2_SPLIT" --d1-root "$C2_D1" --evaluation-root "$C2_EVALUATIONS" --scratch-dir "$C2_SCRATCH/evaluate"
cambacica-tokenizer evaluate --candidate unigram-v32768 --tokenizer-root "$C2_TOKENIZERS" --split-root "$C2_SPLIT" --d1-root "$C2_D1" --evaluation-root "$C2_EVALUATIONS" --scratch-dir "$C2_SCRATCH/evaluate"
```

After all six evaluation JSON files exist, build the tables:

```bash
cambacica-tokenizer compare --evaluation-root "$C2_EVALUATIONS" --output-dir "$C2_EVALUATIONS/comparison-v1" --scratch-dir "$C2_SCRATCH/compare"
```

The compare command requires all six pinned candidate artifacts and writes
`summary.csv`, `by_source.csv`, `by_source_subset_domain.csv`, a comparison
manifest, and checksums. It reports no winner.

## Validation and open decisions

Validation is limited to configuration/CLI checks, unit tests, tiny synthetic
BPE and Unigram training, a small bounded read of published real artifacts,
and Ruff. The production sample has not been used for tokenizer training, the
full validation text sample has not been created, and no six-candidate result
table has been produced.

C1 remains **IN PROGRESS** because rights and policy matters are unresolved.
The C1-S2 B mixture remains a preparation reference, not the final training
mixture. Before interpreting results, the scientist should review the fixed
technical probe list, the equal-source macro interpretation, the EOD contract
against C3's packing design, and the 768-wide tied/untied budget cases. C2
reports measurements without choosing a tokenizer.
