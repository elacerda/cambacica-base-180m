# C2 tokenizer selection — version 1

**Decision recorded 2026-10-14: SCIENTIST APPROVED.** SentencePiece Unigram,
candidate `unigram-v24576`, is the definitive tokenizer for Cambacica-180M's
first training run. The selected production artifact and the comparison inputs
below are **VERIFIED** by their published checksum manifests and matching
identities. This is an immutable decision record; a later scientific change
must create a new version.

## Frozen tokenizer contract

| Field | Frozen value |
| --- | --- |
| Candidate | `unigram-v24576` |
| Algorithm | SentencePiece Unigram |
| Vocabulary size | 24,576 pieces, including special and byte pieces |
| Special tokens | `<unk>` = 0, `<s>` = 1, `</s>` = 2 |
| Padding | No PAD token (`pad_id = -1`) |
| Byte fallback | Enabled; 256 reserved byte pieces |
| SentencePiece | 0.2.2 |
| Production directory | `/mnt/data/cambacica-base-180m/tokenizers/c2-v1/unigram-v24576/` |

The artifact manifest records no PAD token, 256 byte-fallback pieces, and the
same three special-token IDs. The published `.vocab` has 24,576 entries. The
future input pipeline must preserve document boundaries and append one EOS per
document as specified in the C2 experiment contract; the tokenizer does not
automatically insert BOS or EOS during ordinary encoding.

## Production artifact identity

All entries in the artifact's existing `checksums.sha256` passed `sha256sum
-c`. The following hashes were also calculated independently from the
published files:

| File | SHA-256 |
| --- | --- |
| `tokenizer.model` | `c8982a23d3c5ecc0a61d300edc1e354e38f1358aa3db5420d5bf19f7b95cbf23` |
| `tokenizer.vocab` | `f27e38de1a1e0e2e96ac00f50382d1be749d2b357cb3e8ad9ee43bf8ae972d9f` |
| `training_manifest.json` | `4ed655eea79857a07252e4119084a69cb1d440d9b1b2b9c0796be5274318dd53` |
| `checksums.sha256` | `a1a76cc37211fcf019d0cf02d27d1f97233da2f9dc07bcc896fa92ab52295bef` |

The training manifest binds the artifact to candidate `unigram-v24576`, model
type `unigram`, vocabulary size 24,576, training seed 20261015, SentencePiece
0.2.2, and the pinned configuration and input identities below. Its recorded
implementation file digest is
`a5cf942a74cc3cff2dfb23eb17bedc236b4eceb83d22f4c9fe2727848564372e`, matching
the C2 implementation at the initial C2 commit.

## Evaluation and comparison identities

All six published evaluation JSON files name the same fixed evaluation inputs:

| Input | Identity |
| --- | --- |
| C1-S2 training sample manifest SHA-256 | `93c6c5400115c80aa5f5a63b4b593a5bc45bc46ecea429c78ffa5718f978e97c` |
| C2 configuration SHA-256 | `2552ea368fd448dec29843d015210caf24f261b5495ab9292c373795801e1375` |
| C1-S1 split manifest SHA-256 | `8b937f90d40487c643fe48baa6b5c2950713fe2e0d2d534f3e2a3c08e74dbe16` |
| C1-S1 assignment Parquet SHA-256 | `ef52e3cf3033934df13e3d16c207e1783c509dff6fb91954fb8c25ab719ece1e` |
| D1 input manifest SHA-256 | `57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428` |
| Fixed C1-S1 validation sample SHA-256 | `a948a8007a0c30e0ecc84c3db15374e4dde7a608197c1f5ccc18aadbc5ceeac3` |
| Evaluation seed | `20261016` |

Each evaluation selected 616 documents across 21 strata, with at most 32
documents per stratum. The six evaluation file hashes are recorded in the
published comparison manifest and were independently checked against those
files:

| Candidate | Evaluation JSON SHA-256 |
| --- | --- |
| `bpe-v16384` | `48ffd7bd9e89b0b3195f95c047554cc5d7ef1b01c8c8c9d1445942024b397ed6` |
| `bpe-v24576` | `34dd03dd6dfa148e41288d672afa8a5966e72456223f690affac50325005bc67` |
| `bpe-v32768` | `58d38dfa64712fcea73a44faf4f728493c06497b0bfa416b36a09553324f4c8b` |
| `unigram-v16384` | `a73d02bf5cdbc82dc607f399097f4a8ebec5290693c674645b22cdfcc68d4302` |
| `unigram-v24576` | `68f6900332fbb2889ceaee9cdeab8b7901953b99d11fc2f49168c9f972b1604b` |
| `unigram-v32768` | `32d3b34328dcfadf7aca4fb4f9daa0e81217bf931daee3381c6c3145dd9e2e5b44` |

Comparison directory:
`/mnt/data/cambacica-base-180m/tokenizer-evaluations/c2-v1/comparison-v1/`.
Its checksum manifest passed `sha256sum -c`; all three table hashes and the
comparison-manifest hash are:

| Comparison artifact | SHA-256 |
| --- | --- |
| `comparison_manifest.json` | `8c74ce8a3ae960f357c41f7b30598b41a337923406d728b5c761901b176fa330` |
| `summary.csv` | `9a0ba01e44886013aa3c4833797618bd9bb24dfc6fb0065aaaf816713772a467` |
| `by_source.csv` | `878f9523c4ed36369bf05518b9162d29b8939775e8f2af6804144d237dcab675` |
| `by_source_subset_domain.csv` | `d04267e828005cdefc9ef08402114a377b9c57de56e79c1c74568a304996bead` |
| `checksums.sha256` | `3457fa92603749161eb68a3e26a4a2039afbafba6fb5e8cd13ca4eba9d2c65e2` |

The comparison manifest names exactly the six candidates above, pins the C2
configuration digest, and reports `winner_declared: false`. Scientist approval
is recorded separately here; the mechanical comparison did not choose a
winner.

## Six-candidate results

The equal-stratum column is the unweighted descriptive mean across the 21
source/subset/domain rows. All token-per-word values use the same fixed sample.
The embedding column is one input matrix at the C2 reference width of 768.

| Candidate | Global tokens/word | Equal-source macro | Equal-stratum mean | Accented words, one piece | Rare-word mean pieces | Technical-term mean pieces | Embedding parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bpe-v16384` | 1.831 | 1.748 | 1.705 | 21.98% | 3.061 | 2.175 | 12,582,912 |
| `bpe-v24576` | 1.723 | 1.647 | 1.603 | 28.83% | 2.859 | 1.855 | 18,874,368 |
| `bpe-v32768` | 1.668 | 1.593 | 1.548 | 35.68% | 2.727 | 1.621 | 25,165,824 |
| `unigram-v16384` | 1.833 | 1.756 | 1.716 | 29.92% | 3.026 | 1.784 | 12,582,912 |
| **`unigram-v24576`** | **1.732** | **1.658** | **1.615** | **40.59%** | **2.810** | **1.569** | **18,874,368** |
| `unigram-v32768` | 1.677 | 1.605 | 1.559 | 48.17% | 2.680 | 1.383 | 25,165,824 |

All candidates round-tripped exactly 616/616 validation documents and recorded
an unknown-token rate of 0%. Their embedding and output matrices at width 768
are summarized by the experiment report; an untied output matrix costs one
additional matrix of the same size.

The 24k BPE candidate produces slightly fewer tokens per word than the selected
Unigram 24k candidate, but its accented-word single-piece rate is lower
(28.83% versus 40.59%) and its technical-term mean fragmentation is higher
(1.855 versus 1.569 pieces). Unigram 32k reduces fragmentation further, but
adds 6,291,456 parameters to each 768-wide embedding matrix. Unigram 24k is the
approved compromise within the nominal ~180M parameter budget.

These tokenizer measurements do not establish downstream language-model
superiority. The equal-source macro gives each of five sources equal weight;
the equal-stratum mean describes these 21 capped strata. Neither is a final
training-mixture estimate. Tokenization throughput is host-specific and does
not predict model generation speed.

## Byte-fallback finding and C4 consideration

All six candidates produced exactly **965,238 byte-fallback tokens** on the
same validation sample. Targeted probes showed LF, CR, and TAB represented by
byte-fallback pieces, while ordinary accented Portuguese text encoded
correctly. In selected real documents, LF counts exactly matched fallback
counts. This behavior is strongly consistent with fallback preserving
structural characters, especially newlines. The experiment did not
independently classify every fallback token in the full validation set.

C4 must account for structural-character tokens when preserving newlines,
packing documents, and estimating context occupancy. Keep the current
normalization, input corpus, newline preservation, and one-EOS-per-document
policy unchanged. This finding is an accounting consideration, not approval to
alter text treatment.

## Provenance and remaining scope

The C2 framework began at commit `d480a710` (full commit
`d480a7109cea4749a7d2029e3f6f0b62b1a0af43`). The validation-index reuse and
checksum-metadata fix is commit `3b0349a` (full commit
`3b0349a229dc4eb31eafd30406e67dda1cbe3f2e`). The selected model's
implementation digest matches the initial training implementation. The
evaluation and comparison artifacts are linked by their exact hashes above.

C1 remains **IN PROGRESS** while source-rights and usage-policy issues remain
unresolved. Mixtures A/B/C remain candidates; mixture B was the provisional
tokenizer-sampling reference and is not the approved final training mixture.
The selected tokenizer is frozen for the first run. C3 architecture planning is
recorded in [`C3_ARCHITECTURE_PLANNING.md`](C3_ARCHITECTURE_PLANNING.md); no
architecture, context length, optimizer, training-token budget, or final
mixture is frozen by this decision.
