# C1-BD2 Benchmark Snapshot and Matcher Calibration

**Status:** snapshot **COMPLETE**; calibration **COMPLETE**; C1-BD3 production
scan **NOT RUN**; C1-BD4 review/exclusions **NOT RUN**
**BD1:** scientist-approved scope, recorded in
[`C1_BENCHMARK_DECONTAMINATION_APPROVAL.md`](C1_BENCHMARK_DECONTAMINATION_APPROVAL.md)
**Corpus input identity:** `/mnt/data/cambacica-base-180m/deduplicated/exact`,
manifest SHA-256
`57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428`

BD2 makes the approved benchmark direction reviewable and repeatable. It does
not create training splits, exclusions, a modified corpus, or an evaluation
harness. No source corpus Parquet rows were read for this stage; the BD3 CLI
dry-run checks pinned manifest identity, file inventory, Parquet footers, and
row counts only.

## 1. Dataset snapshots

### CALAME-PT

- Repository: `NOVA-vision-language/calame-pt`
- Verified immutable revision:
  `353671bc95cc3d94d488201f67d41b640eb80c55`
- Actual aggregate: **2,076** rows; **406 handwritten** and **1,670 generated**.
- Row IDs are unique source values `0` through `2075`.
- Source fields: `id`, `sentence`, `last_word`. `sentence` is the context;
  `last_word` is preserved literally as the target. Public labels are present.
- Physical pinned Hub metadata: config `default`; the aggregate file is named
  split `train`. The pinned README also maps the handwritten file to `test` and
  names a generated validation file with a trailing underscore. That path,
  `calamept_gen_only_.jsonl`, does not exist in the pinned tree; the actual file
  is `calamept_gen_only.jsonl`.
- Snapshot logical identity: config `all`, split
  `all_evaluation_only`. **Every approved CALAME row is evaluation-only.** The
  source `train` label is not permission to use any row for Cambacica
  pretraining.
- Row `718` is generated and has a 462-character context with a six-space
  `last_word`. The row remains in the snapshot and category count. Its target
  has no match tokens, and its synthetic `complete_item` field is non-matchable
  so that the context alone is not mislabeled as a complete question/answer
  item.
- The pinned card declares MIT. The CALAME introduction reports 406 handwritten
  items and 1,670 generated items; generated contexts were grounded in
  Portuguese Wikipedia, OSCAR, and Arquivo.pt material and then reviewed. The
  snapshot has no per-row upstream source IDs. The MIT label does not settle
  rights to embedded third-party source text.

The official paper describes 2,076 examples. A public Manacá evaluation report
uses `n = 2,075`; its current public evaluator strips target whitespace and
skips empty targets. That behavior would skip row 718, so it is a plausible
explanation for the variant, not an official split definition or a resolved
denominator decision. The future evaluator must pin its own code and record the
decision explicitly.

### Belebele Portuguese

- Repository: `facebook/belebele`
- Verified immutable revision:
  `d4c91dedc9de484dbea7b7d940f898f59fd135e9`
- Logical configuration/split: `por_Latn/test`; physical pinned Hub metadata:
  config `default`, split `por_Latn`.
- Downloaded only `data/por_Latn.jsonl` and the README; no multilingual shard
  data was downloaded.
- Actual count: **900** questions over **488** unique passage identifiers.
  Question-number counts are 482 `1` and 418 `2`.
- Source fields: `link`, `question_number`, `flores_passage`, `question`,
  `mc_answer1` through `mc_answer4`, `correct_answer_num`, `dialect`, and `ds`.
  Stable source row IDs are `link#q<question_number>`. All 900 rows have
  `dialect=por_Latn` and `ds=2023-06-07`.
- `correct_answer_num` is a public, one-indexed string from `1` to `4`; the
  snapshot preserves it and maps it to the corresponding answer text. Choices
  and answer are kept as separate field roles.
- The card declares CC-BY-SA-4.0. The README describes questions over
  FLORES-200 passages. The license label does not independently settle rights
  for every upstream passage or linked source. Attribution and this caveat are
  recorded per row and in the manifest.

The proposed Cambacica evaluation is a Portuguese, zero-shot multiple-choice
adaptation for a causal base LM. It is not automatically comparable with every
published Belebele score, which may use different languages, prompting,
fine-tuning, model classes, or answer-generation conventions.

## 2. Snapshot identity and contract

The snapshot is stored at
`/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2/snapshot/`.
It contains pinned source files, `benchmark_manifest.json`,
`benchmark_registry.parquet`, `match_fields.parquet`, and
`source_checksums.json`. Registry row count is **2,976**; field row count is
**14,328**. Source text is retained unchanged in raw JSON and original-field
columns. Normalized match text and Unicode token/offset representations are
separate.

Stable benchmark IDs hash benchmark name, revision, logical config/split, row
ID, and canonical row checksum. Belebele's row ID is the source passage link
plus question number; CALAME uses its source `id`. Field IDs add the role to the
example ID. Each registry row records repository/revision, logical and physical
config/split, source file SHA-256, row SHA-256, category, field mapping, public
answer availability, declared license, attribution, provenance caveat, and
raw row JSON. The matcher view records original text, field role, matchability,
normalized text, token sequence, normalized checksum, and token offsets into
the unchanged original string.

Pinned source-file SHA-256 values:

| Source file | SHA-256 |
| --- | --- |
| CALAME README | `aed38d8b5e926189c44a8cfd09852a64290363009013bec6a10a7ebe6e3f27c1` |
| CALAME aggregate | `11fb23d6474068186bdc6000334fe48818ee9c26dd3811d3bc6ffdd919480d1c` |
| CALAME generated category | `9bc4346a84215f89fb490b583a36da60bc35b331661ff79ae18dda11c896a04b` |
| CALAME handwritten category | `bd10b5041caa7ae5887ff9b74a0e0ffebf773a9c9c3b1a3570bdae40e1a4eeb2` |
| Belebele README | `62d86838e0959eacbbe7ebe5ce979480ecdd8ee3037237eb18bcd669b83a6309` |
| Belebele Portuguese JSONL | `da63cd6215d550f0ad3df5562567096e8e7731a04322f90e801c554b1c43b431` |

The verified snapshot manifest SHA-256 is
`d9b5bfdec543af37fdb8733239b0f8b0255339454b5ebca8880371bb86f77cda`.
Generated artifact checksums:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `benchmark_registry.parquet` | 1,343,715 | `4b24024be2e9efd56bd41277f6cd436abe6544ec017be0b92e19c769c9b777a3` |
| `match_fields.parquet` | 4,243,090 | `c44df75eb4e333290c11284ebb8ba68518edf130f2a207504636cbaf4acb49a6` |
| `source_checksums.json` | 641 | `8347aaa4b671eca3fc34ffe64c5063072bff2c68e2d4e8250852302c8ee320c2` |

## 3. Matcher and normalization

The matcher is independent of D2c's whole-document MinHash/Jaccard method. Its
versions are `c1-bd2-token-anchor-v2` and
`nfc-casefold-unicode-word-offsets-v1`.

Normalization applies Unicode NFC, casefolding, and Unicode word-token
extraction over letter/number runs (with combining marks attached to a
preceding token). It preserves diacritics and token order. It does not stem,
remove stopwords, strip accents, translate, or reorder. Token offsets point to
Python character positions in the exact original string.

Evidence is field-specific:

- Exact normalized token sequences can identify a complete item, context,
  passage, or question. Punctuation and case changes do not change the token
  sequence.
- Exact questions are reported only for questions of at least six tokens that
  occur in at most two benchmark items and at most 100 distinct corpus records.
- A question and correct answer may be associated when the exact question is
  present and the actual answer text occurs within 128 tokens. Options and
  answer fields are indexed as data but cannot independently trigger a hit.
  CALAME target words are likewise non-triggering by themselves.
- Approximate matching uses 13-token anchors. Anchors repeated in more than two
  distinct benchmark examples are not distinctive. During a scan, a corpus
  anchor is retained only while its final distinct-record document frequency
  is at most 100. If it crosses the cap, evidence from all earlier and current
  documents is removed consistently. Exact full-item evidence bypasses this
  common-anchor filter.
- Approximate candidates meet either a contiguous matching run of at least 50
  tokens or at least 80% distinctive-token coverage with at least two distinct
  rare anchors. These are versioned candidate values, not approved BD3
  thresholds.

For approximate hits, the hit row carries a representative excerpt plus
aggregate coverage, anchor count, and frequency range. A companion
`candidate_anchor_evidence.parquet` is prepared for BD3; it streams every
retained rare-anchor occurrence with corpus and benchmark token/character
offsets, normalized anchor text, and its final corpus document frequency. The
writer batches 4,096 evidence rows. This keeps detailed evidence on disk rather
than collecting all matched spans in memory.

The matcher uses fixed token windows with overlap large enough for the longest
benchmark field and configured question/answer gap. It retains one window per
document, spills document IDs, anchor frequencies/evidence, and final hits to
SQLite, commits every 256 documents, and orders results by stable identities.
The default token window is 8,192 tokens (increased when a longer benchmark
field or question/answer gap requires it); the SQLite page cache is 16 MiB. It
does not cap or silently truncate candidate evidence. Exact complete-item
evidence is retained separately from approximate common-text filtering.

## 4. Calibration controls and results

The calibration uses separate development and held-out controls. Stable
benchmark IDs are disjoint; selected real contexts/passages are also checked
for exact shared 13-token shingles and high 5-token-shingle Jaccard across the
partition. There are **26 documents per policy** (19 development and 7
held-out), evaluated under three predeclared policies. Five positives and one
answer-only negative are derived from selected pinned benchmark rows; the other
controls are synthetic. No source-corpus record was selected or read.

Controls cover exact questions and complete items, embedded full and partial
passages, question-plus-answer, case/punctuation variation, small
insertions/deletions, a short distinctive question, answer-only text, generic
multiple-choice instructions, common Portuguese phrases, topical but different
content, similar questions with changed facts, repeated structural templates,
and D2d-style taxonomy articles with different entities but shared
near-identical references.

| Policy | Development positives | Development negatives | Held-out positives | Held-out negatives | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 50-token-only baseline | 9/10 | 9/9 | 4/4 | 3/3 | Missed `dev_small_insertions_and_deletions`; no selected false positives. |
| Candidate: 50 OR 80% / 2 anchors | 10/10 | 9/9 | 4/4 | 3/3 | 14/14 selected positives; 12/12 selected negatives. |
| Strict: 50 OR 90% / 3 anchors | 10/10 | 9/9 | 4/4 | 3/3 | Same selected-case outcomes as candidate. |

The candidate recovers one development control that the baseline misses. The
candidate and stricter policy do not separate on this small panel; the panel
does not establish the best production threshold. In candidate evidence, the
largest observed contiguous run was 136 tokens, the largest matched anchor
count was 97, coverage reached 1.0, and matched anchor corpus DF ranged from 1
to 3. These observations are fixture-specific. They do not estimate
population precision, recall, contamination prevalence, or the behavior of a
corpus-wide DF cutoff of 100.

The candidate configuration is `c1-bd2-candidate-50-or-80-2-v1`, with
`freeze_for_bd3=false` and
`CANDIDATE_REQUIRES_SCIENTIST_APPROVAL`. The threshold candidates were declared
before fixtures ran; fixtures compare policies but do not select or freeze one.
The smallest useful next calibration is a **40-case blinded panel**: for each
benchmark, 10 independently sourced positives and 10 hard negatives drawn from
at least two post-D1 source families. Positives should include exact, embedded,
and near-copy cases; negatives should include same-topic, boilerplate, and
shared-reference cases. Scientists must approve the bounded record selection
and labels before the panel is assembled. This panel can surface failure modes
and inform threshold review; it still cannot establish population precision or
contamination prevalence.

Known misses include semantic paraphrases, translations with little shared
lexical material, OCR/image-only content, and answer inference when the answer
text is absent. Exact question filtering can miss common exact questions; the
minimum length and corpus DF cutoffs remain provisional. Approximate anchors
can still match shared public source passages or boilerplate, so every hit
requires context/provenance review. The 12 synthetic/selected negative controls
are not a population-level false-positive estimate.

## 5. Memory, I/O, failure and restart behavior

Measured on the BD2 fixture run in this environment:

- Calibration CLI: 26 fixture documents per policy across three policies;
  internal elapsed time **15.79 s**.
- Process high-water RSS: **735,416,320 bytes** (about 701 MiB), including
  Python, PyArrow, benchmark index, and matcher; this is not an incremental
  matcher-only measurement.
- Peak SQLite file during the fixture policies: **335,872 bytes**; no GPU.
- A 30,000-token, no-hit document used **276,313 bytes** of traced Python
  allocation during the matcher scan while the token window was configured for
  512 tokens; the input string was created before tracing, and the test asserts
  less than 8 MiB.
- All approximate anchor spans are iterated from SQLite and written in
  4,096-row Parquet batches; BD2 fixture reports include every anchor span.
- 10,000 synthetic no-hit records with 64-character IDs used **839,680 bytes**
  of SQLite space. Linear extrapolation of that no-hit baseline to 21,603,689
  records is about **1.81 GB** for the ID/accounting table alone. This is an
  estimate from synthetic records; match evidence and final hits add an
  unknown amount. BD3 scratch capacity must be provisioned above this baseline.

BD3 planning scenarios, not measurements: scanning the pinned **21.47 billion
normalized words** at an assumed 25,000–100,000 words/second in one matcher
process would take roughly **60–239 hours** (about 2.5–10 days), before any
Orion/NFS slowdown or unusually dense evidence. No Orion throughput test was
run. The fixture process high-water RSS is about 695 MiB; reserve at least 2 GiB
of process memory for an initial production attempt, with more headroom for
large rows and Arrow buffering. The 1.81 GB no-hit SQLite extrapolation is only
a floor for scratch planning; candidate evidence size is unknown and can add
substantial disk use. Measure a source-approved representative slice and check
local free space before BD3. A failed full pass restarts from the beginning;
BD2 does not checkpoint across records because corpus-wide anchor frequencies
are finalized only after the one-pass input stream completes.

The bounded-memory claim is for the corpus index and token window, not for a
single input string: one document is held at a time, plus a fixed Arrow batch
of 16 rows. Matching state spills to configurable local scratch; final durable
artifacts are staged next to their output and atomically renamed. A failed or
interrupted run publishes no complete manifest and removes the incomplete
stage; its SQLite file is removed by the run context. Runs restart from the
beginning; BD2 does not implement checkpoints or a resumable full scan. The
future BD3 operation must use local scratch with adequate capacity and durable
NFS-backed output, and must check free space before it starts.

## 6. Output contracts and CLI

Bulk artifacts are outside Git at:

```text
/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2/
├── bd2_manifest.json
├── snapshot/
└── calibration/
```

`bd2_manifest.json` records the aggregate state: snapshot `COMPLETE`,
calibration `COMPLETE`, BD3 `NOT_RUN`, and BD4 `NOT_RUN`, with both component
manifest checksums. The snapshot manifest verifies snapshot contents; the
calibration manifest verifies fixture outputs and references the snapshot
manifest. BD2 does not create `candidate_hits.parquet`,
`candidate_anchor_evidence.parquet`, a training view, split, exclusion
manifest, or modified source corpus.

The aggregate manifest SHA-256 is
`b1708653c782c630ee4a6c49a65cf4a251c39381741c146d06c9efc379c565e0`.

The calibration manifest SHA-256 is
`904fd5dbe8175b7b2c216af69e4a592a5cca20e9314ac21c694b10471e2ea28e`. Its
fixture table SHA-256 is
`34bfe1fb61072dcca048d2df912f66fc5da2676001adfa8f221edc5a3e474c69`; the
candidate policy SHA-256 is
`0f84b35c4110aa386ed7c0fcfa88d70b21daa436b743a52a187451b21214d455`.

Commands implemented:

```sh
python3 -m cambacica.corpus.decontamination snapshot
python3 -m cambacica.corpus.decontamination verify-snapshot
python3 -m cambacica.corpus.decontamination calibrate
python3 -m cambacica.corpus.decontamination verify-calibration
python3 -m cambacica.corpus.decontamination scan --help
python3 -m cambacica.corpus.decontamination verify-scan --help
```

`scan` performs a dry-run preflight by default. A production scan requires both
`--execute-bd3` and a policy carrying `SCIENTIST_APPROVED_FOR_BD3`, approver,
and decision reference. It verifies the pinned D1 manifest, Parquet shard
inventory/schema/row totals, per-record text hash, and per-shard accounting; it
does not write source data. It writes candidate evidence and one accounting row
per shard only after explicit execution. Its candidate outputs are
`candidate_hits.parquet` and `candidate_anchor_evidence.parquet`; the latter
contains every retained approximate anchor span with original character
offsets and corpus DF. The candidate policy produced by BD2 fails the approval
gate by design.

Production candidate locators use the unchanged D2c/D1 `occurrence_id_v2`
identity derived from source, normalized shard path, and zero-based row ordinal;
the hit also records source shard/ordinal, source label, pinned corpus manifest
SHA-256, and normalized text SHA-256. The completed scan must reconcile all
21,603,689 input rows by shard, including rows with no candidate hits.

## 7. Future evaluation interface

The preferred future infrastructure remains LM Evaluation Harness. Pin the
evaluator and task adapter before C7.

- **CALAME-PT:** native Portuguese continuation, zero-shot; preserve context
  text and define first-word extraction over decoded greedy output, including
  targets spanning multiple tokenizer tokens. Normalize prompts deterministically
  and report handwritten, generated, and combined metrics separately. The
  2,075-versus-2,076 denominator decision remains open.
- **Belebele:** zero-shot Portuguese multiple choice for a causal base LM, fixed
  option ordering and A/B/C/D labels, one explicit conditional log-probability
  convention, and no chat template or instruction tuning. Report it as an
  adapted base-model protocol, not as automatically comparable to all published
  Belebele results.

No evaluation dependency or evaluator was added in BD2.

## 8. Verification results and next review

BD2 snapshot verification checks all six pinned source-file hashes, both
Parquet schemas and counts, deterministic IDs, canonical row checksums, field
normalization, and offset bounds. Calibration verification checks the result
schema, counts, and all output checksums. The same code produced identical
`fixture_results.parquet` and candidate-policy SHA-256 values in independent
runs. Focused matcher tests cover exact/approximate hits, answer-only negatives,
Unicode offsets and diacritics, high-frequency anchor promotion, repeated
ngrams, chunk boundaries, stable ordering, duplicate document IDs, memory,
interrupt cleanup, and pinned revision failure cleanup.

Before any manual BD3 handoff, scientists should review:

1. Both snapshot revision/count/schema/provenance entries and license caveats.
2. The CALAME whitespace target and intended evaluation denominator.
3. The candidate/strict comparison and whether the additional blinded
   source-stratified calibration is required before a threshold decision.
4. Exact and approximate match rules, common-anchor cutoff, question handling,
   and long-document review policy.
5. A signed/pinned policy file with an approver and decision reference. This is
   a separate decision; BD1 approval does not grant it.

Only after that review, the scientists may run these commands manually on
Orion. They are handoff instructions and **were not executed**:

```sh
python3 -m cambacica.corpus.decontamination verify-snapshot \
  --snapshot-dir /mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2/snapshot
python3 -m cambacica.corpus.decontamination verify-calibration \
  --calibration-dir /mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2/calibration
python3 -m cambacica.corpus.decontamination scan \
  --snapshot-dir /mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2/snapshot \
  --calibration-dir /mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2/calibration \
  --input-root /mnt/data/cambacica-base-180m/deduplicated/exact \
  --output-dir /mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd3/scan-v1 \
  --scratch-dir /path/to/orion/local-scratch/cambacica-bd3 \
  --policy /path/to/scientist-approved-matching-policy.json \
  --execute-bd3
python3 -m cambacica.corpus.decontamination verify-scan \
  --scan-dir /mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd3/scan-v1 \
  --input-root /mnt/data/cambacica-base-180m/deduplicated/exact
```

## 9. References

- [Pinned CALAME-PT repository revision](https://huggingface.co/datasets/NOVA-vision-language/calame-pt/tree/353671bc95cc3d94d488201f67d41b640eb80c55).
- [CALAME-PT introduction in the GlórIA paper, PROPOR 2024](https://aclanthology.org/2024.propor-1.45/).
- [Manacá evaluation report](https://github.com/Instituto-IA-LNCC/manaca-1b-base/blob/main/docs/evaluation/manaca-1b-base-eval-pt.md) and [public evaluator code](https://raw.githubusercontent.com/Instituto-IA-LNCC/manaca-1b-base/main/scripts/eval/eval_base.py). These are protocol precedents on a moving branch, not pinned CALAME specifications.
- [Pinned Belebele repository revision](https://huggingface.co/datasets/facebook/belebele/tree/d4c91dedc9de484dbea7b7d940f898f59fd135e9) and [Belebele paper, ACL 2024](https://aclanthology.org/2024.acl-long.44/).
