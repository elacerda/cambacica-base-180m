# C1-S2 — deterministic tokenizer sample

**Status: COMPLETE / VERIFIED (2026-10-14).** Production selection passed,
materialization completed, and independent verification passed. The published
sample is preserved as C1-S2 v1 at
`/mnt/data/cambacica-base-180m/tokenizer-sample/c1-s2-v1/`.

| Production result | Value |
| --- | --- |
| Selection | `PASS` |
| Materialization | `MATERIALIZED` |
| Independent verification | `PASS` |
| Documents | 46,583 |
| Normalized words | 19,999,380 |
| Shortfall from 20,000,000-word target | 620 words (0.0031%) |
| Selection manifest SHA-256 | `6086185af09357a6393b5de33d5a52442fd41caaa146191129e818bc19b45366` |
| Sample manifest SHA-256 | `93c6c5400115c80aa5f5a63b4b593a5bc45bc46ecea429c78ffa5718f978e97c` |
| Published selection checksum manifest | PASS |
| Published sample checksum manifest | PASS |

The sample uses provisional mixture B, seed **20261014**, and train-only
references from C1-S1. All source and GigaVerbo subset quota tolerances passed.
These results fix the identity of C1-S2 v1; preserve it as published. Any
modified taxonomy-balanced sample must receive a separate version and scientific
approval.

## Carolina taxonomy composition and representativeness

The 5,600,000 Carolina words in this sample are distributed as follows:

| Carolina taxonomy | Normalized words |
| --- | ---: |
| `wik` | 3,375,808 |
| `leg` | 1,746,973 |
| `uni` | 327,365 |
| `dat` | 110,866 |
| `soc` | 38,988 |
| `jud` | 0 |
| `pub` | 0 |

`wik` plus `leg` account for approximately **91.5%** of Carolina words in this
tokenizer sample. This concentration does not violate the approved source-level
quota policy, but it may affect tokenizer representativeness across Portuguese
registers. Do not describe the sample as linguistically unbiased or uniformly
representative. During C2, measure fragmentation and coverage by domain; no such
measurements are part of this C1-S2 production closure.

## Pinned inputs and sample target

The command pins these input artifacts by their exact manifest and payload
identities:

| Input | Path | SHA-256 |
| --- | --- | --- |
| Post-D1 corpus manifest | `/mnt/data/cambacica-base-180m/deduplicated/exact/manifest.json` | `57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428` |
| C1-S1 split manifest | `/mnt/data/cambacica-base-180m/splits/c1-s1-v1/manifest.json` | `8b937f90d40487c643fe48baa6b5c2950713fe2e0d2d534f3e2a3c08e74dbe16` |
| C1-S1 assignments | `/mnt/data/cambacica-base-180m/splits/c1-s1-v1/assignments.parquet` | `ef52e3cf3033934df13e3d16c207e1783c509dff6fb91954fb8c25ab719ece1e` |
| Reference mixture config | `configs/corpus_mix_b.yaml` | `4ff5faac9dc49466a3c19a6f8e020e340aead613a339ee7bcd62820c5a759d95` |

The frozen policy bytes are
[`configs/c1_tokenizer_sample_policy.json`](../configs/c1_tokenizer_sample_policy.json)
(SHA-256 `eb903e329d2f260ac117f4e8730171d728dc76df621f9a533e39903e25df8797`).
The CLI rejects policy or reference-mixture bytes that differ from these
identities.

Target: **20,000,000 normalized words** (`len(text.split())` on stored text),
mixture **B**, seed **20261014**. The shares and corresponding word targets are:

| Source | Share | Target words |
| --- | ---: | ---: |
| Carolina | 28% | 5,600,000 |
| Wikipedia PT | 18% | 3,600,000 |
| ParlamentoPT | 10% | 2,000,000 |
| Gutenberg PT | 4% | 800,000 |
| GigaVerbo residual | 40% | 8,000,000 |

The GigaVerbo residual keeps all eleven weights from `corpus_mix_b.yaml`:

| GigaVerbo subset | Weight within residual | Target words |
| --- | ---: | ---: |
| `finepdfs_por_Latn` | 30% | 2,400,000 |
| `crawlPT_dedup` | 10% | 800,000 |
| `quati` | 5% | 400,000 |
| `blogset` | 5% | 400,000 |
| `fineweb_2_pt` | 35% | 2,800,000 |
| `mc4_pt` | 5% | 400,000 |
| `hplt2_pt` | 5% | 400,000 |
| `hplt1_pt` | 2% | 160,000 |
| `common_crawl` | 1% | 80,000 |
| `oscar` | 1% | 80,000 |
| `culturax` | 1% | 80,000 |

The selector reads only references whose C1-S1 `split` is `train`. It uses the
stable C1-S1 `doc_id`, rejects duplicate IDs or source coordinates, and keeps
whole documents. It ranks documents with SHA-256 over the seed and `doc_id`,
then the ID breaks ties. A document is included only if its complete word count
fits the remaining source or subset quota. No quota is topped up from another
source, and no quota can be exceeded.

### Bounded materialization frame

The D1 files are Parquet. Their sampled footers show row groups of up to 4,096
records, so reading a random 0.1% of documents by row group can decompress most
of the corpus. To keep `materialize` from scanning D1, the proposed policy
selects a deterministic row-group frame first. For each fixed source or
GigaVerbo-subset bucket, row groups are ranked by SHA-256 over the seed, bucket,
source shard, and row-group number. The frame takes groups in that order until
their train-reference word capacity reaches ten times that bucket's target, or
until no groups remain. Document ranking then selects complete records inside
that frame.

This frame is a deterministic locality bound over the approved train-only
population; it does not alter or rewrite D1. Its row-group count, candidate
documents, and word capacity are reported per bucket in `selection_manifest`.
Only D1 row groups containing selected references are read by materialization.
Those groups can contain unselected neighboring records, which are discarded
without being written. A large row group can make physical reads exceed the
ten-times word-capacity target. The scientist should inspect the reported frame
sizes before proceeding to materialization.

## Proposed frozen quota tolerances

These tolerances are part of the proposed policy because whole documents cannot
always land exactly on an integer word target:

- Total sample: at most **1% shortfall** from 20,000,000 words.
- Each source quota: at most **1% shortfall**; Gutenberg PT allows **10%** due
  to whole-book boundaries.
- Each of the eleven GigaVerbo subset quotas: at most **1% shortfall**.
- Overshoot: **0 words** at every bucket and at the total.

Actual words, signed deviation, shortfall basis points, and pass/fail are written
to `selection_manifest.json`. If any tolerance fails, `select` still publishes
the diagnostics with status `QUOTA_TOLERANCE_FAILED`; `materialize` refuses that
selection. There is no redistribution. The production result above passed these
frozen tolerances.

## CLI interface

Run from the repository root with the project environment installed, or set
`PYTHONPATH=src` as in the commands below:

```text
python3 -m cambacica.corpus.tokenizer_sample select \
  --input-root PATH --split-root PATH --policy PATH \
  --scratch-dir LOCAL_PATH --output-dir PATH [--batch-size N]

python3 -m cambacica.corpus.tokenizer_sample materialize \
  --input-root PATH --split-root PATH --policy PATH \
  --selection-dir PATH --scratch-dir LOCAL_PATH --output-dir PATH \
  [--batch-size N]

python3 -m cambacica.corpus.tokenizer_sample verify \
  --input-root PATH --split-root PATH --policy PATH \
  --selection-dir PATH --sample-dir PATH --scratch-dir LOCAL_PATH \
  [--batch-size N]
```

All three commands require an explicit `--scratch-dir`. Defaults are 4,096
assignment rows per batch for `select` and `verify`, and 1,024 D1 rows per batch
for `materialize`. Batch size can change on resume without changing the
selection. The same scratch location must be retained while resuming an
interrupted operation.

## Expected directories and artifacts

Production output root:

```text
/mnt/data/cambacica-base-180m/tokenizer-sample/c1-s2-v1/
├── selection/
│   ├── selected_references.parquet
│   ├── selection_manifest.json
│   └── checksums.sha256
└── sample/
    ├── records/part-<source-shard-hash>.parquet
    ├── sample_manifest.json
    └── checksums.sha256
```

`selected_references.parquet` stores the stable document and group IDs, source,
subset, D1 shard and row coordinates, text hash, normalized word count, train
split, bucket, and both deterministic rank hashes. The sample Parquet preserves
the full D1 row, including original text and source/provenance columns, and adds
the sample identity and split fields.

The SQLite working files belong on node-local storage, for example:

```text
/tmp/c1-tokenizer-sample-v1/
├── selection.sqlite3
├── materialize.sqlite3
├── verification.sqlite3
└── verification_refs.sqlite3
```

Selection checkpoints assignment row groups in SQLite. On a successful
`PASS`, its large selection database is removed after publication; an
interrupted run keeps it for resume. Materialization checkpoints each completed
source shard and retains its smaller SQLite state. Verification keeps its
independent reconstruction state so it can resume. Do not put these databases
under `/mnt/data` or another NFS mount.

Each writer builds a sibling `.partial` directory and atomically renames it to
the final path only after its manifest and checksums are complete. Existing
published output is never overwritten. A compatible final artifact is checked
and returned on a repeated materialization command.
Before writing, the CLI checks Linux mount metadata: scratch cannot resolve to
NFS/NFSv4, and the output path must resolve to the same filesystem type as the
pinned input.

## Archived production command reference for `cenpesor`

Production is complete. The commands below preserve the planned operator
procedure for reference; do not rerun them to recreate C1-S2 v1. The published
output and its identities are recorded above. Any changed sample requires a new
version and scientific approval. The procedure calls for at least 30 GB of
local scratch and 2 GB free on the output filesystem; its example uses `/tmp`.

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
export DATA_ROOT="/mnt/data/cambacica-base-180m"
export SAMPLE_ROOT="$DATA_ROOT/tokenizer-sample/c1-s2-v1"
export SAMPLE_SCRATCH="/tmp/c1-tokenizer-sample-v1"
cd "$REPO_ROOT"
set -euo pipefail
mkdir -p "$SAMPLE_ROOT" "$SAMPLE_ROOT/logs" "$SAMPLE_SCRATCH"
df -h "$SAMPLE_ROOT" "$SAMPLE_SCRATCH"
findmnt -T "$SAMPLE_SCRATCH"
```

Start with selection and inspect the resulting status and frame report:

```bash
PYTHONPATH="$REPO_ROOT/src" python3 -m cambacica.corpus.tokenizer_sample select \
  --input-root "$DATA_ROOT/deduplicated/exact" \
  --split-root "$DATA_ROOT/splits/c1-s1-v1" \
  --policy "$REPO_ROOT/configs/c1_tokenizer_sample_policy.json" \
  --scratch-dir "$SAMPLE_SCRATCH" \
  --output-dir "$SAMPLE_ROOT/selection" \
  --batch-size 4096 \
  2>&1 | tee -a "$SAMPLE_ROOT/logs/select.log"

python3 -c 'import json,sys; p=sys.argv[1]; m=json.load(open(p)); print(json.dumps({"status":m["status"],"selected_documents":m["selected_documents"],"selected_normalized_words":m["selected_normalized_words"],"quotas":m["quotas"],"quota_report":m["quota_report"]},indent=2)); sys.exit(m["status"] != "PASS")' \
  "$SAMPLE_ROOT/selection/selection_manifest.json"
```

Only proceed when the status is `PASS` and the reported frame capacities are
acceptable. Then materialize selected records:

```bash
PYTHONPATH="$REPO_ROOT/src" python3 -m cambacica.corpus.tokenizer_sample materialize \
  --input-root "$DATA_ROOT/deduplicated/exact" \
  --split-root "$DATA_ROOT/splits/c1-s1-v1" \
  --policy "$REPO_ROOT/configs/c1_tokenizer_sample_policy.json" \
  --selection-dir "$SAMPLE_ROOT/selection" \
  --scratch-dir "$SAMPLE_SCRATCH" \
  --output-dir "$SAMPLE_ROOT/sample" \
  --batch-size 1024 \
  2>&1 | tee -a "$SAMPLE_ROOT/logs/materialize.log"
```

After materialization, run independent verification:

```bash
PYTHONPATH="$REPO_ROOT/src" python3 -m cambacica.corpus.tokenizer_sample verify \
  --input-root "$DATA_ROOT/deduplicated/exact" \
  --split-root "$DATA_ROOT/splits/c1-s1-v1" \
  --policy "$REPO_ROOT/configs/c1_tokenizer_sample_policy.json" \
  --selection-dir "$SAMPLE_ROOT/selection" \
  --sample-dir "$SAMPLE_ROOT/sample" \
  --scratch-dir "$SAMPLE_SCRATCH" \
  --batch-size 4096 \
  2>&1 | tee -a "$SAMPLE_ROOT/logs/verify.log"
```

Expected final CLI output is JSON with `"status":"PASS"`, document and word
counts, input identities, and the quota report. `verify` independently rebuilds
the deterministic selection from the full C1-S1 assignment stream, checks
train-only membership and stable IDs, retrieves the selected D1 row groups
again, verifies each selected text hash and whole-document word count, checks
source/subset counts and quotas, and validates all published checksums.
Verification reads the full assignment artifact but does not scan all D1 text.

### Resume commands

After interruption, rerun the exact operation command above with the same
`SAMPLE_SCRATCH`, inputs, policy, and output paths. Selection replays only an
unfinished assignment row group; materialization replays only an unfinished or
damaged staged source-shard part; verification reuses its independent ranking
checkpoint. Identity changes fail closed. Do not remove a `.partial` directory
or SQLite state while trying to resume. If an output has already been atomically
published, rerun `verify` to check it rather than rebuilding it in place.

## Archived optional `tmux` procedure

This is a persistence suggestion for long-running operator commands, not a
record of how the completed production was run:

```bash
tmux new -s c1-s2-sample
```

Inside that session, run the setup and selection commands above. Detach with
`Ctrl-b` then `d`; reconnect with `tmux attach -t c1-s2-sample`. Inspect the
selection manifest before starting `materialize`, then inspect the materialized
manifest before starting `verify`. Keep the log files under `logs/` and retain
the local scratch directory until verification passes.

## Resource estimates and remaining limitations

- **Local scratch:** the pre-production estimate was at least **30 GB free**.
  Selection and independent verification each stream 21,603,689 assignment rows
  through a disk-backed SQLite table and a rank index. The SQLite database was
  expected to use roughly 12–25 GB, depending on text metadata lengths and
  SQLite page overhead. These are planning estimates, not recorded production
  measurements.
- **Memory:** the pre-production recommendation was at least **8 GB RAM**.
  Assignment and source text are processed in bounded batches; a single
  exceptionally long document can still raise peak batch memory. The production
  handoff captured here does not include measured peak memory.
- **Output and runtime:** the pre-production estimate reserved at least **2 GB**
  for the selected sample and temporary publication stage and budgeted an
  overnight run. Actual output size, elapsed times, and production throughput
  measurements are not included in this closure summary.
- **D1 reads:** selection and verification read D1 Parquet footers for the
  pinned shard layout. Materialization reads only selected row groups, but each
  selected group is decompressed and may contain unselected neighbors. Consult
  the selection manifest for the actual frame report; this closure summary does
  not claim a physical-byte read total.
- **Checksums:** materialization recomputes SHA-256 and word counts for each
  selected text, checks source revision/subset/document identity, and validates
  the pinned D1 manifest plus source-file size and read-time stat. It does not
  hash every byte of each full D1 shard, which would defeat selective
  materialization.
- **Whole-document boundaries:** a frame with enough aggregate words can still
  miss a quota because documents are indivisible. The frozen tolerance catches
  this and blocks materialization instead of borrowing from another bucket.
- **Rights:** C1-S2 changes no inherited source licensing decisions. Existing
  C1-S1 rights caveats remain applicable.

## Implementation validation and production verification

The synthetic test module covers changed assignment traversal and batch sizes,
train-only membership, stable IDs, whole-document counts and text hashes,
quota failure reporting, interruption/restart for selection and materialization,
selective row-group reads, independent reconstruction, and artifact tampering.
The real-data checks were limited to pinned manifest/checksum identities, the
2.48 GB assignments payload SHA-256, the assignment Parquet footer,
representative D1 Parquet footers, and 16 records from one D1 row group with
text-hash/source checks. A separate 16-ID lookup through assignment row-group
statistics was stopped after 30 seconds because footer-stat traversal was
slower than expected; it did not decode assignment rows. These bounded
implementation checks preceded production. The full production selection,
materialization, publication, and independent verification are complete as
summarized at the top of this document; both published checksum manifests
passed.
