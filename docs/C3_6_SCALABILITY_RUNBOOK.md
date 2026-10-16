# C3.6 scalability runner runbook

The C3.6 runner implements the proposal in
[`C3_TRAINING_BASELINE_V1.md`](C3_TRAINING_BASELINE_V1.md). Its GPU commands
remain disabled unless `--execute` is supplied. This implementation and its
CPU tests do not authorize the measurements.

## CPU-only validation

From the repository root, the plan can be checked without importing PyTorch,
initializing CUDA, or opening a corpus:

```bash
PYTHONPATH=src /srv/cambacica-c3/venv/bin/python \
  -m cambacica.benchmark.scalability_cli matrix \
  --suite-id c36-review-20261016 --dry-run
```

The dry run validates the versioned baseline and all nine cells, prints the
exact single-cell commands, and prints the restart and finalization commands.
The matrix uses Architecture C, tied weights, 4,096 tokens, BF16 autocast,
five warm-up updates, 20 measured updates, one repeat, and the recorded C3.4
synthetic AdamW fixture. It uses one seeded CPU-generated stream; the same
sample rows are regrouped into the approved microbatches. No corpus or
tokenizer artifact is opened.

## Proposed GPU execution

After separate scientist authorization, run this one foreground command to
execute all nine cells sequentially under the local C3 lock:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  /srv/cambacica-c3/venv/bin/python \
  -m cambacica.benchmark.scalability_cli matrix \
  --suite-id c36-20261016-orion-gpu0 --execute
```

This matrix command holds the lock throughout all cells. The CLI also prints
the exact individual commands below for a scientist who chooses to run one
cell at a time. Each command is foreground-only and must finish before the
next begins:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /srv/cambacica-c3/venv/bin/python -m cambacica.benchmark.scalability_cli run-cell --suite-id c36-20261016-orion-gpu0 --microbatch-size 1 --gradient-accumulation-steps 1 --execute
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /srv/cambacica-c3/venv/bin/python -m cambacica.benchmark.scalability_cli run-cell --suite-id c36-20261016-orion-gpu0 --microbatch-size 1 --gradient-accumulation-steps 2 --execute
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /srv/cambacica-c3/venv/bin/python -m cambacica.benchmark.scalability_cli run-cell --suite-id c36-20261016-orion-gpu0 --microbatch-size 1 --gradient-accumulation-steps 4 --execute
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /srv/cambacica-c3/venv/bin/python -m cambacica.benchmark.scalability_cli run-cell --suite-id c36-20261016-orion-gpu0 --microbatch-size 2 --gradient-accumulation-steps 1 --execute
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /srv/cambacica-c3/venv/bin/python -m cambacica.benchmark.scalability_cli run-cell --suite-id c36-20261016-orion-gpu0 --microbatch-size 2 --gradient-accumulation-steps 2 --execute
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /srv/cambacica-c3/venv/bin/python -m cambacica.benchmark.scalability_cli run-cell --suite-id c36-20261016-orion-gpu0 --microbatch-size 2 --gradient-accumulation-steps 4 --execute
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /srv/cambacica-c3/venv/bin/python -m cambacica.benchmark.scalability_cli run-cell --suite-id c36-20261016-orion-gpu0 --microbatch-size 4 --gradient-accumulation-steps 1 --execute
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /srv/cambacica-c3/venv/bin/python -m cambacica.benchmark.scalability_cli run-cell --suite-id c36-20261016-orion-gpu0 --microbatch-size 4 --gradient-accumulation-steps 2 --execute
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /srv/cambacica-c3/venv/bin/python -m cambacica.benchmark.scalability_cli run-cell --suite-id c36-20261016-orion-gpu0 --microbatch-size 4 --gradient-accumulation-steps 4 --execute
```

Run the separate fresh-process checkpoint/restart validation after the matrix:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  /srv/cambacica-c3/venv/bin/python \
  -m cambacica.benchmark.scalability_cli restart-check \
  --suite-id c36-20261016-orion-gpu0 --execute
```

It compares four uninterrupted updates with two updates, a checksummed local
checkpoint, process exit, validated reload, and two resumed updates. It keeps
each worker under the shared C3 lock and starts the next only after the prior
process exits. The deterministic mode is strict, leaves SDPA on `auto`, and
never changes kernels to force a match.
Unsupported deterministic kernels and numerical mismatches are recorded as
`FAILED`, with the state deltas and failure diagnostics.

After all nine cell artifacts and the terminal restart report exist, verify
and publish the report-only bundle atomically to NFS:

```bash
PYTHONPATH=src /srv/cambacica-c3/venv/bin/python \
  -m cambacica.benchmark.scalability_cli finalize-suite \
  --suite-id c36-20261016-orion-gpu0
```

Active checkpoints and model-state snapshots stay under
`/tmp/cambacica-base-180m/c3.6-scalability/<suite-id>/restart/`. The published
suite is immutable at
`/mnt/data/cambacica-base-180m/c3-scalability/c3.6/<suite-id>/` and contains
cell configurations and results, `matrix.json`, `results.csv`,
`restart_verification.json`, `manifest.json`, and `checksums.sha256`.

## Runtime and memory expectations

The nine cells contain 5,017,600 training tokens including warm-up, plus
32,768 restart-validation tokens: 5,050,368 total and 233 optimizer updates.
The earlier Architecture-C 4,096-token C3.4 measurement observed 59,995.6
tokens/second. That makes about 84 seconds of throughput-equivalent work; a
full invocation should be expected to take several minutes after model
initialization, per-cell profiling, and NVML checks. This is an estimate, not a
measurement of the C3.6 cells.

The previous C3.4 Architecture-C, microbatch-1, context-4,096 process peak was
6,862 MiB. C3.6's largest microbatch is four and its largest effective batch
is 16 sequences. Activation, logits, attention workspaces, and CUDA library
allocations can raise peak memory substantially; the exact peak is unmeasured.
The runner refuses startup below 40,960 MiB free and stops above the 36,864
MiB process target, the 40,960 MiB hard budget, or below 4,096 MiB free. The
PyTorch allocator remains capped at 35,840 MiB. There are no automatic retries,
batch changes, precision changes, or alternate-device paths.

These controls preserve the reviewed physical GPU 0 UUID and CUDA/NVML mapping,
use read-only process monitoring, and never alter GPU 1 or its Qwen service.
They are operational limits rather than hardware isolation. The C3.6 results
remain engineering scalability measurements and do not select a training
optimizer, batch, schedule, token budget, or quality outcome.
