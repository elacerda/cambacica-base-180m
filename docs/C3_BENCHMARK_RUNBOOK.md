# C3 manual benchmark runbook — Orion

**The six primary commands below were executed on Orion GPU 0 on 2026-10-16
under scientist authorization; results are recorded in
[`C3_ARCHITECTURE_SELECTION_V1.md`](C3_ARCHITECTURE_SELECTION_V1.md).** Do
not run any benchmark command — including re-runs of the six primary
combinations or the 8,192 capacity probe — without a new explicit scientist
authorization.

The separate C3.6 scalability proposal is documented in
[`C3_TRAINING_BASELINE_V1.md`](C3_TRAINING_BASELINE_V1.md). That proposal is
not execution authorization; its GPU matrix and restart checks need a new
explicit scientist authorization.

Run the commands in a foreground shell on `cenpesor`. Do not use GPU 1, a
tmux session, a detached process, a production corpus, or a learning pilot.
The commands explicitly select `CUDA_VISIBLE_DEVICES=0`, tied mode, BF16,
microbatch 1, 20 warm-up steps, 100 timed steps, automatic SDPA, no activation
checkpointing, a 40,960 MiB maximum budget, and a 4,096 MiB safety margin.

## Read-only preflight

Before starting, confirm the host is `cenpesor` and the driver/device inventory
still matches the scientist's reviewed snapshot. `nvidia-smi` is read-only:

```bash
hostname
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free,driver_version --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
```

GPU 0 must be UUID
`GPU-7562e34d-b0a2-a725-2ad5-ed1327be3cb0` (NVIDIA H100 PCIe); GPU 1 is used
by the existing healthy `vllm-qwen38-27b` service and must not be used,
stopped, restarted, or reconfigured. The inventory's free-memory values are a
snapshot, not a reservation. If the expected identity differs or the benchmark
does not have at least 40,960 MiB free at preflight, the CLI will refuse to
start. It also verifies the visible-to-physical mapping and NVML process
visibility before allocating model memory.

Use an approved environment with PyTorch 2.3 or newer and `nvidia-ml-py`
available. The project extra is `pip install -e '.[c3]'`; use an isolated
project environment and the site's approved PyTorch wheel if setup is needed.
Do not upgrade drivers or system CUDA packages for this benchmark. The CLI
fails closed if PyTorch's allocator cap or NVML process-level monitoring is
unavailable.

The PyTorch allocator limit is set to 35,840 MiB. Per-step NVML checks stop
Cambacica above the 36,864 MiB initial process peak target, above the
40,960 MiB hard ceiling, or when total free memory drops below 4,096 MiB.
These controls do not create strict hardware isolation; see the memory-policy
limitations in [`C3_BENCHMARK_METHODOLOGY.md`](C3_BENCHMARK_METHODOLOGY.md).
Do not start the run if Orion requires a strict memory partition that has not
been provisioned.

## Six primary commands

Run one command at a time and wait for it to exit before starting the next.
Each command is self-contained and records an immutable run directory.

```bash
CUDA_VISIBLE_DEVICES=0 cambacica-benchmark \
  --architecture A --context-length 2048 --device cuda:0 --precision bf16 \
  --microbatch-size 1 --warmup-steps 20 --timed-steps 100 --seed 20261017 \
  --attention-backend auto --gpu-memory-budget-mib 40960 --safety-margin-mib 4096 \
  --tie-weights tied --activation-checkpointing none \
  --output-root /mnt/data/cambacica-base-180m/c3-benchmarks
```

```bash
CUDA_VISIBLE_DEVICES=0 cambacica-benchmark \
  --architecture A --context-length 4096 --device cuda:0 --precision bf16 \
  --microbatch-size 1 --warmup-steps 20 --timed-steps 100 --seed 20261017 \
  --attention-backend auto --gpu-memory-budget-mib 40960 --safety-margin-mib 4096 \
  --tie-weights tied --activation-checkpointing none \
  --output-root /mnt/data/cambacica-base-180m/c3-benchmarks
```

```bash
CUDA_VISIBLE_DEVICES=0 cambacica-benchmark \
  --architecture B --context-length 2048 --device cuda:0 --precision bf16 \
  --microbatch-size 1 --warmup-steps 20 --timed-steps 100 --seed 20261017 \
  --attention-backend auto --gpu-memory-budget-mib 40960 --safety-margin-mib 4096 \
  --tie-weights tied --activation-checkpointing none \
  --output-root /mnt/data/cambacica-base-180m/c3-benchmarks
```

```bash
CUDA_VISIBLE_DEVICES=0 cambacica-benchmark \
  --architecture B --context-length 4096 --device cuda:0 --precision bf16 \
  --microbatch-size 1 --warmup-steps 20 --timed-steps 100 --seed 20261017 \
  --attention-backend auto --gpu-memory-budget-mib 40960 --safety-margin-mib 4096 \
  --tie-weights tied --activation-checkpointing none \
  --output-root /mnt/data/cambacica-base-180m/c3-benchmarks
```

```bash
CUDA_VISIBLE_DEVICES=0 cambacica-benchmark \
  --architecture C --context-length 2048 --device cuda:0 --precision bf16 \
  --microbatch-size 1 --warmup-steps 20 --timed-steps 100 --seed 20261017 \
  --attention-backend auto --gpu-memory-budget-mib 40960 --safety-margin-mib 4096 \
  --tie-weights tied --activation-checkpointing none \
  --output-root /mnt/data/cambacica-base-180m/c3-benchmarks
```

```bash
CUDA_VISIBLE_DEVICES=0 cambacica-benchmark \
  --architecture C --context-length 4096 --device cuda:0 --precision bf16 \
  --microbatch-size 1 --warmup-steps 20 --timed-steps 100 --seed 20261017 \
  --attention-backend auto --gpu-memory-budget-mib 40960 --safety-margin-mib 4096 \
  --tie-weights tied --activation-checkpointing none \
  --output-root /mnt/data/cambacica-base-180m/c3-benchmarks
```

The CLI also accepts explicit `--tie-weights untied` for validation, and
`--precision fp16` or `fp32` for an explicitly configured comparison. These
choices do not replace the six proposed BF16 tied-mode measurements or select
the training configuration. A context 8,192 run needs a separate review and
the explicit `--enable-8192-capacity-test` flag.

## Monitoring and recovery

The benchmark polls NVML after every warm-up and timed optimizer step, reports
PyTorch allocated/reserved peaks separately from process-level NVML memory,
and keeps all unrelated process memory read-only. The advisory local lock
prevents overlapping Cambacica runs; it never terminates or reclaims another
process. Do not run `nvidia-smi --gpu-reset`, kill a generic Python process,
or alter the Qwen service.

On OOM, non-finite loss/gradients, kernel incompatibility, memory-policy
violation, identity failure, or NVML loss, the foreground command exits
nonzero and writes a failed artifact when storage is available. It does not
retry with a different GPU, precision, architecture, or budget. Inspect
`result.json`, `configuration.json`, and `checksums.sha256` before considering
another scientist-approved run.

Ctrl-C is handled as an interrupted benchmark: the lock is released and the
incomplete result is published with its failure diagnostics. If the process is
forcibly killed, the OS releases the lock; inspect its unique scratch run
directory and any hidden `.partial-*` NFS publication directory before retrying.
Never remove a completed run. Retrying the same configuration creates a new
run ID and cannot overwrite an earlier result. Do not start a new run while
the lock is held.

## Artifact verification

Successful and failed runs publish under
`architecture-{A|B|C}/context-{2048|4096}/<run-id>/`. Verify a result with:

```bash
cd /mnt/data/cambacica-base-180m/c3-benchmarks/architecture-A/context-2048/<run-id>
sha256sum -c checksums.sha256
```

Each run includes `configuration.json`, `result.json`, `manifest.json`, and
`checksums.sha256`. Runtime scratch stays on local `/tmp`; finalized
scientific reports stay on NFS. No benchmark artifact is stored in Git.
