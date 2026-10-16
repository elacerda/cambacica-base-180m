# C3 synthetic benchmark methodology

**Purpose:** compare implementation throughput and memory for the three C3
model shapes at contexts 2,048 and 4,096. These are six synthetic performance
measurements. They do not estimate learning quality or authorize a training
recipe. No benchmark was executed while implementing this framework; the
six-run primary matrix (C3.4) was executed on Orion GPU 0 on 2026-10-16, and
its results and the architecture decision are recorded in
[`C3_ARCHITECTURE_SELECTION_V1.md`](C3_ARCHITECTURE_SELECTION_V1.md).

## Primary matrix

| Architecture | Context 2,048 | Context 4,096 |
| --- | --- | --- |
| A — narrow/deep | One benchmark | One benchmark |
| B — balanced | One benchmark | One benchmark |
| C — wide/shallow | One benchmark | One benchmark |

All six cells were executed once on 2026-10-16 on Orion GPU 0; run IDs,
measurements, and artifact checksums are in
[`C3_ARCHITECTURE_SELECTION_V1.md`](C3_ARCHITECTURE_SELECTION_V1.md).

The default run uses BF16 autocast when GPU 0 supports it, microbatch 1, 20
warm-up steps, 100 timed steps, seed `20261017`, automatic PyTorch SDPA backend
selection, tied embeddings, and no activation checkpointing. Every candidate
uses the same AdamW settings for this synthetic comparison: learning rate
`1e-4`, betas `(0.9, 0.95)`, weight decay `0.1`, and epsilon `1e-8`. This
optimizer exists only to exercise a forward/backward/optimizer step. It does
not freeze the training optimizer or schedule.

Input IDs are generated from a CPU Torch generator using the recorded seed.
Each generated batch has `context_length + 1` IDs; the first `context_length`
are model inputs and the remaining shifted IDs are labels. No production text,
tokenizer model file, or corpus artifact is opened. The tokenizer candidate,
vocabulary, and frozen model SHA-256 are recorded as experiment identity only.

The timed step contains zeroing gradients, forward, next-token loss, backward,
finite loss/gradient validation, and an AdamW step. CUDA events bracket each
step, with synchronization before and after the timed loop. Mean, median, and
nearest-rank P95 report per-step CUDA-event time. Wall-clock mean step time
and training tokens/second include host checks and NVML safety sampling, so
they also reflect the guard overhead. Tokens/second uses
`microbatch_size × context_length × timed_steps` divided by synchronized wall
time.

The model is profiled once outside timed steps. The result includes the
requested SDPA backend and the profiler-observed kernel name when measurable.
If measurement is inconclusive, the result says `unconfirmed`; it does not
claim Flash Attention. A specifically requested backend that cannot be
confirmed fails the run.

## GPU 0 and memory policy

The only accepted manual target is physical GPU 0 on `cenpesor`, UUID
`GPU-7562e34d-b0a2-a725-2ad5-ed1327be3cb0`. Commands must set
`CUDA_VISIBLE_DEVICES=0`. Preflight requires exactly one visible CUDA device,
checks the NVML physical GPU 0 UUID, and matches the visible CUDA device to
that physical device by CUDA UUID or PCI bus ID. An unknown mapping, NVML
failure, or UUID mismatch stops the run. There is no fallback to GPU 1.

The hard operational ceiling is 40,960 MiB. The default initial process peak
target is 36,864 MiB, leaving a 4,096 MiB margin. The code caps the PyTorch
caching allocator at 35,840 MiB, reserving another 1,024 MiB below that peak
target for CUDA context and library allocations. Preflight also requires at
least 40,960 MiB free on GPU 0 at startup. During warm-up and timed steps,
read-only NVML checks require process-level memory visibility, enforce the
36,864 MiB initial peak target and 40,960 MiB hard ceiling, and stop if total
free memory drops below the configured 4,096 MiB margin. The allocator cap is
applied before model parameters are copied to CUDA. A PyTorch allocator backend
other than the native caching allocator is refused because the configured
memory fraction cannot be relied upon there.

PyTorch's caching-allocator fraction is not hardware isolation. It does not
constrain CUDA contexts, all library workspaces, direct allocations outside
the PyTorch allocator, or new allocations by unrelated GPU processes. NVML is
sampled after each optimizer step, not continuously, so a transient outside
allocation can occur between samples. The fixed 1 GiB non-allocator headroom,
4 GiB process target margin, startup free-memory check, per-step NVML monitor,
and fail-on-OOM behavior form the documented operational guard. If Orion's
operator requires strict hardware-enforced isolation, these controls are not a
partition: do not run until an external isolated allocation has been provided
and approved. The benchmark never changes device mode, resets a GPU, or
interferes with another process.

An advisory `flock` lock on local scratch prevents two Cambacica C3 runs from
overlapping on the host. The lock is non-destructive and released by the
kernel when the process exits. Other GPU processes are only read through
NVML; their memory is included in the free-memory preflight and ongoing free
memory check. No unrelated process is stopped or modified.

## Failure and measurements

OOM, unsupported kernels, non-finite loss or gradients, missing NVML process
memory, unsafe identity, lock conflicts, or memory-policy violations stop the
run. Failures are written as candidate/context-specific artifacts where
storage is available. The benchmark does not retry with another device,
architecture, precision, or memory budget. It does not delete a previous
result.

Successful results contain:

- model dimensions, tied/untied mode, analytical and instantiated parameters;
- tokenizer identity, context, microbatch, tokens per step, warm-up and timed
  counts, seed, optimizer and activation-checkpointing mode;
- mean, median, P95 step event time, wall-clock mean step, and training tokens
  per second;
- peak PyTorch allocated/reserved memory and sampled process memory from
  NVML, plus available and used GPU memory at startup;
- GPU UUID, physical index, mapping method, name, PCI bus ID, driver, CUDA and
  PyTorch versions, and requested/observed attention backend;
- loss and gradient stability, generated input SHA-256, configuration, Git
  commit/dirty status, implementation digest, and artifact checksums.

Runtime scratch is written under `/tmp/cambacica-base-180m/`; finalized
candidate-specific results are atomically published under
`/mnt/data/cambacica-base-180m/c3-benchmarks/` on NFS. Each run contains
`configuration.json`, `result.json`, `manifest.json`, and
`checksums.sha256`. A completed run is never overwritten. A failed run receives
a new unique run ID, preserving previous outcomes.

## 8,192-token capacity test

Context 8,192 is disabled unless the operator explicitly supplies
`--enable-8192-capacity-test`. This flag is only for a separate run after the
scientist has reviewed the 4,096-token outcomes and approved the capacity
probe. It is not part of the six primary measurements and does not select the
final context.
