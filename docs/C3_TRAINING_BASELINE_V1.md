# C3.6 Architecture-C training baseline — version 1

**Decision recorded 2026-10-16: SCIENTIST APPROVED baseline fields.** This
version records the approved Architecture-C identity, context, tying,
parameter count, reference compute precision, and Orion target. Scalability
measurements remain pending and require separate explicit authorization. This
record does not freeze a complete training recipe.

The machine-readable contract is
[`architecture_c_c3_6_v1.json`](../configs/architecture_c_c3_6_v1.json).
It is validated by `cambacica.model.baseline.load_training_baseline()` and
`validate_training_baseline()`. A change to an approved field requires a new
versioned config and scientific decision record. The immutable C3.5 selection
record was not changed.

## Approved baseline and frozen inputs

| Field | Approved or frozen value |
| --- | --- |
| Architecture ID | `C` (`wide-shallow`), inherited from C3.5 |
| Shape | 9 layers, hidden size 1,024, 16 heads × 64, SwiGLU width 4,096 |
| Context length | 4,096 tokens |
| Embedding/output weight tying | Enabled |
| Parameter count | 176,180,224 with tying |
| Reference compute precision | BF16 autocast |
| Target | Orion physical GPU 0 only, host `cenpesor`, UUID `GPU-7562e34d-b0a2-a725-2ad5-ed1327be3cb0` |
| Tokenizer | C2-frozen SentencePiece Unigram `unigram-v24576`, vocabulary 24,576 |
| Tokenizer model identity | SHA-256 `c8982a23d3c5ecc0a61d300edc1e354e38f1358aa3db5420d5bf19f7b95cbf23` |
| Tokenizer vocabulary identity | SHA-256 `f27e38de1a1e0e2e96ac00f50382d1be749d2b357cb3e8ad9ee43bf8ae972d9f` |

The baseline validator checks the architecture ID and full frozen shape, the
4,096-token context, enabled tying, exact C2 tokenizer identity (including
model and vocabulary checksums and special-token IDs), the derived tied
parameter count, reference precision, and the pinned physical GPU identity.
It also requires the versioned record to state that no final hyperparameters
have been selected.

## Decision categories

The config keeps four kinds of information separate:

- **Scientist-approved:** the baseline fields in the table above.
- **Measured baseline choices:** C3.4 used microbatch 1, BF16 autocast,
  float32 model parameters and gradients, AdamW with its recorded synthetic
  settings, no activation checkpointing, 20 warm-up updates, and 100 timed
  updates. These describe a synthetic benchmark implementation, not approved
  training settings.
- **Provisional hyperparameters:** none selected in version 1. The microbatch
  and accumulation grid below is a measurement proposal; it does not select an
  effective batch or training hyperparameter.
- **Unresolved training decisions:** final optimizer and settings, learning
  rate schedule and warmup, microbatch and accumulation/effective batch, total
  training tokens, activation checkpointing, parameter/gradient/optimizer
  storage layout, checkpoint cadence and retention, final corpus mixture and
  C1 policy resolution, and C4 packing behavior at context 4,096. C3.7 pilot
  authorization and its token count also remain open.

The C3.4 AdamW learning rate, betas, weight decay, and epsilon remain in the
config only as the exact synthetic benchmark fixture. BF16 is approved as the
reference compute precision; that decision does not by itself select parameter,
gradient, or optimizer-state storage dtypes.

## Bounded scalability proposal

The proposed follow-up is one sequential pass over the nine cells below. Each
cell uses context 4,096, 5 warm-up optimizer updates, 20 measured optimizer
updates, and one repeat. The synthetic token stream is generated from a fixed
seed; identical effective-token cells use the same token IDs split into the
listed microbatches. No corpus or quality evaluation is part of this test.

| Microbatch | Accumulation steps | Tokens per optimizer update |
| ---: | ---: | ---: |
| 1 | 1 | 4,096 |
| 1 | 2 | 8,192 |
| 1 | 4 | 16,384 |
| 2 | 1 | 8,192 |
| 2 | 2 | 16,384 |
| 2 | 4 | 32,768 |
| 4 | 1 | 16,384 |
| 4 | 2 | 32,768 |
| 4 | 4 | 65,536 |

Accumulation steps test throughput and update accounting while retaining one
microbatch's activations at a time. The grid is bounded at microbatch 4 and
accumulation 4. It does not authorize larger values, automatic retries, or a
choice of final effective batch. Record the seed, input-stream hash, optimizer
fixture, observed SDPA backend, and exact cell configuration for each result.

## Full training-state memory accounting

Report each term and the observed process peak; do not infer total fit from
parameter count alone. For `P = 176,180,224` unique tied parameters:

| Memory term | C3 implementation reference | Accounting |
| --- | --- | --- |
| Parameters | FP32 in the current model implementation | `P × 4` bytes = 704,720,896 bytes (672.07 MiB) |
| Gradients | FP32 in the current implementation after backward | `P × 4` bytes = 704,720,896 bytes (672.07 MiB) |
| Optimizer moments | Two FP32 AdamW moments in the C3.4 fixture | `P × 2 × 4` bytes = 1,409,441,792 bytes (1,344.15 MiB) |
| Persistent subtotal | C3.4 reference layout after optimizer state is initialized | 2,818,883,584 bytes (2,688.30 MiB), before activations and workspaces |
| Activations and temporary tensors | Depends on microbatch, kernels, and implementation | Measure forward saved tensors, backward workspaces, logits, float32 cross-entropy conversion, and temporary buffers; report peak PyTorch allocated and reserved memory |
| Non-PyTorch process allocations | CUDA context, driver, and library allocations outside the PyTorch caching allocator | Sample the Cambacica PID with NVML; `NVML process memory − PyTorch reserved` is an estimate, not an exact partition |
| Other GPU use | Processes outside Cambacica | Record NVML process inventory and free memory; include it in preflight and safety-margin checks |

The persistent subtotal is a measured-implementation reference, not a final
optimizer or dtype decision. A different optimizer may use a different number
or dtype of state tensors, and a different storage policy changes parameter
and gradient memory. The full report must include all persistent state,
activation/temporary peaks, and non-PyTorch process memory. NVML sampling after
each update can miss a short-lived peak, so report its sampled process peak
with that limitation and retain PyTorch peak allocated/reserved counters.

Every proposed GPU run must preserve the existing fail-closed controls:

- `CUDA_VISIBLE_DEVICES=0`, exactly one visible CUDA device, physical GPU 0
  UUID validation through NVML, and CUDA-to-NVML identity verification;
- the host-local execution lock and read-only per-process NVML monitoring;
- 40,960 MiB hard process budget, 36,864 MiB process peak target, 35,840 MiB
  PyTorch allocator cap, and at least 4,096 MiB safety margin;
- at least 40,960 MiB free at startup under the existing preflight rule.

NVML is sampled after each warm-up and measured optimizer update. Stop if the
process exceeds 36,864 MiB, the hard 40,960 MiB budget is reached, or total
free memory falls below 4,096 MiB.

Any identity mismatch, unavailable process-memory visibility, lock conflict,
OOM, non-finite value, allocator-policy failure, or memory-limit violation
stops the run. It must not retry on GPU 1, alter a batch or precision, or
change, stop, or reconfigure the Qwen service. These controls do not provide
hardware isolation; the runbook and C3.4 methodology retain that limitation.
The existing host-local lock path is
`/tmp/cambacica-base-180m/c3-benchmark.lock`.

## Checkpoint, resume, and deterministic restart protocol

Use only fixed synthetic token IDs in this verification. The C3.4 AdamW
fixture may populate optimizer state for this protocol, clearly labeled as a
restart-test fixture rather than the selected training optimizer. Checkpoint
only at completed optimizer-update boundaries. Each checkpoint must contain:

- model and optimizer states, optional scaler and scheduler states when the
  tested implementation uses them, and the completed optimizer-update index;
- Python, NumPy (if used), CPU Torch, and CUDA RNG states, plus the configured
  seed and deterministic-kernel settings;
- data-stream seed/hash and the next committed synthetic sample/token offset;
- baseline/configuration identity, tokenizer identity, code commit and dirty
  status, software/hardware identity, and checkpoint schema version.

Write checkpoints to a unique local scratch directory using a temporary file,
flush and fsync it, then atomically rename it. Resume only after validating
the checkpoint checksum and every pinned identity. Keep the prior committed
checkpoint until the new one passes validation. Final reports and checksummed
metadata are published immutably to NFS; active checkpoint files stay on local
storage.

For the determinism comparison, run four updates uninterrupted. In a separate
fresh process with the same baseline, seed, software, device, and backend, run
two updates, save, exit, reload, and run two more. Compare the resumed loss
sequence and final model and optimizer tensors against the uninterrupted
four-update reference. Require bitwise equality when the selected kernels
support deterministic execution. If they do not, report the exact setting and
numeric deltas and mark deterministic restart **FAILED**; do not silently
switch kernels. A mismatch, invalid checkpoint, or config-identity mismatch
fails the verification and preserves both the failure report and checkpoint
for inspection. A failure before a new commit resumes from the last valid
update boundary, replaying uncommitted synthetic batches.

## Artifacts, metadata, and performance report

Keep active runtime data and checkpoints under local
`/tmp/cambacica-base-180m/c3.6-scalability/<suite-id>/`. Validate storage with
the existing storage-layout check, then publish finalized reports atomically
and immutably beneath
`/mnt/data/cambacica-base-180m/c3-scalability/c3.6/<suite-id>/` on NFS.
Preserve the local C3 execution lock. A suite should contain:

- one `configuration.json` per cell and a suite-level `matrix.json`;
- one `result.json` per cell, a summary `results.csv`, and a
  `restart_verification.json`;
- a suite `manifest.json` and `checksums.sha256` covering every published
  report. Completed suite IDs are never overwritten.

Record baseline and cell IDs; context, microbatch, accumulation and tokens per
update; seed and input hash; optimizer fixture and activation-checkpointing
mode; parameter/gradient/moment accounting; warm-up and measured update
counts; Python, PyTorch, CUDA and driver versions; code commit, dirty status
and implementation digest; host, GPU index/UUID/PCI ID and visible-device
mapping; requested/observed attention backend; memory-policy values; failure
status and diagnostics; and all artifact checksums.

Report per-cell optimizer-update latency (mean, median, P95), wall-clock
tokens/second including safety checks, GPU-event timing where available,
PyTorch allocated/reserved peaks, sampled Cambacica process peak, startup and
minimum free memory, and the estimated non-PyTorch allocation. The report is
an engineering scalability result only. It cannot choose a training batch,
optimizer, schedule, token budget, or quality outcome.

## C3.6 status

The versioned baseline and CPU validation are prepared. The nine-cell GPU
matrix, checkpoint round trip, and deterministic-restart comparison have not
run. They require a separate explicit scientist authorization. C3 remains in
progress; this baseline does not authorize a quality pilot, corpus processing,
pretraining, or C4 execution.

The baseline/configuration tests do not require PyTorch. The model-forward
tests use the optional `c3` dependency and are skipped by pytest when PyTorch
is unavailable; that skip does not replace the CPU baseline validation.
