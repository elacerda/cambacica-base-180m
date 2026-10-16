# C3 architecture selection — version 1

**Decision recorded 2026-10-16: SCIENTIST APPROVED.** Architecture C
(wide/shallow) is the selected Cambacica-180M base-model architecture. The
decision rests on the measured computational efficiency of the C3.4 synthetic
benchmark matrix on Orion GPU 0, **not** on demonstrated language-model
quality: no learning-quality pilot was run, the benchmark inputs are synthetic
random token IDs, and their measured losses must not be interpreted as
evidence of linguistic quality. This is an immutable decision record; a later
scientific change must create a new version.

## C3 stage map

| Stage | Scope | Status |
| --- | --- | --- |
| C3.1 | Architecture planning and candidate parameter accounting | COMPLETE — [`C3_ARCHITECTURE_PLANNING.md`](C3_ARCHITECTURE_PLANNING.md) |
| C3.2 | PyTorch implementation and bounded local validation | COMPLETE — [`C3_ARCHITECTURE_IMPLEMENTATION.md`](C3_ARCHITECTURE_IMPLEMENTATION.md) |
| C3.3 | Synthetic benchmark methodology and manual runbook | COMPLETE — [`C3_BENCHMARK_METHODOLOGY.md`](C3_BENCHMARK_METHODOLOGY.md), [`C3_BENCHMARK_RUNBOOK.md`](C3_BENCHMARK_RUNBOOK.md) |
| C3.4 | Six-run synthetic benchmark matrix on Orion GPU 0 | COMPLETE, 2026-10-16 (recorded in this document) |
| C3.5 | Architecture-selection decision | **APPROVED, 2026-10-16** (this document) |
| C3.6 | Training configuration and scalability validation | PLANNED — next stage, see [Next C3 stage](#next-c3-stage-c36---training-configuration-and-scalability-validation) |
| C3.7 | Controlled quality pilot | PLANNED — gated on the corpus gates and a separate approval, see [Quality pilot gate](#c37-quality-pilot-gate) |

## Approved architectural dimensions

The selected shape is proposal C of the C3 planning matrix, preserved exactly
as implemented in `src/cambacica/model/config.py`:

| Field | Approved value |
| --- | --- |
| Architecture ID / name | `C` / `wide-shallow` |
| Transformer layers | 9 |
| Hidden size `d` | 1,024 |
| Attention heads | 16 |
| Head dimension | 64 |
| SwiGLU intermediate size `f` | 4,096 |
| Vocabulary size `V` | 24,576 (C2-frozen `unigram-v24576`) |
| Normalization | Pre-norm RMSNorm before attention and feed-forward in every block, plus a final RMSNorm; float32 reductions |
| Positions | RoPE applied to Q and K; no learned positional parameters |
| Attention | Full multi-head causal attention via PyTorch scaled dot-product attention with `is_causal=True` and zero dropout |
| Projections | Bias-free Q, K, V, output, gate, up, down |
| Parameters (tied) | 176,180,224 |
| Parameters (untied) | 201,346,048 |

The C3.4 matrix ran all six cells in tied mode. **The final tying policy is
not selected by this decision**; the untied total above is a comparison case,
not an approved architecture.

## Selection rationale: measured computational efficiency

In every compared context, Architecture C achieved the best training
throughput and the lowest peak process memory of the three proposals:

| Context | C vs B throughput | C vs A throughput |
| --- | ---: | ---: |
| 2,048 | +15.9% | +37.8% |
| 4,096 | +11.2% | +33.6% |

Peak process memory (NVML) at 2,048 was 4,974 MiB for C versus 5,350 MiB for
B and 5,674 MiB for A; at 4,096 it was 6,862 MiB versus 7,432 MiB and 7,986
MiB. C's wider, shallower shape is more compute-efficient per step on the
H100 at these contexts. This ranking is a measured efficiency result on the
C3.4 synthetic recipe; it says nothing about which shape would learn
Portuguese better, and no quality comparison was made or authorized.

## C3.4 benchmark matrix — six runs

All six primary runs completed on 2026-10-16 (UTC) on `cenpesor`, physical
GPU 0. Runs executed sequentially, one at a time, each after the previous run
exited.

### Shared run environment (identical across all six runs)

| Item | Value |
| --- | --- |
| Device | NVIDIA H100 PCIe, physical index 0, UUID `GPU-7562e34d-b0a2-a725-2ad5-ed1327be3cb0`, PCI `0:27:00.0`, 81,559 MiB total |
| Free memory at each startup | 73,968 MiB (7,590 MiB used by two unrelated resident processes, 3,506 MiB each; left untouched) |
| CUDA-to-NVML mapping | `cuda_driver_pci_bus_id` (verified per run) |
| Code commit | `f73d885a5dad7f6d7294c48e3027d85d8dfd26b1`, working tree clean |
| Implementation digest (SHA-256) | `976b821acbaa5395066f186fb6a3f4242a7a057c0688850f628e4be15ecf9ed7` (all six runs) |
| Software | Python 3.10.12, PyTorch 2.3.1+cu121, CUDA runtime 12.1, driver 550.54.14 |
| Recipe | BF16 autocast, microbatch 1, 20 warm-up + 100 timed steps, seed `20261017`, automatic SDPA backend (flash observed in all six runs), tied weights, no activation checkpointing |
| Synthetic optimizer | AdamW, lr `1e-4`, betas `(0.9, 0.95)`, weight decay `0.1`, epsilon `1e-8` — exercises a forward/backward/optimizer step only; it does not freeze the training recipe |
| Memory policy | 40,960 MiB hard budget, 36,864 MiB initial peak target, 4,096 MiB safety margin, 35,840 MiB allocator cap, 1,024 MiB non-allocator headroom; no violation in any run |
| Tokenizer identity | `unigram-v24576`, model SHA-256 `c8982a23d3c5ecc0a61d300edc1e354e38f1358aa3db5420d5bf19f7b95cbf23` (recorded as identity only; no tokenizer or corpus artifact was opened) |
| Synthetic input IDs | One CPU-generator stream per context, identical across the three architectures: SHA-256 `3419cc8825263e41e49353c08d1b9507916c4d32925d563bf3ba1fe6859064d8` at 2,048; `429a19f7b7dc22430c89c9b0d14a39b5920a5d2129d40e36ce3e466b3432ad19` at 4,096 |
| Timed tokens per run | 204,800 at 2,048; 409,600 at 4,096 |

### Throughput and latency

Wall-clock mean step time includes host checks and per-step NVML sampling;
GPU-event statistics bracket the step on the device.

| Architecture | Context | Training tokens/s | Wall mean step (ms) | Event mean (ms) | Event median (ms) | Event P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A — narrow/deep | 2,048 | 22,229.5 | 92.13 | 92.12 | 91.99 | 97.28 |
| A — narrow/deep | 4,096 | 44,899.8 | 91.23 | 91.22 | 91.03 | 95.96 |
| B — balanced | 2,048 | 26,429.2 | 77.49 | 77.49 | 77.38 | 82.19 |
| B — balanced | 4,096 | 53,954.6 | 75.92 | 75.91 | 75.50 | 79.42 |
| **C — wide/shallow** | 2,048 | **30,622.8** | **66.88** | **66.87** | **64.87** | 78.63 |
| **C — wide/shallow** | 4,096 | **59,995.6** | **68.27** | **68.27** | **68.64** | **71.70** |

### Peak memory

| Architecture | Context | Peak CUDA allocated (MiB) | Peak CUDA reserved (MiB) | Peak process, NVML (MiB) |
| --- | ---: | ---: | ---: | ---: |
| A — narrow/deep | 2,048 | 4,869.5 | 5,004.0 | 5,674 |
| A — narrow/deep | 4,096 | 7,252.7 | 7,316.0 | 7,986 |
| B — balanced | 2,048 | 4,483.7 | 4,680.0 | 5,350 |
| B — balanced | 4,096 | 6,545.2 | 6,762.0 | 7,432 |
| **C — wide/shallow** | 2,048 | **4,274.6** | **4,304.0** | **4,974** |
| **C — wide/shallow** | 4,096 | **6,132.9** | **6,192.0** | **6,862** |

All peaks are far below the 36,864 MiB initial peak target and the 40,960 MiB
hard budget, so the matrix measured compute throughput, not a memory wall.
The 8,192-token capacity probe was not run.

### Run identities and artifact checksums

Each run directory contains `configuration.json`, `result.json`,
`manifest.json`, and `checksums.sha256`. On 2026-10-16, `sha256sum -c
checksums.sha256` passed in every run directory listed below.

| Architecture | Context | Run ID | Configuration SHA-256 | Result SHA-256 |
| --- | ---: | --- | --- | --- |
| A | 2,048 | `run-20261016T152717Z-seed20261017-36328bb8df31` | `badb9712bd0fe0ac5df06ba9082d34814634193f2d9da17d1fc9aefbfd4dcf3f` | `33f419cc3a2f31dc3a857b3d5c2de30bc81b683e6a9af3480b6ddb5b10b56b65` |
| A | 4,096 | `run-20261016T152736Z-seed20261017-d5f948020c83` | `c304bb794e4e8f4beebe082b6b2d7298c7c0557a6fda538c2127a9bada3a422a` | `dc314d219c95f1b6111b28590f894cd4c3084daa391c8fb0492e103298fabbbe` |
| B | 2,048 | `run-20261016T152755Z-seed20261017-b116994ba34d` | `45ac8f57aa52d590fc131cb9f8b4399d4e580dd73f7c07baf7f3585aef89ae49` | `829363c2c791319f43b595ac65a2e33b4c8d239a34d3747861d0752f809f4f93` |
| B | 4,096 | `run-20261016T152811Z-seed20261017-a9a21a7e6fe4` | `f70e1dcad21b555fa598a99a9e923f882b889b822de2605ad1f8748c72a5279e` | `e98420a59dcc068198a8caa52a068e9f87a8717d77810792d331158e39b099f7` |
| C | 2,048 | `run-20261016T152828Z-seed20261017-e912e2173bbe` | `4eeb65d97c68e5cea26ba8962693c971bcaf7b3b10845837711952e11af62099` | `22c6678f82043c127202d64e22a1410aee0a90cc7e5e3ea8328ea4037ed44a47` |
| C | 4,096 | `run-20261016T152843Z-seed20261017-251f6804926d` | `4c562fe87b5910f62765dc086f6ef16cb47f013a9bc71a9645429d3471e9ea7e` | `8de3db1aae489a1b8ad8108ba60a1ba65ee9e7a1c956088c4126f75bb970d175` |

Artifact root: `/mnt/data/cambacica-base-180m/c3-benchmarks/`, with layout
`architecture-{A|B|C}/context-{2048|4096}/<run-id>/`. Completed runs are never
overwritten; no benchmark artifact is stored in Git.

### Synthetic loss (not a quality measure)

Timed-window mean next-token cross-entropy on the synthetic random IDs:
A/2,048 `0.4173`, A/4,096 `0.6196`, B/2,048 `0.1261`, B/4,096 `0.2069`,
C/2,048 `0.0397`, C/4,096 `0.0704`. These values characterize numerical
stability of the synthetic step only. They **must not be interpreted as
evidence of linguistic quality**, and this record makes no cross-architecture
comparison of them.

### Recorded runs outside the primary matrix

- `run-20261015T175125Z-seed20261017-81992a291c01` (B, 2,048): **failed at
  preflight** with `GpuSafetyError: CUDA-to-NVML device mapping cannot be
  established reliably` under commit
  `469a5572d24fe61a2c7b95a38c14b5730b547d12`, the state before the CUDA
  Driver API PCI identity fallback (fix landed in `f73d885`). The fail-closed
  safety policy behaved as designed; the artifact is preserved.
- `run-20261016T152524Z-seed20261017-39ea4f9df0dd` (B, 2,048): complete smoke
  run with 1 warm-up + 2 timed steps, used to validate the repaired pipeline
  before the matrix; excluded from the primary matrix because it does not use
  the approved 20/100 step counts.

## Methodological limitations

1. **Synthetic inputs.** The input IDs are random generator output, not
   Portuguese text. Throughput and memory are valid measurements of the
   model/batch/kernel behavior, but no learning-quality claim follows from
   this matrix.
2. **Wall-clock overhead.** Wall-clock step time and tokens/second include
   host-side finiteness checks and per-step NVML sampling; GPU-event times
   exclude most of that overhead. Both are reported; the two are close in
   these runs.
3. **NVML sampling cadence.** NVML is sampled after each optimizer step, not
   continuously; transient allocations between samples are not captured.
4. **No hardware isolation.** The allocator cap plus NVML guard plus
   fail-closed checks are an operational policy, not a partition; unrelated
   processes are only read.
5. **One run per cell.** There is no within-cell variance estimate. Runs are
   sequential, so they share host state apart from small free-memory drift
   (72,760–72,836 MiB free at the first process sample).
6. **Tied mode only.** The relative ranking was measured with tied weights;
   the untied ranking is not measured by this matrix.
7. **Scope of the recipe.** Microbatch 1 and contexts 2,048/4,096 only; the
   8,192 capacity probe and larger effective batches were not run, and no
   claim extends to them.
8. **Backend selection was automatic.** Flash attention was observed in all
   six runs, but this decision does not pin the training-run SDPA backend.
9. **Benchmark optimizer is synthetic.** The AdamW settings exist only to
   exercise an optimizer step; the training optimizer and schedule remain
   open.
10. **Host-session specificity.** Absolute numbers depend on the 2026-10-16
    session on GPU 0, including two unrelated resident processes.

## Explicitly open decisions (not frozen by this record)

- **Final context length.** 2,048 vs 4,096 vs other; the 8,192 capacity probe
  was not run; C4 packing must account for byte-fallback structural-character
  tokens and the one-EOS-per-document cost in actual packed sequences before a
  context is chosen.
- **Weight tying.** Tied (176,180,224) versus untied (201,346,048) embeddings
  and LM head.
- **Final training recipe.** Precision, optimizer and learning-rate schedule,
  microbatch/effective batch, gradient accumulation, activation checkpointing,
  and the pretraining token budget.
- **Corpus mixture A/B/C** and source-rights/usage-policy resolution (gate
  C1); mixture B remains only the provisional tokenizer-sampling reference.
- **Quality pilot authorization and its separate token count `T_benchmark`**
  (C3.7).

## Preservation of Architectures A and B

Proposals A (narrow/deep) and B (balanced) are preserved as experimental
comparison configurations. Their definitions remain in
`src/cambacica/model/config.py`, their parameter accounting remains in
[`C3_ARCHITECTURE_PLANNING.md`](C3_ARCHITECTURE_PLANNING.md) and
[`C3_ARCHITECTURE_IMPLEMENTATION.md`](C3_ARCHITECTURE_IMPLEMENTATION.md), and
all benchmark artifacts — including the two recorded B/2,048 runs — remain
on NFS under `/mnt/data/cambacica-base-180m/c3-benchmarks/`. Nothing is
deleted or deprecated by this decision.

## Next C3 stage (C3.6) — training configuration and scalability validation

This stage is planned only; no implementation, GPU workload, pretraining, or
pilot training occurs as part of recording this decision.

1. **Freeze the Architecture-C training configuration** (scientist-approved,
   versioned in `configs/`):
   - context length, decided together with C4 packing accounting (structural
     characters, one EOS per document);
   - weight tying policy;
   - precision (BF16 autocast is the measured candidate), optimizer,
     learning-rate schedule, and warmup;
   - microbatch, gradient accumulation, and effective batch;
   - activation checkpointing policy;
   - pretraining token budget, set after the scalability numbers and the
     corpus mixture are known.
2. **Scalability validation on GPU 0** (measurement, no quality claims):
   steady-state memory and throughput at the planned context and effective
   batch with the full training state (BF16 parameters and gradients, FP32
   optimizer moments, activations); verify the configuration fits the 40,960
   MiB budget with the 4,096 MiB safety margin; verify checkpoint
   save/resume and deterministic restart; record tokens/second and the
   wall-clock implication for the token budget. This stage may re-measure
   only under a new scientist-authorized run; it does not rerun C3.4.
3. **Stop line.** C3.6 produces a frozen training recipe plus measured
   scalability evidence. It does not start pretraining and does not authorize
   the quality pilot.

### C3.7 quality pilot gate

A controlled equal-token learning pilot on Architecture C will be proposed
only when both gates hold:

- **Corpus gate (C1):** the final mixture A/B/C is selected and the
  source-rights/usage-policy items are resolved, so an approved training view
  and the fixed C1-S1 validation reference are available;
- **Scientist approval:** a separate authorization that fixes the pilot token
  count `T_benchmark`, the seed, context, optimizer, and evaluation
  references.

The pilot would compare validation loss versus tokens seen on real data; it
is not pretraining, its token count does not set the final pretraining budget,
and its results remain a C3.7 decision input, not a C7 evaluation.

## Provenance

- Decision author: the scientist, recorded 2026-10-16.
- Benchmark execution: 2026-10-16 on `cenpesor` GPU 0, commit
  `f73d885a5dad7f6d7294c48e3027d85d8dfd26b1` (clean tree), implementation
  digest `976b821acbaa5395066f186fb6a3f4242a7a057c0688850f628e4be15ecf9ed7`.
- Methodology and safety policy: [`C3_BENCHMARK_METHODOLOGY.md`](C3_BENCHMARK_METHODOLOGY.md);
  manual runbook: [`C3_BENCHMARK_RUNBOOK.md`](C3_BENCHMARK_RUNBOOK.md);
  planning and accounting: [`C3_ARCHITECTURE_PLANNING.md`](C3_ARCHITECTURE_PLANNING.md);
  implementation: [`C3_ARCHITECTURE_IMPLEMENTATION.md`](C3_ARCHITECTURE_IMPLEMENTATION.md);
  tokenizer contract: [`C2_TOKENIZER_SELECTION_V1.md`](C2_TOKENIZER_SELECTION_V1.md).
