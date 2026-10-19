# C3.6 — Synthetic scalability and deterministic restart: scientific results v1

**Decision recorded 2026-10-19: C3.6 COMPLETE / SCIENTIST APPROVED.**
The scientist reviewed the nine-cell GPU scalability matrix and the
fresh-process deterministic checkpoint/restart validation and authorized
publication of the final scientific result. These are **engineering,
synthetic-token measurements**, not evidence of Portuguese-language quality
or a complete pretraining recipe.

## Scope and provenance

- Suite: `c36-matrix-20261016-gpu0-v1`.
- Matrix and restart execution: 2026-10-17 (Orion local execution timestamps).
- Execution code commit: `9233cd506acafccd5cc491eb6f34adfe7febde5e`,
  clean tree; implementation SHA-256
  `b4a570de2d73373275e4a567ec18514e1b5a150a67141a2ddd58982921e07d4a`.
- Approved baseline: `c3.6-architecture-c-v1`, JSON SHA-256
  `85840c513d8e0ac57804fe41b681d4b78cbcff0417bcedb177580c64bddae5dd`.
- Architecture C: 9 layers, hidden 1024, 16 attention heads, SwiGLU width
  4096, context 4096, tied embedding/output, 176,180,224 unique parameters.
- Frozen C2 tokenizer: SentencePiece Unigram, vocab 24576; model SHA-256
  `c8982a23d3c5ecc0a61d300edc1e354e38f1358aa3db5420d5bf19f7b95cbf23`.
  No production tokenizer/corpus files were opened in these GPU tests.
- Execution host: Orion `cenpesor`, physical GPU 0 H100 PCIe, UUID
  `GPU-7562e34d-b0a2-a725-2ad5-ed1327be3cb0`, PCI `0:27:00.0`.
  CUDA Driver PCI-bus / NVML mapping verified; only one CUDA device visible.
- PyTorch 2.3.1+cu121; driver 550.54.14; BF16 autocast and observed Flash
  SDPA; FP32 parameters, gradients and AdamW moments.
- Synthetic input seed: `20261017`, fixed full-stream SHA-256
  `a09e65ff773b3f426e47f7fbd3373d19838b01e592465f9db63fd976a01ca210`.
- Optimizer: **C3.4 synthetic AdamW fixture only**, learning rate 1e-4,
  betas (0.9, 0.95), weight decay 0.1, epsilon 1e-8.
  This fixture is **not** a scientist-approved training optimizer/recipe.

## C3.6-M: nine-cell matrix — PASS

Each cell: five warm-up optimizer updates, 20 measured optimizer updates,
one repeat; cells run sequentially under the same GPU-0 lock. Throughput
is wall-clock tokens/s for measured updates **including per-update safety
monitoring**, not corpus tokenization/loading.

| Microbatch | Accumulation | Tokens/update | Wall tokens/s | GPU-event mean/update (ms) | Sampled process peak (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 4,096 | 61,525.1 | 66.57 | 6,862 |
| 1 | 2 | 8,192 | 65,644.6 | 124.78 | 7,802 |
| 1 | 4 | 16,384 | 68,247.4 | 240.05 | 7,802 |
| 2 | 1 | 8,192 | 92,459.4 | 88.59 | 10,760 |
| 2 | 2 | 16,384 | 98,276.5 | 166.70 | 11,528 |
| 2 | 4 | 32,768 | 101,202.9 | 323.77 | 11,528 |
| 4 | 1 | 16,384 | 105,095.3 | 155.88 | 18,204 |
| 4 | 2 | 32,768 | 109,037.2 | 300.51 | 19,740 |
| 4 | 4 | 65,536 | **110,673.9** | 592.14 | **19,740** |

All nine cells: `status=complete`, finite numerical results, observed Flash
attention, pinned Architecture C/4096/tied/BF16/GPU0, with per-cell
`configuration.json` and `result.json` SHA-256 verification **PASS**.
Total matrix: **225 optimizer updates**, **5,017,600 synthetic tokens**
(including warm-up), reported elapsed time **83.484 seconds**.
Minimum observed GPU-0 free memory: **54,222 MiB**, above the 4,096 MiB
minimum. The highest sampled Cambacica PID memory was **19,740 MiB**,
below the 36,864 MiB initial process peak target and 40,960 MiB hard budget.
Memory monitoring is sampled, so brief unobserved non-PyTorch spikes are
not ruled out.

The maximum measured throughput was ~79.9% above the 1×1 baseline.
At the same 16,384-token effective batch, 4×1 was faster than 2×2 or
1×4. 4×1 reached ~95.0% of maximum measured throughput, using
18,204 MiB versus 19,740 MiB at 4×4. These comparisons have one repeat
per cell and are not convergence or statistical generalization evidence.

**Provisional engineering candidates, NOT approved effective batches:**
4×1 (16,384 tokens/update, favorable throughput/batch tradeoff);
2×2 (16,384 tokens/update, lower observed memory);
4×4 (65,536 tokens/update, highest synthetic throughput).
Final effective batch requires additional scientific decisions and
evaluation on the approved training view.

## C3.6-R: independent-process deterministic checkpoint/restart — PASS

Protocol executed with fixed synthetic token IDs:

1. Reference process: four uninterrupted optimizer updates.
2. Fresh preparation process: two updates and checksummed checkpoint.
3. Fresh resume process: validate identity/checksum, restore full state and
   synthetic-stream position, then execute two further updates.

Comparison of 4 uninterrupted vs 2 + checkpoint + 2 resumed updates:

- `deterministic_restart_status=PASSED` and `status=complete`.
- **333/333 model/optimizer tensors identical bitwise**; mismatch count 0;
  maximum absolute and relative delta both **0.0**.
- Loss sequences identical bitwise; four losses:
  `[10.309303283691406, 10.318946838378906, 10.317648887634277, 10.31453800201416]`.
- Checkpoint `checkpoint.pt` and reference `reference_state.pt` SHA-256
  verification **PASS**.
- Strict deterministic PyTorch algorithms, cuDNN deterministic mode,
  TF32 matmul disabled and `CUBLAS_WORKSPACE_CONFIG=:4096:8`
  recorded in the pinned restart identity.
- Observed Flash SDPA throughout; no kernel incompatibility reported.
- Checkpoint SHA-256:
  `b9dfae53b4cc71b7556d52486ced0e9432ae918d55222d594f447c197a244431`.
- Reference snapshot SHA-256:
  `2584b6c72ab2b51f994bccb7b71d65ad224748a81ce8fce607acc90e4900491d`.

This verifies deterministic restart for the **tested synthetic fixture,
hardware, software and seed**. It does not guarantee bitwise restart for
future training recipes, longer runs or different kernels.

## Publication and artifact retention

The CPU-only `finalize-suite` command returned `status=published`,
`gpu_execution=false`, and:

`/mnt/data/cambacica-base-180m/c3-scalability/c3.6/c36-matrix-20261016-gpu0-v1/`

The implementation verifies suite artifacts before immutable publication.
The terminal publication response confirms completion; this record does
**not** claim a separately observed post-publication NFS checksum pass.
After publication, a read-only independent verification may be performed
using the existing `verify_suite_artifacts` utility and
`sha256sum -c checksums.sha256` in the destination. Active checkpoint
and reference model snapshots remain on local ext4 scratch, and are **not**
part of the published NFS report-only bundle.

## Gate decision and remaining boundaries

**C3.6 — COMPLETE / PASS / SCIENTIST APPROVED** for bounded synthetic
scalability, operational memory safety and deterministic restart. The
architecture/benchmark **C3 gate is COMPLETE** under the exit criteria
stated in `docs/GATES.md`: C3.4 completed, Architecture C selected
in C3.5, and the C3.6 matrix plus restart reviewed.

This decision does **not** approve a final effective batch, optimizer,
learning-rate schedule, training-token budget, packing/data pipeline,
activation-checkpointing policy, checkpoint cadence or training
state-storage policy. The 4×1 engineering candidate remains provisional.

**C3.7 real-corpus quality pilot remains BLOCKED** until the C1 rights/usage
policies and final corpus mixture/view are resolved and a **separate
scientist authorization** defines pilot tokens, optimizer and evaluation
conditions. C4 remains pending. C5 smoke training, C6 main pretraining
and C7 evaluation are not authorized by this result. GPU 1 and the Qwen
service remain out of scope.

Related contracts: [C3.6 baseline](C3_TRAINING_BASELINE_V1.md),
[C3.6 runner runbook](C3_6_SCALABILITY_RUNBOOK.md),
[C3.5 architecture decision](C3_ARCHITECTURE_SELECTION_V1.md),
[project gates](GATES.md).
