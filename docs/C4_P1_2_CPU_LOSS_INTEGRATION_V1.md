# C4-P1.2 — Bounded synthetic CPU integration of packing and Transformer loss

**Date:** 2026-10-19.

**Status: IMPLEMENTED / CPU EXECUTION AND SCIENTIFIC REVIEW PENDING.**
The scientist explicitly authorized implementation and CPU-only tests of
C4-P1.2 after approving the initial [C4-P1.1 BOS/EOS/loss policy](C4_P1_1_SEQUENCE_LOSS_CONTRACT_V1.md).
This stage does not authorize any real corpus operation, GPU workload,
optimizer update, training or other C4-D/C4-T execution.

## Purpose

Verify numerically that **actual C3 `CambacicaCausalLM.forward(input_ids,
labels)`**, when driven by pre-shifted artificial P1 next-token labels,
agrees with:

1. `torch.nn.functional.cross_entropy(..., ignore_index=-100)`
   explicitly configured; and
2. an independent log-softmax/gather/mean reduction over *only*
   the valid non-`-100` labels.

The C3 Transformer code is **unchanged**; its existing `forward` uses the
PyTorch default `ignore_index=-100`. The independent comparison checks
this on CPU but does not substitute for future production code using
an explicitly pinned loss contract.

## Implementation

- `src/cambacica/packing/p1_cpu_loss_probe.py`: CPU-only,
  bounded numerical probe, imported explicitly; does **not** affect
  imports of the pure-Python P1 reference or make PyTorch required
  for C1/C2/P1 tests.
- `tests/packing/test_p1_cpu_loss_integration.py`: artificial
  token IDs, artificial source/split/provenance, a **tiny**
  `ArchitectureSpec` instantiated with the **actual** C3 model class.
  No production tokenizer or production documents.
- The CPU probe accepts **at most 32 modeled positions**, rejects GPU
  or meta model parameters, zero or all-ignored valid targets, invalid
  input/label IDs, malformed padding/masks and non-finite or
  inconsistent numerical loss. It returns a scored-target count,
  ignored-target count, the three losses and an NLL sum.
  It uses `torch.inference_mode()` and performs no backward or optimizer step.
- The full **4096-position** window geometry is **not reduced or
  modified**: it remains exercised separately through the P1 reference.
  Only a **short valid prefix** of a 4096-position artificial window
  is forwarded through a tiny CPU model, optionally followed by a
  few ignored padding slots. This is intentionally a bounded numerical
  integration check, not a measured full-length training path.
- Tests cover EOS and cross-document next-token predictions, C3
  `next_token_loss` equivalence (no double-shifting), ignored padding
  stability, train/validation/test isolation, causal prefix invariance,
  4096/4097 boundary accounting, non-scored streams, zero-target guards,
  invalid IDs and resource bounds. They also check that per-window
  losses with different valid-token counts are combined by a
  **token-weighted** mean: total negative log likelihood divided by
  total valid scored targets, not a naive arithmetic average of
  per-window means.

## Explicit remaining limitations

- CPU tests have **not yet been executed in the Orion environment**;
  do not declare C4-P1.2 PASS based solely on the implementation commit.
  PyTorch is optional in a plain repo environment; running the focused
  tests requires an environment with CPU-capable PyTorch installed.
- This test does not demonstrate actual v24576 tokenizer behavior,
  production corpus packing, production I/O, FP32/BF16 mixed-precision
  training or large-context (4096) numerical loss performance.
- A production integration must make `ignore_index=-100` explicit
  rather than relying on the C3 model's implicit default. It must
  also skip/reject all-ignored batches and aggregate losses correctly
  across variable valid target counts. A production zero-loss batch
  policy has not been approved.
- No change has been made to frozen C3 model shape, frozen C2 tokenizer,
  C1 source selection, source rights, mixture A/B/C, or approved GPU
  allocation policy.

## Authorized CPU-only verification commands (Orion)

First fast-forward the clean checkout. Then run **with CUDA hidden**
(the commands import PyTorch but never initialize a CUDA device):

```bash
cd ~/dev/cambacica-base-180m
git pull --ff-only
test -z "$(git status --porcelain)"
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src \
  /srv/cambacica-c3/venv/bin/python -m pytest -q \
  tests/packing/test_p1_cpu_loss_integration.py \
  tests/packing/test_p1_reference.py \
  tests/model/test_transformer.py
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src \
  /srv/cambacica-c3/venv/bin/python -m pytest -q
```

**Mandatory:** if PyTorch tests skip rather than execute, the numerical
integration is NOT verified. Capture pass/fail/skip counts and traceback
of any failure without changing hardware or unapproved scientific policy.
No CUDA, NVML, real corpus paths, tokenizer materialization, training,
GPU0/GPU1 operations or published checkpoint changes.

## Gate statement

**C4 remains IN PROGRESS.** C4-P1.1 scientific policy stays approved.
C4-P1.2 is **IMPLEMENTED / VERIFICATION PENDING** until the CPU results are
observed and reviewed. C1 rights and mix remain unresolved, and C3.7,
C5, C6, production C4 and all GPU experiments remain blocked.
