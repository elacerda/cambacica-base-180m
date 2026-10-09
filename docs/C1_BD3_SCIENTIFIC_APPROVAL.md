# C1-BD3 — Scientific authorization

Date: 2026-10-09
Approver identifier: eduardolacerda
Reference: C1-BD3-2026-10-09-scientific-approval

## Approved scope

One read-only full-corpus benchmark-contamination candidate scan
against the pinned CALAME-PT and Belebele Portuguese snapshots.

The approved detection policy is:
`configs/c1_bd3_policy_approved.json`.

Thresholds are fixed for this run, not established as universally
validated contamination thresholds. The coherent monotone-alignment
safeguard is retained.

CALAME retains 2,076 snapshot rows; 2,075 are evaluable under the
approved future metric convention.

Benchmark provenance, source-rights uncertainties, synthetic-only
calibration limitations and undetectable contamination types remain
documented limitations. This authorization does not establish
redistribution rights for third-party source material.

## Explicit restrictions

- No automatic training-record exclusions.
- No corpus rewriting or deletion.
- No changes to corpus mixes A/B/C.
- All candidate hits remain subject to scientific review in BD4.
- No claim of complete contamination detection.
- D1 corpus manifest identity remains unchanged.
- BD3 requires independent output verification after execution.
- C1 remains IN PROGRESS; C2 remains PENDING.
