# C1-BD2.6 Production Recovery Readiness

**Decision:** C1-BD2.6 recovery validation **COMPLETE**. **ENGINEERING READY / SCIENTIFIC APPROVAL PENDING** for one read-only BD3 run after the remaining scientific decisions and the Orion scratch preflight are complete. C1-BD3 was **NOT RUN / NOT APPROVED**.

The work adds persistent restart state, exact resume checks, global document-frequency recovery, and restart-safe output publication. It does not freeze the matcher policy or authorize a production scan.

## 1. Pinned inputs and policy

- D1 exact corpus manifest: `57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428` — 21,603,689 records, 21,470,091,017 normalized words, 532 Parquet files.
- BD2 snapshot manifest: `d9b5bfdec543af37fdb8733239b0f8b0255339454b5ebca8880371bb86f77cda` — CALAME-PT `353671bc95cc3d94d488201f67d41b640eb80c55`; Belebele `d4c91dedc9de484dbea7b7d940f898f59fd135e9`, `por_Latn/test`.
- BD2 calibration manifest: `904fd5dbe8175b7b2c216af69e4a592a5cca20e9314ac21c694b10471e2ea28e`.
- The candidate matcher policy remains `c1-bd2-candidate-50-or-80-2-v1`, SHA-256 `602f9d851cdc038b433ee252011cdddf48834f3ebd1ca1d625396baab7128267`, with `freeze_for_bd3=false`.

The scanner preserves NFC/casefold Unicode tokenization and source offsets, exact complete-item/context/passage/question/question-plus-answer evidence, rare 13-token anchors, provisional contiguous and coverage thresholds, coherent monotone alignment, and review-only candidate classifications.

## 2. Recovery architecture and checkpoint identity

Each production run owns a persistent SQLite database under the operator-selected local scratch directory. Checkpoint schema version 2 stores document identities and source positions, document text hashes for evidence-bearing rows, exact evidence, pending and retained anchor evidence, exact distinct-document anchor frequencies, final hits, per-run counters, and a checkpoint state row. Production `seen_docs` is keyed by ordered shard index and row ordinal, with a unique document-ID safeguard. No corpus text is copied into the checkpoint.

The checkpoint identity includes the D1 manifest SHA-256; the complete, ordered shard inventory with row counts, byte counts, and manifest checksums; the snapshot and calibration manifest SHA-256 values; the complete policy and its SHA-256; matcher and normalization implementation versions; checkpoint schema version; run identifier; output directory; input root; and the approval metadata supplied by the scientist. Resume recomputes this identity and rejects any mismatch. It also runs SQLite `integrity_check`, compares committed row counts and source positions, and checks that the stored inventory agrees with the committed rows.

The human-readable `run.json` remains marked `INCOMPLETE` until publication. It does not authorize row skipping. The SQLite state and its matching evidence are authoritative; manually added progress fields in a sidecar do not change the database resume position.

On resume, the scanner prints the committed record count and the exact next shard index and row ordinal. It verifies the byte checksums of every shard containing committed rows before continuing. It skips completed shards and resumes the current shard at that ordinal, skipping complete row groups where possible. Each new record still passes the normalized-text SHA-256 check, stable occurrence-ID construction, and expected row-count checks.

## 3. Transaction and accounting guarantees

The matching evidence for a record, anchor-frequency updates, duplicate-ID marker, document accounting fields, word counters, and next source position are written in the same SQLite transaction. The default checkpoint interval remains 256 records. A crash rolls back the active batch and its progress together; the maximum replay is 255 records. A clean interrupt closes the connection and rolls back any uncommitted batch. A hard process kill leaves the WAL for SQLite recovery on the next open.

Persistent databases use SQLite WAL mode, `synchronous=FULL`, `wal_autocheckpoint=1000`, file-backed temporary storage, and a 16 MiB page cache. The committed WAL record is synchronized before SQLite reports a successful transaction. The database and WAL must live on local scratch with reliable `fsync`; do not put the live SQLite checkpoint on NFS. The final output stage is created beside its destination so directory publication uses one-filesystem atomic rename.

The SHA-256 record-ID digest is rebuilt from persisted document identities in source-row order after restart. It does not depend on serializing Python's in-memory hash accumulator. Duplicate document IDs and duplicate or out-of-order shard positions fail closed. Final per-shard row totals and digests retain the previous accounting contract.

## 4. Corpus-wide anchor frequencies

Anchor frequencies remain global across every shard and every restart. A document increments each matching anchor once, even when it repeats the text many times. Frequencies continue to increase after an anchor becomes common, so the persisted value is the true distinct-document frequency rather than a saturated `ceiling + 1` value.

When a frequency crosses the configured ceiling, the same transaction marks the anchor frequent and deletes all earlier retained approximate evidence for it. Evidence from that document is not retained, and later documents continue updating its count without recreating evidence. The exact-evidence table is independent of this filter. Candidate finalization begins only after the full pinned inventory is accounted for.

The coherent monotone alignment safeguard from BD2.5 is unchanged. The frequency-accounting correction changes internal counts above the ceiling; it does not change candidate thresholds or evidence eligibility.

## 5. Finalization and publication

Checkpoint status advances through `SCANNING`, `READY_TO_FINALIZE`, `FINALIZING`, and `FINALIZED`. Final candidate construction runs in one SQLite transaction; a crash during it rolls back partial final hits and the finalization state, so resume can repeat it.

The scanner writes Parquet outputs and a final manifest into a sibling directory carrying an `INCOMPLETE` marker. It verifies schemas, row counts, artifact checksums, and a separate `manifest.sha256` before removing the marker. It then atomically renames the stage to the final output path and marks the checkpoint `PUBLISHED`. A crash after the marker is removed but before the rename leaves a verifiable prepared stage that resume can publish. A crash after the rename is detected by output verification and does not create a second output. Existing output directories are never overwritten.

The published scan manifest remains `BD3_SCAN_COMPLETE_PENDING_INDEPENDENT_VERIFICATION`; `verify-scan --verify-inputs` independently recomputes the record-ID digests. Candidate hits remain review evidence and do not create exclusions.

## 6. Recovery test matrix

The focused matcher and recovery suites pass **45 tests**. The new synthetic recovery tests use three Parquet shards with `text`, `source`, and `content_sha256` fields and realistic passage and question benchmark fields.

| Scenario | Result |
| --- | --- |
| Uninterrupted run versus SIGKILL and resume | Candidate hits, anchor evidence, per-shard accounting, exact anchor frequencies, and all three Parquet artifact SHA-256 values match. |
| SIGKILL before the first commit | Progress remains at row 0; the two uncommitted records are replayed. |
| SIGKILL immediately after a commit, mid-shard | Resume starts at the committed next row; no committed record is repeated. |
| SIGKILL exactly between shards | Resume starts at the next shard, row 0. |
| Repeated interruptions and multiple resumes | Final accounting includes each input row exactly once. |
| Failure after partial final-hit insertion | The finalization transaction rolls back; resume reconstructs the same final hits. |
| Anchor crosses the ceiling after resume | Its global frequency reaches 5, all evidence for that anchor is removed, and four exact passage hits remain. |
| Corpus, snapshot, policy, matcher-version, or shard-order mismatch | Resume is rejected. |
| Missing or truncated SQLite checkpoint | Resume is rejected. |
| Edited sidecar progress | It cannot move the database resume position. |
| Already-finalized checkpoint | Reopening and finalizing again returns the same rows without duplicates. |
| Duplicate and out-of-order records | Both fail closed. |
| Interrupted output rename | A complete unmarked stage is verified and atomically published on retry; an existing final directory is refused. |
| Artifact corruption | A changed Parquet file or manifest checksum is rejected. |
| Resume inside a row group larger than the read batch | Row ordinals and occurrence IDs continue without repeating or skipping rows across batches. |

At least one restart test uses `SIGKILL` and verifies SQLite WAL recovery. A separate controlled finalization fault exercises rollback inside the finalization transaction. The baseline/resume comparison checks full sorted records and byte-identical Parquet artifacts in the test environment, not only aggregate hit counts.

## 7. Bounded performance results

The resumable profile used the existing stratified sample and matched BD2.5's deterministic sample identity `099c8ac5e29fc07e79d23a133e0a9219c4854587d8631ecacf3cd2474e024b52` exactly. It processed 13,927 documents and 12,055,769 normalized words, with zero real-sample hits and zero real-sample anchor-evidence rows.

| Measure | BD2.5 non-resumable reference | BD2.6 persistent checkpoint profile |
| --- | ---: | ---: |
| Profile wall time | 52.587 s | 48.340 s |
| Sample processing loop | 50.636 s | 45.899 s |
| End-to-end profile CPU | 52.424 s | 48.232 s |
| End-to-end throughput | 229,252 words/s | 249,393 words/s |
| Peak process RSS | 712,757,248 bytes | 705,843,200 bytes |
| SQLite database plus WAL/SHM at finalization | 1,146,880 bytes | 7,667,656 bytes |
| SQLite checkpoint commits | 54 | 54 |
| Time inside checkpoint commit calls | 0.158 s | 0.085 s |

These are single cached runs, not an interleaved benchmark. The wall-time difference does not establish a speedup. The measured durable commit calls used about 0.19% of the BD2.6 sample-loop time. The larger SQLite footprint includes the expanded recovery/accounting state and WAL/SHM; linear extrapolation of this no-hit profile is about **11.9 GB** for 21.6 million records. Treat that as a conservative planning projection, not a production measurement. The profile wrote no persistent artifacts under `/mnt/data` because this host mounts it read-only.

The completed-checkpoint reopen took 0.0085 s for 13,927 records. A separate persistent match-heavy restart used 19 synthetic wrappers and a 16-record checkpoint interval: 4,484 normalized words, 76 candidate rows, 8,360 anchor-evidence rows, 3 records replayed, 0.0056 s to reopen, 0.1127 s finalization, and 5.47 MB SQLite plus WAL/SHM after finalization. Replay was bounded by the 15 records after the preceding checkpoint. This clean connection restart complements the SIGKILL integration tests; it is not a full-scale restart-time estimate.

The persistent match-heavy run retained substantial evidence through restart. The existing 64-document stress run also completed with 256 candidate rows and 28,160 anchor-evidence rows. These synthetic rates do not predict corpus candidate density. The real bounded sample had no hits, so it does not establish false-positive rate, recall, or evidence-heavy production storage.

## 8. Scratch capacity and host confirmation

The current environment is `cenpesor`, Linux `5.15.0-191-generic`, with `/tmp` on local ext4/NVMe and 581,800,296,448 bytes free at the final profile start. The pinned input mount is NFS and read-only here. No scheduler identity was present, and `nvidia-smi` could not access a driver. This does not establish that the host is Orion; no Orion profile is claimed.

For an initial Orion BD3 run, require **at least 128 GiB free local scratch** before launch, with the SQLite database, WAL, and SHM under a unique run directory. This headroom covers the roughly 11.9 GB no-hit planning projection and adds room above BD2.5's illustrative 1%-hit / four-hit-rows-per-document scenario, which projected about 46 GB scratch before the BD2.6 accounting overhead. Candidate density is unknown and could require hundreds of GB. Keep additional free space on the NFS output filesystem for candidate Parquet files and staging. If SQLite or output writing hits a storage error, the last transaction remains the resume point; do not delete the checkpoint or incomplete stage.

The committed-prefix checksum pass reads the complete compressed bytes of each shard containing committed records. Its Orion NFS cost was not measured here. The operational command below checks filesystem type and free capacity rather than assuming a path such as `/local_nvme`.

## 9. Orion profile and production runbook

The profile and full scan were not run on Orion. On Orion, first set `SCRATCH_PARENT` to a site-confirmed local path, inspect it with `findmnt` and `df`, and require at least 128 GiB free. The 12-million-word profile is bounded at 30 minutes and writes its log and outputs under the new `bd2-recovery` artifact root.

```bash
REPO_ROOT=~/dev/cambacica-base-180m
test -d "$REPO_ROOT/src/cambacica"
SCRATCH_PARENT=/path/to/confirmed/local/scratch  # replace after checking the host
RECOVERY_ROOT=/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2-recovery
findmnt -T "$SCRATCH_PARENT" -o TARGET,SOURCE,FSTYPE,OPTIONS
case "$(findmnt -n -T "$SCRATCH_PARENT" -o FSTYPE)" in nfs*) echo "scratch must be local"; exit 1;; esac
AVAILABLE_BYTES="$(df -B1 --output=avail "$SCRATCH_PARENT" | tail -1 | tr -d ' ')"
test "$AVAILABLE_BYTES" -ge 137438953472 || { echo "need at least 128 GiB free"; exit 1; }
PROFILE_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(hostname -s)"
PROFILE_SCRATCH="$SCRATCH_PARENT/cambacica-bd26-profile-$PROFILE_ID"
PROFILE_OUTPUT="$RECOVERY_ROOT/profile-$PROFILE_ID"
PROFILE_LOG="$RECOVERY_ROOT/logs/profile-$PROFILE_ID.log"
mkdir -p "$RECOVERY_ROOT/logs"
test ! -e "$PROFILE_SCRATCH" && test ! -e "$PROFILE_OUTPUT"
export REPO_ROOT PROFILE_ID PROFILE_SCRATCH PROFILE_OUTPUT PROFILE_LOG
tmux new-session -c "$REPO_ROOT" -s "bd26-profile-$PROFILE_ID"
```

Run this in the attached tmux session:

```bash
set -o pipefail
timeout --signal=INT --kill-after=10s 1800s env PYTHONPATH=src python3 -X faulthandler \
  scripts/profile_decontamination_sample.py \
  --input-root /mnt/data/cambacica-base-180m/deduplicated/exact \
  --snapshot-dir /mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2/snapshot \
  --output-dir "$PROFILE_OUTPUT" \
  --scratch-dir "$PROFILE_SCRATCH" \
  --target-words 12000000 --max-seconds 1740 --stress-documents 64 \
  --persistent-checkpoint 2>&1 | tee "$PROFILE_LOG"
```

For the later BD3 execution, use a policy file only after scientists have approved its thresholds, provenance caveats, reviewer process, approver name, and decision reference. The path below is intentionally an operator-supplied value; the command will fail closed until it names a policy carrying `SCIENTIST_APPROVED_FOR_BD3` and complete approval metadata.

```bash
REPO_ROOT=~/dev/cambacica-base-180m
test -d "$REPO_ROOT/src/cambacica"
SCRATCH_PARENT=/path/to/confirmed/local/scratch  # use the site-verified path from preflight
RECOVERY_ROOT=/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2-recovery
findmnt -T "$SCRATCH_PARENT" -o TARGET,SOURCE,FSTYPE,OPTIONS
case "$(findmnt -n -T "$SCRATCH_PARENT" -o FSTYPE)" in nfs*) echo "scratch must be local"; exit 1;; esac
AVAILABLE_BYTES="$(df -B1 --output=avail "$SCRATCH_PARENT" | tail -1 | tr -d ' ')"
test "$AVAILABLE_BYTES" -ge 137438953472 || { echo "need at least 128 GiB free"; exit 1; }
POLICY_FILE=/path/to/scientist-approved-policy.json
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(hostname -s)"
CHECKPOINT_DIR="$SCRATCH_PARENT/cambacica-bd3-$RUN_ID"
OUTPUT_DIR="/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd3/scan-$RUN_ID"
LOG_DIR="$RECOVERY_ROOT/logs"
LOG_FILE="$LOG_DIR/bd3-$RUN_ID-attempt-1.log"
SESSION="bd3-$RUN_ID"
mkdir -p "$LOG_DIR"
test ! -e "$CHECKPOINT_DIR" && test ! -e "$OUTPUT_DIR"
export REPO_ROOT SCRATCH_PARENT RECOVERY_ROOT POLICY_FILE RUN_ID CHECKPOINT_DIR OUTPUT_DIR LOG_DIR LOG_FILE SESSION
tmux new-session -c "$REPO_ROOT" -s "$SESSION"
```

Run the following in the attached tmux session. The 60-hour timeout sends `SIGINT`; a single Ctrl-C in tmux is the safe early stop. Wait for the Python process to exit before resuming.

```bash
set -o pipefail
/usr/bin/time -v timeout --signal=INT --kill-after=60s 216000s \
  env PYTHONPATH=src python3 -X faulthandler -m cambacica.corpus.decontamination scan \
  --snapshot-dir /mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2/snapshot \
  --calibration-dir /mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2/calibration \
  --input-root /mnt/data/cambacica-base-180m/deduplicated/exact \
  --output-dir "$OUTPUT_DIR" --checkpoint-dir "$CHECKPOINT_DIR" \
  --run-id "$RUN_ID" --policy "$POLICY_FILE" --execute-bd3 \
  2>&1 | tee -a "$LOG_FILE"
```

If the job exits before publication, keep the checkpoint, output path, run ID, and policy unchanged. In the new login shell, replace the placeholders below with the exact values from the first attempt; do not generate a new run ID or checkpoint path. Check the scratch mount and current free space again before resuming.

```bash
SCRATCH_PARENT=/path/to/the/same/confirmed/local/scratch
RECOVERY_ROOT=/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2-recovery
POLICY_FILE=/path/to/the/same/scientist-approved-policy.json
RUN_ID=the-exact-run-id-from-the-first-attempt
CHECKPOINT_DIR="$SCRATCH_PARENT/cambacica-bd3-$RUN_ID"
OUTPUT_DIR="/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd3/scan-$RUN_ID"
LOG_DIR="$RECOVERY_ROOT/logs"
test -f "$CHECKPOINT_DIR/run.json" && test -f "$CHECKPOINT_DIR/matcher.sqlite3"
findmnt -T "$SCRATCH_PARENT" -o TARGET,SOURCE,FSTYPE,OPTIONS
case "$(findmnt -n -T "$SCRATCH_PARENT" -o FSTYPE)" in nfs*) echo "scratch must be local"; exit 1;; esac
df -h "$SCRATCH_PARENT"
RESUME_LOG="$LOG_DIR/bd3-$RUN_ID-resume-$(date -u +%Y%m%dT%H%M%SZ).log"
REPO_ROOT=~/dev/cambacica-base-180m
test -d "$REPO_ROOT/src/cambacica"
mkdir -p "$LOG_DIR"
export REPO_ROOT SCRATCH_PARENT RECOVERY_ROOT POLICY_FILE RUN_ID CHECKPOINT_DIR OUTPUT_DIR LOG_DIR RESUME_LOG
SESSION="bd3-$RUN_ID-resume"
export SESSION
tmux new-session -c "$REPO_ROOT" -s "$SESSION"
set -o pipefail
/usr/bin/time -v timeout --signal=INT --kill-after=60s 216000s \
  env PYTHONPATH=src python3 -X faulthandler -m cambacica.corpus.decontamination scan \
  --snapshot-dir /mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2/snapshot \
  --calibration-dir /mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2/calibration \
  --input-root /mnt/data/cambacica-base-180m/deduplicated/exact \
  --output-dir "$OUTPUT_DIR" --checkpoint-dir "$CHECKPOINT_DIR" \
  --run-id "$RUN_ID" --policy "$POLICY_FILE" --execute-bd3 --resume \
  2>&1 | tee -a "$RESUME_LOG"
```

After publication, independently verify outputs and every per-shard record-ID digest:

```bash
PYTHONPATH=src python3 -X faulthandler -m cambacica.corpus.decontamination verify-scan \
  --scan-dir "$OUTPUT_DIR" \
  --input-root /mnt/data/cambacica-base-180m/deduplicated/exact \
  --verify-inputs
sha256sum "$OUTPUT_DIR/manifest.json" "$OUTPUT_DIR/manifest.sha256"
```

## 10. Scientific limits and remaining approvals

The BD2.5 review packet remains the only calibration panel. Its 40 controls are synthetic constructions, not independently sourced real positives; all scientist-review labels remain blank. The 12-million-word real sample had zero candidates. The provisional alignment drift and matching thresholds have not been independently reviewed, and corpus-wide false-positive rate and recall remain unknown. True paraphrases and heavily edited copies may escape lexical matching. A candidate is evidence for human review, not proof of memorization or grounds for automatic exclusion.

CALAME retains 2,076 snapshot rows; row 718 has a whitespace-only target and the proposed 2,075 metric denominator still needs a scientist decision. CALAME generated-source overlap and Belebele upstream passage rights remain provenance caveats. Scientists must still approve the benchmark caveats, exact and approximate matching rules, alignment safeguard, thresholds and global DF ceiling, long-document review action, reviewer process, and a versioned policy with named approver and decision reference. No approval identity or reference is recorded here.

The existing candidate policy hash `602f9d851cdc038b433ee252011cdddf48834f3ebd1ca1d625396baab7128267` is a concrete review target. It remains unfrozen and is not marked `SCIENTIST_APPROVED_FOR_BD3`.

## 11. Gate status and recommendation

- D1: **COMPLETE / PASS**; D2–D2d: **COMPLETE**.
- Near-deduplication: **APPROVED / CLOSED**, zero removals.
- C1-BD1: **APPROVED**; C1-BD2: **COMPLETE**; C1-BD2.5: **PREFLIGHT COMPLETE**.
- C1-BD2.6: **COMPLETE** for the scoped recovery implementation and bounded validation.
- Engineering: **READY** for one read-only scan once the Orion scratch preflight passes.
- Scientific approval: **PENDING**; C1-BD3: **NOT RUN / NOT APPROVED**; C1-BD4: **NOT RUN**.
- C1: **IN PROGRESS**; C2: **PENDING**.

The production command remains dry-run by default and requires `--execute-bd3`, a scientist-approved policy, named approver, decision reference, and verified pinned inputs for both a fresh run and resume. No full-corpus scan, candidate generation, exclusion manifest, corpus change, mix change, split, model training, or policy freeze occurred.
