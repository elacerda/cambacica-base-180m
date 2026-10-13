# C1-S1 — Split and candidate mixture readiness

2026-10-13. Starting HEAD: `9fb41455a3026930e1fd7e2a7541792084471f28`;
worktree clean. **Bounded validation PASS; split policy PROPOSED, awaiting
scientist approval. C1 IN PROGRESS; C2 PENDING.**

The eligible input is the immutable post-D1 corpus at
`/mnt/data/cambacica-base-180m/deduplicated/exact`: 21,603,689 records,
21,470,091,017 normalized words. Input manifest SHA-256:
`57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428`.
BD4 review manifest:
`9889c448fd67a3728bdd508d1eb41061413bb0b42f13a38569c33a6b9d944be3`.
Near deduplication and benchmark decontamination remain **CLOSED**, with zero
additional exclusions. No source text, D1 record, or BD3/BD4 decision changed.

## Evidence and policy reconciliation

**Verified** quantities below come from D1's checksummed
`accounting_by_source_subset.csv` (SHA-256
`e6ae45146002d722bdad4685d77bd96339e8a6b0a2224a643df611c107bc795b`).
Revisions agree with normalized manifests, sampled post-D1 records, and all
three candidate configurations. Complete pre-D1 metadata characterization is
existing evidence, not a new post-D1 census; its hashes are in `readiness.json`.
Words mean `len(text.split())` on stored normalized text, **not model tokens**.

| Family / verified revision | Post-D1 documents | Normalized words | Retained provenance and language evidence | Quality and rights policy |
| --- | ---: | ---: | --- | --- |
| Carolina v2.0.1, `55e63a519393c70a48dcfa14a558499c6bb0583b` | 2,108,727 | 1,296,640,886 | TEI/file/record IDs, seven taxonomies, headers; PT-BR label, not a measured language confidence. Original URLs absent throughout characterization. | Preserve extraction and inherited per-record license labels (including CC BY and CC BY-SA); licenses vary. Research permission and redistribution must be assessed per component. |
| Wikipedia `20231101.pt`, `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | 1,112,120 | 413,136,890 | Article IDs, titles, URLs; Portuguese, national variety unlabelled; editorial/manual translations possible. | Keep current extraction/title rule. Stored CC BY-SA 3.0/4.0 & GFDL label requires applicable attribution/license compliance; it is not blanket clearance for every use. |
| ParlamentoPT, `08f13e7e63ab9bfbd8c0b40955defe3bb7f68c2b` | 2,670,846 | 396,693,229 | Physical `train.txt` lines with line IDs; shared Parliament URL; PT-PT label. Historical and institutional material, including OCR/formula noise, rather than verified contemporary speaker turns. | Preserve all D1 records, including exact repetitions. Stored “Open Government Data” is a project label, not a complete license grant. Upstream card lists `other`; verify applicable terms for intended use. |
| Gutenberg acquisition `snapshot_2026-10-01`, 655 pinned eBooks | 655 | 21,054,903 | Book IDs and URLs; downloaded payload/manifests pin this local snapshot, no upstream Git revision. Portuguese historical/literary text; variety and native authorship not individually verified. | Keep existing wrapper extraction, whole works. US copyright status is jurisdiction-dependent; permissions for some copyrighted works do not transfer automatically. |
| GigaVerbo v2 residual `edu_high`, `7058ccf19eaeaf4505a96fc7e5305a01fc441fd8` | 15,711,341 | 19,342,565,109 | Subset IDs, upstream shard/row-group/commit, educational/toxicity metadata; `original_url` identifies upstream datasets in inspected records. Portuguese label, variety unknown. | Inherited eleven-subset eligibility and prior exclusion policy remain fixed. No new thresholds or classifier. License null throughout characterization; aggregator metadata does not clear upstream web/PDF rights. |

This reconciles overconfident historical descriptions in
[composition analysis](C1_COMPOSITION_ANALYSIS.md): “100% native,” exact national
variety percentages, uniformly clean parliamentary transcription, and “zero”
Gutenberg licensing uncertainty are not established facts. See the official
[Gutenberg license](https://www.gutenberg.org/policy/license.html) and
[ParlamentoPT card](https://huggingface.co/datasets/PORTULAN/parlamento-pt)
(current contextual checks; neither changes the pinned acquisition).

**Approved decisions:** keep the post-D1 population; no D2 removals; BD4 retains
all reviewed records; no further contamination experiment required. CALAME
keeps **2,076 snapshot rows and 2,075 evaluable rows**, excluding whitespace-only
target row **718** from the future metric denominator. This is already approved
in [BD3 scientific authorization](C1_BD3_SCIENTIFIC_APPROVAL.md), not an open
denominator choice. C7 evaluation code is outside this task.

**Provisional preparation policy:** retain existing source selection, language
labels and quality handling for local scientific preparation. No measured
classifier confidences exist; no claim of perfect Portuguese purity, absent
latent machine translation, or universal native authorship follows. Existing
GigaVerbo exclusions removed identifiable machine translation, noncommercial
components and primary-source overlaps during acquisition; S1 does not rerun
them. Tiny, empty or noisy retained records are assigned, not filtered.

**Unresolved rights:** no new license clearance was established. Before actual
local training, the responsible researcher/institution must confirm the local
research-use basis where existing authorization does not cover Carolina's
components, ParlamentoPT's terms, residual upstream content, and Gutenberg's
applicable jurisdiction. This is a usage determination, not a request for new
acquisition/filtering. There is no newly verified source prohibition here that
requires changing the authorized preparation population. Public corpus,
excerpt, benchmark redistribution and any model release need a separate rights
assessment; scientific approval alone does not resolve those rights. CALAME's
MIT and Belebele's CC BY-SA 4.0 declarations likewise do not clear all embedded
third-party content ([BD4 caveats](C1_BENCHMARK_DECONTAMINATION_BD4.md)).

## Proposed deterministic split contract

Policy: [`c1_split_policy_proposed.json`](../configs/c1_split_policy_proposed.json),
version `c1-s1-v1`, seed **20261013**; file SHA-256
`11ffdfd9521c472106cec8b7d8882770329ea2d8bf3080a817d90634dbd37444`.
The scientist must approve these exact policy bytes before a full manifest pass.

Propose **98% train / 1% validation / 1% test by unique group within each
source/subset**, not by documents or words. At this scale, 1% per held-out split
provides substantial coverage while preserving scarce curated training volume.
Gutenberg yields seven books per held-out split; Carolina `pub` has 26 documents
and minimum coverage matters more than exact percentages. These are practical
defaults, not optimized ratios; rare-domain estimates will have large uncertainty.

1. A document ID is the existing `occurrence-id-v2` hash of source, pinned
   post-D1 shard path **relative to `data/`**, and absolute row ordinal. It agrees
   with BD3/BD4 IDs, remains unambiguous when native IDs repeat, and requires
   original coordinates attached before dispatch/reordering. Batch counters,
   arrival order and a post-shuffle row number must never replace those coordinates.
2. Group keys use `(source, subset, original_id)` for Gutenberg books, Wikipedia
   articles, Carolina TEI records and GigaVerbo records. Native IDs are required;
   retained file/ordinal fallback IDs are usable as records, not evidence of an
   original work. Repeated IDs within a source/subset stay together. For
   ParlamentoPT use `(source, subset, content_sha256)` to keep identical preserved
   lines in one split, with every occurrence retained. This is assignment, not
   a new deduplication or removal operation.
3. Group and rank hashes use SHA-256 with separate versioned namespaces and
   canonical sorted-key UTF-8 JSON. Rank includes pinned manifest SHA, seed and
   group digest; order is `(rank_hash, group_id)`. Within each stratum of `n`
   groups, test receives the first `ceil(n/100)`, validation the next
   `ceil(n/100)`, training the rest. For `n=1`, train only; `n=2`, one train and
   one validation; `n>=3`, at least one group per split. For `pub`, quotas are
   24/1/1; for Gutenberg, 641/7/7. Actual record/word ratios can differ.
4. Entire groups and document boundaries remain immutable. Documents with
   **at least 100,000 words** are flagged and counted by split/domain; none is
   truncated, discarded, or forced into training. Future C4 segments inherit
   their parent split. Large records can dominate word totals: report both
   word-weighted and document-level results, plus length bands. This seed places
   all 22 long Gutenberg books in training, so its held-out books do not assess
   the longest-work regime. Do not claim otherwise or move chapters across splits.
5. Output is one reference Parquet with IDs, original coordinates, content hashes,
   word counts, long flags, source/subset/domain and assignment. SQLite provides
   disk-backed indexing, uniqueness constraints, transactional per-shard
   checkpoints, and full-population group ranking. Fixed output sort/row groups
   make checksums independent of input order and batch sizes under the pinned
   runtime. Restart rejects changed policy, input, implementation or runtime.
   Interrupted shard transactions replay; assignments publish atomically and
   become usable only with the completed manifest and independent verification.

The available metadata does **not** support parliamentary session grouping,
Carolina original-work/URL grouping, original PDF grouping, or grouping by
GigaVerbo URLs. Shared landing/repository URLs would collapse millions of
unrelated records. Distinct editions, shared works across sources, adjacent
parliamentary lines and approximate overlaps may span splits. Even cross-source
exact repetition involving D1's preserved ParlamentoPT population is not ruled
out by these source-scoped groups. The common holdouts measure this retained
distribution, not an independently clean novel-work test. These limitations are
accepted for review without reopening D2 or decontamination.

## Candidate capacities and reference views

A remains **curated-heavy**, B **balanced**, C **residual-heavy experimental
control**. All configs validate, sum to one at both levels, pin the same sources
and share the identical eleven-subset recipe. **B is only the provisional
preparation reference.** The winner is selected after comparable training with
one fixed C2 tokenizer and C3 architecture; selecting it is not a prerequisite
for preparing those common inputs.

| Residual subset | Post-D1 documents | Normalized words | Share within residual |
| --- | ---: | ---: | ---: |
| finepdfs_por_Latn | 909,373 | 4,242,157,200 | 30% |
| crawlPT_dedup | 813,818 | 791,425,795 | 10% |
| quati | 181,372 | 33,854,779 | 5% |
| blogset | 41,411 | 15,892,975 | 5% |
| fineweb_2_pt | 5,881,721 | 5,692,906,141 | 35% |
| mc4_pt | 2,476,815 | 2,493,360,521 | 5% |
| hplt2_pt | 4,411,773 | 4,431,606,538 | 5% |
| hplt1_pt | 415,307 | 904,738,680 | 2% |
| common_crawl | 439,631 | 564,586,897 | 1% |
| oscar | 122,311 | 155,092,956 | 1% |
| culturax | 17,809 | 16,942,627 | 1% |

For total word target `W`, every source/subset requires `W * source_share *
subset_share <= available_training_words`. The no-reuse relaxation is
`floor(min(capacity / weight))`, computed with rational arithmetic. Whole
documents introduce additional granularity; a word ceiling is not a promise
of an exactly achievable whole-document mixture.

| Candidate | Gross post-D1 ceiling before holdout | Binding quota | After proposed holdout |
| --- | ---: | --- | --- |
| A | 263,186,287 words | Gutenberg 8% | At most 259,918,425, using exact remaining Gutenberg words |
| B | 526,372,575 words | Gutenberg 4% | At most 519,836,850, using exact remaining Gutenberg words |
| C | 423,812,666 words | `blogset`: 75% × 5% = 3.75% | Unknown until full split; 415,336,413 if its word retention were exactly 98% (illustration only) |

All positive quotas have capacity, so bounded candidate experiments are
possible. The configurations cannot consume the entire 21.47-billion-word
eligible pool once at these proportions. At one billion words, even before
holdout, A needs 3.80× Gutenberg's capacity, B 1.90×, and C 2.36× `blogset`.
Any such schedule requires explicit repeated exposure/oversampling or a
scientifically approved recipe change. No reuse, reweighting or mixture copies
were introduced. The gross common no-reuse ceiling is A's; the common actual
training ceiling must be recomputed after full assignment and reduced for
whole-document granularity. Training-token budgets remain undecided.

Keep six distinct artifacts/concepts: (1) eligible D1 corpus; (2) fixed shared
validation/test identities; (3) complement training pool; (4) candidate sampling
views referencing only training rows; (5) separately pinned tokenizer sample;
(6) later tokenized training exposure budgets. Views can share training records
across A/B/C experiments, never within a single no-reuse pass or with heldouts.

Proposed future view recipe: pin mixture config hash, split manifest hash,
sampling seed and normalized-word target; rank eligible training document IDs
with a separate sampling namespace per source/subset and take a whole-document
prefix within each quota. Stop before exceeding that quota and report the
shortfall/achieved shares and the next document's length. Exhaustion fails the
requested target rather than silently repeating records. The scientist must
approve tolerated granularity errors or a declared repeated-exposure schedule.
No candidate sampler or large view materialization is needed in S1.

## Common evaluation and tokenizer preparation

Freeze the same held-out references for all candidates before any sampling.
Use validation for model/checkpoint decisions; reserve test for locked comparisons.
Report each family, all available domain/subset labels and length bands, with
counts and word totals. Predeclare a source-balanced macro summary and optional
fixed B-weighted summary; do not change evaluation weights with the candidate.
Fine semantic domains and national variety cannot be inferred from absent labels.
Any later bounded window selection for expensive evaluation must be shared,
seeded and parent-ID preserving. No C7 evaluator is implemented here.

Recommend one **training-only B-reference tokenizer sample** with independent
seed, whole-document references, checksums and achieved word proportions,
shared across every C2 tokenizer trial. Provisionally explore a 20-million-word
target only after sample policy approval; long-record quota shortfalls may be
large and must be reported. Pin the eventual sample size/granularity handling
before sampling. Choose/freeze one C2 tokenizer before A/B/C model comparisons.
This conditions the comparison on a balanced-reference tokenizer, potentially
favoring B; record that assumption, rather than asserting a mixture-neutral
tokenizer or training three candidate-specific tokenizers. Token counts and
equal training exposure budgets must be measured with that fixed tokenizer.

## Bounded results and handoff

Implementation: [`split.py`](../src/cambacica/corpus/split.py); synthetic tests:
[`test_split.py`](../tests/corpus/test_split.py); evidence runner:
[`validate_c1_split_readiness.py`](../scripts/validate_c1_split_readiness.py).
Default execution is bounded; `--full` requires a separate approval record.

Final packet outside Git: `/tmp/cambacica-base-180m/c1-s1/bounded-final/`.
It references **2,991 records, 138,517,734 words, 506 long documents**, sampled
from 46 files, covering all **21 source/subset strata** and all 655 Gutenberg
books. Most sampled records are deliberately long/diagnostic; this is not a
population-prevalence estimate. Other bounded assignments are not future full
identities because their stratum populations are incomplete; Gutenberg's complete
stratum gives exact 641/7/7 assignments and 20,793,474/112,946/148,483 words.

Assignment Parquet SHA-256:
`63d0a8eeba6b2ff6bc7f00b4ecbd86bbfa27bb2e154e5dce0636f17098e0b330`.
Split manifest SHA-256:
`a03742c47d096e69748d8b3da79e98632d6a33392edc6c953bd4fc3462293c65`.
Packet artifact manifest SHA-256:
`2dd4e90995e43237d1cd7b1a6f71967c53e6ca25efb48c06bea55801051e6ffb`.
The packet's `artifact_manifest.json`/`.sha256` also cover `readiness.json`,
state and the split checksum list. `index.sqlite` is restart scratch, not a
scientific artifact. The packet records implementation hashes and runtime.

Fresh reconstruction reverses files and row-group traversal, changes batch
size from 8 to 17, and requires byte-identical assignments/manifests. Validation
checks group integrity, zero ID overlap, correct source/word accounting, all
strata covered, metadata/hash failures and checksums. Synthetic tests additionally
exercise partial-shard replay, completed restarts, missing metadata, tiny strata,
changed policies and long records. Full pytest: **241 passed, 5 optional network
tests skipped**; Ruff and `git diff --check` pass. D1/BD3/BD4 manifest and review
payload hashes plus all D1 output sizes/mtimes are unchanged; no full D1 payload
rehash was performed in bounded validation.

Reproduce the bounded packet with a fresh outside-Git directory:

```bash
PYTHONPATH=src python3 scripts/validate_c1_split_readiness.py \
  --output-root /tmp/cambacica-base-180m/c1-s1/bounded-reproduction
```

Remaining local scientific decisions are approval of this split/group/rare-stratum
policy and its leakage limitations; tokenizer sample policy; comparable budgets,
quota tolerance and any explicit reuse schedule; and confirmation of the local
usage basis described above. Public release rights remain separate. C1 cannot
PASS until approved full split references/statistics and C2 input sample identities
exist and applicable source policies are recorded. **No final A/B/C winner is
required at this stage.**

## Exact next production step — only after scientific authorization

Create a scientist-issued JSON at
`/mnt/data/cambacica-base-180m/splits/c1-s1-v1-approval.json`, with
`scientist`, `decision_reference`, and these exact fields:

```json
{
  "action": "approve_full_split_manifest_and_verification",
  "policy_sha256": "11ffdfd9521c472106cec8b7d8882770329ea2d8bf3080a817d90634dbd37444",
  "input_manifest_sha256": "57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428"
}
```

No such approval was created in S1. From this committed checkout, the next
authorized production action is a read-only D1 pass producing **one reference
manifest**, followed by independent reconstruction:

```bash
PYTHONPATH=src python3 -m cambacica.corpus.split build --full \
  --policy configs/c1_split_policy_proposed.json \
  --approval /mnt/data/cambacica-base-180m/splits/c1-s1-v1-approval.json \
  --input-root /mnt/data/cambacica-base-180m/deduplicated/exact \
  --output-root /mnt/data/cambacica-base-180m/splits/c1-s1-v1
PYTHONPATH=src python3 -m cambacica.corpus.split verify --full \
  --policy configs/c1_split_policy_proposed.json \
  --approval /mnt/data/cambacica-base-180m/splits/c1-s1-v1-approval.json \
  --input-root /mnt/data/cambacica-base-180m/deduplicated/exact \
  --output-root /mnt/data/cambacica-base-180m/splits/c1-s1-v1
```

Each build verifies every input payload SHA-256 before reading it, then hashes
each record's stored text and counts words; the verifier rebuilds independently
in a temporary sibling directory. Thus the pair reads approximately **186 GB
of compressed input** (two checksum plus two parsing passes over 46,567,510,543
bytes), decompresses twice, indexes 21.6 million records twice, and writes only
references. Planning estimate, not a benchmark: **6–24 hours elapsed for
the pair**, provision **8–16 GB RAM and 100 GB free scratch** for original and
temporary SQLite tables/indexes and outputs. NFS random writes may exceed this
estimate; bounded timings in `readiness.json` are diagnostic-heavy and cannot
be extrapolated by document count. No GPU, downloads or external compute cost.

Require verifier PASS, byte-identical manifest/Parquet, 21,603,689 unique IDs,
21,470,091,017 words, exact source/subset totals against D1, expected tiny-stratum
quotas, group integrity and all five families represented in each heldout.
Recheck D1/BD3/BD4 identities, retain verification output, then recompute
training-only capacities from manifest accounting and approve sampling targets.
Stop there; tokenizer training and A/B/C corpus materialization are later actions.
