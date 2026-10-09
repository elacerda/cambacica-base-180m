# Gate C1 Benchmark-Decontamination Plan

**Status:** planning proposal; production scan and benchmark evaluation are **NOT RUN**
**Next:** scientist approval of the benchmark inventory, provisional match rules, and review/exclusion policy
**Corpus input:** post-D1 exact corpus, immutable and identified by manifest SHA-256 `57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428`

## 1. Objective and scope

Reduce the risk that a first-run Cambacica Base 180M evaluation measures recall of public benchmark material already present in pretraining. This is a narrow overlap check against selected evaluation sets. It is not general corpus deduplication and does not estimate or eliminate semantic memorization.

The source corpus stays immutable. D1 exact deduplication is complete; near-deduplication removals are closed at zero for the first run by [C1_NEAR_DEDUP_FINAL_DECISION.md](C1_NEAR_DEDUP_FINAL_DECISION.md). Do not change the A/B/C mix specifications, normalization contract, or source documents. Decontamination is the next C1 stage; it has not run.

The first-run evaluation proposal is deliberately small:

1. **Intrinsic held-out C1 text**, created only after benchmark decontamination and a separately approved deterministic document split.
2. **CALAME-PT**, for native Portuguese next-word prediction.
3. **Belebele por_Latn**, for passage-based multiple-choice reading comprehension.

ENEM is a useful optional Brazilian Portuguese exam evaluation, but its underlying text and image rights need review. Other translated or supervised task sets are deferred below.

## 2. Benchmark inventory and recommendations

The versions below are concrete candidate pins. Before execution, record the retrieved dataset file checksums and confirm that the cited revision contains the expected files. A dataset card’s license label does not by itself settle rights to embedded third-party text.

| Candidate | Capability and language coverage | Dataset/configuration/splits and verified size | License, visibility, base-LM protocol | Recommendation and contamination risk |
| --- | --- | --- | --- | --- |
| **CALAME-PT** ([paper](https://aclanthology.org/2024.propor-1.45/), [dataset card](https://huggingface.co/datasets/NOVA-vision-language/calame-pt)) | Native Portuguese last-word prediction; the public card labels language pt, without a balanced PT-BR/PT-PT breakdown. | NOVA-vision-language/calame-pt, content commit 353671bc95cc3d94d488201f67d41b640eb80c55; config all (also handwritten, generated); no official train/dev/test split. The card describes 406 handwritten and 1,670 generated examples (2,076 total). | Dataset card declares MIT. Contexts and target words are public. Base-LM protocol exists: zero-shot final-word prediction, exact-match accuracy. Use a deterministic first-word greedy continuation; report handwritten, generated, and combined results separately. [Tucano publishes a standalone evaluation script](https://github.com/Nkluge-correa/Tucano/blob/main/evaluations/README.md); a pinned custom task in lm-evaluation-harness is also feasible. The card says the generated set was derived from Portuguese Wikipedia, OSCAR, and Arquivo.pt sources and rewritten/summarized with GPT-3.5 before human review. | **Core, conditional on version/license check.** High source-overlap risk: Cambacica includes Wikipedia PT and web material, and GigaVerbo includes OSCAR-derived/web data. The card does not provide per-example source IDs; exact text matching cannot detect every rewrite. |
| **Belebele** ([ACL paper](https://aclanthology.org/2024.acl-long.44/), [official repository](https://github.com/facebookresearch/belebele), [dataset card](https://huggingface.co/datasets/facebook/belebele)) | Reading comprehension from a passage; 122 language variants, with one general Portuguese variant, por_Latn. It does not split Portuguese into PT-BR and PT-PT. | facebook/belebele, pinned revision d4c91dedc9de484dbea7b7d940f898f59fd135e9; config por_Latn; split test; 900 questions from 488 passages. | CC BY-SA 4.0 for Belebele. The test examples and correct-answer numbers are public. The official project describes four-option accuracy and no-finetuning protocols that use probability to select answer letters. A plain zero-shot Portuguese prompt for a base model is a useful adaptation, not the paper’s directly reported protocol; label it accordingly. A custom lm-evaluation-harness task or a pinned Lighteval task adapter can score causal-LM likelihoods; do not imply either is the official benchmark protocol. The separate assembled few-shot training data has different, largely CC BY-NC terms; it is not needed here. | **Core, conditional on approving the zero-shot scoring adaptation.** Strong fit for a causal LM and short-context evaluation. Passage text comes from FLORES-200 and benchmark items are public, so copied passage or item overlap is plausible. |
| **ENEM** ([Maritaca dataset](https://huggingface.co/datasets/maritaca-ai/enem), [source benchmark repository](https://github.com/piresramon/gpt-4-enem), [Lighteval task](https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/multilingual/tasks/enem.py)) | Brazilian secondary-school examination questions; broad subject knowledge and reading, sometimes image-dependent. | maritaca-ai/enem, candidate revision b2a7257356b0f56b740d57f23f8a575542ff6c0c; year subsets 2022, 2023, 2024; the Hub exposes a train split. Lighteval uses that split for evaluation; it is not an official train/dev/test partition. The 2022 config has 180 rows; confirm counts for 2023/2024 and text-only eligibility at snapshot time. | Hub metadata declares Apache-2.0. Questions, alternatives, labels, descriptions, and linked figures are public. Lighteval supplies multiple-choice formulations and raw/token-/character-normalized log-likelihood accuracy for causal LMs. The license of underlying exam excerpts and linked figures is not established by the Hub card alone. | **Optional; approval required.** Good PT-BR coverage and a usable base-LM scoring protocol, but questions and answer keys are widely public, contamination risk is high, some examples need images, and the underlying rights require review. |
| **Multilingual ARC (m_arc)** ([dataset revision](https://huggingface.co/datasets/alexandrainst/m_arc/tree/83330911ecbd2a9e9b5d59afb09b5db1ff77a7fa/pt)) | Multiple-choice science/commonsense; an actual pt configuration exists. | alexandrainst/m_arc, candidate revision 83330911ecbd2a9e9b5d59afb09b5db1ff77a7fa, config pt, with train, val, and test. Hub lists 10K–100K for the whole multilingual repository; Portuguese split counts need metadata verification. | CC BY-NC 4.0. The pt test config is publicly hosted. A causal LM can score answer continuations; however, exact prompt, option scoring, translation provenance, and usable rights must be pinned before use. | **Deferred.** A real Portuguese configuration exists, but translated content and non-commercial terms make it a poor first-run core without a specific scientific and licensing reason. |
| **ASSIN2** ([shared-task proceedings](https://ceur-ws.org/Vol-2583/), [TFDS catalog](https://www.tensorflow.org/datasets/catalog/assin2)) | Textual entailment and semantic similarity; Brazilian Portuguese only. | TFDS reports 6,500 train pairs, 500 validation pairs, and about 3,000 test pairs; another packaged view reports a smaller test count, so exact distribution must be pinned. | Test examples/labels are available in public dataset implementations. The official metrics are macro-F1 for entailment and Pearson correlation for similarity. [Tucano’s open evaluation harness](https://github.com/Nkluge-correa/Tucano/blob/main/evaluations/README.md) reports 15-shot RTE and 10-shot STS metrics without task fine-tuning, but this is a model-specific protocol rather than a stable zero-shot standard. The redistribution license for the dataset itself remains unclear. | **Deferred.** A base LM could verbalize entailment labels, but that introduces prompt/label calibration; STS is not a natural causal-LM scoring task. Not needed for the first compact suite. |
| **XNLI** ([official repository](https://github.com/facebookresearch/XNLI), [paper](https://aclanthology.org/D18-1269/)) | Cross-lingual natural-language inference. | Official XNLI has 2,500 development and 5,000 test sentence pairs per language; 14 translated languages plus English. | Public examples and labels; original work targets cross-lingual sentence classification. A log-probability verbalized-label adaptation is possible. | **Reject for this suite.** The official language list has no Portuguese variant. A third-party translated XNLI-PT would be a different dataset and would need its own provenance and validation. |
| **HellaSwag-PT / LAMBADA-PT** ([Tucano evaluation documentation](https://github.com/Nkluge-correa/Tucano/blob/main/evaluations/README.md), [Tucano paper](https://doi.org/10.1016/j.patter.2025.101325)) | Translated commonsense completion and next-word prediction used in Portuguese causal-LM studies. | Tucano documents its Portuguese evaluations; no immutable dataset revision for a first-run Cambacica snapshot is selected here. | Tucano classifies ARC-Challenge and HellaSwag as translated, and LAMBADA as translated. A causal-LM likelihood/continuation protocol exists. | **Deferred.** Useful for later matched-protocol comparison, but native CALAME-PT is preferable for first-run language modeling and stable exact dataset/license pins must be established before decontamination. |

### Why this core is proportionate

CALAME-PT measures a causal continuation behavior directly in Portuguese, and its protocol has been used for small Portuguese base models. Belebele adds a different capability—reading a passage and selecting a supported answer—without instruction tuning. Both can be evaluated on a small GPU or CPU inference stack because each has only hundreds or a few thousand items. They are not broad measures of intelligence or factual reliability.

### Small Portuguese base-model comparability

The most relevant near-scale comparator identified here is **Tucano-160m**, a native Portuguese causal base model close to Cambacica’s 180M target. Tucano publishes its training/evaluation artifacts and uses CALAME-PT; its paper also reports Portuguese ARC/HellaSwag/LAMBADA protocols. These are useful protocol precedents, not automatically comparable published numbers.

A fair comparison requires re-evaluating the public base checkpoint and Cambacica with the same pinned benchmark files, prompt, scorer, tokenizer/model code versions, context limit, and deterministic settings. Record each model’s pretraining sources and known benchmark overlap. Tucano’s use of GigaVerbo makes source overlap relevant to CALAME-PT; do not present an unverified cross-model comparison as contamination-free. Compare token perplexity only when tokenization and evaluation procedure match. Avoid comparing base models to instruct checkpoints, or treating a translated task score as direct evidence of native Portuguese language quality.

## 3. Version and provenance contract

Before a production scan, the approved inventory must include for each benchmark:

- canonical name, capability, language/variant, repository URL, immutable revision, dataset configuration and split;
- source dataset file names, byte counts, SHA-256 hashes, license metadata, upstream paper, redistribution notes, and any unresolved rights;
- stable example ID (source ID where supplied; otherwise a documented ID derived from dataset revision, config, split, row key/ordinal, and canonical row hash);
- original passage, question, options, target/answer fields as permitted, plus a checksum for each canonical example;
- task-field mapping and an explicit distinction between benchmark training/dev/test data and any private set.

The core candidate pins are present above, but must still pass an implementation-time revision/schema/checksum check. Do not use a moving main revision. Do not retrieve hidden or restricted answer keys. The chosen core has public evaluation answers. Do not download full multilingual datasets: fetch only the approved Portuguese config/split and metadata once inventory approval is granted. If terms, the expected config, or an immutable revision cannot be confirmed, mark the benchmark unresolved and do not scan against it.

For CALAME-PT, there is no official split; the full pinned all collection is evaluation-only, with source-defined handwritten/generated subsets reported separately. For Belebele, only the pinned por_Latn/test split is used. ENEM’s Hub train name means the full public exam collection used by the evaluator, not model training data. No private benchmark set is proposed.

## 4. Contamination threat model

A candidate training document is potentially contaminated when it contains benchmark-specific text, not merely the same topic.

| Case | Meaning and treatment |
| --- | --- |
| Exact prompt, passage, or question | Detect exact normalized token-sequence presence, retain original offsets, and review context/source. |
| Exact answer-containing material | Flag when the benchmark answer occurs together with its item, passage, options, or benchmark-specific context. A single short answer, option letter, or common phrase is not sufficient. |
| Prompt and answer together | Strong evidence of item-level exposure; confirm that the match is not a generic template before classifying it. |
| Near-identical example | Detect high, rare-token overlap across the benchmark fields. Require multiple distinctive anchors or substantial field coverage and human review. |
| Significant contiguous passage overlap | Flag a long matching run embedded in a longer source document, even if whole-document Jaccard is low. Review quotations, public-domain text, and duplicated exam sources. |
| Short generic text or boilerplate | Examples such as “assinale a alternativa correta”, ordinary greetings, labels, or common connective phrases are not enough. Track their corpus document frequency and downgrade them. |
| Widely quoted/public source text | Treat as a possible exposure signal, not proof of benchmark memorization. A document may quote a source legitimately; preserve evidence and source context for review. |
| Public exam material reproduced online | Treat exact question-plus-answer copies as high-risk exposure. Ordinary references to an exam, subject, or answer alone are not contamination. |
| Translation or paraphrase | Detect only shared lexical spans that pass the overlap rule. Translation without shared anchors and semantic paraphrase are outside reliable coverage. |

The proposed method can detect exact text, sufficiently long shared token runs, and near-copy patterns with distinctive shared n-grams. It cannot reliably detect semantic paraphrases, machine or human translations with little lexical overlap, answer inference without answer text, image-only content, or content in unindexed variants. Topic overlap is not contamination. A match identifies an exposure risk; it does not prove that the model memorized or used the text.

## 5. Detection pipeline

All thresholds below are **provisional**. A small labeled calibration set must include real benchmark/corpus examples, obvious positives, common-template negatives, public quotations, and short-item controls. Scientists approve the final values before a full scan.

### Stage 1 — Snapshot

Create a read-only benchmark snapshot and manifest for only the approved splits/configs. Preserve raw source fields, public labels/answers allowed by the license, stable example IDs, row checksums, and the exact source revision. Keep this snapshot in the decontamination area, separate from the training corpus.

### Stage 2 — Match representation

Keep benchmark and corpus original strings untouched. Create a derived match view using Unicode NFC and the existing D2 token form: lowercase Unicode re.\w+ tokens, in order, with diacritics preserved. Do not strip accents, stem, translate, remove stopwords, or reorder tokens. Store token offsets back to original character offsets. This makes case/punctuation/whitespace variation manageable while preserving traceability and avoiding aggressive collisions.

### Stage 3 — Exact matches

Check canonical full-field and composed-item sequences (passage, question, options, and answer where available), plus question/passage-only sequences. Use hashes for exact token-sequence lookup, but retain and verify the actual matched tokens and source offsets. An answer string alone is a diagnostic, never an exclusion trigger.

### Stage 4 — Bounded approximate overlap

Use benchmark-token n-gram lookup to find matches while streaming corpus
documents. Count per-query n-gram document frequency during the scan and cap
retained postings at `max_ngram_df + 1`; calibration must set this provisional
common-text cutoff, and scientists must approve it. Once an n-gram exceeds
that limit, discard its postings and do not use it as overlap evidence. This
bounds frequent boilerplate without a second corpus pass. Do not reuse D2c’s
global Jaccard cutoff, LSH index, or 256-member bucket cap: a short benchmark
passage may be embedded in a long document, and the D2c candidate system was
designed for document-to-document similarity.

Provisional candidate rules for calibration:

- a normalized contiguous match of at least **50 word tokens**; or
- a near-copy of a complete benchmark item with at least **80%** of its distinctive benchmark tokens covered and at least **two** matched 13-token anchors that are rare in both the benchmark inventory and corpus.

A benchmark item shorter than 50 tokens may still be flagged by an exact full-item match or by the second rule when it contains enough distinctive anchors. A match made only of high-document-frequency n-grams is generic/common overlap. `max_ngram_df` and the coverage/anchor values are starting points, not frozen thresholds; assess precision and miss cases by benchmark type and corpus source. No semantic embedding detector is proposed.

### Stage 5 — Review and decision

Classify each candidate as **confirmed contamination**, **likely / review required**, **generic/common overlap**, or **unresolved**. Reviewers see the benchmark ID and pinned fields, source record ID and provenance, matched offsets/excerpts, source/subset, document frequency, match rule, and enough surrounding context to distinguish an item copy from a legitimate quote.

Do not delete documents during matching. For a confirmed full benchmark item or question-and-answer copy, recommend excluding that immutable source record from the future training view. For a long record containing only a benchmark passage, review the information loss and mix impact: scientists may approve whole-record exclusion, or retain the training record and omit the affected benchmark item from the clean-score denominator. Report both the original and clean denominators if benchmark items are omitted. Likely or unresolved matches are retained unless explicitly approved; report their counts and limitations. Never trim or rewrite source text as an unrecorded fix.

### Stage 6 — Verification

The verifier must check benchmark and corpus manifest identities, all checksums, expected row counts, deterministic benchmark IDs, matcher version/config, complete scan counts, and exclusion references. Synthetic fixtures must cover exact full items, answer-only false positives, punctuation/case differences, accents, long embedded passages, common boilerplate, short texts, and near-copy edits. Re-running the same pinned inputs and policy must produce byte-stable decision artifacts or a documented deterministic equivalent.

The final report must quantify reviewed categories and per-source/subset impact. It must describe what the detector cannot see. Passing verification means the approved policy was applied reproducibly; it does not establish zero contamination.

## 6. Corpus access, compute, and storage

### Measured facts

The exact manifest lists **532 retained Parquet data files**, totaling **46,567,510,543 bytes (43.37 GiB)** and 21,603,689 rows. The current corpus root is `/mnt/data/cambacica-base-180m/deduplicated/exact`; its data should remain read-only. The D1 manifest supplies per-file checksums and row counts. D2c’s signature artifacts also retain stable record identities and row-group locators, which may be reused after input-manifest verification. D2c’s MinHash signatures are not a benchmark-contamination index.

On this host, the D2c full-corpus fingerprint pass took **25,807.8 seconds (7.2 hours)**, with **7.51 GB peak RSS**, while scanning all retained records and producing 5-word signatures. This is a measured analogous pass, not a benchmark-scan runtime.

### Proposed implementation shape and estimates

- Run one ordered, projected text scan over the pinned retained Parquet corpus. Read only text plus the identity/provenance columns needed to emit reviewable hits. Check row totals against the exact manifest.
- Build a small benchmark n-gram matcher from the approved benchmark snapshot. Stream candidate hits and benchmark n-gram document-frequency counters into bounded local scratch batches; publish final compact artifacts and manifests to NFS at `/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/`.
- Reuse D1 identity and D2c locators only for record lookup; do not rebuild or copy D2c signatures or its large LSH index. Retrieve full text only for candidate review, using row-group locators, rather than doing another full corpus pass.
- The known upper bound for a full Parquet read is the measured 46.57 GB compressed data footprint; projected text-column I/O should be lower, but the exact bytes have not been measured. Expect one several-hour, NFS-sensitive scan. Matching/tokenization is CPU-bound; reading/decompression is NFS/I/O-bound. Orion’s H100 is not needed for text matching.
- Candidate-output size depends on the unknown hit rate. Do not claim a precise scratch requirement before a small approved calibration run. Use bounded batches and external sorting/spill so scratch grows with candidate evidence rather than corpus size; keep durable reports and decisions on NFS. Set and record an implementation memory cap, and stop safely rather than dropping candidate rows if it is exceeded.

This is one full corpus pass, not a D2c rebuild. If a strict I/O budget, incomplete manifest, schema drift, or storage error prevents a complete scan, stop with an incomplete status and do not report decontamination as complete.

## 7. Output and training-view contracts

Proposed output root: `/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/`. Do not duplicate the source corpus.

| Artifact | Required content |
| --- | --- |
| `benchmark_manifest.json` | Dataset repos/revisions, config/split, files and hashes, field mapping, license note, example counts, stable benchmark IDs. |
| `benchmark_registry.csv` | One row per example/field with benchmark ID, original text-field role, text checksum, and public answer/label presence. |
| `matching_policy.json` | Versioned normalization, exact rules, provisional/final thresholds, common-text frequency rule, matcher version, exclusions and limitations. |
| `candidate_hits.parquet` | Immutable `record_id`, source/subset, file/row-group locator, benchmark ID and field, matched offsets/length, rule, score/anchor evidence, common-ngram counts, bounded excerpt/checksum. |
| `reviewed_hits.csv` | Candidate ID, human category, reviewer, timestamp, reason, supporting source/provenance, and review status. |
| `exclusion_decisions.parquet` | Record ID, benchmark ID, rule, confirmed/approved reason, reviewer approval, action, and input manifest identity. |
| `source_impact_summary.json` | Counts/words by source and subset, proposed exclusions, retained capacities, and impact on each A/B/C pool without editing mix configurations. |
| `verification_report.json` | Input identity, scan counts, checksum/schema results, matcher checks, determinism, remaining detection limits, and pass/fail. |
| `manifest.json` | Tool commit/runtime, run status/times, input benchmark/corpus identities, and SHA-256/byte counts for every output. |

Every proposed exclusion must resolve to one immutable post-D1 record_id, one pinned benchmark example, one rule and matched evidence, an explicit decision reason, a human approval, and the retained source manifest. The exclusion manifest creates a reproducible **training view** over unchanged source rows. A later materialized view, if needed, must be derived from those same rows and the approved exclusion manifest, then verified against both input identities.

Do not modify source normalization or silently change A/B/C weights. Recompute available documents/words and feasible capacity for all three existing mixes after approved exclusions. If any unchanged mix is no longer feasible, return to the scientists rather than silently redistributing its weights.

## 8. Evaluation contract proposal

### Intrinsic held-out language modeling

Use document-level held-out C1 text for validation and test, not CALAME/Belebele examples. Preserve the project order: benchmark decontamination first, split later. Proposed split rule for approval: group all records with identical normalized content_sha256 under one split, then assign groups deterministically from stable IDs and a pinned seed. D1 has already collapsed eligible exact duplicates; this grouping also handles preserved exact-repeat diagnostics such as parliamentary lines.

The project’s current generic split text also says near-duplicates must not cross splits. Because the first run retains near copies and D2c candidate generation is capped, a corpus-wide near-copy grouping guarantee is unavailable without another investigation. The proposed first-run split contract therefore guarantees document and exact-content-group disjointness, reports known D2c overlap as a limitation, and does not claim near-duplicate-free splits. This contract needs scientist approval before any split is made; it does not reopen or block the approved zero-removal near-dedup decision.

Report mean token negative log-likelihood, token-level perplexity (exp(mean NLL)), and UTF-8 bits per byte on the same held-out text. NLL/PPL depend on the model tokenizer; bits/byte is more useful across different tokenizers, though model context and byte accounting still need to match. Pin the document IDs, tokenizer revision, context-window/stride method, BOS/EOS treatment, precision, and evaluator version. Use validation for choices and reserve test for final reporting.

### Zero-shot downstream

- No SFT, DPO, chat template, or instruction prompt.
- CALAME-PT: zero-shot greedy final-word completion; case-insensitive exact match; report handwritten, generated, and combined results and confidence intervals.
- Belebele: a fixed Portuguese prompt with passage, question, four answer choices, and a Portuguese answer marker; select the answer letter with highest conditional log probability. Report accuracy over the fixed 900-item test set. Freeze exact whitespace, labels, candidate ordering, and answer-marker text in the task config. A secondary choice-text likelihood score may be reported only if its length-normalization rule is pinned.
- No few-shot examples for the core. If few-shot is later approved, examples must come from a benchmark’s official training split, never dev/test; record the fixed example IDs, Portuguese wording, order, and random seed. CALAME and Belebele have no Portuguese task-training split in the described versions.
- Run the full benchmark deterministically. Pin dataset revision and checksum, evaluator release/commit, model/tokenizer revisions, prompt files, scoring code, context limit, and seed. Record tokenization and character normalization details.
- Publish per-example scores or a checksum-protected prediction artifact, aggregate metrics and uncertainty, sample counts, contamination flags, and the clean/original denominator. Report possible contamination instead of claiming zero.
- A 180M base model can be near chance on knowledge-heavy or reasoning sets; interpret such scores as task-specific evidence, not a general quality verdict.

## 9. Failure modes and approval points

Primary risks are an incomplete scan, a changed or mislabeled benchmark revision, insufficient dataset rights, common phrases creating candidate floods, short questions missing the span threshold, passage quotations being misclassified, text normalization creating false collisions, high hit counts exceeding scratch, and unobservable translations/paraphrases. A hit is not proof of memorization; a no-hit report is not proof of no contamination.

Scientists must approve before implementation:

1. Core snapshot: CALAME-PT all plus Belebele por_Latn/test, or a narrower scope; confirm CALAME license/source-overlap caveat.
2. Whether ENEM enters v1 after its source-text and image rights are reviewed.
3. Calibration labels and final approximate-overlap thresholds, including whether the provisional 50-token / 80%-coverage / two-13-gram-anchor rules are acceptable.
4. The confirmed-hit action for long multi-topic documents and the option to exclude an affected benchmark item when record loss is disproportionate.
5. The proposed exact-content-group split contract and the corresponding disclosure of residual near-copy risk.
6. The evaluation framework and immutable software version. [lm-evaluation-harness documents causal-LM log-likelihood and rolling-likelihood interfaces](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md) and is used by Tucano; custom pinned task configs may be needed for the core. [Lighteval documents a multiple-choice ENEM likelihood task](https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/multilingual/tasks/enem.py). Select and pin one implementation before scoring; no evaluator dependency is added in this planning task.

## 10. Implementation phases and success criteria

1. **C1-BD1 — Scientist approval.** Approve core datasets, versions, terms, candidate policy, record-level review action, and proposed split contract. Success: signed decision entry with no unresolved dataset rights/config required for the core.
2. **C1-BD2 — Snapshot and matcher specification.** Fetch only approved Portuguese config/splits, compute file/example hashes, freeze field mappings and match rules. Test on synthetic positives/negatives and a small scientist-approved calibration sample. Success: reproducible snapshot; reviewed candidate behavior; documented limits; no source-corpus mutation.
3. **C1-BD3 — One-pass production scan.** Scan exact D1 rows once with identity checks and bounded candidate output. Success: all 21,603,689 expected records accounted for; zero silent drops; complete manifests and checksums.
4. **C1-BD4 — Human review and training view.** Review candidate hits, approve or reject explicit record exclusions, calculate source/subset/A/B/C impact, and write immutable decision manifests. Success: every exclusion has direct evidence and approval; the input corpus and mix files remain unchanged.
5. **C1-BD5 — Verification and C1 handoff.** Re-run deterministic verification, publish residual-risk report, then separately approve deterministic corpus splitting and final mix materialization. Success: reproducible report and eligible post-decontamination corpus interface. This plan does not mark C1 PASS.

Until those approvals and stages are complete, benchmark decontamination remains **NEXT / NOT RUN**, C1 remains **IN PROGRESS**, and C2 remains **PENDING**.
