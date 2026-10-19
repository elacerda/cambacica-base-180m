# C1-M1 — Documentary rights review: Portuguese Wikipedia and Carolina v2.0.1

**Date:** 2026-10-19.
**Status: DOCUMENTARY REVIEW COMPLETE / SOURCE-USE CLEARANCE NOT GRANTED.**
**Authorization:** documentary review of public licensing terms and
existing C1 repository records only, to support a possible first,
bounded, rights-cleared C5 learning smoke (scientist-approved M1
*priority*, **not** an authorized execution).

**Hard stop:** this document grants **no legal, institutional or
scientific clearance** to read C1 corpus payloads, extract/tokenize
records, materialize a new training view, run GPU or train a model.
No local C1 corpus text, TEI documents, live record rows or token
shards were accessed in this review. Public publisher/dataset cards,
license policy pages and *existing* project scientific documents
were inspected. It is not a substitute for a decision by the
responsible researcher/institution on applicable law and intended use.

## Recorded M1 intended-use scope (scientist decision, 2026-10-19)

**SCIENTIST-DEFINED PURPOSE / NOT RIGHTS CLEARANCE.** The scientist
explicitly clarified that M1 is **entirely internal to CBPF** and that
**no model weights will be published**. Interpret the limited M1
experiment as an internal institutional learning/engineering smoke,
not as a public model, data or inference service. Under the
internal-only scope, no external publication/distribution of
M1 checkpoints, data, token shards, generated samples, or other
artifacts is planned **in this first phase**. Any later dissemination
is a **new use case** requiring separate rights and scientific review.

**Important distinctions:**

- Internal checkpoint files (model/optimizer/RNG/cursor) will be
  required by the future C5 technical design; *storing them privately*
  is not the same as *publishing/distributing weights*. Their creation
  is **not authorized** by this documentary decision.
- The researcher declared an internal scope; that alone does not
  establish a legal classification such as `non-commercial`, a
  statutory training exception, permission to reproduce protected
  texts, or permission to ignore BY/SA/NC or third-party terms.
  Confirm the institutional-purpose and licensing basis for
  training/copying **even without distribution**.
- The Wikipedia M1 review can now focus on **local internal training
  with no public release**, retaining an internal attribution,
  license/version and source-URL ledger. Questions about public
  model release are **out of the current M1 scope**, not permanently
  resolved or waived.
- The Carolina v2.0.1 component/record-license and header-license
  discrepancies persist regardless of the no-release decision;
  internal-only use does not turn heterogeneous document rights
  into a global unrestricted license.
- Record internal-only personnel access, retention, audit, incident
  response and any data-protection constraints in the later C1/C4
  operational approval. Exact access list and retention intervals
  remain **OPEN**.

**Next decision requested from CBPF/responsible researcher:** a
written conclusion that a narrowly specified, immutable
Wikipedia-PT-only C1-S1-compatible M1 view can (or cannot) lawfully
be read, copied and tokenized for **internal local research training**,
with required license compliance and data-protection controls.
**Do not construct or read that view until a separate explicit
scientific/access authorization.** If Wikipedia is not cleared,
evaluate another source; do not silently fall back to Carolina.

## 1. Identified sources and scope

| C1 family | Frozen upstream identity | Existing post-D1 census | Relevant existing C1 evidence |
| --- | --- | ---: | --- |
| Wikipedia Portuguese | `20231101.pt`, revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | 1,112,120 documents; 413,136,890 normalized words | Article IDs, titles, URLs; stored CC BY-SA 3.0/4.0 and GFDL labels |
| Carolina | `carolina-c4ai/corpus-carolina` v2.0.1; HF revision `55e63a519393c70a48dcfa14a558499c6bb0583b` | 2,108,727 documents; 1,296,640,886 normalized words | Seven `dat/jud/leg/pub/soc/uni/wik` taxonomies; TEI-level headers and inherited per-document licenses |

These numbers are the **already published** C1-D1 census, not
newly computed counts, authorized subsets or post-holdout capacity.
The C1-S1 split remains frozen:

- Split manifest SHA-256:
  `8b937f90d40487c643fe48baa6b5c2950713fe2e0d2d534f3e2a3c08e74dbe16`.
- Assignment Parquet SHA-256:
  `ef52e3cf3033934df13e3d16c207e1783c509dff6fb91954fb8c25ab719ece1e`.

A future M1 allowlist cannot silently alter C1 eligibility,
redistribute groups among train/validation/test, or imply that
source rights are resolved for all records in the family.

## 2. Portuguese Wikipedia — primary-document review

### Official terms and 2023 licensing transition

- Wikimedia Foundation **Terms of Use**, especially `7. Licensing
  of Content`, say eligible community-contributed text is under
  CC BY-SA 4.0 and GFDL where applicable; the terms permit
  commercial reuse subject to license compliance and discuss
  attribution via article URL/history. They warn about imported
  material, other underlying licenses, and non-text exceptions.
  <https://foundation.wikimedia.org/wiki/Terms_of_Use>
- Wikimedia's **June 29, 2023** transition announcement explains
  the move from CC BY-SA 3.0 to 4.0; contributions and their
  provenance do not all become identical historical rights.
  <https://diff.wikimedia.org/2023/06/29/stepping-into-the-future-wikimedia-projects-transition-to-creative-commons-4-0-license/>
  Creative Commons separately describes older contributions under
  3.0 and newer contributions under 4.0:
  <https://creativecommons.org/2023/06/29/wikipedia-moves-to-cc-4-0-licenses/>.
- Attribution, license link, change indications and appropriate
  ShareAlike handling need a documented reuse strategy.
  Wikimedia's attribution guidance is available at
  <https://attribution.wikimedia.org/>. The exact requirements
  for any particular reuse derive from applicable licenses and
  relevant source/historical attribution, not merely this overview.
- Wikimedia's 2023 US Copyright Office submission expresses
  general support for AI-model development from Wikipedia content
  while raising **attribution** and possible **ShareAlike** concerns.
  It does **not** waive copyright or unconditionally license every
  training/model-release scenario.
  <https://upload.wikimedia.org/wikipedia/commons/f/f7/Wikimedia_Foundation%27s_Responses_to_the_US_Copyright_Office_Request_for_Comments_on_AI_and_Copyright%2C_2023.pdf>.

### M1 implications — review, not clearance

**Positive evidence:** an identifiable openly licensed source with
existing article IDs/URLs in the C1 record. A controlled attribution
ledger is technically plausible, subject to design/approval.

**Open questions to resolve before a rights-approved Wikipedia view:**

1. Which terms/version(s) govern the exact **20231101.pt** acquisition
   and its transformations, including any imported third-party
   materials or copyright notices? Do not assume every row has
   identical licensing just because the dump is dated after June 2023.
2. What is the legal/institutional basis for **private academic model
   training in Brazil** and any later distribution of training
   excerpts, derived tokens, generated samples, checkpoints,
   model weights and model outputs? These are separate use cases.
3. What attribution/license-notice/version-history policy is
   appropriate to these uses? The current C1 article title and
   URL fields are useful, but not by themselves an audited
   attribution artifact.
4. Does the intended release trigger a ShareAlike obligation for
   a qualifying adaptation, and what if any limitations/exceptions
   apply? **Do not assert** that all trained weights automatically
   inherit CC BY-SA, or that ShareAlike can never apply to a model.
5. Retain only the source/view rows whose original C1-S1 assignment
   is approved for the bounded M1 experiment, without touching
   split identifiers.

**Provisional rights assessment:** `REVIEWABLE FIRST / NOT CLEARED`.
No blanket restriction on commercial purposes follows from
CC BY-SA alone, but attribution, reuse and possible derivatives
require review for the actual operation and jurisdiction.

## 3. Carolina v2.0.1 — upstream/version-specific license analysis

### The important distinction: wrapper/header vs document payload

The current official Hugging Face **v2.0.1** dataset card states
that texts come from multiple repositories and have **multiple
licenses**, while declaring **Carolina headers** under CC BY 4.0:
<https://huggingface.co/datasets/carolina-c4ai/corpus-carolina>.

The **exact upstream commit pinned in C1**, `55e63a519393c70a48dcfa14a558499c6bb0583b`
(June 11, 2025), changes the `_LICENSE` text of
`corpus-carolina.py` from a statement that Carolina headers are
`CC BY-NC-SA 4.0` to one identifying the headers as `CC BY 4.0`.
It **retains** the warning about heterogeneous underlying document
licenses. Evidence:
<https://huggingface.co/datasets/carolina-c4ai/corpus-carolina/commit/55e63a519393c70a48dcfa14a558499c6bb0583b>.

**Version discrepancy requiring reconciliation, not guesswork:**
the official LaViHD/USP Carolina terms page still describes the
**headers** as `CC BY-NC-SA 4.0` and says that every underlying
document has its own licensing metadata:
<https://sites.usp.br/corpuscarolina/creditos/>.
A historic Hugging Face v1.2/v1.3 dataset card likewise reports
CC BY-NC-SA for the headers, while v2.0.1's loader states CC BY 4.0.
The pinned commit is direct evidence of a **metadata/header policy
update**; it is **not evidence that every underlying text was
relicensed**. If the rights of a specific header version matter to
planned distribution, seek clarification from the maintainers/
institution rather than overriding one statement silently.

### Component-level preliminary matrix

| Carolina taxonomy | Preliminary concern to check | M1 status |
| --- | --- | --- |
| `wik` (wikis) | The actual component may include multiple Wikimedia projects/historical licenses, not all `pt.wikipedia.org`; preserve source-specific notices and distinguish overlapping upstream Wikipedia copies | **NOT CLEARED** |
| `dat` (datasets/corpora) | Heterogeneous upstream copyright/licenses; official older repository inventory includes CC BY-NC-SA and other terms; possible third-party rights | **NOT CLEARED** |
| `leg` (legislative) | Distinguish uncopyrightable official acts from protected commentary/compilations and upstream conditions; no global exemption inferred | **NOT CLEARED** |
| `jud` (judicial) | Check document nature, reuse limitations, embedded protected text and provenance; legal texts not synonymous with a clear corpus-wide permission | **NOT CLEARED** |
| `uni` (university domains) | University-hosted does not imply public-domain or general reuse consent; preserve original site/license | **NOT CLEARED** |
| `soc` (social media) | Original author/terms, privacy/personal-data issues and restrictions need special attention | **NOT CLEARED** |
| `pub` ("public domain works") | Taxonomy is not legal proof: older official Carolina repository inventory contains mixed CC BY, CC BY-SA, CC BY-NC and public-domain declarations even under this heading | **NOT CLEARED** |

The official Carolina historical repository list is useful for
locating **known classes of original licenses** but is a v1.1
contextual index, **not** a verified v2.0.1 document-level inventory:
<https://sites.usp.br/corpuscarolina/repositorios-2022/>.
The C1 scientific record already says per-document headers/labels
exist and original URLs are absent from its characterization.
Therefore, do **not** promise reconstruction of complete original
URLs or licenses for every retained v2.0.1 record from the summary
metadata.

**No Carolina taxonomy is approved wholesale for M1.**
Academic research purpose does not override license-specific NC,
attribution, ShareAlike, personal-data or other conditions.
C1's existing historical license-characterization results are not
a fresh record-level v2.0.1 rights census.

### What needs to be answered outside this documentation review

- Have the actual **v2.0.1** license fields been retained correctly
  for each candidate document or group in the frozen C1 metadata?
  Are source URL/author/rights references sufficient for compliance?
- Is any relevant text `BY-NC`, `BY-NC-SA`, another copyleft-like
  condition, or an explicit incompatible/ambiguous license? Define
  M1's institutional-purpose/release intentions *before* assessing
  those tags.
- Can the Carolina maintainers clarify whether the 2025
  `_LICENSE` update and older institutional page represent differing
  header-version conditions, and whether it changes the intended
  terms of v2.0.1 TEI headers?
- Can source-specific provenance and attribution obligations be
  satisfied without unauthorized retrieval of third-party payloads?
- Which subsets, if any, are sufficiently cleared to enter a
  separately scientist-approved fixed M1 view?

**Provisional rights assessment:** `HETEROGENEOUS / NEEDS
COMPONENT- OR RECORD-SPECIFIC REVIEW / NOT CLEARED`.

## 4. CC conditions in AI training: avoid false legal conclusions

Creative Commons explicitly explains that **copyright exceptions
and jurisdiction matter** and that licenses apply where permission
is actually required. Its guidance distinguishes:

- `BY` attribution;
- `SA` conditions applicable to sharing adaptations when triggered;
- `NC` conditions restricting qualifying commercial uses, even
  during dataset-copying/training where copyright permission is
  needed;
- separate assessment for model checkpoints, release/output,
  excerpt redistribution and public training data.

See the Creative Commons guidance:
<https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/>.

These are **guidance and risk-identification**, not an
institutional legal ruling. For the M1 request, seek determination
from the responsible researcher/institution about the exact
intended local research activity, jurisdiction and eventual
release policy. Avoid both blanket claims: "everything CC can
train any model without conditions" and "all BY-SA training
automatically forces weights to BY-SA."

## 5. Decision packet for the responsible researcher/institution

| Required evidence/decision | Wikipedia | Carolina | Owner / status |
| --- | --- | --- | --- |
| Intended activity and dissemination | **Scientist set internal CBPF-only M1 with no published weights or other external artifacts for this phase** | Same intended-use scope if later selected | **PURPOSE DEFINED / institutional rights decision PENDING** |
| License/rights basis appropriate to frozen version and jurisdiction | Wikipedia ToU and 2023 transition identified; reconcile exact content | Heterogeneous v2.0.1 originals and header-version discrepancy | **OPEN — institutional rights review** |
| Attribution and license obligations ledger | Article ID/URL fields inherited from C1, implementation needed | TEI fields/individual upstream rights, sufficiency not verified | **OPEN** |
| Eligible sources/subsets, ineligible sources and fail-closed behavior | Decide for a bounded view | Decide per component/record, not whole corpus | **OPEN** |
| Immutable train/validation/test M1 view | Must inherit C1-S1 group/split hashes | Must inherit C1-S1 group/split hashes | **OPEN — scientist approval; no materialization** |
| Rights release separate from internal research | Test / checkpoints / weights / generated excerpts | Test / checkpoints / weights / generated excerpts | **OPEN — legal/institution** |
| Corpus content/tokenization or compute run | Not requested | Not requested | **NOT AUTHORIZED** |

### Proposed fastest M1 rights resolution

1. Request a documented institutional interpretation for a
   **small Wikipedia-only M1** in the scientist-defined **internal CBPF
   research/no-publication** context; document required compliance,
   private storage, institutional usage basis and retention policy.
   This is a **proposed candidate**, not rights clearance or
   selection of a view.
2. If cleared, separately ask the scientist to approve a fixed
   Wikipedia source-only train/heldout subset under C1-S1 and the
   rights/attribution ledger. Only then seek independent
   authorization for any content read/tokenization.
3. Review Carolina **in parallel** by version and source-specific
   license classes. Reconcile the header-license discrepancy and
   rights metadata completeness before selecting any Carolina
   document for M1.
4. Continue synthetic-only C4-D and recipe planning only under
   their own new scientist approvals. C5 requires a further
   explicit GPU0-only authorization after C4 validation. GPU1
   (the operational Qwen/vLLM service) is excluded.

This order avoids making the entire Carolina rights audit a
prerequisite for a possible limited Wikipedia-only C5, while
preserving Carolina for later scientific review and C6 planning.
It does **not** bypass unresolved source rights.

## 6. Gate and stop conditions

**C1-M1-RIGHTS: DOCUMENTARY EVIDENCE REVIEW COMPLETE /
CLEARANCE AND VIEW SELECTION PENDING.**
**C1: IN PROGRESS; C4: IN PROGRESS; M1 PRIORITY APPROVED; C5/C6: PENDING.**

Nothing in this review selects a corpus mixture A/B/C,
authorizes training/real data access, or upgrades C1 to PASS.
No changes to C1 population, D1, frozen C1-S1/C2/C3 identities
or production data files were made.

### Public official sources checked on 2026-10-19

- Wikimedia Foundation: <https://foundation.wikimedia.org/wiki/Terms_of_Use>
- Wikimedia CC 4.0 transition: <https://diff.wikimedia.org/2023/06/29/stepping-into-the-future-wikimedia-projects-transition-to-creative-commons-4-0-license/>
- Creative Commons on the licensing transition:
  <https://creativecommons.org/2023/06/29/wikipedia-moves-to-cc-4-0-licenses/>
- Wikimedia AI/copyright submission (2023):
  <https://upload.wikimedia.org/wikipedia/commons/f/f7/Wikimedia_Foundation%27s_Responses_to_the_US_Copyright_Office_Request_for_Comments_on_AI_and_Copyright%2C_2023.pdf>
- Wikipedia attribution guidance:
  <https://attribution.wikimedia.org/>
- Carolina v2.0.1 dataset card:
  <https://huggingface.co/datasets/carolina-c4ai/corpus-carolina>
- Pinned Carolina loader license change:
  <https://huggingface.co/datasets/carolina-c4ai/corpus-carolina/commit/55e63a519393c70a48dcfa14a558499c6bb0583b>
- Official Carolina terms:
  <https://sites.usp.br/corpuscarolina/creditos/>
- Carolina historical source inventory (v1.1 contextual only):
  <https://sites.usp.br/corpuscarolina/repositorios-2022/>
- Creative Commons guidance on AI training:
  <https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/>
