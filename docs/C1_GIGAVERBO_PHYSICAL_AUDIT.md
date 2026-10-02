# Gate C1 — GigaVerbo-v2 Physical Inventory

**Gate status: IN PROGRESS**
**Audit date: 2026-10-02 UTC**
**Repository:** `Polygl0t/gigaverbo-v2`
**Partition:** `edu_high`
**Pinned commit:** `7058ccf19eaeaf4505a96fc7e5305a01fc441fd8`

## Physical layout

The pinned partition contains 56 Parquet shards totaling 76,188,793,982 bytes
(76.19 GB decimal; 70.96 GiB) and 16,245,599 records. The files contain 1,355
row groups; a shard has between 16 and 49 row groups. The full inventory records
every shard path, byte size, Git blob OID, LFS SHA-256, row-group size, subset
statistics, and exact row-group subset counts in
[`C1_GIGAVERBO_PHYSICAL_INVENTORY.json`](C1_GIGAVERBO_PHYSICAL_INVENTORY.json).

Footer statistics identify a single subset exactly in 539 row groups where
`subset.min == subset.max`. The other 816 row groups have mixed subset values.
For those groups, the audit read only the `subset` column to obtain exact
membership and counts. A follow-up read of the `source` column in groups
containing the three unexpected labels confirmed their source identities. No
document text was read during the inventory. The initial scan made 927 HTTP
range requests and received 62,470,177 bytes (59.58 MiB), including range
read-ahead.

The inventory found 23 distinct subset labels, rather than the 19 seen in the
earlier diagnostic row-group sample. The initial metadata scan took 103.79
seconds. The three additional labels were
`dolly_15k`, `legal_pt`, and `roots`. `dolly_15k` is sourced from
`Gustrd/dolly-15k-libretranslate-pt`; `legal_pt` is sourced from
`eduagarcia/LegalPT_dedup`; and `roots` is sourced from
`bigscience-data/roots_pt_wikiquote`.

## Exclusion policy reconciliation

The accepted residual has 11 subsets. The exclusion YAML now covers the three
additional labels: the existing machine-translation rule matches the physical
`dolly_15k` subset name, and `legal_pt` and `roots` are explicitly excluded
because they are outside the accepted residual and have no provisional A/B/C
weight. The materializer reads this file as its only exclusion policy; subset
names are not duplicated in implementation code.

Exclusion config SHA-256 used for materialization:

`a26de7a35c08af0ee10da18e1cba9f1917ddcefa274bf86bfa9936dbff22bf6e`

| Excluded subset | Rule | Records |
| --- | --- | ---: |
| `bactrianx` | non-commercial license | 5,984 |
| `baixelivros` | primary-source overlap | 177 |
| `bdtd` | primary-source overlap | 889 |
| `corpus_carolina` | primary-source overlap | 22,500 |
| `cosmos_qa` | machine translated | 1 |
| `dolly_15k` | machine translated | 723 |
| `gpt4all` | machine translated synthetic data | 35,064 |
| `legal_pt` | outside accepted residual | 19,970 |
| `roots` | outside accepted residual | 550 |
| `ultrachat` | machine translated | 328,497 |
| `wikipedia` | primary-source overlap | 73,361 |
| `xlsum` | non-commercial license | 1,204 |
| **Total excluded** |  | **488,920** |

The resulting 15,756,679 eligible records are distributed as follows:

| Eligible subset | Records | Estimated files | Estimated persisted size (GiB) |
| --- | ---: | ---: | ---: |
| `finepdfs_por_Latn` | 909,373 | 143 | 13.87 |
| `crawlPT_dedup` | 813,822 | 146 | 3.10 |
| `quati` | 181,372 | 32 | 0.27 |
| `blogset` | 41,411 | 8 | 0.14 |
| `fineweb_2_pt` | 5,881,723 | 760 | 20.87 |
| `mc4_pt` | 2,476,833 | 433 | 9.15 |
| `hplt2_pt` | 4,431,862 | 577 | 16.35 |
| `hplt1_pt` | 415,307 | 62 | 2.73 |
| `common_crawl` | 464,835 | 65 | 2.06 |
| `oscar` | 122,332 | 28 | 0.52 |
| `culturax` | 17,809 | 6 | 0.07 |
| **Total estimate** | **15,756,679** | **2,260** | **69.12** |

Per-subset sizes are estimates: the full compressed row-group bytes are assigned
to a subset when a row group contains only that subset; in mixed row groups,
bytes are apportioned by exact record counts. This does not assume that every
subset has the same text length. The row-share estimate is 74,218,082,737 bytes
(74.22 GB decimal; 69.12 GiB); rewriting at the configured Parquet compression
level can change the measured output size, which the manifest will record.

## Acquisition decision

There are 1,328 row groups containing eligible records: 1,284 contain only
eligible rows and 44 mix eligible and excluded rows. The remaining 27 groups
can be skipped. The acquisition stream reads the Parquet footer, uses exact
`subset` min/max statistics where they identify one value, reads the `subset`
column first for mixed groups, validates the subset column for the 27 fully
excluded groups, then reads all columns only from groups with eligible rows.
Reading those groups requires an estimated 74,748,794,028 compressed bytes
(74.75 GB decimal; 69.62 GiB), plus 1,723 compressed subset-column bytes for
the fully excluded groups, about 98.1% of the row-group compressed data in this
partition.
The filtered output is estimated at 74.22 GB decimal (69.12 GiB), with the
mixed-group apportionment described above. These measurements replace the
previous 25–35 GB raw-size estimate.

The full eligible residual will be materialized. It is a 15.76-million-record
source reservoir used to measure normalized words and deduplication by subset
before candidate mixture weighting. A bounded reservoir would require an
additional sampling rule and a target corpus budget that Gate C1 has not chosen;
neither can be inferred from the provisional A/B/C percentages. The raw output
will remain partitioned by upstream subset, preserve the original fields and
text, and add source-shard, row-group, and pinned-commit provenance. No
normalization, global deduplication, or final mixture construction is part of
this acquisition.

This document records the pre-acquisition physical audit. Production
materialization status is tracked separately in `docs/GATES.md` and remains
pending until the output manifest is complete and verified.
