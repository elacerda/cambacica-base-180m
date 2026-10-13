"""C1-S1: deterministic group splits and reference manifests, never corpus copies.

The policy is a proposal. Production requires a separate scientist approval
record bound to its exact bytes. SQLite bounds memory and checkpoints whole
input shards; replay of an interrupted shard is atomic.
"""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Any, Iterable, Mapping
import unicodedata

import pyarrow as pa
import pyarrow.parquet as pq

from cambacica.corpus.dedup.exact_pipeline import occurrence_id_v2
from cambacica.corpus.manifest import compute_file_sha256

DEFAULT_INPUT = Path("/mnt/data/cambacica-base-180m/deduplicated/exact")
DEFAULT_POLICY = Path("configs/c1_split_policy_proposed.json")
PINNED_INPUT = "57370cd403f571e36172d19ff4310c52c2a3d1937fcdaef5e1462f56dc44d428"
SOURCES = {
    "carolina": "55e63a519393c70a48dcfa14a558499c6bb0583b",
    "wikipedia_pt": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    "parlamento_pt": "08f13e7e63ab9bfbd8c0b40955defe3bb7f68c2b",
    "gutenberg_pt": "snapshot_2026-10-01",
    "gigaverbo_v2": "7058ccf19eaeaf4505a96fc7e5305a01fc441fd8",
}
COLUMNS = [
    "text",
    "source",
    "subset",
    "source_revision",
    "original_id",
    "content_sha256",
    "raw_source_file",
    "raw_record_identifier",
    "domain_category",
]
SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("group_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("subset", pa.string(), nullable=False),
        pa.field("domain_category", pa.string(), nullable=False),
        pa.field("source_shard", pa.string(), nullable=False),
        pa.field("source_row_ordinal", pa.int64(), nullable=False),
        pa.field("content_sha256", pa.string(), nullable=False),
        pa.field("normalized_words", pa.int64(), nullable=False),
        pa.field("is_long", pa.bool_(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
    ]
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(namespace: str, value: Any) -> str:
    return hashlib.sha256((namespace + "\0" + canonical(value)).encode()).hexdigest()


def load_policy(path: Path) -> dict:
    policy = json.loads(path.read_text())
    if (
        policy.get("version") != "c1-s1-v1"
        or policy.get("status") != "proposed_requires_scientist_approval"
        or policy.get("input_manifest_sha256") != PINNED_INPUT
        or policy.get("seed") != 20261013
        or policy.get("ratios_basis_points")
        != {"train": 9800, "validation": 100, "test": 100}
        or policy.get("stratum") != ["source", "subset"]
        or policy.get("group_rules")
        != {
            s: "content_sha256" if s == "parlamento_pt" else "original_id"
            for s in SOURCES
        }
        or policy.get("long_document_words") != 100000
        or policy.get("tiny_strata")
        != "one_group_train;two_groups_train_validation;three_or_more_minimum_one_per_split"
    ):
        raise ValueError(
            "Unsupported split policy; revise implementation and review together"
        )
    return policy


def authorize(policy_path: Path, approval: Path | None) -> dict:
    if approval is None:
        raise ValueError(
            "Full-corpus execution requires explicit scientist approval JSON"
        )
    record = json.loads(approval.read_text())
    if (
        record.get("policy_sha256") != compute_file_sha256(policy_path)
        or record.get("input_manifest_sha256") != PINNED_INPUT
        or record.get("action") != "approve_full_split_manifest_and_verification"
        or not record.get("scientist")
        or not record.get("decision_reference")
    ):
        raise ValueError(
            "Approval must name scientist, decision and exact policy/input identities"
        )
    return record


def group_counts(n: int) -> tuple[int, int, int]:
    """Integer train/validation/test quotas, by unique groups, never words."""
    if n < 1:
        raise ValueError("Empty stratum")
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0
    held = (n + 99) // 100
    return n - 2 * held, held, held


def reference(row: Mapping[str, Any], shard: str, ordinal: int, policy: dict) -> dict:
    for field in COLUMNS:
        if field not in row:
            raise ValueError(f"Missing required metadata: {field}")
    for field in (
        "source",
        "subset",
        "source_revision",
        "original_id",
        "raw_source_file",
        "raw_record_identifier",
        "content_sha256",
    ):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ValueError(f"Missing required metadata: {field}")
    source = row["source"]
    if source not in SOURCES or row["source_revision"] != SOURCES[source]:
        raise ValueError("Unknown source or unpinned source revision")
    if not isinstance(row["text"], str):
        raise ValueError("Missing text")
    if hashlib.sha256(row["text"].encode()).hexdigest() != row["content_sha256"]:
        raise ValueError("Text checksum mismatch")
    words = len(row["text"].split())
    key = row[policy["group_rules"][source]]
    # URLs are deliberately not used: retained URLs often name entire datasets.
    group = digest("cambacica-s1-group-v1", [source, row["subset"], key])
    return dict(
        doc_id=occurrence_id_v2(source, shard, ordinal),
        group_id=group,
        source=source,
        subset=row["subset"],
        domain_category=row["domain_category"] or "unknown",
        source_shard=shard,
        source_row_ordinal=ordinal,
        content_sha256=row["content_sha256"],
        normalized_words=words,
        is_long=words >= policy["long_document_words"],
    )


def atomic_text(path: Path, text: str) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    partial.replace(path)


def rank_groups(connection: sqlite3.Connection, policy: dict) -> None:
    """Rank with population-wide strata; ties resolve by the group digest."""
    connection.executescript("""
        DROP TABLE IF EXISTS groups;
        CREATE TABLE groups AS
          SELECT group_id, source, subset, '' AS rank_hash, '' AS split
          FROM refs GROUP BY group_id, source, subset;
        CREATE UNIQUE INDEX group_identity ON groups(group_id);
    """)
    for (group,) in connection.execute("SELECT group_id FROM groups"):
        rank = digest(
            "cambacica-s1-rank-v1",
            [policy["input_manifest_sha256"], policy["seed"], group],
        )
        connection.execute(
            "UPDATE groups SET rank_hash=? WHERE group_id=?", (rank, group)
        )
    connection.execute(
        "CREATE INDEX group_rank ON groups(source,subset,rank_hash,group_id)"
    )
    for source, subset, n in connection.execute(
        "SELECT source,subset,count(*) FROM groups GROUP BY source,subset"
    ).fetchall():
        train, val, test = group_counts(n)
        for rank, (group,) in enumerate(
            connection.execute(
                "SELECT group_id FROM groups WHERE source=? AND subset=? ORDER BY rank_hash,group_id",
                (source, subset),
            )
        ):
            split = (
                "test"
                if rank < test
                else "validation"
                if rank < test + val
                else "train"
            )
            connection.execute(
                "UPDATE groups SET split=? WHERE group_id=?", (split, group)
            )
        assert train + val + test == n
    connection.commit()


def capacity_report(
    accounting: Iterable[Mapping[str, Any]], configs: list[dict]
) -> dict:
    """Exact rational no-reuse ceilings; input capacities must be words."""
    capacities = {
        (r["source"], r["subset"]): int(r["normalized_words_after"])
        for r in accounting
        if r["scope"] in ("source_total", "source_subset")
    }
    result = {}
    for config in configs:
        constraints = []
        for source, spec in config["sources"].items():
            share = Fraction(str(spec["share"]))
            if not share:
                continue
            if source == "gigaverbo_v2_residual":
                parts = [
                    ("gigaverbo_v2", subset, share * Fraction(str(weight)))
                    for subset, weight in spec["subsets"].items()
                    if weight
                ]
            else:
                parts = [(source, "", share)]
            for family, subset, weight in parts:
                capacity = capacities[(family, subset)]
                ceiling = Fraction(capacity) / weight
                constraints.append(
                    dict(
                        source=family,
                        subset=subset,
                        words=capacity,
                        total_share=str(weight),
                        max_total_words_floor=math.floor(ceiling),
                    )
                )
        limiting = min(constraints, key=lambda r: r["max_total_words_floor"])
        result[config["name"]] = dict(limiting=limiting, constraints=constraints)
    return result


def selected_files(manifest: dict, bounded: bool) -> list[str]:
    paths = sorted(p for p in manifest["output_files"] if p.startswith("data/"))
    if not bounded:
        return paths
    # All 32 Carolina shards expose rare taxonomies. First file per other
    # source/subset; all 655 Gutenberg books. Selection is not prevalence-weighted.
    chosen, seen = [], set()
    for path in paths:
        key = str(Path(path).parent)
        if path.startswith("data/carolina/") or key not in seen:
            chosen.append(path)
        seen.add(key)
    return chosen


def rows_from_file(path: Path, bounded: bool, batch_size: int, reverse: bool):
    parquet = pq.ParquetFile(path)
    if set(COLUMNS) - set(parquet.schema_arrow.names):
        raise ValueError(f"Missing required columns in {path}")
    starts, start = [], 0
    for i in range(parquet.num_row_groups):
        starts.append(start)
        start += parquet.metadata.row_group(i).num_rows
    groups = list(range(parquet.num_row_groups))
    books = path.parent.name == "gutenberg_pt"
    if bounded and not books:
        groups = sorted({0, len(groups) // 2, len(groups) - 1})
        subset_column = parquet.schema_arrow.get_field_index("subset")
        # Mixed-subset row groups expose taxonomy boundaries and rare strata.
        # Inspect only footer statistics and these small projected columns.
        for i in range(parquet.num_row_groups):
            stats = parquet.metadata.row_group(i).column(subset_column).statistics
            if stats is not None and stats.has_min_max and stats.min != stats.max:
                groups.append(i)
        groups = sorted(set(groups))
    if reverse:
        groups.reverse()
    for group in groups:
        offset = 0
        targets = None
        if bounded and not books:
            targets = set(range(min(16, parquet.metadata.row_group(group).num_rows)))
            subsets = (
                parquet.read_row_group(group, columns=["subset"]).column(0).to_pylist()
            )
            counts = {}
            for index, subset in enumerate(subsets):
                counts[subset] = counts.get(subset, 0) + 1
                if counts[subset] <= 16:
                    targets.add(index)
        for batch in parquet.iter_batches(
            batch_size=batch_size, row_groups=[group], columns=COLUMNS
        ):
            for row in batch.to_pylist():
                if targets is None or offset in targets:
                    yield starts[group] + offset, row
                offset += 1
            if targets is not None and offset > max(targets):
                break


def build(
    input_root: Path,
    output: Path,
    policy_path: Path = DEFAULT_POLICY,
    *,
    bounded: bool = True,
    approval: Path | None = None,
    batch_size: int = 8,
    reverse: bool = False,
) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    policy = load_policy(policy_path)
    authorization = authorize(policy_path, approval) if not bounded else None
    if (
        output.resolve() == input_root.resolve()
        or input_root.resolve() in output.resolve().parents
    ):
        raise ValueError("Output must be outside immutable input")
    input_sha = compute_file_sha256(input_root / "manifest.json")
    if input_sha != policy["input_manifest_sha256"]:
        raise ValueError("Pinned input manifest mismatch")
    manifest = json.loads((input_root / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    state = dict(
        version=policy["version"],
        policy_sha256=compute_file_sha256(policy_path),
        input_manifest_sha256=input_sha,
        bounded=bounded,
        approval=authorization,
        implementation_sha256=compute_file_sha256(Path(__file__)),
        occurrence_implementation_sha256=compute_file_sha256(
            Path(__file__).parent / "dedup" / "exact_pipeline.py"
        ),
        runtime={
            "python": sys.version,
            "unicode": unicodedata.unidata_version,
            "pyarrow": pa.__version__,
        },
    )
    state_path = output / "state.json"
    if state_path.exists() and json.loads(state_path.read_text()) != state:
        raise ValueError("Restart identity mismatch")
    if not state_path.exists():
        atomic_text(state_path, canonical(state) + "\n")
    connection = sqlite3.connect(output / "index.sqlite")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS refs (
          doc_id TEXT PRIMARY KEY, group_id TEXT, source TEXT, subset TEXT,
          domain_category TEXT, source_shard TEXT, source_row_ordinal INTEGER,
          content_sha256 TEXT, normalized_words INTEGER, is_long INTEGER,
          UNIQUE(source_shard,source_row_ordinal));
        CREATE TABLE IF NOT EXISTS completed (path TEXT PRIMARY KEY);
    """)
    paths = selected_files(manifest, bounded)
    if reverse:
        paths.reverse()
    try:
        for relative in paths:
            path = input_root / relative
            entry = manifest["output_files"][relative]
            if path.stat().st_size != entry["bytes"]:
                raise ValueError("Input shard size changed")
            if not bounded and compute_file_sha256(path) != entry["sha256"]:
                raise ValueError("Input shard checksum mismatch")
            if connection.execute(
                "SELECT 1 FROM completed WHERE path=?", (relative,)
            ).fetchone():
                continue
            with connection:
                before_stat = path.stat()
                count = 0
                for ordinal, row in rows_from_file(path, bounded, batch_size, reverse):
                    if row["source"] != Path(relative).parts[1]:
                        raise ValueError("Source/path mismatch")
                    ref = reference(
                        row, str(Path(relative).relative_to("data")), ordinal, policy
                    )
                    connection.execute(
                        "INSERT INTO refs VALUES (?,?,?,?,?,?,?,?,?,?)",
                        tuple(ref.values()),
                    )
                    count += 1
                if not bounded and count != entry["rows"]:
                    raise ValueError("Input row count mismatch")
                after_stat = path.stat()
                if (before_stat.st_size, before_stat.st_mtime_ns) != (
                    after_stat.st_size,
                    after_stat.st_mtime_ns,
                ):
                    raise ValueError("Input shard changed during read")
                connection.execute("INSERT INTO completed VALUES (?)", (relative,))
        if not bounded:
            count = connection.execute("SELECT count(*) FROM refs").fetchone()[0]
            if count != manifest["retained_record_count"]:
                raise ValueError("Full-corpus count mismatch")
            accounting_path = input_root / "accounting_by_source_subset.csv"
            if (
                compute_file_sha256(accounting_path)
                != manifest["output_files"][accounting_path.name]["sha256"]
            ):
                raise ValueError("D1 accounting checksum mismatch")
            actual = {
                (source, subset): (documents, words)
                for source, subset, documents, words in connection.execute(
                    "SELECT source,subset,count(*),sum(normalized_words) FROM refs GROUP BY source,subset"
                )
            }
            expected = {}
            with accounting_path.open() as stream:
                for row in csv.DictReader(stream):
                    if row["scope"] == "source_subset":
                        expected[row["source"], row["subset"]] = (
                            int(row["documents_after"]),
                            int(row["normalized_words_after"]),
                        )
                    elif row["scope"] == "source_total":
                        totals = connection.execute(
                            "SELECT count(*),sum(normalized_words) FROM refs WHERE source=?",
                            (row["source"],),
                        ).fetchone()
                        if totals != (
                            int(row["documents_after"]),
                            int(row["normalized_words_after"]),
                        ):
                            raise ValueError("D1 source accounting mismatch")
            if actual != expected:
                raise ValueError("D1 subset accounting mismatch")
        rank_groups(connection, policy)
        query = """SELECT r.*,g.split FROM refs r JOIN groups g USING(group_id)
                   ORDER BY r.doc_id"""
        partial = output / "assignments.parquet.partial"
        with pq.ParquetWriter(
            partial, SCHEMA, compression="zstd", version="2.6"
        ) as writer:
            cursor = connection.execute(query)
            while rows := cursor.fetchmany(4096):
                records = [dict(zip(SCHEMA.names, row)) for row in rows]
                for record in records:
                    record["is_long"] = bool(record["is_long"])
                writer.write_table(
                    pa.Table.from_pylist(records, schema=SCHEMA), row_group_size=4096
                )
        partial.replace(output / "assignments.parquet")
        accounting = [
            dict(
                zip(
                    (
                        "source",
                        "subset",
                        "domain",
                        "split",
                        "documents",
                        "normalized_words",
                        "long_documents",
                    ),
                    r,
                )
            )
            for r in connection.execute("""
                        SELECT r.source,r.subset,r.domain_category,g.split,count(*),
                               sum(r.normalized_words),sum(r.is_long)
                        FROM refs r JOIN groups g USING(group_id)
                        GROUP BY r.source,r.subset,r.domain_category,g.split
                        ORDER BY r.source,r.subset,r.domain_category,g.split
                      """)
        ]
        group_accounting = [
            dict(zip(("source", "subset", "split", "groups"), r))
            for r in connection.execute("""
                              SELECT source,subset,split,count(*) FROM groups
                              GROUP BY source,subset,split ORDER BY source,subset,split
                            """)
        ]
        result = dict(
            **state,
            status="BOUNDED_BUILT_PROPOSAL"
            if bounded
            else "SPLIT_MANIFEST_BUILT_PENDING_VERIFICATION",
            input_files={p: manifest["output_files"][p] for p in sorted(paths)},
            input_shard_checksum_verification="size_and_selected_text_only"
            if bounded
            else "all_payload_sha256",
            documents=sum(r["documents"] for r in accounting),
            accounting=accounting,
            group_accounting=group_accounting,
            policy=policy,
            output_sha256=compute_file_sha256(output / "assignments.parquet"),
        )
        atomic_text(output / "manifest.json", canonical(result) + "\n")
        checksum = compute_file_sha256(output / "manifest.json")
        atomic_text(
            output / "checksums.sha256",
            f"{checksum}  manifest.json\n{result['output_sha256']}  assignments.parquet\n",
        )
        return result
    finally:
        connection.close()


def verify(
    input_root: Path,
    output: Path,
    policy: Path,
    *,
    bounded: bool,
    approval: Path | None = None,
) -> dict:
    """Independent reconstruction from source, changed order and batch size."""
    with tempfile.TemporaryDirectory(
        prefix="c1-s1-verify-", dir=output.parent
    ) as temporary:
        rebuilt = build(
            input_root,
            Path(temporary),
            policy,
            bounded=bounded,
            approval=approval,
            batch_size=17,
            reverse=True,
        )
        original = json.loads((output / "manifest.json").read_text())
        if (
            rebuilt != original
            or compute_file_sha256(output / "assignments.parquet")
            != rebuilt["output_sha256"]
        ):
            raise ValueError("Independent reconstruction mismatch")
        checksums = (output / "checksums.sha256").read_text()
        expected = (
            f"{compute_file_sha256(output / 'manifest.json')}  manifest.json\n"
            f"{rebuilt['output_sha256']}  assignments.parquet\n"
        )
        if checksums != expected:
            raise ValueError("Checksum manifest mismatch")
    return dict(
        status="PASS",
        documents=original["documents"],
        output_sha256=original["output_sha256"],
        reordered=True,
        batch_size=17,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Requires scientist approval; default is bounded",
    )
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.action == "build":
        result = build(
            args.input_root,
            args.output_root,
            args.policy,
            bounded=not args.full,
            approval=args.approval,
            batch_size=args.batch_size,
        )
    else:
        result = verify(
            args.input_root,
            args.output_root,
            args.policy,
            bounded=not args.full,
            approval=args.approval,
        )
    print(canonical({k: result[k] for k in ("status", "documents", "output_sha256")}))


if __name__ == "__main__":
    main()
