"""Deterministic post-normalization source-pool characterization for Gate C1."""

from __future__ import annotations

from collections import Counter
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

import pyarrow.parquet as pq

from cambacica.corpus.manifest import compute_file_sha256
from cambacica.corpus.mix import validate_mix_files
from cambacica.corpus.normalization import (
    DEFAULT_NORMALIZED_ROOT,
    NORMALIZED_SCHEMA,
    SOURCE_CONFIG,
    _normalized_row_errors,
    count_normalized_words,
    verify_normalized_source,
)


_METADATA_FIELDS = (
    "source",
    "source_revision",
    "subset",
    "original_id",
    "original_url",
    "license",
    "language",
    "language_score",
    "variety",
    "quality_score",
    "publication_date",
    "domain_category",
    "title",
    "raw_source_file",
    "raw_record_identifier",
    "normalization_version",
    "_gv2_upstream_shard",
    "_gv2_upstream_row_group",
    "_gv2_upstream_commit",
    "upstream_metadata_json",
    "content_sha256",
)
_CRITICAL_METADATA_FIELDS = (
    "source",
    "source_revision",
    "subset",
    "original_id",
    "raw_source_file",
    "raw_record_identifier",
    "normalization_version",
    "content_sha256",
)


class _Aggregate:
    def __init__(self) -> None:
        self.documents = 0
        self.normalized_bytes = 0
        self.normalized_characters = 0
        self.normalized_words = 0
        self.word_length_counts: Counter[int] = Counter()
        self.empty_documents = 0
        self.under_20_words = 0
        self.under_100_words = 0
        self.at_least_100k_words = 0
        self.at_least_1m_words = 0
        self.at_least_1m_characters = 0
        self.at_least_10m_characters = 0
        self.documents_missing_critical_metadata = 0
        self.metadata_present: Counter[str] = Counter()

    def add(self, row: Mapping[str, Any]) -> None:
        text = row["text"] or ""
        words = count_normalized_words(text)
        characters = len(text)
        byte_count = len(text.encode("utf-8"))
        self.documents += 1
        self.normalized_bytes += byte_count
        self.normalized_characters += characters
        self.normalized_words += words
        self.word_length_counts[words] += 1
        self.empty_documents += int(text == "")
        self.under_20_words += int(words < 20)
        self.under_100_words += int(words < 100)
        self.at_least_100k_words += int(words >= 100_000)
        self.at_least_1m_words += int(words >= 1_000_000)
        self.at_least_1m_characters += int(characters >= 1_000_000)
        self.at_least_10m_characters += int(characters >= 10_000_000)
        self.documents_missing_critical_metadata += int(
            any(
                row.get(field) is None or row.get(field) == ""
                for field in _CRITICAL_METADATA_FIELDS
            )
        )
        for field in _METADATA_FIELDS:
            value = row.get(field)
            if value is not None and value != "":
                self.metadata_present[field] += 1

    @staticmethod
    def _nearest_rank(counts: Counter[int], n: int, p: float) -> int:
        target_rank = max(1, math.ceil(p * n))
        cumulative = 0
        for value in sorted(counts):
            cumulative += counts[value]
            if cumulative >= target_rank:
                return value
        return 0

    def to_dict(self, normalization_failures: int = 0) -> dict[str, Any]:
        n = self.documents
        counts = self.word_length_counts
        return {
            "document_count": n,
            "normalized_bytes": self.normalized_bytes,
            "normalized_characters": self.normalized_characters,
            "normalized_words": self.normalized_words,
            "word_length_distribution": {
                "mean": round(self.normalized_words / n, 2) if n else 0.0,
                "median": self._nearest_rank(counts, n, 0.50),
                "p90": self._nearest_rank(counts, n, 0.90),
                "p95": self._nearest_rank(counts, n, 0.95),
                "p99": self._nearest_rank(counts, n, 0.99),
                "max": max(counts, default=0),
                "percentile_rule": "nearest_rank: sorted_values[ceil(p*n)-1]",
            },
            "document_length_counts": {
                "empty_normalized_documents": self.empty_documents,
                "under_20_words": self.under_20_words,
                "under_100_words": self.under_100_words,
                "at_least_100000_words": self.at_least_100k_words,
                "at_least_1000000_words": self.at_least_1m_words,
                "at_least_1000000_characters": self.at_least_1m_characters,
                "at_least_10000000_characters": self.at_least_10m_characters,
            },
            "normalization_failures": normalization_failures,
            "documents_missing_critical_metadata": self.documents_missing_critical_metadata,
            "critical_metadata_fields": list(_CRITICAL_METADATA_FIELDS),
            "metadata_coverage": {
                field: {
                    "present_documents": self.metadata_present[field],
                    "coverage_fraction": round(self.metadata_present[field] / n, 8)
                    if n
                    else 0.0,
                }
                for field in _METADATA_FIELDS
            },
        }


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        import os

        os.fsync(stream.fileno())
    import os

    os.replace(temporary, path)


def _characterize_source(
    source: str,
    normalized_root: Path,
    *,
    verify_hashes: bool,
) -> tuple[dict[str, Any], dict[str | None, _Aggregate]]:
    source_dir = normalized_root / SOURCE_CONFIG[source]["output_dir"]
    manifest_path = source_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"COMPLETE", "PARTIAL"}:
        raise ValueError(
            f"Cannot characterize source {source} with status {manifest.get('status')}"
        )
    valid, errors = verify_normalized_source(
        source,
        output_root=normalized_root,
        raw_root=Path(manifest["raw_source_manifest"]).parents[1],
        hash_files=verify_hashes,
        allow_partial=True,
        verify_rows=False,
    )
    if not valid:
        raise ValueError(
            f"Normalized source verification failed for {source}: "
            + "; ".join(errors[:8])
        )

    scopes: dict[str | None, _Aggregate] = {None: _Aggregate()}
    for file_record in manifest.get("normalized_files", []):
        relative_path = file_record["relative_path"]
        parquet_path = source_dir / relative_path
        parquet_file = pq.ParquetFile(parquet_path)
        if parquet_file.schema_arrow.remove_metadata() != NORMALIZED_SCHEMA:
            raise ValueError(f"Normalized Parquet schema mismatch: {parquet_path}")
        file_row_count = 0
        file_bytes = 0
        file_characters = 0
        file_words = 0
        for batch in parquet_file.iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                text = row["text"] or ""
                row_errors = _normalized_row_errors(row, source)
                if row_errors:
                    raise ValueError(
                        f"Invalid normalized row at {row['raw_source_file']}:{row['raw_record_identifier']}: "
                        + "; ".join(row_errors)
                    )
                source_scope = scopes[None]
                source_scope.add(row)
                subset = row.get("subset")
                if subset is not None:
                    scopes.setdefault(str(subset), _Aggregate()).add(row)
                file_row_count += 1
                file_bytes += len(text.encode("utf-8"))
                file_characters += len(text)
                file_words += count_normalized_words(text)
        if file_row_count != int(file_record["documents"]):
            raise ValueError(
                f"Normalized row count mismatch while characterizing: {parquet_path}"
            )
        for observed, field in (
            (file_bytes, "normalized_bytes"),
            (file_characters, "normalized_characters"),
            (file_words, "normalized_words"),
        ):
            if observed != int(file_record[field]):
                raise ValueError(
                    f"Normalized per-file {field} mismatch for {parquet_path}: "
                    f"{observed} != {file_record[field]}"
                )
    top_level = scopes[None].to_dict(
        int(manifest.get("normalization_failure_count", 0))
    )
    for key, expected_key in (
        ("document_count", "output_document_count"),
        ("normalized_bytes", "total_normalized_bytes"),
        ("normalized_characters", "total_normalized_characters"),
        ("normalized_words", "total_normalized_words"),
    ):
        if top_level[key] != manifest.get(expected_key):
            raise ValueError(
                f"Characterization {key} does not match normalized manifest for {source}."
            )
    subset_reports = {
        subset: aggregate.to_dict(
            sum(
                1
                for failure in manifest.get("failures", [])
                if failure.get("raw_source_file")
                and f"subset={quote(str(subset), safe='._-')}"
                in failure["raw_source_file"]
            )
        )
        for subset, aggregate in sorted(scopes.items(), key=lambda item: str(item[0]))
        if subset is not None
    }
    report = {
        "schema_version": 1,
        "source": source,
        "normalization_version": manifest.get("normalization_schema_version"),
        "raw_source_manifest_sha256": manifest.get("raw_source_manifest_sha256"),
        "normalized_manifest_sha256": compute_file_sha256(manifest_path),
        "normalization_status": manifest.get("status"),
        "metrics": top_level,
        "subsets": subset_reports,
    }
    return report, scopes


def _mix_capacities(
    reports: Mapping[str, dict[str, Any]],
    target_words: int | None,
    mix_paths: list[Path],
) -> dict[str, Any]:
    normalized_configs = validate_mix_files(mix_paths)
    source_key_map = {
        "carolina": "carolina",
        "wikipedia_pt": "wikipedia_pt",
        "parlamento_pt": "parlamento_pt",
        "gutenberg_pt": "gutenberg_pt",
        "gigaverbo_v2_residual": "gigaverbo_v2",
    }
    capacities: dict[str, Any] = {}

    def oversampling_factor(required: int | None, available: int) -> float | None:
        if required is None:
            return None
        if available == 0:
            return None
        return round(required / available, 8)

    for config, path in zip(normalized_configs, mix_paths):
        component_capacities: list[tuple[float, str, int, float]] = []
        component_report: dict[str, Any] = {}
        for configured_source, source_config in config["sources"].items():
            source = source_key_map[configured_source]
            total_available = int(reports[source]["metrics"]["normalized_words"])
            share = float(source_config["share"])
            if share <= 0:
                continue
            required_words = target_words * share if target_words is not None else None
            source_capacity = math.floor(total_available / share)
            component_capacities.append(
                (source_capacity, source, total_available, share)
            )
            component_report[source] = {
                "available_normalized_words": total_available,
                "candidate_share": share,
                "source_capacity_as_total_mix_words": source_capacity,
                "required_normalized_words": required_words,
                "source_oversampling_factor": oversampling_factor(
                    required_words, total_available
                ),
                "source_oversampling_required": (
                    total_available < required_words
                    if required_words is not None
                    else None
                ),
            }
            if source == "gigaverbo_v2":
                subset_config = source_config.get("subsets", {})
                for subset, weight in sorted(subset_config.items()):
                    available = int(
                        reports[source]["subsets"]
                        .get(subset, {})
                        .get("normalized_words", 0)
                    )
                    subset_share = share * float(weight)
                    if subset_share <= 0:
                        continue
                    capacity = math.floor(available / subset_share)
                    needed = (
                        target_words * subset_share
                        if target_words is not None
                        else None
                    )
                    component_capacities.append(
                        (capacity, f"{source}/{subset}", available, subset_share)
                    )
                    component_report.setdefault("subsets", {})[subset] = {
                        "available_normalized_words": available,
                        "share_of_total_mix": subset_share,
                        "subset_capacity_as_total_mix_words": capacity,
                        "required_normalized_words": needed,
                        "oversampling_factor": oversampling_factor(needed, available),
                        "oversampling_required": (
                            available < needed if needed is not None else None
                        ),
                    }
        bottleneck = min(component_capacities, key=lambda item: item[0])
        max_words = bottleneck[0]
        oversampling = (
            any(item[2] < target_words * item[3] for item in component_capacities)
            if target_words is not None
            else None
        )
        capacities[str(config["name"])] = {
            "config_path": str(path),
            "target_normalized_words": target_words,
            "max_non_oversampled_total_normalized_words": max_words,
            "limiting_component": bottleneck[1],
            "oversampling_required_at_target": oversampling,
            "feasibility_status": (
                "target_requires_oversampling"
                if oversampling
                else "target_feasible_without_oversampling"
                if oversampling is False
                else "maximum_capacity_reported_target_unspecified"
            ),
            "components": component_report,
        }
    return capacities


def characterize_sources(
    *,
    normalized_root: Path | str = DEFAULT_NORMALIZED_ROOT,
    target_words: int | None = None,
    verify_hashes: bool = True,
    mix_paths: list[Path | str] | None = None,
) -> dict[str, Any]:
    """Write deterministic source metrics and normalized-word accounting."""
    if target_words is not None and target_words <= 0:
        raise ValueError("target_words must be positive when provided.")
    root = Path(normalized_root)
    reports: dict[str, dict[str, Any]] = {}
    scopes_by_source: dict[str, dict[str | None, _Aggregate]] = {}
    for source in SOURCE_CONFIG:
        report, scopes = _characterize_source(source, root, verify_hashes=verify_hashes)
        reports[source] = report
        scopes_by_source[source] = scopes

    total_words = sum(
        int(reports[source]["metrics"]["normalized_words"]) for source in reports
    )
    accounting_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for source in sorted(reports):
        source_metrics = reports[source]["metrics"]
        source_rows.append(
            {
                "source": source,
                "documents": source_metrics["document_count"],
                "normalized_words": source_metrics["normalized_words"],
                "fraction_of_total_normalized_words": (
                    round(source_metrics["normalized_words"] / total_words, 12)
                    if total_words
                    else 0.0
                ),
            }
        )
        scopes = scopes_by_source[source]
        subsets = [subset for subset in scopes if subset is not None]
        if subsets:
            for subset in sorted(subsets):
                metrics = scopes[subset].to_dict(0)
                accounting_rows.append(
                    {
                        "source": source,
                        "subset": subset,
                        "documents": metrics["document_count"],
                        "normalized_words": metrics["normalized_words"],
                        "fraction_of_total_normalized_words": (
                            round(metrics["normalized_words"] / total_words, 12)
                            if total_words
                            else 0.0
                        ),
                    }
                )
        else:
            accounting_rows.append(
                {
                    "source": source,
                    "subset": "",
                    "documents": source_metrics["document_count"],
                    "normalized_words": source_metrics["normalized_words"],
                    "fraction_of_total_normalized_words": (
                        round(source_metrics["normalized_words"] / total_words, 12)
                        if total_words
                        else 0.0
                    ),
                }
            )
    if mix_paths is None:
        project_root = Path(__file__).resolve().parents[3]
        mix_paths = [
            project_root / f"configs/corpus_mix_{letter}.yaml" for letter in "abc"
        ]
    mix_paths = [Path(path) for path in mix_paths]
    mix_feasibility = _mix_capacities(reports, target_words, mix_paths)
    incomplete_sources = sorted(
        source
        for source, report in reports.items()
        if report["normalization_status"] != "COMPLETE"
    )
    if incomplete_sources:
        for mix_report in mix_feasibility.values():
            mix_report["source_pool_status"] = "incomplete"
            mix_report["incomplete_sources"] = incomplete_sources
            mix_report["capacity_interpretation"] = (
                "lower bound from successfully normalized gross words before "
                "deduplication; oversampling feasibility cannot be established"
            )
            mix_report["oversampling_required_at_target"] = None
            mix_report["feasibility_status"] = "not_assessable_incomplete_source_pool"
    else:
        for mix_report in mix_feasibility.values():
            mix_report["source_pool_status"] = "complete"
            mix_report["capacity_interpretation"] = (
                "gross normalized word volume before global deduplication; "
                "post-deduplication capacity is not estimated"
            )

    characterization_dir = root / "characterization"
    for source, report in reports.items():
        _atomic_text(
            characterization_dir / "sources" / f"{source}.json",
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
    summary = {
        "schema_version": 1,
        "normalization_version": "1.0.0",
        "total_normalized_words": total_words,
        "source_count": len(reports),
        "incomplete_sources": incomplete_sources,
        "source_reports": reports,
        "candidate_mix_feasibility": mix_feasibility,
        "target_normalized_words": target_words,
    }
    _atomic_text(
        characterization_dir / "characterization.json",
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    for name, rows, columns in (
        (
            "normalized_word_accounting.csv",
            accounting_rows,
            [
                "source",
                "subset",
                "documents",
                "normalized_words",
                "fraction_of_total_normalized_words",
            ],
        ),
        (
            "normalized_word_accounting_sources.csv",
            source_rows,
            [
                "source",
                "documents",
                "normalized_words",
                "fraction_of_total_normalized_words",
            ],
        ),
    ):
        import io

        content = io.StringIO(newline="")
        writer = csv.DictWriter(content, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        _atomic_text(characterization_dir / name, content.getvalue())
    return summary


__all__ = ["characterize_sources"]
