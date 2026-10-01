"""Sample inspection metrics calculation and reporting for Gate C1.

Computes comprehensive character, word, length percentile, deduplication,
alphabetic ratio, and metadata distribution diagnostics over normalized sample
Parquet files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import pyarrow as pa

from cambacica.corpus.schema import DOCUMENT_FIELDS, load_sample_parquet


@dataclass
class DistributionStats:
    """Summary statistics for numeric metrics."""

    min: float
    median: float
    p90: float
    p99: float
    max: float
    mean: float


@dataclass
class SampleReport:
    """Complete diagnostic report for a corpus sample."""

    file_path: str
    document_count: int
    total_characters: int
    total_words: int
    char_length_stats: DistributionStats
    word_length_stats: DistributionStats
    alphabetic_ratio: float
    empty_or_near_empty_count: int
    empty_or_near_empty_ratio: float
    exact_duplicate_count: int
    exact_duplicate_rate: float
    mean_duplicate_line_ratio: float
    high_repeated_line_doc_count: int
    high_repeated_line_doc_ratio: float
    missing_field_rates: Dict[str, float]
    source_distribution: Dict[str, int]
    subset_distribution: Dict[str, int]
    domain_distribution: Dict[str, int]
    language_distribution: Dict[str, int]
    variety_distribution: Dict[str, int]
    quality_score_stats: Optional[Dict[str, float]] = None
    publication_date_distribution: Dict[str, int] = field(default_factory=dict)
    # Provenance fields loaded from co-located manifest (if present)
    sample_mode: Optional[str] = None
    sampling_frame: Optional[str] = None
    records_examined: Optional[int] = None
    requested_size: Optional[int] = None
    underfill_status: Optional[str] = None
    gigaverbo_exclusion_stats: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to serializable dictionary.

        Returns
        -------
        dict
            Nested dictionary of all report metrics.
        """
        return asdict(self)


def _compute_percentiles(values: List[float | int]) -> DistributionStats:
    """Compute min, median, p90, p99, max, and mean of a sequence.

    Parameters
    ----------
    values : list of float or int
        Non-empty list of numeric values.

    Returns
    -------
    DistributionStats
        Computed percentile statistics.
    """
    if not values:
        return DistributionStats(0, 0, 0, 0, 0, 0)
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def _p(p: float) -> float:
        idx = int(math.ceil(p * n)) - 1
        return float(sorted_vals[max(0, min(idx, n - 1))])

    mean_val = float(sum(sorted_vals) / n)
    return DistributionStats(
        min=float(sorted_vals[0]),
        median=_p(0.50),
        p90=_p(0.90),
        p99=_p(0.99),
        max=float(sorted_vals[-1]),
        mean=round(mean_val, 2),
    )


def _load_manifest_for(parquet_path: Path) -> Optional[dict]:
    """Attempt to load the co-located manifest JSON for a Parquet sample.

    Parameters
    ----------
    parquet_path : Path
        Parquet file path.

    Returns
    -------
    dict or None
        Parsed manifest dictionary, or None if not found.
    """
    stem = parquet_path.stem  # e.g. 'representative', 'audit'
    manifest_path = parquet_path.parent / f"manifest_{stem}.json"
    if manifest_path.is_file():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def compute_sample_metrics(table: pa.Table, file_path: str = "") -> SampleReport:
    """Compute full diagnostic inspection metrics on a sample PyArrow table.

    Parameters
    ----------
    table : pyarrow.Table
        Table containing rows formatted according to PARQUET_SCHEMA.
    file_path : str, default ""
        Originating file path for reporting purposes.

    Returns
    -------
    SampleReport
        Structured diagnostic report.
    """
    doc_count = len(table)
    if doc_count == 0:
        empty_stats = DistributionStats(0, 0, 0, 0, 0, 0)
        return SampleReport(
            file_path=file_path,
            document_count=0,
            total_characters=0,
            total_words=0,
            char_length_stats=empty_stats,
            word_length_stats=empty_stats,
            alphabetic_ratio=0.0,
            empty_or_near_empty_count=0,
            empty_or_near_empty_ratio=0.0,
            exact_duplicate_count=0,
            exact_duplicate_rate=0.0,
            mean_duplicate_line_ratio=0.0,
            high_repeated_line_doc_count=0,
            high_repeated_line_doc_ratio=0.0,
            missing_field_rates={},
            source_distribution={},
            subset_distribution={},
            domain_distribution={},
            language_distribution={},
            variety_distribution={},
        )

    pydict = table.to_pydict()
    texts = pydict.get("text", [])
    hashes = pydict.get("content_sha256", [])

    char_lengths: List[int] = []
    word_lengths: List[int] = []
    total_alpha_chars = 0
    total_chars = 0
    near_empty_count = 0
    duplicate_line_ratios: List[float] = []
    high_repeated_docs = 0

    for text in texts:
        t = text or ""
        c_len = len(t)
        w_len = len(t.split())
        char_lengths.append(c_len)
        word_lengths.append(w_len)
        total_chars += c_len

        alpha_c = sum(1 for c in t if c.isalpha())
        total_alpha_chars += alpha_c

        if c_len < 100 or w_len < 20:
            near_empty_count += 1

        # Repeated lines diagnostic
        lines = [line.strip() for line in t.splitlines() if line.strip()]
        if lines:
            unique_lines = set(lines)
            dup_ratio = 1.0 - (len(unique_lines) / len(lines))
            duplicate_line_ratios.append(dup_ratio)
            if dup_ratio >= 0.20:
                high_repeated_docs += 1
        else:
            duplicate_line_ratios.append(0.0)

    # Exact duplicate rate via content_sha256
    seen_hashes: set = set()
    dup_count = 0
    for h in hashes:
        if h in seen_hashes:
            dup_count += 1
        else:
            seen_hashes.add(h)

    # Missing field rates
    missing_rates: Dict[str, float] = {}
    for field_name in DOCUMENT_FIELDS:
        col_vals = pydict.get(field_name, [])
        missing_c = sum(
            1 for v in col_vals if v is None or v == "" or str(v).lower() == "none"
        )
        missing_rates[field_name] = round(missing_c / doc_count, 4)

    # Categorical distributions (full, not capped)
    def _dist(field_name: str) -> Dict[str, int]:
        d: Dict[str, int] = {}
        for val in pydict.get(field_name, []):
            k = str(val) if val is not None else "(null)"
            d[k] = d.get(k, 0) + 1
        return dict(sorted(d.items(), key=lambda item: item[1], reverse=True))

    source_dist = _dist("source")
    subset_dist = _dist("subset")
    domain_dist = _dist("domain_category")
    lang_dist = _dist("language")
    variety_dist = _dist("variety")

    # Quality score stats if available
    quality_scores = [q for q in pydict.get("quality_score", []) if q is not None]
    qual_stats = None
    if quality_scores:
        s = _compute_percentiles(quality_scores)
        qual_stats = {
            "count": len(quality_scores),
            "min": s.min,
            "mean": s.mean,
            "median": s.median,
            "p90": s.p90,
            "max": s.max,
        }

    # Publication date distribution (top 10)
    pub_dates = [d for d in pydict.get("publication_date", []) if d is not None]
    pub_dist: Dict[str, int] = {}
    for d in pub_dates:
        pub_dist[str(d)] = pub_dist.get(str(d), 0) + 1
    top_dates = dict(sorted(pub_dist.items(), key=lambda x: x[1], reverse=True)[:10])

    overall_alpha_ratio = (
        round(total_alpha_chars / total_chars, 4) if total_chars > 0 else 0.0
    )
    mean_dup_lines = (
        round(sum(duplicate_line_ratios) / len(duplicate_line_ratios), 4)
        if duplicate_line_ratios
        else 0.0
    )

    return SampleReport(
        file_path=file_path,
        document_count=doc_count,
        total_characters=total_chars,
        total_words=sum(word_lengths),
        char_length_stats=_compute_percentiles(char_lengths),
        word_length_stats=_compute_percentiles(word_lengths),
        alphabetic_ratio=overall_alpha_ratio,
        empty_or_near_empty_count=near_empty_count,
        empty_or_near_empty_ratio=round(near_empty_count / doc_count, 4),
        exact_duplicate_count=dup_count,
        exact_duplicate_rate=round(dup_count / doc_count, 4),
        mean_duplicate_line_ratio=mean_dup_lines,
        high_repeated_line_doc_count=high_repeated_docs,
        high_repeated_line_doc_ratio=round(high_repeated_docs / doc_count, 4),
        missing_field_rates=missing_rates,
        source_distribution=source_dist,
        subset_distribution=subset_dist,
        domain_distribution=domain_dist,
        language_distribution=lang_dist,
        variety_distribution=variety_dist,
        quality_score_stats=qual_stats,
        publication_date_distribution=top_dates,
    )


def inspect_sample_file(parquet_path: Path | str) -> SampleReport:
    """Inspect a single sample Parquet file and return its diagnostic report.

    Loads co-located manifest JSON (if present) to enrich the report with
    provenance fields: sample mode, sampling frame, records examined,
    requested size, underfill status, and GigaVerbo exclusion statistics.

    Parameters
    ----------
    parquet_path : Path or str
        Path to Parquet file.

    Returns
    -------
    SampleReport
        Inspection report.
    """
    path = Path(parquet_path)
    table = load_sample_parquet(path)
    report = compute_sample_metrics(table, file_path=str(path))

    # Enrich from manifest
    manifest = _load_manifest_for(path)
    if manifest:
        report.sample_mode = manifest.get("mode")
        report.sampling_frame = manifest.get("sampling_frame")
        report.records_examined = manifest.get("records_examined")
        report.requested_size = manifest.get("target_size")
        stopping = manifest.get("stopping_reason", "")
        if stopping and "underfill" in stopping:
            report.underfill_status = stopping
        else:
            obtained = manifest.get("document_count", report.document_count)
            target = manifest.get("target_size", report.document_count)
            if obtained < target:
                report.underfill_status = f"underfilled: obtained {obtained}/{target}"
            else:
                report.underfill_status = "ok"
        stats = manifest.get("stats", {})
        if "subsets_before_exclusion" in stats or "exclusion_counts_by_subset" in stats:
            report.gigaverbo_exclusion_stats = {
                k: v
                for k, v in stats.items()
                if k
                in (
                    "subsets_before_exclusion",
                    "exclusion_counts_by_subset",
                    "subsets_after_exclusion",
                    "total_excluded",
                )
            }

    return report


def format_report_text(report: SampleReport) -> str:
    """Format SampleReport as a human-readable text string.

    Parameters
    ----------
    report : SampleReport
        Computed sample diagnostics.

    Returns
    -------
    str
        Formatted report text.
    """
    lines = [
        f"=== Sample Inspection Report: {report.file_path or 'In-Memory Table'} ===",
    ]

    # Provenance header
    if report.sample_mode is not None:
        lines.append(f"Sample Mode:              {report.sample_mode}")
    if report.sampling_frame is not None:
        lines.append(f"Sampling Frame:           {report.sampling_frame}")
    if report.requested_size is not None:
        lines.append(
            f"Requested vs Obtained:    {report.requested_size:,} requested / "
            f"{report.document_count:,} obtained"
        )
    if report.records_examined is not None:
        lines.append(f"Records Examined:         {report.records_examined:,}")
    if report.underfill_status is not None:
        flag = "" if report.underfill_status == "ok" else " [!]"
        lines.append(f"Underfill Status:         {report.underfill_status}{flag}")

    lines += [
        "",
        f"Documents: {report.document_count:,}",
        f"Total Characters: {report.total_characters:,}",
        f"Total Words: {report.total_words:,}",
        f"Alphabetic Character Ratio: {report.alphabetic_ratio:.2%}",
        f"Empty / Near-Empty (<100c or <20w): {report.empty_or_near_empty_count:,} "
        f"({report.empty_or_near_empty_ratio:.2%})",
        f"Exact In-Sample Duplicates: {report.exact_duplicate_count:,} "
        f"({report.exact_duplicate_rate:.2%})",
        f"High Repeated Lines (>=20%): {report.high_repeated_line_doc_count:,} "
        f"({report.high_repeated_line_doc_ratio:.2%})",
        "",
        "--- Character Length Percentiles ---",
        f"  Min: {report.char_length_stats.min:,.0f} | "
        f"Median: {report.char_length_stats.median:,.0f} | "
        f"P90: {report.char_length_stats.p90:,.0f} | "
        f"P99: {report.char_length_stats.p99:,.0f} | "
        f"Max: {report.char_length_stats.max:,.0f} | "
        f"Mean: {report.char_length_stats.mean:,.1f}",
        "",
        "--- Word Length Percentiles ---",
        f"  Min: {report.word_length_stats.min:,.0f} | "
        f"Median: {report.word_length_stats.median:,.0f} | "
        f"P90: {report.word_length_stats.p90:,.0f} | "
        f"P99: {report.word_length_stats.p99:,.0f} | "
        f"Max: {report.word_length_stats.max:,.0f} | "
        f"Mean: {report.word_length_stats.mean:,.1f}",
        "",
        "--- Source & Subset Distribution ---",
    ]

    subset_dist = report.subset_distribution
    n_subsets = len(subset_dist)
    # Show all subsets when cardinality is small (<=20), otherwise top 5
    show_all = n_subsets <= 20
    items_to_show = (
        list(subset_dist.items()) if show_all else list(subset_dist.items())[:5]
    )
    for k, v in items_to_show:
        pct = (v / report.document_count) * 100 if report.document_count else 0
        lines.append(f"  {k}: {v:,} ({pct:.1f}%)")
    if not show_all:
        remaining = n_subsets - 5
        lines.append(f"  ... and {remaining} more subsets (use --json for full list)")

    lines.append("")
    lines.append("--- Variety Distribution ---")
    for k, v in report.variety_distribution.items():
        pct = (v / report.document_count) * 100 if report.document_count else 0
        lines.append(f"  {k}: {v:,} ({pct:.1f}%)")

    if report.quality_score_stats:
        lines.append("")
        lines.append("--- Quality Score Stats (Upstream) ---")
        q = report.quality_score_stats
        lines.append(
            f"  Count: {q['count']:,} | Min: {q['min']:.2f} | Mean: {q['mean']:.2f} | "
            f"Median: {q['median']:.2f} | P90: {q['p90']:.2f} | Max: {q['max']:.2f}"
        )

    if report.gigaverbo_exclusion_stats:
        lines.append("")
        lines.append("--- GigaVerbo Exclusion Statistics ---")
        ex = report.gigaverbo_exclusion_stats
        total_excl = ex.get("total_excluded", 0)
        lines.append(f"  Total excluded records: {total_excl:,}")
        excl_by_subset = ex.get("exclusion_counts_by_subset", {})
        if excl_by_subset:
            lines.append("  Exclusion counts by subset:")
            for sub, cnt in sorted(
                excl_by_subset.items(), key=lambda x: x[1], reverse=True
            )[:10]:
                lines.append(f"    {sub}: {cnt:,}")
        subsets_before = ex.get("subsets_before_exclusion", {})
        if subsets_before:
            lines.append(
                f"  Subsets encountered before exclusion: {len(subsets_before)}"
            )
        subsets_after = ex.get("subsets_after_exclusion", {})
        if subsets_after:
            lines.append(f"  Subsets remaining after exclusion: {len(subsets_after)}")

    lines.append("")
    lines.append("--- Missing Field Rates ---")
    for field_name, rate in report.missing_field_rates.items():
        if rate > 0.0:
            lines.append(f"  {field_name}: {rate:.1%}")

    return "\n".join(lines)
