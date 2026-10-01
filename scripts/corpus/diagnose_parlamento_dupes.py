#!/usr/bin/env python3
"""Parlamento PT duplicate analysis (Gate C1 diagnostic).

Reads an existing parlamento_pt sample and produces a diagnostic report
of the top duplicate hashes: occurrence count, text preview, and category
(formula, speaker label, procedural utterance, or unexpectedly long repetition).

Usage
-----
    python3 scripts/corpus/diagnose_parlamento_dupes.py \\
        data/samples/gate_c1/parlamento_pt/representative.parquet

Output is printed to stdout.  Pass --json for machine-readable output.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import pyarrow.parquet as pq


# Simple heuristics for categorizing parliamentary text patterns
_SPEAKER_PATTERNS = (
    "o sr.",
    "o sr ",
    "a sr.",
    "a sr ",
    "o senhor",
    "a senhora",
    "sr. presidente",
    "sra. presidente",
    "presidente:",
    "o presidente:",
)
_PROCEDURAL_PATTERNS = (
    "apoiado",
    "muito bem",
    "palmas",
    "risas",
    "pausa",
    "interrupção",
    "protesto",
    "vozes",
    "nao apoiado",
    "não apoiado",
    "não!",
    "sim!",
)


def _categorize(text: str) -> str:
    """Categorize a parliamentary text snippet.

    Parameters
    ----------
    text : str
        Document text.

    Returns
    -------
    str
        Category label.
    """
    lower = text.lower().strip()
    n_words = len(lower.split())

    # Very short texts are likely formulaic
    if n_words <= 5:
        for p in _PROCEDURAL_PATTERNS:
            if p in lower:
                return "procedural_utterance"
        for p in _SPEAKER_PATTERNS:
            if lower.startswith(p):
                return "speaker_label"
        return "short_formula"

    # Speaker introductions
    for p in _SPEAKER_PATTERNS:
        if lower.startswith(p):
            return "speaker_label"

    # Procedural formulas
    for p in _PROCEDURAL_PATTERNS:
        if lower.strip() == p or lower.startswith(p + " "):
            return "procedural_utterance"

    # Moderately long → likely duplicated long document
    if n_words > 200:
        return "unexpectedly_long_repetition"

    return "parliamentary_formula"


def analyze_parlamento_duplicates(
    parquet_path: Path,
    top_n: int = 20,
    preview_len: int = 120,
) -> dict:
    """Produce a duplicate analysis report for a parlamento_pt sample.

    Parameters
    ----------
    parquet_path : Path
        Path to Parquet sample file.
    top_n : int, default 20
        Number of top duplicate hashes to report.
    preview_len : int, default 120
        Character length for text preview snippets.

    Returns
    -------
    dict
        Diagnostic report.
    """
    table = pq.read_table(
        parquet_path, columns=["text", "content_sha256", "original_id"]
    )
    hashes = table["content_sha256"].to_pylist()
    texts = table["text"].to_pylist()
    ids = table["original_id"].to_pylist()

    total_docs = len(hashes)

    # Build hash -> list of (text, original_id)
    hash_to_entries: dict = {}
    for sha, text, oid in zip(hashes, texts, ids):
        if sha not in hash_to_entries:
            hash_to_entries[sha] = []
        hash_to_entries[sha].append((text, oid))

    # Count occurrences
    dup_hashes = {
        sha: entries for sha, entries in hash_to_entries.items() if len(entries) > 1
    }
    total_dup_occurrences = sum(len(v) - 1 for v in dup_hashes.values())

    # Sort by descending occurrence count
    sorted_dups = sorted(
        dup_hashes.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )

    top_entries = []
    for sha, entries in sorted_dups[:top_n]:
        text = entries[0][0] or ""
        category = _categorize(text)
        preview = text[:preview_len].replace("\n", " ").strip()
        top_entries.append(
            {
                "hash": sha,
                "occurrences": len(entries),
                "category": category,
                "n_words": len(text.split()),
                "n_chars": len(text),
                "preview": preview,
            }
        )

    # Category breakdown across ALL duplicate hashes
    category_counts: Counter = Counter()
    for sha, entries in dup_hashes.items():
        cat = _categorize(entries[0][0] or "")
        category_counts[cat] += len(entries) - 1

    return {
        "parquet_file": str(parquet_path),
        "total_documents": total_docs,
        "total_unique_hashes": len(hash_to_entries),
        "total_duplicate_hashes": len(dup_hashes),
        "total_duplicate_occurrences": total_dup_occurrences,
        "duplicate_rate": round(total_dup_occurrences / total_docs, 4)
        if total_docs
        else 0,
        "category_breakdown": dict(category_counts.most_common()),
        "top_duplicates": top_entries,
        "note": (
            "Duplicates are NOT filtered from representative sampling. "
            "This report is source characterization only."
        ),
    }


def format_parlamento_report(report: dict) -> str:
    """Format the Parlamento duplicate analysis report as readable text.

    Parameters
    ----------
    report : dict
        Output of analyze_parlamento_duplicates.

    Returns
    -------
    str
        Formatted text.
    """
    lines = [
        "=== Parlamento PT Duplicate Analysis ===",
        f"File:                     {report['parquet_file']}",
        f"Total Documents:          {report['total_documents']:,}",
        f"Total Unique Hashes:      {report['total_unique_hashes']:,}",
        f"Total Duplicate Hashes:   {report['total_duplicate_hashes']:,}",
        f"Total Dup Occurrences:    {report['total_duplicate_occurrences']:,} "
        f"({report['duplicate_rate']:.2%})",
        "",
        "--- Category Breakdown (duplicate occurrences) ---",
    ]
    for cat, cnt in report["category_breakdown"].items():
        lines.append(f"  {cat}: {cnt:,}")

    lines += [
        "",
        f"--- Top {len(report['top_duplicates'])} Duplicate Hashes ---",
    ]
    for i, entry in enumerate(report["top_duplicates"], 1):
        lines += [
            f"  {i:>2}. hash={entry['hash'][:16]}...  "
            f"occurrences={entry['occurrences']}  "
            f"category={entry['category']}  "
            f"words={entry['n_words']}  chars={entry['n_chars']}",
            f"      preview: {entry['preview']}",
        ]

    lines.append("")
    lines.append(report["note"])
    return "\n".join(lines)


def main() -> int:
    """Script entrypoint.

    Returns
    -------
    int
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="Diagnose Parlamento PT duplicate hashes in a sample Parquet file."
    )
    parser.add_argument(
        "parquet",
        type=str,
        help="Path to parlamento_pt sample Parquet file.",
    )
    parser.add_argument(
        "--top", type=int, default=20, help="Number of top duplicate hashes to show."
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON.")
    args = parser.parse_args()

    path = Path(args.parquet)
    if not path.is_file():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        return 1

    report = analyze_parlamento_duplicates(path, top_n=args.top)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_parlamento_report(report))
    return 0


if __name__ == "__main__":
    import os

    sys.stdout.flush()
    sys.stderr.flush()
    ret = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(ret)
