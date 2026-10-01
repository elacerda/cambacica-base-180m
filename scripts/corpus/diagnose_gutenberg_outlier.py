#!/usr/bin/env python3
"""Gutenberg PT outlier investigation (Gate C1 diagnostic).

Reads an existing gutenberg_pt sample Parquet file and identifies the
maximum-length document.  Reports ebook ID, word/character counts, text
preview, and a best-effort determination of whether it is:

  - a legitimate single very large text
  - a collected works / anthology
  - a malformed concatenation
  - a duplicated/repeated acquisition

Usage
-----
    python3 scripts/corpus/diagnose_gutenberg_outlier.py \\
        data/samples/gate_c1/gutenberg_pt/representative.parquet

Pass --json for machine-readable output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import pyarrow.parquet as pq


# Common patterns that indicate an anthology / collected works
_ANTHOLOGY_MARKERS = re.compile(
    r"(?i)(\bobras completas\b|\bobras escolhidas\b|\bcoletânea\b"
    r"|\bantologia\b|\bvolume\s+[ivxlcdm0-9]+\b"
    r"|\btomo\s+[ivxlcdm0-9]+\b"
    r"|\b(primeira|segunda|terceira)\s+parte\b"
    r"|\bpart\s+(i|ii|iii|iv|v|vi|vii|viii)\b)"
)

# Patterns that suggest repeated blocks / duplicated concatenation
_REPETITION_MARKER_LEN = 2000  # compare first and last N chars


def _classify_outlier(text: str, ebook_id: str) -> str:
    """Classify why a document is unusually large.

    Parameters
    ----------
    text : str
        Full document text.
    ebook_id : str
        Gutenberg ebook ID.

    Returns
    -------
    str
        Classification label with brief reason.
    """
    # Check for anthology markers
    if _ANTHOLOGY_MARKERS.search(text[:5000]):
        return "collected_works_or_anthology (markers found in first 5 000 chars)"

    # Check for duplicated concatenation: same long block repeated
    n = len(text)
    if n > 10_000:
        quarter = n // 4
        first_chunk = text[:quarter]
        # Look for the first_chunk repeated later
        if first_chunk[:500] in text[quarter * 2 :]:
            return (
                "duplicated_concatenation (first quarter found verbatim in second half)"
            )

    # Very long but no repetition or anthology markers → likely legitimate
    n_words = len(text.split())
    if n_words > 100_000:
        return (
            f"legitimate_very_large_text "
            f"({n_words:,} words; no anthology markers or repetition detected)"
        )

    return "unknown_outlier"


def analyze_gutenberg_outlier(parquet_path: Path, preview_len: int = 500) -> dict:
    """Identify and characterize the maximum-length Gutenberg document.

    Parameters
    ----------
    parquet_path : Path
        Path to gutenberg_pt Parquet sample.
    preview_len : int, default 500
        Character length for text previews.

    Returns
    -------
    dict
        Outlier analysis report.
    """
    table = pq.read_table(
        parquet_path,
        columns=["text", "original_id", "original_url"],
    )
    rows = table.to_pylist()

    if not rows:
        return {"error": "No documents found in sample."}

    # Compute word counts
    def _wc(text: str) -> int:
        return len((text or "").split())

    def _cc(text: str) -> int:
        return len(text or "")

    # Find max-word document
    max_word_row = max(rows, key=lambda r: _wc(r.get("text") or ""))

    # Sort all by word count for percentile context
    sorted_by_words = sorted(rows, key=lambda r: _wc(r.get("text") or ""))
    n = len(sorted_by_words)
    median_wc = _wc(sorted_by_words[n // 2].get("text") or "")
    p90_wc = _wc(sorted_by_words[int(0.9 * n)].get("text") or "")

    outlier_text = max_word_row.get("text") or ""
    ebook_id = max_word_row.get("original_id") or "unknown"
    url = (
        max_word_row.get("original_url")
        or f"https://www.gutenberg.org/ebooks/{ebook_id}"
    )

    classification = _classify_outlier(outlier_text, ebook_id)

    return {
        "parquet_file": str(parquet_path),
        "total_documents": n,
        "word_count_median": median_wc,
        "word_count_p90": p90_wc,
        "outlier": {
            "ebook_id": ebook_id,
            "url": url,
            "word_count": _wc(outlier_text),
            "char_count": _cc(outlier_text),
            "classification": classification,
            "text_preview_start": outlier_text[:preview_len].replace("\n", " "),
            "text_preview_end": outlier_text[-preview_len:].replace("\n", " "),
        },
        "note": (
            "No length cutoff has been introduced. "
            "This report is for source characterization only."
        ),
    }


def format_gutenberg_report(report: dict) -> str:
    """Format the Gutenberg outlier report as readable text.

    Parameters
    ----------
    report : dict
        Output of analyze_gutenberg_outlier.

    Returns
    -------
    str
        Formatted text.
    """
    if "error" in report:
        return f"[ERROR] {report['error']}"

    o = report["outlier"]
    lines = [
        "=== Gutenberg PT Outlier Investigation ===",
        f"File:                 {report['parquet_file']}",
        f"Total Documents:      {report['total_documents']:,}",
        f"Word Count Median:    {report['word_count_median']:,}",
        f"Word Count P90:       {report['word_count_p90']:,}",
        "",
        "--- Maximum-Length Document ---",
        f"Ebook ID:             {o['ebook_id']}",
        f"URL:                  {o['url']}",
        f"Word Count:           {o['word_count']:,}",
        f"Char Count:           {o['char_count']:,}",
        f"Classification:       {o['classification']}",
        "",
        "--- Text Preview (start) ---",
        o["text_preview_start"],
        "",
        "--- Text Preview (end) ---",
        o["text_preview_end"],
        "",
        report["note"],
    ]
    return "\n".join(lines)


def main() -> int:
    """Script entrypoint.

    Returns
    -------
    int
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="Identify and classify the Gutenberg PT outlier document."
    )
    parser.add_argument(
        "parquet",
        type=str,
        help="Path to gutenberg_pt sample Parquet file.",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON.")
    args = parser.parse_args()

    path = Path(args.parquet)
    if not path.is_file():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        return 1

    report = analyze_gutenberg_outlier(path)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_gutenberg_report(report))
    return 0


if __name__ == "__main__":
    import os

    sys.stdout.flush()
    sys.stderr.flush()
    ret = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(ret)
