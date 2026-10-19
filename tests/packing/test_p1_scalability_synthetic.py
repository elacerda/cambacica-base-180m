"""C4-P1.3b: bounded CPU-only synthetic scalability probe tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cambacica.packing.p1_scalability_synthetic import (
    MAX_DOCUMENTS,
    MAX_DOCUMENT_TOKENS,
    SCHEMA,
    benchmark_case,
)


@pytest.mark.parametrize(
    ("documents", "tokens_per_document", "expected_targets", "expected_eos"),
    [
        (1, 0, 0, 0),
        (1, 1, 1, 1),
        (2, 0, 1, 1),
        (2, 2048, 4097, 2),
        (3, 4096, 12290, 3),
        (15, 97, 1469, 15),
    ],
)
def test_synthetic_accounting_and_bounded_index(
    tmp_path: Path,
    documents: int,
    tokens_per_document: int,
    expected_targets: int,
    expected_eos: int,
) -> None:
    result = benchmark_case(
        documents=documents,
        tokens_per_document=tokens_per_document,
        scratch_parent=tmp_path,
    )
    assert result["schema"] == SCHEMA
    assert result["fixture_only"] is True
    assert result["documents"] == documents
    assert result["stream_tokens"] == documents * (tokens_per_document + 1)
    assert result["scored_targets"] == expected_targets
    assert result["scored_eos"] == expected_eos
    assert result["payload_bytes"] == 2 * documents * tokens_per_document
    assert result["sqlite_bytes"] > 0
    assert result["manifest_bytes"] > 0
    assert result["peak_process_rss_kib"] > 0
    assert result["ack_chain_digest"]
    assert list(tmp_path.iterdir()) == []  # No retained fixture data.


def test_restart_replay_and_fragmentation_preserve_digest(tmp_path: Path) -> None:
    cases = [
        benchmark_case(
            documents=67,
            tokens_per_document=97,
            fragment_size=fragment,
            restart_after=restart,
            scratch_parent=tmp_path,
        )
        for fragment, restart in ((1, 1), (257, 0), (4097, 1))
    ]
    for case in cases:
        assert case["windows"] == 2
        assert case["scored_targets"] == 67 * 98 - 1
        assert case["ack_chain_digest"] == cases[0]["ack_chain_digest"]
        assert case["order_digest"] == cases[0]["order_digest"]
        assert case["view_digest"] == cases[0]["view_digest"]
    assert cases[0]["resume_checks"] == cases[2]["resume_checks"] == 1
    assert cases[1]["resume_checks"] == 0
    assert cases[0]["resume_verify_seconds"] >= 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"documents": 0, "tokens_per_document": 0},
        {"documents": True, "tokens_per_document": 1},
        {"documents": MAX_DOCUMENTS + 1, "tokens_per_document": 1},
        {"documents": 1, "tokens_per_document": -1},
        {"documents": 1, "tokens_per_document": MAX_DOCUMENT_TOKENS + 1},
        {"documents": 500, "tokens_per_document": 5000},
        {"documents": 1, "tokens_per_document": 1, "fragment_size": 0},
        {"documents": 2, "tokens_per_document": 1, "restart_after": 1},
        {"documents": 1, "tokens_per_document": 1, "restart_after": -1},
    ],
)
def test_invalid_workloads_fail_before_any_index_write(
    tmp_path: Path, kwargs: dict[str, int]
) -> None:
    with pytest.raises(ValueError):
        benchmark_case(scratch_parent=tmp_path, **kwargs)
    assert list(tmp_path.iterdir()) == []


def test_missing_scratch_parent_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scratch_parent"):
        benchmark_case(
            documents=1,
            tokens_per_document=1,
            scratch_parent=tmp_path / "not-created",
        )


def test_index_bytes_increase_with_more_artificial_documents(tmp_path: Path) -> None:
    small = benchmark_case(
        documents=16, tokens_per_document=64, scratch_parent=tmp_path
    )
    large = benchmark_case(
        documents=128, tokens_per_document=64, scratch_parent=tmp_path
    )
    assert large["payload_bytes"] == small["payload_bytes"] * 8
    assert large["sqlite_bytes"] >= small["sqlite_bytes"]
    assert large["total_index_bytes"] > small["total_index_bytes"]
    assert large["scored_targets"] > small["scored_targets"]


def test_cli_reports_single_machine_readable_case(tmp_path: Path) -> None:
    env = dict(os.environ)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cambacica.packing.p1_scalability_synthetic",
            "--documents",
            "32",
            "--tokens-per-document",
            "128",
            "--fragment-size",
            "31",
            "--restart-after",
            "1",
            "--scratch-parent",
            str(tmp_path),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=25,
    )
    result_json = json.loads(result.stdout)
    assert result_json["schema"] == SCHEMA
    assert result_json["scored_targets"] == 32 * 129 - 1
    assert result_json["resume_checks"] == 1
    assert result_json["peak_process_rss_kib"] > 0
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []


def test_cli_rejects_unbounded_jobs_without_creating_files(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cambacica.packing.p1_scalability_synthetic",
            "--documents",
            str(MAX_DOCUMENTS + 1),
            "--tokens-per-document",
            "2",
            "--scratch-parent",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "documents must be an integer" in result.stderr
    assert not result.stdout
    assert list(tmp_path.iterdir()) == []
