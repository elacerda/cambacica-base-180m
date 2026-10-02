"""Mocked Parquet tests for pinned GigaVerbo residual materialization."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

import cambacica.corpus.materialize_gigaverbo as materializer_module
from cambacica.corpus.materialize_gigaverbo import (
    GigaverboV2Materializer,
    _atomic_write_parquet,
)

REPOSITORY = "Polygl0t/gigaverbo-v2"
PINNED_COMMIT = "7058ccf19eaeaf4505a96fc7e5305a01fc441fd8"
SPLIT = "edu_high"
EXCLUSIONS = Path("configs/gigaverbo_exclusions.yaml").resolve()
PROVENANCE_COLUMNS = {
    "_gv2_upstream_shard",
    "_gv2_upstream_row_group",
    "_gv2_upstream_commit",
}


class _FakeApi:
    def __init__(self, sha: str, siblings: list[SimpleNamespace]) -> None:
        self.sha = sha
        self.siblings = siblings
        self.calls: list[dict] = []

    def repo_info(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(sha=self.sha, siblings=self.siblings)


class _FakeFileSystem:
    def __init__(self, paths: dict[str, Path]) -> None:
        self.paths = paths
        self.opened: list[str] = []

    def open(self, remote_path: str, mode: str):
        self.opened.append(remote_path)
        assert f"@{PINNED_COMMIT}/" in remote_path
        assert "@main/" not in remote_path
        shard = remote_path.split("/", 3)[-1]
        return self.paths[Path(shard).name].open(mode)


def _write_fixture_shard(path: Path, shard_index: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    shard_label = f"fixture-{shard_index}"
    table = pa.table(
        {
            "text": [
                f"Texto cru aceito {shard_label}.",
                f"Texto traduzido {shard_label}.",
                f"Documento técnico {shard_label}.",
                f"Chat bloqueado {shard_label}.",
                f"Segundo chat bloqueado {shard_label}.",
                f"Terceiro chat bloqueado {shard_label}.",
            ],
            "id": [
                f"{shard_label}-fineweb",
                f"{shard_label}-dolly",
                f"{shard_label}-finepdfs",
                f"{shard_label}-ultrachat",
                f"{shard_label}-ultrachat-2",
                f"{shard_label}-ultrachat-3",
            ],
            "source": [
                "https://example.org/fineweb",
                "https://huggingface.co/datasets/Gustrd/dolly-15k-libretranslate-pt",
                "https://example.org/finepdfs",
                "https://example.org/ultrachat",
                "https://example.org/ultrachat",
                "https://example.org/ultrachat",
            ],
            "subset": [
                "fineweb_2_pt",
                "dolly_15k",
                "finepdfs_por_Latn",
                "ultrachat",
                "ultrachat",
                "ultrachat",
            ],
            "token_count": [3, 3, 3, 3, 3, 3],
            "edu_score": [4.1, 4.1, 4.1, 4.1, 4.1, 4.1],
            "edu_int_score": [4, 4, 4, 4, 4, 4],
            "toxic_score": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "toxic_int_score": [1, 1, 1, 1, 1, 1],
        }
    )
    pq.write_table(table, path, row_group_size=2, compression="zstd")
    parquet_file = pq.ParquetFile(path)
    row_groups = []
    for index in range(parquet_file.num_row_groups):
        row_group = parquet_file.metadata.row_group(index)
        subset_table = parquet_file.read_row_group(index, columns=["subset"])
        subset_counts = dict(
            sorted(Counter(subset_table.column("subset").to_pylist()).items())
        )
        stat = row_group.column(
            parquet_file.schema_arrow.get_field_index("subset")
        ).statistics
        row_groups.append(
            {
                "index": index,
                "records": row_group.num_rows,
                "compressed_bytes": sum(
                    row_group.column(column_index).total_compressed_size
                    for column_index in range(row_group.num_columns)
                ),
                "uncompressed_bytes": row_group.total_byte_size,
                "subset_column_compressed_bytes": row_group.column(
                    parquet_file.schema_arrow.get_field_index("subset")
                ).total_compressed_size,
                "subset_counts": subset_counts,
                "subset_statistics_min_max": {
                    "min": stat.min,
                    "max": stat.max,
                },
            }
        )
    payload = path.read_bytes()
    relative = f"{SPLIT}/{path.name}"
    return {
        "path": relative,
        "size_bytes": len(payload),
        "git_blob_oid": hashlib.sha1(payload).hexdigest(),
        "lfs_sha256": hashlib.sha256(payload).hexdigest(),
        "row_group_count": parquet_file.num_row_groups,
        "records": parquet_file.metadata.num_rows,
        "row_groups": row_groups,
    }


def _build_fixture(tmp_path: Path, monkeypatch):
    upstream_dir = tmp_path / "upstream"
    shards = [
        _write_fixture_shard(
            upstream_dir / f"train-{index:05d}-of-00002.parquet", index
        )
        for index in range(2)
    ]
    subset_records: Counter[str] = Counter()
    for shard in shards:
        for row_group in shard["row_groups"]:
            subset_records.update(row_group["subset_counts"])
    inventory_path = tmp_path / "physical_inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "repository": REPOSITORY,
                "partition": SPLIT,
                "pinned_commit_sha": PINNED_COMMIT,
                "summary": {
                    "shard_count": 2,
                    "row_group_count": 6,
                    "upstream_records": 12,
                    "subset_records": dict(sorted(subset_records.items())),
                },
                "shards": shards,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    exclusions_path = tmp_path / "gigaverbo_exclusions.yaml"
    exclusions_path.write_bytes(EXCLUSIONS.read_bytes())
    destination = tmp_path / "raw" / "gigaverbo"
    config_path = tmp_path / "corpus_materialization.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "storage": {"paths": {"raw": str(tmp_path / "raw")}},
                "sources": {
                    "gigaverbo_v2": {
                        "canonical_name": "gigaverbo_v2",
                        "repository": REPOSITORY,
                        "split": SPLIT,
                        "pinned_revision": SPLIT,
                        "pinned_commit_sha": PINNED_COMMIT,
                        "exclusions_config": str(exclusions_path),
                        "physical_inventory": str(inventory_path),
                        "expected_shard_count": 2,
                        "expected_row_group_count": 6,
                        "expected_total_records": 12,
                        "destination": str(destination),
                        "estimated_raw_size": "fixture",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    siblings = [
        SimpleNamespace(
            rfilename=shard["path"],
            size=shard["size_bytes"],
            blob_id=shard["git_blob_oid"],
            lfs=SimpleNamespace(sha256=shard["lfs_sha256"]),
        )
        for shard in shards
    ]
    api = _FakeApi(PINNED_COMMIT, siblings)
    fs = _FakeFileSystem(
        {
            Path(shard["path"]).name: upstream_dir / Path(shard["path"]).name
            for shard in shards
        }
    )
    monkeypatch.setattr(materializer_module, "HfApi", lambda: api)
    monkeypatch.setattr(materializer_module, "HfFileSystem", lambda: fs)
    monkeypatch.setattr(
        materializer_module, "_get_clean_tool_git_commit", lambda: "test-tool-commit"
    )
    materializer = GigaverboV2Materializer(
        config_path=config_path,
        destination_override=destination,
        allow_custom_destination=True,
        exclusions_path=exclusions_path,
        physical_inventory_path=inventory_path,
    )
    return materializer, api, fs, exclusions_path, shards


def test_pinned_upstream_inventory_has_all_56_shards_and_oids(
    tmp_path: Path, monkeypatch
):
    materializer = GigaverboV2Materializer(
        destination_override=tmp_path / "raw" / "gigaverbo",
        allow_custom_destination=True,
    )
    physical, _ = materializer._load_physical_inventory()
    siblings = [
        SimpleNamespace(
            rfilename=shard["path"],
            size=shard["size_bytes"],
            blob_id=shard["git_blob_oid"],
            lfs=SimpleNamespace(sha256=shard["lfs_sha256"]),
        )
        for shard in physical["shards"]
    ]
    api = _FakeApi(PINNED_COMMIT, siblings)
    monkeypatch.setattr(materializer_module, "HfApi", lambda: api)

    inventory = materializer._fetch_upstream_inventory(physical)

    assert len(inventory) == 56
    assert len({item["path"] for item in inventory}) == 56
    assert all(item["git_blob_oid"] for item in inventory)
    assert all(item["lfs_sha256"] for item in inventory)
    assert api.calls == [
        {
            "repo_id": REPOSITORY,
            "repo_type": "dataset",
            "revision": PINNED_COMMIT,
            "files_metadata": True,
        }
    ]


def test_raw_materialization_filters_and_preserves_rowgroup_provenance(
    tmp_path: Path, monkeypatch
):
    materializer, api, fs, _, _ = _build_fixture(tmp_path, monkeypatch)

    manifest = materializer.materialize(max_retries=0)

    assert manifest.status == "COMPLETE"
    assert manifest.pinned_commit_sha == PINNED_COMMIT
    assert manifest.total_files == 4
    metadata = manifest.source_metadata
    assert metadata["records_examined"] == 12
    assert metadata["records_excluded"] == 8
    assert metadata["eligible_records"] == 4
    assert metadata["total_persisted_documents"] == 4
    assert sum(metadata["records_encountered_per_subset"].values()) == 12
    assert sum(metadata["eligible_records_per_subset"].values()) == 4
    assert sum(metadata["persisted_records_per_subset"].values()) == 4
    assert len(metadata["upstream_inventory"]) == 2
    assert len(metadata["row_groups_visited"]) == 6
    assert metadata["exclusion_counts_by_subset_rule"]["dolly_15k"]
    assert metadata["exclusion_counts_by_subset_rule"]["ultrachat"]
    excluded_only_group = next(
        item
        for item in metadata["row_groups_visited"]
        if item["upstream_shard"].endswith("train-00000-of-00002.parquet")
        and item["row_group"] == 2
    )
    assert excluded_only_group["subset_min_max_statistics"][
        "used_for_exact_identification"
    ]
    assert excluded_only_group["subset_min_max_statistics"][
        "validated_by_subset_column"
    ]
    assert not excluded_only_group["persisted_files"]
    assert set(metadata["persisted_records_per_subset"]) == {
        "fineweb_2_pt",
        "finepdfs_por_Latn",
    }
    assert all(f"@{PINNED_COMMIT}/" in path for path in fs.opened)
    assert api.calls[0]["revision"] == PINNED_COMMIT

    seen_text = set()
    for record in manifest.files:
        path = materializer.destination / record["relative_path"]
        table = pq.ParquetFile(path).read()
        assert PROVENANCE_COLUMNS.issubset(table.column_names)
        assert table.column("subset").to_pylist() == [record["upstream_subset"]]
        assert set(table.column("_gv2_upstream_shard").to_pylist()) == {
            record["upstream_shard"]
        }
        assert set(table.column("_gv2_upstream_row_group").to_pylist()) == {
            record["upstream_row_group"]
        }
        assert set(table.column("_gv2_upstream_commit").to_pylist()) == {PINNED_COMMIT}
        seen_text.update(table.column("text").to_pylist())
    assert not any(
        "traduzido" in value or "Chat bloqueado" in value for value in seen_text
    )
    assert "Texto cru aceito fixture-0." in seen_text
    assert "Documento técnico fixture-1." in seen_text
    valid, errors = materializer.verify()
    assert valid, errors


def test_complete_manifest_is_trusted_reused_and_idempotent(
    tmp_path: Path, monkeypatch
):
    materializer, api, fs, _, _ = _build_fixture(tmp_path, monkeypatch)
    first = materializer.materialize(max_retries=0)
    first_hashes = {item["relative_path"]: item["sha256"] for item in first.files}
    first_remote_reads = list(fs.opened)

    monkeypatch.setattr(
        materializer_module,
        "HfApi",
        lambda: (_ for _ in ()).throw(AssertionError("trusted reuse hit network")),
    )
    second = materializer.materialize(max_retries=0)

    assert second.status == "COMPLETE"
    assert {
        item["relative_path"]: item["sha256"] for item in second.files
    } == first_hashes
    assert fs.opened == first_remote_reads
    assert not (materializer.destination / "manifest.in_progress.json").exists()
    assert api.calls[0]["revision"] == PINNED_COMMIT


def test_partial_run_recovers_after_a_completed_shard(tmp_path: Path, monkeypatch):
    materializer, _, fs, _, _ = _build_fixture(tmp_path, monkeypatch)
    original = materializer._write_row_group_subset

    def fail_first_rowgroup_of_second_shard(
        table, *, subset, upstream_shard, row_group_index
    ):
        if (
            upstream_shard.endswith("train-00001-of-00002.parquet")
            and row_group_index == 0
        ):
            raise OSError("simulated interrupted row-group read")
        return original(
            table,
            subset=subset,
            upstream_shard=upstream_shard,
            row_group_index=row_group_index,
        )

    monkeypatch.setattr(
        materializer, "_write_row_group_subset", fail_first_rowgroup_of_second_shard
    )
    partial = materializer.materialize(max_retries=0)
    assert partial.status == "FAILED"
    assert len(partial.source_metadata["row_groups_visited"]) == 3
    assert partial.source_metadata["pending_row_group"] == {
        "upstream_shard": "edu_high/train-00001-of-00002.parquet",
        "row_group": 0,
    }

    monkeypatch.setattr(materializer, "_write_row_group_subset", original)
    resumed = materializer.materialize(max_retries=0)

    assert resumed.status == "COMPLETE"
    assert len(resumed.source_metadata["row_groups_visited"]) == 6
    assert resumed.source_metadata["retry_count"] == 0
    assert resumed.source_metadata["failure_records"]
    assert sum(resumed.source_metadata["persisted_records_per_subset"].values()) == 4
    assert all(f"@{PINNED_COMMIT}/" in path for path in fs.opened)


def test_verify_rejects_manifest_invariant_mismatch(tmp_path: Path, monkeypatch):
    materializer, _, _, _, _ = _build_fixture(tmp_path, monkeypatch)
    materializer.materialize(max_retries=0)
    manifest_path = materializer.destination / "manifest.json"
    serialized = json.loads(manifest_path.read_text(encoding="utf-8"))
    serialized["source_metadata"]["records_examined"] -= 1
    manifest_path.write_text(json.dumps(serialized), encoding="utf-8")

    valid, errors = materializer.verify()

    assert not valid
    assert any("records_examined" in error for error in errors)


def test_corrupted_payload_fails_verify_and_is_never_reblessed(
    tmp_path: Path, monkeypatch
):
    materializer, _, _, _, _ = _build_fixture(tmp_path, monkeypatch)
    manifest = materializer.materialize(max_retries=0)
    payload = materializer.destination / manifest.files[0]["relative_path"]
    payload.write_bytes(b"corrupted local parquet")

    valid, errors = materializer.verify()
    assert not valid
    assert any("SHA-256 mismatch" in error for error in errors)
    with pytest.raises(RuntimeError, match="refusing to re-bless"):
        materializer.materialize(max_retries=0)


def test_exclusion_config_hash_mismatch_is_detected(tmp_path: Path, monkeypatch):
    materializer, _, _, exclusions_path, _ = _build_fixture(tmp_path, monkeypatch)
    materializer.materialize(max_retries=0)
    exclusions_path.write_text(
        exclusions_path.read_text(encoding="utf-8")
        + "\n# changed after materialization\n",
        encoding="utf-8",
    )

    valid, errors = materializer.verify()
    assert not valid
    assert any("Exclusion config SHA-256 mismatch" in error for error in errors)
    with pytest.raises(RuntimeError, match="Exclusion config SHA-256 mismatch"):
        materializer.materialize(max_retries=0)


def test_atomic_parquet_replace_keeps_previous_payload_on_failure(
    tmp_path: Path, monkeypatch
):
    destination = tmp_path / "fragment.parquet"
    destination.write_bytes(b"trusted prior payload")
    original_replace = materializer_module.os.replace

    def fail_payload_replace(source, target):
        if str(source).endswith(".parquet.partial"):
            raise OSError("simulated atomic replace failure")
        return original_replace(source, target)

    monkeypatch.setattr(materializer_module.os, "replace", fail_payload_replace)
    with pytest.raises(OSError, match="simulated atomic replace failure"):
        _atomic_write_parquet(pa.table({"text": ["novo"]}), destination)

    assert destination.read_bytes() == b"trusted prior payload"
    assert destination.with_name("fragment.parquet.partial").is_file()
