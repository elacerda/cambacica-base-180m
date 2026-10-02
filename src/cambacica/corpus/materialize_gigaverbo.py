"""Selective raw materialization for the pinned GigaVerbo-v2 partition."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from huggingface_hub import HfApi, HfFileSystem
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from cambacica.corpus.manifest import compute_file_sha256
from cambacica.corpus.materialize import (
    DEFAULT_CONFIG_PATH,
    MATERIALIZER_REPO_ROOT,
    BaseMaterializer,
    MaterializationManifest,
    _get_clean_tool_git_commit,
)
from cambacica.corpus.sources.gigaverbo_v2 import (
    get_exclusion_rules_map,
    load_gigaverbo_exclusions,
    match_subset_exclusion,
)

logger = logging.getLogger(__name__)

_PROVENANCE_COLUMNS = (
    "_gv2_upstream_shard",
    "_gv2_upstream_row_group",
    "_gv2_upstream_commit",
)
_REQUIRED_SOURCE_COLUMNS = {"text", "id", "source", "subset"}
_PARQUET_SUFFIX = ".parquet"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe_stat_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _counter_dict(values: Counter[str]) -> Dict[str, int]:
    return {key: int(values[key]) for key in sorted(values)}


def _atomic_write_parquet(table: pa.Table, destination: Path) -> Tuple[int, str]:
    """Write a deterministic Parquet payload through a same-directory temp file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.partial")
    temporary.unlink(missing_ok=True)
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            compression_level=6,
            use_dictionary=True,
            write_statistics=True,
            version="2.6",
        )
        with temporary.open("rb") as payload:
            os.fsync(payload.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        # Leave an incomplete temp payload for inspection; a trusted partial
        # manifest only permits its pending row group to remove/rewrite it.
        raise
    return destination.stat().st_size, compute_file_sha256(destination)


class GigaverboV2Materializer(BaseMaterializer):
    """Materialize every non-excluded row from pinned ``edu_high``.

    Rows are emitted as deterministic Parquet files partitioned by upstream
    subset, source shard, and row group. The raw ``text`` values and all
    upstream columns are preserved. Three provenance columns identify the
    pinned shard, row group, and commit for every persisted document.
    """

    def __init__(
        self,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
        destination_override: Optional[Path | str] = None,
        allow_custom_destination: bool = False,
        exclusions_path: Optional[Path | str] = None,
        physical_inventory_path: Optional[Path | str] = None,
    ) -> None:
        super().__init__(
            source_name="gigaverbo_v2",
            config_path=config_path,
            destination_override=destination_override,
            allow_custom_destination=allow_custom_destination,
        )
        self.repository = str(self.source_config["repository"])
        self.split = str(self.source_config["split"])
        self.pinned_revision = str(self.source_config["pinned_revision"])
        self.pinned_commit_sha = str(self.source_config["pinned_commit_sha"])
        self.expected_shard_count = int(self.source_config["expected_shard_count"])
        self.expected_row_group_count = int(
            self.source_config["expected_row_group_count"]
        )
        self.expected_total_records = int(self.source_config["expected_total_records"])

        configured_exclusions = (
            exclusions_path or self.source_config["exclusions_config"]
        )
        self.exclusions_path = self._resolve_project_path(configured_exclusions)

        configured_inventory = (
            physical_inventory_path or self.source_config["physical_inventory"]
        )
        self.physical_inventory_path = self._resolve_project_path(configured_inventory)

    @staticmethod
    def _resolve_project_path(path: Path | str) -> Path:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = MATERIALIZER_REPO_ROOT / resolved
        return resolved.resolve()

    def plan(self) -> Dict[str, Any]:
        """Return the pinned raw-reservoir plan without accessing the network."""
        return {
            "source": "gigaverbo_v2",
            "canonical_name": self.source_config.get("canonical_name", "gigaverbo_v2"),
            "repository": self.repository,
            "split": self.split,
            "pinned_revision": self.pinned_revision,
            "pinned_commit_sha": self.pinned_commit_sha,
            "expected_shard_count": self.expected_shard_count,
            "expected_row_group_count": self.expected_row_group_count,
            "expected_records": self.expected_total_records,
            "exclusions_config": str(self.exclusions_path),
            "physical_inventory": str(self.physical_inventory_path),
            "acquisition_mode": self.source_config.get(
                "acquisition_mode", "selective_row_group_filtered_streaming"
            ),
            "destination": str(self.destination),
            "estimated_transfer_size": self.source_config.get(
                "estimated_transfer_size"
            ),
            "estimated_raw_size": self.source_config.get("estimated_raw_size"),
            "sampling_rule": "persist all rows not matched by the exclusion config",
            "effective_concurrency": 1,
            "execution_order": "ascending pinned shard and row-group order",
        }

    def _load_physical_inventory(self) -> Tuple[dict, str]:
        try:
            raw = self.physical_inventory_path.read_bytes()
            inventory = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Could not load physical inventory {self.physical_inventory_path}: {exc}"
            ) from exc

        if (
            inventory.get("repository") != self.repository
            or inventory.get("partition") != self.split
            or inventory.get("pinned_commit_sha") != self.pinned_commit_sha
        ):
            raise RuntimeError(
                "Physical inventory repository, partition, or pinned commit "
                "does not match the materialization plan."
            )
        shards = inventory.get("shards")
        if not isinstance(shards, list) or len(shards) != self.expected_shard_count:
            raise RuntimeError(
                f"Physical inventory must contain {self.expected_shard_count} shards."
            )
        summary = inventory.get("summary", {})
        if (
            summary.get("shard_count") != self.expected_shard_count
            or summary.get("row_group_count") != self.expected_row_group_count
            or summary.get("upstream_records") != self.expected_total_records
        ):
            raise RuntimeError(
                "Physical inventory row-group or record totals do not match "
                "the accepted pinned layout."
            )
        shard_paths = [item.get("path") for item in shards]
        if len(set(shard_paths)) != self.expected_shard_count:
            raise RuntimeError("Physical inventory contains duplicate shard paths.")
        if sum(item.get("row_group_count", 0) for item in shards) != (
            self.expected_row_group_count
        ):
            raise RuntimeError("Physical inventory row-group counts do not sum.")
        if sum(item.get("records", 0) for item in shards) != (
            self.expected_total_records
        ):
            raise RuntimeError("Physical inventory record counts do not sum.")
        counted_subsets: Counter[str] = Counter()
        for shard in shards:
            row_groups = shard.get("row_groups", [])
            if len(row_groups) != shard.get("row_group_count"):
                raise RuntimeError(
                    f"Physical inventory row-group list is incomplete for {shard.get('path')}"
                )
            for index, row_group in enumerate(row_groups):
                if row_group.get("index") != index:
                    raise RuntimeError(
                        f"Physical inventory row groups are not ordered for {shard.get('path')}"
                    )
                if sum(row_group.get("subset_counts", {}).values()) != row_group.get(
                    "records"
                ):
                    raise RuntimeError(
                        f"Physical inventory subset counts do not sum for {shard.get('path')}"
                    )
                counted_subsets.update(row_group.get("subset_counts", {}))
        if dict(sorted(counted_subsets.items())) != summary.get("subset_records"):
            raise RuntimeError(
                "Physical inventory subset totals do not match row groups."
            )
        return inventory, compute_file_sha256(self.physical_inventory_path)

    def _load_exclusions(self) -> Tuple[set[str], List[str], str, Dict[str, str]]:
        exact, patterns, config_sha = load_gigaverbo_exclusions(self.exclusions_path)
        rules = get_exclusion_rules_map(self.exclusions_path)
        if not rules:
            raise RuntimeError("GigaVerbo exclusion config contains no named rules.")
        return exact, patterns, config_sha, rules

    def _fetch_upstream_inventory(self, physical_inventory: dict) -> List[dict]:
        """Resolve repository metadata at the pinned SHA and compare the audit."""
        info = HfApi().repo_info(
            repo_id=self.repository,
            repo_type="dataset",
            revision=self.pinned_commit_sha,
            files_metadata=True,
        )
        if getattr(info, "sha", None) != self.pinned_commit_sha:
            raise RuntimeError(
                "Hugging Face resolved a revision other than the configured pin: "
                f"{getattr(info, 'sha', None)!r}"
            )

        prefix = f"{self.split}/"
        siblings = [
            item
            for item in (getattr(info, "siblings", None) or [])
            if str(getattr(item, "rfilename", "")).startswith(prefix)
            and str(getattr(item, "rfilename", "")).endswith(_PARQUET_SUFFIX)
        ]
        if len(siblings) != self.expected_shard_count:
            raise RuntimeError(
                f"Pinned partition contains {len(siblings)} Parquet shards; "
                f"expected {self.expected_shard_count}."
            )

        sibling_by_path = {item.rfilename: item for item in siblings}
        if len(sibling_by_path) != len(siblings):
            raise RuntimeError("Pinned upstream inventory contains duplicate paths.")

        baseline = {item["path"]: item for item in physical_inventory["shards"]}
        if set(sibling_by_path) != set(baseline):
            missing = sorted(set(baseline) - set(sibling_by_path))
            unexpected = sorted(set(sibling_by_path) - set(baseline))
            raise RuntimeError(
                "Pinned upstream shard set differs from the audited physical "
                f"inventory (missing={missing[:3]}, unexpected={unexpected[:3]})."
            )

        inventory: List[dict] = []
        shard_pattern = re.compile(
            rf"^{re.escape(self.split)}/train-(\d{{5}})-of-(\d{{5}})\.parquet$"
        )
        for path in sorted(sibling_by_path):
            item = sibling_by_path[path]
            expected = baseline[path]
            blob_oid = getattr(item, "blob_id", None)
            size_bytes = getattr(item, "size", None)
            lfs = getattr(item, "lfs", None)
            lfs_sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
            if not isinstance(size_bytes, int) or not isinstance(blob_oid, str):
                raise RuntimeError(f"Incomplete pinned metadata for shard {path}.")
            if (
                size_bytes != expected["size_bytes"]
                or blob_oid != expected["git_blob_oid"]
                or (expected.get("lfs_sha256") and lfs_sha256 != expected["lfs_sha256"])
            ):
                raise RuntimeError(
                    f"Pinned metadata differs from the physical audit for {path}."
                )
            match = shard_pattern.fullmatch(path)
            if (
                match is None
                or int(match.group(1)) != len(inventory)
                or int(match.group(2)) != self.expected_shard_count
            ):
                raise RuntimeError(f"Unexpected pinned shard identifier: {path}")
            inventory.append(
                {
                    "path": path,
                    "size_bytes": int(size_bytes),
                    "git_blob_oid": blob_oid,
                    "lfs_sha256": lfs_sha256,
                    # Keep the audited footer totals even when resuming after
                    # earlier shards have already been checkpointed. The live
                    # footer is checked against these values when a shard is
                    # first read.
                    "row_group_count": int(expected["row_group_count"]),
                    "records": int(expected["records"]),
                }
            )
        return inventory

    def _new_manifest(
        self,
        tool_git_commit: str,
        exclusion_config_sha256: str,
        physical_inventory_sha256: str,
    ) -> MaterializationManifest:
        return MaterializationManifest(
            source="gigaverbo_v2",
            upstream_repository=self.repository,
            pinned_revision=self.pinned_revision,
            pinned_commit_sha=self.pinned_commit_sha,
            status="PARTIAL",
            tool_git_commit=tool_git_commit,
            checksum_provenance="local_sha256_per_filtered_parquet_payload",
            source_metadata={
                "schema_version": 1,
                "partition": self.split,
                "exclusion_config_path": str(
                    self.exclusions_path.relative_to(MATERIALIZER_REPO_ROOT)
                    if MATERIALIZER_REPO_ROOT in self.exclusions_path.parents
                    else self.exclusions_path
                ),
                "exclusion_config_sha256": exclusion_config_sha256,
                "physical_inventory_path": str(
                    self.physical_inventory_path.relative_to(MATERIALIZER_REPO_ROOT)
                    if MATERIALIZER_REPO_ROOT in self.physical_inventory_path.parents
                    else self.physical_inventory_path
                ),
                "physical_inventory_sha256": physical_inventory_sha256,
                "upstream_inventory": [],
                "row_groups_visited": [],
                "records_examined": 0,
                "records_encountered_per_subset": {},
                "records_excluded": 0,
                "exclusion_counts_by_subset_rule": {},
                "eligible_records": 0,
                "eligible_records_per_subset": {},
                "persisted_records_per_subset": {},
                "local_files_per_subset": {},
                "total_persisted_documents": 0,
                "total_persisted_bytes": 0,
                "sampling_rule": "full residual: persist every row whose subset is not excluded; no cap or sampling",
                "original_identifier_field": "id",
                "provenance_columns": list(_PROVENANCE_COLUMNS),
                "compression": "zstd level 6",
                "pending_row_group": None,
                "pending_output_paths": [],
                "retry_count": 0,
                "failure_records": [],
                "requested_concurrency": 1,
                "effective_concurrency": 1,
            },
        )

    def _check_manifest_compatibility(
        self,
        manifest: MaterializationManifest,
        *,
        tool_git_commit: str,
        exclusion_config_sha256: str,
        physical_inventory_sha256: str,
        require_same_tool_commit: bool,
    ) -> None:
        metadata = manifest.source_metadata or {}
        if (
            manifest.source != "gigaverbo_v2"
            or manifest.upstream_repository != self.repository
            or manifest.pinned_revision != self.pinned_revision
            or manifest.pinned_commit_sha != self.pinned_commit_sha
        ):
            raise RuntimeError(
                "Existing GigaVerbo manifest uses a different repository or pin."
            )
        if metadata.get("exclusion_config_sha256") != exclusion_config_sha256:
            raise RuntimeError(
                "Exclusion config SHA-256 mismatch with existing manifest; "
                "refusing to reuse or resume it."
            )
        if metadata.get("physical_inventory_sha256") != physical_inventory_sha256:
            raise RuntimeError(
                "Physical inventory SHA-256 mismatch with existing manifest."
            )
        if require_same_tool_commit and manifest.tool_git_commit != tool_git_commit:
            raise RuntimeError(
                "Partial materialization was produced by a different tool Git "
                "commit; refusing to mix implementations during recovery."
            )
        if not manifest.tool_git_commit:
            raise RuntimeError("Existing manifest is missing tool_git_commit.")

    def _validate_partial_payloads(self, manifest: MaterializationManifest) -> None:
        """Verify trusted checkpoint files and constrain uncommitted leftovers."""
        root = self.destination.resolve()
        metadata = manifest.source_metadata or {}
        known_paths = {str(record["relative_path"]) for record in manifest.files}
        pending_paths = set(metadata.get("pending_output_paths", []))

        for record in manifest.files:
            path = (root / record["relative_path"]).resolve()
            if root not in path.parents:
                raise RuntimeError(
                    f"Partial manifest path escapes destination: {record['relative_path']}"
                )
            if not path.is_file():
                raise RuntimeError(
                    f"Partial manifest payload is missing: {record['relative_path']}"
                )
            if path.stat().st_size != record["bytes"]:
                raise RuntimeError(
                    f"Partial payload byte count mismatch: {record['relative_path']}"
                )
            if compute_file_sha256(path) != record["sha256"]:
                raise RuntimeError(
                    f"Partial payload SHA-256 mismatch: {record['relative_path']}"
                )

        for path in root.rglob(f"*{_PARQUET_SUFFIX}"):
            relative = path.relative_to(root).as_posix()
            if relative not in known_paths and relative not in pending_paths:
                raise RuntimeError(
                    f"Untrusted Parquet payload found without a checkpoint: {relative}"
                )

        pending_group = metadata.get("pending_row_group")
        for path in root.rglob("*.partial"):
            relative = path.relative_to(root).as_posix()
            base_relative = relative[: -len(".partial")]
            if path.name == "manifest.in_progress.json.partial":
                path.unlink()
            elif pending_group and base_relative in pending_paths:
                path.unlink()
            elif path.name == "manifest.json.partial":
                path.unlink()
            else:
                raise RuntimeError(f"Untrusted partial payload found: {relative}")

    def _validate_partial_invariants(self, manifest: MaterializationManifest) -> None:
        metadata = manifest.source_metadata or {}
        encountered = metadata.get("records_encountered_per_subset", {})
        eligible_per_subset = metadata.get("eligible_records_per_subset", {})
        persisted_per_subset = metadata.get("persisted_records_per_subset", {})
        excluded = metadata.get("records_excluded", 0)
        examined = metadata.get("records_examined", 0)
        eligible = metadata.get("eligible_records", 0)
        persisted = metadata.get("total_persisted_documents", 0)
        if sum(encountered.values()) != examined:
            raise RuntimeError("Partial manifest encountered-record invariant failed.")
        if eligible != examined - excluded:
            raise RuntimeError("Partial manifest eligible-record invariant failed.")
        if sum(eligible_per_subset.values()) != eligible:
            raise RuntimeError("Partial manifest eligible-subset invariant failed.")
        if sum(persisted_per_subset.values()) != persisted:
            raise RuntimeError("Partial manifest persisted-record invariant failed.")
        if eligible_per_subset != persisted_per_subset:
            raise RuntimeError(
                "Full raw materialization checkpoint has eligible records that were "
                "not persisted."
            )

    @staticmethod
    def _completed_group_ids(manifest: MaterializationManifest) -> set[Tuple[str, int]]:
        return {
            (str(item["upstream_shard"]), int(item["row_group"]))
            for item in (manifest.source_metadata or {}).get("row_groups_visited", [])
        }

    def _counts_from_group(
        self,
        parquet_file: pq.ParquetFile,
        row_group_index: int,
        subset_column_index: int,
    ) -> Tuple[Dict[str, int], dict]:
        row_group = parquet_file.metadata.row_group(row_group_index)
        statistics = row_group.column(subset_column_index).statistics
        stat_record = {
            "min": None,
            "max": None,
            "null_count": None,
            "used_for_exact_identification": False,
            "validated_by_subset_column": False,
            "validated_by_full_row_group": False,
        }
        if statistics is not None:
            stat_record.update(
                {
                    "min": _json_safe_stat_value(
                        statistics.min if statistics.has_min_max else None
                    ),
                    "max": _json_safe_stat_value(
                        statistics.max if statistics.has_min_max else None
                    ),
                    "null_count": (
                        int(statistics.null_count)
                        if statistics.null_count is not None
                        else None
                    ),
                }
            )
            if (
                statistics.has_min_max
                and statistics.min == statistics.max
                and statistics.null_count == 0
                and statistics.num_values == row_group.num_rows
            ):
                value = _json_safe_stat_value(statistics.min)
                if value is None:
                    raise RuntimeError(
                        f"Null subset statistic in row group {row_group_index}."
                    )
                stat_record["used_for_exact_identification"] = True
                return {value: int(row_group.num_rows)}, stat_record

        counts = self._read_subset_counts(parquet_file, row_group_index)
        stat_record["validated_by_subset_column"] = True
        return counts, stat_record

    @staticmethod
    def _read_subset_counts(
        parquet_file: pq.ParquetFile, row_group_index: int
    ) -> Dict[str, int]:
        subset_table = parquet_file.read_row_group(row_group_index, columns=["subset"])
        values = subset_table.column("subset").to_pylist()
        if any(value is None for value in values):
            raise RuntimeError(
                f"Null subset value encountered in row group {row_group_index}."
            )
        counts = Counter(str(value) for value in values)
        if (
            sum(counts.values())
            != parquet_file.metadata.row_group(row_group_index).num_rows
        ):
            raise RuntimeError(f"Subset count mismatch in row group {row_group_index}.")
        return _counter_dict(counts)

    @staticmethod
    def _compare_row_group_to_audit(
        upstream_shard: str,
        row_group_index: int,
        subset_counts: Dict[str, int],
        physical_by_path: dict,
    ) -> None:
        expected_shard = physical_by_path[upstream_shard]
        expected_groups = expected_shard["row_groups"]
        if row_group_index >= len(expected_groups):
            raise RuntimeError(
                f"Row group {row_group_index} is absent from physical audit for "
                f"{upstream_shard}."
            )
        expected_counts = expected_groups[row_group_index]["subset_counts"]
        if subset_counts != expected_counts:
            raise RuntimeError(
                f"Subset membership differs from physical audit in "
                f"{upstream_shard} row group {row_group_index}."
            )

    def _write_row_group_subset(
        self,
        table: pa.Table,
        *,
        subset: str,
        upstream_shard: str,
        row_group_index: int,
    ) -> dict:
        encoded_subset = quote(subset, safe="._-")
        source_stem = PurePosixPath(upstream_shard).name.removesuffix(_PARQUET_SUFFIX)
        relative_path = (
            PurePosixPath(f"subset={encoded_subset}")
            / f"{source_stem}__row-group-{row_group_index:05d}{_PARQUET_SUFFIX}"
        ).as_posix()
        output_path = self.destination / relative_path

        if any(name in table.column_names for name in _PROVENANCE_COLUMNS):
            raise RuntimeError(
                "Upstream schema already contains a reserved provenance column."
            )
        row_count = table.num_rows
        table = table.append_column(
            _PROVENANCE_COLUMNS[0],
            pa.array([upstream_shard] * row_count, type=pa.string()),
        )
        table = table.append_column(
            _PROVENANCE_COLUMNS[1],
            pa.array([row_group_index] * row_count, type=pa.int32()),
        )
        table = table.append_column(
            _PROVENANCE_COLUMNS[2],
            pa.array([self.pinned_commit_sha] * row_count, type=pa.string()),
        )
        payload_bytes, payload_sha = _atomic_write_parquet(table, output_path)
        return {
            "relative_path": relative_path,
            "upstream_identifier": (
                f"{upstream_shard}#row_group={row_group_index}#subset={subset}"
            ),
            "url": (
                f"https://huggingface.co/datasets/{self.repository}/resolve/"
                f"{self.pinned_commit_sha}/{upstream_shard}"
            ),
            "bytes": payload_bytes,
            "sha256": payload_sha,
            "checksum_source": "local_sha256",
            "upstream_shard": upstream_shard,
            "upstream_row_group": row_group_index,
            "upstream_subset": subset,
            "upstream_commit": self.pinned_commit_sha,
            "records": row_count,
        }

    def _process_row_group(
        self,
        manifest: MaterializationManifest,
        parquet_file: pq.ParquetFile,
        upstream_shard: str,
        row_group_index: int,
        physical_by_path: dict,
        exact_blocked: set[str],
        patterns: List[str],
        rules: Dict[str, str],
        in_progress_path: Path,
    ) -> None:
        metadata = manifest.source_metadata
        assert metadata is not None
        group_id = {"upstream_shard": upstream_shard, "row_group": row_group_index}
        previous_pending = metadata.get("pending_row_group")
        previous_paths = list(metadata.get("pending_output_paths", []))
        if previous_pending != group_id:
            previous_paths = []
        metadata["pending_row_group"] = group_id
        metadata["pending_output_paths"] = previous_paths
        manifest.status = "PARTIAL"
        manifest.save(in_progress_path)

        schema = parquet_file.schema_arrow
        missing = _REQUIRED_SOURCE_COLUMNS - set(schema.names)
        if missing:
            raise RuntimeError(
                f"Upstream schema is missing required columns: {missing}"
            )
        subset_column_index = schema.get_field_index("subset")
        row_group_metadata = parquet_file.metadata.row_group(row_group_index)
        subset_counts, statistics = self._counts_from_group(
            parquet_file, row_group_index, subset_column_index
        )
        if (
            statistics["used_for_exact_identification"]
            and subset_counts
            and all(
                match_subset_exclusion(subset, exact_blocked, patterns, rules)
                is not None
                for subset in subset_counts
            )
        ):
            # A false single-value statistic could otherwise make an entirely
            # excluded row group appear safe to skip while hiding eligible
            # rows. Validate the small subset column before skipping its text.
            observed_counts = self._read_subset_counts(parquet_file, row_group_index)
            if observed_counts != subset_counts:
                raise RuntimeError(
                    f"Excluded-only subset statistics differ from the subset column "
                    f"for {upstream_shard} row group {row_group_index}."
                )
            statistics["validated_by_subset_column"] = True
        self._compare_row_group_to_audit(
            upstream_shard, row_group_index, subset_counts, physical_by_path
        )

        exclusion_counts: Dict[str, Dict[str, int]] = {}
        eligible_counts: Dict[str, int] = {}
        for subset, count in sorted(subset_counts.items()):
            matched_rule = match_subset_exclusion(
                subset, exact_blocked, patterns, rules
            )
            if matched_rule is None:
                eligible_counts[subset] = int(count)
            else:
                exclusion_counts.setdefault(subset, {})[matched_rule] = int(count)

        eligible_table: Optional[pa.Table] = None
        output_records: List[dict] = []
        if eligible_counts:
            source_table = parquet_file.read_row_group(row_group_index)
            actual = Counter(
                str(value) for value in source_table.column("subset").to_pylist()
            )
            if dict(sorted(actual.items())) != subset_counts:
                raise RuntimeError(
                    f"Full row-group read differs from subset-column inventory for "
                    f"{upstream_shard} row group {row_group_index}."
                )
            statistics["validated_by_full_row_group"] = True
            subset_values = source_table.column("subset").to_pylist()
            eligible_mask = pa.array(
                [
                    match_subset_exclusion(value, exact_blocked, patterns, rules)
                    is None
                    for value in subset_values
                ],
                type=pa.bool_(),
            )
            eligible_table = source_table.filter(eligible_mask)
            if eligible_table.num_rows != sum(eligible_counts.values()):
                raise RuntimeError(
                    f"Filtered row count differs from subset inventory for "
                    f"{upstream_shard} row group {row_group_index}."
                )
            for subset in sorted(eligible_counts):
                mask = pc.equal(
                    eligible_table.column("subset"), pa.scalar(subset, pa.string())
                )
                subset_table = pc.filter(eligible_table, mask)
                if subset_table.num_rows != eligible_counts[subset]:
                    raise RuntimeError(
                        f"Filtered count mismatch for subset {subset} in "
                        f"{upstream_shard} row group {row_group_index}."
                    )
                output_records.append(
                    {
                        "subset": subset,
                        "table": subset_table,
                    }
                )

        expected_paths = [
            (
                PurePosixPath(f"subset={quote(record['subset'], safe='._-')}")
                / f"{PurePosixPath(upstream_shard).name.removesuffix(_PARQUET_SUFFIX)}"
                f"__row-group-{row_group_index:05d}{_PARQUET_SUFFIX}"
            ).as_posix()
            for record in output_records
        ]
        if previous_paths and set(previous_paths) != set(expected_paths):
            raise RuntimeError(
                f"Pending output paths differ on recovery for {upstream_shard} "
                f"row group {row_group_index}."
            )
        metadata["pending_output_paths"] = expected_paths
        manifest.save(in_progress_path)

        file_records = []
        for record in output_records:
            file_records.append(
                self._write_row_group_subset(
                    record["table"],
                    subset=record["subset"],
                    upstream_shard=upstream_shard,
                    row_group_index=row_group_index,
                )
            )

        records_encountered = Counter(metadata["records_encountered_per_subset"])
        eligible_per_subset = Counter(metadata["eligible_records_per_subset"])
        persisted_per_subset = Counter(metadata["persisted_records_per_subset"])
        local_files_per_subset = Counter(metadata["local_files_per_subset"])
        for subset, count in subset_counts.items():
            records_encountered[subset] += int(count)
        for subset, count in eligible_counts.items():
            eligible_per_subset[subset] += int(count)
            persisted_per_subset[subset] += int(count)
        for record in file_records:
            local_files_per_subset[record["upstream_subset"]] += 1

        all_exclusion_counts: Dict[str, Dict[str, int]] = metadata[
            "exclusion_counts_by_subset_rule"
        ]
        for subset, rules_for_subset in exclusion_counts.items():
            target = all_exclusion_counts.setdefault(subset, {})
            for rule, count in rules_for_subset.items():
                target[rule] = int(target.get(rule, 0)) + int(count)

        examined = sum(subset_counts.values())
        excluded = sum(
            count
            for rule_counts in exclusion_counts.values()
            for count in rule_counts.values()
        )
        eligible = sum(eligible_counts.values())
        compressed_bytes = sum(
            row_group_metadata.column(index).total_compressed_size
            for index in range(row_group_metadata.num_columns)
        )
        row_group_record = {
            **group_id,
            "records_examined": int(examined),
            "subset_counts": subset_counts,
            "subset_min_max_statistics": statistics,
            "compressed_row_group_bytes": int(compressed_bytes),
            "uncompressed_row_group_bytes": int(row_group_metadata.total_byte_size),
            "records_excluded_by_subset_rule": exclusion_counts,
            "eligible_records_per_subset": eligible_counts,
            "persisted_records_per_subset": dict(eligible_counts),
            "persisted_files": [record["relative_path"] for record in file_records],
        }

        metadata["records_encountered_per_subset"] = _counter_dict(records_encountered)
        metadata["eligible_records_per_subset"] = _counter_dict(eligible_per_subset)
        metadata["persisted_records_per_subset"] = _counter_dict(persisted_per_subset)
        metadata["local_files_per_subset"] = _counter_dict(local_files_per_subset)
        metadata["records_examined"] = int(metadata["records_examined"] + examined)
        metadata["records_excluded"] = int(metadata["records_excluded"] + excluded)
        metadata["eligible_records"] = int(metadata["eligible_records"] + eligible)
        metadata["total_persisted_documents"] = int(
            metadata["total_persisted_documents"] + eligible
        )
        metadata["total_persisted_bytes"] = int(
            metadata["total_persisted_bytes"]
            + sum(record["bytes"] for record in file_records)
        )
        metadata["row_groups_visited"].append(row_group_record)
        metadata["pending_row_group"] = None
        metadata["pending_output_paths"] = []
        manifest.files.extend(file_records)
        manifest.files.sort(key=lambda item: item["relative_path"])
        manifest.total_files = len(manifest.files)
        manifest.total_bytes = sum(item["bytes"] for item in manifest.files)
        self._validate_partial_invariants(manifest)
        manifest.save(in_progress_path)

    def _verify_output_parquet(self, manifest: MaterializationManifest) -> List[str]:
        errors: List[str] = []
        root = self.destination.resolve()
        try:
            exact, patterns, _, rules = self._load_exclusions()
        except Exception as exc:
            return [str(exc)]

        for record in manifest.files:
            relative_path = str(record.get("relative_path", ""))
            path = (root / relative_path).resolve()
            if root not in path.parents or not path.is_file():
                continue
            try:
                parquet_file = pq.ParquetFile(path)
                required = _REQUIRED_SOURCE_COLUMNS | set(_PROVENANCE_COLUMNS)
                missing = required - set(parquet_file.schema_arrow.names)
                if missing:
                    errors.append(
                        f"Parquet schema missing provenance fields in {relative_path}: {sorted(missing)}"
                    )
                    continue
                if parquet_file.metadata.num_rows != record.get("records"):
                    errors.append(f"Record count mismatch for {relative_path}.")
                    continue
                columns = ["subset", *_PROVENANCE_COLUMNS]
                errors_before_file = len(errors)
                for batch in parquet_file.iter_batches(
                    batch_size=65536, columns=columns
                ):
                    data = batch.to_pydict()
                    for subset, shard, row_group, commit in zip(
                        data["subset"],
                        data[_PROVENANCE_COLUMNS[0]],
                        data[_PROVENANCE_COLUMNS[1]],
                        data[_PROVENANCE_COLUMNS[2]],
                    ):
                        if subset != record.get("upstream_subset"):
                            errors.append(
                                f"Subset mismatch inside {relative_path}: {subset!r}."
                            )
                            break
                        if (
                            match_subset_exclusion(subset, exact, patterns, rules)
                            is not None
                        ):
                            errors.append(
                                f"Excluded subset persisted in {relative_path}: {subset!r}."
                            )
                            break
                        if (
                            shard != record.get("upstream_shard")
                            or row_group != record.get("upstream_row_group")
                            or commit != self.pinned_commit_sha
                        ):
                            errors.append(
                                f"Row-group provenance mismatch in {relative_path}."
                            )
                            break
                    if len(errors) > errors_before_file:
                        break
            except Exception as exc:
                errors.append(
                    f"Could not inspect Parquet payload {relative_path}: {exc}"
                )
        return errors

    def _validate_manifest_invariants(
        self,
        manifest: MaterializationManifest,
        physical_inventory: dict,
        *,
        complete: bool,
    ) -> List[str]:
        errors: List[str] = []
        metadata = manifest.source_metadata or {}
        encountered = metadata.get("records_encountered_per_subset", {})
        eligible_per_subset = metadata.get("eligible_records_per_subset", {})
        persisted_per_subset = metadata.get("persisted_records_per_subset", {})
        local_files_per_subset = metadata.get("local_files_per_subset", {})
        examined = int(metadata.get("records_examined", 0))
        excluded = int(metadata.get("records_excluded", 0))
        eligible = int(metadata.get("eligible_records", 0))
        persisted = int(metadata.get("total_persisted_documents", 0))

        if not manifest.tool_git_commit:
            errors.append("Manifest is missing tool_git_commit provenance")
        if not manifest.acquisition_started_at:
            errors.append("Manifest is missing acquisition start timestamp")
        if complete and not manifest.acquisition_completed_at:
            errors.append(
                "COMPLETE manifest is missing acquisition completion timestamp"
            )
        if sum(encountered.values()) != examined:
            errors.append("sum(records_encountered_per_subset) != records_examined")
        if eligible != examined - excluded:
            errors.append("eligible_records != records_examined - records_excluded")
        if sum(eligible_per_subset.values()) != eligible:
            errors.append("sum(eligible_records_per_subset) != eligible_records")
        if sum(persisted_per_subset.values()) != persisted:
            errors.append(
                "sum(persisted_records_per_subset) != total_persisted_documents"
            )
        if eligible_per_subset != persisted_per_subset:
            errors.append(
                "Full raw reservoir persisted counts differ from eligible counts."
            )
        if persisted != metadata.get("total_persisted_documents"):
            errors.append("total_persisted_documents is not an integer total.")
        if metadata.get("total_persisted_bytes") != manifest.total_bytes:
            errors.append("total_persisted_bytes does not match manifest total_bytes")
        if metadata.get("local_files_per_subset") != local_files_per_subset:
            errors.append("local_files_per_subset is malformed.")

        file_rows: Counter[str] = Counter()
        file_counts: Counter[str] = Counter()
        seen_paths = set()
        for record in manifest.files:
            relative = str(record.get("relative_path", ""))
            subset = str(record.get("upstream_subset", ""))
            if relative in seen_paths:
                errors.append(f"Duplicate local file entry: {relative}")
            seen_paths.add(relative)
            file_rows[subset] += int(record.get("records", 0))
            file_counts[subset] += 1
            if record.get("upstream_commit") != self.pinned_commit_sha or record.get(
                "upstream_shard"
            ) not in {item["path"] for item in metadata.get("upstream_inventory", [])}:
                errors.append(f"Invalid upstream file provenance: {relative}")
        if _counter_dict(file_rows) != persisted_per_subset:
            errors.append(
                "Persisted file rows do not match persisted_records_per_subset"
            )
        if _counter_dict(file_counts) != local_files_per_subset:
            errors.append("File counts do not match local_files_per_subset")

        row_groups = metadata.get("row_groups_visited", [])
        visited_ids = {
            (item.get("upstream_shard"), int(item.get("row_group", -1)))
            for item in row_groups
        }
        if len(visited_ids) != len(row_groups):
            errors.append("Duplicate row group in row_groups_visited")
        row_group_encountered: Counter[str] = Counter()
        row_group_excluded = 0
        row_group_eligible: Counter[str] = Counter()
        try:
            exact_blocked, patterns, _, rules = self._load_exclusions()
        except Exception as exc:
            errors.append(str(exc))
            exact_blocked, patterns, rules = set(), [], {}
        physical_groups = {
            (shard["path"], item["index"]): item
            for shard in physical_inventory["shards"]
            for item in shard["row_groups"]
        }
        for item in row_groups:
            subset_counts = item.get("subset_counts", {})
            group_key = (
                item.get("upstream_shard"),
                int(item.get("row_group", -1)),
            )
            audited_group = physical_groups.get(group_key)
            if audited_group is None:
                errors.append(
                    f"Visited row group is absent from physical audit: {group_key}"
                )
            elif subset_counts != audited_group.get("subset_counts") or item.get(
                "records_examined"
            ) != audited_group.get("records"):
                errors.append(
                    f"Visited row group differs from physical audit: {group_key}"
                )
            if sum(subset_counts.values()) != item.get("records_examined"):
                errors.append(
                    f"Row-group subset counts do not sum for {item.get('upstream_shard')}"
                )
            expected_eligible: Dict[str, int] = {}
            expected_excluded: Dict[str, Dict[str, int]] = {}
            for subset, count in subset_counts.items():
                matched_rule = match_subset_exclusion(
                    subset, exact_blocked, patterns, rules
                )
                if matched_rule is None:
                    expected_eligible[subset] = int(count)
                else:
                    expected_excluded.setdefault(subset, {})[matched_rule] = int(count)
            if item.get("eligible_records_per_subset") != expected_eligible:
                errors.append(
                    f"Eligible row-group counts differ from exclusions: {group_key}"
                )
            if item.get("records_excluded_by_subset_rule") != expected_excluded:
                errors.append(
                    f"Exclusion row-group counts differ from rules: {group_key}"
                )
            row_group_encountered.update(subset_counts)
            for subset, rule_counts in item.get(
                "records_excluded_by_subset_rule", {}
            ).items():
                row_group_excluded += sum(rule_counts.values())
            row_group_eligible.update(item.get("eligible_records_per_subset", {}))
        if _counter_dict(row_group_encountered) != encountered:
            errors.append(
                "Visited row-group counts do not match records_encountered_per_subset"
            )
        if row_group_excluded != excluded:
            errors.append("Visited row-group exclusions do not match records_excluded")
        if _counter_dict(row_group_eligible) != eligible_per_subset:
            errors.append(
                "Visited row-group eligible counts do not match manifest totals"
            )

        inventory = metadata.get("upstream_inventory", [])
        if len(inventory) != self.expected_shard_count:
            errors.append(
                f"Manifest upstream inventory does not contain {self.expected_shard_count} shards"
            )
        if complete:
            expected_ids = {
                (shard["path"], row_group["index"])
                for shard in physical_inventory["shards"]
                for row_group in shard["row_groups"]
            }
            if visited_ids != expected_ids:
                errors.append("COMPLETE manifest did not visit every pinned row group")
            if len(row_groups) != self.expected_row_group_count:
                errors.append(
                    f"COMPLETE manifest visited {len(row_groups)} row groups; "
                    f"expected {self.expected_row_group_count}"
                )
            if examined != self.expected_total_records:
                errors.append(
                    f"COMPLETE manifest examined {examined} records; "
                    f"expected {self.expected_total_records}"
                )
            baseline_by_path = {
                item["path"]: item for item in physical_inventory["shards"]
            }
            inventory_by_path = {item.get("path"): item for item in inventory}
            if len(inventory_by_path) != len(inventory):
                errors.append("Duplicate shard path in upstream inventory")
            if set(inventory_by_path) != set(baseline_by_path):
                errors.append("Upstream inventory paths do not match physical audit")
            for shard in inventory:
                baseline = baseline_by_path.get(shard.get("path"))
                if baseline is None:
                    errors.append(
                        f"Unexpected upstream shard in manifest: {shard['path']}"
                    )
                elif (
                    shard.get("size_bytes") != baseline["size_bytes"]
                    or shard.get("git_blob_oid") != baseline["git_blob_oid"]
                    or shard.get("lfs_sha256") != baseline.get("lfs_sha256")
                    or shard.get("row_group_count") != baseline["row_group_count"]
                    or shard.get("records") != baseline["records"]
                ):
                    errors.append(
                        f"Completed shard metadata differs from physical audit: {shard['path']}"
                    )
            if sum(shard.get("size_bytes", 0) for shard in inventory) != sum(
                shard["size_bytes"] for shard in physical_inventory["shards"]
            ):
                errors.append(
                    "Upstream inventory byte total differs from physical audit"
                )
            expected_oids = {item["path"]: item["git_blob_oid"] for item in inventory}
            if manifest.upstream_shard_oids != expected_oids:
                errors.append("Manifest upstream_shard_oids do not match inventory")
            tracked_paths = {record.get("relative_path") for record in manifest.files}
            actual_paths = {
                path.relative_to(self.destination).as_posix()
                for path in self.destination.rglob(f"*{_PARQUET_SUFFIX}")
            }
            if actual_paths != tracked_paths:
                errors.append(
                    "Local Parquet file set differs from the COMPLETE manifest "
                    f"(untracked={sorted(actual_paths - tracked_paths)[:3]}, "
                    f"missing={sorted(tracked_paths - actual_paths)[:3]})."
                )
        return errors

    def _verify_manifest_object(
        self,
        manifest: MaterializationManifest,
        physical_inventory: dict,
        exclusion_config_sha256: str,
        physical_inventory_sha256: str,
        *,
        check_runtime_state: bool,
    ) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        metadata = manifest.source_metadata or {}
        if metadata.get("exclusion_config_sha256") != exclusion_config_sha256:
            errors.append("Exclusion config SHA-256 mismatch.")
        if metadata.get("physical_inventory_sha256") != physical_inventory_sha256:
            errors.append("Physical inventory SHA-256 mismatch.")
        errors.extend(
            self._validate_manifest_invariants(
                manifest,
                physical_inventory,
                complete=manifest.status == "COMPLETE",
            )
        )
        generic_valid, generic_errors = manifest.verify(
            self.destination,
            check_partial_files=False,
            check_runtime_state=False,
        )
        errors.extend(
            error
            for error in generic_errors
            if not error.startswith("An interrupted or unfinished")
        )
        if not generic_valid and manifest.status != "COMPLETE":
            errors.extend(
                error
                for error in generic_errors
                if error.startswith("Manifest status is")
            )
        if manifest.status == "COMPLETE":
            for partial in self.destination.rglob("*.partial*"):
                errors.append(f"Orphaned partial file detected: {partial}")
            if (
                check_runtime_state
                and (self.destination / "manifest.in_progress.json").exists()
            ):
                errors.append("An unfinished in-progress manifest exists.")
            errors.extend(self._verify_output_parquet(manifest))
        return not errors, errors

    def verify(self) -> Tuple[bool, List[str]]:
        """Verify a completed GigaVerbo manifest, payload hashes, and provenance."""
        manifest_path = self.destination / "manifest.json"
        if not manifest_path.is_file():
            return False, [f"Manifest not found: {manifest_path}"]
        try:
            manifest = MaterializationManifest.load(manifest_path)
            physical_inventory, physical_sha = self._load_physical_inventory()
            _, _, exclusions_sha, _ = self._load_exclusions()
        except Exception as exc:
            return False, [str(exc)]
        try:
            return self._verify_manifest_object(
                manifest,
                physical_inventory,
                exclusions_sha,
                physical_sha,
                check_runtime_state=True,
            )
        except Exception as exc:
            return False, [f"Manifest verification failed: {exc}"]

    def _record_failure(
        self,
        manifest: MaterializationManifest,
        in_progress_path: Path,
        *,
        upstream_shard: Optional[str],
        attempt: int,
        error: Exception,
    ) -> None:
        metadata = manifest.source_metadata
        assert metadata is not None
        metadata["failure_records"].append(
            {
                "at": _utc_now(),
                "upstream_shard": upstream_shard,
                "attempt": attempt,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        if attempt > 0:
            metadata["retry_count"] = int(metadata.get("retry_count", 0)) + 1
        manifest.failure_reasons[upstream_shard or "upstream_inventory"] = (
            f"{type(error).__name__}: {error}"
        )
        manifest.status = "FAILED"
        manifest.save(in_progress_path)

    def _prepare_destination(self, in_progress_path: Path) -> None:
        self.destination.mkdir(parents=True, exist_ok=True)
        if (
            not in_progress_path.exists()
            and not (self.destination / "manifest.json").exists()
        ):
            existing = [path for path in self.destination.rglob("*") if path.is_file()]
            if existing:
                raise RuntimeError(
                    "Destination contains files without a trusted GigaVerbo manifest; "
                    f"refusing to overwrite: {existing[0]}"
                )

    def materialize(
        self,
        concurrency: int = 1,
        timeout: int = 25,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> MaterializationManifest:
        """Stream and persist the complete exclusion-filtered residual reservoir."""
        del timeout, kwargs
        if concurrency < 1:
            raise ValueError("concurrency must be positive")

        tool_git_commit = _get_clean_tool_git_commit()
        physical_inventory, physical_inventory_sha = self._load_physical_inventory()
        exact_blocked, patterns, exclusion_sha, rules = self._load_exclusions()
        attempts_allowed = max(0, int(max_retries)) + 1
        in_progress_path = self.destination / "manifest.in_progress.json"
        manifest_path = self.destination / "manifest.json"
        self._prepare_destination(in_progress_path)

        if manifest_path.is_file():
            manifest = MaterializationManifest.load(manifest_path)
            self._check_manifest_compatibility(
                manifest,
                tool_git_commit=tool_git_commit,
                exclusion_config_sha256=exclusion_sha,
                physical_inventory_sha256=physical_inventory_sha,
                require_same_tool_commit=False,
            )
            valid, errors = self._verify_manifest_object(
                manifest,
                physical_inventory,
                exclusion_sha,
                physical_inventory_sha,
                check_runtime_state=False,
            )
            if not valid:
                raise RuntimeError(
                    "Existing COMPLETE GigaVerbo manifest failed verification; "
                    "refusing to re-bless payloads: " + "; ".join(errors[:8])
                )
            if in_progress_path.is_file():
                partial = MaterializationManifest.load(in_progress_path)
                self._check_manifest_compatibility(
                    partial,
                    tool_git_commit=tool_git_commit,
                    exclusion_config_sha256=exclusion_sha,
                    physical_inventory_sha256=physical_inventory_sha,
                    require_same_tool_commit=False,
                )
                in_progress_path.unlink()
            return manifest

        if in_progress_path.is_file():
            manifest = MaterializationManifest.load(in_progress_path)
            if manifest.status not in {"PARTIAL", "FAILED"}:
                raise RuntimeError(
                    f"Unsupported in-progress manifest status: {manifest.status}"
                )
            self._check_manifest_compatibility(
                manifest,
                tool_git_commit=tool_git_commit,
                exclusion_config_sha256=exclusion_sha,
                physical_inventory_sha256=physical_inventory_sha,
                require_same_tool_commit=True,
            )
            self._validate_partial_payloads(manifest)
            self._validate_partial_invariants(manifest)
            manifest.status = "PARTIAL"
            manifest.failure_reasons = {}
        else:
            manifest = self._new_manifest(
                tool_git_commit,
                exclusion_sha,
                physical_inventory_sha,
            )

        manifest.source_metadata["requested_concurrency"] = int(concurrency)
        manifest.source_metadata["effective_concurrency"] = 1
        manifest.save(in_progress_path)
        upstream_inventory = None
        for attempt in range(attempts_allowed):
            try:
                upstream_inventory = self._fetch_upstream_inventory(physical_inventory)
                break
            except Exception as exc:
                self._record_failure(
                    manifest,
                    in_progress_path,
                    upstream_shard=None,
                    attempt=attempt,
                    error=exc,
                )
                if attempt + 1 >= attempts_allowed:
                    return manifest
                time.sleep(min(2**attempt, 30))
        if upstream_inventory is None:
            return manifest
        manifest.source_metadata["upstream_inventory"] = upstream_inventory
        manifest.failure_reasons.pop("upstream_inventory", None)
        manifest.upstream_shard_oids = {
            item["path"]: item["git_blob_oid"]
            for item in manifest.source_metadata["upstream_inventory"]
        }
        manifest.save(in_progress_path)

        physical_by_path = {item["path"]: item for item in physical_inventory["shards"]}
        upstream_by_path = {
            item["path"]: item
            for item in manifest.source_metadata["upstream_inventory"]
        }
        completed_groups = self._completed_group_ids(manifest)
        try:
            fs = HfFileSystem()
        except Exception as exc:
            self._record_failure(
                manifest,
                in_progress_path,
                upstream_shard=None,
                attempt=0,
                error=exc,
            )
            return manifest

        for upstream_shard in sorted(upstream_by_path):
            if all(
                (upstream_shard, row_group["index"]) in completed_groups
                for row_group in physical_by_path[upstream_shard]["row_groups"]
            ):
                continue
            remote_path = (
                f"datasets/{self.repository}@{self.pinned_commit_sha}/{upstream_shard}"
            )
            shard_complete = False
            for attempt in range(attempts_allowed):
                try:
                    with fs.open(remote_path, "rb") as remote_file:
                        parquet_file = pq.ParquetFile(remote_file)
                        upstream_record = upstream_by_path[upstream_shard]
                        audited_shard = physical_by_path[upstream_shard]
                        if (
                            parquet_file.num_row_groups
                            != audited_shard["row_group_count"]
                            or parquet_file.metadata.num_rows
                            != audited_shard["records"]
                        ):
                            raise RuntimeError(
                                f"Parquet footer differs from physical inventory for "
                                f"{upstream_shard}."
                            )
                        upstream_record["row_group_count"] = int(
                            parquet_file.num_row_groups
                        )
                        upstream_record["records"] = int(parquet_file.metadata.num_rows)
                        manifest.save(in_progress_path)
                        if parquet_file.schema_arrow.get_field_index("subset") < 0:
                            raise RuntimeError(
                                f"Pinned shard has no subset column: {upstream_shard}"
                            )
                        for row_group_index in range(parquet_file.num_row_groups):
                            if (upstream_shard, row_group_index) in completed_groups:
                                continue
                            self._process_row_group(
                                manifest,
                                parquet_file,
                                upstream_shard,
                                row_group_index,
                                physical_by_path,
                                exact_blocked,
                                patterns,
                                rules,
                                in_progress_path,
                            )
                            completed_groups.add((upstream_shard, row_group_index))
                    shard_complete = True
                    break
                except Exception as exc:
                    self._record_failure(
                        manifest,
                        in_progress_path,
                        upstream_shard=upstream_shard,
                        attempt=attempt,
                        error=exc,
                    )
                    if attempt + 1 >= attempts_allowed:
                        return manifest
                    time.sleep(min(2**attempt, 30))
                    # The pending row group is replayed from the pinned shard;
                    # completed groups and their hashes remain checkpointed.
            if not shard_complete:
                return manifest
            manifest.failure_reasons.pop(upstream_shard, None)

        metadata = manifest.source_metadata
        assert metadata is not None
        manifest.files.sort(key=lambda item: item["relative_path"])
        manifest.total_files = len(manifest.files)
        manifest.total_bytes = sum(item["bytes"] for item in manifest.files)
        metadata["total_persisted_bytes"] = manifest.total_bytes
        self._validate_partial_invariants(manifest)
        manifest.status = "COMPLETE"
        manifest.acquisition_completed_at = _utc_now()
        valid, errors = self._verify_manifest_object(
            manifest,
            physical_inventory,
            exclusion_sha,
            physical_inventory_sha,
            check_runtime_state=False,
        )
        if not valid:
            manifest.status = "FAILED"
            metadata["failure_records"].append(
                {
                    "at": _utc_now(),
                    "upstream_shard": None,
                    "attempt": 0,
                    "error": "Final verification failed: " + "; ".join(errors[:8]),
                }
            )
            manifest.save(in_progress_path)
            return manifest

        manifest.save(manifest_path)
        in_progress_path.unlink(missing_ok=True)
        return manifest
