from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cambacica.model.baseline import (
    BaselineValidationError,
    DEFAULT_BASELINE_PATH,
    load_training_baseline,
    validate_training_baseline,
)
from cambacica.model.config import get_architecture


def test_versioned_baseline_loads_and_matches_architecture_c() -> None:
    baseline = load_training_baseline()
    approved = baseline["scientist_approved"]
    architecture = baseline["frozen_inputs"]["architecture"]

    assert baseline["schema_version"] == "cambacica-training-baseline-v1"
    assert approved["architecture_id"] == "C"
    assert approved["context_length_tokens"] == 4096
    assert approved["embedding_output_weight_tying"] is True
    assert approved["parameter_count"] == get_architecture("C").parameter_count(
        tie_weights=True
    )
    assert approved["reference_compute_precision"] == "bf16"
    assert architecture == get_architecture("C").to_dict()
    assert baseline["provisional_hyperparameters"] == {
        "status": "none_selected",
        "values": {},
    }


def test_baseline_file_is_versioned_under_configs() -> None:
    assert DEFAULT_BASELINE_PATH.name == "architecture_c_c3_6_v1.json"
    assert DEFAULT_BASELINE_PATH.parent.name == "configs"
    assert json.loads(DEFAULT_BASELINE_PATH.read_text(encoding="utf-8"))["baseline_id"] == (
        "c3.6-architecture-c-v1"
    )


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("scientist_approved", "architecture_id", "B", "architecture_id"),
        ("scientist_approved", "context_length_tokens", 2048, "context_length_tokens"),
        (
            "scientist_approved",
            "embedding_output_weight_tying",
            False,
            "embedding_output_weight_tying",
        ),
        ("scientist_approved", "parameter_count", 176180225, "parameter_count"),
        (
            "frozen_inputs",
            "tokenizer_identity",
            {"candidate": "unigram-v24576", "vocab_size": 24576},
            "tokenizer_identity",
        ),
    ],
)
def test_baseline_rejects_drift_in_approved_identity(
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    baseline = copy.deepcopy(load_training_baseline())
    target = baseline[section]
    target[field] = value

    with pytest.raises(BaselineValidationError, match=message):
        validate_training_baseline(baseline)


def test_baseline_rejects_boolean_used_as_context_integer() -> None:
    baseline = copy.deepcopy(load_training_baseline())
    baseline["scientist_approved"]["context_length_tokens"] = True

    with pytest.raises(BaselineValidationError, match="context_length_tokens"):
        validate_training_baseline(baseline)


def test_baseline_rejects_scalability_matrix_token_count_drift() -> None:
    baseline = copy.deepcopy(load_training_baseline())
    baseline["scalability_validation_proposal"]["cells"][0]["tokens_per_update"] = 8192

    with pytest.raises(
        BaselineValidationError, match="scalability_validation_proposal.cells"
    ):
        validate_training_baseline(baseline)


def test_baseline_loader_reports_invalid_json(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")

    with pytest.raises(BaselineValidationError, match="could not read baseline"):
        load_training_baseline(invalid)
