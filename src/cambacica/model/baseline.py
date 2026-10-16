"""Validation for the versioned C3.6 Architecture-C training baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cambacica.hardware import ORION_GPU0_PHYSICAL_INDEX, ORION_GPU0_UUID
from cambacica.model.config import get_architecture

BASELINE_SCHEMA_VERSION = "cambacica-training-baseline-v1"
DEFAULT_BASELINE_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "architecture_c_c3_6_v1.json"
)

TOKENIZER_IDENTITY = {
    "candidate": "unigram-v24576",
    "algorithm": "SentencePiece Unigram",
    "vocab_size": 24_576,
    "sentencepiece_version": "0.2.2",
    "model_sha256": "c8982a23d3c5ecc0a61d300edc1e354e38f1358aa3db5420d5bf19f7b95cbf23",
    "vocab_sha256": "f27e38de1a1e0e2e96ac00f50382d1be749d2b357cb3e8ad9ee43bf8ae972d9f",
    "special_token_ids": {"unk": 0, "bos": 1, "eos": 2},
    "pad_id": -1,
    "byte_fallback_pieces": 256,
}

_APPROVED_VALUES = {
    "architecture_id": "C",
    "context_length_tokens": 4_096,
    "embedding_output_weight_tying": True,
    "parameter_count": 176_180_224,
    "reference_compute_precision": "bf16",
    "target_hardware": {
        "site": "Orion",
        "hostname": "cenpesor",
        "physical_gpu_index": ORION_GPU0_PHYSICAL_INDEX,
        "gpu_uuid": ORION_GPU0_UUID,
    },
}
_EXPECTED_SCALABILITY_CELLS = [
    {
        "microbatch_size": microbatch,
        "gradient_accumulation_steps": accumulation,
        "tokens_per_update": microbatch * accumulation * 4_096,
    }
    for microbatch in (1, 2, 4)
    for accumulation in (1, 2, 4)
]


class BaselineValidationError(ValueError):
    """The C3.6 baseline is malformed or differs from approved identities."""


def _strictly_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without treating booleans as integers."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _strictly_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strictly_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BaselineValidationError(f"{label} must be a JSON object")
    return value


def _expect(mapping: dict[str, Any], field: str, expected: Any, label: str) -> None:
    actual = mapping.get(field)
    if not _strictly_equal(actual, expected):
        raise BaselineValidationError(
            f"{label}.{field} must match approved value {expected!r}; got {actual!r}"
        )


def validate_training_baseline(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate all approved C3.6 identity fields and return the payload.

    This deliberately pins the versioned scientific baseline. Changing an
    approved value requires a new baseline version and decision record.
    """

    root = _mapping(payload, "baseline")
    _expect(root, "schema_version", BASELINE_SCHEMA_VERSION, "baseline")
    _expect(
        root,
        "baseline_id",
        "c3.6-architecture-c-v1",
        "baseline",
    )
    _expect(
        root,
        "status",
        "baseline-approved-scalability-validation-pending",
        "baseline",
    )

    approved = _mapping(root.get("scientist_approved"), "scientist_approved")
    for field, expected in _APPROVED_VALUES.items():
        _expect(approved, field, expected, "scientist_approved")

    spec = get_architecture("C")
    frozen_inputs = _mapping(root.get("frozen_inputs"), "frozen_inputs")
    architecture = _mapping(frozen_inputs.get("architecture"), "frozen_inputs.architecture")
    expected_architecture = spec.to_dict()
    if not _strictly_equal(architecture, expected_architecture):
        raise BaselineValidationError(
            "frozen_inputs.architecture must match the frozen Architecture-C dimensions"
        )

    tokenizer = _mapping(
        frozen_inputs.get("tokenizer_identity"),
        "frozen_inputs.tokenizer_identity",
    )
    if not _strictly_equal(tokenizer, TOKENIZER_IDENTITY):
        raise BaselineValidationError(
            "frozen_inputs.tokenizer_identity differs from the frozen C2 artifact"
        )
    if tokenizer["vocab_size"] != spec.vocab_size:
        raise BaselineValidationError(
            "tokenizer vocabulary size does not match Architecture C"
        )

    expected_count = spec.parameter_count(
        tie_weights=approved["embedding_output_weight_tying"]
    )
    if approved["parameter_count"] != expected_count:
        raise BaselineValidationError(
            "scientist_approved.parameter_count does not match Architecture C"
        )

    provisional = _mapping(
        root.get("provisional_hyperparameters"), "provisional_hyperparameters"
    )
    _expect(provisional, "status", "none_selected", "provisional_hyperparameters")
    _expect(provisional, "values", {}, "provisional_hyperparameters")

    proposal = _mapping(
        root.get("scalability_validation_proposal"),
        "scalability_validation_proposal",
    )
    _expect(
        proposal,
        "status",
        "proposed-for-subsequent-scientist-authorization",
        "scalability_validation_proposal",
    )
    _expect(
        proposal,
        "context_length_tokens",
        _APPROVED_VALUES["context_length_tokens"],
        "scalability_validation_proposal",
    )
    _expect(
        proposal,
        "microbatch_sizes",
        [1, 2, 4],
        "scalability_validation_proposal",
    )
    _expect(
        proposal,
        "gradient_accumulation_steps",
        [1, 2, 4],
        "scalability_validation_proposal",
    )
    _expect(
        proposal,
        "optimizer_updates_per_cell",
        {"warmup": 5, "measured": 20, "repeats": 1},
        "scalability_validation_proposal",
    )
    _expect(
        proposal,
        "cells",
        _EXPECTED_SCALABILITY_CELLS,
        "scalability_validation_proposal",
    )
    return payload


def load_training_baseline(
    path: str | Path = DEFAULT_BASELINE_PATH,
) -> dict[str, Any]:
    """Read and validate the default or a supplied versioned baseline file."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineValidationError(f"could not read baseline {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BaselineValidationError("baseline document must be a JSON object")
    return validate_training_baseline(payload)
