from __future__ import annotations

import pytest

from cambacica.model.config import (
    ARCHITECTURES,
    EXPECTED_PARAMETER_COUNTS,
    ArchitectureSpec,
    get_architecture,
    validate_candidate_parameter_counts,
)


def test_exact_candidate_dimensions_and_parameter_counts() -> None:
    validate_candidate_parameter_counts()
    assert {
        key: (
            spec.hidden_size,
            spec.num_layers,
            spec.num_attention_heads,
            spec.head_dim,
            spec.intermediate_size,
        )
        for key, spec in ARCHITECTURES.items()
    } == {
        "A": (768, 17, 12, 64, 3072),
        "B": (896, 12, 14, 64, 3584),
        "C": (1024, 9, 16, 64, 4096),
    }
    assert {
        key: {"tied": spec.parameters_tied, "untied": spec.parameters_untied}
        for key, spec in ARCHITECTURES.items()
    } == EXPECTED_PARAMETER_COUNTS


@pytest.mark.parametrize("candidate", ("A", "B", "C"))
def test_get_architecture_is_case_insensitive(candidate: str) -> None:
    assert get_architecture(candidate.lower()) is ARCHITECTURES[candidate]


def test_spec_rejects_head_dimension_drift() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        ArchitectureSpec("tiny", "invalid", 12, 1, 2, 16, vocab_size=32, head_dim=8)
