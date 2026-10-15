"""C3 model definitions and architecture accounting."""

from cambacica.model.config import (
    ARCHITECTURES,
    EXPECTED_PARAMETER_COUNTS,
    ArchitectureSpec,
    get_architecture,
    validate_candidate_parameter_counts,
)

__all__ = [
    "ARCHITECTURES",
    "EXPECTED_PARAMETER_COUNTS",
    "ArchitectureSpec",
    "get_architecture",
    "validate_candidate_parameter_counts",
]
