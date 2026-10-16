"""C3 model definitions and architecture accounting."""

from cambacica.model.config import (
    ARCHITECTURES,
    EXPECTED_PARAMETER_COUNTS,
    ArchitectureSpec,
    get_architecture,
    validate_candidate_parameter_counts,
)
from cambacica.model.baseline import (
    BaselineValidationError,
    load_training_baseline,
    validate_training_baseline,
)

__all__ = [
    "ARCHITECTURES",
    "EXPECTED_PARAMETER_COUNTS",
    "ArchitectureSpec",
    "BaselineValidationError",
    "get_architecture",
    "load_training_baseline",
    "validate_candidate_parameter_counts",
    "validate_training_baseline",
]
