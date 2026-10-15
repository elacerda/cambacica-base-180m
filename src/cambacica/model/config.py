"""Frozen C3 architecture proposals and analytical parameter accounting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchitectureSpec:
    """Dimensions for one decoder-only Transformer candidate."""

    architecture_id: str
    name: str
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int = 24_576
    head_dim: int = 64

    def __post_init__(self) -> None:
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads × head_dim")
        if min(
            self.hidden_size,
            self.num_layers,
            self.num_attention_heads,
            self.intermediate_size,
            self.vocab_size,
            self.head_dim,
        ) <= 0:
            raise ValueError("architecture dimensions must be positive")

    @property
    def embedding_parameters(self) -> int:
        return self.vocab_size * self.hidden_size

    @property
    def parameters_tied(self) -> int:
        d = self.hidden_size
        per_block = 4 * d * d + 3 * d * self.intermediate_size + 2 * d
        return self.embedding_parameters + self.num_layers * per_block + d

    @property
    def parameters_untied(self) -> int:
        return self.parameters_tied + self.embedding_parameters

    def parameter_count(self, *, tie_weights: bool) -> int:
        return self.parameters_tied if tie_weights else self.parameters_untied

    def to_dict(self) -> dict[str, int | str]:
        return {
            "architecture_id": self.architecture_id,
            "name": self.name,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "parameters_tied": self.parameters_tied,
            "parameters_untied": self.parameters_untied,
        }


ARCHITECTURES: dict[str, ArchitectureSpec] = {
    "A": ArchitectureSpec("A", "narrow-deep", 768, 17, 12, 3_072),
    "B": ArchitectureSpec("B", "balanced", 896, 12, 14, 3_584),
    "C": ArchitectureSpec("C", "wide-shallow", 1_024, 9, 16, 4_096),
}

EXPECTED_PARAMETER_COUNTS = {
    "A": {"tied": 179_333_376, "untied": 198_207_744},
    "B": {"tied": 176_183_168, "untied": 198_203_264},
    "C": {"tied": 176_180_224, "untied": 201_346_048},
}


def get_architecture(architecture_id: str) -> ArchitectureSpec:
    try:
        return ARCHITECTURES[architecture_id.upper()]
    except KeyError as exc:
        raise ValueError(f"Unknown C3 architecture: {architecture_id!r}") from exc


def validate_candidate_parameter_counts() -> None:
    """Fail if candidate dimensions drift from the reviewed C3 accounting."""

    for architecture_id, spec in ARCHITECTURES.items():
        expected = EXPECTED_PARAMETER_COUNTS[architecture_id]
        actual = {
            "tied": spec.parameters_tied,
            "untied": spec.parameters_untied,
        }
        if actual != expected:
            raise AssertionError(
                f"Architecture {architecture_id} parameter counts changed: "
                f"expected {expected}, got {actual}"
            )


validate_candidate_parameter_counts()
