"""Validated, serializable configuration for synthetic C3 benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from cambacica.model.config import ARCHITECTURES

HARD_GPU_MEMORY_BUDGET_MIB = 40_960
DEFAULT_SAFETY_MARGIN_MIB = 4_096
NON_ALLOCATOR_HEADROOM_MIB = 1_024
PRIMARY_CONTEXT_LENGTHS = (2_048, 4_096)
CAPACITY_CONTEXT_LENGTH = 8_192
TOKENIZER_IDENTITY = {
    "candidate": "unigram-v24576",
    "algorithm": "SentencePiece Unigram",
    "vocab_size": 24_576,
    "sentencepiece_version": "0.2.2",
    "model_sha256": "c8982a23d3c5ecc0a61d300edc1e354e38f1358aa3db5420d5bf19f7b95cbf23",
    "production_path": "/mnt/data/cambacica-base-180m/tokenizers/c2-v1/unigram-v24576",
}


@dataclass(frozen=True)
class BenchmarkConfig:
    architecture: str
    context_length: int
    device: str = "cuda:0"
    precision: str = "bf16"
    microbatch_size: int = 1
    warmup_steps: int = 20
    timed_steps: int = 100
    seed: int = 20261017
    attention_backend: str = "auto"
    gpu_memory_budget_mib: int = HARD_GPU_MEMORY_BUDGET_MIB
    safety_margin_mib: int = DEFAULT_SAFETY_MARGIN_MIB
    output_root: Path = Path("/mnt/data/cambacica-base-180m/c3-benchmarks")
    scratch_root: Path = Path("/tmp/cambacica-base-180m/c3-benchmark-scratch")
    lock_path: Path = Path("/tmp/cambacica-base-180m/c3-benchmark.lock")
    tie_weights: bool = True
    activation_checkpointing: str = "none"
    enable_8192_capacity_test: bool = False

    def validate(self) -> "BenchmarkConfig":
        if self.architecture not in ARCHITECTURES:
            raise ValueError("architecture must be A, B, or C")
        if self.device != "cuda:0":
            raise ValueError("C3 production benchmarks are restricted to cuda:0")
        if self.precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError("precision must be bf16, fp16, or fp32")
        if self.microbatch_size <= 0:
            raise ValueError("microbatch-size must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warm-up steps cannot be negative")
        if self.timed_steps <= 0:
            raise ValueError("timed steps must be positive")
        if not 0 <= self.seed <= (2**63 - 1):
            raise ValueError("seed must be between 0 and 2^63-1")
        if self.attention_backend not in {"auto", "flash", "efficient", "math"}:
            raise ValueError("attention backend must be auto, flash, efficient, or math")
        if not 0 < self.gpu_memory_budget_mib <= HARD_GPU_MEMORY_BUDGET_MIB:
            raise ValueError(
                "GPU memory budget must be positive and no greater than 40960 MiB"
            )
        if self.safety_margin_mib < DEFAULT_SAFETY_MARGIN_MIB:
            raise ValueError("safety margin cannot be smaller than 4096 MiB")
        if self.gpu_memory_budget_mib - self.safety_margin_mib <= NON_ALLOCATOR_HEADROOM_MIB:
            raise ValueError("budget must leave room for the safety and allocator margins")
        if self.context_length not in PRIMARY_CONTEXT_LENGTHS:
            if self.context_length != CAPACITY_CONTEXT_LENGTH:
                raise ValueError("context length must be 2048, 4096, or 8192")
            if not self.enable_8192_capacity_test:
                raise ValueError(
                    "8192-token capacity tests require --enable-8192-capacity-test "
                    "after scientist review of 4096-token results"
                )
        if self.activation_checkpointing != "none":
            raise ValueError("activation checkpointing is not implemented; use none")
        for field_name in ("output_root", "scratch_root", "lock_path"):
            value = Path(getattr(self, field_name))
            if not value.is_absolute():
                raise ValueError(f"{field_name.replace('_', '-')} must be an absolute path")
        if self.scratch_root == self.output_root:
            raise ValueError("scratch and finalized output roots must differ")
        return self

    @property
    def peak_target_mib(self) -> int:
        return self.gpu_memory_budget_mib - self.safety_margin_mib

    @property
    def allocator_limit_mib(self) -> int:
        return self.peak_target_mib - NON_ALLOCATOR_HEADROOM_MIB

    @property
    def tokens_per_step(self) -> int:
        return self.microbatch_size * self.context_length

    def to_dict(self) -> dict[str, object]:
        self.validate()
        values = asdict(self)
        values["output_root"] = str(self.output_root)
        values["scratch_root"] = str(self.scratch_root)
        values["lock_path"] = str(self.lock_path)
        values["memory_policy"] = {
            "approved_hard_budget_mib": HARD_GPU_MEMORY_BUDGET_MIB,
            "configured_budget_mib": self.gpu_memory_budget_mib,
            "safety_margin_mib": self.safety_margin_mib,
            "peak_process_target_mib": self.peak_target_mib,
            "torch_allocator_limit_mib": self.allocator_limit_mib,
            "non_allocator_headroom_mib": NON_ALLOCATOR_HEADROOM_MIB,
            "allocator_is_hardware_isolation": False,
        }
        values["tokenizer_identity"] = dict(TOKENIZER_IDENTITY)
        return values
