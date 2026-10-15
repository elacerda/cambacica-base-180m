"""PyTorch decoder-only causal Transformer for the C3 candidate matrix."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cambacica.model.config import ArchitectureSpec

ATTENTION_BACKENDS = ("auto", "flash", "efficient", "math")


def attention_backend_context(backend: str):
    if backend not in ATTENTION_BACKENDS:
        raise ValueError(f"Unsupported attention backend: {backend}")
    if backend == "auto":
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError as exc:  # pragma: no cover - depends on installed torch
        raise RuntimeError(
            "Selecting a specific SDPA backend requires PyTorch 2.3 or newer"
        ) from exc
    selected = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }[backend]
    return sdpa_kernel(backends=[selected])


class RMSNorm(nn.Module):
    """RMSNorm with the reduction performed in float32."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, *, device=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def forward(self, value: Tensor) -> Tensor:
        scale = torch.rsqrt(value.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (value.float() * scale).to(value.dtype) * self.weight.to(value.dtype)


class RotaryEmbedding(nn.Module):
    """RoPE with the split-half rotation convention used by LLaMA-style models."""

    def __init__(self, head_dim: int, theta: float = 10_000.0, *, device=None) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                / head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(self, sequence_length: int, *, device, dtype) -> tuple[Tensor, Tensor]:
        positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
        frequencies = torch.outer(positions, self.inv_freq.to(device=device))
        angles = torch.cat((frequencies, frequencies), dim=-1)
        return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)

    @staticmethod
    def _rotate_half(value: Tensor) -> Tensor:
        first, second = value.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def apply_rotary(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        cos, sin = self.cos_sin(
            query.shape[-2], device=query.device, dtype=torch.float32
        )
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        query_fp32 = query.float()
        key_fp32 = key.float()
        query = (query_fp32 * cos + self._rotate_half(query_fp32) * sin).to(query.dtype)
        key = (key_fp32 * cos + self._rotate_half(key_fp32) * sin).to(key.dtype)
        return query, key


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        spec: ArchitectureSpec,
        attention_backend: str,
        *,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = spec.num_attention_heads
        self.head_dim = spec.head_dim
        self.q_proj = nn.Linear(
            spec.hidden_size, spec.hidden_size, bias=False, device=device
        )
        self.k_proj = nn.Linear(
            spec.hidden_size, spec.hidden_size, bias=False, device=device
        )
        self.v_proj = nn.Linear(
            spec.hidden_size, spec.hidden_size, bias=False, device=device
        )
        self.o_proj = nn.Linear(
            spec.hidden_size, spec.hidden_size, bias=False, device=device
        )
        self.rope = RotaryEmbedding(spec.head_dim, device=device)
        self.attention_backend = attention_backend

    def forward(self, value: Tensor) -> Tensor:
        batch_size, sequence_length, hidden_size = value.shape
        shape = (batch_size, sequence_length, self.num_heads, self.head_dim)
        query = self.q_proj(value).view(shape).transpose(1, 2)
        key = self.k_proj(value).view(shape).transpose(1, 2)
        projected_value = self.v_proj(value).view(shape).transpose(1, 2)
        query, key = self.rope.apply_rotary(query, key)
        with attention_backend_context(self.attention_backend):
            attended = F.scaled_dot_product_attention(
                query,
                key,
                projected_value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, hidden_size)
        )
        return self.o_proj(attended)


class SwiGLU(nn.Module):
    def __init__(self, spec: ArchitectureSpec, *, device=None) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            spec.hidden_size, spec.intermediate_size, bias=False, device=device
        )
        self.up_proj = nn.Linear(
            spec.hidden_size, spec.intermediate_size, bias=False, device=device
        )
        self.down_proj = nn.Linear(
            spec.intermediate_size, spec.hidden_size, bias=False, device=device
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


class DecoderBlock(nn.Module):
    def __init__(
        self,
        spec: ArchitectureSpec,
        attention_backend: str,
        *,
        device=None,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(spec.hidden_size, device=device)
        self.attention = CausalSelfAttention(
            spec, attention_backend=attention_backend, device=device
        )
        self.ffn_norm = RMSNorm(spec.hidden_size, device=device)
        self.feed_forward = SwiGLU(spec, device=device)

    def forward(self, value: Tensor) -> Tensor:
        value = value + self.attention(self.attention_norm(value))
        return value + self.feed_forward(self.ffn_norm(value))


class CambacicaCausalLM(nn.Module):
    """Pre-norm RMSNorm, RoPE, full-MHA, SwiGLU causal language model."""

    def __init__(
        self,
        spec: ArchitectureSpec,
        *,
        tie_weights: bool = True,
        seed: int = 20261017,
        attention_backend: str = "auto",
        activation_dtype: torch.dtype | None = None,
        device=None,
    ) -> None:
        super().__init__()
        if attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(f"Unsupported attention backend: {attention_backend}")
        initial_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        if initial_device.type not in {"cpu", "meta"}:
            raise ValueError(
                "initialize on CPU or meta for seeded weights; move to CUDA only "
                "after GPU safety preflight"
            )
        self.spec = spec
        self.tie_weights = tie_weights
        self.attention_backend = attention_backend
        self.activation_dtype = activation_dtype
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.token_embedding = nn.Embedding(
                spec.vocab_size, spec.hidden_size, device=device
            )
            self.layers = nn.ModuleList(
                DecoderBlock(spec, attention_backend, device=device)
                for _ in range(spec.num_layers)
            )
            self.final_norm = RMSNorm(spec.hidden_size, device=device)
            self.lm_head = nn.Linear(
                spec.hidden_size, spec.vocab_size, bias=False, device=device
            )
            self.apply(self._initialize_weights)
            if tie_weights:
                self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def attention_head_count(self) -> int:
        return self.spec.num_attention_heads

    @property
    def head_dimension(self) -> int:
        return self.spec.head_dim

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, input_ids: Tensor, labels: Tensor | None = None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids sequence length must be positive")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must use torch.long token IDs")
        value = self.token_embedding(input_ids)
        if self.activation_dtype is not None:
            value = value.to(self.activation_dtype)
        for layer in self.layers:
            value = layer(value)
        logits = self.lm_head(self.final_norm(value))
        if labels is None:
            return logits
        if labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")
        if labels.dtype != torch.long:
            raise ValueError("labels must use torch.long token IDs")
        loss = F.cross_entropy(
            logits.float().reshape(-1, self.spec.vocab_size), labels.reshape(-1)
        )
        return logits, loss


def count_instantiated_parameters(model: nn.Module) -> int:
    """Count shared parameters once, matching ``Module.parameters()`` semantics."""

    return sum(parameter.numel() for parameter in model.parameters())


def next_token_loss(
    model: CambacicaCausalLM, token_ids: Tensor
) -> tuple[Tensor, Tensor]:
    """Run the model on all but the final token and align next-token labels."""

    if token_ids.ndim != 2 or token_ids.shape[1] < 2:
        raise ValueError("token_ids must have shape [batch, sequence>=2]")
    return model(token_ids[:, :-1], labels=token_ids[:, 1:])
