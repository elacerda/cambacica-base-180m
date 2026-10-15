from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is an optional C3 dependency")
F = torch.nn.functional

from cambacica.model.config import ARCHITECTURES, ArchitectureSpec  # noqa: E402
from cambacica.model.transformer import (  # noqa: E402
    CambacicaCausalLM,
    RotaryEmbedding,
    count_instantiated_parameters,
    next_token_loss,
)


def tiny_spec() -> ArchitectureSpec:
    return ArchitectureSpec("T", "tiny-test", 16, 2, 4, 32, vocab_size=31, head_dim=4)


@pytest.mark.parametrize("architecture_id", ("A", "B", "C"))
@pytest.mark.parametrize("tie_weights", (True, False))
def test_meta_candidate_counts_match_analytical(architecture_id: str, tie_weights: bool) -> None:
    spec = ARCHITECTURES[architecture_id]
    model = CambacicaCausalLM(
        spec,
        tie_weights=tie_weights,
        seed=4,
        device="meta",
    )
    assert count_instantiated_parameters(model) == spec.parameter_count(
        tie_weights=tie_weights
    )


def test_tied_and_untied_models_share_or_separate_output_weight() -> None:
    tied = CambacicaCausalLM(tiny_spec(), tie_weights=True, seed=2)
    untied = CambacicaCausalLM(tiny_spec(), tie_weights=False, seed=2)
    assert tied.lm_head.weight is tied.token_embedding.weight
    assert untied.lm_head.weight is not untied.token_embedding.weight
    assert tied.parameter_count == tiny_spec().parameters_tied
    assert untied.parameter_count == tiny_spec().parameters_untied


def test_model_refuses_unseeded_direct_cuda_initialization() -> None:
    with pytest.raises(ValueError, match="move to CUDA only after GPU safety preflight"):
        CambacicaCausalLM(tiny_spec(), seed=3, device="cuda:0")


def test_structure_uses_full_attention_rope_rmsnorm_and_bias_free_swiglu() -> None:
    model = CambacicaCausalLM(tiny_spec(), tie_weights=True, seed=2)
    block = model.layers[0]
    assert len(model.layers) == 2
    assert block.attention.num_heads == 4
    assert block.attention.head_dim == 4
    assert block.attention_norm.weight.shape == (16,)
    assert block.ffn_norm.weight.shape == (16,)
    assert block.feed_forward.gate_proj.weight.shape == (32, 16)
    assert block.feed_forward.up_proj.weight.shape == (32, 16)
    assert block.feed_forward.down_proj.weight.shape == (16, 32)
    assert block.attention.q_proj.bias is None
    assert block.attention.k_proj.bias is None
    assert block.attention.v_proj.bias is None
    assert block.attention.o_proj.bias is None
    assert block.feed_forward.gate_proj.bias is None
    assert block.feed_forward.up_proj.bias is None
    assert block.feed_forward.down_proj.bias is None
    assert model.token_embedding.padding_idx is None
    assert not any("position" in name for name, _ in model.named_parameters())
    assert not any("rope" in name for name, _ in model.named_parameters())


def test_forward_loss_shift_finite_gradients_and_generation_logits() -> None:
    model = CambacicaCausalLM(tiny_spec(), tie_weights=True, seed=9)
    token_ids = torch.tensor([[1, 2, 3, 4, 5], [8, 7, 6, 5, 4]], dtype=torch.long)
    logits, loss = next_token_loss(model, token_ids)
    assert logits.shape == (2, 4, 31)
    manual_loss = F.cross_entropy(logits.float().reshape(-1, 31), token_ids[:, 1:].reshape(-1))
    assert torch.allclose(loss, manual_loss)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    generated_logits = model(token_ids[:, :-1])
    assert generated_logits.shape == (2, 4, 31)
    assert torch.isfinite(generated_logits).all()


def test_future_tokens_do_not_change_earlier_logits() -> None:
    model = CambacicaCausalLM(tiny_spec(), tie_weights=True, seed=12).eval()
    left = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    changed_future = torch.tensor([[1, 2, 3, 9, 10]], dtype=torch.long)
    with torch.inference_mode():
        left_logits = model(left)
        changed_logits = model(changed_future)
    assert torch.allclose(left_logits[:, :3], changed_logits[:, :3], atol=1e-6, rtol=1e-6)


def test_rope_identity_at_zero_and_preserves_vector_norms() -> None:
    rope = RotaryEmbedding(4)
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 3, 4)
    rotated_query, rotated_key = rope.apply(query, key)
    assert torch.allclose(rotated_query[:, :, 0], query[:, :, 0], atol=1e-6)
    assert torch.allclose(rotated_key[:, :, 0], key[:, :, 0], atol=1e-6)
    assert torch.allclose(
        rotated_query.norm(dim=-1), query.norm(dim=-1), atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(rotated_key.norm(dim=-1), key.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_initialization_is_seeded() -> None:
    caller_state = torch.random.get_rng_state().clone()
    first = CambacicaCausalLM(tiny_spec(), tie_weights=False, seed=2026)
    assert torch.equal(torch.random.get_rng_state(), caller_state)
    second = CambacicaCausalLM(tiny_spec(), tie_weights=False, seed=2026)
    third = CambacicaCausalLM(tiny_spec(), tie_weights=False, seed=2027)
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )
    assert not torch.equal(first.token_embedding.weight, third.token_embedding.weight)
