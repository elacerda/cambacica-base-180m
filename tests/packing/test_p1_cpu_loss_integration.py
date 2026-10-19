"""C4-P1.2: synthetic-only CPU integration with the real C3 Transformer.

No real corpus, tokenizer, optimizer, GPU, NVML, or training.
Only deliberately tiny CPU models are instantiated.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from cambacica.model.config import ArchitectureSpec
from cambacica.packing.p1_cpu_loss_probe import (
    MAX_CPU_PROBE_POSITIONS,
    probe_p1_window_cpu,
)
from cambacica.packing.p1_reference import (
    CONTEXT_LENGTH,
    EOS_ID,
    IGNORE_INDEX,
    Document,
    PackingPolicy,
    pack_documents_by_split,
)

torch = pytest.importorskip("torch", reason="PyTorch is an optional C3 dependency")
from cambacica.model.transformer import (  # noqa: E402
    CambacicaCausalLM,
    next_token_loss,
)


def toy_document(
    name: str,
    tokens: tuple[int, ...] | list[int],
    *,
    split: str = "train",
    group_id: str | None = None,
) -> Document:
    payload = tuple(tokens)
    return Document(
        document_id=f"c4-p1.2-synthetic:{name}",
        group_id=group_id or f"c4-p1.2-synthetic-group:{name}",
        split=split,
        source="c4-p1.2-artificial",
        subset="cpu-test-only",
        content_hash=hashlib.sha256(
            repr((name, payload)).encode("utf-8")
        ).hexdigest(),
        token_ids=payload,
    )


def tiny_cpu_model() -> CambacicaCausalLM:
    spec = ArchitectureSpec(
        "T", "synthetic-p1.2-tiny", 16, 2, 4, 32,
        vocab_size=31, head_dim=4,
    )
    return CambacicaCausalLM(spec, seed=20261019, device="cpu").eval()


def short_packed_window():
    packed = pack_documents_by_split([
        toy_document("a", [10, 11]),
        toy_document("b", [12]),
        toy_document("empty", []),
    ])["train"]
    assert packed.tokens == (10, 11, EOS_ID, 12, EOS_ID, EOS_ID)
    return packed, packed.windows[0]


def test_model_matches_explicit_ce_and_manual_masked_nll() -> None:
    packed, window = short_packed_window()
    model = tiny_cpu_model()
    result = probe_p1_window_cpu(model, window, ignored_tail_positions=2)

    assert packed.accounting.unique_scored_targets == 5
    assert window.labels[:7] == (
        11, EOS_ID, 12, EOS_ID, EOS_ID, IGNORE_INDEX, IGNORE_INDEX,
    )
    assert result.scored_targets == 5
    assert result.ignored_targets == 2
    assert result.model_loss == pytest.approx(result.explicit_ce, abs=1e-6)
    assert result.model_loss == pytest.approx(result.manual_masked_ce, abs=1e-6)
    assert result.negative_log_likelihood_sum == pytest.approx(
        result.manual_masked_ce * result.scored_targets, abs=1e-6
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_p1_pre_shifted_labels_equal_c3_next_token_loss() -> None:
    packed, window = short_packed_window()
    model = tiny_cpu_model()
    probe = probe_p1_window_cpu(model, window)
    with torch.inference_mode():
        original_tokens = torch.tensor((packed.tokens,), dtype=torch.long)
        logits, expected = next_token_loss(model, original_tokens)
        direct_logits, direct = model(
            original_tokens[:, :-1], labels=original_tokens[:, 1:]
        )

    assert logits.shape == direct_logits.shape == (1, 5, 31)
    torch.testing.assert_close(logits, direct_logits, rtol=1e-6, atol=1e-6)
    assert probe.model_loss == pytest.approx(float(expected.item()), abs=1e-6)
    assert probe.model_loss == pytest.approx(float(direct.item()), abs=1e-6)


def test_ignored_trailing_positions_do_not_change_loss() -> None:
    _, window = short_packed_window()
    model = tiny_cpu_model()
    baseline = probe_p1_window_cpu(model, window)
    with_padding = probe_p1_window_cpu(model, window, ignored_tail_positions=3)

    assert baseline.scored_targets == with_padding.scored_targets == 5
    assert baseline.ignored_targets == 0
    assert with_padding.ignored_targets == 3
    assert with_padding.model_loss == pytest.approx(baseline.model_loss, abs=1e-6)
    assert with_padding.manual_masked_ce == pytest.approx(
        baseline.manual_masked_ce, abs=1e-6
    )


def test_eos_and_cross_document_transition_targets_are_scored() -> None:
    _, window = short_packed_window()
    assert window.input_ids[:5] == (10, 11, EOS_ID, 12, EOS_ID)
    assert window.labels[:5] == (11, EOS_ID, 12, EOS_ID, EOS_ID)
    assert (EOS_ID, 12) in tuple(window.scored_pairs())
    assert (11, EOS_ID) in tuple(window.scored_pairs())
    assert probe_p1_window_cpu(tiny_cpu_model(), window).scored_targets == 5


def test_causal_prefix_is_not_affected_by_later_document_tokens() -> None:
    model = tiny_cpu_model()
    first = pack_documents_by_split([
        toy_document("alpha", [10, 11]), toy_document("beta", [12]),
    ])["train"]
    changed = pack_documents_by_split([
        toy_document("alpha", [10, 11]), toy_document("beta", [15]),
    ])["train"]

    with torch.inference_mode():
        original_logits = model(
            torch.tensor((first.tokens[:-1],), dtype=torch.long)
        )
        changed_logits = model(
            torch.tensor((changed.tokens[:-1],), dtype=torch.long)
        )
    torch.testing.assert_close(
        original_logits[:, :3], changed_logits[:, :3],
        rtol=1e-6, atol=1e-6,
    )


def test_train_validation_test_have_independent_contexts() -> None:
    documents = [
        toy_document("train-one", [10], split="train"),
        toy_document("validation-one", [10], split="validation"),
        toy_document("test-one", [12], split="test"),
        toy_document("train-two", [12], split="train"),
    ]
    packed = pack_documents_by_split(documents)
    assert packed["train"].tokens == (10, EOS_ID, 12, EOS_ID)
    assert packed["validation"].tokens == (10, EOS_ID)
    assert packed["test"].tokens == (12, EOS_ID)
    assert len({packed[key].identity_digest for key in packed}) == 3
    model = tiny_cpu_model()
    for split in ("train", "validation", "test"):
        assert all(span.split == split for span in packed[split].document_spans)
        assert probe_p1_window_cpu(
            model, packed[split].windows[0], ignored_tail_positions=1
        ).scored_targets == packed[split].accounting.unique_scored_targets


def test_unequal_window_lengths_require_token_weighted_mean() -> None:
    packed = pack_documents_by_split([
        toy_document("short", [4], split="train"),
        toy_document("long", [5, 6, 7], split="validation"),
    ])
    model = tiny_cpu_model()
    results = [
        probe_p1_window_cpu(model, packed[split].windows[0])
        for split in ("train", "validation")
    ]
    assert [result.scored_targets for result in results] == [1, 3]
    combined = (
        sum(r.negative_log_likelihood_sum for r in results)
        / sum(r.scored_targets for r in results)
    )
    expected = (
        results[0].manual_masked_ce + 3 * results[1].manual_masked_ce
    ) / 4
    assert combined == pytest.approx(expected, abs=1e-6)


def test_one_empty_document_has_no_supervised_window() -> None:
    packed = pack_documents_by_split([toy_document("empty-only", [])])["train"]
    assert packed.tokens == (EOS_ID,)
    assert packed.accounting.candidate_target_count == 0
    assert packed.accounting.unique_scored_targets == 0
    assert packed.windows == ()


def test_fail_closed_on_zero_or_all_ignored_valid_labels() -> None:
    _, window = short_packed_window()
    model = tiny_cpu_model()
    with pytest.raises(ValueError, match="zero or invalid"):
        probe_p1_window_cpu(model, replace(window, valid_target_count=0))
    fake_all_ignored = replace(
        window, labels=(IGNORE_INDEX,) * CONTEXT_LENGTH,
    )
    with pytest.raises(ValueError, match="valid next-token target"):
        probe_p1_window_cpu(model, fake_all_ignored)


def test_fail_closed_on_bad_padding_and_token_ids() -> None:
    _, window = short_packed_window()
    model = tiny_cpu_model()
    bad_tail = list(window.labels)
    bad_tail[window.valid_target_count] = 8
    with pytest.raises(ValueError, match="trailing invalid"):
        probe_p1_window_cpu(model, replace(window, labels=tuple(bad_tail)))
    bad_input = list(window.input_ids)
    bad_input[0] = -1
    with pytest.raises(ValueError, match="invalid input"):
        probe_p1_window_cpu(model, replace(window, input_ids=tuple(bad_input)))
    high_token = list(window.input_ids)
    high_token[0] = 60  # Allowed by P1's v24576, not by the tiny model's vocab=31.
    with pytest.raises(ValueError, match="tiny model vocabulary"):
        probe_p1_window_cpu(model, replace(window, input_ids=tuple(high_token)))


def test_probe_rejects_unbounded_cpu_model_forward() -> None:
    packed = pack_documents_by_split([
        toy_document("long", list(range(40))),
    ])["train"]
    assert packed.windows[0].valid_target_count == 40
    assert MAX_CPU_PROBE_POSITIONS == 32
    with pytest.raises(ValueError, match="32-position safety bound"):
        probe_p1_window_cpu(tiny_cpu_model(), packed.windows[0])


def test_probe_rejects_non_cpu_model_without_cuda_initialization() -> None:
    _, window = short_packed_window()
    tiny = tiny_cpu_model().spec
    meta_model = CambacicaCausalLM(tiny, seed=20261019, device="meta")
    with pytest.raises(RuntimeError, match="CPU model"):
        probe_p1_window_cpu(meta_model, window)


def test_reference_geometry_remains_4096_4097_even_with_tiny_cpu_probe() -> None:
    packed = pack_documents_by_split([
        toy_document("long-boundary", [10] * (CONTEXT_LENGTH + 1))
    ])["train"]
    first, second = packed.windows
    assert first.valid_target_count == CONTEXT_LENGTH
    assert second.valid_target_count == 1
    assert first.labels[-1] == 10
    assert second.input_ids[0] == 10
    assert second.labels[0] == EOS_ID
    assert len(first.input_ids) == len(first.labels) == CONTEXT_LENGTH
    assert tuple(first.owned_target_indices)[-1] == CONTEXT_LENGTH
    assert tuple(second.owned_target_indices) == (CONTEXT_LENGTH + 1,)
    assert packed.accounting.unique_scored_targets == CONTEXT_LENGTH + 1


def test_bos_eos_alternative_policy_is_not_approved_p1_1() -> None:
    packed = pack_documents_by_split(
        [toy_document("current", [10])]
    )["train"]
    assert packed.bos_offset is None
    assert packed.policy == PackingPolicy(
        insert_bos_at_split_start=False,
        score_eos_targets=True,
        safe_input_padding_id=EOS_ID,
    )
    assert packed.windows[0].labels[0] == EOS_ID
