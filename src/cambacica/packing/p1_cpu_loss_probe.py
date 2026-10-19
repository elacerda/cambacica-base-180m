"""Bounded CPU-only loss probe for artificial C4-P1.1 packed windows.

This is a scientific integration verifier, NOT a production data loader or
training loss implementation. It never uses CUDA, NVML, files, or a tokenizer.
It deliberately rejects large model forwards; full 4,096-position window
geometry is tested independently by the pure-Python P1 reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cambacica.packing.p1_reference import (
    CONTEXT_LENGTH,
    IGNORE_INDEX,
    VOCAB_SIZE,
    TrainingWindow,
)

if TYPE_CHECKING:
    from cambacica.model.transformer import CambacicaCausalLM

MAX_CPU_PROBE_POSITIONS = 32


@dataclass(frozen=True)
class CpuLossProbe:
    """Numerical comparison from one artificial P1 window on CPU."""

    scored_targets: int
    ignored_targets: int
    model_loss: float
    explicit_ce: float
    manual_masked_ce: float
    negative_log_likelihood_sum: float


def probe_p1_window_cpu(
    model: CambacicaCausalLM,
    window: TrainingWindow,
    *,
    ignored_tail_positions: int = 0,
) -> CpuLossProbe:
    """Compare actual Transformer loss with two independent masked CE paths.

    For bounded CPU execution, model only a small valid prefix plus a requested
    number of ignored *trailing* positions. Causal attention guarantees that
    excluded later padding cannot affect earlier logits. The full-size P1
    stream/window structure is validated separately by the reference tests.

    Parameters
    ----------
    model : CambacicaCausalLM
        A tiny CPU-only instance of the real C3 Transformer implementation.
    window : TrainingWindow
        A window generated from artificial C4-P1 tokens.
    ignored_tail_positions : int
        Number of ignored final positions to include in the model forward.

    Returns
    -------
    CpuLossProbe
        Scored/ignored counts and actual, explicit and manual loss values.

    Raises
    ------
    ValueError
        If the window is unsuitable for a finite CPU-only loss comparison.
    RuntimeError
        If the model is not a CPU module or the losses disagree.
    """
    import torch
    from torch.nn import functional as F

    if not isinstance(window, TrainingWindow):
        raise TypeError("window must be a P1 TrainingWindow")
    if type(ignored_tail_positions) is not int or ignored_tail_positions < 0:
        raise ValueError("ignored_tail_positions must be a nonnegative integer")
    if len(window.input_ids) != CONTEXT_LENGTH or len(window.labels) != CONTEXT_LENGTH:
        raise ValueError("window must contain exactly 4096 inputs and labels")
    count = window.valid_target_count
    if not 0 < count <= CONTEXT_LENGTH:
        raise ValueError("zero or invalid valid-target count: do not evaluate loss")
    if any(target == IGNORE_INDEX for target in window.labels[:count]):
        raise ValueError("C4-P1.1 requires every valid next-token target to be scored")
    if any(target != IGNORE_INDEX for target in window.labels[count:]):
        raise ValueError("trailing invalid labels must use ignore_index=-100")
    if any(
        type(token) is not int or not 0 <= token < VOCAB_SIZE
        for token in window.input_ids
    ):
        raise ValueError("window contains invalid input IDs")
    if any(
        type(token) is not int or not 0 <= token < VOCAB_SIZE
        for token in window.labels[:count]
    ):
        raise ValueError("window contains invalid valid-target IDs")
    available = CONTEXT_LENGTH - count
    if ignored_tail_positions > available:
        raise ValueError("requested more ignored positions than the partial tail")
    positions = count + ignored_tail_positions
    if positions > MAX_CPU_PROBE_POSITIONS:
        raise ValueError("CPU probe exceeds the 32-position safety bound")

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a PyTorch module")
    devices = {tensor.device.type for tensor in (*model.parameters(), *model.buffers())}
    if devices != {"cpu"}:
        raise RuntimeError("CPU probe accepts CPU model parameters/buffers only")
    if not hasattr(model, "spec"):
        raise TypeError("model must expose an ArchitectureSpec")
    vocab_size = model.spec.vocab_size
    if (
        max(window.input_ids[:positions]) >= vocab_size
        or max(window.labels[:count]) >= vocab_size
    ):
        raise ValueError("artificial probe tokens exceed the tiny model vocabulary")

    inputs = torch.tensor((window.input_ids[:positions],), dtype=torch.long)
    labels = torch.tensor((window.labels[:positions],), dtype=torch.long)
    mask = labels.ne(IGNORE_INDEX)
    scored_targets = int(mask.sum().item())
    if scored_targets == 0:
        raise ValueError("all target labels are ignored")
    with torch.inference_mode():
        logits, model_loss = model(inputs, labels=labels)
        float_logits = logits.float()
        explicit = F.cross_entropy(
            float_logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        )
        selected_logits = float_logits[mask]
        selected_labels = labels[mask]
        log_probabilities = F.log_softmax(selected_logits, dim=-1)
        nll = -log_probabilities.gather(1, selected_labels.unsqueeze(1)).squeeze(1)
        manual = nll.sum() / scored_targets

    values = (model_loss, explicit, manual)
    if not all(bool(torch.isfinite(value).item()) for value in values):
        raise RuntimeError("non-finite CPU loss comparison")
    if not (
        torch.allclose(model_loss, explicit, atol=1e-6, rtol=1e-6)
        and torch.allclose(model_loss, manual, atol=1e-6, rtol=1e-6)
    ):
        raise RuntimeError("model, explicit CE and masked manual CE disagree")

    result = CpuLossProbe(
        scored_targets=scored_targets,
        ignored_targets=positions - scored_targets,
        model_loss=float(model_loss.item()),
        explicit_ce=float(explicit.item()),
        manual_masked_ce=float(manual.item()),
        negative_log_likelihood_sum=float(nll.sum().item()),
    )
    if not math.isfinite(result.negative_log_likelihood_sum):
        raise RuntimeError("non-finite negative log likelihood sum")
    return result
