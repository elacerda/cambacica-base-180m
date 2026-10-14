# C3 architecture-planning handoff

**Status: planning only, 2026-10-14.** C2 has frozen SentencePiece Unigram
24,576; C3 has not selected an architecture or context length. This document
defines a bounded comparison plan and candidate shapes for scientific review.
It does not authorize GPU work or freeze any training recipe.

The target remains a decoder-only causal base language model with a nominal
scale of approximately 180M parameters. The selected tokenizer is
[`unigram-v24576`](C2_TOKENIZER_SELECTION_V1.md), with vocabulary size 24,576.

## Candidate shapes and full parameter accounting

The following comparable proposals use pre-norm RMSNorm, bias-free full
multi-head causal attention, SwiGLU feed-forward layers, RoPE positions, and a
final RMSNorm. RoPE adds no learned positional parameters. For transparent
counts, the feed-forward block has gate, up, and down matrices; the attention
block has Q, K, V, and output matrices. No biases or other learned parameters
are included.

For vocabulary `V`, hidden width `d`, `L` layers, and feed-forward width `f`:

```text
one token matrix = V × d
one decoder block = 4d² + 3df + 2d
tied total = Vd + L(4d² + 3df + 2d) + d
untied total = tied total + Vd
```

The `2d` term counts two RMSNorm scale vectors per block; the final `d` counts
the final RMSNorm. The untied case adds a separate `V × d` output matrix with
no bias. These totals include input embeddings and the LM head.

| Proposal | Width `d` | Layers | Heads (`d/head`) | SwiGLU `f` | Contexts to compare | Input matrix | Attention/block | SwiGLU/block | Norms/block | Total tied | Total untied |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A — narrow/deep | 768 | 17 | 12 (64) | 3,072 | 2,048; 4,096 | 18,874,368 | 2,359,296 | 7,077,888 | 1,536 | 179,333,376 | 198,207,744 |
| B — balanced | 896 | 12 | 14 (64) | 3,584 | 2,048; 4,096 | 22,020,096 | 3,211,264 | 9,633,792 | 1,792 | 176,183,168 | 198,203,264 |
| C — wide/shallow | 1,024 | 9 | 16 (64) | 4,096 | 2,048; 4,096 | 25,165,824 | 4,194,304 | 12,582,912 | 2,048 | 176,180,224 | 201,346,048 |

The tied proposals range from 176.2M to 179.3M parameters. All three untied
proposals reach roughly 198.2–201.3M because a second vocabulary matrix adds
18.9–25.2M parameters. **Embedding/output-weight tying remains an explicit
architecture decision for the scientist.** The tied totals are a comparison
case, not an approved architecture. Proposal A spends capacity on depth,
proposal C on width, and proposal B sits between them.

Contexts 2,048 and 4,096 are proposed comparison points. An 8,192-token
capacity probe may be added only after a short memory check at 4,096; it is not
a proposed default. RoPE keeps parameter counts constant across these context
lengths, while activation memory and attention work increase. C4 must measure
the selected tokenizer's structural-character tokens and one-EOS-per-document
cost in actual packed sequences before a context is chosen.

## H100 memory and compute planning

The project scope names approximately 40 GB of a single H100 on Orion. The
current inspected host is `cenpesor`; `nvidia-smi` could not communicate with
an NVIDIA driver, and no Orion allocation profile is available here. The
actual GPU model, usable memory, partition/MIG mode, driver, CUDA runtime, and
scheduler allocation therefore remain **unverified**. Before C3 benchmarking,
record these from the allocated Orion node and confirm BF16 and the chosen
attention kernels there. No H100 capability is inferred from the current host.

For a tied 175–180M model, BF16 weights occupy about 0.35–0.36 GB. A rough
persistent-state estimate with BF16 parameters and gradients, two FP32 Adam
moments, and an optional FP32 master copy is 2.1–2.9 GB in decimal units. The
exact footprint depends on the framework and optimizer. An untied proposal is
about 198–202M parameters and raises that state estimate. This is only the
parameter, gradient, and optimizer-state ledger; activations, attention
workspaces, temporary logits, allocator fragmentation, and CUDA context also
consume GPU memory.

At batch size one and context 4,096, one BF16 hidden-state boundary per layer
would hold approximately 102 MiB for proposal A, 91 MiB for B, and 88 MiB for
C. That is a checkpoint-boundary illustration, not a total activation
estimate: backpropagation also needs internal attention and SwiGLU values.
Doubling context to 8,192 doubles linear activations, while a naive attention
score tensor grows with `batch × heads × context²`. Flash-style attention
kernels can avoid materializing that full score tensor; the actual kernel and
activation-checkpointing policy must be measured. A preliminary planning
guardrail is to keep steady peak use below 32 GiB when 40 GiB is actually
available, reserving the remainder for workspaces and runtime variation. This
guardrail is a proposal, not a frozen limit.

Dense-transformer training compute is approximately `6 × parameter_count ×
training_tokens`, plus context-dependent attention overhead. At 180M parameters
that is about 1.08 GFLOP per training token under the usual rough estimate. It
does not set a training-token budget. Longer contexts increase attention cost;
throughput comparisons must report tokens/second and peak allocated and
reserved memory at the same context and effective batch.

BF16 is a candidate precision to measure if Orion supports it. FP16 and FP32
have different numerical range and memory/speed behavior, so the benchmark
must compare numerical stability and loss before a precision is selected.
Activation checkpointing saves memory at the cost of recomputation. Optimizer,
microbatch, gradient accumulation, effective batch, and learning-rate schedule
remain open; compare architecture shapes under one scientist-approved common
benchmark recipe.

## Bounded benchmark proposal

After C3 receives implementation authorization, use at most these three
shapes and the two proposed contexts (six primary shape/context combinations):

1. Verify the actual Orion node and software stack, then calculate exact
   parameter totals from the implementation and check the available memory.
2. Run a short forward/backward/optimizer memory-and-throughput measurement for
   each shape at contexts 2,048 and 4,096, with the same approved precision,
   kernels, microbatch, and effective batch. Use 20 warm-up steps followed by
   100 timed steps per combination, and report tokens/second, step time, peak
   allocated/reserved VRAM, and numerical failures. Add the 8,192 capacity
   probe only if 4,096 leaves adequate measured headroom.
3. If the scientist wants a quality comparison, run a bounded equal-token
   learning pilot on the same approved training view, seed, tokenizer, context,
   optimizer, and validation references. Set its pilot token count `T_benchmark`
   in a separate approval; do not use it as the final pretraining-token budget.
   Compare validation loss versus tokens seen, including available source and
   domain breakdowns.
4. Stop at the approved pilot boundary. Select no shape unless the scientist
   reviews parameter count, held-out loss, throughput, peak memory, stability,
   and the context trade-off together.

The benchmark is bounded to three architecture shapes; the learning-pilot
token count, if authorized, is deliberately left open. C2 tokenizer throughput
was measured on a host CPU and is not a proxy for language-model training or
generation speed.

## Decisions still open

- Tied or untied embeddings/output weights and the final width/depth/head/FFN
  shape.
- Final context length and document-packing accounting for byte-fallback
  structural characters.
- Confirmation of Orion's actual GPU allocation and available memory.
- Precision, attention implementation, activation checkpointing, optimizer,
  microbatch/effective batch, and learning-rate schedule.
- Whether to authorize the equal-token C3 learning pilot and its separate
  `T_benchmark`.
- Source usage/rights resolution and final corpus mixture A/B/C. Mixture B is
  still only the provisional tokenizer-sampling reference.

No C3 training infrastructure, GPU benchmark, production job, optimizer, final
token budget, or corpus mixture is implemented or frozen by this planning
handoff.
