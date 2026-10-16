# C3 architecture implementation

**Status: implementation prepared; Architecture C selected, 2026-10-16.**
This implementation performed no architecture selection. The scientist
approved **Architecture C (wide/shallow)** based on the C3.4 benchmark
matrix; the decision record is in
[`C3_ARCHITECTURE_SELECTION_V1.md`](C3_ARCHITECTURE_SELECTION_V1.md).
Proposals A and B are preserved as experimental comparison configurations.

The PyTorch implementation is in `src/cambacica/model/`. It is a decoder-only
causal Transformer with pre-norm RMSNorm, full multi-head causal attention,
RoPE, SwiGLU, bias-free projections, and a final RMSNorm. It has no learned
position embeddings and no required PAD token. The vocabulary is fixed at
24,576, and every attention head is 64 dimensions.

## Candidate matrix and parameter counts

The analytical calculation follows the reviewed C3 planning contract:

```text
embedding = V × d
attention per block = 4d²
SwiGLU per block = 3df
RMSNorm per block = 2d
final RMSNorm = d
tied = Vd + L(4d² + 3df + 2d) + d
untied = tied + Vd
```

The implementation counts shared parameters once through PyTorch's
`Module.parameters()` behavior. It checks each instantiated count against this
formula. Full candidate instances can be constructed on the meta device for
non-allocating parameter validation.

| Candidate | Width | Layers | Heads | Head dimension | SwiGLU width | Tied parameters | Untied parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A — narrow/deep | 768 | 17 | 12 | 64 | 3,072 | 179,333,376 | 198,207,744 |
| B — balanced | 896 | 12 | 14 | 64 | 3,584 | 176,183,168 | 198,203,264 |
| C — wide/shallow | 1,024 | 9 | 16 | 64 | 4,096 | 176,180,224 | 201,346,048 |

Both tying modes are implemented. Tied mode assigns the LM head weight to the
input embedding `Parameter`, so it is counted once. The initial performance
matrix uses tied weights as the proposed primary hypothesis; this does not
select the final tying policy.

## Model behavior

- Each block applies RMSNorm before attention and before the feed-forward
  layer, then adds both residuals.
- Q, K, V, and output projections use full multi-head attention. The model
  uses PyTorch scaled dot-product attention with `is_causal=True`, zero
  attention dropout, and RoPE applied to Q and K.
- The SDPA backend can be `auto`, `flash`, `efficient`, or `math`. Specific
  backend requests use PyTorch's supported backend context and fail if the
  kernel is unavailable. The benchmark also profiles a forward pass. It
  reports `unconfirmed` when it cannot observe the selected kernel and never
  infers Flash Attention from the GPU model or PyTorch version.
- SwiGLU is `down(silu(gate(x)) * up(x))`; all three matrices are bias-free.
- Loss is next-token cross entropy. `next_token_loss` feeds all tokens except
  the last and aligns labels from token 1 onward. The model also exposes
  generation logits without labels.
- RMSNorm reductions and cross-entropy logits use float32 for stability.
  Benchmark autocast selects BF16, FP16, or FP32; the model parameters and
  AdamW state remain float32.
- Initialization uses a seeded normal distribution with standard deviation
  0.02. The seed is isolated from the caller's CPU RNG state. Models initialize
  on CPU (or meta for count-only validation); the benchmark moves the CPU model
  to CUDA only after device and memory preflight.

Small synthetic CPU tests cover causal masking, RoPE, aligned loss, finite
gradients, output shape, deterministic initialization, and tying. Meta-device
tests cover instantiated parameter counts for all six tied/untied cases. These
PyTorch-dependent tests require the optional `c3` dependency. If that package
is unavailable, pytest skips those tests; the pure-Python candidate formula
tests still run.

## Explicitly open decisions

No context length, weight-tying policy, optimizer or learning rate schedule,
effective batch, training-token budget, final corpus mixture A/B/C, or
learning-quality pilot authorization is selected here. The architecture was
selected separately in
[`C3_ARCHITECTURE_SELECTION_V1.md`](C3_ARCHITECTURE_SELECTION_V1.md). The
`unigram-v24576` tokenizer remains frozen at the C2 SHA-256 recorded in its
decision document. C3 does not read or modify tokenizer or corpus artifacts.
