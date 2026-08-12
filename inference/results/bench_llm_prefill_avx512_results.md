# Zen 5 Full Transformer Prefill (bf16), the layer above the decoder

`inference/inference/results/bench_llm_prefill_avx512.py`, Ryzen 9 9950X3D, single thread. The
complete decoder-only LLM forward pass (prefill / prompt processing):

    x      = embed(ids)        gather rows of the VxD bf16 embedding table
    for l in L: decoder layer  LN, fused QKV, attention (QK^T/softmax/PV),
                               output projection, residual, LN, MLP, residual
    x      = layernorm(x)
    logits = x . E^T           LM head over V, tied to the embeddings

Every GEMM runs on the hand-scheduled bf16 micro-kernel (stc.gemm_asm);
layernorm/GELU/softmax/exp are the SIMD fp32 helpers from
inference/llm_c_common.py (shared with the decoder benchmark,
property-tested). Reference: numpy fp32 (OpenBLAS sgemm, 1 thread) on the
same core, best-of-5 RDTSC with CPU pinning.

## Results (D=128, H=512, V=16384, tied embeddings)

| layers | S | ours (layers + head) | ours total | tokens/s | numpy | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 256 | 0.64 + 1.95 ms | **2.60 ms** | 98,500 | 19.3 ms | 7.4x |
| 6 | 512 | 4.73 + 3.88 ms | **8.61 ms** | 59,500 | 87.8 ms | 10.2x |

Correctness: logits max abs err 0.026 on a 5.3 logit scale (0.5%) vs the
bf16-input fp32 numpy reference - bf16 weights/activations, fp32
accumulation, exact per the reference within bf16 rounding.

## What the breakdown says

- The LM head (S x D x V, tied embedding) is ~45% of the prefill at
  S=512: for small D the vocabulary projection dominates - the standard
  reason LLMs use head-pooling / larger D for the embedding.
- The decoder layer cost grows with S^2 through attention: 0.32 ms/layer at
  S=256 -> 0.79 ms/layer at S=512 (2x the tokens, 2.5x the cost; attention
  is S^2 while QKV/out-proj/MLP are linear in S, so the mix is sub-S^2).
- numpy's reference is ~10x slower; the decoder benchmark measured numpy's
  GELU/softmax elementwise ops as its dominant cost at S=2048 (inherited
  claim; the prefill script does not time numpy per-op).

## Build notes

inference/llm_c_common.py holds the shared C helpers; the prefill gen_c is
self-contained and compiles the same asm kernel + helper set as the decoder
benchmark. The same driver bugs guarded elsewhere (strided q/k/v split,
transpose pack for QK^T, residual vs LN buffer, exp polynomial order,
multi-NR packing) are all exercised by the property tests in
tests/test_properties_*.py.
