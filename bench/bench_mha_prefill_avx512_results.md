# Zen 5 Multi-Head GQA Prefill (bf16), the complete model at the real architecture

`scripts/bench_mha_prefill_avx512.py`, Ryzen 9 9950X3D, single thread. The
full decoder-only LLM forward (prefill) with grouped-query multi-head
attention - the single-head prefill from bench_llm_prefill_avx512.py with
the attention replaced by the reviewed multi-head layer from
bench_mha_avx512.py:

    x = embed(ids);  L x MHA decoder layer;  LN;  logits = x . E^T

H_q=4 query heads of D_head=32, H_kv=2 shared key/value heads. Reference:
numpy fp32 (OpenBLAS, 1 thread); best-of-5 RDTSC, pinned.

## Results (S=512, D=128, MLP 512, V=16384, 6 layers)

| prefill | ours (6 layers + head) | tokens/s | err | vs numpy |
|---|---:|---:|---:|---:|
| single-head (1x1x128) | 8.63 ms | 59.3k | 0.5% | 10.2x |
| **GQA (4x2x32)** | **9.89 ms** (1.65 ms/layer-equiv) | 51.8k | 0.1% | 9.9x |

Correctness: final hidden state matches the bf16-input fp32 numpy reference
to 0.1% (feed-forward; bf16 rounding over 6 layers).

## Finding: the multi-head overhead at the model level

The full GQA prefill is ~15% slower than the single-head prefill at the
same FLOPs (9.89 vs 8.63 ms). The attention portion carries the ~23%
multi-head overhead measured in bench_mha_avx512_results.md (per-head
short-K GEMMs, per-head softmax, head split/scatter); the identical MLP +
LM head (3.9 ms of the total) dilutes the ratio. GQA's kv-sharing saves
the K/V packs versus full MHA.

## Bugs caught while building

- The clean-pass LM head wrote its S x V output into f_g (the S x D MLP
  scratch) instead of the S x V logits buffer - a 4M-float write into a
  128KB buffer that only surfaced at V=16384 (and only at -O3; the -O0
  build survived by .bss layout luck). Fixed; ASan-clean after.
- The timed region initially excluded the LM head (layers-only 5.96 ms
  looked artificially fast); moved the head inside the timing.
- The recurring hidden-size-as-multiplier call bug (H passed as hm),
  fourth occurrence, now caught in review every round.
