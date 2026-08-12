# Zen 5 Multi-Head GQA Attention + Decoder Layer (bf16)

`scripts/bench_mha_avx512.py`, Ryzen 9 9950X3D, single thread. The decoder
layer from bench_decoder_avx512.py with the single-head attention replaced
by grouped-query attention: H_q query heads of head-dim D_head = D / H_q,
H_kv key/value heads shared by H_q/H_kv query heads, per-head QK^T/softmax/
PV, head concatenation, output projection. Same total attention FLOPs as
single-head (H_q * D_head = D), so the benchmark isolates the structural
cost of many smaller GEMMs. Reference: numpy fp32 (OpenBLAS, 1 thread);
timings best-of-5 RDTSC, pinned.

## Results (S=512, D=128, MLP 512, 6 layers)

| attention | H_q x H_kv x D_head | ms/layer | total | err | vs numpy |
|---|---:|---:|---:|---:|---:|
| single-head | 1 x 1 x 128 | 0.82 | 4.91 ms | 0.4% | 16.2x |
| GQA | 4 x 2 x 32 | 1.04 | 6.25 ms | 0.6% | 14.7x |

## Finding: the multi-head structural overhead

At identical FLOPs, GQA multi-head is ~27% slower than single-head on this
kernel. The overhead is per-head work that single-head does once:
- 4 QK^T GEMMs of K=D_head=32 (a 16-chunk K-loop) instead of one K=128
  GEMM - more tile restarts, less load amortization;
- 4 SxS softmaxes plus the -1/sqrt(D_head) scales;
- the head split (strided column blocks) and the PV-output scatter into the
  concatenated D-stride layout (a real cost when the GEMM's C-stride is the
  head width, not the model width).
GQA's kv-sharing already avoids re-packing K/V per query head (2 packs for
4 query heads).

Both attention variants are correct to 0.4-0.6% vs the bf16-input fp32
numpy reference (feed-forward, so the small errors are plain bf16 rounding
over 6 layers, not chaotic amplification).

## Bugs caught while building

- The per-head PV GEMM wrote its output with row-stride N = D_head (32),
  but the concatenated output needs row-stride D (128): head h's rows
  landed inside head h-1's columns. Fixed by writing each head to a
  contiguous S x D_head temp and scattering at the D stride.
- The recurring hidden-size-as-multiplier call bug (main passed H as hm),
  third occurrence across the benchmark family - now guarded by the
  pattern of this round's reviews.
