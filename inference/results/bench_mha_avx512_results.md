# Zen 5 Multi-Head GQA Attention + Decoder Layer (bf16)

`inference/bench/bench_mha_avx512.py`, Ryzen 9 9950X3D, single thread. The decoder
layer from bench_decoder_avx512.py with the single-head attention replaced
by grouped-query attention: H_q query heads of head-dim D_head = D / H_q,
H_kv key/value heads shared by H_q/H_kv query heads, per-head QK^T/softmax/
PV, head concatenation, output projection. Same total attention FLOPs as
single-head (H_q * D_head = D), so the benchmark isolates the structural
cost of many smaller GEMMs. Reference: numpy fp32 (OpenBLAS, 1 thread);
timings best-of-5 RDTSC, pinned.

## Results (S=512, D=128, MLP 512, 6 layers; best-of-5 RDTSC, pinned;
RDTSC runs at the TSC clock below the boosted core clock, so ms/layer is
slightly conservative)

| attention | H_q x H_kv x D_head | ms/layer | total | err | vs numpy |
|---|---:|---:|---:|---:|---:|
| single-head | 1 x 1 x 128 | 0.82 | 4.91 ms | 0.4% | 16.2x |
| GQA | 4 x 2 x 32 | 1.01 | 6.03 ms | 0.15% | 15.2x |

## Finding: the multi-head structural overhead

At identical GEMM FLOPs, GQA multi-head is ~23% slower per layer than
single-head (1.01 vs 0.82 ms/layer). The ratio includes the identical
MLP + projections, so the attention-only overhead is larger; and GQA also
runs 4x the softmaxes and P-quantizations (~3% more elementwise ops).
The overhead is per-head work that single-head does once:
- 4 QK^T GEMMs of K=D_head=32 (a 16-chunk K-loop) instead of one K=128
  GEMM - more tile restarts, less load amortization;
- 4 SxS softmaxes plus the -1/sqrt(D_head) scales;
- the head split (strided column blocks) and the PV-output scatter into the
  concatenated D-stride layout (a real cost when the GEMM's C-stride is the
  head width, not the model width).
GQA's kv-sharing already avoids re-packing K/V per query head (2 packs for
4 query heads).

Both attention variants are correct to 0.15-0.4% vs the bf16-input fp32
numpy reference (feed-forward; bf16 rounding over 6 layers, not chaotic
amplification). numpy is a single unpinned wall-clock shot vs our best-of-5
RDTSC, so the speedups are conservative.

## Bugs caught while building (review + validation)

- The per-head PV GEMM wrote its output with row-stride N = D_head (32),
  but the concatenated output needs row-stride D (128): head h's rows
  landed inside head h-1's columns. Fixed with a contiguous per-head temp
  and a D-stride scatter.
- b_q was declared S x D_head but the LayerNorm quantization wrote S x D
  into it (4x overflow at H_q=4), working only via .bss aliasing; split
  into a full-width b_q and a per-head b_qh. Confirmed clean under ASan.
- A leftover o.bin debug write inside the timed layer contaminated the
  timing; removed (ms/layer dropped ~4%).
- The recurring hidden-size-as-multiplier call bug (main passed H as hm),
  third occurrence; CLI now validates heads*D_head == D and kv-heads
  divides heads.
