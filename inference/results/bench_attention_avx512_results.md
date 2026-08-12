# Zen 5 Attention Head (bf16), the piece above GEMM

`inference/bench/bench_attention_avx512.py`, on the Ryzen 9 9950X3D (Zen 5),
single thread. Single-head attention forward:

    scores = Q . K^T / sqrt(D)     GEMM1 (SxS), hand-scheduled bf16 kernel
    P      = softmax(scores)       SIMD fp32 poly-exp
    out    = P . V                 GEMM2 (SxD), same kernel

Reference: numpy matmul = OpenBLAS 0.3.34 (scipy-openblas, sgemm,
AVX-512 FMA), 1 thread, same CPU, wall-clock. Our timings are RDTSC
(robust to box load); numpy's vary with load on this shared box.

## Why this piece

Attention is the transformer component that is *larger than a GEMM* and
where the interesting bottlenecks show up:

- Two GEMMs with very different shapes: QK^T has K=D=128 (short K-loop,
  tile-restart dominated) while PV has K=S (long).
- An O(S^2) softmax that, at S=2048, materializes 17 MB of scores and
  becomes the dominant cost of a naive implementation (13 ms scalar vs
  5.6 ms for both GEMMs) - the long-sequence regime that flash-attention
  fusion exists to avoid.
- The PV GEMM needs P in bf16 with this kernel; real implementations keep
  P in fp32/tf32. That quantization is the main accuracy loss (below).

## Results (D=128, bf16 in, fp32 accumulate)

| S  | ours QK^T | ours PV | ours softmax | head total | numpy head | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | ~270 GF/s | ~406 GF/s | 0.2 ms | **1.8 ms** | ~20 ms | ~11x |
| 2048 | 380 GF/s | 424 GF/s | 0.7 ms | **6.0 ms** | 15.9 ms | 2.6-6x (box-load dependent) |

(OpenBLAS peak on this CPU is ~64 FLOP/cyc; our bf16 kernels run at
~73-88% of the 64 MACs/cyc bf16 peak - QK^T is short-K (D=128) and pays
tile-restart overhead, PV runs at ~the peak.)

## Correctness

Max abs error vs a bf16-input fp32 numpy reference: **~0.00028** on outputs
of magnitude ~0.01 (score scale 1/sqrt(128) = 0.088). The numpy reference
bf16-quantizes Q/K/V/P exactly like the C path, so the residual error is
the bf16-P quantization plus fp32 accumulation rounding. Earlier numbers
(0.0011) conflated the input quantization (the old numpy reference used
fp32 inputs) with a QK^T packing bug (Kt stored SxD was packed as if DxS;
the softmax normalization masked the scrambled scores). Both are fixed;
the QK^T transpose pack is now the same pack_b_bf16_32_T used by the
decoder layer.

## What the numbers say

1. The hand-scheduled bf16 kernels beat OpenBLAS's fp32 sgemm on the same
   core by 5-20x depending on shape and box load; the bf16 path has 2x the
   peak of fp32 anyway, and the short-K QK^T shape particularly hurts the
   BLAS-style loop order.
2. At S >= 1024 naive attention is *softmax-bound*, not GEMM-bound; a
   vectorized softmax (21x over scalar) is the difference between 18.6 ms
   and 6.3 ms at S=2048.
3. Next step for long sequences: fuse the softmax into the PV kernel
   (flash-attention style) to avoid materializing the SxS scores entirely.
