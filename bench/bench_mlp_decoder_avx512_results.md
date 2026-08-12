# Zen 5 MLP Block + Full Decoder Layer (bf16), vs OpenBLAS/numpy

`scripts/bench_decoder_avx512.py`, Ryzen 9 9950X3D, single thread. The
components above GEMM, in order:

1. **MLP block** (transformer FFN):
   `up = x·W1` (SxD x Dx4D) -> GELU -> `out = g·W2` (Sx4D x 4DxD),
   all on the hand-scheduled bf16 kernel (stc.gemm_asm) with SIMD fp32
   layernorm/GELU/softmax.
2. **Decoder layer** (pre-norm, GPT-2 style): LN1 -> fused QKV projection ->
   attention (QK^T, softmax, PV, output projection) -> residual -> LN2 ->
   MLP -> residual.

Reference: numpy fp32 on this box = OpenBLAS 0.3.34 sgemm (1 thread),
wall-clock; our timings are best-of-7 RDTSC with CPU pinning.

## Results (D=128, H=512, bf16 weights/activations, fp32 accumulate)

| S | ours MLP GEMMs | numpy MLP GEMMs | ours decoder | numpy decoder | speedup |
|---:|---:|---:|---:|---:|---:|
| 1024 | 0.47 ms (588/550 GF/s) | ~11-22 ms | **2.12 ms** | ~60-100 ms | **~31x** |
| 2048 | 0.95 ms (556/572 GF/s) | ~12 ms | **6.69 ms** (attn 5.48 + mlp 1.22) | 83.4 ms | **12.5x** |

The numpy reference numbers vary with box load (wall-clock); ours are
RDTSC. The MLP GEMMs run at ~87-92% of the 64 MACs/cyc bf16 peak (the up
projection is K=128 short-K; down is K=512).

## Correctness

Max abs error vs fp32 numpy: **1e-5** at S=1024/2048 (activation scale
0.09) - the path is: bf16 weights/activations, fp32 accumulation, fp32
layernorm/GELU/softmax, fp32 residuals. This is what bf16 inference should
look like.

## Bugs found while building this (all fixed, all in the C driver, none in the asm kernel)

- q/k/v split of the fused QKV output is strided (columns at 0, D, 2D of
  each S-row), not contiguous - and the in-place split clobbered unread
  data; separate buffers + strided memcpy.
- QK^T needs the transpose packed correctly (k stored SxD, not DxS).
- Residuals must add to the pre-LN input, not the LN working buffer.
- The SIMD exp polynomial was evaluated with the coefficients in reverse
  order (Horner from the wrong end); the softmax normalization masked most
  of the damage (P is normalized) but the decoder's tighter 1e-5 target
  exposed it. Same bug was silently present in the attention benchmark.

## Larger-than-GEMM observations

- At S=2048 the decoder is attention-bound (5.48 ms of 6.69 ms): the O(S^2)
  scores materialization dominates - same flash-attention motivation as
  before. The MLP is only 1.22 ms.
- Fusing the QKV projection into one GEMM (N=3D) is 3x cheaper in tile
  restarts than three separate D-projection GEMMs.
- numpy's GELU (np.tanh) is ~45 ms at S=2048 vs our SIMD poly-exp GELU at
  a fraction of a ms - the elementwise ops dominate the numpy reference,
  not the GEMMs.
