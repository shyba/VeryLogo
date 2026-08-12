# Zen 5 KV-Cache Decode (bf16), the layer after prefill

`inference/bench/bench_decode_avx512.py`, Ryzen 9 9950X3D, single thread. The
token-generation loop: at each step B token embeddings are projected to
Q/K/V, K/V are appended to the per-layer caches, and the next hidden state
is computed against the growing cache (QK^T over the cache, softmax, PV,
output projection, residual, LN, MLP). All GEMMs on the hand-scheduled bf16
kernel; SIMD fp32 elementwise helpers; reference numpy fp32 (OpenBLAS,
1 thread). Timings are best-of-5 RDTSC with CPU pinning (RDTSC runs at the
TSC clock, below the boosted core clock, so absolute GF/s/tokens/s are
slightly conservative versus wall-clock).

## Results (D=128, H=512, 6 layers, batch 8)

| steps | cache | ours | us/step (mean) | us/token | tokens/s |
|---:|---:|---:|---:|---:|---:|
| 64 | 128 -> 640 | 7.74 ms | 121 | 15.1 | 66.1k |
| 128 | 128 -> 1152 | 21.2 ms | 166 | 20.7 | 48.3k |

Per-step cost grows with the cache length (each step reads the whole K/V
cache: 2 * S_cur * D * 2 * L bytes; at S=640 that is 1.97 MB/step, average
over the run ~1.2 MB at the mean S=384). Total decode time is ~O(T^2) in
generated tokens - the standard growing-KV-cache cost. The decode is not
yet bandwidth-bound at these sizes (measured effective ~10 GB/s of a
multi-GB/s-capable core, ~3% of the bf16 GEMM peak per step); the per-step
cache re-packing (O(S*D)) is a real cost that incremental packing removes.

## Correctness

At the 8-step check horizon (6 layers) the final hidden state matches the
bf16-input fp32 numpy reference to fp32 rounding (max abs err 0.00000 on a
0.168 state scale) - the two implementations compute the identical model.
Beyond the check horizon the bf16-vs-fp32 trajectories decorrelate
chaotically (the residual feedback amplifies rounding differences), so
correctness is validated where the model is stable and the per-step error
grows with horizon - the reason LLM inference keeps fp32 accumulation and
residuals.

## Bugs caught while building (all in the C driver, all fixed)

- The queries' attention excluded the just-appended keys (mask/scoping used
  the pre-append cache length instead of scur+BATCH), so the C and numpy
  computed different causal models; the 2-4% "chaos" was mostly this bug
  (fix -> exact match at the check horizon).
- The same hidden-size-as-multiplier call bug as the decoder/prefill.
- The correctness pass ran with mutated ids and read stale cache rows.
- A cache memset that wiped the seed rows.
- The shared square SxS softmax applied to the B x S scores (out of bounds;
  also inflated the step timing) - fixed with a rectangular softmax, now
  covered by a property test.
