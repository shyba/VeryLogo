# Zen 5 KV-Cache Decode (bf16), the layer after prefill

`scripts/bench_decode_avx512.py`, Ryzen 9 9950X3D, single thread. The
token-generation loop: at each step B token embeddings are projected to
Q/K/V, K/V are appended to the per-layer caches, and the next hidden state
is computed against the growing cache (QK^T over the cache, softmax, PV,
output projection, residual, LN, MLP). All GEMMs on the hand-scheduled bf16
kernel; SIMD fp32 elementwise helpers; reference numpy fp32 (OpenBLAS,
1 thread).

## Results (D=128, H=512, 6 layers, batch 8)

| steps | cache | ours | us/step | us/token | tokens/s | numpy | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 128 -> 640 | 7.78 ms | 121.6 | 15.2 | 65.8k | ~40 ms | ~5x |
| 128 | 128 -> 1152 | 21.2 ms | 165.8 | 20.7 | 48.3k | ~90-400 ms | 4-7x |

Correctness: final hidden state matches the bf16-input fp32 numpy reference
to ~2-4% at the 8-step check horizon (6 layers) - the bf16-vs-fp32 decode
trajectories are chaotic feedback loops, so correctness is validated where
the model is stable and the per-step error grows with horizon (the classic
reason LLM inference keeps fp32 accumulation and residuals).

## The memory-bound picture

Decode per step reads ~2 * S_cur * D * 2 * L bytes from the K/V caches. At
S=640, D=128, L=6 that is ~1.2 MB/step; the step cost grows with S_cur
(121 us at S~500 -> 166 us at S~900), i.e. total decode time is O(T^2) in
the number of generated tokens - the standard growing-KV-cache cost.

## Bugs caught while building (all in the C driver, all fixed)

- main passed H=512 as the hm parameter -> H=65536 (128x MLP work) - the
  same hidden-size-as-multiplier bug as the decoder and prefill.
- The clean correctness pass ran with ids already mutated by the timed
  loop, and read stale K/V cache rows the timed loop appended (future
  tokens leak into the check's attention).
- The cache memset for the check wiped the seed rows (started at row 0).
- The shared square SxS softmax was applied to the B x S decode scores -
  out-of-bounds reads/writes that corrupted the trajectory AND slowed the
  step (the decode needs a rectangular softmax).
