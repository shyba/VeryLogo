# Zen 5 Mixed-Precision Vector Overlap Results

Experiment: `inference/inference/results/bench_zen5_mixed_vector.py`.

Question: on the AMD Ryzen 9 9950X3D (Zen 5), do AVX-512 VNNI (INT8 dot)
and FP32 FMA instructions execute concurrently on the FP/vector backend, or
do they contend for the same execution resources (Agner: both P01)?

## Method

- Pinned single core (quietest of 32 logical CPUs), best-of-N perf core
  cycles, inline-asm loops with exactly 8 independent 512-bit accumulators
  per iteration (verified in objdump; no register-copy noise).
- All kernels share the identical 8-op-per-iteration structure; rates are
  computed from the *actual* per-iteration op count.

## Results (vec-ops/cycle, 512-bit, loadavg ~3)

| kernel | ops/cyc |
|---|---:|
| VPDPBUSD (VNNI) | 1.830 |
| VFMADD231PS (FP32 FMA) | 1.829 |
| **VNNI + FMA (alternating 4:4)** | **1.820** |
| VNNI + FMA (grouped / skewed) | ~1.82 |
| VDPBF16PS (8-chain; latency-limited) | 1.332 |

## Conclusion: full contention, Agner's P01/P01 model confirmed

The VNNI+FMA mix runs at 1.820 ops/cyc = 0.99x the single-stream rate of
either alone (1.83). Adding a second op class does not increase throughput:
both instructions share the same multiply/FMA execution resources (Agner's
P01), exactly as the published pipe model assigns them. This holds for every
instruction ordering tested (alternating, grouped, skewed).

An earlier version of this experiment reported ~3.66 ops/cyc ("full
overlap"); that was a **2x op-counting bug** - the mix kernels execute
8 ops/iteration (4+4), not 16. With the count corrected, the result is
unambiguous contention.

The VDPBF16PS 8-chain number (1.332) sits exactly on the dependency ceiling
8/6 = 1.333 (latency 6) and is therefore a *lower bound*, not the execution
throughput; the chain sweep in `inference/inference/results/bench_avx512_chains.py` resolves the
true rate to 2.0 ops/cyc (see `inference/results/avx512_vs_agner_results.md`).

## Practical consequence

For inference kernels on this CPU: interleaving VNNI and FP32-FMA work does
not add arithmetic throughput - they time-share the same pipes. Schedule
them as if they were the same resource class.
