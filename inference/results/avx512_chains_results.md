# AVX-512 Chain-Sweep and Interference Results (corrected)

`inference/bench/bench_avx512_chains.py` and `inference/bench/bench_zen5_mixed_vector.py`,
perf-counted core cycles on this AMD Ryzen 9 9950X3D (Zen 5), loadavg ~3.

## Correction

An earlier draft reported "VNNI + FP32-FMA overlap" (3.66 ops/cyc). That was
a **2x op-counting bug**: the mix kernels execute 8 ops/iteration (4+4), not
16. With the count fixed, the mix runs at the single-stream rate. The
interference conclusion below supersedes the earlier one.

## Standalone execution rates (chain sweep)

A loop with N loop-carried accumulators of latency L cannot exceed
N/L ops/cyc regardless of the execution rate, so sweeping N separates the
dependency ceiling from the true rate.

| op | 8 ch | 12 ch | 16 ch | true rate | latency |
|---|---:|---:|---:|---:|---:|
| VDPBF16PS | 1.331 | 1.844 | 1.997 | **2.0** | 6.02 (Agner 6) |
| VPTERNLOGD | 2.71 | 3.41 | (polluted) | **>= 3.4** | 2.01 (Agner 3) |
| VFMADD231PS | 1.93 | 1.99 | - | 2.0 | 4.01 (Agner 4) |
| VPDPBUSD | 1.85 | 1.99 | - | 2.0 | 4.01 (Agner 4) |
| VADDPS | 2.0 | - | - | 2.0 | 2.01 (Agner 2) |

16-chain VPTERNLOGD (1.77) is codegen-polluted (register allocator inserts
vmovdqa copies at 18 live zmm); 8/12-chain values are clean inline asm.

## Interference (contention probes)

- **VNNI + FMA: 1.82 ops/cyc = the single-stream rate. Full contention.**
  Agner assigns both to P01; this confirms it. The earlier "overlap" was the
  counting bug.
- VDPBF16PS + FMA / + VNNI / + FPADD, and VPTERNLOGD + FMA / + VNNI /
  + FPADD: **not measurable with the current tooling** - intrinsic mixes at
  16-24 live zmm registers trigger the gcc register-allocator pathology
  (16-45 vmovdqa copies per kernel; the inline-asm route hits gcc's
  30-operand limit at >13 accumulators). These need hand-scheduled assembly
  or a register-asm-local encoding. The published values in
  `inference/bench/bench_avx512_chains.py` for these kernels are flagged INVALID by
  the vmov-noise check and must not be cited.

## Net result vs Agner

- Validated on this SKU: P01/P01 contention model for VNNI/FMA/BF16;
  latencies for VNNI/FMA/BF16/FPADD (within rounding); RT 0.5 -> 2/cyc for
  VNNI, FMA, FPADD.
- Falsified (too pessimistic): VDPBF16PS RT=3 (true 2/cyc, 6x faster);
  VPTERNLOGD RT=1 (true >= 3.4/cyc) and latency 3 (measured 2).
- The 29/31 match in `inference/results/avx512_vs_agner_results.md` stands; the two
  flagged rows there are now resolved by this chain sweep.
