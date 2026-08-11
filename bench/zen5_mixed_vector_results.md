# Zen 5 Mixed-Precision Vector Overlap Results

Experiment: `scripts/bench_zen5_mixed_vector.py`.

Question: on the AMD Ryzen 9 9950X3D (Zen 5), do AVX-512 VNNI (INT8 dot),
FP32 FMA, and BF16 dot instructions execute concurrently on the shared
FP/vector backend, or do they contend for the same execution resources?

## Method

- Pinned single core (quietest of the 32 logical CPUs, scanned first).
- Kernels: unrolled loops over 8 independent 512-bit accumulators, emitted
  as a single inline-asm block per iteration (verified in objdump: exactly
  8 vector ops + loop control, no register-copy noise).
- Core cycles counted with `perf stat`, best-of-10.
- All kernels use the identical 8-op-per-iteration structure so the
  comparison is apples-to-apples. A pure 16-op variant (two asm blocks)
  runs at exactly 1.0 vec-op/cyc regardless of op type — a loop-structure
  artifact, not a mixing effect; it is why structure must be held constant.
- Machine: 32 logical CPUs (16 cores, SMT), loadavg ~3 during the runs.

## Results (vec-ops/cycle, 512-bit)

| kernel | structure | ops/cyc |
|---|---|---|
| VPDPBUSD (INT8 dot) | 8× same | 1.835 |
| VFMADD231PS (FP32 FMA) | 8× same | 1.824 |
| VDPBF16PS (BF16 dot) | 8× same | 1.332 |
| **VNNI + FMA** | **alternating 4:4** | **3.665** |
| VNNI + FMA | grouped 4:4 | 1.822 |
| VNNI + FMA | grouped 6:2 / 2:6 | 1.825 / 1.821 |
| **VNNI + BF16** | **alternating 4:4** | **2.664** |

Two independent runs agree to ~0.4% on the alternating mixes.

## Interpretation

**VPDPBUSD + VFMADD231PS, truly interleaved: full arithmetic overlap.**
The alternating 4:4 mix runs at 3.665 ops/cyc = 1.835 + 1.824, exactly the
sum of the two single-stream rates (additive bound; ratio 1.002). The two op
types execute concurrently at their individual peaks. The full-overlap
prediction `max(4/1.835, 4/1.824) = 2.18 cyc/iter -> 3.67 ops/cyc` matches
the measurement exactly.

**The overlap is driven by instruction alternation, not the ratio.**
Grouping the same 8 ops as a burst (4 dpbusd then 4 fmadd) collapses to the
shared rate (1.82 ops/cyc); skewed ratios 6:2 / 2:6 in grouped form also sit
at ~1.82. Only the per-instruction alternation lets the scheduler keep both
vector pipes busy simultaneously.

**VNNI + BF16 overlap only partially.** The alternating mix reaches 2.66
ops/cyc = 84% of the additive bound (3.17), i.e. 1.45x beyond the shared-pipe
bound. VPDPBUSD and VDPBF16PS share some execution capacity (both are
dot-product-with-accumulate ops), so they contend partially.

**Per-stream peaks:** VPDPBUSD (1.835) and VFMADD231PS (1.824) are both near
the 2-per-cycle 512-bit peak. VDPBF16PS is slower per instruction (1.332) —
it performs two BF16 multiplies per lane.

## Practical consequences

- An inference kernel that *interleaves* VNNI and FP32-FMA work instruction
  by instruction can sustain ~2x the arithmetic rate of either alone on this
  CPU. Bursts or phase-separated streams time-share the backend.
- "Phase 1: INT8 matmul, phase 2: FP32 accumulate" does NOT get the overlap;
  software-pipelining the two streams together does.
- The results support the mixed model: scalar/control work and some loads
  overlap with vector work, and (new) VNNI+FP32-FMA arithmetic overlaps when
  interleaved; VNNI+BF16 only partially.

## Caveats

- Single machine, single run environment (shared box; loadavg ~3; quiet-core
  scan + best-of-10). Numbers are reproducible within ~0.5% run-to-run here.
- Inline asm is required: gcc's register allocator inserts `vmovdqa32`
  copies around `vpdpbusd` at 16 accumulators, and 16-op loops cap at
  1.0 ops/cyc — both would corrupt a naive measurement.
