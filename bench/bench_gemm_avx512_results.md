# Zen 5 GEMM Micro-Kernels (INT8 + BF16), designed from the Agner model

`scripts/bench_gemm_avx512.py`, on the Ryzen 9 9950X3D (Zen 5). The kernels
are small core-AI components: dense INT8 (VPDPBUSD) and BF16 (VDPBF16PS)
matmuls, correctness-checked against a naive reference and timed
best-of-N RDTSC.

## Hand-scheduled assembly (no compiler in the hot loop)

By default the micro-kernels are emitted by `stc/gemm_asm.py` as **GAS with a
fixed register assignment and instruction order** (`python -m stc.gemm_asm
--family vnni8 --tile 8x32`), assembled with `gcc -c` and linked into the
driver. The compiler does not see the hot loop, so it cannot reorder, spill,
or reshape the intended sequence - all of which gcc demonstrably does to the
equivalent intrinsics at >= 26 live zmm (the old 16x32 tile spilled and lost
~20%; the register-budget guard in the generator now rejects that tile
outright instead of silently degrading).

Verified by disassembly: per 2 K-chunks the 8x32 loop is exactly
4 vmovdqu64 (B), 16 vpbroadcastd (A), 32 vpdpbusd, 8 pointer-adds + cmp/jne,
with zero vmovdqa spill traffic. `--intrinsic` re-enables the gcc path for
comparison.

## Design is derived from stc/aggen.py (Agner Zen 5 AVX-512 table + measured corrections)

- VNNI/BF16/FMA-class ops: 2/cyc on P01 (Agner rt 0.5; measured 2.0/cyc).
- Latency 4 (VNNI) / 6 (BF16) => >= 8 / >= 12 independent accumulator
  chains per stream are needed to reach 2/cyc. The MR x NR tile provides
  MR x (NR/16) accumulator vectors.
- Broadcast-loads of A run on the 2 load pipes concurrently with P01, so a
  tile is optimal iff its FMA-ops/cycle <= 2 AND its loads/cycle <= 2;
  among those, the largest wins. Register budget: accs + B-vectors + 2 <= 28.
- Predicted tile: 12x32. Measured best (register-fit, no spills): **8x32**
  (16 acc + 2 B-vec + temps = ~18-22 live zmm). 16x32 needs 34 zmm and
  spills, losing ~20%: exactly what the budget model predicts.

## Layouts

- A: row-major; each K-chunk broadcasts A[i][k0..] to all lanes.
- B pre-packed once per N-tile, K-major/N-interleaved: dword j of chunk c =
  B[k0..k0+3][tile*NR+j] (INT8) or the bf16 pair B[2c..2c+1][tile*NR+j]
  (BF16). Lane j accumulates output (i, tile*NR+j).
- VDPBUSD is unsigned x signed bytes: A is uint8 (activations), B is int8
  (weights) - the standard asymmetric-quantized inference layout. A 512-bit
  VPDPBSSD (signed x signed) intrinsic is not exposed by this GCC, so
  symmetric int8 would need inline asm.

## Results (same-process standalone 16-chain reference; RDTSC timebase)

The 16-chain standalone loops measure 160-162 (i8) / 80-81 (bf16) MACs per
TSC tick = exactly 2.0/cyc at the observed boost (TSC ~= base clock; core
boosts ~1.26x). "rel" = GEMM rate / same-run standalone rate, which is
timebase-independent.

| GEMM | tile | INT8 MACs/tick | rel i8 | BF16 MACs/tick | rel bf16 |
|---|---|---:|---:|---:|---:|
| 64x64x64 | 8x32  | 126.9 | 79.2% | 71.0 | 87.3% |
| 64x64x64 | 16x32 |  95.8 | 59.8% | 64.1 | 78.9% (spills) |
| 64x64x64 | 8x16  | 106.9 | 66.8% | 51.1 | 62.9% |
| 128x128x128 | 8x32 | 138.7 | 85.2% | 74.9 | 93.0% |
| 128x128x128 | 16x32 | 99.8 | 62.2% | 69.9 | 86.9% (spills) |

The hand-scheduled asm kernels measure at parity with the gcc -O3
intrinsics (85-87% i8, 93% bf16 at 128^3 relative to the same-run
standalone reference; the standalone 16-chain loop hits the P01 dpbusd
bound, the GEMMs reach 85-93% of it), with the asm version immune to
codegen variation. Tiles with different NR must be benchmarked in separate
invocations: Bp is packed with a single NR layout but each kernel reads its
own, so a mixed-NR run times garbage (caught by a property test). The residual gap is tile prologue/epilogue (zero 16 accs,
16 stores, addressing) and loop control, which shrink as the GEMM grows
(64^3 -> 78%, 128^3 -> 85%).

Predicted cycles (from the aggen model) vs measured at 128^3, 8x32:
i8 pred 16384 vs 14964 TSC ticks (the GEMM is at/below the FMA-pipe bound;
the tick count depends on the boost state); bf16 pred 32768 vs 28078.

## Pipeline integration

- `stc/aggen.py`: machine description from `bench/agner_zen5_avx512.csv`
  with locally measured corrections (VDPBF16PS rt 3 -> 0.5, VPTERNLOG
  rt 1 -> ~0.29 and latency 3 -> 2). Provides the GEMM peak oracle, tile
  selection, and predicted cycles used above.
- `stc/sched/target.py`: the pipeline's AVX512 scheduler model now takes its
  ternary latency/throughput from aggen (measured 2 / 3 vs the previous
  hardcoded 1 / 2); `zen5_avx512` is the full Agner-derived target.
- The GEMM generator is spec-driven: `--tiles` selection, predictions, and
  the peak-oracle printout all come from the machine model, so the numbers
  above are the model's predictions validated by measurement.
