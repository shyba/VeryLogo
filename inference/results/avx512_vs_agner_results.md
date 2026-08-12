# AVX-512 Throughput vs Agner's Zen 5 Tables

`inference/inference/results/bench_avx512_vs_agner.py` measures single-stream 512-bit instruction
throughput on this machine (AMD Ryzen 9 9950X3D, Zen 5) for the instructions
STC's x86 backends emit, and compares against Agner Fog's published Zen 5
reciprocal-throughput values (`inference/results/agner_zen5_avx512.csv`, extracted from
`instruction_tables.ods`, Zen 5 sheet).

Method: one inline-asm block per iteration, 8 independent accumulators,
pinned to the quietest logical CPU, best-of-N perf core cycles; verified by
disassembly (exactly 8 target ops + loop control, no register-copy noise)
and perf instruction counts.

## Results (ops/cycle, 512-bit, loadavg ~3-8)

| instruction | measured | Agner | ratio |
|---|---|---|---:|
| vpandd / vpandnd / vpord / vpxord | 3.87-3.99 | 4.0 | ~1.0 |
| vpaddd / vpaddw / vpsubd | 3.98-3.99 | 4.0 | ~1.0 |
| vpmulld / vpmullw / vpmulhw | 1.996 | 2.0 | 1.0 |
| vpmaddwd | 1.996 | 2.0 | 1.0 |
| vpdpbusd / vpdpwssd | 1.83 | 2.0 | 0.91 |
| **vdpbf16ps** | **1.33** | **0.333** | **3.99** |
| vfmadd231ps | 1.83 | 2.0 | 0.91 |
| vmulps / vaddps | 1.997 | 2.0 | 1.0 |
| vpminuw / vpmaxuw | 3.99 | 4.0 | 1.0 |
| vpslld / vpsrld / vpsrad | 1.996 | 2.0 | 1.0 |
| **vpternlogd** | **2.53** | **1.0** | **2.53** |
| vpcmpeqd / vcmpps | 1.90-2.00 | 2.0 | ~1.0 |
| vpblendmd | 3.87 | 4.0 | 0.97 |
| vpermd / vpshufb / packs / unpacks | 1.996-3.99 | 2.0/4.0 | ~1.0 |

29 of 31 instructions validate Agner within +/-15%. The vpdpbusd/wssd and
vfmadd231ps rows sit at ~0.91 of Agner's 2.0 - the consistent small deficit
of the accumulator kernels (loop-carried chains), not a pipe difference.

## Resolved via chain sweep (inference/inference/results/bench_avx512_chains.py)

An 8-accumulator loop is a *lower bound*, not the execution rate: with N
loop-carried chains of latency L the throughput cannot exceed N/L, so
8 chains pin VDPBF16PS to 8/6 = 1.33 and (with latency 2) VPTERNLOGD to
8/2 = 4.0 regardless of the true execution rate. Sweeping the chain count
resolves the real rates (all measurements perf-counted, codegen verified):

- **VDPBF16PS: 1.33 (8ch) -> 1.85 (12ch) -> 2.00 (16ch) = 2.0 ops/cyc.**
  Agner's RT=3 (0.333) is falsified; the true rate is 6x faster. The
  measured latency 6.02 matches Agner's 6.
- **VPTERNLOGD: 2.53 (8ch) -> 3.38 (12ch), latency 2.01 (not 3).**
  Agner's RT=1 is falsified; the true rate is at least ~3.4 ops/cyc
  (the 12-chain ceiling is 12/2 = 6.0, so 3.38 is still a lower bound;
  16+ chains are needed to test the 4/cyc P0123 hypothesis, but the
  inline-asm operand limit blocks 16 single-block chains).

Both latency measurements (2.01 for VPTERNLOGD vs Agner's 3) are direct
single-chain probes; the rest of the instruction set's latencies validate
Agner within rounding.

## Provenance

Reference data: Agner Fog, "Instruction tables", Zen 5 sheet
(https://www.agner.org/optimize/), downloaded 2026-08-11, extracted into
`inference/results/agner_zen5_avx512.csv` (instruction, operands, ops, latency,
reciprocal throughput, pipes, notes, family). Register-register rows are
used for the comparison.
