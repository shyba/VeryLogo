# AVX-512 Throughput vs Agner's Zen 5 Tables

`scripts/bench_avx512_vs_agner.py` measures single-stream 512-bit instruction
throughput on this machine (AMD Ryzen 9 9950X3D, Zen 5) for the instructions
STC's x86 backends emit, and compares against Agner Fog's published Zen 5
reciprocal-throughput values (`bench/agner_zen5_avx512.csv`, extracted from
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

## Documented conflicts (measured faster than Agner)

Both are verified clean loops (8 ops/iteration confirmed by disassembly and
perf instruction counts; no register-copy noise; core cycles from hardware
counters). The discrepancies are flagged, not adjudicated:

- **VDPBF16PS: 1.33 ops/cyc vs Agner RT=3 (0.333)** - exactly 4x. This SKU
  retires 2.4G vdpbf16ps in 1.8G cycles (implied freq 5.3 GHz, sane).
- **VPTERNLOGD: 2.53 ops/cyc vs Agner RT=1 (1.0)** - 2.5x. 8 independent
  accumulator chains at imm 0x96 (xor3); loop verified in objdump.

A plausible explanation would need port-level counters or Agner's raw
measurements; the benchmark records the silicon as measured.

## Provenance

Reference data: Agner Fog, "Instruction tables", Zen 5 sheet
(https://www.agner.org/optimize/), downloaded 2026-08-11, extracted into
`bench/agner_zen5_avx512.csv` (instruction, operands, ops, latency,
reciprocal throughput, pipes, notes, family). Register-register rows are
used for the comparison.
