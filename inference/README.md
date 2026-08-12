# Zen 5 AVX-512 C + asm Inference Stack

This directory is the *non-VeryLogo* half of the repository: a hand-scheduled
AVX-512 inference component stack, benchmarked and validated on this CPU
(AMD Ryzen 9 9950X3D, Zen 5). It is intentionally self-contained - nothing
here imports `stc/`, and `stc/` imports nothing from here (the scheduler's
Zen 5 target values are inlined in `stc/sched/target.py` with provenance
pointing here).

## Why "C + asm"

The GEMM micro-kernels are emitted as **GAS with a fixed register assignment
and instruction order** (`gemm_asm.py`), so no compiler pass can reorder,
spill, or reshape the intended sequence - gcc demonstrably does all three to
the equivalent intrinsics at >= 26 live zmm. The drivers are C with SIMD fp32
elementwise helpers (`llm_c_common.py`), property-tested.

## Layout

| path | contents |
|---|---|
| `aggen.py` | Agner Zen 5 instruction model (`inference/results/agner_zen5_avx512.csv` + locally measured corrections): per-instruction rt/latency/pipes, the GEMM peak oracle, tile selection, cycle predictions |
| `gemm_asm.py` | hand-scheduled GAS micro-kernel generator (bf16/int8 8x32 VPDPBF16PS/VPDPBUSD), `python -m inference.gemm_asm --family bf16 --tile 8x32` |
| `llm_c_common.py` | shared C template: f2b/b2f, exp_ps, packs, gemm driver, layernorm, gelu, softmax (property-tested) |
| `bench/` | the benchmark drivers (each compiles the asm kernel + a generated C driver, times best-of-N RDTSC pinned, cross-checks vs numpy/OpenBLAS) |
| `results/` | the Agner CSV and the per-component results docs |
| `tests/` | hypothesis property tests + the C checker harness (`pytest inference/tests/`) |

## The stack (bottom to top)

1. **GEMM micro-kernels** - INT8 (VPDPBUSD, uint8-A x int8-B) and BF16
   (VDPBF16PS), 8x32 tiles, 85-95% of the measured instruction ceiling.
2. **GEMM at scale** - 1024^3/128^3, ~93-95% of peak (register-budget tile
   selection from `aggen.py`).
3. **Attention head** - QK^T/sqrt(D) -> softmax -> PV; SIMD poly-exp softmax.
4. **MLP block** - GEMM + GELU + GEMM.
5. **Decoder layer** - LN, fused QKV, attention, out-proj, residuals, LN, MLP.
6. **Prefill** - embedding, L decoder layers, final LN, tied-embedding LM head.
7. **KV-cache decode** - the generation loop against a growing cache
   (rectangular BxS softmax; correctness at a short check horizon because
   bf16-vs-fp32 trajectories are chaotic feedback loops).
8. **Multi-head GQA attention** - H_q heads of D_head = D/H_q with H_kv
   shared key/value heads; measured the ~23% multi-head structural overhead.
9. **Multi-head GQA prefill** - the complete model at the real architecture.

Every rung was reviewed by fresh-context subagent reviewers, fixed, and
cross-checked against OpenBLAS-backed numpy on this CPU.

## Running

```sh
# property tests (needs AVX512-VNNI + AVX512-BF16; skips otherwise)
.venv/bin/python -m pytest inference/tests/

# a benchmark (compiles the asm kernel + C driver, times, checks vs numpy)
.venv/bin/python inference/bench/bench_mha_prefill_avx512.py --seq 512 --layers 6
```

## Known measurement caveats

- RDTSC runs at the TSC clock (below the boosted core clock), so absolute
  GF/s / tokens/s are conservative vs wall-clock; the same-run standalone
  reference and the numpy comparisons are relative and timebase-free.
- numpy (OpenBLAS 0.3.34) is a single unpinned wall-clock shot; its
  elementwise ops dominate its own time.
- bf16 decode trajectories decorrelate from fp32 beyond ~8 steps (chaotic
  feedback); correctness is validated at the stable check horizon.
