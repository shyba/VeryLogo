# CPU Dataflow Lowering (VeryLogo → inference targets): full working state

**Purpose**: self-contained knowledge base for AI dev sessions. This
document survives compaction: a fresh session with no chat history should
be able to continue the work from here alone (plus the repo). It captures
(a) the VeryLogo compiler pipeline mechanics, (b) the Zen 5 AVX-512
inference stack that lives in `inference/`, (c) the verified instruction
timings and the bugs catalog, and (d) the agreed design for lowering a GEMM
from Verilog onto the CPU "device floor" via a component-as-primitive model.

**Branch**: `local_target_c_asm_inference` (local only, no push). The
inference stack was split from the VeryLogo core on this branch
(commits `0f3151d` … `4d82269`). Both halves must stay self-contained.

**Last updated**: 2026-08-13, packed int8 `gemm` hierarchy lift and generated
scalar/VNNI C path implemented; BF16 and hand-scheduled peak-rate lowering
remain follow-on work.

---

## 1. The two halves of the repo

### 1.1 VeryLogo core (`stc/`, `scripts/`, `tests/`, `bench/`, `docs/`)

A "space-time compiler": restricted Verilog → Yosys netlist → Tick-IR
(`S' = f(S, I)`, `O = g(S, I)`) → solver-guided optimization (Z3/SMT,
superopt, autovec, ternary/LUT mapping) → bit-level or packed CircuitState →
scheduled code for `generic` / `avr` / `ptx` / `x86-avx2` / `x86-avx512` /
`futhark`. Research threads: AES S-box circuit optimization (Boyar-Peralta
32-AND, superopt), BP128 bitsliced AES CUDA kernels, Keccak compilation,
SIMD autovectorization, packed word-level lowering, region decomposition,
tick fusion.

Invocation: `.venv/bin/python3 -m stc <input.v> --out <dir> [--backend ...]`
Use `--backend x86-gemm` for the shaped int8 GEMM path.
(`python3 -m stc.cli` does NOT run main; use `python3 -m stc`).

**Crucial frontend fact**: the default Yosys script is
`read_verilog; proc; opt; opt_clean; write_json` — it stops *before* `abc`,
so arithmetic cells (`$add`, `$sub`, …) survive as word-level cells, not
gates. `stc/subset.py:SUPPORTED_CELL_TYPES` allow-lists cells; unsigned
`$mul` is now accepted and signed `$mul` parameters are still rejected until
signed semantics are specified. `$add` and unsigned `$mul` survive to
Tick-IR as word-level `Add`/`Mul` (verified with frontend extraction and
binary-IR round trips).

Pipeline files: `stc/yosys_frontend.py` (yosys script), `stc/yosys_json.py`
(JSON → cells), `stc/subset.py` (cell allowlist; rejects submodules:
"submodule instantiation not supported"), `stc/tick_ir.py` (the IR, incl.
`FloatType`, `SimdType`, scalar + SIMD arithmetic), `stc/reduce.py`
(`optimize_tick_ir`, `arith_classify`, `mul_div_max_width`),
`stc/autovec.py` (z3-verified scalar→SIMD vectorizer), `stc/superopt.py`,
`stc/tick_ir_to_circuit_state.py` (bit-level gates),
`stc/tick_ir_to_packed_circuit_state.py` + `stc/packed_region_emit.py`
(packed word-level), `stc/backend_x86_avx512.py` (int),
`stc/backend_x86_avx512_float.py` (fp), `stc/backend_sched.py`,
`stc/sched/*` (list scheduler, target models), `stc/tech.py`
(**`Primitive` + `Technology` abstraction** — per-target primitives,
`is_legal`, cost/depth models, and the lowering dispatcher — the home for the
GEMM primitive),
`stc/z3_encode.py` / `stc/bounded_equiv.py` (verification).

**Scheduler target**: `stc/sched/target.py` inlines the measured Zen 5
values (ternary latency 2, throughput 3) with provenance pointing at
`inference/aggen.py`; `stc/` has zero imports of `inference/`.

### 1.2 Inference stack (`inference/` — NOT VeryLogo)

Hand-scheduled AVX-512 C+asm inference components, benchmarked on the
9950X3D. Self-contained: imports only within `inference/`; the bench
drivers insert the *repo root* on sys.path (`parent.parent.parent`) and
import `from inference.gemm_asm import ...` / `from
inference.llm_c_common import COMMON_C`.

```
inference/
  aggen.py               # Agner Zen 5 machine model + measured corrections
  gemm_asm.py            # GAS micro-kernel generator (fixed registers)
  llm_c_common.py        # shared C helpers (property-tested)
  bench/                 # 10 benchmark drivers (see §4)
  results/               # agner_zen5_avx512.csv + per-component docs
  tests/                 # hypothesis property tests + C checker harness
  README.md
```

`inference/aggen.py` loads `inference/results/agner_zen5_avx512.csv`
(28 rows extracted from Agner's Zen 5 sheet) and applies
`MEASURED_CORRECTIONS`: **VDPBF16PS rt 3→0.5, VPTERNLOG rt 1→0.294 and
latency 3→2**. It provides `get_machine()`, per-instruction
rt/latency/pipes/MACs, `gemm_peak_macs_per_cycle`, `best_tile`, and
`predicted_gemm_cycles`.

---

## 2. The Zen 5 machine facts (measured, verified, do not re-derive)

All measured on the Ryzen 9 9950X3D (Zen 5), single core, perf or RDTSC.
Sources: `inference/results/avx512_vs_agner_results.md` (31-instruction
validator, 29/31 within ±15%), `inference/results/avx512_chains_results.md`
(chain sweeps), `inference/results/zen5_mixed_vector_results.md`.

| instruction | rt (cyc/instr) | ops/cyc | latency | pipes |
|---|---:|---:|---:|---:|
| VPDPBUSD (VNNI i8 dot, 64 MACs) | 0.5 | 2.0 | 4 | P01 |
| VFMADD231PS (fp32 FMA, 32 MACs) | 0.5 | 2.0 | 4 | P01 |
| VDPBF16PS (bf16, 32 MACs) | **0.5 (Agner 3!)** | 2.0+ | 6 | P01 |
| VPTERNLOGD | **0.294 (Agner 1!)** | ~3.4 | **2 (Agner 3!)** | P0123 |
| VADDPS / PADD | 0.5 | 2.0 | 2 | P0123/P23 |
| VPBROADCASTD (zmm,m32) | ~0.5 | 2.0 | - | load pipes |

Key verified semantics:

- **VDPBUSD operand order**: AT&T `vpdpbusd src1, src2, dst` has **src1
  signed, src2 unsigned**. For the uint8-A × int8-B GEMM, A must be the
  *second* source: `vpdpbusd %b, %a, %acc`. Pinned by a micro-probe.
  (There is no 512-bit VPDPBSSD intrinsic in GCC 14; symmetric int8 would
  need inline asm.)
- **Instruction contention**: VNNI and fp32-FMA **contend** (both P01) —
  a 4:4 mix runs at the single-stream rate, NOT additive. An earlier
  "full overlap" result was a 2× op-counting bug in the mixed-vector
  benchmark (kernel had 8 ops/iter, driver counted 16).
- **TSC vs core clock**: RDTSC runs at the TSC clock (~3.7 GHz nominal,
  below the boosted core ~5.4 GHz). All RDTSC-based GF/s/tokens/s are
  *conservative* vs wall-clock; same-run standalone references are
  timebase-free and are the honest comparison.
- **GEMM peak oracles**: INT8 (VPDPBUSD) = 128 MACs/cyc; BF16
  (VDPBF16PS) = 64 MACs/cyc; fp32 (FMA) = 64 MACs/cyc.
- **Accumulator-chain rule**: to sustain 2 ops/cyc, a stream needs
  >= 2 × latency independent chains: VNNI >= 8, BF16 >= 12, fp32 >= 8.

---

## 3. The hand-scheduled asm GEMM kernel (the reference implementation)

`inference/gemm_asm.py::emit_gemm_kernel(family, mr, nr, name, unroll)`
emits GAS (AT&T) for an 8×32 tile, **fixed register assignment and
instruction order** — the compiler never sees the hot loop. This is the
"result" the Verilog→VeryLogo path must reproduce.

Register map (bf16/vnni8 8×32, unroll=2):
- accs: zmm0–7 then zmm16–23 (16 accumulator vectors = 8 rows × 2
  col-blocks),
- B vectors: zmm24–27 (2 chunks × 2 unrolled),
- broadcast temps: zmm28–29,
- GPR: r9–r15 + rbx/r12–r15 saved (callee-saved pushed/popped), row
  pointers advance by `kpc × a_bytes × unroll` per iteration; B pointer
  advances `nr*4 × unroll`; loop exits on `cmp %rax, %r9` with
  `rax = A + K*a_bytes`.

Layout contract (shared by every driver):
- A: row-major, M×K; per K-chunk the kernel broadcasts a 32-bit group
  `A[i][k0..]` to all lanes.
- B packed (`pack_b_bf16_32` / `pack_b_bf16_32_T`): K-major,
  N-interleaved. `pack_b` takes B as K×N row-major; `pack_b_T` takes B as
  N×K row-major and packs B^T (this is what QK^T needs — **K is stored
  S×D, so QK^T uses pack_T**). dword j of chunk c = the k-group for
  output column j.
- The GEMM driver: `for n0 in steps of 32: for m0 in steps of 8:
  gemm_bf16_8x32_asm(A+m0*K, Bp+(n0/32)*(K/2)*32, C+m0*N+n0, K, N)` — note
  the **C row stride is N** (the GEMM's N), which caused a head-scatter
  bug (§5 item 11). Constraints: M%8==0, N%32==0, K%2==0; the K-loop also
  requires **K % (kpc × unroll) == 0** (8 for vnni8, 4 for bf16).

Measured kernel rates (single thread): 85.2% i8 / 93.0% bf16 at 128³
and 79.2%/87.3% at 64³ of the instruction ceiling (128³ and 64³ are the
measured sizes in inference/results/bench_gemm_avx512_results.md); best tile 8×32 (16 accs + B vecs + temps ≤ 32 zmm; 16×32
spills and loses ~20% — `best_tile` in aggen enforces the register
budget and MR≤8).

The 8×32 tile is the floor-planning primitive: 16 independent i32 ALU
"columns" (width) × the K-loop and ≥8 accumulator chains (time/length).

---

## 4. The benchmark stack (bottom to top) — each is a numbered layer

All drivers: compile the asm kernel + a generated C driver, time
best-of-N RDTSC with `sched_setaffinity(CPU 4)`, check correctness vs a
bf16-input fp32 numpy reference (OpenBLAS 0.3.34 via numpy 2.5.2 in the
venv, `OPENBLAS_NUM_THREADS=1` set *before* the first numpy import).

| # | driver | what | headline result |
|---|---|---|---|
| 1 | `bench_gemm_avx512.py` | INT8 + BF16 GEMM, tile sweep, `--intrinsic` opt-in | 8×32: 85–95% of peak; mixed-NR runs rejected |
| 2 | `bench_attention_avx512.py` | single-head attention (QK^T→softmax→PV) | S=2048: 6.0 ms vs numpy 15.9 (2.6–6×, load-dependent) |
| 3 | `bench_decoder_avx512.py` | decoder layer (LN, fused QKV, attn, MLP) | S=2048: 6.7 ms, ~12–31× vs numpy |
| 4 | `bench_llm_prefill_avx512.py` | full prefill (embed, L layers, LN, tied LM head) | S=512 L=6: 8.6 ms, 59k tok/s |
| 5 | `bench_decode_avx512.py` | KV-cache decode (batch 8, growing cache) | 128 steps: 21 ms, 48k tok/s; O(T²) |
| 6 | `bench_mha_avx512.py` | GQA multi-head decoder layer | multi-head ~23% slower at same FLOPs |
| 7 | `bench_mha_prefill_avx512.py` | full GQA prefill (H_q=4, H_kv=2, D_head=32) | 9.89 ms vs single-head 8.63 ms (~15%) |
| — | `bench_avx512_chains.py` / `vs_agner.py` / `zen5_mixed_vector.py` | instruction sweeps | §2 numbers |

Correctness conventions: bf16 weights/activations, fp32 accumulation and
residuals; the numpy reference replicates the C's exact quantization
points (bf16 nearest-even `f2b`). Decode correctness is validated at a
**short check horizon** (min(T,8)) because bf16-vs-fp32 decode
trajectories are chaotic feedback loops that decorrelate beyond ~8 steps.

The SIMD helpers (`inference/llm_c_common.py`): `f2b`/`b2f` (nearest-even
bf16), `exp_ps` (poly 2^r, **clamped to t ∈ [-126,127]**), packs, gemm
driver, `split_qkv` (strided: columns 0/D/2D of each S-row, must use
separate output buffers), `layernorm`, `gelu`, `softmax` (square S×S) and
`softmax_rect` (B×S for decode).

---

## 5. The bugs catalog (all found and fixed; the property tests guard these classes)

Future sessions MUST NOT re-hit these. The property tests
(`inference/tests/`, `pytest inference/tests/` — 17 pass + 1 xfail) guard
the SIMD semantics and the GEMM; the xfail documents the still-latent
multi-NR Bp packing defect.

1. **Hidden-size-as-multiplier call bug (4 occurrences!)**: every driver's
   `main()` must pass the `--hm` *multiplier* to `run_c(...)`/`gen_c(...)`,
   not the computed hidden size H. Passing H=512 as `hm` → H=65536 → 128×
   MLP work and silent 100× slowdowns. Symptom: absurd ms/layer while the
   binary is fast under perf. **Always check this first when timings look
   insane.**
2. **VPDPBUSD operand order**: src1 signed, src2 unsigned (A must be the
   second source). §2.
3. **QKV split is strided**: the fused QKV GEMM output is (S, 3D); q/k/v
   are column blocks at 0/D/2D of each row — NOT contiguous. An in-place
   split clobbers unread data; use separate buffers + strided memcpy.
4. **QK^T needs pack_T**: K is stored S×D; packing it as if D×S
   scrambles the scores. The softmax normalization masks the damage, so
   the correctness check still "passes" — the error attribution was
   wrong until fixed (attention err ~0.0011 → 0.00028 after the fix + an
   honest bf16-quantized numpy reference).
5. **Residual vs LN buffer**: the residual `x += o` must use the *pre-LN*
   input; LN mutates its buffer in place.
6. **exp polynomial Horner order**: the coefficients were folded from the
   wrong end (exp(0) = 0.0013). The softmax normalization masked most of
   it; the decoder's tighter target exposed it. Property: exp(0)=1,
   exp(a+b)=exp(a)exp(b).
7. **Decode causal mask/scoping**: the QK^T must scope and mask at
   `scur + BATCH` (including the just-appended keys), not the pre-append
   `scur`. Wrong mask = different causal model = 2–4% check error that
   looked like "chaos" until the fix → 0.00000.
8. **Decode check-pass state**: the correctness pass must reset `ids`
   (mutated by the timed loop) and clear the K/V cache rows past S0
   (stale "future" appends leak into the check's attention).
9. **Rectangular softmax**: the shared softmax assumes a square S×S
   matrix; decode scores are B×S → out-of-bounds reads/writes (also
   inflated the timing). Use `softmax_rect`.
10. **MHA b_q overflow**: b_q was declared S×D_head but the LN
    quantization wrote S×D into it (4× overflow at H_q=4), surviving only
    via .bss aliasing. Split into full-width b_q + per-head b_qh. Always
    run the generated C under `-fsanitize=address` before trusting it.
11. **MHA head PV stride collision**: the per-head PV GEMM writes its
    output with row-stride N=D_head, but the concatenated f_o needs
    row-stride D=128 → head h's rows overwrite head h−1's columns. Fix:
    write each head to a contiguous o_h (S×DH) temp, then scatter at the
    D stride.
12. **LM head buffer**: the clean-pass head GEMM wrote S×V logits into
    f_g (S×D scratch) — a 4M-float write into a 128KB buffer; surfaces
    only at V=16384 and only at -O3. Head output needs its own S×V
    buffer.
13. **Best-of-N early break**: the decode timing loop broke after rep 0
    (cache filled SMAX), making "best-of-5" a single cold rep. Don't
    break; reps re-run the same deterministic decode over the same cache
    rows.
14. **Debug dumps in timed regions**: `fopen/fwrite` (o.bin etc.) left
    inside timed loops contaminated ms/layer. Check for stray I/O in
    timed code.
15. **r-string printf escapes**: C templates built with `r"""..."""` need
    single `\n` in the C printf; a doubled `\\n` prints a literal
    backslash-n and the parser fails on `float("123\\n")`.

---

## 6. The GEMM-from-Verilog design (agreed direction — see
`docs/GEMM_FROM_VERILOG_DESIGN.md`)

### 6.1 Why not recognize-passes

Per-kernel `xpto_recognize` passes don't scale (one pass per kernel class,
fragile). Instead the GEMM is a **first-class component**: a Verilog
module with a strict interface contract, lifted to a Tick-IR primitive by
**module name + exact port signatures** (interface identity, not
structural matching), lowered per target, instantiated by ordinary
hierarchy.

### 6.2 The AES/SHA precedent (repo evidence)

- **AES worked** by exactly this shape: the S-box is a primitive with a
  known optimal implementation (Boyar-Peralta 32 AND gates,
  `docs/sboxgates.md`), synthesized once by a targeted technique
  (superopt), then **bitsliced instances-as-lane-width** (BP128 = 128
  parallel instances in vector lanes), then composed. The BP128 dynamic
  family (`docs/AES_BP128_DYNAMIC_FAMILY.md`) generates parameterized
  variants (key_bits / ctr_group / key_source axes) — the shape of a
  parameterized GEMM component.
- **Keccak/SHA3 is the dead-end boundary**: the generic bit-level path
  explodes (40k gates; `scripts/analyze_keccak_gate_explosion.py`
  exists because of it); packed is 0.14× smaller. Designs that produced
  "megabytes of spaghetti" went through generic lowering without the
  primitive decomposition.

### 6.3 The device floor (FPGA/DSP analogy, made concrete)

| FPGA | VeryLogo C/asm target |
|---|---|
| DSP blocks + LUTs | the vector ALUs: **width = 16 independent i32 ALUs per zmm** (64 i8 / 32 bf16 lanes) |
| clock cycles | **length = time**: K-loop cycles + accumulation depth |
| DSP pipeline stages | **piece repetition = instruction timing**: ≥2×latency accumulator chains (VNNI ≥8, BF16 ≥12) |
| device timing tables | `inference/aggen.py` (latency/rt/pipes) |
| router/floorplanner | the backend Technology: places the RTL dataflow onto (lane, cycle, chain) |

Verilog expresses the dataflow (the MAC network, pack, accumulate); the
backend places it on the ALU×time floor using aggen as the timing model.
The emitted kernel is the placed netlist — compact by construction, never
gate-blasted (the S-box precedent).

### 6.4 The component contract

```systemverilog
module gemm #(parameter M = 64, N = 64, K = 64, W = 8, ACCW = 32)
  (input  wire [M*K*W-1:0] a,       // packed row-major A
   input  wire [K*N*W-1:0] b,       // packed row-major B
   output wire [M*N*ACCW-1:0] c);   // packed row-major C
endmodule
```

Body = either a real RTL MAC loop (reference/fallback target) or empty (opaque
primitive). Yosys' normalized derived-module metadata resolves `M/N/K`; the
frontend lifts only a known `gemm` module and checks exact packed widths. A
BF16 variant needs a NEW element type: `FloatType` only supports widths 32/64
with no subformat knob (tick_ir.py:40-52).

### 6.5 Mechanism (mapped onto existing pipeline)

1. **Done (frontend)**: `stc/subset.py` admits only known GEMM module types,
   `stc/yosys_json.py` retains derived-module attributes/parameter defaults,
   and `stc/extract.py` lifts the packed instance to `GemmCall`.
2. **Done (IR)**: `GemmCall` carries shape, element widths, signedness, and
   row-major layout; interpreter, Z3/reference expansion, binary serialization,
   reducer, classifier, and variable replacement are covered. The
   instruction-sized `SimdDotU8S8AccI32` still maps to VPDPBUSD. BF16 remains
   future work.
3. **Done (Technology + target)**: `stc/tech.py` has a lowering dispatcher and
   `x86-gemm` primitive. `stc/backend_x86_gemm.py` emits the packed
   `stc_eval` ABI, using VPDPBUSD for `u8×s8→i32` shapes with `N%16==0,K%4==0`
   and a scalar fallback otherwise. The hand-scheduled 8×32 chain schedule is
   a separate performance phase.
4. **"Inline tick"** (the bridge, and the right model for variable-K
   decode): lower the GEMM to a K-tick program. CAVEAT (verified by
   review): this is NOT a free ride on existing machinery - the
   word-level SIMD backends REJECT designs with state
   (backend_x86_avx512.py:106-108, backend_x86_avx512_float.py:71-72),
   and the tick-stepping emitter (`circuit_steps_shared`, emitted from
   stc/sched/emit/avx512.py / avx512_u64.py / packed_region_emit.py, not
   backend_sched.py) uses one in_io for all ticks. A stateful word-level
   emit is net-new work.
5. **Verification**: `bounded_equiv` / `z3_encode` prove the primitive
   equals the RTL body; the inference property harness is the reference.
6. **Next layer instantiates**: a decoder-layer Verilog module
   instantiates `gemm` instances (QKV, scores, PV, out-proj, MLP) plus
   elementwise primitives — ordinary hierarchy, no composition passes.

### 6.6 First slice (smallest)

1. **Done:** allow unsigned `$mul` → Tick-IR `Mul`; a Verilog
   multiply is extracted, interpreted, and serialized in Tick-IR.
2. **Done:** `SimdDotU8S8AccI32` → `_mm512_dpbusd_epi32`; the
   interpreter, Z3, binary-IR, reference-Verilog, auto-dispatch, and a real
   AVX-512 compile/run equivalence test cover the instruction-sized slice.
3. **Done:** the parameterized packed component reaches generated C through
   `--backend x86-gemm`; generic bounded expansion provides the reference path.
4. Performance phase: replace the straightforward VNNI loop with the measured
   8×32 register-chain kernel and validate against the inference ceilings.

### 6.7 Open questions

- Yosys float-cell synthesis for a pure-Verilog bf16 GEMM vs entering fp
  at the Tick-IR level (recommended: IR level — FloatType exists).
- GEMM K-loop as a multi-tick design vs combinational unrolled K (tick
  path fits decode; unrolled fits prefill).
- Packed B layout: dedicated packing pass vs reusing the packed-region
  emitter (`packed_region_emit`).

---

## 7. Verification and tooling conventions

- **VeryLogo tests**: `.venv/bin/python3 -m unittest discover -s tests`
  (1024 pass, 19 skipped — verified by running on 2026-08-13; the
  pressure-scheduler test needs xor throughput >= 2 on the AVX512 target,
  so keep it at 2 rather than raising it to the measured 3).
- **Inference tests**: `.venv/bin/python3 -m pytest inference/tests/`
  (17 pass + 1 xfail — verified by running on 2026-08-12; requires
  avx512_vnni + avx512_bf16; the SIMD checker C is compiled once per
  process; checker buffers are `buf[65536]` — don't reference an
  undefined BUFSZ. NOTE: the split_qkv round-trip property is mode 6 in
  the checker (mode 3 is softmax_rect) — a past duplicate made it
  unreachable).
- **Running a benchmark**: `cd repo-root; .venv/bin/python3
  inference/bench/bench_mha_prefill_avx512.py --seq 512 --layers 6`
  (the scripts insert the repo root on sys.path and import the
  `inference` package; they compile the asm kernel with `gcc -c` and the
  C driver with `gcc -O3 -march=native`).
- **Before trusting any generated C**: build it with
  `-fsanitize=address` and run; then time with best-of-N RDTSC pinned to
  a quiet CPU (the scripts pin CPU 4 via `sched_setaffinity`).
- **numpy/OpenBLAS**: 2.5.2 in the venv; `OPENBLAS_NUM_THREADS=1` must be
  set BEFORE the first numpy import (after is too late).
- **Measurement caveats**: RDTSC at TSC clock < core boost → absolute
  rates conservative; numpy is a single unpinned wall-clock shot; the
  shared box's load makes wall-clock numbers noisy — prefer the
  same-process relative comparisons.

---

## 8. Immediate next actions (for a fresh session)

1. Read `docs/GEMM_FROM_VERILOG_DESIGN.md` (the full design, incl. the
   floor model) and this document.
2. Keep the component lift, generic fallback, and VNNI ABI tests green.
3. Implement and measure the 8×32 register-chain lowering against
   `inference/aggen.py`, then add BF16 as a separate type/op slice.
4. Keep both halves self-contained (`stc/` no inference imports,
   `inference/` no stc imports).
