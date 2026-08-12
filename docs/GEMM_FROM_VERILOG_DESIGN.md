# Design: GEMM from Verilog through VeryLogo (fp/bf16 support exploration)

Status: **exploration + design only** (no code changes). Goal: get, from a
Verilog GEMM description through the VeryLogo pipeline, a target with the
same functionality and comparable performance to the hand-scheduled
`inference/bench/bench_gemm_avx512.py` kernels — using the instruction
specs in `inference/aggen.py` as the codegen oracle.

## 1. What the pipeline does today (verified by experiment)

Flow: `Verilog -> yosys (proc+opt, no abc) -> JSON netlist -> subset check
-> Tick-IR -> optimize (autovec / superopt / arith_classify) ->
CircuitState (bit-level gates) -> backend (generic / avr / x86-avx2 /
x86-avx512 / ptx / futhark)`.

Verified facts:

- **Word-level arithmetic survives the frontend for supported cells.** A
  Verilog `a + b` comes out of `reduced_tick_ir.bin` as a Tick-IR `Add`
  (`stc/subset.py` allows `$add`, `$sub`; the default yosys script stops
  before `abc`, so cells are not gate-blasted). `stc/tick_ir_to_circuit_state.py`
  is where bit-level lowering happens, downstream.
- **`$mul` is rejected at the frontend.** `stc/subset.py:SUPPORTED_CELL_TYPES`
  has no `$mul`; a Verilog `a * b` fails with
  `SubsetError: unsupported cell type: $mul`. This is the first blocker.
- **The IR already has a rich arithmetic set** (`stc/tick_ir.py`): scalar
  `Mul/Add/Sub/Div`, `SimdMulLo/HiU/HiS`, **`SimdMaddS16`** (i16 x i16 -> i32,
  PMADDWD-class), **`SimdFFma`** (fp32 FMA), `FAdd/FMul/...`, and the
  `FloatType`/`SimdType` type system.
- **The backends already emit the MAC/FMA building blocks.** Verified by
  building a Tick-IR by hand: `SimdMaddS16 -> _mm512_madd_epi16` and
  `SimdFFma -> _mm512_fmadd_ps` in `stc/backend_x86_avx512.py` /
  `stc/backend_x86_avx512_float.py`.
- `stc/autovec.py` is a z3-verified scalar->SIMD vectorizer (Add/Sub/min/max/
  compares); `stc/tick_ir_classify_arith.py` classifies Mul/Div up to
  `mul_div_max_width` (default 32) in `optimize_tick_ir` and can already
  produce `SimdMaddS16` from mul+add structure.

## 2. The gap (what "same result as the inference GEMM" needs)

| need | inference/ has | VeryLogo has today |
|---|---|---|
| i8 dot-product (VPDPBUSD) | hand-scheduled asm kernel | no IR op (`SimdMaddS16` is i16 PMADDWD) |
| BF16 FMA (VDPBF16PS) | hand-scheduled asm kernel | no IR op, **no BF16 type** |
| Verilog `*` accepted | - | **blocked at subset** |
| GEMM-structure recognition (K-loop dot + packed layout + accumulator chains) | the bench generator | only pattern `mul+add -> madd` for i16 |
| instruction-spec-driven schedule (>= 2*latency chains) | `inference/aggen.py` best_tile / predicted cycles | scheduler target has latencies but no GEMM scheduling |

## 3. Design (chosen: component-as-primitive, not recognize-passes)

Rationale: per-kernel `xpto_recognize` passes do not scale - one pass per
kernel class, fragile against code shape. Instead the GEMM is a
**first-class component** (a primitive in the technology library), matched
by interface identity, lowered per target, and instantiated by ordinary
Verilog hierarchy. This maps onto existing machinery:

- `stc/tech.py` already defines `Primitive` + `Technology` (per-target
  primitives with cost/depth models and `is_legal(expr)`) - the primitive
  library exists; a `Gemm` primitive slots in.
- The frontend keeps module names in the cell graph; submodules are
  currently rejected (`SubsetError: submodule instantiation not
  supported`). A *known* module name (the GEMM component) is lifted to a
  Tick-IR primitive instead of being flattened - the same mechanism, with
  an allowlist of primitive module names instead of a rejection.
- The instruction specs (`inference/aggen.py`) drive the primitive's
  per-target lowering (op choice, tile, accumulator-chain count).

### 3.1 The GEMM component (the interface contract)

```systemverilog
module gemm #(parameter M = 64, N = 64, K = 64, W = 8, ACCW = 32)
  (input  wire [W-1:0] a[M*K],   // row-major A
   input  wire [W-1:0] b[K*N],   // row-major B
   output wire [ACCW-1:0] c[M*N]); // row-major C, zero-init accumulate
endmodule
```

The module declares the *interface*; the body is either a real RTL MAC loop
(used for the bit-level reference / fallback target) or empty (opaque
primitive). The frontend matches the module by **name + exact port
signatures** (like a hardened macro in a standard-cell flow): a
`gemm`-named module with these ports lifts to the Tick-IR `Gemm` primitive.
Strict matching avoids accidental collisions; a Verilog attribute
(`(* verylogo_primitive = "gemm" *)`) can make the intent explicit.

### 3.2 The Tick-IR primitive + verification

- A `Gemm(a, b, M, N, K, W, ACCW)` expr (or a `GemmOp` with typed ports);
  a BF16 variant uses a new `FloatType(subformat=bf16)` on the ports.
- Verification is the pipeline's own machinery: `bounded_equiv` / `z3_encode`
  prove the primitive equals the bit-level reference (the RTL MAC body) -
  so "compiles naturally" is not a trust-me shortcut. The property tests
  from `inference/tests/` are the reference harness.

### 3.3 Per-target lowering (the Technology)

- `Technology.primitives()` declares the GEMM primitive per target:
  - Zen 5 AVX-512: lower to the VPDPBUSD/VDPBF16PS kernel, scheduling
    accumulator chains and the 8x32 tile from `inference/aggen.py`
    (>= 2*latency chains, packed K-major/N-interleaved layout). This is
    the existing "pipeline uses the instruction specs" integration.
  - PTX: dp4a / bf16.fma; generic: gates fallback (the RTL body).
- `is_legal` gates the choice; `cost_model`/`depth_model` let the
  optimizer schedule around it.

### 3.4 "Compile naturally" vs "force with inline tick"

- **Natural**: the frontend lifts a `gemm` module instance to the
  primitive; higher-level Verilog (a decoder layer, the transformer)
  instantiates the component like any module - no composition passes.
- **Force with inline tick** (the bridge, also useful long-term): lower
  the GEMM to a **tick-sequenced program** using the pipeline's existing
  tick machinery (`TickIR` steps, `fuse_ticks`, the scheduled backend's
  `circuit_steps_shared`): the K-loop is a K-tick accumulation, the
  component's ports are the tick inputs/outputs. This is the right model
  for variable-K decode and for getting the primitive working before the
  frontend plumbing is complete.

### 3.5 The next layer instantiates

A decoder-layer Verilog module instantiates `gemm` instances (QKV, scores,
PV, out-proj, MLP) plus elementwise primitives (layernorm/softmax as their
own components or the existing SIMD fp ops). The frontend lifts the known
instances, composes, and the backends emit per-target code. One primitive
per kernel class; everything else is hierarchy.

## 4. The fp/bf16 part ("change a lot", localized by the component model)

- A `FloatType(subformat=bf16)` (16-bit, 8-bit mantissa) on the component's
  data ports; `FloatConst` rounding via the property-tested f2b.
- The fp32 accumulator (SimdFAdd / SimdFFma already emit).
- The new primitive's per-target lowering (vdpbf16ps) - the fp support is
  localized to the type + the primitive, not spread across recognition
  passes. A pure-Verilog bf16 GEMM body for the fallback/verification path
  needs Yosys float cells or a bf16 cell library (flagged).

## 5. Acceptance ("the same result")

- Functionality: emitted kernel == naive reference (z3-provable + runtime
  check, as the inference property tests do).
- Performance: the pass must hit the measured instruction ceilings from
  `inference/results/`: VPDPBUSD 2/cyc (128 i8-MACs/cyc), VDPBF16PS 2/cyc
  (64 bf16-MACs/cyc), with the register-fit 8x32 tile — validated by
  compiling the emitted C and timing it with the existing bench harness
  (best-of-N RDTSC pinned).

## 6. Suggested first slice (smallest)

1. Subset: allow `$mul` -> `Mul`; a Verilog 8x8 mul reaches the avx512
   backend.
2. IR+backend: `SimdDotS8` -> `_mm512_dpbusd_epi32`; property-test the
   emitted kernel against a naive reference (reuse the inference harness).
3. Recognition: turn a Verilog MAC module (K-iterated mul+add) into
   `SimdDotS8` + accumulate; then a small GEMM driver with the packed
   layout, scheduled from `inference/aggen.py`.

## Open questions

- Yosys float-cell synthesis for a pure-Verilog bf16 GEMM vs IR-level entry.
- The GEMM K-loop as a multi-tick design (the pipeline's tick/fusion passes
  already handle sequential structure) vs a combinational unrolled K.
- Whether the packed-region emitter (`tick_ir_to_packed_circuit_state` +
  `packed_region_emit`) is the right vehicle for the packed B layout, or a
  dedicated packing pass.
