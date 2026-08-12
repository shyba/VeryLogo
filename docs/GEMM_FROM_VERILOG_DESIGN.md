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

## 3. Design (minimal change set, in dependency order)

### Phase 0 — frontend: accept arithmetic cells
- Add `$mul` (and `$macc`, `$div` for later) to `stc/subset.py`, and map them
  in the yosys-JSON -> Tick-IR path to `Mul`/`Add` (the `$add` path already
  proves the mechanism). Constrain by `mul_div_max_width` as today.
- Result check: `a * b` survives to `reduced_tick_ir.bin` as `Mul`, and the
  avx512 backend emits a correct (if gated) result.

### Phase 1 — IR: the two new ops + BF16 type
- `SimdDotS8(a, b)` — i8 lanes, 4-element dot -> i32 lane (VPDPBUSD). Encode
  the **verified operand semantics**: AT&T `vpdpbusd src1, src2` has src1
  signed, src2 unsigned; for the uint8-A x int8-B GEMM, A must be the second
  source (`inference/results/bench_gemm_avx512_results.md`).
- `SimdFmaBF16(a, b, c)` — bf16 lane pairs -> fp32 accumulator (VDPBF16PS);
  pair layout (2 bf16 per 32-bit lane) is part of the type contract.
- `FloatType(subformat=bf16)` or a `Bf16Type(16, mantissa=8)`; `FloatConst`
  rounding via the f2b nearest-even function already property-tested in
  `inference/tests/`.

### Phase 2 — recognition/lowering: the GEMM pass
- A `gemm_recognize` pass on the Tick-IR: given the Verilog MAC structure
  (a `$macc`-shaped expression or a K-iterated mul+add), rewrite the
  dot-product chain into `SimdDotS8` / `SimdFmaBF16` + the fp32 accumulator
  sum, and emit the N-interleaved packing (reuse `SimdUnpack/Pack` or a
  packing loop) + the K-loop.
- **Schedule from the specs**: the pass reads `inference/aggen.py`
  (latency/rt/pipes) to choose the accumulator-chain count
  (>= 2 * latency -> >= 8 chains for VNNI, >= 12 for BF16) and the tile, so
  the emitted loop reaches the measured 2 ops/cyc. This is the existing
  "compiler pipeline uses the instruction specs" integration, now for GEMM.

### Phase 3 — backends
- `stc/backend_x86_avx512.py`: `SimdDotS8 -> _mm512_dpbusd_epi32(src, a, b)`
  with the A-unsigned/B-signed operand order; `SimdFmaBF16` in
  `stc/backend_x86_avx512_float.py` -> `_mm512_dpbf16_ps` + the f2b
  conversion for bf16 constants (helpers exist in the inference stack and are
  property-tested — copy them into the backend or a shared place).

### Phase 4 — verification
- Extend `stc/z3_encode.py` / `stc/bounded_equiv.py` to prove
  `SimdDotS8(a, b) ==` the bit-level i8 dot (the z3 machinery already
  encodes the other Simd ops), and `SimdFmaBF16` against an fp32-FMA
  reference on bf16-rounded inputs with tolerance.
- The acceptance property is the same as the inference tests: emitted kernel
  == naive reference over random shapes (reuse the hypothesis harness ideas
  from `inference/tests/`).

## 4. The fp/bf16 part ("change a lot")

- **Entry point**: Verilog fp arithmetic needs float cells; the pragmatic
  path is to enter fp at the Tick-IR level (FloatType already exists, the
  float backend already emits `_mm512_fmadd_ps`), and add a `FloatType(
  bf16)` variant + `SimdFmaBF16` + the f2b const rounding. A pure-Verilog
  bf16 GEMM would additionally need Yosys float synthesis or a bf16 cell
  library — flagged for a separate exploration.
- The fp32 accumulate (SimdFAdd chain) exists; the bf16-specific parts are
  the input rounding, the lane-pair layout, and the op semantics.

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
