# A* Score SIMD Experiment (Verilog → Tick-IR → Z3 Autovec → SSE2)

Goal

Validate an end-to-end SIMD path on a minimal A* building block: the score function `f = g + h`, evaluated lane-wise for multiple independent candidates.

Setup

- Verilog fixture: `fixtures/verilog/astar_score_simd_slices.v`
- Lanes: 8
- Lane width: 16 bits
- Packed width: 128 bits
- Baseline: raw scalar C (per-lane `uint16_t` add, wrap modulo 2^16)
- Optimizer: Z3-backed autovectorization (`stc.autovec_pass.autovectorize_tick_ir`)
- Backend: x86 SSE2 (`stc.backend_x86_sse2.emit_x86_sse2_c`)

What the experiment proves

- The Verilog frontend produces Tick-IR that preserves lane boundaries through `Slice` and `Concat`.
- `infer_simd_types` identifies the packed bus as `SimdType(lane_width=16, lanes=8)`.
- Z3-guarded autovectorization replaces the scalar lane-concat form with `SimdAdd(g, h)`.
- The SSE2 backend output matches a raw C reference across randomized test vectors.

Follow-up improvements

- Add autovec patterns for `Mux`/select so common A* building blocks like `abs`, `min`, and conditional updates can become `SimdBlend` and masked ops.
- Extend superopt candidate ops to include compares and blends (currently bitwise + add/sub only) for more realistic vector synthesis.
- Add a SIMD backend that supports non-128-bit SIMD values (e.g., mask vectors with `total_width != 128`) or define a canonical mask representation for x86 backends.
- Add a CLI experiment runner that emits both scalar and SIMD C and compares results automatically.

