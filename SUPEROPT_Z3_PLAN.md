# Z3 Superoptimizer Plan (Typed Search for SIMD)

Goal

Extend the Z3-backed superoptimizer to synthesize SIMD expressions that require intermediate results of different types, especially SIMD masks, so it can discover patterns like `SimdEq`, `SimdUlt`, and `SimdBlend` from scalar-lane specifications emitted by the Verilog frontend.

Current blockers

- Single-output-type search space: `superopt_expr` only enumerates candidates that have the same type as the final output, so it cannot build masks (e.g. `SimdType(lane_width=1, lanes=N)`) as intermediates when the output is a value vector, and it cannot build value vectors as intermediates when the output is a mask.
- No ternary operators in the enumerator: `SimdBlend(mask, a, b)` and scalar `Mux(cond, a, b)` are not part of the candidate grammar.
- Compare ops not in the candidate grammar: unsigned/signed SIMD compares are not enumerated, so masks cannot be synthesized even if multi-type terms were allowed.
- Cost model gaps: `expr_cost` does not assign stable costs to the missing ops, which makes “best program” selection unstable once they are added.

Target patterns (first-class, solver-verified)

- Mask synthesis
  - Lane-wise scalar compares lowered as `SimdEq/SimdUlt/SimdUle/SimdUgt/SimdUge` (output type is a SIMD mask).
- Blend/select synthesis
  - Lane-wise scalar `Mux(cond_lane, a_lane, b_lane)` lowered as `SimdBlend(mask, a, b)` where `mask` is a SIMD mask.
  - Concrete A* building blocks:
    - Score: `f = g + h` (already covered by `SimdAdd`)
    - Select: `o = (a < b) ? a : b` (covered by `SimdMinU` and also representable as `SimdBlend(SimdUlt(...), a, b)`)

Design: typed candidate enumeration

Core change

- Replace the single `by_size[size] -> [Expr]` store with a typed store:
  - `by_size[(type_key, size)] -> [Expr]`
- Enumerate expressions for a bounded set of types derived from the spec:
  - Always include the spec output type.
  - For SIMD outputs, also include the corresponding mask type `SimdType(lane_width=1, lanes=out.lanes)`.
  - Include any “source” SIMD value types found among the used variables (this is required to build masks via compares).
  - Include bitcasts between `SimdType(total_width=W)` and `BitVecType(width=W)` as optional glue when needed.

Operator grammar (phased, keep caps tight)

Phase A: binary and unary ops (typed)

- Boolean: `Not`, `And`, `Or`, `Xor` (if needed for non-SIMD specs).
- BitVec: `Not`, `And`, `Or`, `Xor`, `Add`, `Sub` (existing).
- SIMD value: `SimdNot`, `SimdAnd`, `SimdOr`, `SimdXor`, `SimdAdd`, `SimdSub`, `SimdMinU`, `SimdMaxU`, `SimdMinS`, `SimdMaxS`.
- SIMD compare (value -> mask): `SimdEq`, `SimdUlt`, `SimdUle`, `SimdUgt`, `SimdUge`, plus signed compares if required by fixtures.

Phase B: ternary ops

- `SimdBlend(mask, a, b)`:
  - `mask` must be `SimdType(lane_width=1, lanes=N)`
  - `a` and `b` must match SIMD value type `SimdType(lane_width=W, lanes=N)`
  - returns SIMD value type
- Scalar `Mux(cond, a, b)`:
  - keep out of the SIMD candidate grammar initially; prefer `SimdBlend` for lane-wise select.

Search strategy

- Keep current CEGIS loop structure for the final output type:
  - Use counterexamples to cheaply reject candidates by evaluation.
  - Use Z3 to obtain new counterexamples and to prove equivalence.
- For typed enumeration, only “score/prove” candidates whose type matches the spec output type.
- Maintain strict caps:
  - `max_nodes` remains the main knob.
  - Keep per-type candidate caps and a global cap.
  - Prefer generating fewer, more relevant operator combinations over broad grammars.

Cost model updates

- Extend `expr_cost` with stable costs for:
  - `SimdEq` and SIMD compares (mask-producing ops)
  - `SimdBlend`
  - `SimdMin*/SimdMax*` if not already covered in tests
- Add tests that enforce deterministic ordering for a small set of competing equivalent candidates.

Tests (TDD matrix)

Always-on unit tests (no toolchain required)

- Typed mask synthesis:
  - Spec: lane-wise `Eq` over slices into a `Concat`, `Bitcast` to mask type.
  - Expect: `SimdEq(a, b)` (or unsigned compare variants).
- Typed blend synthesis:
  - Spec: lane-wise `Mux(Eq(...), x_lane, y_lane)` into a `Concat`, `Bitcast` to SIMD value type.
  - Expect: `SimdBlend(mask=SimdEq(a, b), a=y, b=x)` in the canonical convention used by the interpreter.
- Typed blend via unsigned compare:
  - Spec: lane-wise `Mux(Ult(...), a_lane, b_lane)`.
  - Expect: either `SimdMinU(a, b)` or `SimdBlend(mask=SimdUlt(a, b), a=b, b=a)` depending on the configured cost model.
- Negative tests:
  - Ensure non-equivalent candidates are not accepted as equivalent (SAT checks stay SAT).
  - Ensure the superopt does not return a different type (type checker rejects).

Optional end-to-end tests (toolchain gated)

- Verilog -> Tick-IR -> infer SIMD -> superopt -> x86 compile+run vs scalar C baseline:
  - `astar_select_min_simd_slices.v` should superopt to `SimdMinU(a, b)` or to the blend form.

Implementation task list (checklist)

- Add a `TypeKey` and typed `by_size` storage in `stc/superopt.py`.
- Enumerate base terms for all relevant types:
  - variables of those types
  - constants of those types (as provided)
- Add typed operator signatures and a generator that only combines operands with compatible types.
- Add ternary operator support for `SimdBlend`.
- Add SIMD compare operators as mask-producing ops.
- Update `stc/cost.py` to include the new ops with stable costs.
- Add unit tests for typed mask and typed blend synthesis.
- Add optional Verilog end-to-end tests that confirm the new synthesis path (when Yosys and the compiler are available).

Done criteria

- Unit tests for mask and blend synthesis pass reliably with strict timeouts.
- Existing tests remain green.
- The optimizer can discover `SimdEq` masks and `SimdBlend` selects from scalar-lane specifications without any hand-written pattern-matching in the autovec pass.

