# Refactor Plan: Fix Regressions + Make Changes Reviewable

This plan addresses current regressions (SIMD/autovec + CUDA PTX) and cleans up the working set so changes are easy to review.

## Goals

- `python -m unittest discover -s tests` passes (optional tests skip cleanly when deps are missing).
- Keccak benchmark remains reproducible and not slower than baseline.
- Changes are deterministic and easy to troubleshoot.

## 1) Triage and lock repro

Capture failing list + stack traces and create minimal repro entrypoints:

- SIMD/autovec failures:
  - Run the failing fixtures through `infer_simd_types()` / `optimize_tick_ir()` and print:
    - input/output/state types before/after
    - key output expression kind(s)
- CUDA PTX failure:
  - Save the generated PTX to an `out/` path on failure and print the failing PTX line with context.

## 2) Fix SIMD/autovec regression (types + node rewriting)

Symptoms observed:
- Tests expect `BitVecType(width=128)` to become `SimdType(lane_width=16, lanes=8)` etc.
- Tests expect arithmetic nodes like `SimdAdd`, but currently see patterns like:
  - `Bitcast(SimdType, Concat(parts=(Add(Slice(...)), ...)))`
- SSE2 backend tests error because `emit_x86_sse2_c()` requires actual SIMD types.

Work:

1) Define canonical SIMD-form invariants
   - After autovec/infer-simd, prefer `Simd*` nodes (`SimdAdd`, `SimdSub`, `Simd*Cmp`, etc.)
     over “bitcast + concat of lane ops”.
   - Ensure both `TickIR.inputs/outputs/state` *and* expressions are rewritten.

2) Add/restore a rewrite pass
   - Implement or repair a pass in `stc/infer_simd.py` (or a post-pass in `stc/reduce.py`) that recognizes:
     - `Bitcast(SimdType, Concat(parts=(lane_op(Slice(...)), ...)))`
     - where the concat corresponds to lanes and each lane op matches.
   - Rewrite to the corresponding `Simd*` node.

3) Keep emitter expectations consistent
   - SSE2 backend requires SIMD types: ensure tests that call `emit_x86_sse2_c()` only do so
     after SIMD inference succeeded.
   - If a design cannot be autovectorized, optional tests should skip instead of failing.

4) Confirm type inference consistency
   - Verify `stc/interp.py::infer_type()` returns `SimdType` for `Simd*` nodes and
     maintains bitcast invariants.

5) Add a direct unit test for the rewrite
   - A small TickIR that matches the “concat of lane ops” pattern should rewrite into `SimdAdd`
     (and similarly for other ops covered by the failing tests).

Acceptance:
- All SIMD/autovec optional tests pass on machines with the required toolchain/CPU features.
- No SSE2 backend errors due to missing SIMD types.

## 3) Fix CUDA PTX regression (ptxas parse error)

Symptom observed:
- `ptxas ... fatal: Parsing error near '-'`

Work:

1) Reproduce exact PTX
   - Ensure the failing path saves PTX to `out/ptx_debug.ptx` on error.

2) Inspect the failing line (common causes)
   - Negative immediates emitted in PTX-illegal form
   - Invalid symbol/register names containing `-`
   - `.version`/`.target` mismatch for installed `ptxas`
   - Predicate/inline formatting issues

3) Fix PTX generation/assembly
   - Likely touchpoints:
     - `stc/ptx_lop3.py` (PTX generation + assembly)
     - `stc/cuda_driver.py` (tool invocation / flags)
   - Emit immediates in PTX-legal forms (prefer hex masks over negative literals).
   - Use conservative `.version` / `.target` compatible with `sm_61` and installed toolchain.

4) Make optional test robust
   - Skip if `ptxas` is missing/unsupported or GPU not available (with clear reason).

Acceptance:
- `tests/test_cuda_lop3_optional.py` passes on machines with CUDA toolchain + compatible GPU;
  otherwise skips cleanly.

## 4) Repository hygiene (reduce review noise)

1) Separate intentional source changes from local artifacts
   - Keep: `stc/*`, `tests/*`, `scripts/*`, `docs/*` that are referenced and used.
   - Avoid tracking scratch files and large binary artifacts.

2) Add/extend `.gitignore` for common build outputs
   - `out/`, `*.so`, `*.cubin`, `*.ptx`, and local temp outputs used by scripts.
   - Treat `external-*` checkouts as inputs unless explicitly meant to be tracked.

Acceptance:
- `git status` only includes intentional source changes.

## 5) Validation runs

Run in increasing scope:

1) Targeted
   - Only previously failing SIMD/autovec optional tests.
   - CUDA optional test (or confirm skip).

2) Full suite
   - `python -m unittest discover -s tests -q`

3) Bench sanity
   - Keccak AVX-512:
     - `PYTHONPATH=. .venv/bin/python scripts/bench_keccak_steps_avx512.py --msg abc --reps 20 --idle-chunk 64`
   - CUDA LOP3 bench (if CUDA available):
     - `PYTHONPATH=. .venv/bin/python scripts/bench_lop3_cuda.py ...` minimal settings

