# AVX / AVX2 Backend Plan (x86)

This plan extends the x86 backend support beyond SSE into AVX and AVX2. It is test-driven and keeps `./scripts/test.sh` green.

## Rules

- All changes are TDD-first: add a failing test, implement the smallest change, keep tests green.
- No code comments.
- Black formatting.
- Prefer `<immintrin.h>` intrinsics and compile-time feature flags (`-mavx`, `-mavx2`) in optional compile/run tests.
- Treat Tick-IR SIMD values as fixed-width packed bitvectors with per-lane modular arithmetic unless an op is explicitly saturating.

## AVX (AVX1) Scope

AVX1 primarily adds 256-bit floating-point vectors and VEX-encoded forms of many SSE instructions.

- [x] Add Tick-IR `f32/f64` types with IEEE-754 semantics.
- [x] Add Tick-IR float ops needed for a minimal AVX1 backend (`fadd/fmul/fcmp`, casts).
- [x] Add Z3 encoding for float ops (use Z3 FP theory; define rounding mode policy).
- [x] Add interpreter semantics for float ops.
- [x] Add a minimal AVX1 backend for `__m256/__m256d`.

Non-goals until float semantics exist:

- AVX1 as “faster SSE2 integers” is not a target: integer semantics remain covered by SSE2/SSE4.1 and AVX2.

## AVX (Float) Expansion (Post-Minimal)

This section covers AVX work that goes beyond the minimal float backend and aims for broader float instruction coverage with stable semantics.

### Phase A — Float Semantics Hardening

- [x] Decide and document NaN handling policy for float ops:
  - Current policy: float ops canonicalize NaN payloads in interpreter, Z3 encoding, and x86 backends; tests include NaN payload cases.
- [x] If canonicalization is chosen: implement it consistently in interpreter, Z3 encoding, and x86 backends.

### Phase B — Scalar Float Ops

- [x] Add Tick-IR scalar float ops: `FSub`, `FDiv`, `FSqrt`, `FNeg`, `FAbs`.
- [x] Add interpreter semantics for those ops.
- [x] Add Z3 FP encoding for those ops (fixed rounding mode policy).
- [x] Add unit tests that check Z3 consistency on concrete samples (include NaN payload cases once canonicalization is implemented).

### Phase C — SIMD Float Ops (256-bit)

- [x] Add Tick-IR SIMD float ops: `SimdFSub`, `SimdFDiv`, `SimdFSqrt`, `SimdFNeg`, `SimdFAbs`.
- [x] Add AVX lowering:
  - `_mm256_sub_ps/pd`, `_mm256_div_ps/pd`, `_mm256_sqrt_ps/pd`
  - `FNeg/FAbs` via bitwise sign manipulation using `_mm256_xor_ps/pd` / `_mm256_and_ps/pd` with constant masks.
- [x] Add optional compile/run tests (skip unless `cc` + `x86_64` + `avx`) verifying backend results match the interpreter bit patterns.

### Phase D — Comparisons and Blends

- [x] Add Tick-IR float comparisons beyond `{eq, lt}`:
  - scalar: `FLe`, `FNe`
  - SIMD: `SimdFCmpLe`, `SimdFCmpNe`
- [x] Add AVX lowering using `_mm256_cmp_ps/pd` with explicit predicates for ordered comparisons.
- [x] Add `SimdBlend` lowering for float vectors:
  - synthesize a per-lane all-ones mask vector from packed mask bits
  - use `_mm256_blendv_ps/pd` or `and/andnot/or` forms
- [x] Add tests covering signed zero behavior and a small set of edge cases.

### Phase E — FMA (Optional)

- [x] Add Tick-IR `SimdFFma` gated by `fma` CPU flag and `-mfma`.
- [x] Add Z3 encoding as `fpFMA` (with the chosen rounding mode).
- [x] Add optional compile/run tests that compare backend vs interpreter for representative vectors.

## AVX2 Scope (Integer 256-bit)

AVX2 is the primary target for 256-bit integer SIMD (`__m256i`). The existing AVX2 backend covers core ops; this plan focuses on correctness gaps (especially cross-128-lane semantics) and increasing Tick-IR instruction coverage.

### Phase 1 — Semantics and ABI

- [x] Document the AVX2 ABI and mask conventions in `X86_SIMD_ABI.md` (what is returned for vector vs mask results, calling convention for generated helpers).
- [x] Add an explicit “256-bit lane-group boundary” section to the Tick-IR SIMD docs (AVX2 `vpshufb` and some ops operate independently on 128-bit lanes; Tick-IR ops must define cross-lane behavior explicitly).
- [x] Add a dedicated AVX2 test vector document (`AVX2_TESTCASES.md`) mirroring `SSE2_TESTCASES.md`.

### Phase 2 — Pack/Unpack (Cross-Lane Correctness)

Tick-IR `SimdUnpackLo/Hi` and `SimdPack*` define full-vector behavior; AVX2 needs permutes to match that when the semantics cross 128-bit lanes.

- [x] Add failing interpreter-vs-backend tests for `SimdUnpackLo` / `SimdUnpackHi` at 256-bit widths (u8/u16/u32).
- [x] Implement AVX2 lowering for `SimdUnpackLo` / `SimdUnpackHi` using the appropriate `unpack*` intrinsics plus `permute2x128` when needed.
- [x] Add failing tests for `SimdPackSS16To8`, `SimdPackUS16To8`, `SimdPackSS32To16` at 256-bit widths.
- [x] Implement AVX2 lowering for `SimdPack*` using pack intrinsics plus `permute2x128` when needed.

### Phase 3 — Shuffle and Permute

- [x] Add failing tests for `SimdShuffle` on `simd[8,32]` at 256-bit widths.
- [x] Implement AVX2 lowering for `SimdShuffle`:
  - `vpshufb` for byte-level shuffles using `permute2x128` + `_mm256_shuffle_epi8` and OR.
  - `_mm256_permutevar8x32_epi32` for 32-bit lane shuffles.
- [x] Add a reducer rule that canonicalizes common shuffle patterns into `SimdUnpack*` / `SimdPack*` / blend ops when that makes backend support simpler.

### Phase 4 — Widen/Narrow and Blend at 256-bit

- [x] Add failing tests for `SimdZExtLo` / `SimdSExtLo` for 256-bit outputs (e.g., u8→u16, s8→s16, u16→u32, s16→s32).
- [x] Implement AVX2 lowering using `_mm256_cvtepu8_epi16`, `_mm256_cvtepi8_epi16`, `_mm256_cvtepu16_epi32`, `_mm256_cvtepi16_epi32`, etc.
- [x] Add failing tests for `SimdBlend` at 256-bit widths using packed-bit masks.
- [x] Implement AVX2 lowering for `SimdBlend` via `_mm256_blendv_epi8` with a mask vector derived from packed mask bits.

### Phase 5 — Mask Handling (Ergonomics)

- [x] Add tests that exercise `SimdMaskPack` on 256-bit values across lane widths (8/16/32) and verify packed-bit layout.
- [x] Add support for “mask-only Tick-IR programs” (no vector outputs) in the AVX2 compile/run tests, ensuring the ABI returns masks consistently.
- [x] Add a normalization pass that rewrites `SimdMaskExpand(SimdMaskPack(x))` patterns into stable canonical forms (so Z3 and codegen see the same structure).

## Optional: Integration and Tooling

- [x] Add a `scripts/emit_x86.py` CLI that emits C for a given `tick_ir.bin` and runs the optional compile/run tests for the selected backend (SSE2/AVX/AVX2/AVX-512).
- [x] Add a `scripts/minimize_counterexample.py` helper that shrinks failing SIMD test vectors using the reducer (useful for AVX2 cross-lane bugs).
