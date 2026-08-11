# AVX-512 Backend Plan (x86)

This plan extends the x86 backend support to AVX-512 for both integer (`__m512i`) and floating-point (`__m512/__m512d`) vectors. It is test-driven and keeps `./scripts/test.sh` green.

## Rules

- All changes are TDD-first: add a failing test, implement the smallest change, keep tests green.
- No code comments.
- Black formatting.
- Prefer `<immintrin.h>` intrinsics and compile-time feature flags in optional compile/run tests.
- Feature-gate instruction families by CPU flags and refuse codegen when a required flag is missing.

## Feature Gates

AVX-512 is a family; the backend must be precise about which sub-features are required:

- `avx512f`: baseline for `__m512i`, `__m512`, `__m512d`, masks, and core ops.
- `avx512bw`: 8-bit/16-bit integer lanes (`epi8/epi16`) and byte shuffles in 512-bit registers.
- `avx512dq`: additional integer ops for 32/64-bit lanes and some conversions.
- `avx512vl`: 128/256-bit AVX-512 forms (optional; not required for a pure 512-bit backend).
- `avx512vbmi`: more powerful byte permutes (optional, for advanced `SimdShuffle`).

## Phase 1 — Backend Structure and Auto-Selection

- [x] Decide backend organization:
  - Option A: extend `stc/backend_x86_avx512.py` to support both integer and float ops.
  - Option B: keep `stc/backend_x86_avx512.py` as the integer backend and add `stc/backend_x86_avx512_float.py` for floats.
- [x] Extend `stc/backend_x86_auto.py` so width=512 selects:
  - AVX-512 float backend when Tick-IR contains SIMD float ops (`SimdF*`).
  - AVX-512 integer backend otherwise.
- [x] Add a single “required flags” pre-scan for width=512 so failure is a clear `AutoBackendError`.

## Phase 2 — AVX-512 Float (512-bit)

Target types:

- `simd[32,16]` (`__m512`)
- `simd[64,8]` (`__m512d`)

### Core Ops

- [x] Add optional compile/run tests for `SimdFAdd`, `SimdFMul`, `SimdFCmpEq`, `SimdFCmpLt` at 512-bit width (skip unless `cc` + `x86_64` + `avx512f`).
- [x] Implement codegen lowering:
  - `_mm512_add_ps/pd`, `_mm512_mul_ps/pd`
  - `_mm512_cmp_ps_mask/_mm512_cmp_pd_mask` for mask results
  - Return packed masks as `uint64_t` in `out[0]`.

### AVX Float-Expansion Compatibility

- [x] Keep float semantics consistent across widths (256-bit AVX and 512-bit AVX-512):
  - same rounding mode policy (`RNE`)
  - same comparison predicate policy (ordered comparisons for existing ops)

## Phase 3 — AVX-512 Integer (512-bit)

### Broaden Lane Width Coverage

- [x] Add tests for `simd[64,8]` integer add/sub/bitwise/shifts/compares (skip unless `avx512f`).
- [x] Add tests for `simd[16,32]` and `simd[8,64]` integer add/sub/bitwise/compares (skip unless `avx512bw`).
- [x] Implement lowering for `epi8/epi16` families guarded by `avx512bw`.

### Multiply, Saturating, Pack/Unpack

- [x] Add failing tests (interpreter-vs-backend) for:
  - saturating add/sub (`SimdAddSat*`, `SimdSubSat*`) on `simd[8,64]` and `simd[16,32]` (requires `avx512bw`)
  - mul and madd (`SimdMul*`, `SimdMaddS16`) for `simd[16,32]` (requires `avx512bw`)
  - pack/unpack (`SimdUnpack*`, `SimdPack*`) for 512-bit (requires `avx512bw` for 8/16)
- [x] Implement AVX-512 lowering (intrinsics where available; scalar fallback for pack/unpack to preserve full-vector semantics).

### Shuffle and Permute

- [x] Define supported `SimdShuffle` cases for 512-bit:
  - `simd[32,16]` via `_mm512_permutexvar_epi32` (requires `avx512f`)
  - `simd[64,8]` via `_mm512_permutexvar_epi64` (requires `avx512f`)
  - `simd[8,64]` via `_mm512_permutexvar_epi8` when `avx512vbmi` is available; otherwise provide a restricted subset or refuse.
- [x] Add tests gated by the required flags.
- [x] Ensure `SimdShuffle` remains full-vector semantic (no hidden lane-group boundaries).

### Min/Max, Blend, Extend

- [x] Add tests and implement `SimdMin*` / `SimdMax*` for:
  - 8/16 lanes (requires `avx512bw`)
  - 32/64 lanes (requires `avx512f`)
- [x] Add tests and implement `SimdBlend` for 512-bit integer vectors using mask synthesis from packed mask bits.
- [x] Add tests and implement `SimdZExtLo` / `SimdSExtLo` for 512-bit outputs where supported by AVX-512.

## Phase 4 — Mask Handling and ABI

- [x] Expand `SimdMaskPack` tests to cover 512-bit across lane widths {8,16,32,64} (gated by required flags).
- [x] Add a doc `AVX512_TESTCASES.md` mirroring `SSE2_TESTCASES.md` / `AVX2_TESTCASES.md`.
- [x] Update `X86_SIMD_ABI.md` with an AVX-512 section clarifying:
  - when mask results are `__mmask*` directly
  - the packed-bit layout and lane order

## Phase 5 — AVX-512 Optional Extensions

- [x] Add optional support for masked arithmetic (AVX-512 predication) when Tick-IR grows a first-class mask-on-op representation.
- [x] Add `avx512vl` usage only if a 128/256-bit AVX-512 backend variant is introduced.
