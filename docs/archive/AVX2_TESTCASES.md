# AVX2 Backend Test Matrix

This document maps the current Tick-IR SIMD ops to AVX2 intrinsics and points to the unit tests that cover each mapping.

## Supported (Implemented + Tested)

| Tick-IR op | Lane widths | AVX2 lowering | Tests |
| --- | --- | --- | --- |
| `SimdAnd` | any (256-bit total) | `_mm256_and_si256` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdOr` | any (256-bit total) | `_mm256_or_si256` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdXor` | any (256-bit total) | `_mm256_xor_si256` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdNot` | any (256-bit total) | `_mm256_xor_si256(x, _mm256_set1_epi32(-1))` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdAnd(SimdNot(a), b)` | any (256-bit total) | `_mm256_andnot_si256(a, b)` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdAdd` | 8/16/32/64 | `_mm256_add_epi8/16/32/64` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdSub` | 8/16/32/64 | `_mm256_sub_epi8/16/32/64` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdAddSatU` | 8/16 | `_mm256_adds_epu8/16` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdSubSatU` | 8/16 | `_mm256_subs_epu8/16` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdAddSatS` | 8/16 | `_mm256_adds_epi8/16` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdSubSatS` | 8/16 | `_mm256_subs_epi8/16` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdShl` | 16/32/64, const shift | `_mm256_slli_epi16/32/64` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdLShr` | 16/32/64, const shift | `_mm256_srli_epi16/32/64` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdAShr` | 16/32, const shift | `_mm256_srai_epi16/32` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdEq` (mask output) | 8/16/32 | `vpcmpeq*` + `_mm256_movemask_epi8` packing | `tests/test_backend_x86_avx2_optional.py` |
| `SimdSlt` (mask output) | 8/16/32 | `vpcmpgt*` swapped + `_mm256_movemask_epi8` packing | `tests/test_backend_x86_avx2_optional.py` |
| `SimdUlt` (mask output) | 8/16/32 | XOR-bias + `vpcmpgt*` + `_mm256_movemask_epi8` packing | `tests/test_backend_x86_avx2_optional.py` |
| `SimdMaskPack` (mask output) | 8/16/32 | `cmpeq(x,0)` + invert + `_mm256_movemask_epi8` packing | `tests/test_backend_x86_avx2_optional.py` |
| `SimdMulLo` | 16 | `_mm256_mullo_epi16` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdMulHiU` | 16 | `_mm256_mulhi_epu16` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdMulHiS` | 16 | `_mm256_mulhi_epi16` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdMaddS16` | 16→32 | `_mm256_madd_epi16` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdUnpackLo` | 8/16/32/64 | split to `__m128i` low/high + `_mm_unpacklo/hi_epi*` + `inserti128` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdUnpackHi` | 8/16/32/64 | split to `__m128i` low/high + `_mm_unpacklo/hi_epi*` + `inserti128` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdPackSS16To8` | 16→8 | pack each input separately via `_mm_packs_epi16(lo,hi)` + `inserti128` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdPackUS16To8` | 16→8 | pack each input separately via `_mm_packus_epi16(lo,hi)` + `inserti128` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdPackSS32To16` | 32→16 | pack each input separately via `_mm_packs_epi32(lo,hi)` + `inserti128` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdShuffle` | 8 (32 lanes) | `permute2x128` (dup lo/hi) + `vpshufb` per half + OR | `tests/test_backend_x86_avx2_optional.py` |
| `SimdShuffle` | 32 (8 lanes) | `_mm256_permutevar8x32_epi32` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdZExtLo` | 8→16,16→32,32→64 | `_mm256_cvtepu8/16/32_*` from low `__m128i` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdSExtLo` | 8→16,16→32,32→64 | `_mm256_cvtepi8/16/32_*` from low `__m128i` | `tests/test_backend_x86_avx2_optional.py` |
| `SimdBlend` | 8/16/32/64 | `_mm256_blendv_epi8` (mask vector synthesized from packed bits) | `tests/test_backend_x86_avx2_optional.py` |

## Not Yet Supported

These are not implemented in `stc/backend_x86_avx2.py` yet.

| Area | Notes |
| --- | --- |
| `SimdMin*` / `SimdMax*` at 256-bit | AVX2 has integer min/max intrinsics; Tick-IR ops exist but the AVX2 backend does not lower them yet. |
| General `SimdShuffle` | The AVX2 backend supports only `simd[8,32] → simd[8,32]` and `simd[32,8] → simd[32,8]`. |

