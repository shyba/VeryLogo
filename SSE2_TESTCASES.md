# SSE2 Backend Test Matrix

This document maps the current Tick-IR SIMD ops to SSE2 intrinsics and points to the unit tests that cover each mapping.

## Supported (Implemented + Tested)

| Tick-IR op | Lane widths | SSE2 lowering | Tests |
| --- | --- | --- | --- |
| `SimdAnd` | any (128-bit total) | `_mm_and_si128` | `tests/test_backend_x86_sse2_optional.py` |
| `SimdOr` | any (128-bit total) | `_mm_or_si128` | `tests/test_backend_x86_sse2_optional.py` |
| `SimdXor` | any (128-bit total) | `_mm_xor_si128` | `tests/test_backend_x86_sse2_optional.py` |
| `SimdNot` | any (128-bit total) | `_mm_xor_si128(x, _mm_set1_epi32(-1))` | `tests/test_backend_x86_sse2_optional.py` |
| `SimdAnd(SimdNot(a), b)` | any (128-bit total) | `_mm_andnot_si128(a, b)` | `tests/test_backend_x86_sse2_optional.py` |
| `SimdAdd` | 8/16/32/64 | `_mm_add_epi8/16/32/64` | `tests/test_backend_x86_sse2_lane_widths_optional.py` |
| `SimdSub` | 8/16/32/64 | `_mm_sub_epi8/16/32/64` | `tests/test_backend_x86_sse2_lane_widths_optional.py` |
| `SimdAddSatU` | 8/16 | `_mm_adds_epu8/16` | `tests/test_backend_x86_sse2_lane_widths_optional.py` |
| `SimdSubSatU` | 8/16 | `_mm_subs_epu8/16` | `tests/test_backend_x86_sse2_lane_widths_optional.py` |
| `SimdAddSatS` | 8/16 | `_mm_adds_epi8/16` | `tests/test_backend_x86_sse2_lane_widths_optional.py` |
| `SimdSubSatS` | 8/16 | `_mm_subs_epi8/16` | `tests/test_backend_x86_sse2_lane_widths_optional.py` |
| `SimdShl` | 16/32/64, const shift | `_mm_slli_epi16/32/64` | `tests/test_backend_x86_sse2_lane_widths_optional.py` |
| `SimdLShr` | 16/32/64, const shift | `_mm_srli_epi16/32/64` | `tests/test_backend_x86_sse2_lane_widths_optional.py` |
| `SimdAShr` | 16/32, const shift | `_mm_srai_epi16/32` | `tests/test_backend_x86_sse2_lane_widths_optional.py` |
| `SimdEq` (mask output) | 8/16/32 | `pcmpeq*` + `_mm_movemask_epi8` packing | `tests/test_backend_x86_sse2_cmp_optional.py` |
| `SimdSlt` (mask output) | 8/16/32 | `pcmpgt*` swapped + `_mm_movemask_epi8` packing | `tests/test_backend_x86_sse2_cmp_optional.py` |
| `SimdUlt` (mask output) | 8/16/32 | XOR-bias + `pcmpgt*` + `_mm_movemask_epi8` packing | `tests/test_backend_x86_sse2_cmp_optional.py` |
| `SimdMulLo` | 16 | `_mm_mullo_epi16` | `tests/test_backend_x86_sse2_lane_widths_optional.py` |
| `SimdMulHiU` | 16 | `_mm_mulhi_epu16` | `tests/test_backend_x86_sse2_lane_widths_optional.py` |
| `SimdMulHiS` | 16 | `_mm_mulhi_epi16` | `tests/test_backend_x86_sse2_lane_widths_optional.py` |
| `SimdMaddS16` | 16→32 | `_mm_madd_epi16` | `tests/test_backend_x86_sse2_lane_widths_optional.py` |
| `SimdUnpackLo` | 8/16/32/64 | `_mm_unpacklo_epi*` | `tests/test_backend_x86_sse2_lane_widths_optional.py` |
| `SimdUnpackHi` | 8/16/32/64 | `_mm_unpackhi_epi*` | `tests/test_backend_x86_sse2_lane_widths_optional.py` |
| `SimdPackSS16To8` | 16→8 | `_mm_packs_epi16` | `tests/test_backend_x86_sse2_lane_widths_optional.py` |
| `SimdPackUS16To8` | 16→8 | `_mm_packus_epi16` | `tests/test_backend_x86_sse2_lane_widths_optional.py` |
| `SimdPackSS32To16` | 32→16 | `_mm_packs_epi32` | `tests/test_backend_x86_sse2_lane_widths_optional.py` |
| `SimdMinU` | 8/16/32 | `_mm_min_epu8/16/32` | `tests/test_backend_x86_sse41_optional.py` |
| `SimdMaxU` | 8/16/32 | `_mm_max_epu8/16/32` | `tests/test_backend_x86_sse41_optional.py` |
| `SimdMinS` | 8/16/32 | `_mm_min_epi8/16/32` | `tests/test_backend_x86_sse41_optional.py` |
| `SimdMaxS` | 8/16/32 | `_mm_max_epi8/16/32` | `tests/test_backend_x86_sse41_optional.py` |
| `SimdZExtLo` | 8→16,16→32,32→64 | `_mm_cvtepu8_epi16/_mm_cvtepu16_epi32/_mm_cvtepu32_epi64` | `tests/test_backend_x86_sse41_optional.py` |
| `SimdSExtLo` | 8→16,16→32,32→64 | `_mm_cvtepi8_epi16/_mm_cvtepi16_epi32/_mm_cvtepi32_epi64` | `tests/test_backend_x86_sse41_optional.py` |
| `SimdBlend` | 8/16/32/64 | `_mm_blendv_epi8` (mask from compare packed bits) | `tests/test_backend_x86_sse41_optional.py` |
| `SimdShuffle` | 8 (16 lanes) | `_mm_shuffle_epi8` (SSSE3; indices `0..15`) | `tests/test_backend_x86_ssse3_shuffle_optional.py` |
