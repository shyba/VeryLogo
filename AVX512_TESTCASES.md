# AVX-512 Backend Test Matrix

This document maps the current Tick-IR SIMD ops to AVX-512 intrinsics and points to the unit tests that cover each mapping.

## Supported (Implemented + Tested)

| Tick-IR op | Lane widths | AVX-512 lowering | Tests |
| --- | --- | --- | --- |
| `SimdAnd` | 32 (16 lanes) | `_mm512_and_si512` | `tests/test_backend_x86_avx512_optional.py` |
| `SimdOr` | 32 (16 lanes) | `_mm512_or_si512` | `tests/test_backend_x86_avx512_optional.py` |
| `SimdXor` | 32 (16 lanes) | `_mm512_xor_si512` | `tests/test_backend_x86_avx512_optional.py` |
| `SimdNot` | 32 (16 lanes) | `_mm512_xor_si512(x, _mm512_set1_epi32(-1))` | `tests/test_backend_x86_avx512_optional.py` |
| `SimdAdd` | 32 (16 lanes) | `_mm512_add_epi32` | `tests/test_backend_x86_avx512_optional.py` |
| `SimdSub` | 32 (16 lanes) | `_mm512_sub_epi32` | `tests/test_backend_x86_avx512_optional.py` |
| `SimdShl` | 32, const shift | `_mm512_slli_epi32` | `tests/test_backend_x86_avx512_optional.py` |
| `SimdLShr` | 32, const shift | `_mm512_srli_epi32` | `tests/test_backend_x86_avx512_optional.py` |
| `SimdMinU/SimdMaxU` | 32 (16 lanes) | `_mm512_min_epu32/_mm512_max_epu32` | `tests/test_backend_x86_avx512_optional.py` |
| `SimdMinS/SimdMaxS` | 32 (16 lanes) | `_mm512_min_epi32/_mm512_max_epi32` | `tests/test_backend_x86_avx512_optional.py` |
| `SimdBlend` | 32 (16 lanes) | `_mm512_mask_blend_epi32` (mask from packed bits) | `tests/test_backend_x86_avx512_optional.py` |
| `SimdShuffle` | 32 (16 lanes) | `_mm512_permutexvar_epi32` | `tests/test_backend_x86_avx512_optional.py` |
| `SimdEq` (mask output) | 32 (16 lanes) | `_mm512_cmpeq_epi32_mask` | `tests/test_backend_x86_avx512_optional.py` |
| `SimdUlt` (mask output) | 32 (16 lanes) | `_mm512_cmplt_epu32_mask` | `tests/test_backend_x86_avx512_optional.py` |
| `SimdMaskPack(x)` | 32 (16 lanes) | `~_mm512_cmpeq_epi32_mask(x, 0)` | `tests/test_backend_x86_avx512_optional.py` |
| `SimdAdd` | 8/16 | `_mm512_add_epi8/16` | `tests/test_backend_x86_avx512_bw_optional.py` |
| `SimdSub` | 8/16 | `_mm512_sub_epi8/16` | `tests/test_backend_x86_avx512_bw_optional.py` |
| `SimdAddSatU/SimdSubSatU` | 8/16 | `_mm512_adds_epu8/16`, `_mm512_subs_epu8/16` | `tests/test_backend_x86_avx512_bw_optional.py` |
| `SimdAddSatS/SimdSubSatS` | 8/16 | `_mm512_adds_epi8/16`, `_mm512_subs_epi8/16` | `tests/test_backend_x86_avx512_bw_optional.py` |
| `SimdMulLo/SimdMulHiU/SimdMulHiS` | 16 (32 lanes) | `_mm512_mullo_epi16`, `_mm512_mulhi_epu16`, `_mm512_mulhi_epi16` | `tests/test_backend_x86_avx512_bw_optional.py` |
| `SimdMaddS16` | 16→32 | `_mm512_madd_epi16` | `tests/test_backend_x86_avx512_bw_optional.py` |
| `SimdAnd`/`SimdOr`/`SimdXor`/`SimdNot` | 8/16 | bitwise on `__m512i` | `tests/test_backend_x86_avx512_bw_optional.py` |
| `SimdShl`/`SimdLShr` | 16, const shift | `_mm512_slli_epi16/_mm512_srli_epi16` | `tests/test_backend_x86_avx512_bw_optional.py` |
| `SimdMinU/SimdMaxU` | 8/16 | `_mm512_min_epu8/16`, `_mm512_max_epu8/16` | `tests/test_backend_x86_avx512_bw_optional.py` |
| `SimdMinS/SimdMaxS` | 8/16 | `_mm512_min_epi8/16`, `_mm512_max_epi8/16` | `tests/test_backend_x86_avx512_bw_optional.py` |
| `SimdBlend` | 8/16 | `_mm512_mask_blend_epi8/16` (mask from packed bits) | `tests/test_backend_x86_avx512_bw_optional.py` |
| `SimdZExtLo/SimdSExtLo` | 8→16 | `_mm512_cvtepu8_epi16/_mm512_cvtepi8_epi16` | `tests/test_backend_x86_avx512_bw_optional.py` |
| `SimdEq` (mask output) | 8/16 | `_mm512_cmpeq_epi8/16_mask` | `tests/test_backend_x86_avx512_bw_optional.py` |
| `SimdSlt` (mask output) | 8/16 | `_mm512_cmplt_epi8/16_mask` | `tests/test_backend_x86_avx512_bw_optional.py` |
| `SimdUlt` (mask output) | 8/16 | `_mm512_cmplt_epu8/16_mask` | `tests/test_backend_x86_avx512_bw_optional.py` |
| `SimdMaskPack(x)` | 8/16 | `~_mm512_cmpeq_epi8/16_mask(x, 0)` | `tests/test_backend_x86_avx512_bw_optional.py` |
| `SimdZExtLo/SimdSExtLo` | 16→32 | `_mm512_cvtepu16_epi32/_mm512_cvtepi16_epi32` | `tests/test_backend_x86_avx512_bw_optional.py` |
| `SimdUnpackLo/SimdUnpackHi` | 16 (32 lanes) | scalar `u512` interleave | `tests/test_backend_x86_avx512_pack_unpack_optional.py` |
| `SimdPackSS16To8/SimdPackUS16To8` | 16→8 | scalar `u512` pack + saturate | `tests/test_backend_x86_avx512_pack_unpack_optional.py` |
| `SimdPackSS32To16` | 32→16 | scalar `u512` pack + saturate | `tests/test_backend_x86_avx512_pack_unpack_optional.py` |
| `SimdAdd`/`SimdSub`/bitwise/shifts/compares/masks | 64 (8 lanes) | `*_epi64*` forms | `tests/test_backend_x86_avx512_epi64_optional.py` |
| `SimdMinU/SimdMaxU` | 64 (8 lanes) | `_mm512_min_epu64/_mm512_max_epu64` | `tests/test_backend_x86_avx512_epi64_optional.py` |
| `SimdMinS/SimdMaxS` | 64 (8 lanes) | `_mm512_min_epi64/_mm512_max_epi64` | `tests/test_backend_x86_avx512_epi64_optional.py` |
| `SimdBlend` | 64 (8 lanes) | `_mm512_mask_blend_epi64` (mask from packed bits) | `tests/test_backend_x86_avx512_epi64_optional.py` |
| `SimdShuffle` | 64 (8 lanes) | `_mm512_permutexvar_epi64` | `tests/test_backend_x86_avx512_epi64_optional.py` |
| `SimdShuffle` | 8 (64 lanes) | `_mm512_permutexvar_epi8` (requires `avx512vbmi`) | `tests/test_backend_x86_avx512_vbmi_shuffle_optional.py` |
