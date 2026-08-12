"""Property-based tests for the hand-scheduled AVX-512 GEMM kernels.

Targets:
  - stc.gemm_asm 8x32 asm kernels (vnni8: uint8 A x int8 B; bf16: A/B bf16)
    against naive references over random shapes/data (hypothesis-driven
    seeds; the C harness regenerates the data from the seed, so hypothesis
    shrinks to (shape, seed)).
  - The pack functions (pack_b_i8_32, pack_b_bf16_32, pack_b_bf16_32_T):
    pack-then-GEMM == naive, and gemm(A, pack_T(B)) == A@B^T.
  - stc.aggen predictions (positive, monotone in work size) and the known
    latent bug: best_tile can return mr=12 which emit_gemm_kernel rejects
    (expected failure).
  - The benchmark's multi-NR shared-Bp bug: packing once with one NR while
    kernels read their own NR layout gives garbage (expected failure).

Run:  .venv/bin/python -m pytest tests/test_properties_gemm.py -q
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from stc.aggen import get_machine
from stc.gemm_asm import emit_gemm_kernel

from property_harness import (
    EDGE_BF16,
    EDGE_VNNI,
    K_BF16,
    K_VNNI,
    MUL32,
    MUL8,
    SEED,
    build_binary,
    run_check,
)

I8_CHECKER_C = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern void gemm_vnni8_8x32_asm(const uint8_t* A, const int32_t* Bp, int32_t* C, int K, int N);
extern void gemm_vnni8_8x16_asm(const uint8_t* A, const int32_t* Bp, int32_t* C, int K, int N);
static uint64_t st;
static uint64_t rng(void) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; }
static void pack_b_i8_32(const int8_t* B, int32_t* Bp, int K, int N) {
    for (int n0 = 0; n0 < N; n0 += 32)
        for (int c = 0; c < K/4; c++)
            for (int j = 0; j < 32; j++) {
                uint32_t d = 0; int col = n0 + j;
                for (int t = 0; t < 4; t++)
                    d |= (uint32_t)(uint8_t)B[(4*c + t)*N + col] << (8*t);
                Bp[((n0/32)*(K/4) + c)*32 + j] = d;
            }
}
static void gemm_i8_32(const uint8_t* A, const int32_t* Bp, int32_t* C, int M, int N, int K) {
    for (int n0 = 0; n0 < N; n0 += 32)
        for (int m0 = 0; m0 < M; m0 += 8)
            gemm_vnni8_8x32_asm(A + m0*K, Bp + (n0/32)*(K/4)*32, C + m0*N + n0, K, N);
}
static void gemm_i8_16(const uint8_t* A, const int32_t* Bp, int32_t* C, int M, int N, int K) {
    for (int n0 = 0; n0 < N; n0 += 16)
        for (int m0 = 0; m0 < M; m0 += 8)
            gemm_vnni8_8x16_asm(A + m0*K, Bp + (n0/16)*(K/4)*16, C + m0*N + n0, K, N);
}
static uint8_t A[8192]; static int8_t B[16384]; static int32_t Bp[16384];
static int32_t Cv[16384], Cr[16384];
int main(int argc, char** argv) {
    int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]);
    st = strtoull(argv[4], NULL, 0); if (st == 0) st = 0x9E3779B97F4A7C15ULL;
    int mode = argc > 5 ? atoi(argv[5]) : 0;
    for (int i = 0; i < M*K; i++) A[i] = (uint8_t)(rng() & 0xFF);   /* 0..255 unsigned */
    for (int i = 0; i < K*N; i++) B[i] = (int8_t)(rng() & 0xFF);    /* -128..127 signed */
    pack_b_i8_32(B, Bp, K, N);
    if (mode == 1)
        gemm_i8_16(A, Bp, Cv, M, N, K);   /* 8x16 kernel on a 32-packed Bp */
    else
        gemm_i8_32(A, Bp, Cv, M, N, K);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            long long s = 0;
            for (int k = 0; k < K; k++) s += (long long)A[i*K+k] * (long long)B[k*N+j];
            Cr[i*N+j] = (int32_t)s;
            if (Cv[i*N+j] != Cr[i*N+j]) {
                printf("FAIL %d %d %lld %lld\n", i, j, (long long)Cv[i*N+j], (long long)Cr[i*N+j]);
                return 0;
            }
        }
    printf("OK\n");
    return 0;
}
"""

BF16_CHECKER_C = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
extern void gemm_bf16_8x32_asm(const uint16_t* A, const int32_t* Bp, float* C, int K, int N);
static uint64_t st;
static uint64_t rng(void) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; }
static uint16_t f2b(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16;
    return (uint16_t)r;
}
static float b2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
/* mixed normal/denormal bf16 generation */
static uint16_t gen_bf16(void) {
    uint64_t r = rng();
    if ((r & 0xF) == 0) {                       /* ~6% denormals: exp=0, random mantissa */
        uint16_t m = (uint16_t)(r >> 16) & 0x7F;
        uint16_t s = (uint16_t)((r >> 8) & 0x8000);
        return (uint16_t)(s | m);
    }
    float f = ((float)(r & 0xFFFFFF) / 16777216.0f) * 2.0f - 1.0f;
    return f2b(f);
}
static void pack_b_bf16_32(const uint16_t* B, int32_t* Bp, int K, int N) {
    for (int n0 = 0; n0 < N; n0 += 32)
        for (int c = 0; c < K/2; c++)
            for (int j = 0; j < 32; j++) {
                uint32_t v = 0; int col = n0 + j;
                for (int t = 0; t < 2; t++)
                    v |= (uint32_t)B[(2*c + t)*N + col] << (16*t);
                Bp[((n0/32)*(K/2) + c)*32 + j] = v;
            }
}
static void pack_b_bf16_32_T(const uint16_t* B, int32_t* Bp, int K, int N) {
    /* B is N x K row-major; packs B^T (K x N) */
    for (int n0 = 0; n0 < N; n0 += 32)
        for (int c = 0; c < K/2; c++)
            for (int j = 0; j < 32; j++) {
                uint32_t v = 0; int col = n0 + j;
                for (int t = 0; t < 2; t++)
                    v |= (uint32_t)B[col*K + (2*c + t)] << (16*t);
                Bp[((n0/32)*(K/2) + c)*32 + j] = v;
            }
}
static void gemm_bf16(const uint16_t* A, const int32_t* Bp, float* C, int M, int N, int K) {
    for (int n0 = 0; n0 < N; n0 += 32)
        for (int m0 = 0; m0 < M; m0 += 8)
            gemm_bf16_8x32_asm(A + m0*K, Bp + (n0/32)*(K/2)*32, C + m0*N + n0, K, N);
}
static uint16_t A[8192]; static uint16_t B[16384]; static int32_t Bp[16384];
static float Cv[16384], Cr[16384];
int main(int argc, char** argv) {
    int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]);
    st = strtoull(argv[4], NULL, 0); if (st == 0) st = 0x9E3779B97F4A7C15ULL;
    int mode = argc > 5 ? atoi(argv[5]) : 0;   /* 0 = A@B, 1 = A@B^T via pack_T */
    for (int i = 0; i < M*K; i++) A[i] = gen_bf16();
    for (int i = 0; i < K*N; i++) B[i] = gen_bf16();
    if (mode == 1)
        pack_b_bf16_32_T(B, Bp, K, N);
    else
        pack_b_bf16_32(B, Bp, K, N);
    gemm_bf16(A, Bp, Cv, M, N, K);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            if (mode == 1)
                for (int k = 0; k < K; k++) s += b2f(A[i*K+k]) * b2f(B[j*K+k]);
            else
                for (int k = 0; k < K; k++) s += b2f(A[i*K+k]) * b2f(B[k*N+j]);
            Cr[i*N+j] = s;
            if (fabsf(Cv[i*N+j] - s) > 1e-3f * fmaxf(1.0f, fabsf(s))) {
                printf("FAIL %d %d %.6f %.6f\n", i, j, Cv[i*N+j], s);
                return 0;
            }
        }
    printf("OK\n");
    return 0;
}
"""


_BINS: dict[str, str] = {}


def _require_cpu():
    from property_harness import cpu_has_avx512_vnni

    if not cpu_has_avx512_vnni():
        pytest.skip("requires AVX512-VNNI + AVX512-BF16 (Zen 5)")


def _i8_bin() -> str:
    _require_cpu()
    if "i8" not in _BINS:
        try:
            _BINS["i8"] = build_binary(
                "gemm_i8",
                I8_CHECKER_C,
                {
                    "gemm_vnni8_8x32_asm": ("vnni8", 8, 32),
                    "gemm_vnni8_8x16_asm": ("vnni8", 8, 16),
                },
            )
        except Exception as exc:  # pragma: no cover - env without gcc/avx512
            pytest.skip(f"i8 kernel build failed: {exc}")
    return _BINS["i8"]


def _bf16_bin() -> str:
    _require_cpu()
    if "bf16" not in _BINS:
        try:
            _BINS["bf16"] = build_binary(
                "gemm_bf16", BF16_CHECKER_C, {"gemm_bf16_8x32_asm": ("bf16", 8, 32)}
            )
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"bf16 kernel build failed: {exc}")
    return _BINS["bf16"]


# ---------------------------------------------------------------------------
# GEMM correctness over random shapes/data
# ---------------------------------------------------------------------------


@settings(max_examples=40, deadline=None)
@given(MUL8, MUL32, K_VNNI, SEED)
def test_vnni8_random_shapes(m, n, k, seed):
    out = run_check(_i8_bin(), [m, n, k, seed, 0])
    assert out == "OK", f"vnni8 {m}x{n}x{k} seed={seed}: {out}"


@settings(max_examples=20, deadline=None)
@given(EDGE_VNNI, SEED)
def test_vnni8_edge_shapes(shape, seed):
    m, n, k = shape
    out = run_check(_i8_bin(), [m, n, k, seed, 0])
    assert out == "OK", f"vnni8 edge {m}x{n}x{k} seed={seed}: {out}"


@settings(max_examples=40, deadline=None)
@given(MUL8, MUL32, K_BF16, SEED)
def test_bf16_random_shapes(m, n, k, seed):
    out = run_check(_bf16_bin(), [m, n, k, seed, 0])
    assert out == "OK", f"bf16 {m}x{n}x{k} seed={seed}: {out}"


@settings(max_examples=20, deadline=None)
@given(EDGE_BF16, SEED)
def test_bf16_edge_shapes(shape, seed):
    m, n, k = shape
    out = run_check(_bf16_bin(), [m, n, k, seed, 0])
    assert out == "OK", f"bf16 edge {m}x{n}x{k} seed={seed}: {out}"


# ---------------------------------------------------------------------------
# Transpose pack property: gemm(A, pack_T(B)) == A @ B^T
# ---------------------------------------------------------------------------


@settings(max_examples=30, deadline=None)
@given(MUL8, MUL32, K_BF16, SEED)
def test_bf16_transpose_pack(m, n, k, seed):
    out = run_check(_bf16_bin(), [m, n, k, seed, 1])
    assert out == "OK", f"bf16 transpose {m}x{n}x{k} seed={seed}: {out}"


# ---------------------------------------------------------------------------
# Known latent bug: benchmark packs Bp once with the last tile's NR while
# kernels read their own NR layout -> the timed path is garbage (bad=0,
# because the per-tile verify re-packs correctly). Replicated here: the 8x16
# kernel reading a 32-packed buffer must NOT equal the naive result.
# ---------------------------------------------------------------------------


@settings(max_examples=10, deadline=None)
@given(SEED)
@pytest.mark.xfail(
    reason="bench_gemm_avx512.py packs Bp once with the last tile's NR while "
    "kernels read their own NR layout: the 8x16 kernel on a 32-packed buffer "
    "returns garbage. Bug is real; see PROPERTY_TEST_NOTES.md.",
    strict=True,
)
def test_multi_nr_shared_bp_incorrect(seed):
    """The mixed-NR shared-Bp scenario must NOT equal the naive result."""
    out = run_check(_i8_bin(), [8, 32, 8, seed, 1])
    assert out == "OK", f"8x16 kernel on 32-packed Bp differs (seed={seed}): {out}"


# ---------------------------------------------------------------------------
# stc.aggen model properties
# ---------------------------------------------------------------------------


def _machine():
    return get_machine()


@given(
    st.integers(min_value=1, max_value=6).map(lambda t: 8 * t),
    st.integers(min_value=1, max_value=4).map(lambda t: 32 * t),
    K_VNNI,
)
def test_aggen_predictions_positive(m, n, k):
    mach = _machine()
    for fam in ("vnni8", "bf16"):
        cyc = mach.predicted_gemm_cycles(fam, m, n, k, 8, 32)
        assert cyc > 0
        assert (
            mach.gemm_tile_cycles_per_chunk(fam, 8, 32, 4 if fam == "vnni8" else 2) > 0
        )


@given(
    st.integers(min_value=2, max_value=6).map(lambda t: 8 * t),
    st.integers(min_value=1, max_value=4).map(lambda t: 32 * t),
    K_VNNI,
)
def test_aggen_predictions_monotone_in_m(m, n, k):
    mach = _machine()
    for fam in ("vnni8", "bf16"):
        small = mach.predicted_gemm_cycles(fam, m // 2, n, k, 8, 32)
        large = mach.predicted_gemm_cycles(fam, m, n, k, 8, 32)
        assert large > small


@given(
    st.integers(min_value=1, max_value=6).map(lambda t: 8 * t),
    st.integers(min_value=2, max_value=4).map(lambda t: 32 * t),
    K_VNNI,
)
def test_aggen_predictions_monotone_in_n(m, n, k):
    mach = _machine()
    for fam in ("vnni8", "bf16"):
        small = mach.predicted_gemm_cycles(fam, m, n // 2, k, 8, 32)
        large = mach.predicted_gemm_cycles(fam, m, n, k, 8, 32)
        assert large > small


@given(MUL8, MUL32, st.integers(min_value=2, max_value=8).map(lambda t: 8 * t))
def test_aggen_predictions_monotone_in_k(m, n, k):
    mach = _machine()
    for fam in ("vnni8", "bf16"):
        small = mach.predicted_gemm_cycles(fam, m, n, k // 2, 8, 32)
        large = mach.predicted_gemm_cycles(fam, m, n, k, 8, 32)
        assert large > small


@given(st.sampled_from(["vnni8", "bf16"]))
def test_best_tile_acceptable_to_generator(fam):
    """aggen.best_tile output must be emittable by stc.gemm_asm.

    Was a strict xfail (best_tile returned mr=12, which the fixed-register
    generator rejects); stc/aggen.py now constrains best_tile to what the
    generator can emit, so this is a hard property.
    """
    mach = _machine()
    mr, nr = mach.best_tile(fam)
    assert mr >= 1 and nr >= 16
    emit_gemm_kernel(fam, mr, nr)  # raises ValueError for mr > 8
