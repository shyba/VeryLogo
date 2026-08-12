"""Property-based tests for the SIMD fp32 helper functions used by the
attention/decoder C drivers (exp_ps, softmax, layernorm, split_qkv,
f2b/b2f).

The functions under test are copied verbatim from the current
inference/inference/results/bench_decoder_avx512.py template so the tests validate the real
implementations. hypothesis drives (mode, seed, S, D); the C checker
generates its own data from the seed and prints OK / FAIL + detail.

Known bug classes these target:
  - reversed-Horner exp (exp(0) != 1) - fixed, must pass now,
  - softmax masking a scrambled QK^T (rows still sum to 1),
  - the strided split_qkv and its in-place clobbering,
  - overflow/NaN for larger |x| in exp_ps.

Run:  .venv/bin/python -m pytest tests/test_properties_simd.py -q
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from property_harness import build_binary, run_check

SIMD_CHECKER_C = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
/* --- functions under test, copied from inference/inference/results/bench_decoder_avx512.py --- */
static inline uint16_t f2b(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16;
    return (uint16_t)r;
}
static inline float b2f(uint16_t b) {
    uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f;
}
static inline __m512 exp_ps(__m512 x) {
    __m512 t = _mm512_mul_ps(x, _mm512_set1_ps(1.4426950408889634f));
    t = _mm512_min_ps(_mm512_max_ps(t, _mm512_set1_ps(-126.0f)), _mm512_set1_ps(127.0f));
    __m512i n = _mm512_cvtps_epi32(_mm512_roundscale_ps(t, 0));
    __m512 r = _mm512_sub_ps(t, _mm512_cvtepi32_ps(n));
    __m512 p = _mm512_set1_ps(0.001333355815f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.009618129108f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.055504108665f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.240226506959f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.693147180560f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.000000000000f));
    __m512i e = _mm512_slli_epi32(_mm512_add_epi32(n, _mm512_set1_epi32(127)), 23);
    return _mm512_mul_ps(p, _mm512_castsi512_ps(e));
}
static void softmax(float* x, int S) {
    const int V = S / 16;
    for (int i = 0; i < S; i++) {
        float* r = x + i*S;
        __m512 mx = _mm512_loadu_ps(r);
        for (int j = 1; j < V; j++)
            mx = _mm512_max_ps(mx, _mm512_loadu_ps(r + 16*j));
        float m = _mm512_reduce_max_ps(mx);
        __m512 mvec = _mm512_set1_ps(m);
        __m512 sum = _mm512_setzero_ps();
        for (int j = 0; j < V; j++) {
            __m512 e = exp_ps(_mm512_sub_ps(_mm512_loadu_ps(r + 16*j), mvec));
            _mm512_storeu_ps(r + 16*j, e);
            sum = _mm512_add_ps(sum, e);
        }
        float inv = 1.0f / _mm512_reduce_add_ps(sum);
        __m512 iv = _mm512_set1_ps(inv);
        for (int j = 0; j < V; j++)
            _mm512_storeu_ps(r + 16*j, _mm512_mul_ps(_mm512_loadu_ps(r + 16*j), iv));
    }
}
/* rectangular softmax (B x C): decode attention; same math as softmax */
static void softmax_rect(float* x, int rows, int cols) {
    const int V = cols / 16;
    for (int i = 0; i < rows; i++) {
        float* r = x + i*cols;
        __m512 mx = _mm512_loadu_ps(r);
        for (int j = 1; j < V; j++)
            mx = _mm512_max_ps(mx, _mm512_loadu_ps(r + 16*j));
        float m = _mm512_reduce_max_ps(mx);
        __m512 mvec = _mm512_set1_ps(m);
        __m512 sum = _mm512_setzero_ps();
        for (int j = 0; j < V; j++) {
            __m512 e = exp_ps(_mm512_sub_ps(_mm512_loadu_ps(r + 16*j), mvec));
            _mm512_storeu_ps(r + 16*j, e);
            sum = _mm512_add_ps(sum, e);
        }
        float inv = 1.0f / _mm512_reduce_add_ps(sum);
        __m512 iv = _mm512_set1_ps(inv);
        for (int j = 0; j < V; j++)
            _mm512_storeu_ps(r + 16*j, _mm512_mul_ps(_mm512_loadu_ps(r + 16*j), iv));
    }
}
static void layernorm(float* x, int S, int D) {
    const int V = D / 16;
    for (int i = 0; i < S; i++) {
        float* r = x + i*D;
        __m512 sum = _mm512_setzero_ps();
        for (int j = 0; j < V; j++) sum = _mm512_add_ps(sum, _mm512_loadu_ps(r + 16*j));
        float mean = _mm512_reduce_add_ps(sum) / (float)D;
        __m512 mvec = _mm512_set1_ps(mean);
        __m512 ss = _mm512_setzero_ps();
        for (int j = 0; j < V; j++) {
            __m512 t = _mm512_sub_ps(_mm512_loadu_ps(r + 16*j), mvec);
            ss = _mm512_fmadd_ps(t, t, ss);
        }
        float var = _mm512_reduce_add_ps(ss) / (float)D;
        __m512 iv = _mm512_set1_ps(1.0f / sqrtf(var + 1e-5f));
        for (int j = 0; j < V; j++)
            _mm512_storeu_ps(r + 16*j,
                _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(r + 16*j), mvec), iv));
    }
}
static void split_qkv(const float* qkv, float* q, float* k, float* v, int S, int D) {
    for (int r = 0; r < S; r++) {
        memcpy(q + r*D, qkv + r*3*D, (size_t)D*4);
        memcpy(k + r*D, qkv + r*3*D + D, (size_t)D*4);
        memcpy(v + r*D, qkv + r*3*D + 2*D, (size_t)D*4);
    }
}
/* --- test driver ------------------------------------------------------- */
static uint64_t st;
static uint64_t rng(void) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; }
static float rndf(void) { return ((float)(rng() & 0xFFFFFF) / 8388608.0f) - 1.0f; } /* [-1,1] */

static float ev(float* buf, int idx) {
    union { __m512 v; float f[16]; } u;
    u.v = ((__m512*)buf)[idx];
    return u.f[0];
}
static float exps(float x) { __m512 v = exp_ps(_mm512_set1_ps(x)); union { __m512 v; float f[16]; } u; u.v = v; return u.f[0]; }

int main(int argc, char** argv) {
    int mode = atoi(argv[1]);
    st = strtoull(argv[2], NULL, 0); if (st == 0) st = 0x9E3779B97F4A7C15ULL;
    int S = atoi(argv[3]), D = atoi(argv[4]);
    static float buf[65536], buf2[65536];

    if (mode == 0) {  /* exp properties */
        float e0 = exps(0.0f);
        if (fabsf(e0 - 1.0f) > 1e-5f) { printf("FAIL exp(0)=%.6f\n", e0); return 0; }
        for (int i = 0; i < 500; i++) {
            float a = rndf() * 5.0f, b = rndf() * 5.0f;
            float lhs = exps(a + b), rhs = exps(a) * exps(b);
            if (fabsf(lhs - rhs) > 1e-4f * fmaxf(1.0f, fabsf(rhs))) {
                printf("FAIL exp(a+b) a=%.4f b=%.4f %.6f vs %.6f\n", a, b, lhs, rhs);
                return 0;
            }
        }
        float prev = -1e30f;
        for (int i = 0; i < 1000; i++) {
            float x = -6.0f + 12.0f * (float)i / 999.0f;
            float e = exps(x);
            if (e < prev) { printf("FAIL monotone at x=%.4f %.6f < %.6f\n", x, e, prev); return 0; }
            prev = e;
        }
        for (int i = 0; i < 200; i++) {
            float x = rndf() * 80.0f;
            float e = exps(x);
            if (!isfinite(e)) { printf("FAIL non-finite exp(%.2f)=%f\n", x, e); return 0; }
        }
        printf("OK\n");
        return 0;
    }
    if (mode == 1) {  /* softmax: sums, argmax, shift invariance */
        for (int i = 0; i < S*S; i++) buf[i] = rndf() * 3.0f;
        memcpy(buf2, buf, sizeof(float) * S * S);
        softmax(buf, S);
        for (int i = 0; i < S; i++) {
            double s = 0;
            for (int j = 0; j < S; j++) s += buf[i*S + j];
            if (fabs(s - 1.0) > 1e-4) { printf("FAIL softmax row %d sum %.6f\n", i, s); return 0; }
        }
        for (int i = 0; i < S; i++) {
            int mi = 0;
            for (int j = 1; j < S; j++) if (buf2[i*S + j] > buf2[i*S + mi]) mi = j;
            int mo = 0;
            for (int j = 1; j < S; j++) if (buf[i*S + j] > buf[i*S + mo]) mo = j;
            if (mi != mo) { printf("FAIL argmax row %d: %d vs %d\n", i, mi, mo); return 0; }
        }
        for (int i = 0; i < S; i++)
            for (int j = 0; j < S; j++) buf2[i*S + j] += 0.7f;   /* shift by constant */
        softmax(buf2, S);
        for (int i = 0; i < S*S; i++)
            if (fabsf(buf[i] - buf2[i]) > 1e-5f) { printf("FAIL shift-invariance at %d: %.8f vs %.8f\n", i, buf[i], buf2[i]); return 0; }
        printf("OK\n");
        return 0;
    }
    if (mode == 3) {  /* softmax_rect: B x C rows sum to 1, argmax preserved */
        int B = S, C = S * 2;  /* non-square: 8 x 16 etc. */
        if (C > 4096) C = 4096;  /* buf[65536] floats */
        for (int i = 0; i < B*C; i++) buf[i] = rndf() * 3.0f;
        memcpy(buf2, buf, sizeof(float) * B * C);
        softmax_rect(buf, B, C);
        for (int i = 0; i < B; i++) {
            double s = 0;
            for (int j = 0; j < C; j++) s += buf[i*C + j];
            if (fabs(s - 1.0) > 1e-4) { printf("FAIL softmax_rect row %d sum %.6f\n", i, s); return 0; }
        }
        for (int i = 0; i < B; i++) {
            int mi = 0;
            for (int j = 1; j < C; j++) if (buf2[i*C + j] > buf2[i*C + mi]) mi = j;
            int mo = 0;
            for (int j = 1; j < C; j++) if (buf[i*C + j] > buf[i*C + mo]) mo = j;
            if (mi != mo) { printf("FAIL softmax_rect argmax row %d\n", i); return 0; }
        }
        printf("OK\n");
        return 0;
    }
    if (mode == 2) {  /* layernorm: mean ~ 0, var ~ 1 */
        for (int i = 0; i < S*D; i++) buf[i] = rndf() * 2.0f;
        layernorm(buf, S, D);
        for (int i = 0; i < S; i++) {
            double mean = 0, var = 0;
            for (int j = 0; j < D; j++) mean += buf[i*D + j];
            mean /= D;
            for (int j = 0; j < D; j++) var += (buf[i*D + j] - mean) * (buf[i*D + j] - mean);
            var /= D;
            if (fabs(mean) > 1e-4 || fabs(var - 1.0) > 2e-3) {
                printf("FAIL layernorm row %d mean %.6f var %.6f\n", i, mean, var);
                return 0;
            }
        }
        printf("OK\n");
        return 0;
    }
    if (mode == 6) {  /* split_qkv round-trip: concat(split(x)) == x */
        for (int i = 0; i < S*3*D; i++) buf[i] = rndf();
        static float q[65536], k[65536], v[65536];
        split_qkv(buf, q, k, v, S, D);
        int bad = 0;
        for (int r = 0; r < S; r++) {
            if (memcmp(q + r*D, buf + r*3*D, D*4) || memcmp(k + r*D, buf + r*3*D + D, D*4) ||
                memcmp(v + r*D, buf + r*3*D + 2*D, D*4)) { bad = 1; break; }
        }
        if (bad) { printf("FAIL split_qkv round-trip\n"); return 0; }
        printf("OK\n");
        return 0;
    }
    if (mode == 4) {  /* residual: x + o preserves the o signal */
        for (int i = 0; i < 1000; i++) {
            float x = rndf(), o = rndf() * 1e-3f;
            float y = x + o;
            if (fabsf(y - x) > 2e-3f) { printf("FAIL residual x=%.6f o=%.6f y=%.6f\n", x, o, y); return 0; }
        }
        printf("OK\n");
        return 0;
    }
    if (mode == 5) {  /* f2b/b2f round-trip accuracy */
        for (int i = 0; i < 2000; i++) {
            float x = rndf();
            float y = b2f(f2b(x));
            if (fabsf(y - x) > 1.1f * 0.00390625f * fmaxf(1e-6f, fabsf(x))) {
                printf("FAIL f2b round-trip x=%.8f y=%.8f\n", x, y);
                return 0;
            }
        }
        printf("OK\n");
        return 0;
    }
    printf("FAIL unknown mode %d\n", mode);
    return 0;
}
"""


_SIMD_BIN: str | None = None


def _require_cpu():
    from property_harness import cpu_has_avx512_vnni

    if not cpu_has_avx512_vnni():
        pytest.skip("requires AVX512-VNNI + AVX512-BF16 (Zen 5)")


def _simd_bin() -> str:
    global _SIMD_BIN
    _require_cpu()
    if _SIMD_BIN is None:
        try:
            _SIMD_BIN = build_binary("simd", SIMD_CHECKER_C)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"SIMD checker build failed: {exc}")
    return _SIMD_BIN


S = st.integers(min_value=1, max_value=8).map(lambda t: 16 * t)  # 16..128
D = st.integers(min_value=1, max_value=8).map(lambda t: 16 * t)  # 16..128
SEED = st.integers(min_value=0, max_value=2**32 - 1)


@settings(max_examples=20, deadline=None)
@given(SEED)
def test_exp_properties(seed):
    out = run_check(_simd_bin(), [0, seed, 64, 64])
    assert out == "OK", f"exp properties seed={seed}: {out}"


@settings(max_examples=15, deadline=None)
@given(S, SEED)
def test_softmax_properties(s, seed):
    out = run_check(_simd_bin(), [1, seed, s, s])
    assert out == "OK", f"softmax S={s} seed={seed}: {out}"


@settings(max_examples=15, deadline=None)
@given(S, D, SEED)
def test_layernorm_properties(s, d, seed):
    out = run_check(_simd_bin(), [2, seed, s, d])
    assert out == "OK", f"layernorm S={s} D={d} seed={seed}: {out}"


@settings(max_examples=15, deadline=None)
@given(S, SEED)
def test_softmax_rect_properties(s, seed):
    """Rectangular B x C softmax used by KV-cache decode: rows sum to 1,
    argmax preserved (non-square B != C)."""
    out = run_check(_simd_bin(), [3, seed, s, s])
    assert out == "OK", f"softmax_rect S={s} seed={seed}: {out}"


@settings(max_examples=15, deadline=None)
@given(S, D, SEED)
def test_split_qkv_roundtrip(s, d, seed):
    out = run_check(_simd_bin(), [6, seed, s, d])
    assert out == "OK", f"split_qkv S={s} D={d} seed={seed}: {out}"


@settings(max_examples=10, deadline=None)
@given(SEED)
def test_residual_signal(seed):
    out = run_check(_simd_bin(), [4, seed, 64, 64])
    assert out == "OK", f"residual seed={seed}: {out}"


@settings(max_examples=10, deadline=None)
@given(SEED)
def test_f2b_roundtrip(seed):
    out = run_check(_simd_bin(), [5, seed, 64, 64])
    assert out == "OK", f"f2b seed={seed}: {out}"
