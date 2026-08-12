#!/usr/bin/env python3
"""MLP block and full transformer decoder layer (bf16), vs OpenBLAS/numpy.

The stack above GEMM, in order:
  1. MLP block:  up   = x . W1          (S x D) x (D x 4D)
                 g    = GELU(up)
                 out  = g . W2          (S x 4D) x (4D x D)
  2. Decoder layer (pre-norm, GPT-2 style):
                 xh   = LayerNorm(x)
                 qkv  = xh . Wqkv       (fused 3 heads-worth of projections)
                 q,k,v split; scores = q.k^T/sqrt(d); P = softmax; o = P.v
                 o    = o . Wo          (output projection)
                 x    = x + o           (residual 1)
                 xh2  = LayerNorm(x)
                 h    = xh2 . W1; g = GELU(h); h2 = g . W2
                 x    = x + h2          (residual 2)

Everything runs on the hand-scheduled bf16 micro-kernel from stc.gemm_asm
(VPDPBF16PS) with SIMD fp32 layernorm / GELU / softmax. The reference is a
numpy fp32 implementation (OpenBLAS sgemm for the GEMMs) on the same core.

Usage:
  python inference/bench/bench_decoder_avx512.py [--seq 2048] [--d 128] [--hm 4]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root
from inference.gemm_asm import emit_gemm_kernel  # noqa: E402


def gen_c(seq: int, d: int, hm: int) -> str:
    S, D, H = seq, d, d * hm
    # tokens replaced below; C braces stay single (no f-string escaping)
    t = r"""#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <x86intrin.h>
#include <sched.h>
extern void gemm_bf16_8x32_asm(const uint16_t* A, const int32_t* Bp, float* C, int K, int N);
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
    for (int n0 = 0; n0 < N; n0 += 32)
        for (int c = 0; c < K/2; c++)
            for (int j = 0; j < 32; j++) {
                uint32_t v = 0; int col = n0 + j;
                for (int t = 0; t < 2; t++)
                    v |= (uint32_t)B[col*K + (2*c + t)] << (16*t);
                Bp[((n0/32)*(K/2) + c)*32 + j] = v;
            }
}
static void gemm(const uint16_t* A, const int32_t* Bp, float* C, int M, int N, int K) {
    for (int n0 = 0; n0 < N; n0 += 32)
        for (int m0 = 0; m0 < M; m0 += 8)
            gemm_bf16_8x32_asm(A + m0*K, Bp + (n0/32)*(K/2)*32, C + m0*N + n0, K, N);
}
static void to_bf16(const float* x, uint16_t* y, int n) {
    for (int i = 0; i < n; i++) y[i] = f2b(x[i]);
}
static void split_qkv(const float* qkv, float* q, float* k, float* v, int S, int D) {
    for (int r = 0; r < S; r++) {
        memcpy(q + r*D, qkv + r*3*D, (size_t)D*4);
        memcpy(k + r*D, qkv + r*3*D + D, (size_t)D*4);
        memcpy(v + r*D, qkv + r*3*D + 2*D, (size_t)D*4);
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
            _mm512_storeu_ps(r + 16*j, _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(r + 16*j), mvec), iv));
    }
}
static void gelu(float* x, int n) {
    const float c = 0.7978845608f;
    for (int i = 0; i < n/16; i++) {
        __m512 v = _mm512_loadu_ps(x + 16*i);
        __m512 v3 = _mm512_mul_ps(v, _mm512_mul_ps(v, v));
        __m512 z = _mm512_mul_ps(_mm512_add_ps(v, _mm512_mul_ps(_mm512_set1_ps(0.044715f), v3)), _mm512_set1_ps(c));
        __m512 e = exp_ps(_mm512_add_ps(z, z));
        __m512 t = _mm512_div_ps(_mm512_sub_ps(e, _mm512_set1_ps(1.0f)), _mm512_add_ps(e, _mm512_set1_ps(1.0f)));
        __m512 g = _mm512_mul_ps(_mm512_set1_ps(0.5f), _mm512_mul_ps(v, _mm512_add_ps(_mm512_set1_ps(1.0f), t)));
        _mm512_storeu_ps(x + 16*i, g);
    }
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
#define NREP 7
static uint64_t t_gemm(const uint16_t* A, const int32_t* Bp, float* C, int M, int N, int K) {
    gemm(A, Bp, C, M, N, K);
    uint64_t best = ~0ULL;
    for (int r = 0; r < NREP; r++) {
        uint64_t t0 = __rdtsc();
        gemm(A, Bp, C, M, N, K);
        uint64_t d = __rdtsc() - t0;
        if (d < best) best = d;
    }
    return best;
}
int main(void) {
    const int S = @S@, D = @D@, H = @H@;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(4, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
    static uint16_t X[@S2@];
    static uint16_t W1[@DH@], W2[@HD@], Wqkv[@D3D@], Wo[@DD@];
    static int32_t Bp1[@B1@], Bp2[@B2@], Bpq[@BQ@], Bpo[@BO@];
    static float f_up[@SH@], f_g[@SH@], f_qkv[@S3D@], f_scores[@SS@], f_o[@S2@];
    static float xbuf[@S2@], wbuf[@S2@], fq[@S2@], fk[@S2@], fv[@S2@];
    static uint16_t b_up[@SH@], b_q[@S2@], b_k[@S2@], b_v[@S2@], b_o[@S2@];
    static int32_t Bpk[@BK@], Bpv[@BV@];
    static uint16_t b_p[@SS@];
    for (int i = 0; i < S*D; i++) {
        float v = ((i * 7) % 1001) / 1000.0f - 0.5f;
        X[i] = f2b(v * 0.1f);
        xbuf[i] = (float)b2f(X[i]);
    }
    for (int i = 0; i < D*H; i++) { float v = ((i * 11) % 401) / 400.0f - 0.5f; W1[i] = f2b(v * 0.05f); }
    for (int i = 0; i < H*D; i++) { float v = ((i * 13) % 401) / 400.0f - 0.5f; W2[i] = f2b(v * 0.05f); }
    for (int i = 0; i < D*3*D; i++) { float v = ((i * 17) % 401) / 400.0f - 0.5f; Wqkv[i] = f2b(v * 0.05f); }
    for (int i = 0; i < D*D; i++) { float v = ((i * 7) % 1001) / 1000.0f - 0.5f; Wo[i] = f2b(v * 0.05f); }
    pack_b_bf16_32(W1, Bp1, D, H);
    pack_b_bf16_32(W2, Bp2, H, D);
    pack_b_bf16_32(Wqkv, Bpq, D, 3*D);
    pack_b_bf16_32(Wo, Bpo, D, D);
    double cyc_up = (double)t_gemm(X, Bp1, f_up, S, H, D);
    gelu(f_up, S*H);
    to_bf16(f_up, b_up, S*H);
    double cyc_down = (double)t_gemm(b_up, Bp2, f_g, S, D, H);
    double cyc_attn = ~0ULL;
    for (int r = 0; r < NREP; r++) {
        for (int i = 0; i < S*D; i++) wbuf[i] = xbuf[i];
        layernorm(wbuf, S, D);
        to_bf16(wbuf, b_q, S*D);
        uint64_t t0 = __rdtsc();
        gemm(b_q, Bpq, f_qkv, S, 3*D, D);
        split_qkv(f_qkv, fq, fk, fv, S, D);
        to_bf16(fq, b_q, S*D); to_bf16(fk, b_k, S*D); to_bf16(fv, b_v, S*D);
        pack_b_bf16_32_T(b_k, Bpk, D, S);
        pack_b_bf16_32(b_v, Bpv, S, D);
        gemm(b_q, Bpk, f_scores, S, S, D);
        for (int i = 0; i < S*S; i++) f_scores[i] *= 1.0f / sqrtf((float)D);
        softmax(f_scores, S);
        to_bf16(f_scores, b_p, S*S);
        gemm(b_p, Bpv, f_o, S, D, S);
        to_bf16(f_o, b_o, S*D);
        gemm(b_o, Bpo, f_o, S, D, D);
        uint64_t d = __rdtsc() - t0;
        if (d < cyc_attn) cyc_attn = d;
    }
    double cyc_mlp = ~0ULL;
    for (int r = 0; r < NREP; r++) {
        for (int i = 0; i < S*D; i++) xbuf[i] += f_o[i];
        for (int i = 0; i < S*D; i++) wbuf[i] = xbuf[i];
        layernorm(wbuf, S, D);
        to_bf16(wbuf, b_q, S*D);
        uint64_t t0 = __rdtsc();
        gemm(b_q, Bp1, f_up, S, H, D);
        gelu(f_up, S*H);
        to_bf16(f_up, b_up, S*H);
        gemm(b_up, Bp2, f_g, S, D, H);
        uint64_t d = __rdtsc() - t0;
        if (d < cyc_mlp) cyc_mlp = d;
    }
    /* clean single pass for the correctness dump */
    for (int i = 0; i < S*D; i++) xbuf[i] = (float)b2f(X[i]);
    for (int i = 0; i < S*D; i++) wbuf[i] = xbuf[i];
    layernorm(wbuf, S, D);
    to_bf16(wbuf, b_q, S*D);
    gemm(b_q, Bpq, f_qkv, S, 3*D, D);
    split_qkv(f_qkv, fq, fk, fv, S, D);
    to_bf16(fq, b_q, S*D); to_bf16(fk, b_k, S*D); to_bf16(fv, b_v, S*D);
    pack_b_bf16_32_T(b_k, Bpk, D, S);
    pack_b_bf16_32(b_v, Bpv, S, D);
    gemm(b_q, Bpk, f_scores, S, S, D);
    for (int i = 0; i < S*S; i++) f_scores[i] *= 1.0f / sqrtf((float)D);
#ifdef STAGE
    { FILE* f = fopen("stage_scores.bin","wb"); fwrite(f_scores,4,S*S,f); fclose(f); }
#endif
    softmax(f_scores, S);
#ifdef STAGE
    { FILE* f = fopen("stage_p.bin","wb"); fwrite(f_scores,4,S*S,f); fclose(f); }
#endif
    to_bf16(f_scores, b_p, S*S);
    gemm(b_p, Bpv, f_o, S, D, S);
#ifdef STAGE
    { FILE* f = fopen("stage_pv.bin","wb"); fwrite(f_o,4,S*D,f); fclose(f); }
    { FILE* f = fopen("bp.bin","wb"); fwrite(b_p,2,S*S,f); fclose(f); }
    { FILE* f = fopen("bv.bin","wb"); fwrite(b_v,2,S*D,f); fclose(f); }
#endif
    to_bf16(f_o, b_o, S*D);
    gemm(b_o, Bpo, f_o, S, D, D);
#ifdef STAGE
    { FILE* f = fopen("stage_o.bin","wb"); fwrite(f_o,4,S*D,f); fclose(f); }
#endif
    for (int i = 0; i < S*D; i++) xbuf[i] += f_o[i];
    for (int i = 0; i < S*D; i++) wbuf[i] = xbuf[i];
    layernorm(wbuf, S, D);
    to_bf16(wbuf, b_q, S*D);
    gemm(b_q, Bp1, f_up, S, H, D);
#ifdef STAGE
    { FILE* f = fopen("stage_h.bin","wb"); fwrite(f_up,4,S*H,f); fclose(f); }
#endif
    gelu(f_up, S*H);
    to_bf16(f_up, b_up, S*H);
    gemm(b_up, Bp2, f_g, S, D, H);
#ifdef STAGE
    { FILE* f = fopen("stage_mlp.bin","wb"); fwrite(f_g,4,S*D,f); fclose(f); }
#endif
    for (int i = 0; i < S*D; i++) xbuf[i] += f_g[i];
    printf("RESULT cyc_up=%.0f cyc_down=%.0f cyc_attn=%.0f cyc_mlp=%.0f\n",
           cyc_up, cyc_down, cyc_attn, cyc_mlp);
    FILE* f = fopen("out.bin", "wb");
    fwrite(xbuf, sizeof(float), S*D, f);
    fclose(f);
    return 0;
}
"""
    return (
        t.replace("@S@", str(S))
        .replace("@D@", str(D))
        .replace("@H@", str(H))
        .replace("@S2@", str(S * D))
        .replace("@SS@", str(S * S))
        .replace("@SH@", str(S * H))
        .replace("@S3D@", str(S * 3 * D))
        .replace("@DH@", str(D * H))
        .replace("@HD@", str(H * D))
        .replace("@D3D@", str(D * 3 * D))
        .replace("@DD@", str(D * D))
        .replace("@B1@", str((D // 2) * H))
        .replace("@B2@", str((H // 2) * D))
        .replace("@BQ@", str((D // 2) * 3 * D))
        .replace("@BO@", str((D // 2) * D))
        .replace("@BK@", str((D // 2) * S))
        .replace("@BV@", str((S // 2) * D))
    )


def _ref(seq: int, d: int, hm: int) -> dict:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import numpy as np

    S, D, H = seq, d, d * hm

    def bf16(f):
        u = f.astype(np.float32).view(np.uint32)
        r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
        return (r.astype(np.uint32) << 16).view(np.float32)

    i = np.arange(S * D)
    base = ((i * 7) % 1001) / 1000.0 - 0.5
    x = bf16(base * 0.1).reshape(S, D).astype(np.float32)
    i2 = np.arange(D * H)
    w1 = bf16((((i2 * 11) % 401) / 400.0 - 0.5) * 0.05).reshape(D, H)
    w2 = bf16((((i2 * 13) % 401) / 400.0 - 0.5) * 0.05).reshape(H, D)
    i3 = np.arange(D * 3 * D)
    wqkv = bf16((((i3 * 17) % 401) / 400.0 - 0.5) * 0.05).reshape(D, 3 * D)
    i4 = np.arange(D * D)
    wo = bf16((((i4 * 7) % 1001) / 1000.0 - 0.5) * 0.05).reshape(D, D)

    def gelu_np(z):
        return 0.5 * z * (1.0 + np.tanh(0.7978845608 * (z + 0.044715 * z**3)))

    def layernorm_np(xr):
        m = xr.mean(axis=1, keepdims=True)
        v = ((xr - m) ** 2).mean(axis=1, keepdims=True)
        return (xr - m) / np.sqrt(v + 1e-5)

    # MLP
    t0 = time.perf_counter()
    up = x @ w1
    t_up = time.perf_counter() - t0
    t0 = time.perf_counter()
    g = gelu_np(up)
    t_gelu = time.perf_counter() - t0
    t0 = time.perf_counter()
    out_mlp = bf16(g) @ w2
    t_down = time.perf_counter() - t0

    # decoder layer
    t0 = time.perf_counter()
    xh = layernorm_np(x)
    qkv = bf16(xh) @ wqkv
    q, k, v = np.split(qkv, 3, axis=1)
    scores = bf16(q) @ bf16(k).T
    scores *= 1.0 / np.sqrt(D)
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    o = bf16(p) @ bf16(v)
    o = bf16(o) @ wo
    xr = x + o
    xh2 = layernorm_np(xr)
    h = bf16(xh2) @ w1
    g2 = gelu_np(h)
    h2 = bf16(g2) @ w2
    xr = xr + h2
    t_layer = time.perf_counter() - t0
    return {
        "t_up": t_up,
        "t_gelu": t_gelu,
        "t_down": t_down,
        "t_layer": t_layer,
        "out": xr.astype(np.float32),
    }


def run_c(seq: int, d: int, hm: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfile = td / "dec.c"
        cfile.write_text(gen_c(seq, d, hm))
        asm_file = td / "gemm.S"
        asm_file.write_text(emit_gemm_kernel("bf16", 8, 32))
        r = subprocess.run(
            ["gcc", "-c", str(asm_file), "-o", str(td / "gemm.o")],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stderr)
            sys.exit(1)
        r = subprocess.run(
            [
                "gcc",
                "-O3",
                "-march=native",
                "-o",
                str(td / "dec"),
                str(cfile),
                str(td / "gemm.o"),
                "-lm",
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stderr)
            sys.exit(1)
        out = subprocess.run([str(td / "dec")], capture_output=True, text=True, cwd=td)
        if out.returncode != 0:
            print(out.stderr)
            sys.exit(1)
        res = {}
        for line in out.stdout.splitlines():
            if line.startswith("RESULT"):
                for kv in line.split()[1:]:
                    k, v = kv.split("=")
                    res[k] = float(v)
        res["out"] = (td / "out.bin").read_bytes()
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--hm", type=int, default=4, help="hidden multiplier (H = D*hm)")
    args = ap.parse_args()
    S, D, H = args.seq, args.d, args.d * args.hm
    tsc = 3.7e9

    print(f"bf16 MLP + decoder layer: S={S} D={D} H={H}")
    res = run_c(S, D, args.hm)
    ref = _ref(S, D, args.hm)

    import numpy as np

    our = np.frombuffer(res["out"], dtype=np.float32).reshape(S, D)
    abs_err = np.abs(our - ref["out"]).max()
    act_scale = np.abs(ref["out"]).max()
    print(
        f"decoder out max abs err vs fp32 numpy: {abs_err:.5f} "
        f"(activation scale {act_scale:.3f})"
    )

    macs_mlp = 2.0 * S * (2.0 * D * H)
    print(
        f"MLP:   up  {S*D*H/1e9:6.2f} GFLOP in {res['cyc_up']/1e6:6.1f} Mcyc = "
        f"{S*D*H*2/1e9/(res['cyc_up']/tsc):6.1f} GF/s"
    )
    print(
        f"       gelu (SIMD) not timed; down {S*H*D/1e9:6.2f} GFLOP in "
        f"{res['cyc_down']/1e6:6.1f} Mcyc = {S*H*D*2/1e9/(res['cyc_down']/tsc):6.1f} GF/s"
    )
    tot_mlp_ours = (res["cyc_up"] + res["cyc_down"]) / tsc
    tot_mlp_np = ref["t_up"] + ref["t_down"]
    print(
        f"MLP GEMMs: ours {tot_mlp_ours*1e3:6.2f} ms vs numpy {tot_mlp_np*1e3:6.2f} ms "
        f"({tot_mlp_np/tot_mlp_ours:.2f}x; +numpy gelu {ref['t_gelu']*1e3:.2f} ms)"
    )

    tot_ours = (res["cyc_attn"] + res["cyc_mlp"]) / tsc
    print(
        f"decoder layer: attn-part {res['cyc_attn']/tsc*1e3:6.2f} ms + "
        f"mlp-part {res['cyc_mlp']/tsc*1e3:6.2f} ms = {tot_ours*1e3:6.2f} ms"
    )
    print(
        f"decoder layer (numpy): {ref['t_layer']*1e3:6.2f} ms "
        f"({ref['t_layer']/tot_ours:.2f}x slower)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
