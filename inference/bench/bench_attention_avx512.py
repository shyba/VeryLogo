#!/usr/bin/env python3
"""Single-head attention forward pass (bf16), the piece above GEMM.

Attention is the core transformer component that is *larger than a GEMM*:
  scores = Q . K^T / sqrt(D)      (GEMM1, SxS output)
  P      = softmax(scores, axis=1)
  out    = P . V                  (GEMM2, SxD output)

Both GEMMs run on the hand-scheduled bf16 micro-kernel emitted by
stc.gemm_asm (VPDPBF16PS, 2/cyc on P01 -> 64 MACs/cyc peak), with the same
K-major/N-interleaved packing. Softmax is a SIMD fp32 pass (poly-exp; memory-bound,
~10% of the head at S=2048). The reference is numpy's matmul on this box, which is
OpenBLAS 0.3.34 (scipy-openblas, sgemm, single-threaded) - i.e. real tuned
existing code running on the same CPU.

Usage:
  python inference/inference/results/bench_attention_avx512.py [--seq 2048] [--d 128]
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


def gen_c(seq: int, d: int) -> str:
    S, D = seq, d
    return f"""#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <x86intrin.h>
extern void gemm_bf16_8x32_asm(const uint16_t* A, const int32_t* Bp, float* C, int K, int N);
static inline uint16_t f2b(float f) {{
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16;
    return (uint16_t)r;
}}
static inline float b2f(uint16_t b) {{
    uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f;
}}
/* pack B (KxN, row-major) into the K-major/N-interleaved layout */
static void pack_b_bf16_32_T(const uint16_t* B, int32_t* Bp, int K, int N) {{
    for (int n0 = 0; n0 < N; n0 += 32)
        for (int c = 0; c < K/2; c++)
            for (int j = 0; j < 32; j++) {{
                uint32_t v = 0; int col = n0 + j;
                for (int t = 0; t < 2; t++)
                    v |= (uint32_t)B[col*K + (2*c + t)] << (16*t);
                Bp[((n0/32)*(K/2) + c)*32 + j] = v;
            }}
}}
static void pack_b_bf16_32(const uint16_t* B, int32_t* Bp, int K, int N) {{
    for (int n0 = 0; n0 < N; n0 += 32)
        for (int c = 0; c < K/2; c++)
            for (int j = 0; j < 32; j++) {{
                uint32_t v = 0; int col = n0 + j;
                for (int t = 0; t < 2; t++)
                    v |= (uint32_t)B[(2*c + t)*N + col] << (16*t);
                Bp[((n0/32)*(K/2) + c)*32 + j] = v;
            }}
}}
/* C(MxN) = A(MxK) . Bp(KxN, packed), tiles 8x32; M%8==0, N%32==0, K%2==0 */
static void gemm(const uint16_t* A, const int32_t* Bp, float* C, int M, int N, int K) {{
    for (int n0 = 0; n0 < N; n0 += 32)
        for (int m0 = 0; m0 < M; m0 += 8)
            gemm_bf16_8x32_asm(
                A + m0*K,
                Bp + (n0/32)*(K/2)*32,
                C + m0*N + n0, K, N);
}}
/* exp2-free fast exp: e^x = 2^(x*log2e); n=round, r in [-0.5,0.5],
 * 2^r by degree-5 minimax, scale by 2^n via exponent arithmetic. */
static inline __m512 exp_ps(__m512 x) {{
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
}}
/* softmax rows of SxS fp32 scores in place; vectorized (memory-bound) */
static void softmax(float* x, int S) {{
    const int V = S / 16;
    for (int i = 0; i < S; i++) {{
        float* r = x + i*S;
        __m512 mx = _mm512_loadu_ps(r);
        for (int j = 1; j < V; j++)
            mx = _mm512_max_ps(mx, _mm512_loadu_ps(r + 16*j));
        float m = _mm512_reduce_max_ps(mx);
        __m512 mvec = _mm512_set1_ps(m);
        __m512 sum = _mm512_setzero_ps();
        for (int j = 0; j < V; j++) {{
            __m512 e = exp_ps(_mm512_sub_ps(_mm512_loadu_ps(r + 16*j), mvec));
            _mm512_storeu_ps(r + 16*j, e);
            sum = _mm512_add_ps(sum, e);
        }}
        float inv = 1.0f / _mm512_reduce_add_ps(sum);
        __m512 iv = _mm512_set1_ps(inv);
        for (int j = 0; j < V; j++)
            _mm512_storeu_ps(r + 16*j, _mm512_mul_ps(_mm512_loadu_ps(r + 16*j), iv));
    }}
}}
int main(void) {{
    const int S = {S}, D = {D};
    static uint16_t Q[{S*D}], Kt[{S*D}], V[{S*D}];
    static int32_t Bp1[{(D//2)*S}], Bp2[{(S//2)*D}];
    static float scores[{S*S}], out[{S*D}];
    static uint16_t P16[{S*S}];
    for (int i = 0; i < S*D; i++) {{
        float q = ((i * 7) % 1001) / 1000.0f - 0.5f;
        Q[i] = f2b(q * 0.1f);
        Kt[i] = f2b(q * 0.1f);
        V[i] = f2b(q);
    }}
    /* GEMM1: scores = Q . Kt  (Kt is K^T: D x S) */
    pack_b_bf16_32_T(Kt, Bp1, D, S);  /* Kt stored SxD -> pack the transpose */
    uint64_t t0 = __rdtsc();
    gemm(Q, Bp1, scores, S, S, D);
    uint64_t t1 = __rdtsc();
    double cyc_qk = (double)(t1 - t0);
    float inv = 1.0f / sqrtf((float)D);
    for (int i = 0; i < S*S; i++) scores[i] *= inv;
    uint64_t s0 = __rdtsc();
    softmax(scores, S);
    uint64_t s1 = __rdtsc();
    double cyc_sm = (double)(s1 - s0);
    for (int i = 0; i < S*S; i++) P16[i] = f2b(scores[i]);
    pack_b_bf16_32(V, Bp2, S, D);
    t0 = __rdtsc();
    gemm(P16, Bp2, out, S, D, S);
    t1 = __rdtsc();
    double cyc_pv = (double)(t1 - t0);
    printf("RESULT cyc_qk=%.0f cyc_pv=%.0f cyc_sm=%.0f\\n", cyc_qk, cyc_pv, cyc_sm);
    double chk = 0;
    for (int i = 0; i < S*D; i++) chk += out[i] != 0.0f ? 1 : 0;
    printf("chk=%.0f\\n", chk);
    FILE* f = fopen("out.bin", "wb");
    fwrite(out, sizeof(float), S*D, f);
    fclose(f);
    return 0;
}}
"""


def run_c(seq: int, d: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        cfile = Path(td) / "attn.c"
        cfile.write_text(gen_c(seq, d))
        asm_file = Path(td) / "gemm.S"
        asm_file.write_text(emit_gemm_kernel("bf16", 8, 32))
        exe = Path(td) / "attn"
        r = subprocess.run(
            ["gcc", "-c", str(asm_file), "-o", str(Path(td) / "gemm.o")],
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
                str(exe),
                str(cfile),
                str(Path(td) / "gemm.o"),
                "-lm",
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stderr)
            sys.exit(1)
        out = subprocess.run([str(exe)], capture_output=True, text=True, cwd=td)
        if out.returncode != 0:
            print(out.stderr)
            sys.exit(1)
        res = {}
        for line in out.stdout.splitlines():
            if line.startswith("RESULT"):
                for kv in line.split()[1:]:
                    k, v = kv.split("=")
                    res[k] = float(v)
        res["out"] = (Path(td) / "out.bin").read_bytes()
    return res


def _bf16_np(f):
    """Round fp32 to bf16 exactly like the C's f2b (nearest-even at bit 16)."""
    import numpy as np

    u = f.astype(np.float32).view(np.uint32)
    r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return (r.astype(np.uint32) << 16).view(np.float32)


def numpy_reference(seq: int, d: int) -> dict:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import numpy as np

    # identical data to the C kernel, bf16-quantized like the C's f2b inputs
    i = np.arange(seq * d)
    base = ((i * 7) % 1001) / 1000.0 - 0.5
    q = _bf16_np(base * 0.1).reshape(seq, d)
    kt = _bf16_np(base * 0.1).reshape(seq, d)
    v = _bf16_np(base).reshape(seq, d)

    t0 = time.perf_counter()
    scores = q @ kt.T
    t1 = time.perf_counter()
    qk_s = t1 - t0

    scores *= 1.0 / np.sqrt(d).astype(np.float32)
    t0 = time.perf_counter()
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    t1 = time.perf_counter()
    softmax_s = t1 - t0

    t0 = time.perf_counter()
    out = p @ v
    t1 = time.perf_counter()
    pv_s = t1 - t0
    return {
        "qk_s": qk_s,
        "pv_s": pv_s,
        "softmax_s": softmax_s,
        "out": out.astype(np.float32),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--d", type=int, default=128)
    args = ap.parse_args()
    S, D = args.seq, args.d

    print(
        f"single-head attention, bf16: S={S} D={D} "
        f"(scores {S}x{S} fp32 = {S*S*4/1e6:.0f} MB)"
    )
    res = run_c(S, D)
    ref = numpy_reference(S, D)

    import numpy as np

    our = np.frombuffer(res["out"], dtype=np.float32).reshape(S, D)
    abs_err = np.abs(our - ref["out"]).max()
    score_scale = 1.0 / np.sqrt(D)  # scores ~ N(0,1) * 1/sqrt(D)
    print(
        f"max abs err vs fp32 numpy: {abs_err:.5f} "
        f"(score scale {score_scale:.3f}; bf16-P quantization is the main loss)"
    )

    macs_qk = 2.0 * S * S * D
    macs_pv = 2.0 * S * S * D
    tsc = 3.7e9  # TSC clock on this box
    gflop = (macs_qk + macs_pv) / 1e9
    print(
        f"ours:   QK^T {macs_qk/1e9:7.1f} GFLOP in {res['cyc_qk']/1e6:7.1f} Mcyc "
        f"= {macs_qk/1e9/(res['cyc_qk']/tsc):7.1f} GF/s @TSC"
    )
    print(
        f"        PV   {macs_pv/1e9:7.1f} GFLOP in {res['cyc_pv']/1e6:7.1f} Mcyc "
        f"= {macs_pv/1e9/(res['cyc_pv']/tsc):7.1f} GF/s @TSC"
    )
    print(
        f"numpy:  QK^T {macs_qk/1e9/ref['qk_s']:7.1f} GF/s (OpenBLAS sgemm, 1 thread)"
    )
    print(
        f"        PV   {macs_pv/1e9/ref['pv_s']:7.1f} GF/s (OpenBLAS sgemm, 1 thread)"
    )
    print(f"        softmax {ref['softmax_s']*1e3:6.1f} ms")
    ours_sm = res["cyc_sm"] / tsc
    tot_ours = (res["cyc_qk"] + res["cyc_pv"] + res["cyc_sm"]) / tsc
    tot_np = ref["qk_s"] + ref["pv_s"] + ref["softmax_s"]
    print(f"ours:   softmax {ours_sm*1e3:6.1f} ms (SIMD fp32 poly-exp)")
    print(
        f"full head: ours {tot_ours*1e3:6.1f} ms vs numpy {tot_np*1e3:6.1f} ms "
        f"({tot_np/tot_ours:.2f}x)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
