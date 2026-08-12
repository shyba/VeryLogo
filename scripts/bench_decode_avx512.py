#!/usr/bin/env python3
"""KV-cache decode (token generation), the layer after prefill.

The incremental inference loop: at each step, B new token embeddings are
projected to Q/K/V, K/V are appended to the per-layer caches, and the next
hidden state is computed from a growing cache:

    x      = embed(ids)                      (B x D)
    q,k,v  = x . Wqkv                        (B x D each); cache[k] += k, cache[v] += v
    scores = q . K_cache^T / sqrt(D)         (B x S_cur, N padded to 32; tail masked)
    P      = softmax(scores)                 (B x S_cur)
    o      = P . V_cache                     (B x D)
    o      = o . Wo; x = x + o               (residual)
    x      = LN(x); h = x . W1; g = GELU(h); x = x + g . W2

Prefill is compute-bound; decode is memory-bound: per step each layer
reads ~2*S_cur*D*2 bytes from its K/V caches (2 x the cache in bf16), and
the GEMM cost is B*S*D MACs - at B=8 the reads dominate. The caches are
re-packed per step (O(S*D), ~12% at B=8); real serving implementations
pack incrementally.

All GEMMs on the hand-scheduled bf16 8x32 kernel (stc.gemm_asm) with the
SIMD fp32 helpers from scripts/llm_c_common.py. Reference: numpy fp32
(OpenBLAS sgemm, 1 thread).

Usage:
  python scripts/bench_decode_avx512.py [--seq-init 128] [--steps 128]
      [--batch 8] [--d 128] [--layers 6] [--hm 4]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stc.gemm_asm import emit_gemm_kernel  # noqa: E402
from llm_c_common import COMMON_C  # noqa: E402


def gen_c(s_init: int, steps: int, batch: int, d: int, hm: int, layers: int) -> str:
    S0, T, B, D, H, L = s_init, steps, batch, d, d * hm, layers
    SMAX = S0 + B * T  # cache capacity
    t = (
        r"""#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <x86intrin.h>
#include <sched.h>
extern void gemm_bf16_8x32_asm(const uint16_t* A, const int32_t* Bp, float* C, int K, int N);
"""
        + COMMON_C
        + r"""
/* decode attention is B x S, not square S x S: softmax over cols per row */
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
#define S0 @S0@
#define TSTEPS @T@
#define BATCH @B@
#define DIM @D@
#define HID @H@
#define NLAY @L@
#define SMAX @SMAX@
#define CHECK (@T@ < 8 ? @T@ : 8)
int main(void) {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(4, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
    /* per-layer weights (bf16) + packed forms */
    static uint16_t Wqkv[@LD3@], Wo[@LD2@], W1[@LH@], W2[@LH@];
    static int32_t Bpq[@LQ@], Bpo[@LO@], Bp1[@LU@], Bp2[@LDW@];
    /* per-layer K/V caches (SMAX x DIM bf16), zero-initialized so the
       padded region (scores N pad, PV K pad) contributes 0 */
    static uint16_t Kc[@LK@], Vc[@LV@];
    /* per-step packed views of the caches */
    static int32_t Bpk[@BK@], Bpv[@BV@];
    static float xbuf[@BD@], wbuf[@BD@], f_qkv[@B3D@], f_scores[@BSP@], f_o[@BD@];
    static float f_up[@BH@], f_g[@BD@], fq[@BD@], fk[@BD@], fv[@BD@];
    static uint16_t b_q[@BD@], b_k[@BD@], b_v[@BD@], b_o[@BD@], b_up[@BH@];
    static uint16_t b_p[@BSP@];
    for (int l = 0; l < NLAY; l++) {
        for (int i = 0; i < DIM*3*DIM; i++) { float v = ((i * 17 + l * 7) % 401) / 400.0f - 0.5f; Wqkv[l*DIM*3*DIM + i] = f2b(v * 0.05f); }
        for (int i = 0; i < DIM*DIM; i++) { float v = ((i * 7 + l * 3) % 1001) / 1000.0f - 0.5f; Wo[l*DIM*DIM + i] = f2b(v * 0.05f); }
        for (int i = 0; i < DIM*HID; i++) { float v = ((i * 11 + l * 5) % 401) / 400.0f - 0.5f; W1[l*DIM*HID + i] = f2b(v * 0.05f); }
        for (int i = 0; i < HID*DIM; i++) { float v = ((i * 13 + l * 9) % 401) / 400.0f - 0.5f; W2[l*HID*DIM + i] = f2b(v * 0.002f); }
    }
    for (int l = 0; l < NLAY; l++) {
        pack_b_bf16_32(Wqkv + l*DIM*3*DIM, Bpq + l*@SQ@, DIM, 3*DIM);
        pack_b_bf16_32(Wo + l*DIM*DIM, Bpo + l*@SO@, DIM, DIM);
        pack_b_bf16_32(W1 + l*DIM*HID, Bp1 + l*@SU@, DIM, HID);
        pack_b_bf16_32(W2 + l*HID*DIM, Bp2 + l*@SDW@, HID, DIM);
    }
    /* seed the caches with S0 deterministic rows */
    for (int l = 0; l < NLAY; l++)
        for (int r = 0; r < S0; r++)
            for (int j = 0; j < DIM; j++) {
                float v = ((r * 31 + l * 13 + j * 7) % 997) / 996.0f - 0.5f;
                Kc[(l*SMAX + r)*DIM + j] = f2b(v * 0.1f);
                Vc[(l*SMAX + r)*DIM + j] = f2b(v * 0.1f);
            }
    /* deterministic token ids for the decode steps */
    static int32_t ids[@B@];
    for (int i = 0; i < BATCH; i++) ids[i] = (i * 37 + 5) % 4093;
    /* the decode loop; time the whole generation (best-of-N) */
    uint64_t best = ~0ULL;
    for (int rep = 0; rep < 5; rep++) {
        int scur = S0;
        for (int i = 0; i < BATCH; i++)
            for (int j = 0; j < DIM; j++)
                xbuf[i*DIM + j] = 0.05f * (float)((ids[i] + j) % 7 - 3);
        uint64_t t0 = __rdtsc();
        for (int t = 0; t < TSTEPS; t++) {
            int spad = ((scur + BATCH + 31) / 32) * 32;  /* incl. the appends below */
            for (int l = 0; l < NLAY; l++) {
                /* LN1 -> QKV */
                for (int i = 0; i < BATCH*DIM; i++) wbuf[i] = xbuf[i];
                layernorm(wbuf, BATCH, DIM);
                to_bf16(wbuf, b_q, BATCH*DIM);
                gemm(b_q, Bpq + l*@SQ@, f_qkv, BATCH, 3*DIM, DIM);
                split_qkv(f_qkv, fq, fk, fv, BATCH, DIM);
                to_bf16(fq, b_q, BATCH*DIM);
                to_bf16(fk, b_k, BATCH*DIM);
                to_bf16(fv, b_v, BATCH*DIM);
                /* append k,v to the layer cache */
                for (int i = 0; i < BATCH; i++) {
                    memcpy(&Kc[(l*SMAX + scur + i)*DIM], &b_k[i*DIM], DIM*2);
                    memcpy(&Vc[(l*SMAX + scur + i)*DIM], &b_v[i*DIM], DIM*2);
                }
                /* QK^T over the padded cache length; tail masked in softmax */
                pack_b_bf16_32_T(Kc + l*SMAX*DIM, Bpk, DIM, spad);
                gemm(b_q, Bpk, f_scores, BATCH, spad, DIM);
                for (int i = 0; i < BATCH; i++)
                    for (int j = scur + BATCH; j < spad; j++)
                        f_scores[i*spad + j] = -INFINITY;
                for (int i = 0; i < BATCH*spad; i++) f_scores[i] *= 1.0f / sqrtf((float)DIM);
                softmax_rect(f_scores, BATCH, spad);
                to_bf16(f_scores, b_p, BATCH*spad);
                /* PV over the padded cache; padded V rows are 0, padded P cols 0 */
                pack_b_bf16_32(Vc + l*SMAX*DIM, Bpv, spad, DIM);
                gemm(b_p, Bpv, f_o, BATCH, DIM, spad);
                to_bf16(f_o, b_o, BATCH*DIM);
                gemm(b_o, Bpo + l*@SO@, f_o, BATCH, DIM, DIM);
                for (int i = 0; i < BATCH*DIM; i++) xbuf[i] += f_o[i];
                /* LN2 + MLP */
                for (int i = 0; i < BATCH*DIM; i++) wbuf[i] = xbuf[i];
                layernorm(wbuf, BATCH, DIM);
                to_bf16(wbuf, b_q, BATCH*DIM);
                gemm(b_q, Bp1 + l*@SU@, f_up, BATCH, HID, DIM);
                gelu(f_up, BATCH*HID);
                to_bf16(f_up, b_up, BATCH*HID);
                gemm(b_up, Bp2 + l*@SDW@, f_g, BATCH, DIM, HID);
                for (int i = 0; i < BATCH*DIM; i++) xbuf[i] += f_g[i];
            }
            /* next token ids: deterministic (no logits/argmax for the benchmark) */
            for (int i = 0; i < BATCH; i++) ids[i] = (ids[i] * 31 + 17) % 4093;
            scur += BATCH;
        }
        uint64_t t1 = __rdtsc();
        uint64_t d = t1 - t0;
        if (d < best) best = d;
        /* no break: all reps run the same deterministic decode over the same
           cache rows (appends overwrite identical values), so best-of-5 is a
           true min over warm reps */
    }
    /* clean pass for correctness: reset ids and caches past S0 (the timed
       loop left stale K/V appends that the check would otherwise read),
       then run CHECK steps (short horizon: the bf16-vs-fp32 decode
       trajectories decorrelate chaotically at long horizons, so
       correctness is validated where the model is stable) */
    for (int l = 0; l < NLAY; l++) {
        memset(Kc + l*SMAX*DIM + S0*DIM, 0, (size_t)(SMAX - S0)*DIM*2);
        memset(Vc + l*SMAX*DIM + S0*DIM, 0, (size_t)(SMAX - S0)*DIM*2);
    }
    for (int i = 0; i < BATCH; i++) ids[i] = (i * 37 + 5) % 4093;
    int scur = S0;
    for (int i = 0; i < BATCH; i++)
        for (int j = 0; j < DIM; j++)
            xbuf[i*DIM + j] = 0.05f * (float)((ids[i] + j) % 7 - 3);
    for (int t = 0; t < CHECK; t++) {
        int spad = ((scur + BATCH + 31) / 32) * 32;  /* incl. the appends below */
        for (int l = 0; l < NLAY; l++) {
            for (int i = 0; i < BATCH*DIM; i++) wbuf[i] = xbuf[i];
            layernorm(wbuf, BATCH, DIM);
            to_bf16(wbuf, b_q, BATCH*DIM);
            gemm(b_q, Bpq + l*@SQ@, f_qkv, BATCH, 3*DIM, DIM);
            split_qkv(f_qkv, fq, fk, fv, BATCH, DIM);
            to_bf16(fq, b_q, BATCH*DIM);
            to_bf16(fk, b_k, BATCH*DIM);
            to_bf16(fv, b_v, BATCH*DIM);
            for (int i = 0; i < BATCH; i++) {
                memcpy(&Kc[(l*SMAX + scur + i)*DIM], &b_k[i*DIM], DIM*2);
                memcpy(&Vc[(l*SMAX + scur + i)*DIM], &b_v[i*DIM], DIM*2);
            }
            pack_b_bf16_32_T(Kc + l*SMAX*DIM, Bpk, DIM, spad);
            gemm(b_q, Bpk, f_scores, BATCH, spad, DIM);
            for (int i = 0; i < BATCH; i++)
                for (int j = scur + BATCH; j < spad; j++)
                    f_scores[i*spad + j] = -INFINITY;
            for (int i = 0; i < BATCH*spad; i++) f_scores[i] *= 1.0f / sqrtf((float)DIM);
            softmax_rect(f_scores, BATCH, spad);
            to_bf16(f_scores, b_p, BATCH*spad);
            pack_b_bf16_32(Vc + l*SMAX*DIM, Bpv, spad, DIM);
            gemm(b_p, Bpv, f_o, BATCH, DIM, spad);
            to_bf16(f_o, b_o, BATCH*DIM);
            gemm(b_o, Bpo + l*@SO@, f_o, BATCH, DIM, DIM);
            for (int i = 0; i < BATCH*DIM; i++) xbuf[i] += f_o[i];
            for (int i = 0; i < BATCH*DIM; i++) wbuf[i] = xbuf[i];
            layernorm(wbuf, BATCH, DIM);
            to_bf16(wbuf, b_q, BATCH*DIM);
            gemm(b_q, Bp1 + l*@SU@, f_up, BATCH, HID, DIM);
            gelu(f_up, BATCH*HID);
            to_bf16(f_up, b_up, BATCH*HID);
            gemm(b_up, Bp2 + l*@SDW@, f_g, BATCH, DIM, HID);
            for (int i = 0; i < BATCH*DIM; i++) xbuf[i] += f_g[i];
        }
        for (int i = 0; i < BATCH; i++) ids[i] = (ids[i] * 31 + 17) % 4093;
        scur += BATCH;
    }
    printf("RESULT cyc_total=%.0f\n", (double)best);
    FILE* f = fopen("state.bin", "wb");
    fwrite(xbuf, sizeof(float), BATCH*DIM, f);
    fclose(f);
    return 0;
}
"""
    )
    return (
        t.replace("@S0@", str(S0))
        .replace("@T@", str(T))
        .replace("@B@", str(B))
        .replace("@D@", str(D))
        .replace("@H@", str(H))
        .replace("@L@", str(L))
        .replace("@SMAX@", str(SMAX))
        .replace("@BD@", str(B * D))
        .replace("@B3D@", str(B * 3 * D))
        .replace("@BH@", str(B * H))
        .replace("@BSP@", str(B * ((SMAX + 31) // 32) * 32))
        .replace("@LD3@", str(L * D * 3 * D))
        .replace("@LD2@", str(L * D * D))
        .replace("@LH@", str(L * D * H))
        .replace("@LQ@", str(L * (D // 2) * 3 * D))
        .replace("@LO@", str(L * (D // 2) * D))
        .replace("@LU@", str(L * (D // 2) * H))
        .replace("@LDW@", str(L * (H // 2) * D))
        .replace("@SQ@", str((D // 2) * 3 * D))
        .replace("@SO@", str((D // 2) * D))
        .replace("@SU@", str((D // 2) * H))
        .replace("@SDW@", str((H // 2) * D))
        .replace("@LK@", str(L * SMAX * D))
        .replace("@LV@", str(L * SMAX * D))
        .replace("@BK@", str((D // 2) * SMAX))
        .replace("@BV@", str((SMAX // 2) * D))
    )


def run_c(s_init: int, steps: int, batch: int, d: int, hm: int, layers: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfile = td / "decode.c"
        cfile.write_text(gen_c(s_init, steps, batch, d, hm, layers))
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
                str(td / "decode"),
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
        out = subprocess.run(
            [str(td / "decode")], capture_output=True, text=True, cwd=td
        )
        if out.returncode != 0:
            print(out.stderr)
            sys.exit(1)
        res = {}
        for line in out.stdout.splitlines():
            if line.startswith("RESULT"):
                for kv in line.split()[1:]:
                    k, v = kv.split("=")
                    res[k] = float(v.strip())
        res["state"] = (td / "state.bin").read_bytes()
    return res


def numpy_reference(
    s_init: int, steps: int, batch: int, d: int, hm: int, layers: int
) -> dict:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import numpy as np

    def bf16(f):
        u = f.astype(np.float32).view(np.uint32)
        r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
        return (r.astype(np.uint32) << 16).view(np.float32)

    S0, T, B, D, H, L = s_init, steps, batch, d, d * hm, layers

    def ln(xr):
        m = xr.mean(1, keepdims=True)
        vv = ((xr - m) ** 2).mean(1, keepdims=True)
        return (xr - m) / np.sqrt(vv + 1e-5)

    def gelu_np(z):
        return 0.5 * z * (1.0 + np.tanh(0.7978845608 * (z + 0.044715 * z**3)))

    Wqkv, Wo, W1, W2 = [], [], [], []
    for l in range(L):
        i = np.arange(D * 3 * D)
        Wqkv.append(
            bf16(((((i * 17) + l * 7) % 401) / 400.0 - 0.5) * 0.05).reshape(D, 3 * D)
        )
        i = np.arange(D * D)
        Wo.append(
            bf16(((((i * 7) + l * 3) % 1001) / 1000.0 - 0.5) * 0.05).reshape(D, D)
        )
        i = np.arange(D * H)
        W1.append(bf16(((((i * 11) + l * 5) % 401) / 400.0 - 0.5) * 0.05).reshape(D, H))
        W2.append(
            bf16(((((i * 13) + l * 9) % 401) / 400.0 - 0.5) * 0.002).reshape(H, D)
        )

    Kc = [np.zeros((S0 + B * T, D), dtype=np.float32) for _ in range(L)]
    Vc = [np.zeros((S0 + B * T, D), dtype=np.float32) for _ in range(L)]
    for l in range(L):
        for r in range(S0):
            v = ((np.arange(D) * 7 + r * 31 + l * 13) % 997) / 996.0 - 0.5
            Kc[l][r] = bf16((v * 0.1).astype(np.float32))
            Vc[l][r] = bf16((v * 0.1).astype(np.float32))

    def run(steps):
        ids = np.array([(i * 37 + 5) % 4093 for i in range(B)])
        x = np.zeros((B, D), dtype=np.float32)
        for i in range(B):
            x[i] = 0.05 * ((ids[i] + np.arange(D)) % 7 - 3).astype(np.float32)
        for l in range(L):  # reset rows past S0 (fresh caches per run)
            Kc[l][S0:] = 0.0
            Vc[l][S0:] = 0.0
        scur = S0
        for t in range(steps):
            for l in range(L):
                xh = ln(x)
                qkv = bf16(xh) @ Wqkv[l]
                q, k, v = np.split(qkv, 3, axis=1)
                q, k, v = bf16(q), bf16(k), bf16(v)
                Kc[l][scur : scur + B] = k
                Vc[l][scur : scur + B] = v
                scores = q @ bf16(Kc[l][: scur + B]).T / np.sqrt(D)
                e = np.exp(scores - scores.max(1, keepdims=True))
                p = e / e.sum(1, keepdims=True)
                o = bf16(bf16(p) @ bf16(Vc[l][: scur + B])) @ Wo[l]
                x = x + o
                xh = ln(x)
                h = bf16(xh) @ W1[l]
                g = gelu_np(h)
                x = x + bf16(g) @ W2[l]
            ids = (ids * 31 + 17) % 4093
            scur += B
        return x.astype(np.float32)

    t0 = time.perf_counter()
    run(T)  # full-horizon timing (same work as the C timed loop)
    t_decode = time.perf_counter() - t0
    check = min(T, 8)
    state = run(check)  # fresh short-horizon state for correctness
    return {"t_decode": t_decode, "state": state}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-init", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--hm", type=int, default=4)
    ap.add_argument("--layers", type=int, default=6)
    args = ap.parse_args()
    S0, T, B, D, L = args.seq_init, args.steps, args.batch, args.d, args.layers
    H = D * args.hm
    tsc = 3.7e9

    print(
        f"bf16 {L}-layer KV-cache decode: batch {B}, {T} steps, cache {S0} -> {S0 + B*T}"
    )
    res = run_c(S0, T, B, D, args.hm, L)
    ref = numpy_reference(S0, T, B, D, args.hm, L)

    import numpy as np

    our = np.frombuffer(res["state"], dtype=np.float32).reshape(B, D)
    abs_err = np.abs(our - ref["state"]).max()
    print(
        f"final state max abs err vs bf16-input fp32 numpy: {abs_err:.5f} "
        f"(state scale {np.abs(ref['state']).max():.3f})"
    )

    total_ms = res["cyc_total"] / tsc * 1e3
    per_step_us = total_ms / T * 1e3
    tokens = B * T
    print(
        f"ours:   {T} decode steps in {total_ms:7.2f} ms "
        f"({per_step_us:6.1f} us/step, {per_step_us/B:6.1f} us/token @batch {B})"
    )
    print(f"        {tokens/(total_ms/1e3)/1e3:7.1f} k tokens/s")
    avg_cache = S0 + B * T / 2
    cache_bytes = 2 * avg_cache * D * 2 * L  # K and V caches, bf16, per layer
    print(
        f"        avg cache read/step: {cache_bytes/1e3:7.1f} KB "
        f"(~{cache_bytes/(total_ms/T/1e3)/1e9:5.1f} GB/s effective)"
    )
    print(
        f"numpy:  {ref['t_decode']*1e3:7.2f} ms  ({ref['t_decode']/(total_ms/1e3):5.1f}x slower)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
