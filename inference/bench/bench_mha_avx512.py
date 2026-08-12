#!/usr/bin/env python3
"""Multi-head (GQA) attention + decoder layer (bf16), the layer above single-head.

Real transformers use H query heads of head-dim D_head = D / H_q, with
grouped-query attention (GQA): H_kv < H_q key/value heads, each shared by
H_q / H_kv query heads. This is the same decoder layer as
bench_decoder_avx512.py but with the single-head attention replaced by:

    qkv  = x . Wqkv                      (S x 3D)
    q_h, k_g, v_g split into heads       (column blocks of D_head)
    for kv group g: pack K_g, V_g        (S x D_head)
    for query head h: scores_h = q_h . K_g^T   (S x S x D_head)
                      P_h = softmax(scores_h)
                      o_h = P_h . V_g          (S x D_head)
    o    = concat(o_h) . Wo              (S x D)

The total attention FLOPs equal the single-head case (H_q * D_head = D),
so the benchmark isolates the structural cost of many smaller GEMMs and the
per-head softmax, plus the GQA kv-sharing saving. Reference: numpy fp32
(OpenBLAS, 1 thread).

Usage:
  python inference/inference/results/bench_mha_avx512.py [--seq 512] [--d 128] [--heads 4]
      [--kv-heads 2] [--hm 4] [--layers 2]
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
from inference.llm_c_common import COMMON_C  # noqa: E402


def gen_c(seq: int, d: int, hm: int, layers: int, heads: int, kv_heads: int) -> str:
    S, D, H, L = seq, d, d * hm, layers
    HQ, HKV, DH = heads, kv_heads, d // heads
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
#define S @S@
#define D @D@
#define H @H@
#define L @L@
#define HQ @HQ@
#define HKV @HKV@
#define DH @DH@
static void layer_mha(
    float* xbuf, float* wbuf,
    uint16_t* b_q, uint16_t* b_k, uint16_t* b_v, uint16_t* b_qh,
    uint16_t* b_o, uint16_t* b_p, uint16_t* b_up,
    float* f_qkv, float* f_scores, float* f_o, float* o_h,
    float* f_up, float* f_g, float* fq, float* fk, float* fv,
    const int32_t* Bpq, const int32_t* Bpo, const int32_t* Bp1, const int32_t* Bp2,
    int32_t* Bpk, int32_t* Bpv) {
    for (int i = 0; i < S*D; i++) wbuf[i] = xbuf[i];
    layernorm(wbuf, S, D);
    to_bf16(wbuf, b_q, S*D);
    gemm(b_q, Bpq, f_qkv, S, 3*D, D);
    split_qkv(f_qkv, fq, fk, fv, S, D);
    /* kv groups: pack each group's K^T and V once */
    for (int g = 0; g < HKV; g++) {
        /* b_k = fk[:, g*DH:(g+1)*DH] as S x DH (strided cols); materialize contiguous */
        for (int r = 0; r < S; r++)
            for (int j = 0; j < DH; j++) b_k[r*DH + j] = f2b(fk[r*D + g*DH + j]);
        pack_b_bf16_32_T(b_k, Bpk + g*@BK1@, DH, S);
        for (int r = 0; r < S; r++)
            for (int j = 0; j < DH; j++) b_v[r*DH + j] = f2b(fv[r*D + g*DH + j]);
        pack_b_bf16_32(b_v, Bpv + g*@BV1@, S, DH);
        /* query heads that share this kv group */
        for (int h = g; h < HQ; h += HKV) {
            for (int r = 0; r < S; r++)
                for (int j = 0; j < DH; j++) b_qh[r*DH + j] = f2b(fq[r*D + h*DH + j]);
            gemm(b_qh, Bpk + g*@BK1@, f_scores, S, S, DH);
            for (int i = 0; i < S*S; i++) f_scores[i] *= 1.0f / sqrtf((float)DH);
            softmax(f_scores, S);
            to_bf16(f_scores, b_p, S*S);
            gemm(b_p, Bpv + g*@BV1@, o_h, S, DH, S);
            /* scatter into the concatenated output at the D row stride */
            for (int r = 0; r < S; r++)
                for (int j = 0; j < DH; j++)
                    f_o[r*D + h*DH + j] = o_h[r*DH + j];
        }
    }
    /* concat heads are already in f_o column blocks; out projection */
    to_bf16(f_o, b_o, S*D);
    gemm(b_o, Bpo, f_o, S, D, D);
    for (int i = 0; i < S*D; i++) xbuf[i] += f_o[i];
    /* LN2 + MLP (unchanged from the single-head layer) */
    for (int i = 0; i < S*D; i++) wbuf[i] = xbuf[i];
    layernorm(wbuf, S, D);
    to_bf16(wbuf, b_q, S*D);
    gemm(b_q, Bp1, f_up, S, H, D);
    gelu(f_up, S*H);
    to_bf16(f_up, b_up, S*H);
    gemm(b_up, Bp2, f_g, S, D, H);
    for (int i = 0; i < S*D; i++) xbuf[i] += f_g[i];
}
int main(void) {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(4, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
    static uint16_t Wqkv[@LD3@], Wo[@LD2@], W1[@LH@], W2[@LH@];
    static int32_t Bpq[@LQ@], Bpo[@LO@], Bp1[@LU@], Bp2[@LDW@];
    static int32_t Bpk[@KB@], Bpv[@VB@];
    static float xbuf[@S2@], wbuf[@S2@], f_qkv[@S3D@], f_scores[@SS@], f_o[@S2@];
    static float f_up[@SH@], f_g[@S2@], fq[@S2@], fk[@S2@], fv[@S2@];
    static float o_h[@SDH@];  /* per-head PV output, contiguous S x DH */
    static uint16_t b_q[@S2@], b_k[@SDH@], b_v[@SDH@], b_qh[@SDH@], b_o[@S2@], b_p[@SS@], b_up[@SH@];
    for (int l = 0; l < L; l++) {
        for (int i = 0; i < D*3*D; i++) { float v = ((i * 17 + l * 7) % 401) / 400.0f - 0.5f; Wqkv[l*D*3*D + i] = f2b(v * 0.05f); }
        for (int i = 0; i < D*D; i++) { float v = ((i * 7 + l * 3) % 1001) / 1000.0f - 0.5f; Wo[l*D*D + i] = f2b(v * 0.05f); }
        for (int i = 0; i < D*H; i++) { float v = ((i * 11 + l * 5) % 401) / 400.0f - 0.5f; W1[l*D*H + i] = f2b(v * 0.05f); }
        for (int i = 0; i < H*D; i++) { float v = ((i * 13 + l * 9) % 401) / 400.0f - 0.5f; W2[l*H*D + i] = f2b(v * 0.002f); }
    }
    for (int l = 0; l < L; l++) {
        pack_b_bf16_32(Wqkv + l*D*3*D, Bpq + l*@SQ@, D, 3*D);
        pack_b_bf16_32(Wo + l*D*D, Bpo + l*@SO@, D, D);
        pack_b_bf16_32(W1 + l*D*H, Bp1 + l*@SU@, D, H);
        pack_b_bf16_32(W2 + l*H*D, Bp2 + l*@SDW@, H, D);
    }
    for (int i = 0; i < S*D; i++)
        xbuf[i] = (float)b2f(f2b(0.05f * (float)((i % 1001) / 1000.0f - 0.5f)));
    uint64_t best = ~0ULL;
    for (int rep = 0; rep < 5; rep++) {
        uint64_t t0 = __rdtsc();
        for (int l = 0; l < L; l++)
            layer_mha(xbuf, wbuf, b_q, b_k, b_v, b_qh, b_o, b_p, b_up,
                      f_qkv, f_scores, f_o, o_h, f_up, f_g, fq, fk, fv,
                      Bpq + l*@SQ@, Bpo + l*@SO@, Bp1 + l*@SU@, Bp2 + l*@SDW@,
                      Bpk, Bpv);
        uint64_t t1 = __rdtsc();
        uint64_t d = t1 - t0;
        if (d < best) best = d;
    }
    /* clean pass for the correctness dump */
    for (int i = 0; i < S*D; i++)
        xbuf[i] = (float)b2f(f2b(0.05f * (float)((i % 1001) / 1000.0f - 0.5f)));
    for (int l = 0; l < L; l++)
        layer_mha(xbuf, wbuf, b_q, b_k, b_v, b_qh, b_o, b_p, b_up,
                  f_qkv, f_scores, f_o, o_h, f_up, f_g, fq, fk, fv,
                  Bpq + l*@SQ@, Bpo + l*@SO@, Bp1 + l*@SU@, Bp2 + l*@SDW@,
                  Bpk, Bpv);
    printf("RESULT cyc_total=%.0f\n", (double)best);
    FILE* f = fopen("state.bin", "wb");
    fwrite(xbuf, sizeof(float), S*D, f);
    fclose(f);
    return 0;
}
"""
    )
    return (
        t.replace("@S@", str(S))
        .replace("@D@", str(D))
        .replace("@H@", str(H))
        .replace("@L@", str(L))
        .replace("@HQ@", str(HQ))
        .replace("@HKV@", str(HKV))
        .replace("@DH@", str(DH))
        .replace("@S2@", str(S * D))
        .replace("@S3D@", str(S * 3 * D))
        .replace("@SS@", str(S * S))
        .replace("@SH@", str(S * H))
        .replace("@SD@", str(S * DH))
        .replace("@SDH@", str(S * DH))
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
        .replace("@KB@", str(HKV * (DH // 2) * S))
        .replace("@VB@", str(HKV * (S // 2) * DH))
        .replace("@BK1@", str((DH // 2) * S))
        .replace("@BV1@", str((S // 2) * DH))
    )


def run_c(seq: int, d: int, hm: int, layers: int, heads: int, kv_heads: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfile = td / "mha.c"
        cfile.write_text(gen_c(seq, d, hm, layers, heads, kv_heads))
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
                str(td / "mha"),
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
        out = subprocess.run([str(td / "mha")], capture_output=True, text=True, cwd=td)
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
    seq: int, d: int, hm: int, layers: int, heads: int, kv_heads: int
) -> dict:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import numpy as np

    def bf16(f):
        u = f.astype(np.float32).view(np.uint32)
        r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
        return (r.astype(np.uint32) << 16).view(np.float32)

    S, D, H, L = seq, d, d * hm, layers
    HQ, HKV, DH = heads, kv_heads, d // heads

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
        i = np.arange(H * D)
        W2.append(
            bf16(((((i * 13) + l * 9) % 401) / 400.0 - 0.5) * 0.002).reshape(H, D)
        )

    i = np.arange(S * D)
    x = bf16(((((i % 1001) / 1000.0 - 0.5) * 0.05)).astype(np.float32)).reshape(S, D)
    t0 = time.perf_counter()
    for l in range(L):
        xh = ln(x)
        qkv = bf16(xh) @ Wqkv[l]
        q, k, v = np.split(qkv, 3, axis=1)
        q, k, v = bf16(q), bf16(k), bf16(v)
        o = np.zeros((S, D), dtype=np.float32)
        for g in range(HKV):
            kg = bf16(k[:, g * DH : (g + 1) * DH])
            vg = bf16(v[:, g * DH : (g + 1) * DH])
            for h in range(g, HQ, HKV):
                qh = bf16(q[:, h * DH : (h + 1) * DH])
                scores = qh @ kg.T / np.sqrt(DH)
                e = np.exp(scores - scores.max(1, keepdims=True))
                p = e / e.sum(1, keepdims=True)
                o[:, h * DH : (h + 1) * DH] = bf16(p) @ vg
        o = bf16(o) @ Wo[l]
        x = x + o
        xh = ln(x)
        h = bf16(xh) @ W1[l]
        g = gelu_np(h)
        x = x + bf16(g) @ W2[l]
    t_layer = time.perf_counter() - t0
    return {"t_layer": t_layer, "state": x.astype(np.float32)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--hm", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    args = ap.parse_args()
    S, D, H, L = args.seq, args.d, args.d * args.hm, args.layers
    HQ, HKV, DH = args.heads, args.kv_heads, args.d // args.heads
    if HQ * DH != args.d:
        raise SystemExit("heads * (D/heads) != D: heads must divide D")
    if HQ % HKV != 0:
        raise SystemExit("kv-heads must divide heads (GQA)")
    tsc = 3.7e9

    print(
        f"bf16 {L}-layer GQA decoder: S={S} D={D} H_q={HQ} H_kv={HKV} "
        f"D_head={DH} (MLP {H})"
    )
    res = run_c(S, D, args.hm, L, HQ, HKV)
    ref = numpy_reference(S, D, args.hm, L, HQ, HKV)

    import numpy as np

    our = np.frombuffer(res["state"], dtype=np.float32).reshape(S, D)
    abs_err = np.abs(our - ref["state"]).max()
    print(
        f"final state max abs err vs fp32 numpy: {abs_err:.5f} "
        f"(state scale {np.abs(ref['state']).max():.3f})"
    )

    total_ms = res["cyc_total"] / tsc * 1e3
    print(f"ours:   {L} layers in {total_ms:6.2f} ms ({total_ms/L:5.2f} ms/layer)")
    print(
        f"        attention FLOPs {2*S*S*D*L/1e9:6.2f} GFLOP, "
        f"total layer FLOPs ~{2*S*D*(3*D + D + 2*H + S)*L/1e9:6.2f} GFLOP"
    )
    print(
        f"numpy:  {ref['t_layer']*1e3:6.2f} ms ({ref['t_layer']/(total_ms/1e3):5.1f}x slower)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
