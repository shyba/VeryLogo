#!/usr/bin/env python3
"""Full transformer prefill forward pass (bf16), the layer above the decoder.

The complete decoder-only LLM forward (prefill / prompt processing):
  x      = embed(token ids)          (gather VxD bf16 embedding table)
  for l in L:  decoder layer l       (LN, fused QKV, attention, out-proj,
                                     residual, LN, MLP, residual)
  x      = layernorm(x)
  logits = x . E^T                   (tied embeddings, LM head over V)

All GEMMs run on the hand-scheduled bf16 micro-kernel (stc.gemm_asm) with
the SIMD fp32 layernorm/GELU/softmax from inference/llm_c_common.py (shared
with the decoder benchmark, property-tested). Reference: numpy fp32
(OpenBLAS sgemm, 1 thread) on the same core.

Usage:
  python inference/bench/bench_llm_prefill_avx512.py [--seq 512] [--d 128]
      [--layers 6] [--vocab 16384]
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


def gen_c(seq: int, d: int, hm: int, layers: int, vocab: int) -> str:
    S, D, H, L, V = seq, d, d * hm, layers, vocab
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
static void layer(
    float* xbuf, float* wbuf,
    uint16_t* b_q, uint16_t* b_k, uint16_t* b_v, uint16_t* b_o, uint16_t* b_p,
    uint16_t* b_up, float* f_qkv, float* f_scores, float* f_o,
    float* f_up, float* f_g, float* fq, float* fk, float* fv,
    const int32_t* Bpq, const int32_t* Bpo, const int32_t* Bp1, const int32_t* Bp2,
    int32_t* Bpk, int32_t* Bpv, int S, int D, int H) {
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
    softmax(f_scores, S);
    to_bf16(f_scores, b_p, S*S);
    gemm(b_p, Bpv, f_o, S, D, S);
    to_bf16(f_o, b_o, S*D);
    gemm(b_o, Bpo, f_o, S, D, D);
    for (int i = 0; i < S*D; i++) xbuf[i] += f_o[i];
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
    const int S = @S@, D = @D@, H = @H@, L = @L@, V = @V@;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(4, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
    /* embedding table (bf16) and per-layer weights (bf16) */
    static uint16_t E[@VD@], Wqkv[@L3D@], Wo[@LD2@], W1[@LH@], W2[@LH@];
    static int32_t Bp_head[@BH@];
    static float xbuf[@S2@], wbuf[@S2@], f_qkv[@S3D@], f_scores[@SS@], f_o[@S2@];
    static float f_up[@SH@], f_g[@SH@], fq[@S2@], fk[@S2@], fv[@S2@];
    static uint16_t b_q[@S2@], b_k[@S2@], b_v[@S2@], b_o[@S2@], b_p[@SS@], b_up[@SH@];
    static int32_t Bpk[@BK@], Bpv[@BV@];
    static uint16_t b_x[@S2@];
    static float logits[@SV@];
    for (int i = 0; i < V*D; i++) { float v = ((i * 3) % 4093) / 4092.0f - 0.5f; E[i] = f2b(v * 0.1f); }
    for (int l = 0; l < L; l++) {
        for (int i = 0; i < D*3*D; i++) { float v = ((i * 17 + l * 7) % 401) / 400.0f - 0.5f; Wqkv[l*D*3*D + i] = f2b(v * 0.05f); }
        for (int i = 0; i < D*D; i++) { float v = ((i * 7 + l * 3) % 1001) / 1000.0f - 0.5f; Wo[l*D*D + i] = f2b(v * 0.05f); }
        for (int i = 0; i < D*H; i++) { float v = ((i * 11 + l * 5) % 401) / 400.0f - 0.5f; W1[l*D*H + i] = f2b(v * 0.05f); }
        for (int i = 0; i < H*D; i++) { float v = ((i * 13 + l * 9) % 401) / 400.0f - 0.5f; W2[l*H*D + i] = f2b(v * 0.05f); }
    }
    /* pack all L layers' weights (they differ per layer), + the tied
       embedding transpose for the LM head */
    static int32_t Bpq_all[@LQ@], Bpo_all[@LO@], Bp1_all[@LU@], Bp2_all[@LDW@];
    for (int l = 0; l < L; l++) {
        pack_b_bf16_32(Wqkv + l*D*3*D, Bpq_all + l*@SQ@, D, 3*D);
        pack_b_bf16_32(Wo + l*D*D, Bpo_all + l*@SO@, D, D);
        pack_b_bf16_32(W1 + l*D*H, Bp1_all + l*@SU@, D, H);
        pack_b_bf16_32(W2 + l*H*D, Bp2_all + l*@SDW@, H, D);
    }
    pack_b_bf16_32_T(E, Bp_head, D, V);
    /* embed: xbuf = E[ids]; ids deterministic */
    static int32_t ids[@S@];
    for (int i = 0; i < S; i++) ids[i] = (i * 7) % V;
    for (int i = 0; i < S; i++)
        for (int j = 0; j < D; j++)
            xbuf[i*D + j] = (float)b2f(E[ids[i]*D + j]);
    /* timing, best-of-N: embed gather runs before t0 (it is tiny and not part
       of the GEMM-bound work); cyc_layers covers L layers + final LN + bf16
       conversion; cyc_total additionally covers the LM-head GEMM. */
    uint64_t embed_cyc = 0; (void)embed_cyc;
    uint64_t best_total = ~0ULL, best_layers = ~0ULL;
    for (int r = 0; r < 5; r++) {
        for (int i = 0; i < S; i++)
            for (int j = 0; j < D; j++)
                xbuf[i*D + j] = (float)b2f(E[ids[i]*D + j]);
        uint64_t t0 = __rdtsc();
        for (int l = 0; l < L; l++)
            layer(xbuf, wbuf, b_q, b_k, b_v, b_o, b_p, b_up,
                  f_qkv, f_scores, f_o, f_up, f_g, fq, fk, fv,
                  Bpq_all + l*@SQ@, Bpo_all + l*@SO@,
                  Bp1_all + l*@SU@, Bp2_all + l*@SDW@,
                  Bpk, Bpv, S, D, H);
        for (int i = 0; i < S*D; i++) wbuf[i] = xbuf[i];
        layernorm(wbuf, S, D);
        to_bf16(wbuf, b_x, S*D);
        uint64_t t1 = __rdtsc();
        uint64_t d = t1 - t0;
        if (d < best_layers) best_layers = d;
        gemm(b_x, Bp_head, logits, S, V, D);  /* LM head over tied embeddings */
        uint64_t t2 = __rdtsc();
        d = t2 - t0;
        if (d < best_total) best_total = d;
    }
    /* clean single pass for the correctness dump (logits) */
    for (int i = 0; i < S; i++)
        for (int j = 0; j < D; j++)
            xbuf[i*D + j] = (float)b2f(E[ids[i]*D + j]);
    for (int l = 0; l < L; l++)
        layer(xbuf, wbuf, b_q, b_k, b_v, b_o, b_p, b_up,
              f_qkv, f_scores, f_o, f_up, f_g, fq, fk, fv,
              Bpq_all + l*@SQ@, Bpo_all + l*@SO@,
              Bp1_all + l*@SU@, Bp2_all + l*@SDW@,
              Bpk, Bpv, S, D, H);
    for (int i = 0; i < S*D; i++) wbuf[i] = xbuf[i];
    layernorm(wbuf, S, D);
    to_bf16(wbuf, b_x, S*D);
    gemm(b_x, Bp_head, logits, S, V, D);
    printf("RESULT cyc_layers=%.0f cyc_total=%.0f\n", (double)best_layers, (double)best_total);
    FILE* f = fopen("logits.bin", "wb");
    fwrite(logits, sizeof(float), S*V, f);
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
        .replace("@V@", str(V))
        .replace("@S2@", str(S * D))
        .replace("@SS@", str(S * S))
        .replace("@SH@", str(S * H))
        .replace("@S3D@", str(S * 3 * D))
        .replace("@SV@", str(S * V))
        .replace("@VD@", str(V * D))
        .replace("@L3D@", str(L * D * 3 * D))
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
        .replace("@BK@", str((D // 2) * S))
        .replace("@BV@", str((S // 2) * D))
        .replace("@BH@", str((D // 2) * V))
    )


def run_c(seq: int, d: int, hm: int, layers: int, vocab: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfile = td / "prefill.c"
        cfile.write_text(gen_c(seq, d, hm, layers, vocab))
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
                str(td / "prefill"),
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
            [str(td / "prefill")], capture_output=True, text=True, cwd=td
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
        res["logits"] = (td / "logits.bin").read_bytes()
    return res


def numpy_reference(seq: int, d: int, hm: int, layers: int, vocab: int) -> dict:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import numpy as np

    def bf16(f):
        u = f.astype(np.float32).view(np.uint32)
        r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
        return (r.astype(np.uint32) << 16).view(np.float32)

    S, D, H, L, V = seq, d, d * hm, layers, vocab
    i = np.arange(V * D)
    E = bf16((((i * 3) % 4093) / 4092.0 - 0.5) * 0.1).reshape(V, D)
    ids = (np.arange(S) * 7) % V

    def gelu_np(z):
        return 0.5 * z * (1.0 + np.tanh(0.7978845608 * (z + 0.044715 * z**3)))

    def ln(xr):
        m = xr.mean(1, keepdims=True)
        vv = ((xr - m) ** 2).mean(1, keepdims=True)
        return (xr - m) / np.sqrt(vv + 1e-5)

    t0 = time.perf_counter()
    x = bf16(E[ids]).astype(np.float32)
    for l in range(L):
        i2 = np.arange(D * 3 * D)
        wqkv = bf16(((((i2 * 17) + l * 7) % 401) / 400.0 - 0.5) * 0.05).reshape(
            D, 3 * D
        )
        i3 = np.arange(D * D)
        wo = bf16(((((i3 * 7) + l * 3) % 1001) / 1000.0 - 0.5) * 0.05).reshape(D, D)
        i4 = np.arange(D * H)
        w1 = bf16(((((i4 * 11) + l * 5) % 401) / 400.0 - 0.5) * 0.05).reshape(D, H)
        w2 = bf16(((((i4 * 13) + l * 9) % 401) / 400.0 - 0.5) * 0.05).reshape(H, D)
        xh = ln(x)
        qkv = bf16(xh) @ wqkv
        q, k, vv = np.split(qkv, 3, axis=1)
        scores = bf16(q) @ bf16(k).T / np.sqrt(D)
        e = np.exp(scores - scores.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
        o = bf16(bf16(p) @ bf16(vv)) @ wo
        x = x + o
        xh = ln(x)
        h = bf16(xh) @ w1
        g = gelu_np(h)
        x = x + bf16(g) @ w2
    xh = ln(x)
    logits = bf16(xh) @ bf16(E).T
    t_layer = time.perf_counter() - t0
    return {"t_layer": t_layer, "logits": logits.astype(np.float32)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--hm", type=int, default=4, help="hidden multiplier (H = D*hm)")
    args = ap.parse_args()
    S, D, H, L, V = args.seq, args.d, args.d * args.hm, args.layers, args.vocab
    tsc = 3.7e9

    print(f"bf16 {L}-layer prefill: S={S} D={D} H={H} V={V}")
    res = run_c(S, D, args.hm, L, V)
    ref = numpy_reference(S, D, args.hm, L, V)

    import numpy as np

    our = np.frombuffer(res["logits"], dtype=np.float32).reshape(S, V)
    abs_err = np.abs(our - ref["logits"]).max()
    print(
        f"logits max abs err vs bf16-input fp32 numpy: {abs_err:.5f} "
        f"(logit scale {np.abs(ref['logits']).max():.3f})"
    )

    layers_ms = res["cyc_layers"] / tsc * 1e3
    total_ms = res["cyc_total"] / tsc * 1e3
    print(
        f"ours:   {L} layers {layers_ms:6.2f} ms ({layers_ms/L:6.2f} ms/layer) + head "
        f"{total_ms-layers_ms:6.2f} ms = {total_ms:6.2f} ms total"
    )
    print(f"        {S/total_ms*1e3:8.1f} tokens/s  ({total_ms/S:5.3f} ms/token)")
    print(
        f"numpy:  {ref['t_layer']*1e3:6.2f} ms total  ({ref['t_layer']/(total_ms/1e3):5.1f}x slower)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
