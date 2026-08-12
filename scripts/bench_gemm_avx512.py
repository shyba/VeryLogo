#!/usr/bin/env python3
"""Zen 5 GEMM micro-kernels, designed from the Agner machine model.

Small core-AI component: INT8 (VPDPBUSD, 128 MACs/cyc peak) and BF16
(VDPBF16PS, 64 MACs/cyc peak) dense matmuls. The micro-kernel structure is
derived from stc/aggen.py (Agner Zen 5 AVX-512 table + locally measured
corrections):

- FMA-class ops run 2/cyc on P01 (Agner rt 0.5; measured 2.0/cyc).
- Latency 4 (VNNI) / 6 (BF16) -> need >= 8 / >= 12 independent accumulator
  chains per stream to reach 2/cyc; the MR x NR tile provides MR of them.
- Broadcast-loads of A operands run on the 2 load pipes concurrently with
  P01, so a tile is optimal (peak MACs/cyc) iff its FMA-ops/cycle does not
  exceed 2/cyc AND its loads/cycle stay under 2/cyc; among such tiles the
  largest wins (best amortization). See stc.aggen.best_tile.

Layouts (K-major, N-interleaved):
- A: row-major int8/bf16[M][K]; each K-chunk broadcasts a 32-bit A[i][k0..]
  group to all lanes.
- B packed once (outside the timed loop): per N-tile t, per K-chunk c,
  NR dwords; dword j = B[k0..k0+3][t*NR+j] (INT8) or the bf16 pair
  B[32c..32c+1][t*NR+j] (BF16). Lane j then accumulates output (i, t*NR+j).

Usage:
  python scripts/bench_gemm_avx512.py [--tiles 16x32] [--reps 7]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stc.aggen import get_machine  # noqa: E402


def int8_kernel(mr: int, nr: int) -> str:
    nb = nr // 16
    b_loads = "\n".join(
        "        __m512i b%d = _mm512_loadu_si512((const __m512i*)(Bp + c*%d + %d));"
        % (j, nr, j * 16)
        for j in range(nb)
    )
    decl = "\n".join(
        "    __m512i acc%d_%d = _mm512_setzero_si512();" % (i, j)
        for i in range(mr)
        for j in range(nb)
    )
    rows = []
    for i in range(mr):
        rows.append(
            "        __m512i a%d = _mm512_set1_epi32(*(const uint32_t*)(A + %d*K + 4*c));"
            % (i, i)
        )
        for j in range(nb):
            rows.append(
                "        acc%d_%d = _mm512_dpbusd_epi32(acc%d_%d, a%d, b%d);"
                % (i, j, i, j, i, j)
            )
    # K-loop unrolled by 2 (halves loop-control overhead): rename temps
    b_loads2 = "\n".join(
        "        __m512i b2_%d = _mm512_loadu_si512((const __m512i*)(Bp + (c+1)*%d + %d));"
        % (j, nr, j * 16)
        for j in range(nb)
    )
    rows2 = []
    for r in rows:
        r2 = (
            r.replace("__m512i a", "__m512i a2_").replace(
                "a%d, b%d" % (r_i, r_j), "a2_%d, b2_%d" % (r_i, r_j)
            )
            if False
            else r
        )
        rows2.append(r2)
    # rebuild rows2 properly: rename a -> a2_ and b%d -> b2_%d in dpbusd lines
    rows2 = []
    for i in range(mr):
        rows2.append(
            "        __m512i a2_%d = _mm512_set1_epi32(*(const uint32_t*)(A + %d*K + 4*(c+1)));"
            % (i, i)
        )
        for j in range(nb):
            rows2.append(
                "        acc%d_%d = _mm512_dpbusd_epi32(acc%d_%d, a2_%d, b2_%d);"
                % (i, j, i, j, i, j)
            )
    body = "\n".join([b_loads, "\n".join(rows), b_loads2, "\n".join(rows2)])
    store = "\n".join(
        "    _mm512_storeu_si512((__m512i*)(C + %d*N + %d*16), acc%d_%d);"
        % (i, j, i, j)
        for i in range(mr)
        for j in range(nb)
    )
    return """static void gemm_i8_%dx%d(const uint8_t* A, const int32_t* Bp, int32_t* C, int K, int N) {
%s
    for (int c = 0; c < K/4; c += 2) {
%s
    }
%s
}
""" % (
        mr,
        nr,
        decl,
        body,
        store,
    )


def bf16_kernel(mr: int, nr: int) -> str:
    nb = nr // 16
    b_loads = "\n".join(
        "        __m512i b%d = _mm512_loadu_si512((const __m512i*)(Bp + c*%d + %d));"
        % (j, nr, j * 16)
        for j in range(nb)
    )
    decl = "\n".join(
        "    __m512 acc%d_%d = _mm512_setzero_ps();" % (i, j)
        for i in range(mr)
        for j in range(nb)
    )
    rows = []
    for i in range(mr):
        rows.append(
            "        __m512i a%d = _mm512_set1_epi32(*(const uint32_t*)(A + %d*K + 2*c));"
            % (i, i)
        )
        for j in range(nb):
            rows.append(
                "        acc%d_%d = _mm512_dpbf16_ps(acc%d_%d, bh_from_i(a%d), bh_from_i(b%d));"
                % (i, j, i, j, i, j)
            )
    store = "\n".join(
        "    _mm512_storeu_ps(C + %d*N + %d*16, acc%d_%d);" % (i, j, i, j)
        for i in range(mr)
        for j in range(nb)
    )
    return """static void gemm_bf16_%dx%d(const uint16_t* A, const int32_t* Bp, float* C, int K, int N) {
%s
    for (int c = 0; c < K/2; c++) {
%s
%s
    }
%s
}
""" % (
        mr,
        nr,
        decl,
        b_loads,
        "\n".join(rows),
        store,
    )


def pack_b_i8(nr: int) -> str:
    return """static void pack_b_i8_%d(const int8_t* B, int32_t* Bp, int K, int N) {
    for (int n0 = 0; n0 < N; n0 += %d)
        for (int c = 0; c < K/4; c++)
            for (int j = 0; j < %d; j++) {
                uint32_t d = 0; int col = n0 + j;
                for (int t = 0; t < 4; t++)
                    d |= (uint32_t)(uint8_t)B[(4*c + t)*N + col] << (8*t);
                Bp[((n0/%d)*(K/4) + c)*%d + j] = d;
            }
}""" % (
        nr,
        nr,
        nr,
        nr,
        nr,
    )


def pack_b_bf16(nr: int) -> str:
    return """static void pack_b_bf16_%d(const uint16_t* B, int32_t* Bp, int K, int N) {
    for (int n0 = 0; n0 < N; n0 += %d)
        for (int c = 0; c < K/2; c++)
            for (int j = 0; j < %d; j++) {
                uint32_t d = 0; int col = n0 + j;
                for (int t = 0; t < 2; t++)
                    d |= (uint32_t)B[(2*c + t)*N + col] << (16*t);
                Bp[((n0/%d)*(K/2) + c)*%d + j] = d;
            }
}""" % (
        nr,
        nr,
        nr,
        nr,
        nr,
    )


def gen_c(
    tiles: list[tuple[int, int]], pred: dict, reps: int = 7, size: int = 64
) -> str:
    parts = []
    parts.append(
        "#include <stdint.h>\n#include <immintrin.h>\n#include <stdio.h>\n"
        "#include <math.h>\n#include <stdlib.h>\n#include <string.h>\n"
    )
    parts.append(
        "\n".join(
            "#define pred_i8_%dx%d %d\n#define pred_bf16_%dx%d %d"
            % (mr, nr, pred[(mr, nr)][0], mr, nr, pred[(mr, nr)][1])
            for mr, nr in tiles
        )
    )
    parts.append("\n".join(int8_kernel(mr, nr) for mr, nr in tiles))
    parts.append(
        "static inline __m512bh bh_from_i(__m512i v) "
        "{ union { __m512i i; __m512bh h; } u; u.i = v; return u.h; }\n"
        "static inline uint16_t f2b(float f) {\n"
        "    uint32_t u; memcpy(&u, &f, 4);\n"
        "    uint32_t r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16;\n"
        "    return (uint16_t)r;\n"
        "}\n"
        "static inline float b2f(uint16_t b) {\n"
        "    uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f;\n"
        "}\n"
    )
    parts.append("\n".join(bf16_kernel(mr, nr) for mr, nr in tiles))
    parts.append(
        """static uint64_t stand_dpbusd(uint64_t iters, int seed) {
    __m512i a[16];
    for (int i = 0; i < 16; i++) a[i] = _mm512_set1_epi32(1234567 + seed);
    __m512i b = _mm512_set1_epi8(3 + seed), c = _mm512_set1_epi8(5 - seed);
    for (uint64_t i = 0; i < iters; i++) {
        __m512i t = _mm512_set1_epi32((int)(i & 0xFFFF));
        for (int j = 0; j < 16; j++) a[j] = _mm512_dpbusd_epi32(a[j], b, c);
        a[0] = _mm512_xor_si512(a[0], t);
    }
    __m512i s = _mm512_setzero_si512();
    for (int j = 0; j < 16; j++) s = _mm512_add_epi32(s, a[j]);
    return (uint64_t)_mm512_reduce_add_epi32(s);
}
static uint64_t stand_dpbf16(uint64_t iters, int seed) {
    __m512 acc[16];
    for (int i = 0; i < 16; i++) acc[i] = _mm512_set1_ps(1.0f + seed);
    __m512bh b = bh_from_i(_mm512_set1_epi32(0x3f803f80)), ch = b;
    for (uint64_t i = 0; i < iters; i++) {
        __m512 t = _mm512_castsi512_ps(_mm512_set1_epi32((int)(i & 0xFFFF)));
        for (int j = 0; j < 16; j++) acc[j] = _mm512_dpbf16_ps(acc[j], b, ch);
        acc[0] = _mm512_xor_ps(acc[0], t);
    }
    __m512 s = _mm512_setzero_ps();
    for (int j = 0; j < 16; j++) s = _mm512_add_ps(s, acc[j]);
    return (uint64_t)_mm512_reduce_add_ps(s);
}
"""
    )
    parts.append("\n".join(pack_b_i8(nr) for nr in sorted(set(nr for _, nr in tiles))))
    parts.append(
        "\n".join(pack_b_bf16(nr) for nr in sorted(set(nr for _, nr in tiles)))
    )

    for mr, nr in tiles:

        def F(t: str) -> str:
            return (
                t.replace("@MR@", str(mr))
                .replace("@NR@", str(nr))
                .replace("@NR2@", str(nr))
            )

        parts.append(
            F(
                """static uint64_t run_i8_@MR@x@NR@(const uint8_t* A, const int32_t* Bp, int32_t* C, int M, int N, int K) {
    for (int n0 = 0; n0 < N; n0 += @NR@)
        for (int m0 = 0; m0 < M; m0 += @MR@)
            gemm_i8_@MR@x@NR@(A + m0*K, Bp + (n0/@NR@)*(K/4)*@NR@, C + m0*N + n0, K, N);
    uint64_t chk = 0;
    for (int i = 0; i < M*N; i++) chk += (uint32_t)C[i];
    return chk;
}
static uint64_t run_bf16_@MR@x@NR@(const uint16_t* A, const int32_t* Bp, float* C, int M, int N, int K) {
    for (int n0 = 0; n0 < N; n0 += @NR@)
        for (int m0 = 0; m0 < M; m0 += @MR@)
            gemm_bf16_@MR@x@NR@(A + m0*K, Bp + (n0/@NR@)*(K/2)*@NR@, C + m0*N + n0, K, N);
    uint64_t chk = 0;
    for (int i = 0; i < M*N; i++) chk += (uint64_t)(C[i] != 0.0f);
    return chk;
}
static int verify_i8_@MR@x@NR@(int M, int N, int K) {
    static uint8_t A[@SZ2@]; static int8_t B[@SZ2@]; static int32_t Cv[@SZ2@], Cr[@SZ2@]; static int32_t Bp[@SZ4@];
    for (int i = 0; i < M*K; i++) A[i] = (uint8_t)((i * 37) % 251);
    for (int i = 0; i < K*N; i++) B[i] = (int8_t)((i * 71) % 17) - 8;
    pack_b_i8_@NR@(B, Bp, K, N);
    run_i8_@MR@x@NR@(A, Bp, Cv, M, N, K);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t s = 0;
            for (int k = 0; k < K; k++) s += (int32_t)A[i*K + k] * (int32_t)B[k*N + j];
            Cr[i*N + j] = s;
        }
    int bad = 0;
    for (int i = 0; i < M*N; i++) if (Cv[i] != Cr[i]) bad++;
    return bad;
}
static int verify_bf16_@MR@x@NR@(int M, int N, int K) {
    static uint16_t A[@SZ2@], B[@SZ2@]; static float Cv[@SZ2@], Cr[@SZ2@]; static int32_t Bp[@SZ4@];
    for (int i = 0; i < M*K; i++) A[i] = f2b(-5.0f + (float)(i % 41) * 0.25f);
    for (int i = 0; i < K*N; i++) B[i] = f2b(-5.0f + (float)(i % 37) * 0.25f);
    pack_b_bf16_@NR@(B, Bp, K, N);
    run_bf16_@MR@x@NR@(A, Bp, Cv, M, N, K);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += b2f(A[i*K + k]) * b2f(B[k*N + j]);
            Cr[i*N + j] = s;
        }
    int bad = 0;
    for (int i = 0; i < M*N; i++) if (fabsf(Cv[i] - Cr[i]) > 0.05f * (fabsf(Cr[i]) + 1.0f)) bad++;
    return bad;
}
"""
            )
        )

    cases = "\n".join(
        '  {"%dx%d", verify_i8_%dx%d, verify_bf16_%dx%d, run_i8_%dx%d, run_bf16_%dx%d,\n'
        "   pred_i8_%dx%d, pred_bf16_%dx%d},"
        % (mr, nr, mr, nr, mr, nr, mr, nr, mr, nr, mr, nr, mr, nr)
        for mr, nr in tiles
    )
    main = """typedef uint64_t (*r8_t)(const uint8_t*, const int32_t*, int32_t*, int, int, int);
typedef uint64_t (*rb_t)(const uint16_t*, const int32_t*, float*, int, int, int);
typedef int (*v8_t)(int, int, int); typedef int (*vb_t)(int, int, int);
struct Case { const char* name; v8_t vi; vb_t vb; r8_t fi; rb_t fb; int p8; int p16; };
static struct Case cases[] = {
%s
};
int main(void) {
    const int M = @SZ@, N = @SZ@, K = @SZ@;
    static uint8_t A8[@SZ2@]; static int8_t B8[@SZ2@]; static uint16_t A16[@SZ2@], B16[@SZ2@];
    static int32_t Bp8[@SZ4@], Bp16[@SZ4@], C8[@SZ2@]; static float Cf[@SZ2@];
    for (int i = 0; i < M*K; i++) { A8[i] = (int8_t)((i*37)%%19)-9; A16[i] = (uint16_t)((i*13)%%1000)+1; }
    for (int i = 0; i < K*N; i++) { B8[i] = (int8_t)((i*71)%%17)-8; B16[i] = (uint16_t)((i*29)%%1000)+1; }
    pack_b_i8_%d(B8, Bp8, K, N); pack_b_bf16_%d(B16, Bp16, K, N);
    printf("M=%%d N=%%d K=%%d (MACs=%%d)\\n", M, N, K, M*N*K);
    double macs = (double)M*N*K;
    volatile uint64_t sink = 0;
    uint64_t total_tsc = 0;
    uint64_t best_s = ~0ULL;
    for (int r = 0; r < %d; r++) {
        uint64_t t0 = __rdtsc(), t1;
        sink += stand_dpbusd(2000000, r);
        t1 = __rdtsc(); uint64_t d = t1 - t0; if (d < best_s) best_s = d;
        total_tsc += d;
    }
    double stand_i8 = 64.0 * 16.0 * 2000000.0 / best_s;
    best_s = ~0ULL;
    for (int r = 0; r < %d; r++) {
        uint64_t t0 = __rdtsc(), t1;
        sink += stand_dpbf16(2000000, r);
        t1 = __rdtsc(); uint64_t d = t1 - t0; if (d < best_s) best_s = d;
        total_tsc += d;
    }
    double stand_b16 = 32.0 * 16.0 * 2000000.0 / best_s;
    printf("standalone 16ch: i8 %%.1f MACs/cyc  bf16 %%.1f MACs/cyc (same-timebase reference)\\n",
           stand_i8, stand_b16);
    for (unsigned t = 0; t < sizeof(cases)/sizeof(cases[0]); t++) {
        struct Case* cs = &cases[t];
        int bi = cs->vi(M, N, K), bb = cs->vb(M, N, K);
        sink += cs->fi(A8, Bp8, C8, M, N, K); sink += cs->fb(A16, Bp16, Cf, M, N, K);
        uint64_t best_i = ~0ULL, best_b = ~0ULL;
        for (int r = 0; r < %d; r++) {
            uint64_t t0 = __rdtsc(), t1;
            for (uint64_t q = 0; q < 100; q++) sink += cs->fi(A8, Bp8, C8, M, N, K);
            t1 = __rdtsc(); uint64_t d = t1 - t0; if (d < best_i) best_i = d;
            total_tsc += d;
        }
        for (int r = 0; r < %d; r++) {
            uint64_t t0 = __rdtsc(), t1;
            for (uint64_t q = 0; q < 100; q++) sink += cs->fb(A16, Bp16, Cf, M, N, K);
            t1 = __rdtsc(); uint64_t d = t1 - t0; if (d < best_b) best_b = d;
            total_tsc += d;
        }
        printf("%%s  i8: bad=%%d cyc=%%llu MACs/cyc=%%7.1f rel=%%5.1f%%%% | "
               "bf16: bad=%%d cyc=%%llu MACs/cyc=%%7.1f rel=%%5.1f%%%% (vs pred %%d/%%d)\\n",
               cs->name, bi, (unsigned long long)(best_i/100), macs/(best_i/100.0),
               100.0*macs/(best_i/100.0)/stand_i8,
               bb, (unsigned long long)(best_b/100), macs/(best_b/100.0),
               100.0*macs/(best_b/100.0)/stand_b16, cs->p8, cs->p16);
        fflush(stdout);
    }
    printf("sink=%%llu\\n", (unsigned long long)sink);
    printf("total_tsc=%%llu\\n", (unsigned long long)total_tsc);
    return 0;
}
""" % (
        cases,
        nr,
        nr,
        reps,
        reps,
        reps,
        reps,
    )
    all_src = "\n".join(parts)
    main = (
        main.replace("@SZ@", str(size))
        .replace("@SZ2@", str(size * size))
        .replace("@SZ4@", str(size * size * 4))
    )
    all_src = (
        all_src.replace("@SZ@", str(size))
        .replace("@SZ2@", str(size * size))
        .replace("@SZ4@", str(size * size * 4))
    )
    parts = [all_src]
    parts.append(main)
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="8x32", help="comma list MRxNR")
    ap.add_argument("--size", type=int, default=64, help="M=N=K (multiple of 64)")
    args = ap.parse_args()
    tiles = []
    for t in args.tiles.split(","):
        mr, nr = (int(x) for x in t.split("x"))
        tiles.append((mr, nr))

    m = get_machine()
    print(
        "machine:",
        m.name,
        "| peak INT8:",
        m.gemm_peak_macs_per_cycle("vnni8"),
        "MACs/cyc | peak BF16:",
        m.gemm_peak_macs_per_cycle("bf16"),
        "MACs/cyc",
    )
    S = args.size
    pred = {}
    for mr, nr in tiles:
        pred[(mr, nr)] = (
            round(m.predicted_gemm_cycles("vnni8", S, S, S, mr, nr)),
            round(m.predicted_gemm_cycles("bf16", S, S, S, mr, nr)),
        )
    rep = {"64": 7}.get(str(S), 5)
    with tempfile.TemporaryDirectory() as d:
        cfile = Path(d) / "gemm.c"
        cfile.write_text(gen_c(tiles, pred, reps=rep, size=S))
        exe = Path(d) / "gemm"
        r = subprocess.run(
            [
                "gcc",
                "-O3",
                "-march=native",
                "-funroll-loops",
                "-o",
                str(exe),
                str(cfile),
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stderr)
            return 1
        out = subprocess.run([str(exe)], capture_output=True, text=True)
        print(out.stdout)
        if out.returncode != 0:
            print(out.stderr)
            return 1
        obj = subprocess.run(
            ["objdump", "-d", str(exe)], capture_output=True, text=True
        ).stdout
        # count stack spills in the hot kernels (zmm saves to [rsp])
        import re

        hot = re.findall(r"<(gemm_i8_|gemm_bf16_)[^>]+>:", obj)
        print(f"[disasm] gemm kernels found: {len(hot)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
