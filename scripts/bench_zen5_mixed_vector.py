#!/usr/bin/env python3
"""Zen 5 mixed-precision vector throughput experiment.

Determines whether the AMD Zen 5 FP/vector backend can execute VNNI (INT8
dot), FP32 FMA, and BF16 dot instructions simultaneously at their individual
peak rates, or whether they contend for shared execution resources.

Method: pinned single-core kernels, each an unrolled loop over 8 independent
512-bit vector accumulators (enough to saturate ILP without register-allocation
noise), cycle-counted with perf (exact core cycles). For each op type we
report MACs/cycle. The interleaved kernels reveal the sharing model:

  macs_cyc(mix) ~ macs_cyc(a) + macs_cyc(b)  ->  full overlap
  macs_cyc(mix) ~ max(macs_cyc(a), macs_cyc(b)) ->  full contention

Opcodes counted (per 512-bit instruction):
  VPDPBUSD   : 64 u8 x u8 -> i32 MACs (128 INT8 ops)
  VFMADD231PS: 16 FMA (32 FLOPs)
  VDPBF16PS  : 32 BF16 MACs (64 FLOPs)

Results on this 9950X3D (Zen 5, see bench/zen5_mixed_vector_results.md):
  pure VPDPBUSD / VFMADD231PS: ~1.83 vec-ops/cyc each (near the 2/cyc peak)
  alternating 4:4 mix: ~3.66 vec-ops/cyc = sum of the singles (full overlap)
  grouped 4:4 or skewed 6:2/2:6: ~1.82 vec-ops/cyc (time-shared)
  alternating VNNI + BF16: ~2.66 vec-ops/cyc (partial overlap)
  => The overlap is driven by INSTRUCTION ALTERNATION, not the ratio.
     True interleaving lets VNNI and FP32-FMA run concurrently on this
     silicon; bursts of one type serialize at the single-stream rate.

Usage:
  python scripts/bench_zen5_mixed_vector.py [--cpu N] [--iters N] [--reps N]
  --cpu -1 (default) scans the 32 logical CPUs for the quietest one first.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

C_SOURCE = r"""
#define _GNU_SOURCE
#include <immintrin.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <x86intrin.h>

typedef __m512i v16i;
typedef __m512 v16f;

static uint64_t k_vnni(uint64_t iters) {
    v16i a0 = _mm512_set1_epi32(1), a1 = _mm512_set1_epi32(2);
    v16i a2 = _mm512_set1_epi32(3), a3 = _mm512_set1_epi32(4);
    v16i a4 = _mm512_set1_epi32(5), a5 = _mm512_set1_epi32(6);
    v16i a6 = _mm512_set1_epi32(7), a7 = _mm512_set1_epi32(8);
    v16i b = _mm512_set1_epi8(3);
    v16i c = _mm512_set1_epi8(5);
    for (uint64_t i = 0; i < iters; i++) {
        asm volatile(
            "vpdpbusd %[b], %[c], %[a0]\n\t"
            "vpdpbusd %[b], %[c], %[a1]\n\t"
            "vpdpbusd %[b], %[c], %[a2]\n\t"
            "vpdpbusd %[b], %[c], %[a3]\n\t"
            "vpdpbusd %[b], %[c], %[a4]\n\t"
            "vpdpbusd %[b], %[c], %[a5]\n\t"
            "vpdpbusd %[b], %[c], %[a6]\n\t"
            "vpdpbusd %[b], %[c], %[a7]"
            : [a0] "+v"(a0), [a1] "+v"(a1), [a2] "+v"(a2), [a3] "+v"(a3),
              [a4] "+v"(a4), [a5] "+v"(a5), [a6] "+v"(a6), [a7] "+v"(a7)
            : [b] "v"(b), [c] "v"(c)
            : "cc");
    }
    v16i s = a0;
    s = _mm512_add_epi32(s, a1); s = _mm512_add_epi32(s, a2);
    s = _mm512_add_epi32(s, a3); s = _mm512_add_epi32(s, a4);
    s = _mm512_add_epi32(s, a5); s = _mm512_add_epi32(s, a6);
    s = _mm512_add_epi32(s, a7);
    return (uint64_t)_mm512_reduce_add_epi32(s);
}


static uint64_t k_fma(uint64_t iters) {
    v16f a0 = _mm512_set1_ps(1.0f), a1 = _mm512_set1_ps(2.0f);
    v16f a2 = _mm512_set1_ps(3.0f), a3 = _mm512_set1_ps(4.0f);
    v16f a4 = _mm512_set1_ps(5.0f), a5 = _mm512_set1_ps(6.0f);
    v16f a6 = _mm512_set1_ps(7.0f), a7 = _mm512_set1_ps(8.0f);
    v16f b = _mm512_set1_ps(1.0001f);
    v16f c = _mm512_set1_ps(0.9999f);
    for (uint64_t i = 0; i < iters; i++) {
        asm volatile(
            "vfmadd231ps %[b], %[c], %[a0]\n\t"
            "vfmadd231ps %[b], %[c], %[a1]\n\t"
            "vfmadd231ps %[b], %[c], %[a2]\n\t"
            "vfmadd231ps %[b], %[c], %[a3]\n\t"
            "vfmadd231ps %[b], %[c], %[a4]\n\t"
            "vfmadd231ps %[b], %[c], %[a5]\n\t"
            "vfmadd231ps %[b], %[c], %[a6]\n\t"
            "vfmadd231ps %[b], %[c], %[a7]"
            : [a0] "+v"(a0), [a1] "+v"(a1), [a2] "+v"(a2), [a3] "+v"(a3),
              [a4] "+v"(a4), [a5] "+v"(a5), [a6] "+v"(a6), [a7] "+v"(a7)
            : [b] "v"(b), [c] "v"(c)
            : "cc");
    }
    v16f s = a0;
    s = _mm512_add_ps(s, a1); s = _mm512_add_ps(s, a2);
    s = _mm512_add_ps(s, a3); s = _mm512_add_ps(s, a4);
    s = _mm512_add_ps(s, a5); s = _mm512_add_ps(s, a6);
    s = _mm512_add_ps(s, a7);
    return (uint64_t)_mm512_reduce_add_ps(s);
}


static uint64_t k_bf16(uint64_t iters) {
    v16i a0 = _mm512_set1_epi32(1), a1 = _mm512_set1_epi32(2);
    v16i a2 = _mm512_set1_epi32(3), a3 = _mm512_set1_epi32(4);
    v16i a4 = _mm512_set1_epi32(5), a5 = _mm512_set1_epi32(6);
    v16i a6 = _mm512_set1_epi32(7), a7 = _mm512_set1_epi32(8);
    union {
        v16i i;
        __m512bh h;
    } ub, uc;
    ub.i = _mm512_set1_epi16(0x3F80);
    uc.i = _mm512_set1_epi16(0x3F80);
    __m512bh bh = ub.h;
    __m512bh ch = uc.h;
    v16f fa0 = _mm512_castsi512_ps(a0), fa1 = _mm512_castsi512_ps(a1);
    v16f fa2 = _mm512_castsi512_ps(a2), fa3 = _mm512_castsi512_ps(a3);
    v16f fa4 = _mm512_castsi512_ps(a4), fa5 = _mm512_castsi512_ps(a5);
    v16f fa6 = _mm512_castsi512_ps(a6), fa7 = _mm512_castsi512_ps(a7);
    for (uint64_t i = 0; i < iters; i++) {
        asm volatile(
            "vdpbf16ps %[b], %[c], %[a0]\n\t"
            "vdpbf16ps %[b], %[c], %[a1]\n\t"
            "vdpbf16ps %[b], %[c], %[a2]\n\t"
            "vdpbf16ps %[b], %[c], %[a3]\n\t"
            "vdpbf16ps %[b], %[c], %[a4]\n\t"
            "vdpbf16ps %[b], %[c], %[a5]\n\t"
            "vdpbf16ps %[b], %[c], %[a6]\n\t"
            "vdpbf16ps %[b], %[c], %[a7]"
            : [a0] "+v"(fa0), [a1] "+v"(fa1), [a2] "+v"(fa2), [a3] "+v"(fa3),
              [a4] "+v"(fa4), [a5] "+v"(fa5), [a6] "+v"(fa6), [a7] "+v"(fa7)
            : [b] "v"(bh), [c] "v"(ch)
            : "cc");
    }
    v16f s = fa0;
    s = _mm512_add_ps(s, fa1); s = _mm512_add_ps(s, fa2);
    s = _mm512_add_ps(s, fa3); s = _mm512_add_ps(s, fa4);
    s = _mm512_add_ps(s, fa5); s = _mm512_add_ps(s, fa6);
    s = _mm512_add_ps(s, fa7);
    return (uint64_t)_mm512_reduce_add_ps(s);
}


static uint64_t k_mix_vnni_fma(uint64_t iters) {
    v16i i0 = _mm512_set1_epi32(1), i1 = _mm512_set1_epi32(2);
    v16i i2 = _mm512_set1_epi32(3), i3 = _mm512_set1_epi32(4);
    v16f f0 = _mm512_set1_ps(1.0f), f1 = _mm512_set1_ps(2.0f);
    v16f f2 = _mm512_set1_ps(3.0f), f3 = _mm512_set1_ps(4.0f);
    v16i b = _mm512_set1_epi8(3);
    v16i c = _mm512_set1_epi8(5);
    v16f fb = _mm512_set1_ps(1.0001f);
    v16f fc = _mm512_set1_ps(0.9999f);
    for (uint64_t i = 0; i < iters; i++) {
        asm volatile(
            "vpdpbusd %[b], %[c], %[i0]\n\t"
            "vfmadd231ps %[fb], %[fc], %[f0]\n\t"
            "vpdpbusd %[b], %[c], %[i1]\n\t"
            "vfmadd231ps %[fb], %[fc], %[f1]\n\t"
            "vpdpbusd %[b], %[c], %[i2]\n\t"
            "vfmadd231ps %[fb], %[fc], %[f2]\n\t"
            "vpdpbusd %[b], %[c], %[i3]\n\t"
            "vfmadd231ps %[fb], %[fc], %[f3]"
            : [i0] "+v"(i0), [i1] "+v"(i1), [i2] "+v"(i2), [i3] "+v"(i3),
              [f0] "+v"(f0), [f1] "+v"(f1), [f2] "+v"(f2), [f3] "+v"(f3)
            : [b] "v"(b), [c] "v"(c), [fb] "v"(fb), [fc] "v"(fc)
            : "cc");
    }
    v16i si = i0;
    si = _mm512_add_epi32(si, i1); si = _mm512_add_epi32(si, i2);
    si = _mm512_add_epi32(si, i3);
    v16f sf = f0;
    sf = _mm512_add_ps(sf, f1); sf = _mm512_add_ps(sf, f2);
    sf = _mm512_add_ps(sf, f3);
    return (uint64_t)(_mm512_reduce_add_epi32(si) + (uint64_t)_mm512_reduce_add_ps(sf));
}

static uint64_t k_mix_vnni_bf16(uint64_t iters) {
    v16i i0 = _mm512_set1_epi32(1), i1 = _mm512_set1_epi32(2);
    v16i i2 = _mm512_set1_epi32(3), i3 = _mm512_set1_epi32(4);
    v16i a0 = _mm512_set1_epi32(1), a1 = _mm512_set1_epi32(2);
    v16i a2 = _mm512_set1_epi32(3), a3 = _mm512_set1_epi32(4);
    v16i b = _mm512_set1_epi8(3);
    v16i c = _mm512_set1_epi8(5);
    union {
        v16i i;
        __m512bh h;
    } ub, uc;
    ub.i = _mm512_set1_epi16(0x3F80);
    uc.i = _mm512_set1_epi16(0x3F80);
    __m512bh bh = ub.h;
    __m512bh ch = uc.h;
    v16f f0 = _mm512_castsi512_ps(a0), f1 = _mm512_castsi512_ps(a1);
    v16f f2 = _mm512_castsi512_ps(a2), f3 = _mm512_castsi512_ps(a3);
    for (uint64_t i = 0; i < iters; i++) {
        asm volatile(
            "vpdpbusd %[b], %[c], %[i0]\n\t"
            "vdpbf16ps %[bh], %[ch], %[f0]\n\t"
            "vpdpbusd %[b], %[c], %[i1]\n\t"
            "vdpbf16ps %[bh], %[ch], %[f1]\n\t"
            "vpdpbusd %[b], %[c], %[i2]\n\t"
            "vdpbf16ps %[bh], %[ch], %[f2]\n\t"
            "vpdpbusd %[b], %[c], %[i3]\n\t"
            "vdpbf16ps %[bh], %[ch], %[f3]"
            : [i0] "+v"(i0), [i1] "+v"(i1), [i2] "+v"(i2), [i3] "+v"(i3),
              [f0] "+v"(f0), [f1] "+v"(f1), [f2] "+v"(f2), [f3] "+v"(f3)
            : [b] "v"(b), [c] "v"(c), [bh] "v"(bh), [ch] "v"(ch)
            : "cc");
    }
    v16i si = i0;
    si = _mm512_add_epi32(si, i1); si = _mm512_add_epi32(si, i2);
    si = _mm512_add_epi32(si, i3);
    v16f sf = f0;
    sf = _mm512_add_ps(sf, f1); sf = _mm512_add_ps(sf, f2);
    sf = _mm512_add_ps(sf, f3);
    return (uint64_t)(_mm512_reduce_add_epi32(si) + (uint64_t)_mm512_reduce_add_ps(sf));
}

typedef uint64_t (*kfn)(uint64_t);

static uint64_t run_kernel(kfn fn, uint64_t iters, int reps, uint64_t *check_out) {
    fn(iters / 10);
    uint64_t best = ~0ULL;
    uint64_t chk = 0;
    for (int r = 0; r < reps; r++) {
        unsigned aux;
        uint64_t t0 = __rdtscp(&aux);
        chk = fn(iters);
        uint64_t t1 = __rdtscp(&aux);
        uint64_t cyc = t1 - t0;
        if (cyc < best)
            best = cyc;
    }
    *check_out = chk;
    return best;
}

static uint64_t k_single(kfn fn, uint64_t iters) {
    unsigned aux;
    uint64_t t0 = __rdtscp(&aux);
    uint64_t chk = fn(iters);
    uint64_t t1 = __rdtscp(&aux);
    return chk ^ (t1 - t0);
}

int main(int argc, char **argv) {
    int cpu = argc > 1 ? atoi(argv[1]) : 0;
    uint64_t iters = argc > 2 ? strtoull(argv[2], NULL, 10) : 300000000ULL;
    int reps = argc > 3 ? atoi(argv[3]) : 5;
    int only = argc > 4 ? atoi(argv[4]) : -1;

    struct { const char *name; kfn fn; uint64_t per_instr; int cnt; } ks[] = {
        {"vnni8", k_vnni, 64, 8},
        {"fma32", k_fma, 16, 8},
        {"bf16", k_bf16, 32, 8},
        {"mix_vnni_fma", k_mix_vnni_fma, 0, 8},
        {"mix_vnni_bf16", k_mix_vnni_bf16, 0, 8},
    };
    int nk = (int)(sizeof(ks) / sizeof(ks[0]));

    cpu_set_t set;
    CPU_ZERO(&set);
    if (only >= 0) {
        if (only >= nk)
            return 2;
        CPU_SET(cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0) {
            perror("sched_setaffinity");
            return 1;
        }
        uint64_t chk = k_single(ks[only].fn, iters);
        printf("%llu\n", (unsigned long long)chk);
        return 0;
    }

    if (cpu < 0) {
        uint64_t best = ~0ULL;
        int best_cpu = 0;
        for (int c = 0; c < 32; c++) {
            CPU_ZERO(&set);
            CPU_SET(c, &set);
            if (sched_setaffinity(0, sizeof(set), &set) != 0)
                continue;
            uint64_t chk;
            uint64_t cyc = run_kernel(k_fma, iters / 5, 3, &chk);
            if (cyc < best) {
                best = cyc;
                best_cpu = c;
            }
        }
        cpu = best_cpu;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        sched_setaffinity(0, sizeof(set), &set);
    } else {
        CPU_SET(cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0) {
            perror("sched_setaffinity");
            return 1;
        }
    }
    printf("cpu=%d iters=%llu reps=%d\n", cpu, (unsigned long long)iters, reps);
    for (int k = 0; k < nk; k++) {
        uint64_t chk;
        uint64_t cyc = run_kernel(ks[k].fn, iters, reps, &chk);
        double ops_cyc = (double)(ks[k].cnt * iters) / (double)cyc;
        printf(
            "%-14s cycles=%12llu vec_ops=%12llu ops_cyc=%7.3f chk=%llu\n",
            ks[k].name, (unsigned long long)cyc,
            (unsigned long long)(ks[k].cnt * iters), ops_cyc,
            (unsigned long long)chk);
    }
    return 0;
}
"""


def load_avg() -> float:
    try:
        return float(open("/proc/loadavg").read().split()[0])
    except OSError:
        return float("nan")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", type=int, default=-1, help="-1 = scan for quietest core")
    p.add_argument("--iters", type=int, default=300_000_000)
    p.add_argument("--reps", type=int, default=5)
    args = p.parse_args()

    names = ["vnni8", "fma32", "bf16", "mix_vnni_fma", "mix_vnni_bf16"]
    vec_per_iter = {
        "vnni8": 8,
        "fma32": 8,
        "bf16": 8,
        "mix_vnni_fma": 16,
        "mix_vnni_bf16": 16,
    }

    with tempfile.TemporaryDirectory() as d:
        cfile = Path(d) / "bench_mixed.c"
        cfile.write_text(C_SOURCE)
        exe = Path(d) / "bench_mixed"
        subprocess.run(
            ["gcc", "-O3", "-march=native", "-o", str(exe), str(cfile)],
            check=True,
        )
        out = subprocess.run(
            [str(exe), str(args.cpu), str(args.iters), str(args.reps)],
            capture_output=True,
            text=True,
        )
        if out.returncode != 0:
            print(out.stderr)
            return 1
        cpu_line = out.stdout.splitlines()[0] if out.stdout else ""
        run_cpu = args.cpu
        if run_cpu < 0:
            try:
                run_cpu = int(cpu_line.split("cpu=")[1].split()[0])
            except (IndexError, ValueError):
                run_cpu = 0

        print(
            f"loadavg={load_avg():.1f}  (shared box; RDTSCP cycles, best-of-{args.reps})  "
            f"{cpu_line}"
        )
        rows = {}
        best = {}
        for idx, name in enumerate(names):
            best_cyc = None
            best_inst = None
            for rep in range(max(3, args.reps)):
                pr = subprocess.run(
                    [
                        "perf",
                        "stat",
                        "-e",
                        "cycles,instructions",
                        str(exe),
                        str(run_cpu),
                        str(args.iters),
                        str(args.reps),
                        str(idx),
                    ],
                    capture_output=True,
                    text=True,
                )
                cycles = instructions = None
                for line in pr.stderr.splitlines():
                    if "cycles" in line:
                        cycles = int(line.split()[0].replace(",", ""))
                    if "instructions" in line and "cycles" not in line:
                        instructions = int(line.split()[0].replace(",", ""))
                if cycles is None or instructions is None:
                    continue
                if best_cyc is None or cycles < best_cyc:
                    best_cyc = cycles
                    best_inst = instructions
            if best_cyc is None:
                print(f"{name:14s} perf parse failed")
                continue
            niter = args.iters
            ipc = best_inst / best_cyc
            vec_ops = vec_per_iter[name] * niter
            ops_cyc = vec_ops / best_cyc
            print(
                f"{name:14s} cycles={best_cyc:12d} vec_ops={vec_ops:12d} "
                f"ops_cyc={ops_cyc:7.3f}"
            )
            rows[name] = ops_cyc
            best[name] = (best_cyc, best_inst)

    print()
    print("overlap model (vector ops/cycle; identical 8-op/iter structure):")
    for a, b, m in [
        ("vnni8", "fma32", "mix_vnni_fma"),
        ("vnni8", "bf16", "mix_vnni_bf16"),
    ]:
        shared = max(rows[a], rows[b])
        additive = rows[a] + rows[b]
        ratio = rows[m] / additive
        print(
            f"  {a}+{b}: mix={rows[m]:.3f} ops/cyc  "
            f"shared-pipe bound={shared:.3f}  additive bound={additive:.3f}  "
            f"mix/additive={ratio:.3f}"
        )
        if rows[m] >= shared * 1.5:
            print(
                f"    -> OVERLAP: mix exceeds the shared-pipe bound ({shared:.2f}) by {rows[m] / shared:.2f}x"
            )
        elif rows[m] > shared * 1.05:
            print(
                f"    -> partial overlap: mix at {rows[m] / shared:.2f}x of the shared-pipe bound"
            )
        else:
            print(
                f"    -> contention: mix at {rows[m] / shared:.2f}x of the shared-pipe bound"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
