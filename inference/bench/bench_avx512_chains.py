#!/usr/bin/env python3
"""Chain-count sweep and interference matrix for the AVX-512 conflicts.

Finds the true execution throughput of VDPBF16PS and VPTERNLOGD (which an
8-chain benchmark pinned to their latency ceilings: 8/6 = 1.33 and
8/3 = 2.67 ops/cyc respectively, falsifying Agner's RT but not resolving
the real rate), then probes pairwise instruction-stream interference.

Method (intrinsics verified clean by objdump/perf instruction counts):
- standalone kernels at N independent loop-carried accumulator chains;
  the dependency ceiling is chains / latency. If the measured rate
  plateaus below the ceiling, the plateau is the execution throughput.
- interference kernels alternate two streams within one loop; reported as
  total ops/cyc vs the standalone rates (additive = no contention,
  max = full contention).

Reference (Agner Zen 5): VDPBF16PS lat 6 RT 3 P01; VPTERNLOGD lat 3 RT 1
P0123; VFMADD231PS lat 4 RT 0.5 P01; VPDPBUSD lat 4 RT 0.5 P01;
VADDPS lat 2 RT 0.5 P23.

Usage:
  python scripts/bench_avx512_chains.py [--cpu N] [--iters N] [--reps N]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

STANDS = [
    ("bf16", 8, "bf16"),
    ("bf16", 12, "bf16"),
    ("bf16", 16, "bf16"),
    ("tern", 8, "tern"),
    ("tern", 12, "tern"),
    ("tern", 16, "tern"),
    ("fma", 8, "fma"),
    ("fma", 12, "fma"),
    ("vnni", 8, "vnni"),
    ("vnni", 12, "vnni"),
    ("fpadd", 8, "fpadd"),
]

MIXES = [
    ("bf16", 16, "fma", 8),
    ("bf16", 16, "vnni", 8),
    ("bf16", 16, "fpadd", 8),
    ("tern", 12, "fma", 8),
    ("tern", 12, "vnni", 8),
    ("tern", 12, "fpadd", 8),
    ("vnni", 8, "fma", 8),
]

LATENCY = {"bf16": 6, "tern": 3, "fma": 4, "vnni": 4, "fpadd": 2}


def emit_op(kind: str, dst: str) -> str:
    if kind == "bf16":
        return "%s = _mm512_dpbf16_ps(%s, bh, ch);" % (dst, dst)
    if kind == "tern":
        return "%s = _mm512_ternarylogic_epi32(%s, bb, cc, 0x96);" % (dst, dst)
    if kind == "fma":
        return "%s = _mm512_fmadd_ps(%s, fb, fc);" % (dst, dst)
    if kind == "vnni":
        return "%s = _mm512_dpbusd_epi32(%s, b, c);" % (dst, dst)
    if kind == "fpadd":
        return "%s = _mm512_add_ps(%s, fb);" % (dst, dst)
    raise ValueError(kind)


def acc_decl(kind: str, idx: int) -> str:
    v = idx + 1
    if kind == "bf16":
        return "v16f a%d = _mm512_castsi512_ps(_mm512_set1_epi32(%d));" % (idx, v)
    if kind == "tern":
        return "v16i a%d = _mm512_set1_epi32(%d);" % (idx, v)
    if kind == "fma" or kind == "fpadd":
        return "v16f a%d = _mm512_set1_ps(%d.0f);" % (idx, v)
    if kind == "vnni":
        return "v16i a%d = _mm512_set1_epi32(%d);" % (idx, v)
    raise ValueError(kind)


def fold(kinds: list[tuple[str, int]]) -> str:
    lines = []
    acc = 0
    for kind, n in kinds:
        if kind in ("tern", "vnni"):
            s = "v16i s%d = a%d;" % (acc, acc)
            for j in range(1, n):
                s += " s%d = _mm512_add_epi32(s%d, a%d);" % (acc, acc, acc + j)
            lines.append(s)
            lines.append("acc |= (uint64_t)_mm512_reduce_add_epi32(s%d);" % acc)
        else:
            s = "v16f s%d = a%d;" % (acc, acc)
            for j in range(1, n):
                s += " s%d = _mm512_add_ps(s%d, a%d);" % (acc, acc, acc + j)
            lines.append(s)
            lines.append("acc |= (uint64_t)_mm512_reduce_add_ps(s%d);" % acc)
        acc += n
    return lines


def gen_stand(kind: str, n: int) -> str:
    decls = [acc_decl(kind, i) for i in range(n)]
    if kind == "bf16":
        decls += [
            "union { v16i i; __m512bh h; } ub;",
            "ub.i = _mm512_set1_epi16(0x3F80);",
            "__m512bh bh = ub.h;",
            "__m512bh ch = bh;",
        ]
    elif kind == "tern":
        decls += [
            "v16i bb = _mm512_set1_epi32(0x11111111);",
            "v16i cc = _mm512_set1_epi32(0x22222222);",
        ]
    elif kind in ("fma", "fpadd"):
        decls += [
            "v16f fb = _mm512_set1_ps(1.0f);",
            "v16f fc = fb;",
        ]
    elif kind == "vnni":
        decls += [
            "v16i b = _mm512_set1_epi8(3);",
            "v16i c = b;",
        ]
    if kind == "tern" and n <= 12:
        outs = ", ".join('[a%d] "+v"(a%d)' % (i, i) for i in range(n))
        asm_ops = "\\n\\t".join(
            "vpternlogd $0x96, %%[b], %%[c], %%[a%d]" % i for i in range(n)
        )
        body = [
            "        asm volatile(\n"
            '            "%s"\n            : %s\n'
            '            : [b] "v"(bb), [c] "v"(cc)\n            : "cc");'
            % (asm_ops, outs)
        ]
    else:
        body = [emit_op(kind, "a%d" % i) for i in range(n)]
    folds = fold([(kind, n)])
    return """static uint64_t k_%s_%d(uint64_t iters) {
%s
    uint64_t acc = 0;
    for (uint64_t i = 0; i < iters; i++) {
%s
    }
%s
    return acc;
}

""" % (
        kind,
        n,
        "\n".join("    " + d for d in decls),
        "\n".join(body),
        "\n".join("    " + l for l in folds),
    )


def gen_mix(kind_a: str, na: int, kind_b: str, nb: int) -> str:
    decls = [acc_decl(kind_a, i) for i in range(na)] + [
        acc_decl(kind_b, na + i) for i in range(nb)
    ]
    extras = {
        "bf16": [
            "union { v16i i; __m512bh h; } ub;",
            "ub.i = _mm512_set1_epi16(0x3F80);",
            "__m512bh bh = ub.h;",
            "__m512bh ch = bh;",
        ],
        "tern": [
            "v16i bb = _mm512_set1_epi32(0x11111111);",
            "v16i cc = bb;",
        ],
        "fma": [
            "v16f fb = _mm512_set1_ps(1.0f);",
            "v16f fc = fb;",
        ],
        "fpadd": [
            "v16f fb = _mm512_set1_ps(1.0f);",
            "v16f fc = fb;",
        ],
        "vnni": [
            "v16i b = _mm512_set1_epi8(3);",
            "v16i c = b;",
        ],
    }
    for k in (kind_a, kind_b):
        decls.extend(extras[k])
    body = []
    m = max(na, nb)
    for j in range(m):
        if j < na:
            body.append(emit_op(kind_a, "a%d" % j))
        if j < nb:
            body.append(emit_op(kind_b, "a%d" % (na + j)))
    folds = fold([(kind_a, na), (kind_b, nb)])
    return """static uint64_t k_%s_%s(uint64_t iters) {
%s
    uint64_t acc = 0;
    for (uint64_t i = 0; i < iters; i++) {
%s
    }
%s
    return acc;
}

""" % (
        kind_a,
        kind_b,
        "\n".join("    " + d for d in decls),
        "\n".join("        " + b for b in body),
        "\n".join("    " + l for l in folds),
    )


def gen_c() -> str:
    kern = []
    entries = []
    for kind, n, _ in STANDS:
        kern.append(gen_stand(kind, n))
        entries.append('        {"stand_%s_%d", k_%s_%d, %d},' % (kind, n, kind, n, n))
    for a, na, b, nb in MIXES:
        kern.append(gen_mix(a, na, b, nb))
        entries.append('        {"mix_%s_%s", k_%s_%s, %d},' % (a, b, a, b, na + nb))
    return r"""
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

@KERNELS@

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
    uint64_t iters = argc > 2 ? strtoull(argv[2], NULL, 10) : 150000000ULL;
    int reps = argc > 3 ? atoi(argv[3]) : 5;
    int only = argc > 4 ? atoi(argv[4]) : -1;

    struct { const char *name; kfn fn; int cnt; } ks[] = {
@KS@
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
            uint64_t cyc = run_kernel(k_fma_8, iters / 5, 3, &chk);
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
        printf("%s %.3f %llu\n", ks[k].name, ops_cyc, (unsigned long long)chk);
    }
    return 0;
}
""".replace(
        "@KERNELS@", "\n".join(kern)
    ).replace(
        "@KS@", "\n".join(entries)
    )


def load_avg() -> float:
    try:
        return float(open("/proc/loadavg").read().split()[0])
    except OSError:
        return float("nan")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", type=int, default=-1)
    p.add_argument("--iters", type=int, default=150_000_000)
    p.add_argument("--reps", type=int, default=5)
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as d:
        cfile = Path(d) / "chains.c"
        cfile.write_text(gen_c())
        exe = Path(d) / "chains"
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
        obj = subprocess.run(
            ["objdump", "-d", str(exe)], capture_output=True, text=True
        ).stdout.split("\n")
        vmov: dict[str, int] = {}
        cur = None
        for line in obj:
            if "<k_" in line and ">:" in line:
                cur = line.split("<")[1].split(">")[0]
            if cur and "vmovdqa" in line:
                vmov[cur] = vmov.get(cur, 0) + 1
        lines = out.stdout.splitlines()
        cpu_line = lines[0]
        run_cpu = args.cpu
        if run_cpu < 0:
            try:
                run_cpu = int(cpu_line.split("cpu=")[1].split()[0])
            except (IndexError, ValueError):
                run_cpu = 0
        print(f"loadavg={load_avg():.1f}  {cpu_line}")

        names = []
        for idx in range(len(lines) - 1):
            parts = lines[1 + idx].split()
            if len(parts) >= 2:
                names.append(parts[0])

        rates: dict[str, float] = {}
        for idx, name in enumerate(names):
            best = None
            best_inst = None
            cnt = 0
            if name.startswith("mix"):
                for a, na, b, nb in MIXES:
                    if name == "mix_%s_%s" % (a, b):
                        cnt = na + nb
            else:
                cnt = int(name.split("_")[-1])
            for _ in range(max(3, args.reps)):
                pr = subprocess.run(
                    [
                        "perf",
                        "stat",
                        "-e",
                        "cycles",
                        str(exe),
                        str(run_cpu),
                        str(args.iters),
                        str(args.reps),
                        str(idx),
                    ],
                    capture_output=True,
                    text=True,
                )
                cyc = inst = None
                for line in pr.stderr.splitlines():
                    if "cycles" in line:
                        cyc = int(line.split()[0].replace(",", ""))
                    if "instructions" in line and "cycles" not in line:
                        inst = int(line.split()[0].replace(",", ""))
                if cyc is not None and (best is None or cyc < best):
                    best = cyc
                if inst is not None:
                    best_inst = inst
            if best is None:
                print(f"{name:18s} perf failed")
                continue
            expected = cnt * args.iters
            if best_inst is not None and best_inst < expected:
                print(
                    f"{name:18s} INVALID: only {best_inst / args.iters:.1f} "
                    f"inst/iter vs {cnt} ops (dead-code elimination)"
                )
                continue
            rates[name] = (cnt * args.iters) / best
            noisy = vmov.get("k_" + name, 0)
            tag = "  [vmov-noise: %d, INVALID]" % noisy if noisy > 8 else ""
            print(f"{name:18s} {rates[name]:8.3f} ops/cyc{tag}")

    print()
    print("== standalone: measured vs latency ceiling (chains/latency) ==")
    for kind, n, _ in STANDS:
        key = "stand_%s_%d" % (kind, n)
        meas = rates.get(key)
        if meas is None:
            continue
        ceil = n / LATENCY[kind]
        tag = (
            "saturated (at ceiling)"
            if meas >= ceil * 0.98
            else ("-> true rate >= %.3f" % meas if meas < ceil else "unexpected")
        )
        print(
            f"  {kind:7s} {n:2d} chains: {meas:6.3f} ops/cyc  "
            f"latency ceiling {ceil:6.3f}  {tag}"
        )

    print()
    print("== interference: total ops/cyc vs standalone rates ==")
    stand = {}
    for kind, n, _ in STANDS:
        stand.setdefault(kind, {}).setdefault(n, rates.get("stand_%s_%d" % (kind, n)))
    for a, na, b, nb in MIXES:
        key = "mix_%s_%s" % (a, b)
        meas = rates.get(key)
        if meas is None:
            continue
        ra = stand[a][na]
        rb = stand[b][nb]
        if ra is None or rb is None:
            continue
        additive = ra + rb
        contended = max(ra, rb)
        print(
            f"  {a}({na}ch {ra:.3f}) + {b}({nb}ch {rb:.3f}): "
            f"mix {meas:.3f}  additive {additive:.3f}  max {contended:.3f}  "
            f"mix/additive {meas / additive:.2f}  mix/max {meas / contended:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
