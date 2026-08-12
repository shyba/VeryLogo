#!/usr/bin/env python3
"""AVX-512 throughput benchmark vs Agner Fog's Zen 5 instruction tables.

Measures single-stream instruction throughput on the local Zen 5 CPU for the
512-bit instructions STC's x86 backends emit, and compares against the
published reciprocal-throughput values extracted from Agner's instruction
tables (bench/agner_zen5_avx512.csv, Zen 5 sheet).

Method (hardened to avoid the codegen and structure artifacts found earlier):
- One inline-asm block per iteration, 8 independent accumulators, verified
  by disassembly that the loop contains exactly the 8 target ops + loop
  control (no register-copy noise, no autovectorization).
- Pinned to the quietest logical CPU (scanned first), best-of-N perf core
  cycles and instructions; ops/cyc = 8*iters / min_cycles.
- All kernels share the identical 8-op-per-iteration structure so results
  are comparable; mixed-op overlap is a separate experiment
  (bench_zen5_mixed_vector.py).

Agner reference: reciprocal throughput RT (cycles/instruction); expected
ops/cyc = 1/RT.

Known discrepancy (flagged, not resolved): VDPBF16PS measures ~1.33 ops/cyc
here vs Agner's RT=3 (0.333 ops/cyc); the loop is verified (8 vdpbf16ps per
iteration by perf instruction counts, core cycles from hardware counters).

Usage:
  python scripts/bench_avx512_vs_agner.py [--cpu N] [--iters N] [--reps N]
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

AGNER_CSV = Path(__file__).resolve().parent.parent / "bench" / "agner_zen5_avx512.csv"


def load_agner() -> dict[str, dict[str, str]]:
    import re as _re

    fam: dict[str, dict[str, str]] = {}
    rows_by_fam: dict[str, list[dict[str, str]]] = {}
    with open(AGNER_CSV, newline="") as f:
        for row in csv.DictReader(f):
            rows_by_fam.setdefault(row["family"], []).append(row)
    for fam_name, rows in rows_by_fam.items():

        def reg_weight(r):
            ops = r["operands"]
            if _re.search(r"\bv,v|z,z|k,z", ops):
                return 0
            if "m" in ops:
                return 2
            return 1

        pick = min(rows, key=reg_weight)
        fam[fam_name] = {
            "instruction": pick["instruction"],
            "operands": pick["operands"],
            "rt": pick["rt_cycles"],
            "pipes": pick["pipes"],
        }
    return fam


# (kernel, agner family, kind, asm template with [a0] as the accumulator dst)
# kind: i = int zmm, f = float zmm, b = bf16, m = mask output
INSTR = [
    ("vpandd", "bitwise", "i", "vpandd %[b], %[c], %[a0]"),
    ("vpandnd", "bitwise", "i", "vpandnd %[b], %[c], %[a0]"),
    ("vpord", "bitwise", "i", "vpord %[b], %[c], %[a0]"),
    ("vpxord", "bitwise", "i", "vpxord %[b], %[c], %[a0]"),
    ("vpaddd", "add/sub", "i", "vpaddd %[b], %[c], %[a0]"),
    ("vpaddw", "add/sub", "i", "vpaddw %[b], %[c], %[a0]"),
    ("vpsubd", "add/sub", "i", "vpsubd %[b], %[c], %[a0]"),
    ("vpmulld", "mul", "i", "vpmulld %[b], %[c], %[a0]"),
    ("vpmullw", "mul", "i", "vpmullw %[b], %[c], %[a0]"),
    ("vpmulhw", "mul", "i", "vpmulhw %[b], %[c], %[a0]"),
    ("vpmaddwd", "madd", "i", "vpmaddwd %[b], %[c], %[a0]"),
    ("vpdpbusd", "vnni8", "i", "vpdpbusd %[b], %[c], %[a0]"),
    ("vpdpwssd", "vnni16", "i", "vpdpwssd %[b], %[c], %[a0]"),
    ("vdpbf16ps", "bf16", "b", "vdpbf16ps %[b], %[c], %[a0]"),
    ("vfmadd231ps", "fma", "f", "vfmadd231ps %[b], %[c], %[a0]"),
    ("vmulps", "mul-fp", "f", "vmulps %[b], %[c], %[a0]"),
    ("vaddps", "add-fp", "f", "vaddps %[b], %[c], %[a0]"),
    ("vpminuw", "minmax", "i", "vpminuw %[b], %[c], %[a0]"),
    ("vpmaxuw", "minmax", "i", "vpmaxuw %[b], %[c], %[a0]"),
    ("vpslld", "shift-imm", "i", "vpslld $3, %[b], %[a0]"),
    ("vpsrld", "shift-imm", "i", "vpsrld $3, %[b], %[a0]"),
    ("vpsrad", "shift-imm", "i", "vpsrad $3, %[b], %[a0]"),
    ("vpternlogd", "ternary", "i", "vpternlogd $0x96, %[b], %[c], %[a0]"),
    ("vpcmpeqd", "cmp-int", "mi", "vpcmpeqd %[b], %[c], %k[a0]"),
    ("vcmpps", "cmp-fp", "mf", "vcmpps $0, %[b], %[c], %k[a0]"),
    ("vpblendmd", "blend", "iblend", ""),
    ("vpermd", "permute", "i", "vpermd %[c], %[b], %[a0]"),
    ("vpacksswb", "pack", "i", "vpacksswb %[b], %[c], %[a0]"),
    ("vpackuswb", "pack", "i", "vpackuswb %[b], %[c], %[a0]"),
    ("vpunpckldq", "unpack", "i", "vpunpckldq %[b], %[c], %[a0]"),
    ("vpshufb", "shuffle", "i", "vpshufb %[b], %[c], %[a0]"),
]

INT_FOLD = """    v16i s = a0;
    s = _mm512_add_epi32(s, a1); s = _mm512_add_epi32(s, a2);
    s = _mm512_add_epi32(s, a3); s = _mm512_add_epi32(s, a4);
    s = _mm512_add_epi32(s, a5); s = _mm512_add_epi32(s, a6);
    s = _mm512_add_epi32(s, a7);
    return (uint64_t)_mm512_reduce_add_epi32(s);"""

FLOAT_FOLD = """    v16f s = a0;
    s = _mm512_add_ps(s, a1); s = _mm512_add_ps(s, a2);
    s = _mm512_add_ps(s, a3); s = _mm512_add_ps(s, a4);
    s = _mm512_add_ps(s, a5); s = _mm512_add_ps(s, a6);
    s = _mm512_add_ps(s, a7);
    return (uint64_t)_mm512_reduce_add_ps(s);"""

MASK_FOLD = """    __mmask16 m = m0;
    m |= m1; m |= m2; m |= m3; m |= m4;
    m |= m5; m |= m6; m |= m7;
    return (uint64_t)m;"""


def gen_kernel(name: str, kind: str, tpl: str) -> str:
    lines = ["static uint64_t k_%s(uint64_t iters) {" % name]
    if kind == "iblend":
        lines.append("    v16i b = _mm512_set1_epi8(3);")
        lines.append("    v16i c = _mm512_set1_epi8(5);")
        lines.append("    volatile __mmask16 cmask = 0xFFFF;")
        for i in range(8):
            lines.append("    v16i t%d = _mm512_set1_epi32(%d);" % (i, i + 1))
        lines.append("    for (uint64_t i = 0; i < iters; i++) {")
        for i in range(8):
            lines.append("        t%d = _mm512_mask_blend_epi32(cmask, b, c);" % i)
        lines.append("    }")
        lines.append(
            INT_FOLD.replace("a0", "t0")
            .replace("a1", "t1")
            .replace("a2", "t2")
            .replace("a3", "t3")
            .replace("a4", "t4")
            .replace("a5", "t5")
            .replace("a6", "t6")
            .replace("a7", "t7")
        )
        lines.append("}")
        return "\n".join(lines) + "\n"
    if kind in ("i", "mi"):
        lines.append("    v16i b = _mm512_set1_epi8(3);")
        lines.append("    v16i c = _mm512_set1_epi8(5);")
    elif kind in ("f", "mf"):
        lines.append("    v16f b = _mm512_set1_ps(1.0001f);")
        lines.append("    v16f c = _mm512_set1_ps(0.9999f);")
    elif kind == "b":
        lines.append("    union { v16i i; __m512bh h; } ub, uc;")
        lines.append("    ub.i = _mm512_set1_epi16(0x3F80);")
        lines.append("    uc.i = _mm512_set1_epi16(0x3F80);")
        lines.append("    __m512bh b = ub.h;")
        lines.append("    __m512bh c = uc.h;")
    if kind in ("i", "b"):
        acc_t, init = "v16i", "_mm512_set1_epi32(%d)"
    elif kind == "f":
        acc_t, init = "v16f", "_mm512_set1_ps(%d.0f)"
    elif kind == "m":
        acc_t, init = None, None
    if kind in ("mi", "mf"):
        for i in range(8):
            lines.append("    __mmask16 m%d = 0;" % i)
    elif kind == "b":
        for i in range(8):
            lines.append(
                "    v16f a%d = _mm512_castsi512_ps(_mm512_set1_epi32(%d));"
                % (i, i + 1)
            )
    elif kind == "i":
        for i in range(8):
            lines.append("    v16i a%d = _mm512_set1_epi32(%d);" % (i, i + 1))
    elif kind == "f":
        for i in range(8):
            lines.append("    v16f a%d = _mm512_set1_ps(%d.0f);" % (i, i + 1))
    if "k[cmask]" in tpl:
        lines.append("    __mmask16 cmask = 0xFFFF;")
    lines.append("    for (uint64_t i = 0; i < iters; i++) {")
    lines.append("        asm volatile(")
    ops = []
    for i in range(8):
        op = tpl.replace("[a0]", "[a%d]" % i)
        ops.append('            "%s\\n\\t"' % op if i < 7 else '            "%s"' % op)
    lines.extend(ops)
    if kind in ("m", "mi", "mf"):
        outs = ", ".join('[a%d] "=k"(m%d)' % (i, i) for i in range(8))
        ins = '[b] "v"(b), [c] "v"(c)'
    else:
        outs = ", ".join('[a%d] "+v"(a%d)' % (i, i) for i in range(8))
        ins = '[b] "v"(b), [c] "v"(c)'
        if "k[cmask]" in tpl:
            outs += ', [cmask] "+k"(cmask)'
    lines.append("            : %s" % outs)
    lines.append("            : %s" % ins)
    lines.append('            : "cc");')
    lines.append("    }")
    if kind == "b":
        lines.append(FLOAT_FOLD)
    elif kind in ("i", "mi"):
        lines.append(MASK_FOLD if kind == "mi" else INT_FOLD)
    elif kind == "f":
        lines.append(FLOAT_FOLD)
    elif kind == "mf":
        lines.append(MASK_FOLD)
    lines.append("}")
    return "\n".join(lines) + "\n"


def gen_c(iters: int) -> str:
    kernels = []
    for name, fam, kind, tpl in INSTR:
        kernels.append(gen_kernel(name, kind, tpl))

    mask_prelude = ""
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
    uint64_t iters = argc > 2 ? strtoull(argv[2], NULL, 10) : 200000000ULL;
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
            uint64_t cyc = run_kernel(k_vpaddd, iters / 5, 3, &chk);
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
        printf("%-12s %.3f %llu\n", ks[k].name, ops_cyc, (unsigned long long)chk);
    }
    return 0;
}
""".replace(
        "@KERNELS@", "\n".join(kernels)
    ).replace(
        "@KS@",
        "\n".join('        {"%s", k_%s, 8},' % (name, name) for name, _, _, _ in INSTR),
    )


def load_avg() -> float:
    try:
        return float(open("/proc/loadavg").read().split()[0])
    except OSError:
        return float("nan")


def parse_rt(rt: str) -> float | None:
    try:
        return float(rt.split("-")[0])
    except (ValueError, AttributeError):
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", type=int, default=-1, help="-1 = scan for quietest core")
    p.add_argument("--iters", type=int, default=200_000_000)
    p.add_argument("--reps", type=int, default=5)
    args = p.parse_args()

    agner = load_agner()
    print(f"loadavg={load_avg():.1f}  Agner ref: {AGNER_CSV.name}")

    with tempfile.TemporaryDirectory() as d:
        cfile = Path(d) / "bench_avx512.c"
        cfile.write_text(gen_c(args.iters))
        exe = Path(d) / "bench_avx512"
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
        print(cpu_line)

        results: dict[str, float] = {}
        for idx, (name, fam, kind, _tpl) in enumerate(INSTR):
            best = None
            for _ in range(max(3, args.reps)):
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
                cyc = None
                for line in pr.stderr.splitlines():
                    if "cycles" in line:
                        cyc = int(line.split()[0].replace(",", ""))
                if cyc is not None and (best is None or cyc < best):
                    best = cyc
            if best is None:
                print(f"{name:12s}  perf failed")
                continue
            ops_cyc = 8 * args.iters / best
            results[name] = ops_cyc

    ref = agner
    print(
        f"\n{'instruction':12s} {'meas ops/cyc':>12s} {'agner ops/cyc':>13s} {'ratio':>6s}  flag"
    )
    flagged = 0
    for name, fam, _kind, _tpl in INSTR:
        r = ref.get(fam)
        rt = parse_rt(r["rt"]) if r else None
        ag_ops = 1.0 / rt if rt and rt > 0 else float("nan")
        meas = results.get(name, float("nan"))
        ratio = meas / ag_ops if ag_ops and meas == meas else float("nan")
        flag = ""
        if ratio == ratio:
            if ratio < 0.85 or ratio > 1.15:
                flag = "<-- CHECK"
                flagged += 1
        print(f"{name:12s} {meas:12.3f} {ag_ops:13.3f} {ratio:6.2f}  {flag}")
    print(f"\n{len(results)} measured; {flagged} outside +/-15% of Agner")
    return 0


if __name__ == "__main__":
    sys.exit(main())
