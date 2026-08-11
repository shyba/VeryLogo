#!/usr/bin/env python3
"""Baseline benchmark for Z3 usage across the pipeline.

Measures, for the *current* code, the cost of the five Z3 roles:
  1. autovec equivalence proofs (fresh solver per proof, wall timeout)
  2. superopt CEGIS (pattern + hard negative cases)
  3. exact synthesis (marginal-size SAT search)
  4. bounded reachability / constant-state
  5. CLI pipeline with --autovec / --superopt

Reports wall time per workload (median over reps) plus solver-result
counts where available. Run it before and after Z3-related changes.
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import z3

from stc.tick_ir import (
    Add,
    And,
    BitVecConst,
    BitVecType,
    Bitcast,
    Concat,
    Mux,
    SimdAdd,
    SimdMaxU,
    SimdType,
    Slice,
    TickIR,
    Ult,
    Var,
)
from stc.z3_encode import encode_expr
from stc.autovec import _equiv
from stc.superopt import SuperoptStats, superopt_expr

REPS = 5


def median(xs):
    return statistics.median(xs)


def bench(name, fn, reps=REPS):
    times = []
    cpus = []
    results = []
    for _ in range(reps):
        t0 = time.time()
        c0 = time.process_time()
        r = fn()
        times.append(time.time() - t0)
        cpus.append(time.process_time() - c0)
        results.append(r)
    med = median(times)
    cmed = median(cpus)
    print(
        f"{name:42s} wall={med * 1000:9.1f}ms  cpu={cmed * 1000:9.1f}ms  "
        f"range=[{min(times) * 1000:.1f},{max(times) * 1000:.1f}]  {results[0]}"
    )
    return med


def load_avg():
    try:
        return float(open("/proc/loadavg").read().split()[0])
    except OSError:
        return float("nan")


def make_spec(lane_width, lanes):
    simd = SimdType(lane_width=lane_width, lanes=lanes)
    bv = BitVecType(width=lane_width * lanes)
    types = {"x": simd, "y": simd}
    xb = Bitcast(to=bv, x=Var(name="x"))
    yb = Bitcast(to=bv, x=Var(name="y"))
    parts = []
    for i in range(lanes - 1, -1, -1):
        off = i * lane_width
        parts.append(
            Add(
                a=Slice(x=xb, offset=off, width=lane_width),
                b=Slice(x=yb, offset=off, width=lane_width),
            )
        )
    spec = Bitcast(to=simd, x=Concat(parts=parts))
    return spec, types


def make_mux_min_spec(lane_width, lanes):
    simd = SimdType(lane_width=lane_width, lanes=lanes)
    bv = BitVecType(width=lane_width * lanes)
    types = {"x": simd, "y": simd}
    xb = Bitcast(to=bv, x=Var(name="x"))
    yb = Bitcast(to=bv, x=Var(name="y"))
    parts = []
    for i in range(lanes - 1, -1, -1):
        off = i * lane_width
        parts.append(
            Mux(
                cond=Ult(
                    a=Slice(x=xb, offset=off, width=lane_width),
                    b=Slice(x=yb, offset=off, width=lane_width),
                ),
                a=Slice(x=yb, offset=off, width=lane_width),
                b=Slice(x=xb, offset=off, width=lane_width),
            )
        )
    return Bitcast(to=simd, x=Concat(parts=parts)), types


def make_no_equiv_spec(lane_width, lanes):
    simd = SimdType(lane_width=lane_width, lanes=lanes)
    bv = BitVecType(width=lane_width * lanes)
    types = {"x": simd}
    xb = Bitcast(to=bv, x=Var(name="x"))
    parts = []
    for i in range(lanes - 1, -1, -1):
        off = i * lane_width
        parts.append(
            And(
                a=Slice(x=xb, offset=off, width=lane_width),
                b=BitVecConst(value=0xFE, width=lane_width),
            )
        )
    return Bitcast(to=simd, x=Concat(parts=parts)), types


def counter_ir():
    t = BitVecType(width=8)
    return TickIR(
        name="counter",
        inputs={"inc": t},
        outputs={"o": t},
        state={"s": t},
        reset_state={"s": BitVecConst(value=0, width=8)},
        next_state={"s": Var("s")},
        output_exprs={"o": Var("s")},
    )


def main() -> int:
    print(f"loadavg={load_avg():.1f}  z3={z3.get_version_string()}")
    print()

    print("== 1. autovec equivalence proofs (CPU budget + retry) ==")

    def w1_128():
        spec, types = make_spec(16, 8)
        cand = SimdAdd(a=Var("x"), b=Var("y"))
        return _equiv(spec, cand, types, timeout_ms=2000)

    def w1_512():
        spec, types = make_spec(32, 16)
        cand = SimdAdd(a=Var("x"), b=Var("y"))
        return _equiv(spec, cand, types, timeout_ms=2000)

    bench("equiv proof 128-bit (CPU budget 2000ms)", w1_128)
    bench("equiv proof 512-bit (CPU budget 2000ms)", w1_512)

    print()
    print("== 2. superopt CEGIS ==")

    def w2_pattern():
        spec, types = make_mux_min_spec(8, 4)
        st = SuperoptStats()
        r = superopt_expr(spec, types, max_nodes=5, timeout_ms=2000, stats=st)
        return (
            type(r).__name__ if r else None,
            st.candidates_checked,
            st.solver_checks,
        )

    def w2_hard():
        spec, types = make_no_equiv_spec(8, 4)
        st = SuperoptStats()
        try:
            superopt_expr(spec, types, max_nodes=5, timeout_ms=2000, stats=st)
            r = "found"
        except Exception as e:
            r = type(e).__name__
        return (r, st.candidates_checked, st.solver_checks)

    bench("superopt mux-min pattern n=5", w2_pattern)
    bench("superopt no-equiv n=5", w2_hard)

    print()
    print("== 3. exact synthesis (marginal-size SAT) ==")
    sbox = [
        0xC,
        0x5,
        0x6,
        0xB,
        0x9,
        0x0,
        0xA,
        0xD,
        0x3,
        0xE,
        0xF,
        0x8,
        0x4,
        0x7,
        0x1,
        0x2,
    ]
    bt = [(sbox[i] >> 1) & 1 for i in range(16)]

    from stc.circuit_synth import _synthesize_single_fixed_size

    def w3():
        r = _synthesize_single_fixed_size(bt, 4, 7, 60000)
        return "found" if r else "none"

    bench("synthesize bit1 @ 7 gates (60s budget)", w3, reps=3)

    print()
    print("== 4. bounded reachability / constant-state (200ms) ==")
    from stc.reachability import constant_state_within_bound

    def w4():
        r = constant_state_within_bound(counter_ir(), 8, timeout_ms=2000)
        return sorted(r.items())

    bench("constant_state counter bound=8 (2000ms)", w4)

    print()
    print("== 5. CLI pipeline with --autovec / --superopt ==")
    root = Path(__file__).resolve().parent.parent
    src = root / "fixtures" / "verilog" / "simd_lane_add_slices.v"

    def run_cli(extra):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run(
                [sys.executable, "-m", "stc", str(src), "--out", d, "--no-backend"]
                + extra,
                capture_output=True,
                text=True,
                cwd=root,
                timeout=300,
            )
            return r.returncode

    def w5_autovec():
        return run_cli(["--infer-simd", "--autovec"])

    def w5_superopt():
        return run_cli(["--infer-simd", "--autovec", "--superopt"])

    bench("cli --infer-simd --autovec (no backend)", w5_autovec, reps=3)
    bench("cli --infer-simd --autovec --superopt", w5_superopt, reps=3)

    print()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
