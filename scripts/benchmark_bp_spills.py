#!/usr/bin/env python3
"""
Benchmark S-box circuits under forced register pressure (spills).

This uses the scheduler + linear-scan regalloc + AVX2Emitter, and compares:
  - "no_spills": allocation with AVX2's register count
  - "forced_spills": allocation with a small register budget

It measures circuit-only throughput on bit-planes (no byte↔bitplane transpose).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bp_circuit_sbox import build_bp_sbox
from stc.bitslice import AES_SBOX_TABLE
from stc.circuit_synth import CircuitState, IncrementalOptimizer
from stc.sched import AVX2, AVX512, list_schedule
from stc.sched.emit import AVX2Emitter, AVX512Emitter
from stc.sched.liveness import compute_live_ranges
from stc.sched.regalloc import allocate_registers


def _emit_bench_c(code_a: str, code_b: str, width: int) -> str:
    return f"""
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

{code_a}
{code_b}

static inline double now_sec(void) {{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}}

static void run_one(const char* name, void (*fn)(__m{width}i*, __m{width}i*), int iters) {{
  volatile __m{width}i vin[8];
  __m{width}i in[8], out[8];
  for (int i = 0; i < 8; i++) vin[i] = _mm{width}_set1_epi32(0xA5A5A5A5u ^ (0x11111111u * i));

  double t0 = now_sec();
  for (int i = 0; i < iters; i++) {{
    for (int j = 0; j < 8; j++) in[j] = vin[j];
    fn(in, out);
    // Make the result observable.
    vin[0] = out[0];
  }}
  double t1 = now_sec();

  double elapsed = t1 - t0;
  // Bitsliced bit-planes: each vector bit is one independent evaluation.
  double evals = (double)iters * (double){width};
  printf("%s: %.3f ns/eval, %.0f evals/sec\\n", name, (elapsed / evals) * 1e9, evals / elapsed);
}}

int main(int argc, char** argv) {{
  int iters = 2000000;
  if (argc > 1) iters = atoi(argv[1]);
  printf("BP spill benchmark (circuit-only, bit-planes)\\n");
  printf("Iterations: %d\\n\\n", iters);
  run_one("no_spills", sbox_bp_no_spills, iters);
  run_one("forced_spills", sbox_bp_forced_spills, iters);
  return 0;
}}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=2_000_000)
    ap.add_argument(
        "--target",
        choices=["avx2", "avx512"],
        default="avx2",
        help="Scheduling target model (codegen stays 256-bit via AVX2Emitter).",
    )
    ap.add_argument(
        "--circuit",
        choices=["bp128", "anf1145", "ternary828", "json"],
        default="bp128",
        help="Which circuit to benchmark.",
    )
    ap.add_argument(
        "--json",
        type=str,
        default="",
        help="Path to CircuitState JSON (used when --circuit=json).",
    )
    ap.add_argument(
        "--regs", type=int, default=None, help="Register budget to force spills"
    )
    ap.add_argument(
        "--spill-mem",
        action="store_true",
        help="Force spill slots to be volatile (closer to real memory spills, but slower).",
    )
    args = ap.parse_args()

    if args.circuit == "bp128":
        circuit = build_bp_sbox()
    elif args.circuit == "anf1145":
        opt = IncrementalOptimizer(AES_SBOX_TABLE, input_bits=8, output_bits=8)
        circuit = opt.best_state
    elif args.circuit == "ternary828":
        path = "out/sbox_anf_optimized.json"
        if not os.path.exists(path):
            raise SystemExit(
                f"{path} not found; run `PYTHONPATH=. .venv/bin/python scripts/depth_aware_avx512_sbox.py` first"
            )
        with open(path) as f:
            circuit = CircuitState.from_dict(json.load(f))
    else:
        if not args.json:
            raise SystemExit("--json is required when --circuit=json")
        with open(args.json) as f:
            circuit = CircuitState.from_dict(json.load(f))

    gates = list(circuit.gates)
    input_bits = circuit.input_bits
    outputs = list(circuit.outputs)

    target = AVX512 if args.target == "avx512" else AVX2
    schedule = list_schedule(gates, input_bits, outputs, target, "slack")
    live_ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

    regs_ok = target.registers
    regs_spill = args.regs if args.regs is not None else 4
    alloc_ok = allocate_registers(live_ranges, schedule, num_registers=regs_ok)
    alloc_spill = allocate_registers(live_ranges, schedule, num_registers=regs_spill)

    print("S-box spill benchmark (circuit-only, bit-planes)")
    print(f"Circuit: {args.circuit} ({circuit.gate_count} gates)")
    print(f"Target: {args.target}, schedule {schedule.total_cycles} cycles")
    print(f"no_spills: regs={regs_ok}, spills={alloc_ok.num_spills}")
    print(f"forced_spills: regs={regs_spill}, spills={alloc_spill.num_spills}")

    emitter = AVX512Emitter() if args.target == "avx512" else AVX2Emitter()
    code_ok = emitter.emit(
        schedule, alloc_ok, gates, input_bits, outputs, "sbox_bp_no_spills"
    )
    code_spill = emitter.emit(
        schedule, alloc_spill, gates, input_bits, outputs, "sbox_bp_forced_spills"
    )

    if args.spill_mem:
        # Make spill slots "real" by forcing them into memory in this benchmark.
        # (Otherwise the compiler may keep `stackN` in registers and the benchmark
        # doesn't reflect spill traffic.)
        code_ok = code_ok.replace("__m256i stack", "volatile __m256i stack")
        code_spill = code_spill.replace("__m256i stack", "volatile __m256i stack")

    needs_ternary = any(len(g) == 5 for g in gates)
    cflags = ["-O3", "-march=native"]
    if args.target == "avx512":
        cflags += ["-mavx512f"]
    else:
        cflags += ["-mavx2"]
        if needs_ternary:
            cflags += ["-mavx512f", "-mavx512vl"]

    with tempfile.TemporaryDirectory() as td:
        c_path = os.path.join(td, "bench.c")
        exe_path = os.path.join(td, "bench")
        width = 512 if args.target == "avx512" else 256
        Path(c_path).write_text(_emit_bench_c(code_ok, code_spill, width))

        res = subprocess.run(
            ["gcc", *cflags, "-o", exe_path, c_path],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            print(res.stderr)
            return 1

        res = subprocess.run(
            [exe_path, str(args.iterations)],
            capture_output=False,
            text=True,
            timeout=300,
        )
        return res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
