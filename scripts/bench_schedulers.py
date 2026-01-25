#!/usr/bin/env python3
"""
Benchmark comparing scheduling strategies across targets.

Measures:
- Schedule length (cycles)
- Register pressure (max live)
- Spill count
- Estimated throughput

Usage:
    python scripts/bench_schedulers.py
    python scripts/bench_schedulers.py --circuit bp
    python scripts/bench_schedulers.py --compile  # Actually compile and measure
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bp_circuit_sbox import build_bp_sbox
from stc.sched import (
    SSE2,
    SSE2_X64,
    AVX2,
    AVX512,
    PTX,
    list_schedule,
    pipelined_schedule,
    compute_depth,
)
from stc.sched.liveness import compute_live_ranges, max_live
from stc.sched.regalloc import allocate_registers

try:
    from stc.sched.regpressure_scheduler import regpressure_schedule

    HAS_REGPRESSURE = True
except ImportError:
    HAS_REGPRESSURE = False


def make_schedulers(target_registers: int) -> dict:
    schedulers = {
        "list_slack": lambda g, i, o, t: list_schedule(g, i, o, t, "slack"),
        "list_depth": lambda g, i, o, t: list_schedule(g, i, o, t, "depth"),
        "pipelined": pipelined_schedule,
    }
    if HAS_REGPRESSURE:
        schedulers["regpressure"] = (
            lambda g, i, o, t, max_reg=target_registers: regpressure_schedule(
                g, i, o, t, max_reg
            )
        )
    return schedulers


TARGETS = [
    ("SSE2", SSE2),
    ("SSE2-x64", SSE2_X64),
    ("AVX2", AVX2),
    ("AVX-512", AVX512),
    ("PTX", PTX),
]

SCHEDULERS_BASE = {
    "list_slack": lambda g, i, o, t: list_schedule(g, i, o, t, "slack"),
    "list_depth": lambda g, i, o, t: list_schedule(g, i, o, t, "depth"),
    "pipelined": pipelined_schedule,
}


def build_chain_circuit(n: int) -> tuple[list, int, list]:
    gates = []
    for i in range(n):
        if i == 0:
            gates.append(("xor", 0, 1))
        else:
            gates.append(("xor", 2 + i - 1, (i % 2)))
    outputs = [(2 + n - 1, False)]
    return gates, 2, outputs


def build_tree_circuit(depth: int) -> tuple[list, int, list]:
    input_bits = 2**depth
    gates = []
    layer_start = 0
    layer_size = input_bits

    while layer_size > 1:
        new_layer_size = layer_size // 2
        for i in range(new_layer_size):
            if layer_start == 0:
                left = 2 * i
                right = 2 * i + 1
            else:
                left = input_bits + layer_start - layer_size + 2 * i
                right = input_bits + layer_start - layer_size + 2 * i + 1
            gates.append(("xor", left, right))
        layer_start += new_layer_size
        layer_size = new_layer_size

    outputs = [(input_bits + len(gates) - 1, False)]
    return gates, input_bits, outputs


def benchmark_circuit(gates, input_bits, outputs, name="circuit"):
    depth = compute_depth(gates, input_bits)
    print(f"\n{name}: {len(gates)} gates, depth {depth}")
    print("=" * 80)

    print(
        f"{'Target':<12} {'Scheduler':<15} {'Cycles':<8} {'MaxLive':<8} "
        f"{'Spills':<8} {'Util':<8}"
    )
    print("-" * 80)

    for target_name, target in TARGETS:
        schedulers = make_schedulers(target.registers)
        for sched_name, sched_fn in schedulers.items():
            schedule = sched_fn(gates, input_bits, outputs, target)
            live_ranges = compute_live_ranges(schedule, gates, input_bits, outputs)
            ml = max_live(live_ranges)
            allocation = allocate_registers(live_ranges, schedule, target.registers)

            cycles = schedule.total_cycles
            spills = allocation.num_spills
            util = len(gates) / cycles if cycles > 0 else 0

            fits = "ok" if ml <= target.registers else ""
            print(
                f"{target_name:<12} {sched_name:<15} {cycles:<8} {ml:<8} "
                f"{spills:<8} {util:<8.1f} {fits}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark scheduling strategies across targets"
    )
    parser.add_argument(
        "--circuit",
        default="bp",
        choices=["bp", "chain", "tree", "all"],
        help="Circuit to benchmark (default: bp)",
    )
    parser.add_argument(
        "--chain-length",
        type=int,
        default=64,
        help="Length for chain circuit (default: 64)",
    )
    parser.add_argument(
        "--tree-depth",
        type=int,
        default=6,
        help="Depth for tree circuit (default: 6, giving 63 gates)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Actually compile and measure (not yet implemented)",
    )
    args = parser.parse_args()

    if args.compile:
        print("--compile not yet implemented")
        return 1

    print("Scheduler Benchmark")
    print("=" * 80)
    sched_names = list(SCHEDULERS_BASE.keys())
    if HAS_REGPRESSURE:
        sched_names.append("regpressure")
    print(f"Schedulers: {', '.join(sched_names)}")
    print(f"Targets: {', '.join(name for name, _ in TARGETS)}")

    circuits_to_run = []

    if args.circuit in ("bp", "all"):
        circuit = build_bp_sbox()
        circuits_to_run.append(
            (
                list(circuit.gates),
                circuit.input_bits,
                list(circuit.outputs),
                f"BP S-box ({circuit.gate_count} gates)",
            )
        )

    if args.circuit in ("chain", "all"):
        gates, input_bits, outputs = build_chain_circuit(args.chain_length)
        circuits_to_run.append(
            (gates, input_bits, outputs, f"Chain ({args.chain_length} gates)")
        )

    if args.circuit in ("tree", "all"):
        gates, input_bits, outputs = build_tree_circuit(args.tree_depth)
        circuits_to_run.append(
            (
                gates,
                input_bits,
                outputs,
                f"Tree (depth {args.tree_depth}, {len(gates)} gates)",
            )
        )

    for gates, input_bits, outputs, name in circuits_to_run:
        benchmark_circuit(gates, input_bits, outputs, name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
