#!/usr/bin/env python3
"""
Test scheduling of the Boyar-Peralta AES S-box circuit.

Compares schedule quality across different targets and strategies.
"""

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
    ListScheduler,
    PipelinedScheduler,
    list_schedule,
    pipelined_schedule,
    compute_asap,
    compute_depth,
)
from stc.sched.analysis import critical_path


def main():
    print("Building BP tower field S-box circuit...")
    circuit = build_bp_sbox()

    gates = list(circuit.gates)
    input_bits = circuit.input_bits
    outputs = list(circuit.outputs)

    print(f"\nCircuit Statistics:")
    print(f"  Gates: {len(gates)} ({circuit.and_count} AND, {circuit.xor_count} XOR)")
    print(f"  Inputs: {input_bits}")
    print(f"  Outputs: {len(outputs)}")

    depth = compute_depth(gates, input_bits)
    print(f"  Depth: {depth}")

    print("\n" + "=" * 70)
    print("List Scheduler Results")
    print("=" * 70)

    targets = [
        ("SSE2 (32-bit)", SSE2),
        ("SSE2 (64-bit)", SSE2_X64),
        ("AVX2", AVX2),
        ("AVX-512", AVX512),
        ("PTX (CUDA)", PTX),
    ]

    strategies = ["slack", "depth", "fifo"]

    print(
        f"\n{'Target':<15} {'Strategy':<10} {'Cycles':<8} {'Max Live':<10} {'Avg Par':<10}"
    )
    print("-" * 60)

    for target_name, target in targets:
        for strategy in strategies:
            scheduler = ListScheduler(priority_fn=strategy)
            schedule = scheduler.schedule(gates, input_bits, outputs, target)
            stats = schedule.compute_stats(gates, input_bits)

            errors = schedule.validate(gates, input_bits, target.latencies)
            valid = "✓" if not errors else f"✗ ({len(errors)} errors)"

            print(
                f"{target_name:<15} {strategy:<10} {stats.total_cycles:<8} "
                f"{stats.max_live:<10} {stats.avg_parallelism:<10.1f} {valid}"
            )

    print("\n" + "=" * 70)
    print("Pipelined Scheduler Results")
    print("=" * 70)

    print(f"\n{'Target':<15} {'Cycles':<8} {'Max Live':<10} {'Avg Par':<10}")
    print("-" * 50)

    for target_name, target in targets:
        scheduler = PipelinedScheduler()
        schedule = scheduler.schedule(gates, input_bits, outputs, target)
        stats = schedule.compute_stats(gates, input_bits)

        errors = schedule.validate(gates, input_bits, target.latencies)
        valid = "✓" if not errors else f"✗ ({len(errors)} errors)"

        print(
            f"{target_name:<15} {stats.total_cycles:<8} "
            f"{stats.max_live:<10} {stats.avg_parallelism:<10.1f} {valid}"
        )

    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)

    print(f"\nCircuit depth: {depth} cycles (theoretical minimum)")
    print(f"Circuit has {len(gates)} gates")

    for target_name, target in targets:
        schedule = list_schedule(gates, input_bits, outputs, target, "slack")
        stats = schedule.compute_stats(gates, input_bits)

        print(f"\n{target_name}:")
        print(f"  Registers: {target.registers}")
        print(f"  Schedule cycles: {stats.total_cycles}")
        print(f"  Max live values: {stats.max_live}")
        print(f"  Utilization: {len(gates) / stats.total_cycles:.1f} gates/cycle")

        if stats.max_live > target.registers:
            print(f"  ⚠️  SPILLS NEEDED: {stats.max_live - target.registers} extra regs")
        else:
            print(f"  ✓ Fits in {target.registers} registers")

    print("\n" + "=" * 70)
    print("Gates per Cycle Distribution (AVX2, slack)")
    print("=" * 70)

    schedule = list_schedule(gates, input_bits, outputs, AVX2, "slack")
    stats = schedule.compute_stats(gates, input_bits)

    for cycle, count in enumerate(stats.gates_per_cycle):
        bar = "█" * count + "░" * (10 - min(count, 10))
        print(f"  Cycle {cycle:2d}: {bar} {count}")


if __name__ == "__main__":
    main()
