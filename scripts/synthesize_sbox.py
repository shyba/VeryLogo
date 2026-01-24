#!/usr/bin/env python3
"""
AES S-box Circuit Synthesizer and Optimizer

Usage:
    python scripts/synthesize_sbox.py                    # Basic synthesis
    python scripts/synthesize_sbox.py --optimize         # Run all optimizations
    python scripts/synthesize_sbox.py --checkpoint best.json --optimize
    python scripts/synthesize_sbox.py --resume best.json --optimize
"""

import argparse
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.setrecursionlimit(10000)

from stc.bitslice import AES_SBOX_TABLE
from stc.circuit_synth import (
    CircuitState,
    IncrementalOptimizer,
)

best_circuit: CircuitState | None = None
checkpoint_file: str | None = None


def handle_interrupt(signum, frame):
    """Save best circuit on Ctrl+C."""
    global best_circuit, checkpoint_file
    if best_circuit and checkpoint_file:
        print(
            f"\nInterrupted. Saving best ({best_circuit.gate_count} gates) to {checkpoint_file}"
        )
        save_circuit(best_circuit, checkpoint_file)
    sys.exit(0)


def save_circuit(circuit: CircuitState, filepath: str):
    """Save circuit to JSON."""
    import json

    with open(filepath, "w") as f:
        json.dump(circuit.to_dict(), f)


def load_circuit(filepath: str) -> CircuitState:
    """Load circuit from JSON."""
    import json

    with open(filepath) as f:
        data = json.load(f)
    return CircuitState.from_dict(data)


def main():
    parser = argparse.ArgumentParser(description="AES S-box Circuit Optimizer")
    parser.add_argument("--optimize", action="store_true", help="Run optimization")
    parser.add_argument("--checkpoint", type=str, help="Save progress to file")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint file")
    parser.add_argument(
        "--iterations", type=int, default=100, help="Optimization iterations"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    global best_circuit, checkpoint_file
    checkpoint_file = args.checkpoint
    signal.signal(signal.SIGINT, handle_interrupt)

    if args.resume:
        circuit = load_circuit(args.resume)
        print(f"Resumed from {args.resume}: {circuit.gate_count} gates")
    else:
        opt = IncrementalOptimizer(AES_SBOX_TABLE, input_bits=8, output_bits=8)
        circuit = opt.best_state
        print(
            f"Initial ANF: {circuit.gate_count} gates ({circuit.and_count} AND, {circuit.xor_count} XOR)"
        )

    best_circuit = circuit

    if args.optimize:
        print("Running Phase 1 optimizations...")
        circuit = circuit.eliminate_common_subexpressions()
        circuit = circuit.apply_algebraic_rewrites()
        circuit = circuit.eliminate_dead_code()
        circuit = circuit.flatten_xor_trees()
        circuit = circuit.try_local_rewrites()
        print(
            f"After Phase 1: {circuit.gate_count} gates ({circuit.and_count} AND, {circuit.xor_count} XOR)"
        )
        best_circuit = circuit

        if args.iterations > 0:
            print(f"Running Phase 3 SAT resynthesis ({args.iterations} iterations)...")
            circuit = circuit.sat_window_resynthesis(
                max_iterations=args.iterations, checkpoint_file=args.checkpoint
            )
            print(
                f"After Phase 3: {circuit.gate_count} gates ({circuit.and_count} AND, {circuit.xor_count} XOR)"
            )
            best_circuit = circuit

    print("Verifying correctness...")
    for i in range(256):
        if best_circuit.evaluate(i) != AES_SBOX_TABLE[i]:
            print(f"ERROR: Mismatch at input {i}")
            sys.exit(1)
    print("All 256 inputs verified correct")

    if checkpoint_file:
        save_circuit(best_circuit, checkpoint_file)
        print(f"Saved to {checkpoint_file}")


if __name__ == "__main__":
    main()
