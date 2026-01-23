#!/usr/bin/env python3
"""
Script to synthesize and incrementally optimize AES S-box circuits.

Usage:
    # Start fresh with ANF synthesis
    python scripts/synthesize_sbox.py --anf

    # Run incremental optimization
    python scripts/synthesize_sbox.py --optimize --iterations 100

    # Resume from saved state
    python scripts/synthesize_sbox.py --resume circuit.json --iterations 100

    # Save progress
    python scripts/synthesize_sbox.py --optimize --save circuit.json
"""
import argparse
import sys
import time

sys.setrecursionlimit(10000)

from stc.bitslice import AES_SBOX_TABLE
from stc.circuit_synth import (
    IncrementalOptimizer,
    synthesize_multi_output_anf,
    circuit_to_state,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize and optimize AES S-box circuits"
    )
    parser.add_argument("--anf", action="store_true", help="Generate initial ANF circuit")
    parser.add_argument("--optimize", action="store_true", help="Run incremental optimization")
    parser.add_argument("--anneal", action="store_true", help="Run simulated annealing")
    parser.add_argument("--iterations", type=int, default=100, help="Optimization iterations")
    parser.add_argument("--timeout", type=int, default=5, help="Timeout (seconds for optimize, minutes for anneal)")
    parser.add_argument("--temp", type=float, default=10.0, help="Initial temperature for annealing")
    parser.add_argument("--cooling", type=float, default=0.995, help="Cooling rate for annealing")
    parser.add_argument("--target", type=int, default=None, help="Target gate count")
    parser.add_argument("--resume", type=str, default=None, help="Resume from saved JSON file")
    parser.add_argument("--save", type=str, default=None, help="Save result to JSON file")
    args = parser.parse_args()

    if args.resume:
        print(f"Loading from {args.resume}...")
        opt = IncrementalOptimizer.load(args.resume)
        print(f"Loaded circuit with {opt.get_gate_count()} gates")
        print(f"Previous iterations: {opt.iterations}, improvements: {opt.improvements}")
    elif args.anf or args.optimize:
        print("Generating initial ANF circuit...")
        opt = IncrementalOptimizer(AES_SBOX_TABLE, input_bits=8, output_bits=8)
        print(f"Initial circuit: {opt.get_gate_count()} gates")
    else:
        parser.print_help()
        return

    if not opt.verify():
        print("ERROR: Circuit verification failed!")
        return

    print(f"Circuit verified OK")

    if args.anneal:
        print(f"\nRunning simulated annealing ({args.iterations} iterations)...")
        print(f"Temperature: {args.temp}, cooling: {args.cooling}")

        start_gates = opt.get_gate_count()
        start_time = time.time()

        def anneal_callback(iteration, gates, improved):
            if improved:
                elapsed = time.time() - start_time
                print(f"  [{elapsed:.1f}s] Iteration {iteration}: {gates} gates (improved!)")
            elif iteration % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  [{elapsed:.1f}s] Iteration {iteration}: {gates} gates")

        opt.anneal(
            max_iterations=args.iterations,
            initial_temp=args.temp,
            cooling_rate=args.cooling,
            timeout_seconds=args.timeout * 60,
            callback=anneal_callback,
        )

        elapsed = time.time() - start_time
        final_gates = opt.get_gate_count()
        print(f"\nAnnealing complete in {elapsed:.1f}s")
        print(f"Gates: {start_gates} -> {final_gates} ({start_gates - final_gates} saved)")
        print(f"Total iterations: {opt.iterations}, improvements: {opt.improvements}")

        if not opt.verify():
            print("ERROR: Circuit verification failed after annealing!")
            return

    elif args.optimize or args.resume:
        print(f"\nRunning optimization ({args.iterations} iterations, {args.timeout}s timeout)...")
        if args.target:
            print(f"Target: {args.target} gates")

        start_gates = opt.get_gate_count()
        start_time = time.time()

        def callback(iteration, gates, improved):
            if improved:
                elapsed = time.time() - start_time
                print(f"  [{elapsed:.1f}s] Iteration {iteration}: {gates} gates (improved!)")
            elif iteration % 20 == 0:
                elapsed = time.time() - start_time
                print(f"  [{elapsed:.1f}s] Iteration {iteration}: {gates} gates")

        opt.optimize(
            max_iterations=args.iterations,
            timeout_ms_per_step=args.timeout * 1000,
            target_gates=args.target,
            callback=callback,
        )

        elapsed = time.time() - start_time
        final_gates = opt.get_gate_count()
        print(f"\nOptimization complete in {elapsed:.1f}s")
        print(f"Gates: {start_gates} -> {final_gates} ({start_gates - final_gates} saved)")
        print(f"Total iterations: {opt.iterations}, improvements: {opt.improvements}")

        if not opt.verify():
            print("ERROR: Circuit verification failed after optimization!")
            return

    if args.save:
        opt.save(args.save)
        print(f"Saved to {args.save}")

    print(f"\nFinal circuit: {opt.get_gate_count()} gates")


if __name__ == "__main__":
    main()
