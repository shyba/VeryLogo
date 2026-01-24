#!/usr/bin/env python3
"""
Test and benchmark AES S-box circuit implementations.

Usage:
    # Test correctness of ANF circuit
    python scripts/test_sbox_circuit.py --test

    # Benchmark evaluation speed
    python scripts/test_sbox_circuit.py --bench --iterations 10000

    # Load and test a saved circuit
    python scripts/test_sbox_circuit.py --test --load circuit.json

    # Generate test vectors
    python scripts/test_sbox_circuit.py --vectors --output test_vectors.json

    # Compare circuit vs table lookup
    python scripts/test_sbox_circuit.py --compare
"""
import argparse
import json
import sys
import time

sys.setrecursionlimit(10000)

from stc.bitslice import AES_SBOX_TABLE
from stc.circuit_synth import IncrementalOptimizer, CircuitState


def test_correctness(opt: IncrementalOptimizer) -> tuple[bool, int, list[int]]:
    """Test circuit correctness for all 256 inputs. Returns (passed, errors, failed_inputs)."""
    errors = 0
    failed_inputs = []

    for i in range(256):
        expected = AES_SBOX_TABLE[i]
        got = opt.best_state.evaluate(i)
        if got != expected:
            errors += 1
            failed_inputs.append(i)
            if errors <= 5:
                print(f"  FAIL: input={i:02x}, expected={expected:02x}, got={got:02x}")

    return errors == 0, errors, failed_inputs


def benchmark_circuit(opt: IncrementalOptimizer, iterations: int) -> dict:
    """Benchmark circuit evaluation speed."""
    state = opt.best_state

    start = time.perf_counter()
    for _ in range(iterations):
        for i in range(256):
            state.evaluate(i)
    elapsed = time.perf_counter() - start

    total_evals = iterations * 256
    evals_per_sec = total_evals / elapsed
    ns_per_eval = (elapsed / total_evals) * 1e9

    return {
        "total_evaluations": total_evals,
        "elapsed_seconds": elapsed,
        "evaluations_per_second": evals_per_sec,
        "nanoseconds_per_evaluation": ns_per_eval,
        "gate_count": state.gate_count,
    }


def benchmark_table_lookup(iterations: int) -> dict:
    """Benchmark direct table lookup speed."""
    table = AES_SBOX_TABLE

    start = time.perf_counter()
    for _ in range(iterations):
        for i in range(256):
            _ = table[i]
    elapsed = time.perf_counter() - start

    total_evals = iterations * 256
    evals_per_sec = total_evals / elapsed
    ns_per_eval = (elapsed / total_evals) * 1e9

    return {
        "total_evaluations": total_evals,
        "elapsed_seconds": elapsed,
        "evaluations_per_second": evals_per_sec,
        "nanoseconds_per_evaluation": ns_per_eval,
    }


def generate_test_vectors() -> list[dict]:
    """Generate test vectors for all 256 inputs."""
    vectors = []
    for i in range(256):
        vectors.append({"input": i, "expected_output": AES_SBOX_TABLE[i]})
    return vectors


def analyze_circuit(opt: IncrementalOptimizer) -> dict:
    """Analyze circuit structure."""
    state = opt.best_state
    gates = state.gates

    xor_count = sum(1 for op, _, _ in gates if op == "xor")
    and_count = sum(1 for op, _, _ in gates if op == "and")
    const_count = sum(1 for op, _, _ in gates if op == "const")
    other_count = len(gates) - xor_count - and_count - const_count

    depth = 0
    depths = {i: 0 for i in range(opt.input_bits)}
    for idx, (op, left, right) in enumerate(gates):
        full_idx = opt.input_bits + idx
        left_depth = depths.get(left, 0)
        right_depth = depths.get(right, 0) if op not in ("const", "not") else 0
        depths[full_idx] = max(left_depth, right_depth) + 1
        depth = max(depth, depths[full_idx])

    return {
        "gate_count": state.gate_count,
        "xor_gates": xor_count,
        "and_gates": and_count,
        "const_gates": const_count,
        "other_gates": other_count,
        "critical_path_depth": depth,
        "input_bits": opt.input_bits,
        "output_bits": opt.output_bits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test and benchmark AES S-box circuits"
    )
    parser.add_argument("--test", action="store_true", help="Test correctness")
    parser.add_argument("--bench", action="store_true", help="Benchmark performance")
    parser.add_argument(
        "--compare", action="store_true", help="Compare circuit vs table"
    )
    parser.add_argument("--vectors", action="store_true", help="Generate test vectors")
    parser.add_argument(
        "--analyze", action="store_true", help="Analyze circuit structure"
    )
    parser.add_argument("--load", type=str, help="Load circuit from JSON file")
    parser.add_argument("--output", type=str, help="Output file for vectors")
    parser.add_argument(
        "--iterations", type=int, default=1000, help="Benchmark iterations"
    )
    args = parser.parse_args()

    if not any([args.test, args.bench, args.compare, args.vectors, args.analyze]):
        args.test = True
        args.analyze = True

    if args.load:
        print(f"Loading circuit from {args.load}...")
        opt = IncrementalOptimizer.load(args.load)
    else:
        print("Generating ANF circuit...")
        opt = IncrementalOptimizer(AES_SBOX_TABLE, input_bits=8, output_bits=8)

    print(f"Circuit: {opt.get_gate_count()} gates")

    if args.analyze:
        print("\n=== Circuit Analysis ===")
        analysis = analyze_circuit(opt)
        print(f"  Total gates: {analysis['gate_count']}")
        print(f"  XOR gates: {analysis['xor_gates']}")
        print(f"  AND gates: {analysis['and_gates']}")
        print(f"  CONST gates: {analysis['const_gates']}")
        print(f"  Critical path depth: {analysis['critical_path_depth']}")

    if args.test:
        print("\n=== Correctness Test ===")
        passed, errors, failed = test_correctness(opt)
        if passed:
            print(f"  PASSED: All 256 inputs correct")
        else:
            print(f"  FAILED: {errors} incorrect outputs")
            if len(failed) > 5:
                print(f"  First 5 failures shown, {len(failed) - 5} more...")

    if args.bench:
        print(f"\n=== Performance Benchmark ({args.iterations} iterations) ===")
        results = benchmark_circuit(opt, args.iterations)
        print(f"  Total evaluations: {results['total_evaluations']:,}")
        print(f"  Elapsed time: {results['elapsed_seconds']:.3f}s")
        print(f"  Evaluations/sec: {results['evaluations_per_second']:,.0f}")
        print(f"  Time per eval: {results['nanoseconds_per_evaluation']:.1f} ns")

    if args.compare:
        print(f"\n=== Comparison: Circuit vs Table ({args.iterations} iterations) ===")

        print("  Benchmarking circuit...")
        circuit_results = benchmark_circuit(opt, args.iterations)

        print("  Benchmarking table lookup...")
        table_results = benchmark_table_lookup(args.iterations)

        print(
            f"\n  Circuit: {circuit_results['nanoseconds_per_evaluation']:.1f} ns/eval"
        )
        print(f"  Table:   {table_results['nanoseconds_per_evaluation']:.1f} ns/eval")

        ratio = circuit_results["nanoseconds_per_evaluation"] / max(
            table_results["nanoseconds_per_evaluation"], 0.001
        )
        print(f"  Ratio:   {ratio:.1f}x slower")

    if args.vectors:
        vectors = generate_test_vectors()
        output_file = args.output or "test_vectors.json"
        with open(output_file, "w") as f:
            json.dump(
                {
                    "description": "AES S-box test vectors",
                    "vectors": vectors,
                    "circuit_gates": opt.get_gate_count(),
                },
                f,
                indent=2,
            )
        print(f"\nTest vectors written to {output_file}")


if __name__ == "__main__":
    main()
