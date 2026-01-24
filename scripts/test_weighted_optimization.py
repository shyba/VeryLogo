#!/usr/bin/env python3
"""
Test weighted gate optimization.

Demonstrates how weighting AND gates higher leads to circuits with fewer ANDs.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stc.circuit_synth import CircuitState
from stc.window_synth import synthesize_exact


def test_weighted_cost():
    """Test the weighted_cost method."""
    # Create a simple circuit with known gate counts
    # x0 XOR x1 AND x2
    circuit = CircuitState(
        input_bits=3,
        output_bits=1,
        gates=[
            ("xor", 0, 1),  # g0 = x0 ^ x1
            ("and", 3, 2),  # g1 = g0 & x2
        ],
        outputs=[(4, False)],
        gate_count=2,
    )

    assert circuit.and_count == 1
    assert circuit.xor_count == 1

    # Equal weights
    assert circuit.weighted_cost(1.0, 1.0) == 2.0

    # AND weight higher
    assert circuit.weighted_cost(10.0, 1.0) == 11.0  # 10*1 + 1*1

    # XOR weight higher
    assert circuit.weighted_cost(1.0, 10.0) == 11.0  # 1*1 + 10*1

    print("weighted_cost method: OK")


def test_weighted_synthesis():
    """Test that weighted synthesis prefers fewer AND gates."""
    # Create truth tables that can be implemented multiple ways
    # A simple function: (x0 AND x1) XOR x2
    # This needs 2 gates: 1 AND + 1 XOR
    # Truth table for 3 inputs:
    # x0 x1 x2 | (x0 & x1) ^ x2
    #  0  0  0 |       0
    #  1  0  0 |       0
    #  0  1  0 |       0
    #  1  1  0 |       1
    #  0  0  1 |       1
    #  1  0  1 |       1
    #  0  1  1 |       1
    #  1  1  1 |       0
    truth_table = 0b01111000  # bits 3,4,5,6 are set

    # Synthesize without weights (default)
    result_default = synthesize_exact(
        [truth_table],
        n_inputs=3,
        max_gates=5,
        timeout_ms=5000,
    )
    print(f"Default synthesis: {result_default}")
    if result_default:
        ands = sum(1 for op, _, _ in result_default if op == "and")
        xors = sum(1 for op, _, _ in result_default if op == "xor")
        print(f"  Gates: {len(result_default)}, ANDs: {ands}, XORs: {xors}")

    # Synthesize with high AND weight (prefer XORs)
    result_weighted = synthesize_exact(
        [truth_table],
        n_inputs=3,
        max_gates=5,
        timeout_ms=5000,
        and_weight=10.0,
        xor_weight=1.0,
    )
    print(f"Weighted synthesis (AND=10, XOR=1): {result_weighted}")
    if result_weighted:
        ands = sum(1 for op, _, _ in result_weighted if op == "and")
        xors = sum(1 for op, _, _ in result_weighted if op == "xor")
        print(f"  Gates: {len(result_weighted)}, ANDs: {ands}, XORs: {xors}")


def test_xor_only_function():
    """Test synthesis of a function that can be XOR-only."""
    # XOR of all 3 inputs: x0 ^ x1 ^ x2
    # Truth table:
    # bits where odd number of inputs are 1
    truth_table = 0b10010110  # 1,2,4,7 have odd parity

    result_default = synthesize_exact(
        [truth_table],
        n_inputs=3,
        max_gates=5,
        timeout_ms=5000,
    )
    print(f"\nXOR-only function:")
    print(f"Default: {result_default}")
    if result_default:
        ands = sum(1 for op, _, _ in result_default if op == "and")
        xors = sum(1 for op, _, _ in result_default if op == "xor")
        wires = sum(1 for op, _, _ in result_default if op == "wire")
        print(f"  Gates: {len(result_default) - wires}, ANDs: {ands}, XORs: {xors}")

    result_weighted = synthesize_exact(
        [truth_table],
        n_inputs=3,
        max_gates=5,
        timeout_ms=5000,
        and_weight=100.0,
        xor_weight=1.0,
    )
    print(f"Weighted (AND=100): {result_weighted}")
    if result_weighted:
        ands = sum(1 for op, _, _ in result_weighted if op == "and")
        xors = sum(1 for op, _, _ in result_weighted if op == "xor")
        wires = sum(1 for op, _, _ in result_weighted if op == "wire")
        print(f"  Gates: {len(result_weighted) - wires}, ANDs: {ands}, XORs: {xors}")


def test_bp_circuit_weighted_optimization():
    """Test weighted optimization on BP circuit."""
    from scripts.bp_circuit_sbox import build_bp_sbox, verify_sbox

    print("\n" + "="*60)
    print("BP Circuit Weighted Optimization Test")
    print("="*60)

    circuit = build_bp_sbox()
    print(f"\nOriginal BP circuit:")
    print(f"  Total gates: {circuit.gate_count}")
    print(f"  AND gates: {circuit.and_count}")
    print(f"  XOR gates: {circuit.xor_count}")
    print(f"  Weighted cost (AND=1, XOR=1): {circuit.weighted_cost(1.0, 1.0)}")
    print(f"  Weighted cost (AND=10, XOR=1): {circuit.weighted_cost(10.0, 1.0)}")

    # Try optimization with default weights (minimize total gates)
    print(f"\nOptimizing with default weights (minimize total gates)...")
    opt_default = circuit.sat_window_resynthesis(
        max_iterations=5,
        max_window_inputs=5,
        timeout_per_window=2000,
    )
    print(f"  After optimization: {opt_default.gate_count} gates "
          f"({opt_default.and_count} AND, {opt_default.xor_count} XOR)")

    # Try optimization with high AND weight (minimize ANDs)
    print(f"\nOptimizing with weighted (AND=10, XOR=1)...")
    opt_weighted = circuit.sat_window_resynthesis(
        max_iterations=5,
        max_window_inputs=5,
        timeout_per_window=2000,
        and_weight=10.0,
        xor_weight=1.0,
    )
    print(f"  After optimization: {opt_weighted.gate_count} gates "
          f"({opt_weighted.and_count} AND, {opt_weighted.xor_count} XOR)")

    # Verify both are still correct
    if not verify_sbox(opt_default, "opt_default"):
        print("ERROR: Default optimization broke correctness!")
    if not verify_sbox(opt_weighted, "opt_weighted"):
        print("ERROR: Weighted optimization broke correctness!")

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Original:     {circuit.gate_count} gates ({circuit.and_count} AND, {circuit.xor_count} XOR)")
    print(f"Default opt:  {opt_default.gate_count} gates ({opt_default.and_count} AND, {opt_default.xor_count} XOR)")
    print(f"Weighted opt: {opt_weighted.gate_count} gates ({opt_weighted.and_count} AND, {opt_weighted.xor_count} XOR)")


if __name__ == "__main__":
    test_weighted_cost()
    print()
    test_weighted_synthesis()
    test_xor_only_function()
    test_bp_circuit_weighted_optimization()
