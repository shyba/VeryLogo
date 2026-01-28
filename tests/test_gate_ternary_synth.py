"""Tests for gate-level ternary synthesis."""

import unittest

from stc.circuit_synth import CircuitState
from stc.gate_ternary_synth import (
    apply_ternary_synthesis,
    compute_imm8,
    compute_imm8_swapped,
    build_gate_info,
    find_ternary_patterns,
)


class TestComputeImm8(unittest.TestCase):
    """Test imm8 computation for known patterns."""

    def test_and_and(self):
        """AND(AND(a,b), c) = a & b & c"""
        imm8 = compute_imm8("and", "and")
        self.assertEqual(imm8, 0x80)

    def test_or_or(self):
        """OR(OR(a,b), c) = a | b | c"""
        imm8 = compute_imm8("or", "or")
        self.assertEqual(imm8, 0xFE)

    def test_xor_xor(self):
        """XOR(XOR(a,b), c) = a ^ b ^ c"""
        imm8 = compute_imm8("xor", "xor")
        self.assertEqual(imm8, 0x96)

    def test_and_or(self):
        """AND(OR(a,b), c) = (a | b) & c"""
        imm8 = compute_imm8("and", "or")
        self.assertEqual(imm8, 0xA8)

    def test_or_and(self):
        """OR(AND(a,b), c) = (a & b) | c"""
        imm8 = compute_imm8("or", "and")
        self.assertEqual(imm8, 0xEA)


class TestTernarySynthesis(unittest.TestCase):
    """Test the full ternary synthesis pass."""

    def test_simple_and_or_pattern(self):
        """Test AND(OR(a,b), c) -> ternary."""
        circuit = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("or", 0, 1),
                ("and", 3, 2),
            ],
            outputs=[(4, False)],
            gate_count=2,
        )

        new_circuit, stats = apply_ternary_synthesis(circuit)

        self.assertEqual(stats.gates_before, 2)
        self.assertEqual(stats.gates_after, 1)
        self.assertEqual(stats.ternary_gates_created, 1)
        self.assertEqual(len(new_circuit.gates), 1)
        self.assertEqual(new_circuit.gates[0][0], "ternary")

    def test_no_pattern_high_fanout(self):
        """Don't merge if inner gate has fanout > 1."""
        circuit = CircuitState(
            input_bits=3,
            output_bits=2,
            gates=[
                ("or", 0, 1),
                ("and", 3, 2),
                ("xor", 3, 2),
            ],
            outputs=[(4, False), (5, False)],
            gate_count=3,
        )

        new_circuit, stats = apply_ternary_synthesis(circuit)

        self.assertEqual(stats.gates_after, 3)
        self.assertEqual(stats.ternary_gates_created, 0)

    def test_chain_of_patterns(self):
        """Multiple patterns in sequence."""
        circuit = CircuitState(
            input_bits=4,
            output_bits=1,
            gates=[
                ("and", 0, 1),
                ("or", 4, 2),
                ("xor", 5, 3),
            ],
            outputs=[(6, False)],
            gate_count=3,
        )

        new_circuit, stats = apply_ternary_synthesis(circuit)

        self.assertLess(stats.gates_after, 3)

    def test_preserves_correctness(self):
        """Verify output is functionally equivalent."""
        circuit = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("or", 0, 1),
                ("and", 3, 2),
            ],
            outputs=[(4, False)],
            gate_count=2,
        )

        new_circuit, _ = apply_ternary_synthesis(circuit)

        for i in range(8):
            original = circuit.evaluate(i)
            optimized = new_circuit.evaluate(i)
            self.assertEqual(original, optimized, f"Mismatch for input {i}")


class TestBuildGateInfo(unittest.TestCase):
    """Test gate info construction."""

    def test_fanout_counting(self):
        """Verify fanout counts are correct."""
        gates = [
            ("or", 0, 1),
            ("and", 3, 2),
            ("xor", 3, 2),
        ]

        infos = build_gate_info(gates, input_bits=3)

        self.assertEqual(infos[0].fanout, 2)
        self.assertEqual(infos[1].fanout, 0)
        self.assertEqual(infos[2].fanout, 0)


class TestAndnotOptimization(unittest.TestCase):

    def test_simple_andnot_pattern(self):
        """AND(NOT(a), b) -> ANDNOT(a, b)."""
        circuit = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("not", 0, 0),
                ("and", 2, 1),
            ],
            outputs=[(3, False)],
            gate_count=2,
        )

        from stc.gate_ternary_synth import apply_andnot_optimization

        new_circuit, stats = apply_andnot_optimization(circuit)

        self.assertEqual(stats["andnot_created"], 1)
        self.assertEqual(len(new_circuit.gates), 1)
        self.assertEqual(new_circuit.gates[0][0], "andn")

    def test_andnot_preserves_correctness(self):
        """Verify ANDNOT output matches original."""
        circuit = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("not", 0, 0),
                ("and", 2, 1),
            ],
            outputs=[(3, False)],
            gate_count=2,
        )

        from stc.gate_ternary_synth import apply_andnot_optimization

        new_circuit, _ = apply_andnot_optimization(circuit)

        for i in range(4):
            original = circuit.evaluate(i)
            optimized = new_circuit.evaluate(i)
            self.assertEqual(original, optimized)


class TestDoubleNotElimination(unittest.TestCase):

    def test_simple_double_not(self):
        """NOT(NOT(x)) -> x."""
        circuit = CircuitState(
            input_bits=1,
            output_bits=1,
            gates=[
                ("not", 0, 0),
                ("not", 1, 0),
            ],
            outputs=[(2, False)],
            gate_count=2,
        )

        from stc.gate_ternary_synth import eliminate_double_nots

        new_circuit, stats = eliminate_double_nots(circuit)

        self.assertEqual(stats["double_nots_eliminated"], 2)
        self.assertEqual(len(new_circuit.gates), 0)
        self.assertEqual(new_circuit.outputs[0][0], 0)

    def test_double_not_preserves_correctness(self):
        """Verify double-NOT elimination preserves behavior."""
        circuit = CircuitState(
            input_bits=1,
            output_bits=1,
            gates=[
                ("not", 0, 0),
                ("not", 1, 0),
            ],
            outputs=[(2, False)],
            gate_count=2,
        )

        from stc.gate_ternary_synth import eliminate_double_nots

        new_circuit, _ = eliminate_double_nots(circuit)

        for i in range(2):
            original = circuit.evaluate(i)
            optimized = new_circuit.evaluate(i)
            self.assertEqual(original, optimized)

    def test_double_not_with_other_gates(self):
        """Double-NOT within larger circuit."""
        circuit = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("not", 0, 0),
                ("not", 2, 0),
                ("xor", 3, 1),
            ],
            outputs=[(4, False)],
            gate_count=3,
        )

        from stc.gate_ternary_synth import eliminate_double_nots

        new_circuit, stats = eliminate_double_nots(circuit)

        self.assertEqual(stats["double_nots_eliminated"], 2)
        self.assertEqual(len(new_circuit.gates), 1)
        self.assertEqual(new_circuit.gates[0][0], "xor")


if __name__ == "__main__":
    unittest.main()
