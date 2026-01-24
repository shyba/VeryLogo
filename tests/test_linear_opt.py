import unittest

from stc.circuit_synth import CircuitState
from stc.linear_opt import (
    LinearCone,
    boyar_peralta_minimize,
    evaluate_linear_cone,
    hamming_distance,
    partition_linear_nonlinear,
)


class TestHammingDistance(unittest.TestCase):
    def test_hamming_same(self) -> None:
        self.assertEqual(hamming_distance(0b1010, 0b1010), 0)

    def test_hamming_all_different(self) -> None:
        self.assertEqual(hamming_distance(0b0000, 0b1111), 4)

    def test_hamming_one_bit(self) -> None:
        self.assertEqual(hamming_distance(0b1000, 0b0000), 1)


class TestBoyarPeraltaMinimize(unittest.TestCase):
    def test_bp_identity_matrix(self) -> None:
        matrix = [0b01, 0b10]
        ops = boyar_peralta_minimize(matrix, n_inputs=2)
        self.assertEqual(ops, [])

    def test_bp_simple_2x2(self) -> None:
        matrix = [0b11, 0b01]
        ops = boyar_peralta_minimize(matrix, n_inputs=2)
        self.assertEqual(len(ops), 1)

    def test_bp_known_case(self) -> None:
        matrix = [0b111, 0b001, 0b010]
        ops = boyar_peralta_minimize(matrix, n_inputs=3)
        self.assertLessEqual(len(ops), 2)

    def test_bp_produces_valid_result(self) -> None:
        matrix = [0b11, 0b10, 0b01]
        n_inputs = 2
        ops = boyar_peralta_minimize(matrix, n_inputs=n_inputs)

        base = [1 << i for i in range(n_inputs)]
        for idx_a, idx_b in ops:
            base.append(base[idx_a] ^ base[idx_b])

        base_set = set(base)
        for target in matrix:
            self.assertIn(target, base_set)

    def test_bp_larger_matrix(self) -> None:
        matrix = [0b1111, 0b1010, 0b0101, 0b1100]
        n_inputs = 4
        ops = boyar_peralta_minimize(matrix, n_inputs=n_inputs)

        base = [1 << i for i in range(n_inputs)]
        for idx_a, idx_b in ops:
            base.append(base[idx_a] ^ base[idx_b])

        base_set = set(base)
        for target in matrix:
            self.assertIn(target, base_set)

    def test_bp_single_target_needs_all_inputs(self) -> None:
        matrix = [0b111]
        n_inputs = 3
        ops = boyar_peralta_minimize(matrix, n_inputs=n_inputs)
        self.assertEqual(len(ops), 2)

        base = [1 << i for i in range(n_inputs)]
        for idx_a, idx_b in ops:
            base.append(base[idx_a] ^ base[idx_b])

        self.assertIn(0b111, set(base))


class TestLinearConeExtraction(unittest.TestCase):
    def test_extract_simple_cone(self) -> None:
        """Single XOR x0 ^ x1."""
        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[("xor", 0, 1)],
            outputs=[(2, False)],
            gate_count=1,
        )
        cone = LinearCone.from_circuit(state, [2], stop_at=set())

        self.assertEqual(sorted(cone.inputs), [0, 1])
        self.assertEqual(cone.outputs, [2])
        self.assertEqual(len(cone.matrix), 1)
        self.assertEqual(cone.matrix[0], 0b11)

    def test_extract_with_and_boundary(self) -> None:
        """XOR cone stops at AND gate."""
        state = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("and", 0, 1),
                ("xor", 3, 2),
            ],
            outputs=[(4, False)],
            gate_count=2,
        )
        and_signal = 3
        cone = LinearCone.from_circuit(state, [4], stop_at={and_signal})

        self.assertEqual(sorted(cone.inputs), [2, 3])
        self.assertEqual(cone.outputs, [4])
        self.assertEqual(len(cone.matrix), 1)
        self.assertEqual(cone.matrix[0], 0b11)

    def test_matrix_representation(self) -> None:
        """Verify matrix correctness for XOR chain."""
        state = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("xor", 3, 2),
            ],
            outputs=[(4, False)],
            gate_count=2,
        )
        cone = LinearCone.from_circuit(state, [4], stop_at=set())

        self.assertEqual(sorted(cone.inputs), [0, 1, 2])
        self.assertEqual(cone.matrix[0], 0b111)

        input_values = {0: 1, 1: 0, 2: 1}
        result = evaluate_linear_cone(cone, input_values)
        expected = 1 ^ 0 ^ 1
        self.assertEqual(result[0], expected)

        input_values = {0: 1, 1: 1, 2: 0}
        result = evaluate_linear_cone(cone, input_values)
        expected = 1 ^ 1 ^ 0
        self.assertEqual(result[0], expected)

    def test_roundtrip(self) -> None:
        """from_circuit -> to_xor_circuit produces equivalent."""
        state = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("xor", 3, 2),
            ],
            outputs=[(4, False)],
            gate_count=2,
        )
        cone = LinearCone.from_circuit(state, [4], stop_at=set())

        gates = cone.to_xor_circuit()
        output_signals = cone.get_output_signals()

        self.assertEqual(len(gates), 2)
        self.assertEqual(gates[0][0], "xor")
        self.assertEqual(gates[1][0], "xor")

        for test_val in range(8):
            input_values = {i: (test_val >> i) & 1 for i in range(3)}
            original_result = evaluate_linear_cone(cone, input_values)

            node_vals = [input_values[i] for i in cone.inputs]
            for op, left, right in gates:
                node_vals.append(node_vals[left] ^ node_vals[right])

            out_signal = output_signals[0]
            reconstructed = node_vals[out_signal]

            self.assertEqual(
                original_result[0],
                reconstructed,
                f"Mismatch at input {test_val}: original={original_result[0]}, reconstructed={reconstructed}",
            )

    def test_xor_cancellation(self) -> None:
        """Test that x ^ x cancels to 0 (empty deps)."""
        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 0),
            ],
            outputs=[(2, False)],
            gate_count=1,
        )
        cone = LinearCone.from_circuit(state, [2], stop_at=set())

        self.assertEqual(cone.matrix[0], 0)

    def test_chained_xor_cancellation(self) -> None:
        """Test (x0 ^ x1) ^ x0 = x1."""
        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("xor", 2, 0),
            ],
            outputs=[(3, False)],
            gate_count=2,
        )
        cone = LinearCone.from_circuit(state, [3], stop_at=set())

        self.assertEqual(cone.inputs, [1])
        self.assertEqual(cone.matrix[0], 0b1)

    def test_not_gate_passthrough(self) -> None:
        """NOT gate should be treated as passthrough for linear analysis."""
        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("not", 2, 0),
            ],
            outputs=[(3, False)],
            gate_count=2,
        )
        cone = LinearCone.from_circuit(state, [3], stop_at=set())

        self.assertEqual(sorted(cone.inputs), [0, 1])
        self.assertEqual(cone.matrix[0], 0b11)


class TestPartitionLinearNonlinear(unittest.TestCase):
    def test_partition_simple(self) -> None:
        """Simple circuit with one AND gate."""
        state = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("and", 2, 3),
                ("xor", 4, 2),
            ],
            outputs=[(5, False)],
            gate_count=3,
        )
        cones, and_gates = partition_linear_nonlinear(state)

        self.assertEqual(len(and_gates), 1)
        self.assertEqual(and_gates[0], 4)

    def test_partition_no_and_gates(self) -> None:
        """Circuit with only XOR gates."""
        state = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("xor", 3, 2),
            ],
            outputs=[(4, False)],
            gate_count=2,
        )
        cones, and_gates = partition_linear_nonlinear(state)

        self.assertEqual(len(and_gates), 0)

        self.assertEqual(len(cones), 1)
        self.assertEqual(cones[0].matrix[0], 0b111)

    def test_partition_multiple_and_gates(self) -> None:
        """Circuit with multiple AND gates."""
        state = CircuitState(
            input_bits=4,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("and", 4, 2),
                ("xor", 5, 3),
                ("and", 6, 0),
            ],
            outputs=[(7, False)],
            gate_count=4,
        )
        cones, and_gates = partition_linear_nonlinear(state)

        self.assertEqual(len(and_gates), 2)
        and_gate_set = set(and_gates)
        self.assertIn(5, and_gate_set)
        self.assertIn(7, and_gate_set)

    def test_partition_input_as_and_operand(self) -> None:
        """AND gate with primary input as operand."""
        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("and", 0, 1),
            ],
            outputs=[(2, False)],
            gate_count=1,
        )
        cones, and_gates = partition_linear_nonlinear(state)

        self.assertEqual(len(and_gates), 1)
        self.assertEqual(and_gates[0], 2)
        self.assertEqual(len(cones), 0)


class TestLinearConeToXorCircuit(unittest.TestCase):
    def test_single_input(self) -> None:
        """Single input passthrough (no gates needed)."""
        cone = LinearCone(inputs=[0], outputs=[2], matrix=[0b1])
        gates = cone.to_xor_circuit()
        self.assertEqual(len(gates), 0)

    def test_two_inputs(self) -> None:
        """Two inputs XORed together."""
        cone = LinearCone(inputs=[0, 1], outputs=[2], matrix=[0b11])
        gates = cone.to_xor_circuit()
        self.assertEqual(len(gates), 1)
        self.assertEqual(gates[0], ("xor", 0, 1))

    def test_three_inputs(self) -> None:
        """Three inputs XORed together."""
        cone = LinearCone(inputs=[0, 1, 2], outputs=[3], matrix=[0b111])
        gates = cone.to_xor_circuit()
        self.assertEqual(len(gates), 2)

        for op, _, _ in gates:
            self.assertEqual(op, "xor")

    def test_multiple_outputs(self) -> None:
        """Multiple outputs with different dependencies."""
        cone = LinearCone(
            inputs=[0, 1, 2],
            outputs=[3, 4],
            matrix=[0b011, 0b110],
        )
        gates = cone.to_xor_circuit()

        self.assertGreaterEqual(len(gates), 2)

    def test_zero_row(self) -> None:
        """Zero row (constant 0) produces no gates."""
        cone = LinearCone(inputs=[0, 1], outputs=[2], matrix=[0])
        gates = cone.to_xor_circuit()
        self.assertEqual(len(gates), 0)


class TestEvaluateLinearCone(unittest.TestCase):
    def test_evaluate_xor(self) -> None:
        """Evaluate simple XOR."""
        cone = LinearCone(inputs=[0, 1], outputs=[2], matrix=[0b11])

        result = evaluate_linear_cone(cone, {0: 0, 1: 0})
        self.assertEqual(result, [0])

        result = evaluate_linear_cone(cone, {0: 1, 1: 0})
        self.assertEqual(result, [1])

        result = evaluate_linear_cone(cone, {0: 0, 1: 1})
        self.assertEqual(result, [1])

        result = evaluate_linear_cone(cone, {0: 1, 1: 1})
        self.assertEqual(result, [0])

    def test_evaluate_three_inputs(self) -> None:
        """Evaluate three-input XOR."""
        cone = LinearCone(inputs=[0, 1, 2], outputs=[3], matrix=[0b111])

        for i in range(8):
            x0 = i & 1
            x1 = (i >> 1) & 1
            x2 = (i >> 2) & 1
            expected = x0 ^ x1 ^ x2
            result = evaluate_linear_cone(cone, {0: x0, 1: x1, 2: x2})
            self.assertEqual(result[0], expected)

    def test_evaluate_partial_deps(self) -> None:
        """Evaluate with partial dependencies (not all inputs used)."""
        cone = LinearCone(inputs=[0, 1, 2], outputs=[3], matrix=[0b101])

        result = evaluate_linear_cone(cone, {0: 1, 1: 1, 2: 1})
        expected = 1 ^ 1
        self.assertEqual(result[0], expected)

        result = evaluate_linear_cone(cone, {0: 0, 1: 1, 2: 1})
        expected = 0 ^ 1
        self.assertEqual(result[0], expected)
