import unittest

from stc.circuit_synth import CircuitState
from stc.window_synth import (
    Window,
    compute_fanouts,
    compute_depths,
    compute_mffc,
    extract_window,
    compute_truth_table,
    splice_window,
    synthesize_exact,
)


class TestComputeFanouts(unittest.TestCase):
    def test_compute_fanouts_simple(self) -> None:
        """Simple circuit fanout structure."""
        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
            ],
            outputs=[(2, False)],
            gate_count=1,
        )

        fanouts = compute_fanouts(state)

        self.assertEqual(fanouts[0], [2])
        self.assertEqual(fanouts[1], [2])
        self.assertEqual(fanouts[2], [])

    def test_compute_fanouts_multiple_uses(self) -> None:
        """Node used by multiple gates."""
        state = CircuitState(
            input_bits=2,
            output_bits=2,
            gates=[
                ("xor", 0, 1),
                ("and", 0, 1),
            ],
            outputs=[(2, False), (3, False)],
            gate_count=2,
        )

        fanouts = compute_fanouts(state)

        self.assertIn(2, fanouts[0])
        self.assertIn(3, fanouts[0])
        self.assertIn(2, fanouts[1])
        self.assertIn(3, fanouts[1])

    def test_compute_fanouts_chain(self) -> None:
        """Chain of gates: g0 = x0 ^ x1, g1 = g0 ^ x0."""
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

        fanouts = compute_fanouts(state)

        self.assertIn(2, fanouts[0])
        self.assertIn(3, fanouts[0])
        self.assertEqual(fanouts[2], [3])


class TestComputeDepths(unittest.TestCase):
    def test_compute_depths_inputs_zero(self) -> None:
        """Inputs have depth 0."""
        state = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[("xor", 0, 1)],
            outputs=[(3, False)],
            gate_count=1,
        )

        depths = compute_depths(state)

        self.assertEqual(depths[0], 0)
        self.assertEqual(depths[1], 0)
        self.assertEqual(depths[2], 0)

    def test_compute_depths_single_gate(self) -> None:
        """Single gate has depth 1."""
        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[("xor", 0, 1)],
            outputs=[(2, False)],
            gate_count=1,
        )

        depths = compute_depths(state)

        self.assertEqual(depths[2], 1)

    def test_compute_depths_chain(self) -> None:
        """Chain: g0 = x0 ^ x1 (depth 1), g1 = g0 ^ x0 (depth 2)."""
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

        depths = compute_depths(state)

        self.assertEqual(depths[0], 0)
        self.assertEqual(depths[1], 0)
        self.assertEqual(depths[2], 1)
        self.assertEqual(depths[3], 2)

    def test_compute_depths_parallel(self) -> None:
        """Parallel gates have same depth."""
        state = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("and", 1, 2),
                ("xor", 3, 4),
            ],
            outputs=[(5, False)],
            gate_count=3,
        )

        depths = compute_depths(state)

        self.assertEqual(depths[3], 1)
        self.assertEqual(depths[4], 1)
        self.assertEqual(depths[5], 2)


class TestComputeMFFC(unittest.TestCase):
    def test_mffc_simple(self) -> None:
        """Single output MFFC."""
        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[("xor", 0, 1)],
            outputs=[(2, False)],
            gate_count=1,
        )

        fanouts = compute_fanouts(state)
        mffc = compute_mffc(state, 2, fanouts)

        self.assertIn(2, mffc)

    def test_mffc_chain_single_fanout(self) -> None:
        """Chain where each gate has single fanout includes all gates."""
        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("and", 2, 0),
            ],
            outputs=[(3, False)],
            gate_count=2,
        )

        fanouts = compute_fanouts(state)
        mffc = compute_mffc(state, 3, fanouts)

        self.assertIn(3, mffc)
        self.assertIn(2, mffc)

    def test_mffc_shared_node_excluded(self) -> None:
        """Node with external fanout is excluded from MFFC."""
        state = CircuitState(
            input_bits=2,
            output_bits=2,
            gates=[
                ("xor", 0, 1),
                ("and", 2, 0),
            ],
            outputs=[(3, False), (2, False)],
            gate_count=2,
        )

        fanouts = compute_fanouts(state)
        mffc = compute_mffc(state, 3, fanouts)

        self.assertIn(3, mffc)
        self.assertNotIn(2, mffc)

    def test_mffc_input_returns_empty(self) -> None:
        """MFFC of input node returns empty set."""
        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[("xor", 0, 1)],
            outputs=[(2, False)],
            gate_count=1,
        )

        fanouts = compute_fanouts(state)
        mffc = compute_mffc(state, 0, fanouts)

        self.assertEqual(mffc, set())


class TestWindowExtraction(unittest.TestCase):
    def test_window_extraction_simple(self) -> None:
        """Extract valid window from simple circuit."""
        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[("xor", 0, 1)],
            outputs=[(2, False)],
            gate_count=1,
        )

        window = extract_window(state, 2, max_inputs=6)

        self.assertIsNotNone(window)
        self.assertEqual(set(window.inputs), {0, 1})
        self.assertEqual(window.outputs, [2])
        self.assertIn(2, window.internal)

    def test_window_extraction_chain(self) -> None:
        """Extract window from chain circuit."""
        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("and", 2, 0),
            ],
            outputs=[(3, False)],
            gate_count=2,
        )

        window = extract_window(state, 3, max_inputs=6)

        self.assertIsNotNone(window)
        self.assertEqual(set(window.inputs), {0, 1})
        self.assertEqual(window.outputs, [3])
        self.assertIn(3, window.internal)
        self.assertIn(2, window.internal)

    def test_window_extraction_too_many_inputs(self) -> None:
        """Return None if too many inputs."""
        state = CircuitState(
            input_bits=8,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("xor", 2, 3),
                ("xor", 4, 5),
                ("xor", 6, 7),
                ("xor", 8, 9),
                ("xor", 10, 11),
                ("xor", 12, 13),
            ],
            outputs=[(14, False)],
            gate_count=7,
        )

        window = extract_window(state, 14, max_inputs=4)

        self.assertIsNone(window)

    def test_window_extraction_input_node(self) -> None:
        """Return None for input node."""
        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[("xor", 0, 1)],
            outputs=[(2, False)],
            gate_count=1,
        )

        window = extract_window(state, 0, max_inputs=6)

        self.assertIsNone(window)


class TestTruthTable(unittest.TestCase):
    def test_truth_table_xor(self) -> None:
        """Verify truth table for XOR."""
        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[("xor", 0, 1)],
            outputs=[(2, False)],
            gate_count=1,
        )

        tt = compute_truth_table(state, 2, [0, 1])

        self.assertEqual(tt, 0b0110)

    def test_truth_table_and(self) -> None:
        """Verify truth table for AND."""
        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[("and", 0, 1)],
            outputs=[(2, False)],
            gate_count=1,
        )

        tt = compute_truth_table(state, 2, [0, 1])

        self.assertEqual(tt, 0b1000)

    def test_truth_table_or(self) -> None:
        """Verify truth table for OR."""
        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[("or", 0, 1)],
            outputs=[(2, False)],
            gate_count=1,
        )

        tt = compute_truth_table(state, 2, [0, 1])

        self.assertEqual(tt, 0b1110)

    def test_truth_table_not(self) -> None:
        """Verify truth table for NOT."""
        state = CircuitState(
            input_bits=1,
            output_bits=1,
            gates=[("not", 0, 0)],
            outputs=[(1, False)],
            gate_count=1,
        )

        tt = compute_truth_table(state, 1, [0])

        self.assertEqual(tt, 0b01)

    def test_truth_table_chain(self) -> None:
        """Verify truth table for chain: (x0 ^ x1) & x0."""
        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("and", 2, 0),
            ],
            outputs=[(3, False)],
            gate_count=2,
        )

        tt = compute_truth_table(state, 3, [0, 1])

        self.assertEqual(tt, 0b0010)

    def test_truth_table_matches_evaluation(self) -> None:
        """Truth table matches direct circuit evaluation."""
        state = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("and", 3, 2),
            ],
            outputs=[(4, False)],
            gate_count=2,
        )

        tt = compute_truth_table(state, 4, [0, 1, 2])

        for i in range(8):
            expected = state.evaluate(i) & 1
            actual = (tt >> i) & 1
            self.assertEqual(actual, expected, f"Mismatch at input {i}")


class TestSynthesizeExact(unittest.TestCase):
    def test_synth_xor(self) -> None:
        """Synthesize XOR (needs 1 gate)."""
        tt_xor = 0b0110
        result = synthesize_exact([tt_xor], n_inputs=2, max_gates=5, timeout_ms=5000)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "xor")

        self._verify_circuit(result, [tt_xor], n_inputs=2)

    def test_synth_and(self) -> None:
        """Synthesize AND (needs 1 gate)."""
        tt_and = 0b1000
        result = synthesize_exact([tt_and], n_inputs=2, max_gates=5, timeout_ms=5000)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "and")

        self._verify_circuit(result, [tt_and], n_inputs=2)

    def test_synth_xor3(self) -> None:
        """Synthesize 3-input XOR (needs 2 gates)."""
        tt_xor3 = 0b10010110
        result = synthesize_exact([tt_xor3], n_inputs=3, max_gates=5, timeout_ms=5000)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

        self._verify_circuit(result, [tt_xor3], n_inputs=3)

    def test_synth_majority(self) -> None:
        """Synthesize MAJ function (majority of 3 bits)."""
        tt_maj = 0b11101000
        result = synthesize_exact([tt_maj], n_inputs=3, max_gates=10, timeout_ms=10000)
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result), 5)

        self._verify_circuit(result, [tt_maj], n_inputs=3)

    def test_synth_timeout(self) -> None:
        """Handle timeout gracefully."""
        tt_complex = 0b10110100
        result = synthesize_exact([tt_complex], n_inputs=3, max_gates=0, timeout_ms=100)
        self.assertIsNone(result)

    def test_synth_identity(self) -> None:
        """Identity function requires 0 gates."""
        tt_id = 0b10
        result = synthesize_exact([tt_id], n_inputs=1, max_gates=5, timeout_ms=5000)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 0)

    def test_synth_two_outputs(self) -> None:
        """Synthesize circuit with two outputs."""
        tt_a_xor_b = 0b0110
        tt_a_and_b = 0b1000
        result = synthesize_exact(
            [tt_a_xor_b, tt_a_and_b], n_inputs=2, max_gates=5, timeout_ms=5000
        )
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result), 2)

        self._verify_circuit(result, [tt_a_xor_b, tt_a_and_b], n_inputs=2)

    def _verify_circuit(
        self, gates: list[tuple[str, int, int]], targets: list[int], n_inputs: int
    ) -> None:
        """Verify that a synthesized circuit produces the correct truth tables."""
        num_entries = 1 << n_inputs

        input_patterns = []
        for bit in range(n_inputs):
            pattern = sum(((j >> bit) & 1) << j for j in range(num_entries))
            input_patterns.append(pattern)

        node_vals = list(input_patterns)

        for op, left, right in gates:
            left_val = node_vals[left]
            right_val = node_vals[right]
            if op == "xor":
                node_vals.append(left_val ^ right_val)
            elif op == "and":
                node_vals.append(left_val & right_val)
            elif op == "or":
                node_vals.append(left_val | right_val)

        for out_idx, target in enumerate(targets):
            found = False
            for node_val in node_vals:
                if node_val == target:
                    found = True
                    break
            self.assertTrue(found, f"Output {out_idx} not found with target {target}")


class TestSpliceWindow(unittest.TestCase):
    def test_splice_simple(self) -> None:
        """Replace window with optimized version."""
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

        window = extract_window(state, root=3, max_inputs=4)
        self.assertIsNotNone(window)

        tt = window.truth_tables[0]
        new_gates = synthesize_exact([tt], n_inputs=len(window.inputs), max_gates=3)
        self.assertIsNotNone(new_gates)

        spliced = splice_window(state, window, new_gates)

        for i in range(4):
            original = state.evaluate(i)
            new_val = spliced.evaluate(i)
            self.assertEqual(original, new_val, f"Mismatch at input {i}")

    def test_splice_single_gate(self) -> None:
        """Replace a multi-gate window with a single gate."""
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

        window = extract_window(state, root=4, max_inputs=4)
        self.assertIsNotNone(window)

        tt = window.truth_tables[0]
        new_gates = synthesize_exact([tt], n_inputs=len(window.inputs), max_gates=3)
        self.assertIsNotNone(new_gates)

        spliced = splice_window(state, window, new_gates)

        for i in range(8):
            original = state.evaluate(i)
            new_val = spliced.evaluate(i)
            self.assertEqual(original, new_val, f"Mismatch at input {i}")

    def test_splice_preserves_outputs(self) -> None:
        """Splicing preserves circuit output behavior."""
        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("and", 0, 1),
                ("xor", 2, 3),
            ],
            outputs=[(4, False)],
            gate_count=3,
        )

        window = Window(
            inputs=[0, 1],
            outputs=[4],
            internal=[2, 3, 4],
            truth_tables=[0],
        )

        tt = compute_truth_table(state, 4, [0, 1], set([2, 3, 4]))
        window.truth_tables = [tt]

        new_gates = synthesize_exact([tt], n_inputs=2, max_gates=5)
        self.assertIsNotNone(new_gates)

        spliced = splice_window(state, window, new_gates)

        for i in range(4):
            original = state.evaluate(i)
            new_val = spliced.evaluate(i)
            self.assertEqual(original, new_val, f"Mismatch at input {i}")


if __name__ == "__main__":
    unittest.main()
