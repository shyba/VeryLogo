import unittest

from scripts.benchmark_ternary_sbox import compute_truth_table, find_3input_cones
from stc.bitslice import AES_SBOX_TABLE
from stc.circuit_synth import CircuitState, IncrementalOptimizer


class TestTernaryConeMappingRegression(unittest.TestCase):
    def test_script_style_cone_mapping_preserves_aes_sbox(self) -> None:
        """
        Regression test for ternary cone mapping:

        The script-style cone finder + imm8 computation used for AVX-512 vpternlogd
        should not change the circuit's truth table.
        """
        opt = IncrementalOptimizer(AES_SBOX_TABLE, input_bits=8, output_bits=8)
        state = opt.best_state

        cones = find_3input_cones(state)
        self.assertGreater(len(cones), 0)

        new_gates = list(state.gates)
        for cone in cones:
            root = cone["root"]
            leaves = cone["leaves"]
            imm8 = compute_truth_table(state, root, leaves)
            new_gates[root] = ("ternary", leaves[0], leaves[1], leaves[2], imm8)

        mapped = CircuitState(
            input_bits=state.input_bits,
            output_bits=state.output_bits,
            gates=new_gates,
            outputs=list(state.outputs),
            gate_count=len(new_gates),
        )

        for x in range(256):
            self.assertEqual(mapped.evaluate(x), AES_SBOX_TABLE[x])
