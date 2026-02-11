import unittest

from stc.circuit_synth import CircuitState
from stc.slice_circuit import slice_circuit_by_block_size


class TestSliceCircuit(unittest.TestCase):
    def test_slice_identity_blocks(self) -> None:
        circuit = CircuitState(
            input_bits=16,
            output_bits=16,
            gates=[],
            outputs=[(i, False) for i in range(16)],
            gate_count=0,
        )
        slices = slice_circuit_by_block_size(circuit, 8)
        self.assertIsNotNone(slices)
        assert slices is not None
        self.assertEqual(len(slices), 2)
        self.assertEqual(slices[0].output_offset, 0)
        self.assertEqual(slices[1].output_offset, 8)

    def test_slice_cross_block_rejected(self) -> None:
        # output = input0 xor input9 (crosses 8-bit blocks)
        circuit = CircuitState(
            input_bits=16,
            output_bits=1,
            gates=[("xor", 0, 9)],
            outputs=[(16, False)],
            gate_count=1,
        )
        self.assertIsNone(slice_circuit_by_block_size(circuit, 8))


if __name__ == "__main__":
    unittest.main()
