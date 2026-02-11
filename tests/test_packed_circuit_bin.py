import unittest

from stc.packed_circuit import PackedCircuitState
from stc.packed_circuit_bin import read_packed_circuit_bin, write_packed_circuit_bin


class TestPackedCircuitBin(unittest.TestCase):
    def test_roundtrip(self) -> None:
        circuit = PackedCircuitState(
            word_bits=64,
            input_words=2,
            output_words=1,
            gates=(
                ("xor", 0, 1),
                ("and", 0, 1),
                ("not", 0, 0),
                ("const", 0x1234, 0),
                ("shl", 0, 3, 0),
                ("lshr", 1, 2, 0),
            ),
            outputs=((2 + 2, False),),
        )
        path = "out/test_packed_circuit.bin"
        write_packed_circuit_bin(circuit, path)
        loaded = read_packed_circuit_bin(path)
        self.assertEqual(circuit.word_bits, loaded.word_bits)
        self.assertEqual(circuit.input_words, loaded.input_words)
        self.assertEqual(circuit.output_words, loaded.output_words)
        self.assertEqual(circuit.gates, loaded.gates)
        self.assertEqual(circuit.outputs, loaded.outputs)
