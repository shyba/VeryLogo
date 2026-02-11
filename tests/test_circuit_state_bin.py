import unittest

from stc.circuit_state_bin import read_circuit_state_bin, write_circuit_state_bin
from stc.circuit_synth import CircuitState


class TestCircuitStateBin(unittest.TestCase):
    def test_roundtrip(self) -> None:
        circuit = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("and", 1, 2),
                ("not", 0, 0),
                ("const", 1, 1),
                ("ternary", 0, 1, 2, 0xCA),
            ],
            outputs=[(5 + 3, True)],
            gate_count=5,
        )
        path = "out/test_circuit_state.bin"
        write_circuit_state_bin(circuit, path)
        loaded = read_circuit_state_bin(path)
        self.assertEqual(circuit.input_bits, loaded.input_bits)
        self.assertEqual(circuit.output_bits, loaded.output_bits)
        self.assertEqual(circuit.gates, loaded.gates)
        self.assertEqual(circuit.outputs, loaded.outputs)

    def test_roundtrip_andn_gate(self) -> None:
        circuit = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[("andn", 0, 1)],
            outputs=[(2, False)],
            gate_count=1,
        )
        path = "out/test_circuit_state_andn.bin"
        write_circuit_state_bin(circuit, path)
        loaded = read_circuit_state_bin(path)
        self.assertEqual(loaded.gates, [("andn", 0, 1)])
