import tempfile
import unittest
from pathlib import Path

from stc.circuit_synth import CircuitState
from stc.layout_bin import read_packed_word_layout_bin, write_packed_word_layout_bin
from stc.packed_bitslice import bitslice_circuit_to_packed
from stc.tick_ir_to_circuit_state import PackedLayout
from stc.tick_ir_to_packed_circuit_state import PackedWordLayout


class TestPackedWordLayout(unittest.TestCase):
    def test_io_words_packed(self) -> None:
        layout = PackedWordLayout(
            inputs={
                "a": {"lsw": 0, "width_bits": 65},
                "b": {"lsw": 2, "width_bits": 7},
            },
            state={},
            outputs={"y": {"lsw": 0, "width_bits": 129}},
            next_state={},
            input_words=0,
            output_words=0,
            mode="packed",
        )
        input_io, output_io = layout.io_words()
        self.assertEqual(input_io, 3)
        self.assertEqual(output_io, 3)

    def test_io_words_bitslice(self) -> None:
        layout = PackedWordLayout(
            inputs={
                "a": {"lsw": 0, "width_bits": 65},
                "b": {"lsw": 2, "width_bits": 7},
            },
            state={},
            outputs={"y": {"lsw": 0, "width_bits": 129}},
            next_state={},
            input_words=0,
            output_words=0,
            mode="bitslice",
        )
        input_io, output_io = layout.io_words()
        self.assertEqual(input_io, 72)
        self.assertEqual(output_io, 129)

    def test_layout_bin_roundtrip_mode(self) -> None:
        layout = PackedWordLayout(
            inputs={"a": {"lsw": 0, "width_bits": 2}},
            state={},
            outputs={"y": {"lsw": 0, "width_bits": 1}},
            next_state={},
            input_words=2,
            output_words=1,
            mode="bitslice",
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "layout.bin"
            write_packed_word_layout_bin(layout, path)
            loaded = read_packed_word_layout_bin(path)
        self.assertEqual(loaded.mode, "bitslice")
        self.assertEqual(loaded.input_words, 2)
        self.assertEqual(loaded.output_words, 1)


class TestPackedBitsliceConversion(unittest.TestCase):
    def test_andn_remap(self) -> None:
        circuit = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[("andn", 0, 1)],
            outputs=[(2, False)],
            gate_count=1,
        )
        layout = PackedLayout(
            inputs={"a": {"lsb": 0, "width": 1}, "b": {"lsb": 1, "width": 1}},
            state={},
            outputs={"y": {"lsb": 0, "width": 1}},
            next_state={},
            input_bits=2,
            output_bits=1,
        )
        packed, packed_layout = bitslice_circuit_to_packed(circuit, layout)
        self.assertEqual(packed.gates[0][0], "andnot")
        self.assertEqual(packed_layout.mode, "bitslice")
        self.assertEqual(packed_layout.io_words(), (2, 1))


if __name__ == "__main__":
    unittest.main()
