import unittest

from stc.layout_bin import (
    read_packed_layout_bin,
    read_packed_word_layout_bin,
    write_packed_layout_bin,
    write_packed_word_layout_bin,
)
from stc.tick_ir_to_circuit_state import PackedLayout
from stc.tick_ir_to_packed_circuit_state import PackedWordLayout


class TestLayoutBin(unittest.TestCase):
    def test_bit_layout_roundtrip(self) -> None:
        layout = PackedLayout(
            inputs={"a": {"lsb": 0, "width": 4}},
            state={"s": {"lsb": 4, "width": 2}},
            outputs={"y": {"lsb": 0, "width": 1}},
            next_state={"s": {"lsb": 1, "width": 2}},
            input_bits=6,
            output_bits=3,
        )
        path = "out/test_layout.bin"
        write_packed_layout_bin(layout, path)
        loaded = read_packed_layout_bin(path)
        self.assertEqual(layout.inputs, loaded.inputs)
        self.assertEqual(layout.state, loaded.state)
        self.assertEqual(layout.outputs, loaded.outputs)
        self.assertEqual(layout.next_state, loaded.next_state)
        self.assertEqual(layout.input_bits, loaded.input_bits)
        self.assertEqual(layout.output_bits, loaded.output_bits)

    def test_word_layout_roundtrip(self) -> None:
        layout = PackedWordLayout(
            inputs={"a": {"lsw": 0, "width_bits": 64}},
            state={"s": {"lsw": 1, "width_bits": 32}},
            outputs={"y": {"lsw": 0, "width_bits": 16}},
            next_state={"s": {"lsw": 1, "width_bits": 32}},
            input_words=2,
            output_words=2,
        )
        path = "out/test_word_layout.bin"
        write_packed_word_layout_bin(layout, path)
        loaded = read_packed_word_layout_bin(path)
        self.assertEqual(layout.inputs, loaded.inputs)
        self.assertEqual(layout.state, loaded.state)
        self.assertEqual(layout.outputs, loaded.outputs)
        self.assertEqual(layout.next_state, loaded.next_state)
        self.assertEqual(layout.input_words, loaded.input_words)
        self.assertEqual(layout.output_words, loaded.output_words)
