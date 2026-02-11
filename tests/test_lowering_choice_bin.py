import unittest

from stc.lowering_choice_bin import (
    read_lowering_choice_bin,
    write_lowering_choice_bin,
)
from stc.lowering_coordinator import LoweringChoice


class TestLoweringChoiceBin(unittest.TestCase):
    def test_roundtrip(self) -> None:
        choice = LoweringChoice(
            path="bit",
            reason="fallback",
            unsupported=["Add", "Mul"],
            stats={"input_bits": 8, "gates": 42},
            gate_comparison={"packed_gates": 10, "bitsliced_gates": 100, "ratio": 0.1},
        )
        path = "out/test_lowering_choice.bin"
        write_lowering_choice_bin(choice, path)
        loaded = read_lowering_choice_bin(path)
        self.assertEqual(choice.path, loaded.path)
        self.assertEqual(choice.reason, loaded.reason)
        self.assertEqual(choice.unsupported, loaded.unsupported)
        self.assertEqual(choice.stats, loaded.stats)
        self.assertEqual(
            list(choice.gate_comparison.keys()),
            list(loaded.gate_comparison.keys()),
        )
