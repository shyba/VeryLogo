import unittest

from stc.tick_ir import (
    BitVecConst,
    BitVecType,
    Concat,
    Slice,
    TickIR,
    Var,
    Xor,
)
from stc.tick_ir_to_circuit_state import lower_tick_ir_to_circuit_state
from stc.tick_ir_to_packed_circuit_state import lower_tick_ir_to_packed_circuit_state


class TestPackedLoweringGateExplosion(unittest.TestCase):
    def test_word_aligned_xor(self) -> None:
        expr = Xor(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="xor64",
            inputs={"x": BitVecType(64), "y": BitVecType(64)},
            outputs={"o": BitVecType(64)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )

        bit_circuit, _ = lower_tick_ir_to_circuit_state(ir)
        packed_circuit, _ = lower_tick_ir_to_packed_circuit_state(ir)

        print(f"\ntest_word_aligned_xor:")
        print(f"  Bit-level gates: {bit_circuit.gate_count}")
        print(f"  Packed gates: {packed_circuit.gate_count}")

        self.assertLess(
            packed_circuit.gate_count,
            bit_circuit.gate_count * 2,
            "Packed XOR should be much more efficient than bit-level",
        )

    def test_single_slice(self) -> None:
        slice_expr = Slice(x=Var("x"), offset=32, width=32)
        ir = TickIR(
            name="slice32",
            inputs={"x": BitVecType(64)},
            outputs={"o": BitVecType(32)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": slice_expr},
        )

        bit_circuit, _ = lower_tick_ir_to_circuit_state(ir)
        packed_circuit, _ = lower_tick_ir_to_packed_circuit_state(ir)

        print(f"\ntest_single_slice:")
        print(f"  Bit-level gates: {bit_circuit.gate_count}")
        print(f"  Packed gates: {packed_circuit.gate_count}")

        self.assertLess(
            packed_circuit.gate_count,
            100,
            "Single word slice should use <100 gates (should be ~5-10 for shift+mask)",
        )

    def test_single_concat(self) -> None:
        concat_expr = Concat(parts=[Var("a"), Var("b")])
        ir = TickIR(
            name="concat32",
            inputs={"a": BitVecType(32), "b": BitVecType(32)},
            outputs={"o": BitVecType(64)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": concat_expr},
        )

        bit_circuit, _ = lower_tick_ir_to_circuit_state(ir)
        packed_circuit, _ = lower_tick_ir_to_packed_circuit_state(ir)

        print(f"\ntest_single_concat:")
        print(f"  Bit-level gates: {bit_circuit.gate_count}")
        print(f"  Packed gates: {packed_circuit.gate_count}")

        self.assertLess(
            packed_circuit.gate_count,
            100,
            "Single word concat should use <100 gates (should be ~5-10 for shift+or)",
        )

    def test_rotate_pattern(self) -> None:
        rotate_expr = Concat(
            parts=[
                Slice(x=Var("x"), offset=0, width=61),
                Slice(x=Var("x"), offset=61, width=3),
            ]
        )
        ir = TickIR(
            name="rotate3",
            inputs={"x": BitVecType(64)},
            outputs={"o": BitVecType(64)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": rotate_expr},
        )

        bit_circuit, _ = lower_tick_ir_to_circuit_state(ir)
        packed_circuit, _ = lower_tick_ir_to_packed_circuit_state(ir)

        print(f"\ntest_rotate_pattern:")
        print(f"  Bit-level gates: {bit_circuit.gate_count}")
        print(f"  Packed gates: {packed_circuit.gate_count}")

        self.assertLess(
            packed_circuit.gate_count,
            200,
            "Rotate pattern should use <200 gates (should be ~3 gates: shl+shr+or)",
        )

    def test_multi_word_xor(self) -> None:
        expr = Xor(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="xor128",
            inputs={"x": BitVecType(128), "y": BitVecType(128)},
            outputs={"o": BitVecType(128)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )

        bit_circuit, _ = lower_tick_ir_to_circuit_state(ir)
        packed_circuit, _ = lower_tick_ir_to_packed_circuit_state(ir)

        print(f"\ntest_multi_word_xor:")
        print(f"  Bit-level gates: {bit_circuit.gate_count}")
        print(f"  Packed gates: {packed_circuit.gate_count}")

        self.assertLess(
            packed_circuit.gate_count,
            bit_circuit.gate_count * 2,
            "Packed multi-word XOR should be much more efficient than bit-level",
        )

    def test_multi_word_slice(self) -> None:
        slice_expr = Slice(x=Var("x"), offset=13, width=96)
        ir = TickIR(
            name="slice_multi",
            inputs={"x": BitVecType(192)},
            outputs={"o": BitVecType(96)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": slice_expr},
        )

        bit_circuit, _ = lower_tick_ir_to_circuit_state(ir)
        packed_circuit, _ = lower_tick_ir_to_packed_circuit_state(ir)

        print(f"\ntest_multi_word_slice:")
        print(f"  Bit-level gates: {bit_circuit.gate_count}")
        print(f"  Packed gates: {packed_circuit.gate_count}")

        self.assertLess(
            packed_circuit.gate_count,
            500,
            "Multi-word slice should use <500 gates",
        )

    def test_multi_word_concat(self) -> None:
        concat_expr = Concat(parts=[Var("a"), Var("b")])
        ir = TickIR(
            name="concat_multi",
            inputs={"a": BitVecType(100), "b": BitVecType(100)},
            outputs={"o": BitVecType(200)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": concat_expr},
        )

        bit_circuit, _ = lower_tick_ir_to_circuit_state(ir)
        packed_circuit, _ = lower_tick_ir_to_packed_circuit_state(ir)

        print(f"\ntest_multi_word_concat:")
        print(f"  Bit-level gates: {bit_circuit.gate_count}")
        print(f"  Packed gates: {packed_circuit.gate_count}")

        self.assertLess(
            packed_circuit.gate_count,
            500,
            "Multi-word concat should use <500 gates",
        )

    def test_complex_expression(self) -> None:
        slice1 = Slice(x=Var("x"), offset=0, width=64)
        slice2 = Slice(x=Var("x"), offset=64, width=64)
        xor_expr = Xor(a=slice1, b=slice2)
        concat_expr = Concat(parts=[xor_expr, slice1])
        ir = TickIR(
            name="complex",
            inputs={"x": BitVecType(128)},
            outputs={"o": BitVecType(128)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": concat_expr},
        )

        bit_circuit, _ = lower_tick_ir_to_circuit_state(ir)
        packed_circuit, _ = lower_tick_ir_to_packed_circuit_state(ir)

        print(f"\ntest_complex_expression:")
        print(f"  Bit-level gates: {bit_circuit.gate_count}")
        print(f"  Packed gates: {packed_circuit.gate_count}")
        print(
            f"  Ratio: {packed_circuit.gate_count / max(1, bit_circuit.gate_count):.2f}x"
        )

        self.assertLess(
            packed_circuit.gate_count,
            1000,
            "Complex expression should use <1000 gates",
        )

    def test_keccak_like_pattern(self) -> None:
        parts = []
        for i in range(25):
            parts.append(Slice(x=Var(f"s{i}"), offset=0, width=64))

        xor_tree = parts[0]
        for part in parts[1:]:
            xor_tree = Xor(a=xor_tree, b=part)

        outputs_dict = {}
        for i in range(25):
            outputs_dict[f"o{i}"] = BitVecType(64)

        output_exprs_dict = {}
        for i in range(25):
            output_exprs_dict[f"o{i}"] = xor_tree

        inputs_dict = {}
        for i in range(25):
            inputs_dict[f"s{i}"] = BitVecType(64)

        ir = TickIR(
            name="keccak_like",
            inputs=inputs_dict,
            outputs=outputs_dict,
            state={},
            reset_state={},
            next_state={},
            output_exprs=output_exprs_dict,
        )

        bit_circuit, _ = lower_tick_ir_to_circuit_state(ir)
        packed_circuit, _ = lower_tick_ir_to_packed_circuit_state(ir)

        print(f"\ntest_keccak_like_pattern:")
        print(f"  Bit-level gates: {bit_circuit.gate_count}")
        print(f"  Packed gates: {packed_circuit.gate_count}")
        print(
            f"  Ratio: {packed_circuit.gate_count / max(1, bit_circuit.gate_count):.2f}x"
        )

        self.assertLess(
            packed_circuit.gate_count,
            bit_circuit.gate_count * 3,
            f"Keccak-like pattern should not explode: {packed_circuit.gate_count} vs {bit_circuit.gate_count}",
        )


if __name__ == "__main__":
    unittest.main()
