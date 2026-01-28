from __future__ import annotations

import unittest

from stc.tick_ir import (
    BitVecConst,
    BitVecType,
    Div,
    LShr,
    Mul,
    Shl,
    TickIR,
    Var,
)
from stc.tick_ir_classify_arith import classify_arithmetic
from stc.tick_ir_to_circuit_state import lower_tick_ir_to_circuit_state
from stc.tick_ir_to_packed_circuit_state import (
    PackedLoweringError,
    lower_tick_ir_to_packed_circuit_state,
)


class TestMulDivStrengthReduction(unittest.TestCase):
    def test_mul_by_power_of_2_becomes_shl(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8)},
            outputs={"y": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": Mul(a=Var(name="x"), b=BitVecConst(width=8, value=8))},
        )
        new_ir, report = classify_arithmetic(ir)
        self.assertIsInstance(new_ir.output_exprs["y"], Shl)
        self.assertEqual(new_ir.output_exprs["y"].b.value, 3)
        self.assertEqual(report["Mul"], 0)

    def test_mul_by_power_of_2_left_operand(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8)},
            outputs={"y": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": Mul(a=BitVecConst(width=8, value=16), b=Var(name="x"))},
        )
        new_ir, report = classify_arithmetic(ir)
        self.assertIsInstance(new_ir.output_exprs["y"], Shl)
        self.assertEqual(new_ir.output_exprs["y"].b.value, 4)
        self.assertEqual(report["Mul"], 0)

    def test_mul_by_zero(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8)},
            outputs={"y": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": Mul(a=Var(name="x"), b=BitVecConst(width=8, value=0))},
        )
        new_ir, report = classify_arithmetic(ir)
        self.assertIsInstance(new_ir.output_exprs["y"], BitVecConst)
        self.assertEqual(new_ir.output_exprs["y"].value, 0)
        self.assertEqual(report["Mul"], 0)

    def test_mul_by_one(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8)},
            outputs={"y": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": Mul(a=Var(name="x"), b=BitVecConst(width=8, value=1))},
        )
        new_ir, report = classify_arithmetic(ir)
        self.assertIsInstance(new_ir.output_exprs["y"], Var)
        self.assertEqual(new_ir.output_exprs["y"].name, "x")
        self.assertEqual(report["Mul"], 0)

    def test_mul_non_power_of_2_remains(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8)},
            outputs={"y": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": Mul(a=Var(name="x"), b=BitVecConst(width=8, value=3))},
        )
        new_ir, report = classify_arithmetic(ir)
        self.assertIsInstance(new_ir.output_exprs["y"], Mul)
        self.assertEqual(report["Mul"], 1)

    def test_div_by_power_of_2_becomes_lshr(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8)},
            outputs={"y": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": Div(a=Var(name="x"), b=BitVecConst(width=8, value=4))},
        )
        new_ir, report = classify_arithmetic(ir)
        self.assertIsInstance(new_ir.output_exprs["y"], LShr)
        self.assertEqual(new_ir.output_exprs["y"].b.value, 2)
        self.assertEqual(report["Div"], 0)

    def test_div_by_one(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8)},
            outputs={"y": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": Div(a=Var(name="x"), b=BitVecConst(width=8, value=1))},
        )
        new_ir, report = classify_arithmetic(ir)
        self.assertIsInstance(new_ir.output_exprs["y"], Var)
        self.assertEqual(new_ir.output_exprs["y"].name, "x")
        self.assertEqual(report["Div"], 0)

    def test_div_non_power_of_2_remains(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8)},
            outputs={"y": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": Div(a=Var(name="x"), b=BitVecConst(width=8, value=3))},
        )
        new_ir, report = classify_arithmetic(ir)
        self.assertIsInstance(new_ir.output_exprs["y"], Div)
        self.assertEqual(report["Div"], 1)


class TestMulDivPackedLowering(unittest.TestCase):
    def test_mul_rejected_in_packed_lowering(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8)},
            outputs={"y": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": Mul(a=Var(name="x"), b=BitVecConst(width=8, value=3))},
        )
        with self.assertRaises(PackedLoweringError) as cm:
            lower_tick_ir_to_packed_circuit_state(ir)
        self.assertIn("Mul", str(cm.exception))
        self.assertIn("power-of-2", str(cm.exception))

    def test_div_rejected_in_packed_lowering(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8)},
            outputs={"y": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": Div(a=Var(name="x"), b=BitVecConst(width=8, value=3))},
        )
        with self.assertRaises(PackedLoweringError) as cm:
            lower_tick_ir_to_packed_circuit_state(ir)
        self.assertIn("Div", str(cm.exception))
        self.assertIn("power-of-2", str(cm.exception))


class TestMulDivBitLevelLowering(unittest.TestCase):
    def test_mul_small_width_succeeds(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8), "y": BitVecType(width=8)},
            outputs={"z": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"z": Mul(a=Var(name="x"), b=Var(name="y"))},
        )
        circuit, layout = lower_tick_ir_to_circuit_state(ir)
        self.assertIsNotNone(circuit)
        self.assertEqual(circuit.input_bits, 16)
        self.assertEqual(circuit.output_bits, 8)

    def test_mul_16_bit_succeeds(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=16), "y": BitVecType(width=16)},
            outputs={"z": BitVecType(width=16)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"z": Mul(a=Var(name="x"), b=Var(name="y"))},
        )
        circuit, layout = lower_tick_ir_to_circuit_state(ir)
        self.assertIsNotNone(circuit)

    def test_mul_large_width_rejected(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=32), "y": BitVecType(width=32)},
            outputs={"z": BitVecType(width=32)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"z": Mul(a=Var(name="x"), b=Var(name="y"))},
        )
        with self.assertRaises(Exception) as cm:
            lower_tick_ir_to_circuit_state(ir)
        self.assertIn("16 bits", str(cm.exception))

    def test_div_small_width_succeeds(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8), "y": BitVecType(width=8)},
            outputs={"z": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"z": Div(a=Var(name="x"), b=Var(name="y"))},
        )
        circuit, layout = lower_tick_ir_to_circuit_state(ir)
        self.assertIsNotNone(circuit)
        self.assertEqual(circuit.input_bits, 16)
        self.assertEqual(circuit.output_bits, 8)

    def test_div_16_bit_succeeds(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=16), "y": BitVecType(width=16)},
            outputs={"z": BitVecType(width=16)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"z": Div(a=Var(name="x"), b=Var(name="y"))},
        )
        circuit, layout = lower_tick_ir_to_circuit_state(ir)
        self.assertIsNotNone(circuit)

    def test_div_large_width_rejected(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=32), "y": BitVecType(width=32)},
            outputs={"z": BitVecType(width=32)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"z": Div(a=Var(name="x"), b=Var(name="y"))},
        )
        with self.assertRaises(Exception) as cm:
            lower_tick_ir_to_circuit_state(ir)
        self.assertIn("16 bits", str(cm.exception))


def _eval_circuit_state_once(circuit, in_bits):
    nodes = list(in_bits)
    for gate in circuit.gates:
        if len(gate) == 5:
            op, a, b, c, imm8 = gate
            assert op == "ternary"
            ia = nodes[a] & 1
            ib = nodes[b] & 1
            ic = nodes[c] & 1
            idx = (ic << 2) | (ib << 1) | ia
            nodes.append((imm8 >> idx) & 1)
            continue

        op = gate[0]
        if op == "const":
            nodes.append(gate[1] & 1)
        elif op == "not":
            nodes.append((~nodes[gate[1]]) & 1)
        elif op == "and":
            nodes.append(nodes[gate[1]] & nodes[gate[2]])
        elif op == "or":
            nodes.append(nodes[gate[1]] | nodes[gate[2]])
        elif op == "xor":
            nodes.append(nodes[gate[1]] ^ nodes[gate[2]])
        else:
            raise ValueError(f"unknown gate op: {op}")

    out_bits = []
    for idx, inv in circuit.outputs:
        v = nodes[idx] & 1
        if inv:
            v = (~v) & 1
        out_bits.append(v)
    return out_bits


class TestMulDivCorrectness(unittest.TestCase):
    def test_mul_correctness_4bit(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=4), "y": BitVecType(width=4)},
            outputs={"z": BitVecType(width=4)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"z": Mul(a=Var(name="x"), b=Var(name="y"))},
        )
        circuit, layout = lower_tick_ir_to_circuit_state(ir)

        test_cases = [
            (0, 0, 0),
            (1, 1, 1),
            (2, 3, 6),
            (3, 5, 15),
            (7, 2, 14),
            (15, 1, 15),
            (5, 3, 15),
        ]

        for x, y, expected in test_cases:
            x_bits = [(x >> i) & 1 for i in range(4)]
            y_bits = [(y >> i) & 1 for i in range(4)]
            input_bits = x_bits + y_bits
            output_bits = _eval_circuit_state_once(circuit, input_bits)
            result = sum(bit << i for i, bit in enumerate(output_bits))
            self.assertEqual(result, expected & 0xF, f"mul({x}, {y}) failed")

    def test_div_correctness_4bit(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=4), "y": BitVecType(width=4)},
            outputs={"z": BitVecType(width=4)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"z": Div(a=Var(name="x"), b=Var(name="y"))},
        )
        circuit, layout = lower_tick_ir_to_circuit_state(ir)

        test_cases = [
            (0, 1, 0),
            (1, 1, 1),
            (6, 2, 3),
            (15, 3, 5),
            (14, 7, 2),
            (15, 1, 15),
            (8, 4, 2),
        ]

        for x, y, expected in test_cases:
            x_bits = [(x >> i) & 1 for i in range(4)]
            y_bits = [(y >> i) & 1 for i in range(4)]
            input_bits = x_bits + y_bits
            output_bits = _eval_circuit_state_once(circuit, input_bits)
            result = sum(bit << i for i, bit in enumerate(output_bits))
            self.assertEqual(result, expected, f"div({x}, {y}) failed")


if __name__ == "__main__":
    unittest.main()
