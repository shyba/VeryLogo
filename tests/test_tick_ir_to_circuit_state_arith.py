import unittest
import random

from stc.tick_ir import (
    TickIR,
    BitVecType,
    BoolType,
    Var,
    Add,
    Sub,
    BitVecConst,
)
from stc.tick_ir_to_circuit_state import lower_tick_ir_to_circuit_state


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
            nodes.append((nodes[gate[1]] ^ 1) & 1)
        elif op == "xor":
            nodes.append((nodes[gate[1]] ^ nodes[gate[2]]) & 1)
        elif op == "and":
            nodes.append((nodes[gate[1]] & nodes[gate[2]]) & 1)
        elif op == "or":
            nodes.append((nodes[gate[1]] | nodes[gate[2]]) & 1)
        else:
            raise AssertionError(f"unexpected gate op: {op}")
    return [nodes[idx] ^ (1 if inv else 0) for idx, inv in circuit.outputs]


def _bits_to_int(bits):
    return sum((bit & 1) << i for i, bit in enumerate(bits))


def _int_to_bits(val, width):
    return [(val >> i) & 1 for i in range(width)]


class TestTickIrToCircuitStateArith(unittest.TestCase):
    def test_add_4bit(self):
        a = Var("a")
        b = Var("b")
        o_expr = Add(a=a, b=b)

        ir = TickIR(
            name="add4",
            inputs={"a": BitVecType(4), "b": BitVecType(4)},
            outputs={"o": BitVecType(4)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": o_expr},
        )

        circuit, layout = lower_tick_ir_to_circuit_state(ir)
        self.assertEqual(layout.input_bits, 8)
        self.assertEqual(layout.output_bits, 4)

        for a_val in range(16):
            for b_val in range(16):
                in_bits = _int_to_bits(a_val, 4) + _int_to_bits(b_val, 4)
                out_bits = _eval_circuit_state_once(circuit, in_bits)
                o_val = _bits_to_int(out_bits[:4])
                expected = (a_val + b_val) & 0xF
                self.assertEqual(o_val, expected)

    def test_sub_4bit(self):
        a = Var("a")
        b = Var("b")
        o_expr = Sub(a=a, b=b)

        ir = TickIR(
            name="sub4",
            inputs={"a": BitVecType(4), "b": BitVecType(4)},
            outputs={"o": BitVecType(4)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": o_expr},
        )

        circuit, layout = lower_tick_ir_to_circuit_state(ir)
        self.assertEqual(layout.input_bits, 8)
        self.assertEqual(layout.output_bits, 4)

        for a_val in range(16):
            for b_val in range(16):
                in_bits = _int_to_bits(a_val, 4) + _int_to_bits(b_val, 4)
                out_bits = _eval_circuit_state_once(circuit, in_bits)
                o_val = _bits_to_int(out_bits[:4])
                expected = (a_val - b_val) & 0xF
                self.assertEqual(o_val, expected)

    def test_add_8bit_exhaustive(self):
        a = Var("a")
        b = Var("b")
        o_expr = Add(a=a, b=b)

        ir = TickIR(
            name="add8",
            inputs={"a": BitVecType(8), "b": BitVecType(8)},
            outputs={"o": BitVecType(8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": o_expr},
        )

        circuit, layout = lower_tick_ir_to_circuit_state(ir)
        self.assertEqual(layout.input_bits, 16)
        self.assertEqual(layout.output_bits, 8)

        test_cases = [
            (0, 0),
            (255, 255),
            (0, 255),
            (255, 0),
            (1, 1),
            (127, 1),
            (128, 128),
        ]

        random.seed(42)
        for _ in range(100):
            test_cases.append((random.randint(0, 255), random.randint(0, 255)))

        for a_val, b_val in test_cases:
            in_bits = _int_to_bits(a_val, 8) + _int_to_bits(b_val, 8)
            out_bits = _eval_circuit_state_once(circuit, in_bits)
            o_val = _bits_to_int(out_bits[:8])
            expected = (a_val + b_val) & 0xFF
            self.assertEqual(o_val, expected, f"add({a_val}, {b_val})")

    def test_sub_8bit_exhaustive(self):
        a = Var("a")
        b = Var("b")
        o_expr = Sub(a=a, b=b)

        ir = TickIR(
            name="sub8",
            inputs={"a": BitVecType(8), "b": BitVecType(8)},
            outputs={"o": BitVecType(8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": o_expr},
        )

        circuit, layout = lower_tick_ir_to_circuit_state(ir)
        self.assertEqual(layout.input_bits, 16)
        self.assertEqual(layout.output_bits, 8)

        test_cases = [
            (0, 0),
            (255, 255),
            (0, 255),
            (255, 0),
            (1, 1),
            (127, 1),
            (128, 128),
            (0, 1),
            (100, 200),
        ]

        random.seed(42)
        for _ in range(100):
            test_cases.append((random.randint(0, 255), random.randint(0, 255)))

        for a_val, b_val in test_cases:
            in_bits = _int_to_bits(a_val, 8) + _int_to_bits(b_val, 8)
            out_bits = _eval_circuit_state_once(circuit, in_bits)
            o_val = _bits_to_int(out_bits[:8])
            expected = (a_val - b_val) & 0xFF
            self.assertEqual(o_val, expected, f"sub({a_val}, {b_val})")

    def test_add_16bit_random(self):
        a = Var("a")
        b = Var("b")
        o_expr = Add(a=a, b=b)

        ir = TickIR(
            name="add16",
            inputs={"a": BitVecType(16), "b": BitVecType(16)},
            outputs={"o": BitVecType(16)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": o_expr},
        )

        circuit, layout = lower_tick_ir_to_circuit_state(ir)
        self.assertEqual(layout.input_bits, 32)
        self.assertEqual(layout.output_bits, 16)

        test_cases = [
            (0, 0),
            (0xFFFF, 0xFFFF),
            (0x8000, 0x8000),
            (0x7FFF, 1),
            (0x1234, 0x5678),
        ]

        random.seed(42)
        for _ in range(50):
            test_cases.append((random.randint(0, 0xFFFF), random.randint(0, 0xFFFF)))

        for a_val, b_val in test_cases:
            in_bits = _int_to_bits(a_val, 16) + _int_to_bits(b_val, 16)
            out_bits = _eval_circuit_state_once(circuit, in_bits)
            o_val = _bits_to_int(out_bits[:16])
            expected = (a_val + b_val) & 0xFFFF
            self.assertEqual(o_val, expected, f"add({a_val}, {b_val})")

    def test_sub_16bit_random(self):
        a = Var("a")
        b = Var("b")
        o_expr = Sub(a=a, b=b)

        ir = TickIR(
            name="sub16",
            inputs={"a": BitVecType(16), "b": BitVecType(16)},
            outputs={"o": BitVecType(16)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": o_expr},
        )

        circuit, layout = lower_tick_ir_to_circuit_state(ir)
        self.assertEqual(layout.input_bits, 32)
        self.assertEqual(layout.output_bits, 16)

        test_cases = [
            (0, 0),
            (0xFFFF, 0xFFFF),
            (0x8000, 0x8000),
            (0x7FFF, 1),
            (0x1234, 0x5678),
            (0, 1),
            (100, 200),
        ]

        random.seed(42)
        for _ in range(50):
            test_cases.append((random.randint(0, 0xFFFF), random.randint(0, 0xFFFF)))

        for a_val, b_val in test_cases:
            in_bits = _int_to_bits(a_val, 16) + _int_to_bits(b_val, 16)
            out_bits = _eval_circuit_state_once(circuit, in_bits)
            o_val = _bits_to_int(out_bits[:16])
            expected = (a_val - b_val) & 0xFFFF
            self.assertEqual(o_val, expected, f"sub({a_val}, {b_val})")

    def test_add_32bit_random(self):
        a = Var("a")
        b = Var("b")
        o_expr = Add(a=a, b=b)

        ir = TickIR(
            name="add32",
            inputs={"a": BitVecType(32), "b": BitVecType(32)},
            outputs={"o": BitVecType(32)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": o_expr},
        )

        circuit, layout = lower_tick_ir_to_circuit_state(ir)
        self.assertEqual(layout.input_bits, 64)
        self.assertEqual(layout.output_bits, 32)

        test_cases = [
            (0, 0),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x80000000, 0x80000000),
            (0x7FFFFFFF, 1),
            (0x12345678, 0x9ABCDEF0),
        ]

        random.seed(42)
        for _ in range(30):
            test_cases.append(
                (random.randint(0, 0xFFFFFFFF), random.randint(0, 0xFFFFFFFF))
            )

        for a_val, b_val in test_cases:
            in_bits = _int_to_bits(a_val, 32) + _int_to_bits(b_val, 32)
            out_bits = _eval_circuit_state_once(circuit, in_bits)
            o_val = _bits_to_int(out_bits[:32])
            expected = (a_val + b_val) & 0xFFFFFFFF
            self.assertEqual(o_val, expected, f"add({a_val}, {b_val})")

    def test_sub_32bit_random(self):
        a = Var("a")
        b = Var("b")
        o_expr = Sub(a=a, b=b)

        ir = TickIR(
            name="sub32",
            inputs={"a": BitVecType(32), "b": BitVecType(32)},
            outputs={"o": BitVecType(32)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": o_expr},
        )

        circuit, layout = lower_tick_ir_to_circuit_state(ir)
        self.assertEqual(layout.input_bits, 64)
        self.assertEqual(layout.output_bits, 32)

        test_cases = [
            (0, 0),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x80000000, 0x80000000),
            (0x7FFFFFFF, 1),
            (0x12345678, 0x9ABCDEF0),
            (0, 1),
            (100, 200),
        ]

        random.seed(42)
        for _ in range(30):
            test_cases.append(
                (random.randint(0, 0xFFFFFFFF), random.randint(0, 0xFFFFFFFF))
            )

        for a_val, b_val in test_cases:
            in_bits = _int_to_bits(a_val, 32) + _int_to_bits(b_val, 32)
            out_bits = _eval_circuit_state_once(circuit, in_bits)
            o_val = _bits_to_int(out_bits[:32])
            expected = (a_val - b_val) & 0xFFFFFFFF
            self.assertEqual(o_val, expected, f"sub({a_val}, {b_val})")

    def test_add_64bit_random(self):
        a = Var("a")
        b = Var("b")
        o_expr = Add(a=a, b=b)

        ir = TickIR(
            name="add64",
            inputs={"a": BitVecType(64), "b": BitVecType(64)},
            outputs={"o": BitVecType(64)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": o_expr},
        )

        circuit, layout = lower_tick_ir_to_circuit_state(ir)
        self.assertEqual(layout.input_bits, 128)
        self.assertEqual(layout.output_bits, 64)

        test_cases = [
            (0, 0),
            (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF),
            (0x8000000000000000, 0x8000000000000000),
            (0x7FFFFFFFFFFFFFFF, 1),
            (0x123456789ABCDEF0, 0xFEDCBA9876543210),
        ]

        random.seed(42)
        for _ in range(20):
            test_cases.append(
                (
                    random.randint(0, 0xFFFFFFFFFFFFFFFF),
                    random.randint(0, 0xFFFFFFFFFFFFFFFF),
                )
            )

        for a_val, b_val in test_cases:
            in_bits = _int_to_bits(a_val, 64) + _int_to_bits(b_val, 64)
            out_bits = _eval_circuit_state_once(circuit, in_bits)
            o_val = _bits_to_int(out_bits[:64])
            expected = (a_val + b_val) & 0xFFFFFFFFFFFFFFFF
            self.assertEqual(o_val, expected, f"add({a_val}, {b_val})")

    def test_sub_64bit_random(self):
        a = Var("a")
        b = Var("b")
        o_expr = Sub(a=a, b=b)

        ir = TickIR(
            name="sub64",
            inputs={"a": BitVecType(64), "b": BitVecType(64)},
            outputs={"o": BitVecType(64)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": o_expr},
        )

        circuit, layout = lower_tick_ir_to_circuit_state(ir)
        self.assertEqual(layout.input_bits, 128)
        self.assertEqual(layout.output_bits, 64)

        test_cases = [
            (0, 0),
            (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF),
            (0x8000000000000000, 0x8000000000000000),
            (0x7FFFFFFFFFFFFFFF, 1),
            (0x123456789ABCDEF0, 0xFEDCBA9876543210),
            (0, 1),
            (100, 200),
        ]

        random.seed(42)
        for _ in range(20):
            test_cases.append(
                (
                    random.randint(0, 0xFFFFFFFFFFFFFFFF),
                    random.randint(0, 0xFFFFFFFFFFFFFFFF),
                )
            )

        for a_val, b_val in test_cases:
            in_bits = _int_to_bits(a_val, 64) + _int_to_bits(b_val, 64)
            out_bits = _eval_circuit_state_once(circuit, in_bits)
            o_val = _bits_to_int(out_bits[:64])
            expected = (a_val - b_val) & 0xFFFFFFFFFFFFFFFF
            self.assertEqual(o_val, expected, f"sub({a_val}, {b_val})")

    def test_add_carry_propagation(self):
        a = Var("a")
        b = Var("b")
        o_expr = Add(a=a, b=b)

        ir = TickIR(
            name="add_carry",
            inputs={"a": BitVecType(8), "b": BitVecType(8)},
            outputs={"o": BitVecType(8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": o_expr},
        )

        circuit, layout = lower_tick_ir_to_circuit_state(ir)

        test_cases = [
            (0xFF, 0x01),
            (0x7F, 0x01),
            (0x0F, 0x01),
            (0x01, 0x01),
            (0x55, 0xAA),
            (0xAA, 0x55),
        ]

        for a_val, b_val in test_cases:
            in_bits = _int_to_bits(a_val, 8) + _int_to_bits(b_val, 8)
            out_bits = _eval_circuit_state_once(circuit, in_bits)
            o_val = _bits_to_int(out_bits[:8])
            expected = (a_val + b_val) & 0xFF
            self.assertEqual(o_val, expected, f"add({a_val:#x}, {b_val:#x})")

    def test_sub_borrow_propagation(self):
        a = Var("a")
        b = Var("b")
        o_expr = Sub(a=a, b=b)

        ir = TickIR(
            name="sub_borrow",
            inputs={"a": BitVecType(8), "b": BitVecType(8)},
            outputs={"o": BitVecType(8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": o_expr},
        )

        circuit, layout = lower_tick_ir_to_circuit_state(ir)

        test_cases = [
            (0x00, 0x01),
            (0x80, 0x81),
            (0x10, 0x11),
            (0x01, 0x02),
            (0x55, 0xAA),
            (0xAA, 0x55),
        ]

        for a_val, b_val in test_cases:
            in_bits = _int_to_bits(a_val, 8) + _int_to_bits(b_val, 8)
            out_bits = _eval_circuit_state_once(circuit, in_bits)
            o_val = _bits_to_int(out_bits[:8])
            expected = (a_val - b_val) & 0xFF
            self.assertEqual(o_val, expected, f"sub({a_val:#x}, {b_val:#x})")

    def test_add_with_state(self):
        a = Var("a")
        s = Var("s")
        o_expr = Add(a=a, b=s)
        nx_expr = Add(a=s, b=BitVecConst(width=8, value=1))

        ir = TickIR(
            name="add_state",
            inputs={"a": BitVecType(8)},
            outputs={"o": BitVecType(8)},
            state={"s": BitVecType(8)},
            reset_state={"s": BitVecConst(width=8, value=0)},
            next_state={"s": nx_expr},
            output_exprs={"o": o_expr},
        )

        circuit, layout = lower_tick_ir_to_circuit_state(ir)
        self.assertEqual(layout.input_bits, 16)
        self.assertEqual(layout.output_bits, 16)

        test_cases = [(5, 10), (100, 200), (255, 0), (0, 255)]

        for a_val, s_val in test_cases:
            in_bits = _int_to_bits(a_val, 8) + _int_to_bits(s_val, 8)
            out_bits = _eval_circuit_state_once(circuit, in_bits)
            o_val = _bits_to_int(out_bits[:8])
            nx_val = _bits_to_int(out_bits[8:16])

            expected_o = (a_val + s_val) & 0xFF
            expected_nx = (s_val + 1) & 0xFF

            self.assertEqual(o_val, expected_o)
            self.assertEqual(nx_val, expected_nx)

    def test_sub_with_state(self):
        a = Var("a")
        s = Var("s")
        o_expr = Sub(a=a, b=s)
        nx_expr = Sub(a=s, b=BitVecConst(width=8, value=1))

        ir = TickIR(
            name="sub_state",
            inputs={"a": BitVecType(8)},
            outputs={"o": BitVecType(8)},
            state={"s": BitVecType(8)},
            reset_state={"s": BitVecConst(width=8, value=0)},
            next_state={"s": nx_expr},
            output_exprs={"o": o_expr},
        )

        circuit, layout = lower_tick_ir_to_circuit_state(ir)
        self.assertEqual(layout.input_bits, 16)
        self.assertEqual(layout.output_bits, 16)

        test_cases = [(5, 10), (100, 200), (255, 0), (0, 255)]

        for a_val, s_val in test_cases:
            in_bits = _int_to_bits(a_val, 8) + _int_to_bits(s_val, 8)
            out_bits = _eval_circuit_state_once(circuit, in_bits)
            o_val = _bits_to_int(out_bits[:8])
            nx_val = _bits_to_int(out_bits[8:16])

            expected_o = (a_val - s_val) & 0xFF
            expected_nx = (s_val - 1) & 0xFF

            self.assertEqual(o_val, expected_o)
            self.assertEqual(nx_val, expected_nx)
