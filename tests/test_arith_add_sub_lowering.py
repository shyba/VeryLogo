import random
import unittest

from stc.tick_ir import (
    Add,
    BitVecType,
    BitVecConst,
    Sub,
    Var,
    TickIR,
)
from stc.tick_ir_to_packed_circuit_state import (
    lower_tick_ir_to_packed_circuit_state,
)
from stc.packed_circuit import eval_packed_circuit_words


class TestArithAddSubLowering(unittest.TestCase):
    def test_add_64bit_no_carry(self) -> None:
        expr = Add(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="add64",
            inputs={"x": BitVecType(64), "y": BitVecType(64)},
            outputs={"o": BitVecType(64)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 2)
        self.assertEqual(layout.output_words, 1)

        for _ in range(50):
            xv = random.getrandbits(63)
            yv = random.getrandbits(63)
            inputs = [xv, yv]
            outs = eval_packed_circuit_words(circuit, inputs)
            expected = (xv + yv) & ((1 << 64) - 1)
            self.assertEqual(outs[0], expected)

    def test_add_64bit_with_overflow(self) -> None:
        expr = Add(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="add64_overflow",
            inputs={"x": BitVecType(64), "y": BitVecType(64)},
            outputs={"o": BitVecType(64)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)

        for _ in range(50):
            xv = random.getrandbits(64)
            yv = random.getrandbits(64)
            inputs = [xv, yv]
            outs = eval_packed_circuit_words(circuit, inputs)
            expected = (xv + yv) & ((1 << 64) - 1)
            self.assertEqual(outs[0], expected)

    def test_add_128bit_no_carry(self) -> None:
        expr = Add(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="add128",
            inputs={"x": BitVecType(128), "y": BitVecType(128)},
            outputs={"o": BitVecType(128)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 4)
        self.assertEqual(layout.output_words, 2)

        for _ in range(50):
            xv = random.getrandbits(127)
            yv = random.getrandbits(127)
            x_inputs = [xv & ((1 << 64) - 1), (xv >> 64) & ((1 << 64) - 1)]
            y_inputs = [yv & ((1 << 64) - 1), (yv >> 64) & ((1 << 64) - 1)]
            inputs = x_inputs + y_inputs
            outs = eval_packed_circuit_words(circuit, inputs)
            expected = (xv + yv) & ((1 << 128) - 1)
            self.assertEqual(outs[0], expected & ((1 << 64) - 1))
            self.assertEqual(outs[1], (expected >> 64) & ((1 << 64) - 1))

    def test_add_128bit_with_carry_propagation(self) -> None:
        expr = Add(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="add128_carry",
            inputs={"x": BitVecType(128), "y": BitVecType(128)},
            outputs={"o": BitVecType(128)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)

        for _ in range(50):
            xv = random.getrandbits(128)
            yv = random.getrandbits(128)
            x_inputs = [xv & ((1 << 64) - 1), (xv >> 64) & ((1 << 64) - 1)]
            y_inputs = [yv & ((1 << 64) - 1), (yv >> 64) & ((1 << 64) - 1)]
            inputs = x_inputs + y_inputs
            outs = eval_packed_circuit_words(circuit, inputs)
            expected = (xv + yv) & ((1 << 128) - 1)
            reconstructed = outs[0] | (outs[1] << 64)
            self.assertEqual(reconstructed, expected)

    def test_add_256bit_with_carry_propagation(self) -> None:
        expr = Add(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="add256",
            inputs={"x": BitVecType(256), "y": BitVecType(256)},
            outputs={"o": BitVecType(256)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 8)
        self.assertEqual(layout.output_words, 4)

        for _ in range(50):
            xv = random.getrandbits(256)
            yv = random.getrandbits(256)
            x_inputs = [(xv >> (i * 64)) & ((1 << 64) - 1) for i in range(4)]
            y_inputs = [(yv >> (i * 64)) & ((1 << 64) - 1) for i in range(4)]
            inputs = x_inputs + y_inputs
            outs = eval_packed_circuit_words(circuit, inputs)
            expected = (xv + yv) & ((1 << 256) - 1)
            reconstructed = sum(w << (i * 64) for i, w in enumerate(outs))
            self.assertEqual(reconstructed, expected)

    def test_add_non_multiple_of_64_width_100(self) -> None:
        expr = Add(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="add100",
            inputs={"x": BitVecType(100), "y": BitVecType(100)},
            outputs={"o": BitVecType(100)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 4)
        self.assertEqual(layout.output_words, 2)

        for _ in range(50):
            xv = random.getrandbits(100)
            yv = random.getrandbits(100)
            x_inputs = [xv & ((1 << 64) - 1), (xv >> 64) & ((1 << 36) - 1)]
            y_inputs = [yv & ((1 << 64) - 1), (yv >> 64) & ((1 << 36) - 1)]
            inputs = x_inputs + y_inputs
            outs = eval_packed_circuit_words(circuit, inputs)
            expected = (xv + yv) & ((1 << 100) - 1)
            reconstructed = outs[0] | (outs[1] << 64)
            reconstructed &= (1 << 100) - 1
            self.assertEqual(reconstructed, expected)

    def test_add_edge_case_all_zeros(self) -> None:
        expr = Add(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="add_zeros",
            inputs={"x": BitVecType(128), "y": BitVecType(128)},
            outputs={"o": BitVecType(128)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)

        xv = 0
        yv = 0
        x_inputs = [0, 0]
        y_inputs = [0, 0]
        inputs = x_inputs + y_inputs
        outs = eval_packed_circuit_words(circuit, inputs)
        self.assertEqual(outs[0], 0)
        self.assertEqual(outs[1], 0)

    def test_add_edge_case_all_ones(self) -> None:
        expr = Add(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="add_ones",
            inputs={"x": BitVecType(128), "y": BitVecType(128)},
            outputs={"o": BitVecType(128)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)

        xv = (1 << 128) - 1
        yv = (1 << 128) - 1
        x_inputs = [(1 << 64) - 1, (1 << 64) - 1]
        y_inputs = [(1 << 64) - 1, (1 << 64) - 1]
        inputs = x_inputs + y_inputs
        outs = eval_packed_circuit_words(circuit, inputs)
        expected = (xv + yv) & ((1 << 128) - 1)
        reconstructed = outs[0] | (outs[1] << 64)
        self.assertEqual(reconstructed, expected)

    def test_add_alternating_pattern(self) -> None:
        expr = Add(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="add_alt",
            inputs={"x": BitVecType(128), "y": BitVecType(128)},
            outputs={"o": BitVecType(128)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)

        xv = 0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        yv = 0x55555555555555555555555555555555
        x_inputs = [xv & ((1 << 64) - 1), (xv >> 64) & ((1 << 64) - 1)]
        y_inputs = [yv & ((1 << 64) - 1), (yv >> 64) & ((1 << 64) - 1)]
        inputs = x_inputs + y_inputs
        outs = eval_packed_circuit_words(circuit, inputs)
        expected = (xv + yv) & ((1 << 128) - 1)
        reconstructed = outs[0] | (outs[1] << 64)
        self.assertEqual(reconstructed, expected)

    def test_sub_64bit_no_borrow(self) -> None:
        expr = Sub(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="sub64",
            inputs={"x": BitVecType(64), "y": BitVecType(64)},
            outputs={"o": BitVecType(64)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 2)
        self.assertEqual(layout.output_words, 1)

        for _ in range(50):
            yv = random.getrandbits(63)
            xv = yv + random.getrandbits(63)
            inputs = [xv, yv]
            outs = eval_packed_circuit_words(circuit, inputs)
            expected = (xv - yv) & ((1 << 64) - 1)
            self.assertEqual(outs[0], expected)

    def test_sub_64bit_with_underflow(self) -> None:
        expr = Sub(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="sub64_underflow",
            inputs={"x": BitVecType(64), "y": BitVecType(64)},
            outputs={"o": BitVecType(64)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)

        for _ in range(50):
            xv = random.getrandbits(64)
            yv = random.getrandbits(64)
            inputs = [xv, yv]
            outs = eval_packed_circuit_words(circuit, inputs)
            expected = (xv - yv) & ((1 << 64) - 1)
            self.assertEqual(outs[0], expected)

    def test_sub_128bit_no_borrow(self) -> None:
        expr = Sub(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="sub128",
            inputs={"x": BitVecType(128), "y": BitVecType(128)},
            outputs={"o": BitVecType(128)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 4)
        self.assertEqual(layout.output_words, 2)

        for _ in range(50):
            yv = random.getrandbits(127)
            xv = yv + random.getrandbits(127)
            x_inputs = [xv & ((1 << 64) - 1), (xv >> 64) & ((1 << 64) - 1)]
            y_inputs = [yv & ((1 << 64) - 1), (yv >> 64) & ((1 << 64) - 1)]
            inputs = x_inputs + y_inputs
            outs = eval_packed_circuit_words(circuit, inputs)
            expected = (xv - yv) & ((1 << 128) - 1)
            reconstructed = outs[0] | (outs[1] << 64)
            self.assertEqual(reconstructed, expected)

    def test_sub_128bit_with_borrow_propagation(self) -> None:
        expr = Sub(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="sub128_borrow",
            inputs={"x": BitVecType(128), "y": BitVecType(128)},
            outputs={"o": BitVecType(128)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)

        for _ in range(50):
            xv = random.getrandbits(128)
            yv = random.getrandbits(128)
            x_inputs = [xv & ((1 << 64) - 1), (xv >> 64) & ((1 << 64) - 1)]
            y_inputs = [yv & ((1 << 64) - 1), (yv >> 64) & ((1 << 64) - 1)]
            inputs = x_inputs + y_inputs
            outs = eval_packed_circuit_words(circuit, inputs)
            expected = (xv - yv) & ((1 << 128) - 1)
            reconstructed = outs[0] | (outs[1] << 64)
            self.assertEqual(reconstructed, expected)

    def test_sub_256bit_with_borrow_propagation(self) -> None:
        expr = Sub(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="sub256",
            inputs={"x": BitVecType(256), "y": BitVecType(256)},
            outputs={"o": BitVecType(256)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 8)
        self.assertEqual(layout.output_words, 4)

        for _ in range(50):
            xv = random.getrandbits(256)
            yv = random.getrandbits(256)
            x_inputs = [(xv >> (i * 64)) & ((1 << 64) - 1) for i in range(4)]
            y_inputs = [(yv >> (i * 64)) & ((1 << 64) - 1) for i in range(4)]
            inputs = x_inputs + y_inputs
            outs = eval_packed_circuit_words(circuit, inputs)
            expected = (xv - yv) & ((1 << 256) - 1)
            reconstructed = sum(w << (i * 64) for i, w in enumerate(outs))
            self.assertEqual(reconstructed, expected)

    def test_sub_non_multiple_of_64_width_100(self) -> None:
        expr = Sub(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="sub100",
            inputs={"x": BitVecType(100), "y": BitVecType(100)},
            outputs={"o": BitVecType(100)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 4)
        self.assertEqual(layout.output_words, 2)

        for _ in range(50):
            xv = random.getrandbits(100)
            yv = random.getrandbits(100)
            x_inputs = [xv & ((1 << 64) - 1), (xv >> 64) & ((1 << 36) - 1)]
            y_inputs = [yv & ((1 << 64) - 1), (yv >> 64) & ((1 << 36) - 1)]
            inputs = x_inputs + y_inputs
            outs = eval_packed_circuit_words(circuit, inputs)
            expected = (xv - yv) & ((1 << 100) - 1)
            reconstructed = outs[0] | (outs[1] << 64)
            reconstructed &= (1 << 100) - 1
            self.assertEqual(reconstructed, expected)

    def test_sub_edge_case_all_zeros(self) -> None:
        expr = Sub(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="sub_zeros",
            inputs={"x": BitVecType(128), "y": BitVecType(128)},
            outputs={"o": BitVecType(128)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)

        xv = 0
        yv = 0
        x_inputs = [0, 0]
        y_inputs = [0, 0]
        inputs = x_inputs + y_inputs
        outs = eval_packed_circuit_words(circuit, inputs)
        self.assertEqual(outs[0], 0)
        self.assertEqual(outs[1], 0)

    def test_sub_edge_case_all_ones(self) -> None:
        expr = Sub(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="sub_ones",
            inputs={"x": BitVecType(128), "y": BitVecType(128)},
            outputs={"o": BitVecType(128)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)

        xv = (1 << 128) - 1
        yv = (1 << 128) - 1
        x_inputs = [(1 << 64) - 1, (1 << 64) - 1]
        y_inputs = [(1 << 64) - 1, (1 << 64) - 1]
        inputs = x_inputs + y_inputs
        outs = eval_packed_circuit_words(circuit, inputs)
        expected = (xv - yv) & ((1 << 128) - 1)
        reconstructed = outs[0] | (outs[1] << 64)
        self.assertEqual(reconstructed, expected)

    def test_sub_alternating_pattern(self) -> None:
        expr = Sub(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="sub_alt",
            inputs={"x": BitVecType(128), "y": BitVecType(128)},
            outputs={"o": BitVecType(128)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)

        xv = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        yv = 0x55555555555555555555555555555555
        x_inputs = [xv & ((1 << 64) - 1), (xv >> 64) & ((1 << 64) - 1)]
        y_inputs = [yv & ((1 << 64) - 1), (yv >> 64) & ((1 << 64) - 1)]
        inputs = x_inputs + y_inputs
        outs = eval_packed_circuit_words(circuit, inputs)
        expected = (xv - yv) & ((1 << 128) - 1)
        reconstructed = outs[0] | (outs[1] << 64)
        self.assertEqual(reconstructed, expected)


if __name__ == "__main__":
    unittest.main()
