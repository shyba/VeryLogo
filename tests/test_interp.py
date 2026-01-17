import unittest

from stc.interp import reset_state, tick
from stc.tick_ir import (
    Add,
    And,
    BitVecConst,
    BitVecType,
    BoolConst,
    BoolType,
    TickIR,
    Var,
)


class TestInterp(unittest.TestCase):
    def test_combinational_passthrough(self) -> None:
        ir = TickIR(
            name="top",
            inputs={"i": BoolType()},
            outputs={"o": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Var(name="i")},
        )

        state = reset_state(ir)
        state2, outputs = tick(ir, state, {"i": True})
        self.assertEqual(state, state2)
        self.assertEqual(outputs, {"o": True})

    def test_state_update(self) -> None:
        ir = TickIR(
            name="top",
            inputs={"i": BoolType()},
            outputs={"o": BoolType()},
            state={"s": BoolType()},
            reset_state={"s": BoolConst(value=False)},
            next_state={"s": Var(name="i")},
            output_exprs={"o": Var(name="s")},
        )

        state0 = reset_state(ir)
        state1, out0 = tick(ir, state0, {"i": True})
        self.assertEqual(out0["o"], False)
        state2, out1 = tick(ir, state1, {"i": True})
        self.assertEqual(out1["o"], True)

    def test_bitvec_add_wrap(self) -> None:
        ir = TickIR(
            name="top",
            inputs={"a": BitVecType(width=4), "b": BitVecType(width=4)},
            outputs={"y": BitVecType(width=4)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": Add(a=Var(name="a"), b=Var(name="b"))},
        )

        state = reset_state(ir)
        _, outputs = tick(ir, state, {"a": 15, "b": 1})
        self.assertEqual(outputs["y"], 0)

    def test_bitvec_and(self) -> None:
        ir = TickIR(
            name="top",
            inputs={"a": BitVecType(width=4)},
            outputs={"y": BitVecType(width=4)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": And(a=Var(name="a"), b=BitVecConst(width=4, value=3))},
        )

        state = reset_state(ir)
        _, outputs = tick(ir, state, {"a": 10})
        self.assertEqual(outputs["y"], 2)
