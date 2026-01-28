import unittest

from stc.reduce import optimize_tick_ir
from stc.tick_ir import BoolConst, BoolType, TickIR, Var


class TestConstantState(unittest.TestCase):
    def test_constant_reg_is_eliminated(self) -> None:
        ir = TickIR(
            name="top",
            inputs={},
            outputs={"o": BoolType()},
            state={"s": BoolType()},
            reset_state={"s": BoolConst(value=False)},
            next_state={"s": BoolConst(value=False)},
            output_exprs={"o": Var(name="s")},
        )

        opt, _ = optimize_tick_ir(ir, bound=2)
        self.assertEqual(opt.state, {})
        self.assertEqual(opt.output_exprs["o"], BoolConst(value=False))
