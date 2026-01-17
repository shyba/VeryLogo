import unittest

from stc.hashcons import hashcons_tick_ir
from stc.tick_ir import Add, BitVecType, TickIR, Var
from stc.tick_ir_validate import validate_tick_ir


class TestHashcons(unittest.TestCase):
    def test_hashcons_deduplicates_identical_subexpressions(self) -> None:
        bv8 = BitVecType(width=8)
        e1 = Add(a=Var("x"), b=Var("y"))
        e2 = Add(a=Var("x"), b=Var("y"))
        self.assertIsNot(e1, e2)

        ir = TickIR(
            name="t",
            inputs={"x": bv8, "y": bv8},
            outputs={"o1": bv8, "o2": bv8},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o1": e1, "o2": e2},
        )
        validate_tick_ir(ir)
        out = hashcons_tick_ir(ir)
        self.assertIs(out.output_exprs["o1"], out.output_exprs["o2"])
