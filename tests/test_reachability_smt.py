import unittest

from stc.reduce import optimize_tick_ir
from stc.tick_ir import BitVecConst, BitVecType, TickIR, Var
from stc.tick_ir_validate import validate_tick_ir


class TestReachabilitySmt(unittest.TestCase):
    def test_constant_state_detection_scales_past_enumeration_limit(self) -> None:
        bv16 = BitVecType(width=16)
        ir = TickIR(
            name="t",
            inputs={},
            outputs={"o": bv16},
            state={"s": bv16},
            reset_state={"s": BitVecConst(width=16, value=0x1234)},
            next_state={"s": Var("s")},
            output_exprs={"o": Var("s")},
        )
        validate_tick_ir(ir)

        out, _ = optimize_tick_ir(ir, bound=4)
        self.assertEqual(out.state, {})
        self.assertEqual(out.output_exprs["o"], BitVecConst(width=16, value=0x1234))
