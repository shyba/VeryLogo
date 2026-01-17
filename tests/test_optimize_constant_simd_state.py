import unittest

from stc.reduce import optimize_tick_ir
from stc.tick_ir import SimdConst, SimdType, TickIR, Var
from stc.tick_ir_validate import validate_tick_ir


class TestOptimizeConstantSimdState(unittest.TestCase):
    def test_optimize_removes_constant_simd_state(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        ir = TickIR(
            name="t",
            inputs={},
            outputs={"o": simd},
            state={"s": simd},
            reset_state={"s": SimdConst(lane_width=4, lanes=2, value=0xAB)},
            next_state={"s": Var("s")},
            output_exprs={"o": Var("s")},
        )
        validate_tick_ir(ir)
        out = optimize_tick_ir(ir, bound=3)
        self.assertEqual(out.state, {})
        self.assertEqual(
            out.output_exprs["o"], SimdConst(lane_width=4, lanes=2, value=0xAB)
        )
