import unittest

from stc.interp import eval_expr
from stc.tick_ir import SimdConst, SimdSub, SimdType, Var


class TestSimdSub(unittest.TestCase):
    def test_lane_wise_sub_wraps(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        types = {"x": simd}
        expr = SimdSub(a=Var(name="x"), b=SimdConst(lane_width=4, lanes=2, value=0x11))
        self.assertEqual(eval_expr(expr, types, {"x": 0x00}), 0xFF)
