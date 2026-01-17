import unittest

from stc.interp import eval_expr
from stc.tick_ir import SimdSge, SimdSlt, SimdType, Var


class TestSimdCmpSigned(unittest.TestCase):
    def test_simd_slt_mask(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        types = {"x": simd, "y": simd}
        expr = SimdSlt(a=Var("x"), b=Var("y"))
        self.assertEqual(eval_expr(expr, types, {"x": 0xF0, "y": 0x10}), 0b10)

    def test_simd_sge_mask(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        types = {"x": simd, "y": simd}
        expr = SimdSge(a=Var("x"), b=Var("y"))
        self.assertEqual(eval_expr(expr, types, {"x": 0xF0, "y": 0x10}), 0b01)
