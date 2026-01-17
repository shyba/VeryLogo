import unittest

from stc.interp import eval_expr
from stc.tick_ir import SimdEq, SimdType, Var


class TestSimdMasks(unittest.TestCase):
    def test_simd_eq_mask_layout(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        types = {"x": simd, "y": simd}

        expr = SimdEq(a=Var(name="x"), b=Var(name="y"))
        self.assertEqual(eval_expr(expr, types, {"x": 0xAB, "y": 0xAB}), 0b11)
        self.assertEqual(eval_expr(expr, types, {"x": 0xAB, "y": 0xAC}), 0b10)
        self.assertEqual(eval_expr(expr, types, {"x": 0xAB, "y": 0xBB}), 0b01)
