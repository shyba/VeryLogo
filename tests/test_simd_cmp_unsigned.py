import unittest

from stc.interp import eval_expr
from stc.tick_ir import SimdType, SimdUge, SimdUgt, SimdUle, SimdUlt, Var


class TestSimdCmpUnsigned(unittest.TestCase):
    def test_simd_ult_mask(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        types = {"x": simd, "y": simd}
        expr = SimdUlt(a=Var(name="x"), b=Var(name="y"))
        self.assertEqual(eval_expr(expr, types, {"x": 0x21, "y": 0x12}), 0b01)

    def test_simd_ule_mask(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        types = {"x": simd, "y": simd}
        expr = SimdUle(a=Var(name="x"), b=Var(name="y"))
        self.assertEqual(eval_expr(expr, types, {"x": 0x22, "y": 0x22}), 0b11)

    def test_simd_ugt_mask(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        types = {"x": simd, "y": simd}
        expr = SimdUgt(a=Var(name="x"), b=Var(name="y"))
        self.assertEqual(eval_expr(expr, types, {"x": 0x21, "y": 0x12}), 0b10)

    def test_simd_uge_mask(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        types = {"x": simd, "y": simd}
        expr = SimdUge(a=Var(name="x"), b=Var(name="y"))
        self.assertEqual(eval_expr(expr, types, {"x": 0x21, "y": 0x12}), 0b10)
