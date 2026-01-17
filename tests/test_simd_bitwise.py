import unittest

from stc.interp import eval_expr
from stc.tick_ir import (
    SimdAnd,
    SimdNot,
    SimdOr,
    SimdType,
    SimdXor,
    Var,
)


class TestSimdBitwise(unittest.TestCase):
    def test_simd_not(self) -> None:
        t = SimdType(lane_width=4, lanes=2)
        types = {"x": t}
        self.assertEqual(eval_expr(SimdNot(x=Var("x")), types, {"x": 0x00}), 0xFF)

    def test_simd_and_or_xor(self) -> None:
        t = SimdType(lane_width=4, lanes=2)
        types = {"x": t, "y": t}
        self.assertEqual(
            eval_expr(SimdAnd(a=Var("x"), b=Var("y")), types, {"x": 0xF0, "y": 0x0F}),
            0x00,
        )
        self.assertEqual(
            eval_expr(SimdOr(a=Var("x"), b=Var("y")), types, {"x": 0xF0, "y": 0x0F}),
            0xFF,
        )
        self.assertEqual(
            eval_expr(SimdXor(a=Var("x"), b=Var("y")), types, {"x": 0xAA, "y": 0x0F}),
            0xA5,
        )
