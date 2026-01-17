import unittest

from stc.interp import eval_expr
from stc.tick_ir import (
    BitVecConst,
    SimdAdd,
    SimdConst,
    SimdEq,
    SimdSub,
    SimdType,
    SimdUlt,
    Var,
)


class TestSimdMatrix(unittest.TestCase):
    def test_core_identities_matrix(self) -> None:
        for lane_width in [1, 2, 4, 8]:
            for lanes in [2, 4]:
                t = SimdType(lane_width=lane_width, lanes=lanes)
                types = {"x": t}
                total = lane_width * lanes
                x_val = (1 << total) - 1
                zero = SimdConst(lane_width=lane_width, lanes=lanes, value=0)
                ones_mask = (1 << lanes) - 1

                self.assertEqual(
                    eval_expr(SimdAdd(a=Var("x"), b=zero), types, {"x": x_val}),
                    x_val,
                )
                self.assertEqual(
                    eval_expr(SimdSub(a=Var("x"), b=zero), types, {"x": x_val}),
                    x_val,
                )
                self.assertEqual(
                    eval_expr(SimdEq(a=Var("x"), b=Var("x")), types, {"x": x_val}),
                    ones_mask,
                )
                self.assertEqual(
                    eval_expr(SimdUlt(a=Var("x"), b=Var("x")), types, {"x": x_val}),
                    0,
                )
