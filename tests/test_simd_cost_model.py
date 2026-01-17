import unittest

from stc.cost import expr_cost
from stc.tick_ir import SimdShuffle, SimdType, SimdXor, Var, Xor


class TestSimdCostModel(unittest.TestCase):
    def test_cost_prefers_explicit_simd_bitwise(self) -> None:
        t = SimdType(lane_width=4, lanes=2)
        types = {"x": t, "y": t}
        self.assertLess(
            expr_cost(SimdXor(a=Var("x"), b=Var("y")), types),
            expr_cost(Xor(a=Var("x"), b=Var("y")), types),
        )

    def test_cost_penalizes_shuffle(self) -> None:
        t = SimdType(lane_width=4, lanes=4)
        types = {"x": t}
        self.assertGreater(
            expr_cost(SimdShuffle(x=Var("x"), indices=[0, 1, 2, 3]), types),
            expr_cost(Var("x"), types),
        )
