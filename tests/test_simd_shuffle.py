import unittest

from stc.interp import eval_expr
from stc.tick_ir import SimdConst, SimdShuffle, SimdType


class TestSimdShuffle(unittest.TestCase):
    def test_shuffle(self) -> None:
        t = SimdType(lane_width=4, lanes=4)
        v = SimdConst(lane_width=4, lanes=4, value=0x3210)
        expr = SimdShuffle(x=v, indices=[1, 0, 3, 2])
        self.assertEqual(eval_expr(expr, {}, {}), 0x2301)
