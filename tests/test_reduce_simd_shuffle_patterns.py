import unittest

from stc.reduce import reduce_expr
from stc.tick_ir import SimdShuffle, SimdType, SimdUnpackHi, SimdUnpackLo, Var


class TestReduceSimdShufflePatterns(unittest.TestCase):
    def test_shuffle_identity_reduces_to_source(self) -> None:
        t = SimdType(lane_width=8, lanes=16)
        types = {"x": t}
        expr = SimdShuffle(x=Var("x"), indices=list(range(16)))
        self.assertEqual(reduce_expr(expr, types), Var("x"))

    def test_shuffle_duplicate_low_half_reduces_to_unpacklo(self) -> None:
        t = SimdType(lane_width=16, lanes=8)
        types = {"x": t}
        indices = [0, 0, 1, 1, 2, 2, 3, 3]
        expr = SimdShuffle(x=Var("x"), indices=indices)
        self.assertEqual(reduce_expr(expr, types), SimdUnpackLo(a=Var("x"), b=Var("x")))

    def test_shuffle_duplicate_high_half_reduces_to_unpackhi(self) -> None:
        t = SimdType(lane_width=16, lanes=8)
        types = {"x": t}
        indices = [4, 4, 5, 5, 6, 6, 7, 7]
        expr = SimdShuffle(x=Var("x"), indices=indices)
        self.assertEqual(reduce_expr(expr, types), SimdUnpackHi(a=Var("x"), b=Var("x")))
