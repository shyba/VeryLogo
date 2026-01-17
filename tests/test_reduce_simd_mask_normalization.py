import unittest

from stc.reduce import reduce_expr
from stc.tick_ir import SimdMaskExpand, SimdMaskPack, SimdType, Var


class TestReduceSimdMaskNormalization(unittest.TestCase):
    def test_pack_of_expand_reduces_to_mask(self) -> None:
        m_t = SimdType(lane_width=1, lanes=16)
        v_t = SimdType(lane_width=8, lanes=16)
        types = {"m": m_t}
        expr = SimdMaskPack(x=SimdMaskExpand(to=v_t, x=Var("m")))
        self.assertEqual(reduce_expr(expr, types), Var("m"))

    def test_expand_pack_expand_simplifies(self) -> None:
        m_t = SimdType(lane_width=1, lanes=16)
        v_t = SimdType(lane_width=8, lanes=16)
        types = {"m": m_t}
        expr = SimdMaskExpand(
            to=v_t, x=SimdMaskPack(x=SimdMaskExpand(to=v_t, x=Var("m")))
        )
        self.assertEqual(reduce_expr(expr, types), SimdMaskExpand(to=v_t, x=Var("m")))
