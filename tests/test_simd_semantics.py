import unittest

from stc.interp import eval_expr
from stc.tick_ir import Bitcast, BitVecType, SimdAdd, SimdConst, SimdType, Slice, Var


class TestSimdSemantics(unittest.TestCase):
    def test_lane_packing_lsb_lane0(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        bv8 = BitVecType(width=8)
        types = {"x": simd}

        xb = Bitcast(to=bv8, x=Var(name="x"))
        lo = Slice(x=xb, offset=0, width=4)
        hi = Slice(x=xb, offset=4, width=4)

        self.assertEqual(eval_expr(lo, types, {"x": 0xAB}), 0xB)
        self.assertEqual(eval_expr(hi, types, {"x": 0xAB}), 0xA)

    def test_lane_wise_add_wraps(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        types = {"x": simd}
        spec = SimdAdd(a=Var(name="x"), b=SimdConst(lane_width=4, lanes=2, value=0x11))
        self.assertEqual(eval_expr(spec, types, {"x": 0xFF}), 0x00)
