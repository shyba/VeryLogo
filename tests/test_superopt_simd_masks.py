import unittest

from stc.superopt import superopt_expr
from stc.tick_ir import (
    Bitcast,
    BitVecType,
    Concat,
    Eq,
    SimdEq,
    SimdType,
    Slice,
    Var,
)


class TestSuperoptSimdMasks(unittest.TestCase):
    def test_superopt_finds_simd_eq_mask_from_lane_eq(self) -> None:
        v = SimdType(lane_width=16, lanes=8)
        m = SimdType(lane_width=1, lanes=8)
        bv128 = BitVecType(width=128)
        types = {"a": v, "b": v}

        ab = Bitcast(to=bv128, x=Var("a"))
        bb = Bitcast(to=bv128, x=Var("b"))
        parts = []
        for lane in range(7, -1, -1):
            off = lane * 16
            al = Slice(x=ab, offset=off, width=16)
            bl = Slice(x=bb, offset=off, width=16)
            parts.append(Eq(a=al, b=bl))
        spec = Bitcast(to=m, x=Concat(parts=parts))

        out = superopt_expr(spec, types, max_nodes=3, timeout_ms=200)
        self.assertEqual(out, SimdEq(a=Var("a"), b=Var("b")))
