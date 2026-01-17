import unittest

from stc.superopt import superopt_expr
from stc.tick_ir import Add, Bitcast, BitVecType, Concat, SimdAdd, SimdType, Slice, Var


class TestSuperoptSimdCanonical(unittest.TestCase):
    def test_superopt_finds_simd_add_from_lane_add(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        bv8 = BitVecType(width=8)
        types = {"x": simd, "y": simd}

        xb = Bitcast(to=bv8, x=Var("x"))
        yb = Bitcast(to=bv8, x=Var("y"))
        lo = Add(a=Slice(x=xb, offset=0, width=4), b=Slice(x=yb, offset=0, width=4))
        hi = Add(a=Slice(x=xb, offset=4, width=4), b=Slice(x=yb, offset=4, width=4))
        spec = Bitcast(to=simd, x=Concat(parts=[hi, lo]))

        out = superopt_expr(spec, types, max_nodes=3, timeout_ms=200)
        self.assertEqual(out, SimdAdd(a=Var("x"), b=Var("y")))
