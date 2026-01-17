import unittest

from stc.autovec import autovectorize_expr
from stc.tick_ir import (
    Bitcast,
    BitVecType,
    Concat,
    SimdType,
    SimdUge,
    SimdUlt,
    Slice,
    Uge,
    Ult,
    Var,
)


class TestAutovecCmpUnsigned(unittest.TestCase):
    def test_autovec_simd_ult_mask(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        mask = SimdType(lane_width=1, lanes=2)
        bv8 = BitVecType(width=8)
        types = {"x": simd, "y": simd}

        xb = Bitcast(to=bv8, x=Var(name="x"))
        yb = Bitcast(to=bv8, x=Var(name="y"))
        lo = Ult(a=Slice(x=xb, offset=0, width=4), b=Slice(x=yb, offset=0, width=4))
        hi = Ult(a=Slice(x=xb, offset=4, width=4), b=Slice(x=yb, offset=4, width=4))
        spec = Bitcast(to=mask, x=Concat(parts=[hi, lo]))

        self.assertEqual(
            autovectorize_expr(spec, types), SimdUlt(a=Var("x"), b=Var("y"))
        )

    def test_autovec_simd_uge_mask(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        mask = SimdType(lane_width=1, lanes=2)
        bv8 = BitVecType(width=8)
        types = {"x": simd, "y": simd}

        xb = Bitcast(to=bv8, x=Var(name="x"))
        yb = Bitcast(to=bv8, x=Var(name="y"))
        lo = Uge(a=Slice(x=xb, offset=0, width=4), b=Slice(x=yb, offset=0, width=4))
        hi = Uge(a=Slice(x=xb, offset=4, width=4), b=Slice(x=yb, offset=4, width=4))
        spec = Bitcast(to=mask, x=Concat(parts=[hi, lo]))

        self.assertEqual(
            autovectorize_expr(spec, types), SimdUge(a=Var("x"), b=Var("y"))
        )
