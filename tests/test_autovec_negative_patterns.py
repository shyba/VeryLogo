import unittest

from stc.autovec import autovectorize_expr
from stc.tick_ir import (
    Add,
    Bitcast,
    BitVecType,
    Concat,
    SimdType,
    Slice,
    Var,
)


class TestAutovecNegativePatterns(unittest.TestCase):
    def test_autovec_does_not_match_misaligned_slices(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        bv8 = BitVecType(width=8)
        types = {"x": simd, "y": simd}

        xb = Bitcast(to=bv8, x=Var(name="x"))
        yb = Bitcast(to=bv8, x=Var(name="y"))
        lo = Add(a=Slice(x=xb, offset=1, width=4), b=Slice(x=yb, offset=1, width=4))
        hi = Add(a=Slice(x=xb, offset=3, width=4), b=Slice(x=yb, offset=3, width=4))
        spec = Bitcast(to=simd, x=Concat(parts=[hi, lo]))

        self.assertEqual(autovectorize_expr(spec, types), spec)

    def test_autovec_does_not_match_mixed_operands(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        bv8 = BitVecType(width=8)
        types = {"x": simd, "y": simd, "z": simd}

        xb = Bitcast(to=bv8, x=Var(name="x"))
        yb = Bitcast(to=bv8, x=Var(name="y"))
        zb = Bitcast(to=bv8, x=Var(name="z"))
        lo = Add(a=Slice(x=xb, offset=0, width=4), b=Slice(x=yb, offset=0, width=4))
        hi = Add(a=Slice(x=xb, offset=4, width=4), b=Slice(x=zb, offset=4, width=4))
        spec = Bitcast(to=simd, x=Concat(parts=[hi, lo]))

        self.assertEqual(autovectorize_expr(spec, types), spec)

    def test_autovec_does_not_match_cross_lane_carry(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        bv8 = BitVecType(width=8)
        types = {"x": simd, "y": simd}

        xb = Bitcast(to=bv8, x=Var(name="x"))
        yb = Bitcast(to=bv8, x=Var(name="y"))
        packed = Add(a=xb, b=yb)
        lo = Slice(x=packed, offset=0, width=4)
        hi = Slice(x=packed, offset=4, width=4)
        spec = Bitcast(to=simd, x=Concat(parts=[hi, lo]))

        self.assertEqual(autovectorize_expr(spec, types), spec)

    def test_autovec_does_not_match_partial_lanes(self) -> None:
        simd3 = SimdType(lane_width=4, lanes=3)
        bv12 = BitVecType(width=12)
        types = {"x": simd3, "y": simd3}

        xb = Bitcast(to=bv12, x=Var(name="x"))
        yb = Bitcast(to=bv12, x=Var(name="y"))
        lane0 = Slice(x=xb, offset=0, width=4)
        lane1 = Add(a=Slice(x=xb, offset=4, width=4), b=Slice(x=yb, offset=4, width=4))
        lane2 = Add(a=Slice(x=xb, offset=8, width=4), b=Slice(x=yb, offset=8, width=4))
        spec = Bitcast(to=simd3, x=Concat(parts=[lane2, lane1, lane0]))

        self.assertEqual(autovectorize_expr(spec, types), spec)
