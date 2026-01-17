import unittest

from stc.autovec import autovectorize_expr
from stc.tick_ir import (
    Add,
    And,
    Bitcast,
    BitVecType,
    Concat,
    Eq,
    Or,
    SimdAdd,
    SimdEq,
    SimdType,
    Slice,
    Var,
    Xor,
)


class TestAutovec(unittest.TestCase):
    def test_autovec_simd_add(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        bv8 = BitVecType(width=8)
        types = {"x": simd, "y": simd}

        xb = Bitcast(to=bv8, x=Var(name="x"))
        yb = Bitcast(to=bv8, x=Var(name="y"))
        lo = Add(a=Slice(x=xb, offset=0, width=4), b=Slice(x=yb, offset=0, width=4))
        hi = Add(a=Slice(x=xb, offset=4, width=4), b=Slice(x=yb, offset=4, width=4))
        spec = Bitcast(to=simd, x=Concat(parts=[hi, lo]))

        self.assertEqual(
            autovectorize_expr(spec, types), SimdAdd(a=Var("x"), b=Var("y"))
        )

    def test_autovec_simd_xor(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        bv8 = BitVecType(width=8)
        types = {"x": simd, "y": simd}

        xb = Bitcast(to=bv8, x=Var(name="x"))
        yb = Bitcast(to=bv8, x=Var(name="y"))
        lo = Xor(a=Slice(x=xb, offset=0, width=4), b=Slice(x=yb, offset=0, width=4))
        hi = Xor(a=Slice(x=xb, offset=4, width=4), b=Slice(x=yb, offset=4, width=4))
        spec = Bitcast(to=simd, x=Concat(parts=[hi, lo]))

        self.assertEqual(autovectorize_expr(spec, types), Xor(a=Var("x"), b=Var("y")))

    def test_autovec_simd_and(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        bv8 = BitVecType(width=8)
        types = {"x": simd, "y": simd}

        xb = Bitcast(to=bv8, x=Var(name="x"))
        yb = Bitcast(to=bv8, x=Var(name="y"))
        lo = And(a=Slice(x=xb, offset=0, width=4), b=Slice(x=yb, offset=0, width=4))
        hi = And(a=Slice(x=xb, offset=4, width=4), b=Slice(x=yb, offset=4, width=4))
        spec = Bitcast(to=simd, x=Concat(parts=[hi, lo]))

        self.assertEqual(autovectorize_expr(spec, types), And(a=Var("x"), b=Var("y")))

    def test_autovec_simd_or(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        bv8 = BitVecType(width=8)
        types = {"x": simd, "y": simd}

        xb = Bitcast(to=bv8, x=Var(name="x"))
        yb = Bitcast(to=bv8, x=Var(name="y"))
        lo = Or(a=Slice(x=xb, offset=0, width=4), b=Slice(x=yb, offset=0, width=4))
        hi = Or(a=Slice(x=xb, offset=4, width=4), b=Slice(x=yb, offset=4, width=4))
        spec = Bitcast(to=simd, x=Concat(parts=[hi, lo]))

        self.assertEqual(autovectorize_expr(spec, types), Or(a=Var("x"), b=Var("y")))

    def test_autovec_does_not_match_packed_add(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        bv8 = BitVecType(width=8)
        types = {"x": simd, "y": simd}

        xb = Bitcast(to=bv8, x=Var(name="x"))
        yb = Bitcast(to=bv8, x=Var(name="y"))
        spec = Bitcast(to=simd, x=Add(a=xb, b=yb))

        self.assertEqual(autovectorize_expr(spec, types), spec)

    def test_autovec_simd_eq_mask(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        mask = SimdType(lane_width=1, lanes=2)
        bv8 = BitVecType(width=8)
        types = {"x": simd, "y": simd}

        xb = Bitcast(to=bv8, x=Var(name="x"))
        yb = Bitcast(to=bv8, x=Var(name="y"))
        lo = Eq(a=Slice(x=xb, offset=0, width=4), b=Slice(x=yb, offset=0, width=4))
        hi = Eq(a=Slice(x=xb, offset=4, width=4), b=Slice(x=yb, offset=4, width=4))
        spec = Bitcast(to=mask, x=Concat(parts=[hi, lo]))
        self.assertEqual(
            autovectorize_expr(spec, types), SimdEq(a=Var("x"), b=Var("y"))
        )
