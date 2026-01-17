import unittest

from stc.autovec import autovectorize_expr
from stc.tick_ir import (
    Bitcast,
    BitVecType,
    Concat,
    Eq,
    Mux,
    SimdBlend,
    SimdEq,
    SimdType,
    Slice,
    Var,
)


class TestAutovecBlend(unittest.TestCase):
    def test_autovec_blend_from_lane_mux_eq(self) -> None:
        simd = SimdType(lane_width=16, lanes=8)
        mask_t = SimdType(lane_width=1, lanes=8)
        bv128 = BitVecType(width=128)
        types = {"a": simd, "b": simd, "x": simd, "y": simd}

        ab = Bitcast(to=bv128, x=Var("a"))
        bb = Bitcast(to=bv128, x=Var("b"))
        xb = Bitcast(to=bv128, x=Var("x"))
        yb = Bitcast(to=bv128, x=Var("y"))

        parts = []
        for lane in range(7, -1, -1):
            off = lane * 16
            al = Slice(x=ab, offset=off, width=16)
            bl = Slice(x=bb, offset=off, width=16)
            xl = Slice(x=xb, offset=off, width=16)
            yl = Slice(x=yb, offset=off, width=16)
            parts.append(Mux(cond=Eq(a=al, b=bl), a=xl, b=yl))
        spec = Bitcast(to=simd, x=Concat(parts=parts))

        out = autovectorize_expr(spec, types, timeout_ms=200)
        self.assertEqual(
            out,
            SimdBlend(
                mask=SimdEq(a=Var("a"), b=Var("b")),
                a=Var("y"),
                b=Var("x"),
            ),
        )
        self.assertEqual(
            autovectorize_expr(
                Bitcast(to=mask_t, x=SimdEq(a=Var("a"), b=Var("b"))), types
            ),
            Bitcast(to=mask_t, x=SimdEq(a=Var("a"), b=Var("b"))),
        )
