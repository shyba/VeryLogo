import unittest

from stc.autovec import autovectorize_expr
from stc.tick_ir import (
    Bitcast,
    BitVecType,
    Concat,
    Mux,
    SimdMaxU,
    SimdMinU,
    SimdType,
    Slice,
    Ult,
    Var,
)


class TestAutovecMinMax(unittest.TestCase):
    def test_autovec_minu_from_lane_mux(self) -> None:
        simd = SimdType(lane_width=16, lanes=8)
        bv128 = BitVecType(width=128)
        types = {"a": simd, "b": simd}

        ab = Bitcast(to=bv128, x=Var("a"))
        bb = Bitcast(to=bv128, x=Var("b"))
        parts = []
        for lane in range(7, -1, -1):
            off = lane * 16
            al = Slice(x=ab, offset=off, width=16)
            bl = Slice(x=bb, offset=off, width=16)
            parts.append(Mux(cond=Ult(a=al, b=bl), a=al, b=bl))
        spec = Bitcast(to=simd, x=Concat(parts=parts))

        out = autovectorize_expr(spec, types, timeout_ms=200)
        self.assertEqual(out, SimdMinU(a=Var("a"), b=Var("b")))

    def test_autovec_maxu_from_lane_mux(self) -> None:
        simd = SimdType(lane_width=16, lanes=8)
        bv128 = BitVecType(width=128)
        types = {"a": simd, "b": simd}

        ab = Bitcast(to=bv128, x=Var("a"))
        bb = Bitcast(to=bv128, x=Var("b"))
        parts = []
        for lane in range(7, -1, -1):
            off = lane * 16
            al = Slice(x=ab, offset=off, width=16)
            bl = Slice(x=bb, offset=off, width=16)
            parts.append(Mux(cond=Ult(a=al, b=bl), a=bl, b=al))
        spec = Bitcast(to=simd, x=Concat(parts=parts))

        out = autovectorize_expr(spec, types, timeout_ms=200)
        self.assertEqual(out, SimdMaxU(a=Var("a"), b=Var("b")))
