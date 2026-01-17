import unittest

from stc.autovec import autovectorize_expr
from stc.tick_ir import (
    Add,
    Sub,
    Bitcast,
    BitVecType,
    Concat,
    SimdAdd,
    SimdSub,
    SimdType,
    SimdUlt,
    Slice,
    Ult,
    Var,
)


class TestAutovecMatrix(unittest.TestCase):
    def test_autovec_add_matrix(self) -> None:
        for lane_width in [2, 4, 8]:
            for lanes in [2, 4]:
                simd = SimdType(lane_width=lane_width, lanes=lanes)
                bv = BitVecType(width=lane_width * lanes)
                types = {"x": simd, "y": simd}

                xb = Bitcast(to=bv, x=Var("x"))
                yb = Bitcast(to=bv, x=Var("y"))
                parts = []
                for lane in reversed(range(lanes)):
                    off = lane * lane_width
                    parts.append(
                        Add(
                            a=Slice(x=xb, offset=off, width=lane_width),
                            b=Slice(x=yb, offset=off, width=lane_width),
                        )
                    )
                spec = Bitcast(to=simd, x=Concat(parts=parts))
                self.assertEqual(
                    autovectorize_expr(spec, types, timeout_ms=200),
                    SimdAdd(a=Var("x"), b=Var("y")),
                )

    def test_autovec_sub_matrix(self) -> None:
        for lane_width in [2, 4, 8]:
            for lanes in [2, 4]:
                simd = SimdType(lane_width=lane_width, lanes=lanes)
                bv = BitVecType(width=lane_width * lanes)
                types = {"x": simd, "y": simd}

                xb = Bitcast(to=bv, x=Var("x"))
                yb = Bitcast(to=bv, x=Var("y"))
                parts = []
                for lane in reversed(range(lanes)):
                    off = lane * lane_width
                    parts.append(
                        Sub(
                            a=Slice(x=xb, offset=off, width=lane_width),
                            b=Slice(x=yb, offset=off, width=lane_width),
                        )
                    )
                spec = Bitcast(to=simd, x=Concat(parts=parts))
                self.assertEqual(
                    autovectorize_expr(spec, types, timeout_ms=200),
                    SimdSub(a=Var("x"), b=Var("y")),
                )

    def test_autovec_ult_mask_matrix(self) -> None:
        for lane_width in [2, 4, 8]:
            for lanes in [2, 4]:
                simd = SimdType(lane_width=lane_width, lanes=lanes)
                mask = SimdType(lane_width=1, lanes=lanes)
                bv = BitVecType(width=lane_width * lanes)
                types = {"x": simd, "y": simd}

                xb = Bitcast(to=bv, x=Var("x"))
                yb = Bitcast(to=bv, x=Var("y"))
                parts = []
                for lane in reversed(range(lanes)):
                    off = lane * lane_width
                    parts.append(
                        Ult(
                            a=Slice(x=xb, offset=off, width=lane_width),
                            b=Slice(x=yb, offset=off, width=lane_width),
                        )
                    )
                spec = Bitcast(to=mask, x=Concat(parts=parts))
                self.assertEqual(
                    autovectorize_expr(spec, types, timeout_ms=200),
                    SimdUlt(a=Var("x"), b=Var("y")),
                )
