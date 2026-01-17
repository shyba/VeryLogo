import unittest

import z3

from stc.interp import eval_expr
from stc.tick_ir import (
    SimdBlend,
    SimdMaxS,
    SimdMaxU,
    SimdMinS,
    SimdMinU,
    SimdSExtLo,
    SimdType,
    SimdZExtLo,
    Var,
)
from stc.z3_encode import encode_expr


def _sat_matches(expr, types, env) -> bool:
    z3_vars = {}
    out = encode_expr(expr, types, z3_vars)
    s = z3.Solver()
    expected = eval_expr(expr, types, env)
    for name, val in env.items():
        if name not in z3_vars:
            continue
        v = z3_vars[name]
        t = types[name]
        assert isinstance(t, SimdType)
        s.add(v == z3.BitVecVal(int(val), t.total_width))
    assert isinstance(out, z3.BitVecRef)
    s.add(out == z3.BitVecVal(int(expected), out.size()))
    return s.check() == z3.sat


def _pack_u8(lanes: list[int]) -> int:
    out = 0
    for i, v in enumerate(lanes):
        out |= (int(v) & 0xFF) << (8 * i)
    return out


def _to_s8(x: int) -> int:
    x &= 0xFF
    return x - 0x100 if (x & 0x80) else x


def _pack_u16(lanes: list[int]) -> int:
    out = 0
    for i, v in enumerate(lanes):
        out |= (int(v) & 0xFFFF) << (16 * i)
    return out


def _pack_u32(lanes: list[int]) -> int:
    out = 0
    for i, v in enumerate(lanes):
        out |= (int(v) & 0xFFFFFFFF) << (32 * i)
    return out


def _to_s32(x: int) -> int:
    x &= 0xFFFFFFFF
    return x - 0x100000000 if (x & 0x80000000) else x


class TestSimdSse41Ops(unittest.TestCase):
    def test_min_max_unsigned_epi8(self) -> None:
        t = SimdType(lane_width=8, lanes=16)
        types = {"x": t, "y": t}
        x = _pack_u8([0, 1, 2, 255, 128, 127, 10, 20] + [0] * 8)
        y = _pack_u8([5, 0, 3, 1, 200, 126, 10, 21] + [0] * 8)

        mn = SimdMinU(a=Var("x"), b=Var("y"))
        mx = SimdMaxU(a=Var("x"), b=Var("y"))
        mnv = int(eval_expr(mn, types, {"x": x, "y": y}))
        mxv = int(eval_expr(mx, types, {"x": x, "y": y}))

        lanes_min = []
        lanes_max = []
        for i in range(16):
            a = (x >> (8 * i)) & 0xFF
            b = (y >> (8 * i)) & 0xFF
            lanes_min.append(min(a, b))
            lanes_max.append(max(a, b))
        self.assertEqual(mnv, _pack_u8(lanes_min))
        self.assertEqual(mxv, _pack_u8(lanes_max))
        self.assertTrue(_sat_matches(mn, types, {"x": x, "y": y}))
        self.assertTrue(_sat_matches(mx, types, {"x": x, "y": y}))

    def test_min_max_signed_epi32(self) -> None:
        t = SimdType(lane_width=32, lanes=4)
        types = {"x": t, "y": t}
        x = _pack_u32([0x7FFFFFFF, 0x80000000, 1, 0xFFFFFFFF])
        y = _pack_u32([1, 0xFFFFFFFF, 2, 0x80000000])

        mn = SimdMinS(a=Var("x"), b=Var("y"))
        mx = SimdMaxS(a=Var("x"), b=Var("y"))
        mnv = int(eval_expr(mn, types, {"x": x, "y": y}))
        mxv = int(eval_expr(mx, types, {"x": x, "y": y}))

        lanes_min = []
        lanes_max = []
        for i in range(4):
            a = _to_s32((x >> (32 * i)) & 0xFFFFFFFF)
            b = _to_s32((y >> (32 * i)) & 0xFFFFFFFF)
            lanes_min.append(a if a < b else b)
            lanes_max.append(a if a > b else b)
        self.assertEqual(mnv, _pack_u32([v & 0xFFFFFFFF for v in lanes_min]))
        self.assertEqual(mxv, _pack_u32([v & 0xFFFFFFFF for v in lanes_max]))
        self.assertTrue(_sat_matches(mn, types, {"x": x, "y": y}))
        self.assertTrue(_sat_matches(mx, types, {"x": x, "y": y}))

    def test_blend_per_lane(self) -> None:
        v = SimdType(lane_width=32, lanes=4)
        m = SimdType(lane_width=1, lanes=4)
        types = {"a": v, "b": v, "m": m}
        a = _pack_u32([1, 2, 3, 4])
        b = _pack_u32([10, 20, 30, 40])
        mask = 0b1010
        expr = SimdBlend(mask=Var("m"), a=Var("a"), b=Var("b"))
        out = int(eval_expr(expr, types, {"a": a, "b": b, "m": mask}))
        expected = _pack_u32([1, 20, 3, 40])
        self.assertEqual(out, expected)
        self.assertTrue(_sat_matches(expr, types, {"a": a, "b": b, "m": mask}))

    def test_zext_sext_lo(self) -> None:
        x8 = SimdType(lane_width=8, lanes=16)
        z16 = SimdType(lane_width=16, lanes=8)
        types = {"x": x8}
        x = _pack_u8([0, 1, 2, 255, 128, 127, 10, 20] + [0] * 8)

        zext = SimdZExtLo(to=z16, x=Var("x"))
        outz = int(eval_expr(zext, types, {"x": x}))
        lanes = []
        for i in range(8):
            lanes.append(((x >> (8 * i)) & 0xFF) & 0xFFFF)
        self.assertEqual(outz, _pack_u16(lanes))
        self.assertTrue(_sat_matches(zext, types, {"x": x}))

        sext = SimdSExtLo(to=z16, x=Var("x"))
        outs = int(eval_expr(sext, types, {"x": x}))
        lanes = []
        for i in range(8):
            v = _to_s8((x >> (8 * i)) & 0xFF)
            lanes.append(v & 0xFFFF)
        self.assertEqual(outs, _pack_u16(lanes))
        self.assertTrue(_sat_matches(sext, types, {"x": x}))
