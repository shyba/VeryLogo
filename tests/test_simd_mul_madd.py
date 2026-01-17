import unittest

from stc.interp import eval_expr
from stc.tick_ir import (
    SimdMaddS16,
    SimdMulHiS,
    SimdMulHiU,
    SimdMulLo,
    SimdType,
    Var,
)


def _pack_u16(lanes: list[int]) -> int:
    out = 0
    for i, v in enumerate(lanes):
        out |= (int(v) & 0xFFFF) << (16 * i)
    return out


def _pack_s16(lanes: list[int]) -> int:
    out = 0
    for i, v in enumerate(lanes):
        out |= (int(v) & 0xFFFF) << (16 * i)
    return out


def _to_s16(x: int) -> int:
    x &= 0xFFFF
    return x - 0x10000 if (x & 0x8000) else x


class TestSimdMulMadd(unittest.TestCase):
    def test_mullo_epi16_wraps(self) -> None:
        t = SimdType(lane_width=16, lanes=8)
        types = {"x": t, "y": t}
        expr = SimdMulLo(a=Var("x"), b=Var("y"))

        x = _pack_u16([0xFFFF, 0x8000, 0x1234, 0x0002, 0x0000, 0x0001, 0x00FF, 0x00FF])
        y = _pack_u16([0x0002, 0x0002, 0x0002, 0xFFFF, 0x1234, 0x8000, 0x0101, 0xFFFF])
        out = int(eval_expr(expr, types, {"x": x, "y": y}))

        lanes = []
        for i in range(8):
            a = (x >> (16 * i)) & 0xFFFF
            b = (y >> (16 * i)) & 0xFFFF
            lanes.append((a * b) & 0xFFFF)
        self.assertEqual(out, _pack_u16(lanes))

    def test_mulhi_unsigned_epi16(self) -> None:
        t = SimdType(lane_width=16, lanes=8)
        types = {"x": t, "y": t}
        expr = SimdMulHiU(a=Var("x"), b=Var("y"))

        x = _pack_u16([0x8000, 0xFFFF, 0x1234, 0x0001, 0, 0, 0, 0])
        y = _pack_u16([0x8000, 0x0002, 0x1000, 0xFFFF, 0, 0, 0, 0])
        out = int(eval_expr(expr, types, {"x": x, "y": y}))

        lanes = []
        for i in range(8):
            a = (x >> (16 * i)) & 0xFFFF
            b = (y >> (16 * i)) & 0xFFFF
            lanes.append(((a * b) >> 16) & 0xFFFF)
        self.assertEqual(out, _pack_u16(lanes))

    def test_mulhi_signed_epi16(self) -> None:
        t = SimdType(lane_width=16, lanes=8)
        types = {"x": t, "y": t}
        expr = SimdMulHiS(a=Var("x"), b=Var("y"))

        x = _pack_s16([0xFFFF, 0x8000, 0x7FFF, 0x0001, 0, 0, 0, 0])
        y = _pack_s16([0x0002, 0x0002, 0x0002, 0xFFFF, 0, 0, 0, 0])
        out = int(eval_expr(expr, types, {"x": x, "y": y}))

        lanes = []
        for i in range(8):
            a = _to_s16((x >> (16 * i)) & 0xFFFF)
            b = _to_s16((y >> (16 * i)) & 0xFFFF)
            prod = a * b
            lanes.append((prod >> 16) & 0xFFFF)
        self.assertEqual(out, _pack_u16(lanes))

    def test_madd_epi16_outputs_epi32(self) -> None:
        a_t = SimdType(lane_width=16, lanes=8)
        out_t = SimdType(lane_width=32, lanes=4)
        types = {"x": a_t, "y": a_t}
        expr = SimdMaddS16(a=Var("x"), b=Var("y"))

        x = _pack_s16([1, 2, -3, 4, 5, -6, 7, 8])
        y = _pack_s16([10, 10, 10, 10, -2, -2, 1, 1])
        out = int(eval_expr(expr, types, {"x": x, "y": y}))

        lanes = []
        for i in range(4):
            a0 = _to_s16((x >> (16 * (2 * i))) & 0xFFFF)
            a1 = _to_s16((x >> (16 * (2 * i + 1))) & 0xFFFF)
            b0 = _to_s16((y >> (16 * (2 * i))) & 0xFFFF)
            b1 = _to_s16((y >> (16 * (2 * i + 1))) & 0xFFFF)
            s = (a0 * b0) + (a1 * b1)
            lanes.append(int(s) & 0xFFFFFFFF)
        expected = 0
        for i, v in enumerate(lanes):
            expected |= int(v) << (32 * i)
        self.assertEqual(out, expected & ((1 << out_t.total_width) - 1))
