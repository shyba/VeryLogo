import unittest

from stc.interp import eval_expr
from stc.tick_ir import (
    SimdPackSS16To8,
    SimdPackSS32To16,
    SimdPackUS16To8,
    SimdType,
    SimdUnpackHi,
    SimdUnpackLo,
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


def _pack_s32(lanes: list[int]) -> int:
    out = 0
    for i, v in enumerate(lanes):
        out |= (int(v) & 0xFFFFFFFF) << (32 * i)
    return out


def _to_s(x: int, w: int) -> int:
    mask = (1 << w) - 1
    x &= mask
    sign = 1 << (w - 1)
    return x - (1 << w) if (x & sign) else x


def _from_s(x: int, w: int) -> int:
    return x & ((1 << w) - 1)


class TestSimdPackUnpack(unittest.TestCase):
    def test_unpacklo_unpackhi_epi16(self) -> None:
        t = SimdType(lane_width=16, lanes=8)
        types = {"x": t, "y": t}

        x_lanes = list(range(8))
        y_lanes = [100 + i for i in range(8)]
        x = _pack_u16(x_lanes)
        y = _pack_u16(y_lanes)

        lo = int(
            eval_expr(SimdUnpackLo(a=Var("x"), b=Var("y")), types, {"x": x, "y": y})
        )
        hi = int(
            eval_expr(SimdUnpackHi(a=Var("x"), b=Var("y")), types, {"x": x, "y": y})
        )

        lo_expected = _pack_u16(
            [
                x_lanes[0],
                y_lanes[0],
                x_lanes[1],
                y_lanes[1],
                x_lanes[2],
                y_lanes[2],
                x_lanes[3],
                y_lanes[3],
            ]
        )
        hi_expected = _pack_u16(
            [
                x_lanes[4],
                y_lanes[4],
                x_lanes[5],
                y_lanes[5],
                x_lanes[6],
                y_lanes[6],
                x_lanes[7],
                y_lanes[7],
            ]
        )
        self.assertEqual(lo, lo_expected)
        self.assertEqual(hi, hi_expected)

    def test_packsswb_epi16_to_epi8(self) -> None:
        a_t = SimdType(lane_width=16, lanes=8)
        out_t = SimdType(lane_width=8, lanes=16)
        types = {"x": a_t, "y": a_t}
        expr = SimdPackSS16To8(a=Var("x"), b=Var("y"))

        x = _pack_s16([0, 127, 128, -128, -129, 200, -200, 42])
        y = _pack_s16([1, -1, 32767, -32768, 300, -300, 7, -7])
        out = int(eval_expr(expr, types, {"x": x, "y": y}))

        lanes = []
        for src in [x, y]:
            for i in range(8):
                v = _to_s((src >> (16 * i)) & 0xFFFF, 16)
                if v > 127:
                    v = 127
                if v < -128:
                    v = -128
                lanes.append(_from_s(v, 8))
        expected = 0
        for i, v in enumerate(lanes):
            expected |= int(v) << (8 * i)
        self.assertEqual(out, expected & ((1 << out_t.total_width) - 1))

    def test_packuswb_epi16_to_epi8(self) -> None:
        a_t = SimdType(lane_width=16, lanes=8)
        out_t = SimdType(lane_width=8, lanes=16)
        types = {"x": a_t, "y": a_t}
        expr = SimdPackUS16To8(a=Var("x"), b=Var("y"))

        x = _pack_s16([0, 127, 128, -1, -128, 255, 256, 1000])
        y = _pack_s16([-200, 1, 300, 32767, -32768, 42, 500, -500])
        out = int(eval_expr(expr, types, {"x": x, "y": y}))

        lanes = []
        for src in [x, y]:
            for i in range(8):
                v = _to_s((src >> (16 * i)) & 0xFFFF, 16)
                if v < 0:
                    v = 0
                if v > 255:
                    v = 255
                lanes.append(v & 0xFF)
        expected = 0
        for i, v in enumerate(lanes):
            expected |= int(v) << (8 * i)
        self.assertEqual(out, expected & ((1 << out_t.total_width) - 1))

    def test_packssdw_epi32_to_epi16(self) -> None:
        a_t = SimdType(lane_width=32, lanes=4)
        out_t = SimdType(lane_width=16, lanes=8)
        types = {"x": a_t, "y": a_t}
        expr = SimdPackSS32To16(a=Var("x"), b=Var("y"))

        x = _pack_s32([0, 32767, 32768, -32769])
        y = _pack_s32([1, -1, 2**31 - 1, -(2**31)])
        out = int(eval_expr(expr, types, {"x": x, "y": y}))

        lanes = []
        for src in [x, y]:
            for i in range(4):
                v = _to_s((src >> (32 * i)) & 0xFFFFFFFF, 32)
                if v > 32767:
                    v = 32767
                if v < -32768:
                    v = -32768
                lanes.append(_from_s(v, 16))
        expected = _pack_u16(lanes)
        self.assertEqual(out, expected & ((1 << out_t.total_width) - 1))
