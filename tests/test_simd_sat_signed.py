import unittest

from stc.interp import eval_expr
from stc.tick_ir import SimdAddSatS, SimdSubSatS, SimdType, Var


def _to_signed(x: int, w: int) -> int:
    sign = 1 << (w - 1)
    x &= (1 << w) - 1
    return x - (1 << w) if (x & sign) else x


def _from_signed(x: int, w: int) -> int:
    return x & ((1 << w) - 1)


class TestSimdSatSigned(unittest.TestCase):
    def test_adds_epi8_saturates(self) -> None:
        t = SimdType(lane_width=8, lanes=16)
        types = {"x": t, "y": t}
        expr = SimdAddSatS(a=Var("x"), b=Var("y"))

        x_bytes = [0x7F, 0x80, 0x01, 0xFE, 0x40, 0xC0, 0x10, 0xF0] + [0] * 8
        y_bytes = [0x01, 0xFF, 0x7F, 0x80, 0x40, 0xC0, 0x70, 0x90] + [0] * 8
        x = sum(int(b) << (8 * i) for i, b in enumerate(x_bytes))
        y = sum(int(b) << (8 * i) for i, b in enumerate(y_bytes))
        out = int(eval_expr(expr, types, {"x": x, "y": y}))

        lanes = []
        for i in range(16):
            a = _to_signed((x >> (i * 8)) & 0xFF, 8)
            b = _to_signed((y >> (i * 8)) & 0xFF, 8)
            s = a + b
            if s > 127:
                s = 127
            if s < -128:
                s = -128
            lanes.append(_from_signed(s, 8))
        expected = sum(int(v) << (i * 8) for i, v in enumerate(lanes))
        self.assertEqual(out, expected)

    def test_subs_epi16_saturates(self) -> None:
        t = SimdType(lane_width=16, lanes=8)
        types = {"x": t, "y": t}
        expr = SimdSubSatS(a=Var("x"), b=Var("y"))

        x_words = [0x7FFF, 0x8000, 0x0001, 0xFFFE, 0x4000, 0xC000, 0x0000, 0x0000]
        y_words = [0xFFFF, 0x0001, 0x7FFF, 0x8000, 0xC000, 0x4000, 0x0001, 0xFFFF]
        x = sum(int(w) << (16 * i) for i, w in enumerate(x_words))
        y = sum(int(w) << (16 * i) for i, w in enumerate(y_words))
        out = int(eval_expr(expr, types, {"x": x, "y": y}))

        lanes = []
        for i in range(8):
            a = _to_signed((x >> (i * 16)) & 0xFFFF, 16)
            b = _to_signed((y >> (i * 16)) & 0xFFFF, 16)
            s = a - b
            if s > 32767:
                s = 32767
            if s < -32768:
                s = -32768
            lanes.append(_from_signed(s, 16))
        expected = sum(int(v) << (i * 16) for i, v in enumerate(lanes))
        self.assertEqual(out, expected)
