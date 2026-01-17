import unittest

from stc.interp import eval_expr
from stc.tick_ir import SimdAddSatU, SimdSubSatU, SimdType, Var


class TestSimdSatUnsigned(unittest.TestCase):
    def test_adds_epu8_saturates(self) -> None:
        t = SimdType(lane_width=8, lanes=16)
        types = {"x": t, "y": t}
        expr = SimdAddSatU(a=Var("x"), b=Var("y"))

        x = 0x00_FF_01_FE_10_F0_80_7F_00_01_02_03_04_05_06_07
        y = 0x00_01_FF_02_F0_20_80_01_FF_FE_FD_FC_FB_FA_F9_F8
        out = int(eval_expr(expr, types, {"x": x, "y": y}))

        lanes = []
        for i in range(16):
            a = (x >> (i * 8)) & 0xFF
            b = (y >> (i * 8)) & 0xFF
            lanes.append(min(255, a + b))
        expected = sum(int(v) << (i * 8) for i, v in enumerate(lanes))
        self.assertEqual(out, expected)

    def test_subs_epu16_saturates(self) -> None:
        t = SimdType(lane_width=16, lanes=8)
        types = {"x": t, "y": t}
        expr = SimdSubSatU(a=Var("x"), b=Var("y"))

        x = 0x0001_0000_FFFF_8000_0002_0003_0004_0005
        y = 0x0002_0001_0001_8001_0001_0004_0005_0006
        out = int(eval_expr(expr, types, {"x": x, "y": y}))

        lanes = []
        for i in range(8):
            a = (x >> (i * 16)) & 0xFFFF
            b = (y >> (i * 16)) & 0xFFFF
            lanes.append(0 if a < b else (a - b))
        expected = sum(int(v) << (i * 16) for i, v in enumerate(lanes))
        self.assertEqual(out, expected)
