import unittest

import z3

from stc.interp import eval_expr
from stc.tick_ir import SimdMaskExpand, SimdMaskPack, SimdType, Var
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


class TestSimdMaskPackUnpack(unittest.TestCase):
    def test_expand_and_pack_roundtrip(self) -> None:
        m_t = SimdType(lane_width=1, lanes=16)
        v_t = SimdType(lane_width=8, lanes=16)
        types = {"m": m_t}
        mask = 0b1010_1100_0011_0101

        expanded = SimdMaskExpand(to=v_t, x=Var("m"))
        packed = SimdMaskPack(x=expanded)

        got = int(eval_expr(packed, types, {"m": mask}))
        self.assertEqual(got, mask & ((1 << 16) - 1))
        self.assertTrue(_sat_matches(packed, types, {"m": mask}))

    def test_expand_sets_all_ones_per_lane(self) -> None:
        m_t = SimdType(lane_width=1, lanes=16)
        v_t = SimdType(lane_width=8, lanes=16)
        types = {"m": m_t}
        mask = 0b0101_0101_0101_0101

        expr = SimdMaskExpand(to=v_t, x=Var("m"))
        out = int(eval_expr(expr, types, {"m": mask}))
        for i in range(16):
            lane = (out >> (8 * i)) & 0xFF
            self.assertEqual(lane, 0xFF if ((mask >> i) & 1) else 0x00)

    def test_pack_treats_nonzero_as_true(self) -> None:
        v_t = SimdType(lane_width=16, lanes=8)
        types = {"x": v_t}
        x = 0
        for i in range(8):
            lane = 0xFFFF if (i % 2 == 0) else 0x0000
            x |= lane << (16 * i)
        expr = SimdMaskPack(x=Var("x"))
        out = int(eval_expr(expr, types, {"x": x}))
        self.assertEqual(out, 0b0101_0101)
        self.assertTrue(_sat_matches(expr, types, {"x": x}))
