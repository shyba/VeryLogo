import unittest

import z3

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


class TestSimdPackUnpackVsZ3(unittest.TestCase):
    def test_pack_unpack_matches_on_concrete_samples(self) -> None:
        t16 = SimdType(lane_width=16, lanes=8)
        t8 = SimdType(lane_width=8, lanes=16)
        t32 = SimdType(lane_width=32, lanes=4)
        t16o = SimdType(lane_width=16, lanes=8)

        exprs = [
            SimdUnpackLo(a=Var("x16"), b=Var("y16")),
            SimdUnpackHi(a=Var("x16"), b=Var("y16")),
            SimdPackSS16To8(a=Var("x16"), b=Var("y16")),
            SimdPackUS16To8(a=Var("x16"), b=Var("y16")),
            SimdPackSS32To16(a=Var("x32"), b=Var("y32")),
        ]

        types = {"x16": t16, "y16": t16, "x32": t32, "y32": t32}
        samples = [
            {
                "x16": 0x0000_0001_0002_0003_0004_0005_0006_0007,
                "y16": 0x0064_0065_0066_0067_0068_0069_006A_006B,
                "x32": 0x00000000_00007FFF_00008000_FFFF7FFF,
                "y32": 0x7FFFFFFF_80000000_00000001_FFFFFFFF,
            },
            {
                "x16": 0x7FFF_8000_00FF_FF00_0000_007F_0080_FF80,
                "y16": 0xFFFF_0001_0100_FF00_00C8_FF38_012C_FED4,
                "x32": 0x00010000_FFFE0000_7FFFFFFF_80000000,
                "y32": 0x00000000_00000000_00000000_00000000,
            },
        ]

        for e in exprs:
            for env in samples:
                self.assertTrue(_sat_matches(e, types, env))
