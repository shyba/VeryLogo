import unittest

import z3

from stc.interp import eval_expr
from stc.tick_ir import (
    BitVecConst,
    BitVecType,
    SimdAdd,
    SimdAddSatS,
    SimdAddSatU,
    SimdConst,
    SimdEq,
    SimdMulHiS,
    SimdMulHiU,
    SimdMulLo,
    SimdShl,
    SimdType,
    SimdSubSatS,
    SimdSubSatU,
    SimdUlt,
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
        if isinstance(t, BitVecType):
            s.add(v == z3.BitVecVal(int(val), t.width))
        else:
            assert isinstance(t, SimdType)
            s.add(v == z3.BitVecVal(int(val), t.total_width))
    if isinstance(out, z3.BoolRef):
        s.add(out == z3.BoolVal(bool(expected)))
    else:
        assert isinstance(out, z3.BitVecRef)
        s.add(out == z3.BitVecVal(int(expected), out.size()))
    return s.check() == z3.sat


class TestSimdInterpVsZ3(unittest.TestCase):
    def test_concrete_evaluation_matches(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        sh_t = BitVecType(width=4)
        types = {"x": simd, "y": simd, "sh": sh_t}
        exprs = [
            SimdAdd(a=Var("x"), b=Var("y")),
            SimdAddSatU(a=Var("x"), b=Var("y")),
            SimdSubSatU(a=Var("x"), b=Var("y")),
            SimdAddSatS(a=Var("x"), b=Var("y")),
            SimdSubSatS(a=Var("x"), b=Var("y")),
            SimdMulLo(a=Var("x"), b=Var("y")),
            SimdMulHiU(a=Var("x"), b=Var("y")),
            SimdMulHiS(a=Var("x"), b=Var("y")),
            SimdEq(a=Var("x"), b=Var("y")),
            SimdUlt(a=Var("x"), b=Var("y")),
            SimdShl(a=Var("x"), sh=Var("sh")),
            SimdAdd(a=Var("x"), b=SimdConst(lane_width=4, lanes=2, value=0)),
            SimdShl(a=Var("x"), sh=BitVecConst(width=4, value=1)),
        ]

        samples = [
            {"x": 0x00, "y": 0x00, "sh": 0},
            {"x": 0xAB, "y": 0xAC, "sh": 1},
            {"x": 0x81, "y": 0x13, "sh": 3},
            {"x": 0xFF, "y": 0x01, "sh": 7},
        ]

        for e in exprs:
            for env in samples:
                self.assertTrue(_sat_matches(e, types, env))
