import unittest

from stc.tick_ir import (
    BitVecConst,
    SimdAShr,
    SimdAdd,
    SimdConst,
    SimdEq,
    SimdLShr,
    SimdShl,
    SimdSub,
    SimdType,
    SimdUge,
    SimdUlt,
    Var,
    Xor,
)
from stc.z3_prove import prove_equiv


class TestZ3SimdIdentities(unittest.TestCase):
    def test_simd_add_zero(self) -> None:
        t = SimdType(lane_width=4, lanes=2)
        types = {"x": t}
        spec = SimdAdd(a=Var("x"), b=SimdConst(lane_width=4, lanes=2, value=0))
        self.assertTrue(prove_equiv(spec, Var("x"), types))

    def test_simd_sub_zero(self) -> None:
        t = SimdType(lane_width=4, lanes=2)
        types = {"x": t}
        spec = SimdSub(a=Var("x"), b=SimdConst(lane_width=4, lanes=2, value=0))
        self.assertTrue(prove_equiv(spec, Var("x"), types))

    def test_simd_xor_self(self) -> None:
        t = SimdType(lane_width=4, lanes=2)
        types = {"x": t}
        spec = Xor(a=Var("x"), b=Var("x"))
        self.assertTrue(
            prove_equiv(spec, SimdConst(lane_width=4, lanes=2, value=0), types)
        )

    def test_simd_eq_reflexive(self) -> None:
        t = SimdType(lane_width=4, lanes=2)
        types = {"x": t}
        spec = SimdEq(a=Var("x"), b=Var("x"))
        self.assertTrue(
            prove_equiv(spec, SimdConst(lane_width=1, lanes=2, value=0b11), types)
        )

    def test_simd_shift_by_zero(self) -> None:
        t = SimdType(lane_width=4, lanes=2)
        sh0 = BitVecConst(width=4, value=0)
        self.assertTrue(prove_equiv(SimdShl(a=Var("x"), sh=sh0), Var("x"), {"x": t}))
        self.assertTrue(prove_equiv(SimdLShr(a=Var("x"), sh=sh0), Var("x"), {"x": t}))
        self.assertTrue(prove_equiv(SimdAShr(a=Var("x"), sh=sh0), Var("x"), {"x": t}))

    def test_simd_unsigned_cmp_reflexive(self) -> None:
        t = SimdType(lane_width=4, lanes=2)
        types = {"x": t}
        self.assertTrue(
            prove_equiv(
                SimdUge(a=Var("x"), b=Var("x")),
                SimdConst(lane_width=1, lanes=2, value=0b11),
                types,
            )
        )
        self.assertTrue(
            prove_equiv(
                SimdUlt(a=Var("x"), b=Var("x")),
                SimdConst(lane_width=1, lanes=2, value=0),
                types,
            )
        )
