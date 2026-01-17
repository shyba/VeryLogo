import unittest

from stc.tick_ir import (
    Add,
    Bitcast,
    BitVecType,
    Concat,
    SimdAdd,
    SimdConst,
    SimdType,
    Slice,
    Var,
)
from stc.z3_prove import prove_not_equiv


class TestZ3SimdNonIdentities(unittest.TestCase):
    def test_packed_add_is_not_simd_add(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        bv8 = BitVecType(width=8)
        types = {"x": simd, "y": simd}

        packed = Bitcast(
            to=simd, x=Add(a=Bitcast(to=bv8, x=Var("x")), b=Bitcast(to=bv8, x=Var("y")))
        )
        lane = SimdAdd(a=Var("x"), b=Var("y"))
        self.assertTrue(prove_not_equiv(packed, lane, types))

    def test_scalar_lane_concat_is_not_simd_const(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        bv8 = BitVecType(width=8)
        types = {"x": simd}
        xb = Bitcast(to=bv8, x=Var("x"))
        lo = Slice(x=xb, offset=0, width=4)
        hi = Slice(x=xb, offset=4, width=4)
        spec = Bitcast(to=simd, x=Concat(parts=[lo, hi]))
        self.assertTrue(
            prove_not_equiv(spec, SimdConst(lane_width=4, lanes=2, value=0), types)
        )
