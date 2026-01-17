import unittest

from stc.interp import eval_expr
from stc.tick_ir import (
    BitVecConst,
    BitVecType,
    SimdAShr,
    SimdConst,
    SimdLShr,
    SimdShl,
    SimdType,
    Var,
)


class TestSimdShifts(unittest.TestCase):
    def test_simd_shl_scalar_shift(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        sh_t = BitVecType(width=4)
        types = {"x": simd, "sh": sh_t}
        expr = SimdShl(a=Var(name="x"), sh=Var(name="sh"))
        self.assertEqual(eval_expr(expr, types, {"x": 0x81, "sh": 1}), 0x02)

    def test_simd_lshr_scalar_shift(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        sh_t = BitVecType(width=4)
        types = {"x": simd, "sh": sh_t}
        expr = SimdLShr(a=Var(name="x"), sh=Var(name="sh"))
        self.assertEqual(eval_expr(expr, types, {"x": 0x81, "sh": 1}), 0x40)

    def test_simd_ashr_scalar_shift(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        sh_t = BitVecType(width=4)
        types = {"x": simd, "sh": sh_t}
        expr = SimdAShr(a=Var(name="x"), sh=Var(name="sh"))
        self.assertEqual(eval_expr(expr, types, {"x": 0x81, "sh": 1}), 0xC0)

    def test_simd_ashr_shift_ge_width(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        expr = SimdAShr(
            a=SimdConst(lane_width=4, lanes=2, value=0x81),
            sh=BitVecConst(width=4, value=7),
        )
        self.assertEqual(eval_expr(expr, {}, {}), 0xF0)
