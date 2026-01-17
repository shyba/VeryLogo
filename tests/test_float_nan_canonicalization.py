import unittest

import z3

from stc.interp import eval_expr
from stc.tick_ir import (
    FloatConst,
    FloatType,
    FAbs,
    FNeg,
    SimdFAbs,
    SimdFNeg,
    SimdType,
    Var,
)
from stc.z3_encode import encode_expr


class TestFloatNaNCanonicalization(unittest.TestCase):
    def test_interp_canonicalizes_f32_nan_results(self) -> None:
        nan_payload = 0x7FA12345
        canon = 0x7FC00000
        t = FloatType(width=32)
        types = {"x": t}
        env = {"x": nan_payload}
        self.assertEqual(int(eval_expr(FAbs(x=Var("x")), types, env)), canon)
        self.assertEqual(int(eval_expr(FNeg(x=Var("x")), types, env)), canon)

    def test_interp_canonicalizes_f64_nan_results(self) -> None:
        nan_payload = 0x7FF0123456789ABC
        canon = 0x7FF8000000000000
        t = FloatType(width=64)
        types = {"x": t}
        env = {"x": nan_payload}
        self.assertEqual(int(eval_expr(FAbs(x=Var("x")), types, env)), canon)
        self.assertEqual(int(eval_expr(FNeg(x=Var("x")), types, env)), canon)

    def test_interp_canonicalizes_simd_nan_lanes(self) -> None:
        nan_payload = 0x7FA12345
        canon = 0x7FC00000
        t = SimdType(lane_width=32, lanes=8)
        types = {"x": t}
        x = 0
        x |= nan_payload << (32 * 0)
        x |= 0x3F800000 << (32 * 1)
        env = {"x": x}
        got_abs = int(eval_expr(SimdFAbs(x=Var("x")), types, env))
        got_neg = int(eval_expr(SimdFNeg(x=Var("x")), types, env))
        self.assertEqual((got_abs >> 0) & 0xFFFFFFFF, canon)
        self.assertEqual((got_neg >> 0) & 0xFFFFFFFF, canon)

    def test_z3_canonicalizes_f32_nan_results(self) -> None:
        nan_payload = 0x7FA12345
        canon = 0x7FC00000
        t = FloatType(width=32)
        types = {"x": t}

        out_abs = encode_expr(FAbs(x=Var("x")), types, {})
        out_neg = encode_expr(FNeg(x=Var("x")), types, {})
        assert isinstance(out_abs, z3.BitVecRef)
        assert isinstance(out_neg, z3.BitVecRef)

        x = z3.BitVec("x", 32)
        s = z3.Solver()
        s.add(x == z3.BitVecVal(nan_payload, 32))
        s.add(out_abs == z3.BitVecVal(canon, 32))
        s.add(out_neg == z3.BitVecVal(canon, 32))
        self.assertEqual(s.check(), z3.sat)

    def test_z3_canonicalizes_simd_nan_lanes(self) -> None:
        nan_payload = 0x7FA12345
        canon = 0x7FC00000
        t = SimdType(lane_width=32, lanes=8)
        types = {"x": t}

        out_abs = encode_expr(SimdFAbs(x=Var("x")), types, {})
        out_neg = encode_expr(SimdFNeg(x=Var("x")), types, {})
        assert isinstance(out_abs, z3.BitVecRef)
        assert isinstance(out_neg, z3.BitVecRef)

        x = z3.BitVec("x", 256)
        s = z3.Solver()
        in_bits = 0
        in_bits |= nan_payload << (32 * 0)
        in_bits |= 0x3F800000 << (32 * 1)
        s.add(x == z3.BitVecVal(in_bits, 256))
        s.add(z3.Extract(31, 0, out_abs) == z3.BitVecVal(canon, 32))
        s.add(z3.Extract(31, 0, out_neg) == z3.BitVecVal(canon, 32))
        self.assertEqual(s.check(), z3.sat)

    def test_const_is_not_canonicalized_outside_ops(self) -> None:
        nan_payload = 0x7FA12345
        t = FloatType(width=32)
        self.assertEqual(
            int(eval_expr(FloatConst(width=32, bits=nan_payload), {}, {})), nan_payload
        )
