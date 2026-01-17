import struct
import unittest

import z3

from stc.interp import eval_expr
from stc.tick_ir import (
    FloatConst,
    FloatType,
    FAdd,
    FAbs,
    FDiv,
    FEq,
    FLe,
    FLt,
    FNeg,
    FMul,
    FNe,
    FSqrt,
    FSub,
    SimdFAdd,
    SimdFCmpEq,
    SimdFCmpLe,
    SimdFCmpLt,
    SimdFCmpNe,
    SimdFDiv,
    SimdFFma,
    SimdFNeg,
    SimdFSqrt,
    SimdFSub,
    SimdFMul,
    SimdFAbs,
    SimdType,
    Var,
)
from stc.z3_encode import encode_expr


def _f32_bits(x: float) -> int:
    return int.from_bytes(struct.pack("<f", float(x)), "little")


def _f64_bits(x: float) -> int:
    return int.from_bytes(struct.pack("<d", float(x)), "little")


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
        if isinstance(t, FloatType):
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


class TestFloatOpsVsZ3(unittest.TestCase):
    def test_signed_zero_f32_scalar_and_simd_matches(self) -> None:
        pos0 = 0x00000000
        neg0 = 0x80000000

        t = FloatType(width=32)
        types = {"a": t, "b": t}
        env = {"a": neg0, "b": pos0}
        exprs = [
            FNeg(x=Var("b")),
            FAbs(x=Var("a")),
            FEq(a=Var("a"), b=Var("b")),
            FNe(a=Var("a"), b=Var("b")),
            FLt(a=Var("a"), b=Var("b")),
            FLe(a=Var("a"), b=Var("b")),
        ]
        for e in exprs:
            self.assertTrue(_sat_matches(e, types, env))

        vx = 0
        for i, bits in enumerate([pos0, neg0] * 4):
            vx |= int(bits) << (32 * i)
        vt = SimdType(lane_width=32, lanes=8)
        types2 = {"x": vt}
        env2 = {"x": vx}
        exprs2 = [
            SimdFNeg(x=Var("x")),
            SimdFAbs(x=Var("x")),
            SimdFCmpEq(a=Var("x"), b=SimdFAbs(x=Var("x"))),
            SimdFCmpNe(a=Var("x"), b=SimdFAbs(x=Var("x"))),
            SimdFCmpLt(a=Var("x"), b=SimdFAbs(x=Var("x"))),
            SimdFCmpLe(a=Var("x"), b=SimdFAbs(x=Var("x"))),
        ]
        for e in exprs2:
            self.assertTrue(_sat_matches(e, types2, env2))

    def test_signed_zero_f64_scalar_and_simd_matches(self) -> None:
        pos0 = 0x0000000000000000
        neg0 = 0x8000000000000000

        t = FloatType(width=64)
        types = {"a": t, "b": t}
        env = {"a": neg0, "b": pos0}
        exprs = [
            FNeg(x=Var("b")),
            FAbs(x=Var("a")),
            FEq(a=Var("a"), b=Var("b")),
            FNe(a=Var("a"), b=Var("b")),
            FLt(a=Var("a"), b=Var("b")),
            FLe(a=Var("a"), b=Var("b")),
        ]
        for e in exprs:
            self.assertTrue(_sat_matches(e, types, env))

        vx = 0
        for i, bits in enumerate([pos0, neg0] * 4):
            vx |= int(bits) << (64 * i)
        vt = SimdType(lane_width=64, lanes=8)
        types2 = {"x": vt}
        env2 = {"x": vx}
        exprs2 = [
            SimdFNeg(x=Var("x")),
            SimdFAbs(x=Var("x")),
            SimdFCmpEq(a=Var("x"), b=SimdFAbs(x=Var("x"))),
            SimdFCmpNe(a=Var("x"), b=SimdFAbs(x=Var("x"))),
            SimdFCmpLt(a=Var("x"), b=SimdFAbs(x=Var("x"))),
            SimdFCmpLe(a=Var("x"), b=SimdFAbs(x=Var("x"))),
        ]
        for e in exprs2:
            self.assertTrue(_sat_matches(e, types2, env2))

    def test_f32_add_mul_cmp_matches(self) -> None:
        t = FloatType(width=32)
        types = {"a": t, "b": t}
        env = {"a": _f32_bits(1.5), "b": _f32_bits(-2.0)}
        exprs = [
            FAdd(a=Var("a"), b=Var("b")),
            FSub(a=Var("a"), b=Var("b")),
            FMul(a=Var("a"), b=Var("b")),
            FDiv(a=Var("a"), b=Var("b")),
            FSqrt(x=Var("a")),
            FNeg(x=Var("a")),
            FAbs(x=Var("a")),
            FEq(a=Var("a"), b=Var("b")),
            FLt(a=Var("a"), b=Var("b")),
            FLe(a=Var("a"), b=Var("b")),
            FNe(a=Var("a"), b=Var("b")),
        ]
        for e in exprs:
            self.assertTrue(_sat_matches(e, types, env))

    def test_f64_add_mul_cmp_matches(self) -> None:
        t = FloatType(width=64)
        types = {"a": t, "b": t}
        env = {"a": _f64_bits(3.25), "b": _f64_bits(0.5)}
        exprs = [
            FAdd(a=Var("a"), b=Var("b")),
            FSub(a=Var("a"), b=Var("b")),
            FMul(a=Var("a"), b=Var("b")),
            FDiv(a=Var("a"), b=Var("b")),
            FSqrt(x=Var("a")),
            FNeg(x=Var("a")),
            FAbs(x=Var("a")),
            FEq(a=Var("a"), b=Var("b")),
            FLt(a=Var("a"), b=Var("b")),
            FLe(a=Var("a"), b=Var("b")),
            FNe(a=Var("a"), b=Var("b")),
        ]
        for e in exprs:
            self.assertTrue(_sat_matches(e, types, env))

    def test_simd_f32_add_mul_cmp_matches(self) -> None:
        t = SimdType(lane_width=32, lanes=8)
        m = SimdType(lane_width=1, lanes=8)
        types = {"x": t, "y": t}

        xs = [1.0, 2.0, -3.0, 4.5, 0.25, -0.5, 10.0, -1.25]
        ys = [0.5, -2.0, 3.0, -1.5, 1.25, 0.5, -2.0, 2.25]
        x_bits = 0
        y_bits = 0
        for i, (a, b) in enumerate(zip(xs, ys, strict=True)):
            x_bits |= _f32_bits(a) << (32 * i)
            y_bits |= _f32_bits(b) << (32 * i)

        env = {"x": x_bits, "y": y_bits}
        exprs = [
            SimdFAdd(a=Var("x"), b=Var("y")),
            SimdFSub(a=Var("x"), b=Var("y")),
            SimdFMul(a=Var("x"), b=Var("y")),
            SimdFDiv(a=Var("x"), b=Var("y")),
            SimdFFma(a=Var("x"), b=Var("y"), c=Var("x")),
            SimdFSqrt(x=SimdFAbs(x=Var("x"))),
            SimdFNeg(x=Var("x")),
            SimdFAbs(x=Var("x")),
            SimdFCmpEq(a=Var("x"), b=Var("y")),
            SimdFCmpLt(a=Var("x"), b=Var("y")),
            SimdFCmpLe(a=Var("x"), b=Var("y")),
            SimdFCmpNe(a=Var("x"), b=Var("y")),
        ]
        for e in exprs:
            self.assertTrue(_sat_matches(e, types, env))

    def test_simd_f64_fma_matches(self) -> None:
        t = SimdType(lane_width=64, lanes=4)
        types = {"x": t, "y": t, "z": t}

        xs = [1.0, -2.0, 0.5, 10.0]
        ys = [3.0, 0.25, -4.0, -0.125]
        zs = [0.0, 1.0, -1.0, 2.0]
        x_bits = 0
        y_bits = 0
        z_bits = 0
        for i, (a, b, c) in enumerate(zip(xs, ys, zs, strict=True)):
            x_bits |= _f64_bits(a) << (64 * i)
            y_bits |= _f64_bits(b) << (64 * i)
            z_bits |= _f64_bits(c) << (64 * i)

        env = {"x": x_bits, "y": y_bits, "z": z_bits}
        e = SimdFFma(a=Var("x"), b=Var("y"), c=Var("z"))
        self.assertTrue(_sat_matches(e, types, env))

    def test_float_const_roundtrip(self) -> None:
        t = FloatType(width=32)
        c = FloatConst(width=32, bits=_f32_bits(0.75))
        types = {}
        self.assertTrue(_sat_matches(c, types, {}))
