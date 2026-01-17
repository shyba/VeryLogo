from __future__ import annotations

from dataclasses import dataclass

import hashlib
import z3

from stc.interp import infer_type
from stc.tick_ir import (
    AShr,
    Add,
    And,
    BitVecConst,
    BitVecType,
    BoolConst,
    BoolType,
    Concat,
    Eq,
    Expr,
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
    LShr,
    Mux,
    Not,
    Or,
    Shl,
    Slice,
    Lut8,
    Bitcast,
    FloatConst,
    FloatType,
    SimdAdd,
    SimdAddMasked,
    SimdAddSatS,
    SimdAddSatU,
    SimdAShr,
    SimdAnd,
    SimdConst,
    SimdEq,
    SimdExtractLane,
    SimdInsertLane,
    SimdLShr,
    SimdMaddS16,
    SimdMaskExpand,
    SimdMaskPack,
    SimdMaxS,
    SimdMaxU,
    SimdMinS,
    SimdMinU,
    SimdBlend,
    SimdSExtLo,
    SimdMulHiS,
    SimdMulHiU,
    SimdMulLo,
    SimdNot,
    SimdOr,
    SimdPackSS16To8,
    SimdPackSS32To16,
    SimdPackUS16To8,
    SimdShl,
    SimdShuffle,
    SimdSplat,
    SimdSub,
    SimdSubMasked,
    SimdSubSatS,
    SimdSubSatU,
    SimdFAdd,
    SimdFAbs,
    SimdFCmpLe,
    SimdFCmpNe,
    SimdFCmpEq,
    SimdFCmpLt,
    SimdFDiv,
    SimdFFma,
    SimdFNeg,
    SimdFSqrt,
    SimdFSub,
    SimdFMul,
    SimdType,
    SimdUnpackHi,
    SimdUnpackLo,
    SimdZExtLo,
    SimdSge,
    SimdSgt,
    SimdSle,
    SimdSlt,
    SimdUge,
    SimdUgt,
    SimdUle,
    SimdUlt,
    SimdXor,
    Sub,
    Type,
    Uge,
    Ugt,
    Ule,
    Ult,
    Var,
    Xor,
)


@dataclass(frozen=True)
class Z3EncodeError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def _is_nan_bits(width: int, x: z3.BitVecRef) -> z3.BoolRef:
    if width == 32:
        exp = z3.Extract(30, 23, x)
        frac = z3.Extract(22, 0, x)
        return z3.And(exp == z3.BitVecVal(0xFF, 8), frac != z3.BitVecVal(0, 23))
    exp = z3.Extract(62, 52, x)
    frac = z3.Extract(51, 0, x)
    return z3.And(exp == z3.BitVecVal(0x7FF, 11), frac != z3.BitVecVal(0, 52))


def _canonical_nan_bits(width: int) -> int:
    if width == 32:
        return 0x7FC00000
    return 0x7FF8000000000000


def _canonicalize_nan(width: int, x: z3.BitVecRef) -> z3.BitVecRef:
    if x.size() != width:
        raise Z3EncodeError("float width mismatch")
    return z3.If(
        _is_nan_bits(width, x), z3.BitVecVal(_canonical_nan_bits(width), width), x
    )


def _bitwidth(t: Type) -> int:
    if isinstance(t, BitVecType):
        return t.width
    if isinstance(t, FloatType):
        return t.width
    if isinstance(t, SimdType):
        return t.total_width
    raise Z3EncodeError("expected bitvector-like type")


def _bv_to_width(x: z3.BitVecRef, width: int) -> z3.BitVecRef:
    if x.size() == width:
        return x
    if x.size() < width:
        return z3.ZeroExt(width - x.size(), x)
    return z3.Extract(width - 1, 0, x)


def _z3_var(name: str, t: Type) -> z3.ExprRef:
    if isinstance(t, BoolType):
        return z3.Bool(name)
    if isinstance(t, (BitVecType, FloatType, SimdType)):
        return z3.BitVec(name, _bitwidth(t))
    raise Z3EncodeError("unknown type")


def encode_expr(
    expr: Expr,
    types: dict[str, Type],
    vars: dict[str, z3.ExprRef] | None = None,
) -> z3.ExprRef:
    if vars is None:
        vars = {}

    if isinstance(expr, BoolConst):
        return z3.BoolVal(bool(expr.value))
    if isinstance(expr, BitVecConst):
        return z3.BitVecVal(int(expr.value), expr.width)
    if isinstance(expr, FloatConst):
        return z3.BitVecVal(int(expr.bits), expr.width)
    if isinstance(expr, SimdConst):
        return z3.BitVecVal(int(expr.value), expr.lane_width * expr.lanes)
    if isinstance(expr, Var):
        t = types.get(expr.name)
        if t is None:
            raise Z3EncodeError(f"unknown var {expr.name}")
        if expr.name not in vars:
            vars[expr.name] = _z3_var(expr.name, t)
        return vars[expr.name]

    if isinstance(expr, Bitcast):
        src = encode_expr(expr.x, types, vars)
        dst_t = expr.to
        if isinstance(dst_t, BoolType):
            raise Z3EncodeError("bitcast does not support bool")
        if not isinstance(src, z3.BitVecRef):
            raise Z3EncodeError("bitcast source must be bitvector")
        want = _bitwidth(dst_t)
        if src.size() != want:
            raise Z3EncodeError("bitcast width mismatch")
        return src

    if isinstance(
        expr, (FAdd, FMul, FSub, FDiv, FEq, FLt, FLe, FNe, FSqrt, FNeg, FAbs)
    ):
        t = infer_type(
            expr.x if isinstance(expr, (FSqrt, FNeg, FAbs)) else expr.a, types
        )
        if not isinstance(t, FloatType):
            raise Z3EncodeError("float op requires float operands")
        sort = z3.Float32() if t.width == 32 else z3.Float64()
        if isinstance(expr, (FSqrt, FNeg, FAbs)):
            x = encode_expr(expr.x, types, vars)
            if not isinstance(x, z3.BitVecRef) or x.size() != t.width:
                raise Z3EncodeError("float width mismatch")
            x = _canonicalize_nan(t.width, x)
            xf = z3.fpBVToFP(x, sort)
            if isinstance(expr, FSqrt):
                return _canonicalize_nan(
                    t.width, z3.fpToIEEEBV(z3.fpSqrt(z3.RNE(), xf))
                )
            if isinstance(expr, FNeg):
                return _canonicalize_nan(t.width, z3.fpToIEEEBV(z3.fpNeg(xf)))
            return _canonicalize_nan(t.width, z3.fpToIEEEBV(z3.fpAbs(xf)))

        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        if not isinstance(a, z3.BitVecRef) or not isinstance(b, z3.BitVecRef):
            raise Z3EncodeError("float op requires bitvector encoding")
        if a.size() != t.width or b.size() != t.width:
            raise Z3EncodeError("float width mismatch")

        a = _canonicalize_nan(t.width, a)
        b = _canonicalize_nan(t.width, b)
        af = z3.fpBVToFP(a, sort)
        bf = z3.fpBVToFP(b, sort)
        if isinstance(expr, FAdd):
            return _canonicalize_nan(t.width, z3.fpToIEEEBV(z3.fpAdd(z3.RNE(), af, bf)))
        if isinstance(expr, FSub):
            return _canonicalize_nan(t.width, z3.fpToIEEEBV(z3.fpSub(z3.RNE(), af, bf)))
        if isinstance(expr, FMul):
            return _canonicalize_nan(t.width, z3.fpToIEEEBV(z3.fpMul(z3.RNE(), af, bf)))
        if isinstance(expr, FDiv):
            return _canonicalize_nan(t.width, z3.fpToIEEEBV(z3.fpDiv(z3.RNE(), af, bf)))
        if isinstance(expr, FEq):
            return z3.fpEQ(af, bf)
        if isinstance(expr, FLt):
            return z3.fpLT(af, bf)
        if isinstance(expr, FLe):
            return z3.fpLEQ(af, bf)
        return z3.Not(z3.fpEQ(af, bf))

    if isinstance(expr, SimdSplat):
        to = expr.to
        if not isinstance(to, SimdType):
            raise Z3EncodeError("simd_splat target must be simd")
        x = encode_expr(expr.x, types, vars)
        if not isinstance(x, z3.BitVecRef) or x.size() != to.lane_width:
            raise Z3EncodeError("simd_splat source must be bitvector lane_width")
        out = x
        for _ in range(to.lanes - 1):
            out = z3.Concat(out, x)
        return out

    if isinstance(expr, SimdExtractLane):
        src_t = infer_type(expr.x, types)
        if not isinstance(src_t, SimdType):
            raise Z3EncodeError("simd_extract_lane requires simd source")
        if expr.lane >= src_t.lanes:
            raise Z3EncodeError("lane out of bounds")
        x = encode_expr(expr.x, types, vars)
        if not isinstance(x, z3.BitVecRef):
            raise Z3EncodeError("simd_extract_lane requires bitvector encoding")
        off = expr.lane * src_t.lane_width
        hi = off + src_t.lane_width - 1
        part = z3.Extract(hi, off, x)
        if src_t.lane_width == 1:
            return part == z3.BitVecVal(1, 1)
        return part

    if isinstance(expr, SimdInsertLane):
        src_t = infer_type(expr.x, types)
        if not isinstance(src_t, SimdType):
            raise Z3EncodeError("simd_insert_lane requires simd base")
        if expr.lane >= src_t.lanes:
            raise Z3EncodeError("lane out of bounds")
        x = encode_expr(expr.x, types, vars)
        v = encode_expr(expr.value, types, vars)
        if not isinstance(x, z3.BitVecRef):
            raise Z3EncodeError("simd_insert_lane requires bitvector base")
        if src_t.lane_width == 1:
            if isinstance(v, z3.BoolRef):
                v_bv = z3.If(v, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1))
            elif isinstance(v, z3.BitVecRef) and v.size() == 1:
                v_bv = v
            else:
                raise Z3EncodeError("insert value type mismatch")
        else:
            if not isinstance(v, z3.BitVecRef) or v.size() != src_t.lane_width:
                raise Z3EncodeError("insert value type mismatch")
            v_bv = v

        lanes: list[z3.ExprRef] = []
        for i in reversed(range(src_t.lanes)):
            off = i * src_t.lane_width
            hi = off + src_t.lane_width - 1
            if i == expr.lane:
                lanes.append(v_bv)
            else:
                lanes.append(z3.Extract(hi, off, x))
        out = lanes[0]
        for part in lanes[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, SimdShuffle):
        src_t = infer_type(expr.x, types)
        if not isinstance(src_t, SimdType):
            raise Z3EncodeError("simd_shuffle requires simd source")
        if not expr.indices:
            raise Z3EncodeError("indices must be non-empty")
        x = encode_expr(expr.x, types, vars)
        if not isinstance(x, z3.BitVecRef):
            raise Z3EncodeError("simd_shuffle requires bitvector encoding")
        out_parts: list[z3.ExprRef] = []
        for out_lane in reversed(range(len(expr.indices))):
            src_lane = int(expr.indices[out_lane])
            if src_lane < 0 or src_lane >= src_t.lanes:
                raise Z3EncodeError("shuffle index out of bounds")
            off = src_lane * src_t.lane_width
            hi = off + src_t.lane_width - 1
            out_parts.append(z3.Extract(hi, off, x))
        out = out_parts[0]
        for part in out_parts[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, Not):
        x = encode_expr(expr.x, types, vars)
        t = infer_type(expr.x, types)
        if isinstance(t, BoolType):
            return z3.Not(x)
        return ~x

    if isinstance(
        expr, (And, Or, Xor, Add, Sub, Shl, LShr, AShr, Eq, Ult, Ule, Ugt, Uge)
    ):
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        t = infer_type(expr.a, types)

        if isinstance(expr, (Eq, Ult, Ule, Ugt, Uge)):
            if not isinstance(t, BitVecType) and not isinstance(expr, Eq):
                raise Z3EncodeError("unsigned compares require bitvec operands")
            if isinstance(expr, Ult):
                return z3.ULT(a, b)
            if isinstance(expr, Ule):
                return z3.ULE(a, b)
            if isinstance(expr, Ugt):
                return z3.UGT(a, b)
            if isinstance(expr, Uge):
                return z3.UGE(a, b)
            return a == b

        if isinstance(t, BoolType):
            if isinstance(expr, And):
                return z3.And(a, b)
            if isinstance(expr, Or):
                return z3.Or(a, b)
            if isinstance(expr, Xor):
                return z3.Xor(a, b)
            raise Z3EncodeError("add requires bitvec operands")

        if isinstance(expr, Add):
            if not isinstance(t, BitVecType):
                raise Z3EncodeError("add requires bitvec operands")
            return a + b
        if isinstance(expr, Sub):
            if not isinstance(t, BitVecType):
                raise Z3EncodeError("sub requires bitvec operands")
            return a - b
        if isinstance(expr, Shl):
            if not isinstance(t, BitVecType):
                raise Z3EncodeError("shl requires bitvec operands")
            return a << b
        if isinstance(expr, LShr):
            if not isinstance(t, BitVecType):
                raise Z3EncodeError("lshr requires bitvec operands")
            return z3.LShR(a, b)
        if isinstance(expr, AShr):
            if not isinstance(t, BitVecType):
                raise Z3EncodeError("ashr requires bitvec operands")
            return a >> b

        if isinstance(expr, And):
            return a & b
        if isinstance(expr, Or):
            return a | b
        if isinstance(expr, Xor):
            return a ^ b
        raise Z3EncodeError("unsupported binary op")

    if isinstance(expr, SimdAdd):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise Z3EncodeError("simd_add requires simd operands")
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        lanes: list[z3.ExprRef] = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            hi = off + t.lane_width - 1
            la = z3.Extract(hi, off, a)
            lb = z3.Extract(hi, off, b)
            lanes.append(la + lb)
        if not lanes:
            raise Z3EncodeError("simd lanes must be >= 1")
        out = lanes[0]
        for part in lanes[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, SimdAddMasked):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise Z3EncodeError("simd_add_masked requires simd operands")
        mask_t = infer_type(expr.mask, types)
        if not isinstance(mask_t, SimdType) or mask_t.lane_width != 1:
            raise Z3EncodeError("simd_add_masked requires simd mask")
        if mask_t.lanes != t.lanes:
            raise Z3EncodeError("simd_add_masked mask lane mismatch")
        mask = encode_expr(expr.mask, types, vars)
        p = encode_expr(expr.passthru, types, vars)
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        if not isinstance(mask, z3.BitVecRef) or mask.size() != mask_t.lanes:
            raise Z3EncodeError("simd_add_masked mask encoding mismatch")
        lanes: list[z3.ExprRef] = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            hi = off + t.lane_width - 1
            lp = z3.Extract(hi, off, p)
            la = z3.Extract(hi, off, a)
            lb = z3.Extract(hi, off, b)
            bit = z3.Extract(i, i, mask)
            lanes.append(
                z3.If(bit == z3.BitVecVal(1, 1), la + lb, lp),
            )
        if not lanes:
            raise Z3EncodeError("simd lanes must be >= 1")
        out = lanes[0]
        for part in lanes[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, (SimdFAdd, SimdFMul, SimdFSub, SimdFDiv)):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise Z3EncodeError("simd float op requires simd operands")
        if t.lane_width not in {32, 64}:
            raise Z3EncodeError("simd float op requires lane_width in {32,64}")
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        if not isinstance(a, z3.BitVecRef) or not isinstance(b, z3.BitVecRef):
            raise Z3EncodeError("simd float op requires bitvector encoding")
        sort = z3.Float32() if t.lane_width == 32 else z3.Float64()
        lanes: list[z3.ExprRef] = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            hi = off + t.lane_width - 1
            la = _canonicalize_nan(t.lane_width, z3.Extract(hi, off, a))
            lb = _canonicalize_nan(t.lane_width, z3.Extract(hi, off, b))
            af = z3.fpBVToFP(la, sort)
            bf = z3.fpBVToFP(lb, sort)
            if isinstance(expr, SimdFAdd):
                rf = z3.fpAdd(z3.RNE(), af, bf)
            elif isinstance(expr, SimdFSub):
                rf = z3.fpSub(z3.RNE(), af, bf)
            elif isinstance(expr, SimdFMul):
                rf = z3.fpMul(z3.RNE(), af, bf)
            else:
                rf = z3.fpDiv(z3.RNE(), af, bf)
            lanes.append(_canonicalize_nan(t.lane_width, z3.fpToIEEEBV(rf)))
        if not lanes:
            raise Z3EncodeError("simd lanes must be >= 1")
        out = lanes[0]
        for part in lanes[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, SimdFFma):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise Z3EncodeError("simd fma requires simd operands")
        if t.lane_width not in {32, 64}:
            raise Z3EncodeError("simd fma requires lane_width in {32,64}")
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        c = encode_expr(expr.c, types, vars)
        if (
            not isinstance(a, z3.BitVecRef)
            or not isinstance(b, z3.BitVecRef)
            or not isinstance(c, z3.BitVecRef)
        ):
            raise Z3EncodeError("simd fma requires bitvector encoding")
        sort = z3.Float32() if t.lane_width == 32 else z3.Float64()
        lanes: list[z3.ExprRef] = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            hi = off + t.lane_width - 1
            la = _canonicalize_nan(t.lane_width, z3.Extract(hi, off, a))
            lb = _canonicalize_nan(t.lane_width, z3.Extract(hi, off, b))
            lc = _canonicalize_nan(t.lane_width, z3.Extract(hi, off, c))
            af = z3.fpBVToFP(la, sort)
            bf = z3.fpBVToFP(lb, sort)
            cf = z3.fpBVToFP(lc, sort)
            rf = z3.fpFMA(z3.RNE(), af, bf, cf)
            lanes.append(_canonicalize_nan(t.lane_width, z3.fpToIEEEBV(rf)))
        if not lanes:
            raise Z3EncodeError("simd lanes must be >= 1")
        out = lanes[0]
        for part in lanes[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, (SimdFSqrt, SimdFNeg, SimdFAbs)):
        t = infer_type(expr.x, types)
        if not isinstance(t, SimdType):
            raise Z3EncodeError("simd float op requires simd operand")
        if t.lane_width not in {32, 64}:
            raise Z3EncodeError("simd float op requires lane_width in {32,64}")
        x = encode_expr(expr.x, types, vars)
        if not isinstance(x, z3.BitVecRef):
            raise Z3EncodeError("simd float op requires bitvector encoding")
        sort = z3.Float32() if t.lane_width == 32 else z3.Float64()
        lanes: list[z3.ExprRef] = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            hi = off + t.lane_width - 1
            part = _canonicalize_nan(t.lane_width, z3.Extract(hi, off, x))
            xf = z3.fpBVToFP(part, sort)
            if isinstance(expr, SimdFSqrt):
                rf = z3.fpSqrt(z3.RNE(), xf)
            elif isinstance(expr, SimdFNeg):
                rf = z3.fpNeg(xf)
            else:
                rf = z3.fpAbs(xf)
            lanes.append(_canonicalize_nan(t.lane_width, z3.fpToIEEEBV(rf)))
        if not lanes:
            raise Z3EncodeError("simd lanes must be >= 1")
        out = lanes[0]
        for part in lanes[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, (SimdFCmpEq, SimdFCmpLt, SimdFCmpLe, SimdFCmpNe)):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise Z3EncodeError("simd float cmp requires simd operands")
        if t.lane_width not in {32, 64}:
            raise Z3EncodeError("simd float cmp requires lane_width in {32,64}")
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        if not isinstance(a, z3.BitVecRef) or not isinstance(b, z3.BitVecRef):
            raise Z3EncodeError("simd float cmp requires bitvector encoding")
        sort = z3.Float32() if t.lane_width == 32 else z3.Float64()
        bits: list[z3.ExprRef] = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            hi = off + t.lane_width - 1
            la = _canonicalize_nan(t.lane_width, z3.Extract(hi, off, a))
            lb = _canonicalize_nan(t.lane_width, z3.Extract(hi, off, b))
            af = z3.fpBVToFP(la, sort)
            bf = z3.fpBVToFP(lb, sort)
            if isinstance(expr, SimdFCmpEq):
                ok = z3.fpEQ(af, bf)
            elif isinstance(expr, SimdFCmpLt):
                ok = z3.fpLT(af, bf)
            elif isinstance(expr, SimdFCmpLe):
                ok = z3.fpLEQ(af, bf)
            else:
                ok = z3.Not(z3.fpEQ(af, bf))
            bits.append(z3.If(ok, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)))
        if not bits:
            raise Z3EncodeError("simd lanes must be >= 1")
        out = bits[0]
        for part in bits[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, SimdSub):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise Z3EncodeError("simd_sub requires simd operands")
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        lanes: list[z3.ExprRef] = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            hi = off + t.lane_width - 1
            la = z3.Extract(hi, off, a)
            lb = z3.Extract(hi, off, b)
            lanes.append(la - lb)
        if not lanes:
            raise Z3EncodeError("simd lanes must be >= 1")
        out = lanes[0]
        for part in lanes[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, SimdSubMasked):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise Z3EncodeError("simd_sub_masked requires simd operands")
        mask_t = infer_type(expr.mask, types)
        if not isinstance(mask_t, SimdType) or mask_t.lane_width != 1:
            raise Z3EncodeError("simd_sub_masked requires simd mask")
        if mask_t.lanes != t.lanes:
            raise Z3EncodeError("simd_sub_masked mask lane mismatch")
        mask = encode_expr(expr.mask, types, vars)
        p = encode_expr(expr.passthru, types, vars)
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        if not isinstance(mask, z3.BitVecRef) or mask.size() != mask_t.lanes:
            raise Z3EncodeError("simd_sub_masked mask encoding mismatch")
        lanes: list[z3.ExprRef] = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            hi = off + t.lane_width - 1
            lp = z3.Extract(hi, off, p)
            la = z3.Extract(hi, off, a)
            lb = z3.Extract(hi, off, b)
            bit = z3.Extract(i, i, mask)
            lanes.append(
                z3.If(bit == z3.BitVecVal(1, 1), la - lb, lp),
            )
        if not lanes:
            raise Z3EncodeError("simd lanes must be >= 1")
        out = lanes[0]
        for part in lanes[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, SimdAddSatU):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise Z3EncodeError("simd_add_satu requires simd operands")
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        lanes: list[z3.ExprRef] = []
        lane_max = (1 << t.lane_width) - 1
        maxv = z3.BitVecVal(lane_max, t.lane_width)
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            hi = off + t.lane_width - 1
            la = z3.Extract(hi, off, a)
            lb = z3.Extract(hi, off, b)
            sum_ext = z3.ZeroExt(1, la) + z3.ZeroExt(1, lb)
            carry = z3.Extract(t.lane_width, t.lane_width, sum_ext)
            low = z3.Extract(t.lane_width - 1, 0, sum_ext)
            lanes.append(z3.If(carry == z3.BitVecVal(1, 1), maxv, low))
        if not lanes:
            raise Z3EncodeError("simd lanes must be >= 1")
        out = lanes[0]
        for part in lanes[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, SimdSubSatU):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise Z3EncodeError("simd_sub_satu requires simd operands")
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        lanes: list[z3.ExprRef] = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            hi = off + t.lane_width - 1
            la = z3.Extract(hi, off, a)
            lb = z3.Extract(hi, off, b)
            lanes.append(
                z3.If(
                    z3.ULT(la, lb),
                    z3.BitVecVal(0, t.lane_width),
                    la - lb,
                )
            )
        if not lanes:
            raise Z3EncodeError("simd lanes must be >= 1")
        out = lanes[0]
        for part in lanes[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, (SimdAddSatS, SimdSubSatS)):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise Z3EncodeError("simd saturating signed op requires simd operands")
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        lanes: list[z3.ExprRef] = []
        w = t.lane_width
        max_val = (1 << (w - 1)) - 1
        min_ext_val = (1 << (w + 1)) - (1 << (w - 1))
        max_ext = z3.BitVecVal(max_val, w + 1)
        min_ext = z3.BitVecVal(min_ext_val, w + 1)
        for i in reversed(range(t.lanes)):
            off = i * w
            hi = off + w - 1
            la = z3.Extract(hi, off, a)
            lb = z3.Extract(hi, off, b)
            la_ext = z3.SignExt(1, la)
            lb_ext = z3.SignExt(1, lb)
            s = la_ext + lb_ext if isinstance(expr, SimdAddSatS) else la_ext - lb_ext
            clamped = z3.If(s > max_ext, max_ext, z3.If(s < min_ext, min_ext, s))
            lanes.append(z3.Extract(w - 1, 0, clamped))
        if not lanes:
            raise Z3EncodeError("simd lanes must be >= 1")
        out = lanes[0]
        for part in lanes[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, (SimdMulLo, SimdMulHiU, SimdMulHiS)):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise Z3EncodeError("simd mul requires simd operands")
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        lanes: list[z3.ExprRef] = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            hi = off + t.lane_width - 1
            la = z3.Extract(hi, off, a)
            lb = z3.Extract(hi, off, b)
            if isinstance(expr, SimdMulLo):
                lanes.append(la * lb)
            elif isinstance(expr, SimdMulHiU):
                prod = z3.ZeroExt(t.lane_width, la) * z3.ZeroExt(t.lane_width, lb)
                lanes.append(z3.Extract((2 * t.lane_width) - 1, t.lane_width, prod))
            else:
                prod = z3.SignExt(t.lane_width, la) * z3.SignExt(t.lane_width, lb)
                lanes.append(z3.Extract((2 * t.lane_width) - 1, t.lane_width, prod))
        if not lanes:
            raise Z3EncodeError("simd lanes must be >= 1")
        out = lanes[0]
        for part in lanes[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, SimdMaddS16):
        out_t = infer_type(expr, types)
        if not isinstance(out_t, SimdType):
            raise Z3EncodeError("simd_madd_s16 requires simd operands")
        a_t = infer_type(expr.a, types)
        if not isinstance(a_t, SimdType):
            raise Z3EncodeError("simd_madd_s16 requires simd operands")
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        lanes: list[z3.ExprRef] = []
        for i in reversed(range(out_t.lanes)):
            off0 = 16 * (2 * i)
            off1 = 16 * (2 * i + 1)
            a0 = z3.Extract(off0 + 15, off0, a)
            a1 = z3.Extract(off1 + 15, off1, a)
            b0 = z3.Extract(off0 + 15, off0, b)
            b1 = z3.Extract(off1 + 15, off1, b)
            a0e = z3.SignExt(16, a0)
            a1e = z3.SignExt(16, a1)
            b0e = z3.SignExt(16, b0)
            b1e = z3.SignExt(16, b1)
            lanes.append((a0e * b0e) + (a1e * b1e))
        if not lanes:
            raise Z3EncodeError("simd lanes must be >= 1")
        out = lanes[0]
        for part in lanes[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, (SimdUnpackLo, SimdUnpackHi)):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise Z3EncodeError("simd unpack requires simd operands")
        if t.lanes % 2 != 0:
            raise Z3EncodeError("simd unpack requires even lanes")
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        half = t.lanes // 2
        out_lanes: list[z3.ExprRef] = []
        for i in reversed(range(half)):
            src = i if isinstance(expr, SimdUnpackLo) else (i + half)
            off = src * t.lane_width
            hi = off + t.lane_width - 1
            la = z3.Extract(hi, off, a)
            lb = z3.Extract(hi, off, b)
            out_lanes.append(lb)
            out_lanes.append(la)
        if not out_lanes:
            raise Z3EncodeError("simd lanes must be >= 1")
        out = out_lanes[0]
        for part in out_lanes[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, (SimdPackSS16To8, SimdPackUS16To8)):
        a_t = infer_type(expr.a, types)
        if not isinstance(a_t, SimdType):
            raise Z3EncodeError("simd pack requires simd operands")
        if a_t.lane_width != 16:
            raise Z3EncodeError("simd_pack_*16_to_8 requires lane_width=16")
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        out_lanes: list[z3.ExprRef] = []
        for src_vec in [b, a]:
            for i in reversed(range(a_t.lanes)):
                off = i * 16
                lane = z3.Extract(off + 15, off, src_vec)
                v = z3.SignExt(16, lane)
                if isinstance(expr, SimdPackUS16To8):
                    clamped = z3.If(
                        v < z3.BitVecVal(0, 32),
                        z3.BitVecVal(0, 32),
                        z3.If(
                            v > z3.BitVecVal(255, 32),
                            z3.BitVecVal(255, 32),
                            v,
                        ),
                    )
                else:
                    minv = z3.BitVecVal((1 << 32) - 128, 32)
                    maxv = z3.BitVecVal(127, 32)
                    clamped = z3.If(v > maxv, maxv, z3.If(v < minv, minv, v))
                out_lanes.append(z3.Extract(7, 0, clamped))
        out = out_lanes[0]
        for part in out_lanes[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, SimdPackSS32To16):
        a_t = infer_type(expr.a, types)
        if not isinstance(a_t, SimdType):
            raise Z3EncodeError("simd pack requires simd operands")
        if a_t.lane_width != 32:
            raise Z3EncodeError("simd_pack_ss32_to_16 requires lane_width=32")
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        out_lanes: list[z3.ExprRef] = []
        for src_vec in [b, a]:
            for i in reversed(range(a_t.lanes)):
                off = i * 32
                lane = z3.Extract(off + 31, off, src_vec)
                v = lane
                minv = z3.BitVecVal((1 << 32) - 32768, 32)
                maxv = z3.BitVecVal(32767, 32)
                clamped = z3.If(v > maxv, maxv, z3.If(v < minv, minv, v))
                out_lanes.append(z3.Extract(15, 0, clamped))
        out = out_lanes[0]
        for part in out_lanes[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, SimdMaskExpand):
        if not isinstance(expr.to, SimdType):
            raise Z3EncodeError("simd_mask_expand target must be simd")
        src_t = infer_type(expr.x, types)
        if not (isinstance(src_t, SimdType) and src_t.lane_width == 1):
            raise Z3EncodeError("simd_mask_expand source must be simd mask")
        if src_t.lanes != expr.to.lanes:
            raise Z3EncodeError("simd_mask_expand lane mismatch")
        x = encode_expr(expr.x, types, vars)
        if not isinstance(x, z3.BitVecRef) or x.size() != src_t.lanes:
            raise Z3EncodeError("simd_mask_expand requires bitvector encoding")
        w = expr.to.lane_width
        ones = z3.BitVecVal((1 << w) - 1, w)
        zeros = z3.BitVecVal(0, w)
        lanes: list[z3.ExprRef] = []
        for i in reversed(range(expr.to.lanes)):
            bit = z3.Extract(i, i, x)
            lanes.append(z3.If(bit == z3.BitVecVal(1, 1), ones, zeros))
        out = lanes[0]
        for part in lanes[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, SimdMaskPack):
        src_t = infer_type(expr.x, types)
        if not isinstance(src_t, SimdType):
            raise Z3EncodeError("simd_mask_pack source must be simd")
        x = encode_expr(expr.x, types, vars)
        if not isinstance(x, z3.BitVecRef):
            raise Z3EncodeError("simd_mask_pack requires bitvector encoding")
        bits: list[z3.ExprRef] = []
        for i in reversed(range(src_t.lanes)):
            off = i * src_t.lane_width
            hi = off + src_t.lane_width - 1
            lane = z3.Extract(hi, off, x)
            bits.append(
                z3.If(
                    lane != z3.BitVecVal(0, src_t.lane_width),
                    z3.BitVecVal(1, 1),
                    z3.BitVecVal(0, 1),
                )
            )
        out = bits[0]
        for part in bits[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, (SimdMinU, SimdMaxU, SimdMinS, SimdMaxS)):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise Z3EncodeError("simd min/max requires simd operands")
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        lanes: list[z3.ExprRef] = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            hi = off + t.lane_width - 1
            la = z3.Extract(hi, off, a)
            lb = z3.Extract(hi, off, b)
            if isinstance(expr, SimdMinU):
                pred = z3.ULT(la, lb)
            elif isinstance(expr, SimdMaxU):
                pred = z3.UGT(la, lb)
            elif isinstance(expr, SimdMinS):
                pred = la < lb
            else:
                pred = la > lb
            lanes.append(z3.If(pred, la, lb))
        out = lanes[0]
        for part in lanes[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, SimdBlend):
        m_t = infer_type(expr.mask, types)
        if not isinstance(m_t, SimdType) or m_t.lane_width != 1:
            raise Z3EncodeError("simd_blend mask must be simd mask")
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise Z3EncodeError("simd_blend requires simd operands")
        mask = encode_expr(expr.mask, types, vars)
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        if not isinstance(mask, z3.BitVecRef) or mask.size() != m_t.lanes:
            raise Z3EncodeError("simd_blend mask encoding mismatch")
        lanes: list[z3.ExprRef] = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            hi = off + t.lane_width - 1
            la = z3.Extract(hi, off, a)
            lb = z3.Extract(hi, off, b)
            bit = z3.Extract(i, i, mask)
            lanes.append(z3.If(bit == z3.BitVecVal(1, 1), lb, la))
        out = lanes[0]
        for part in lanes[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, (SimdZExtLo, SimdSExtLo)):
        if not isinstance(expr.to, SimdType):
            raise Z3EncodeError("simd extend target must be simd")
        src_t = infer_type(expr.x, types)
        if not isinstance(src_t, SimdType):
            raise Z3EncodeError("simd extend source must be simd")
        dst_t = expr.to
        if dst_t.total_width != src_t.total_width:
            raise Z3EncodeError("simd extend total_width mismatch")
        if src_t.lanes % 2 != 0 or dst_t.lanes != src_t.lanes // 2:
            raise Z3EncodeError("simd extend requires lanes halved")
        if dst_t.lane_width != src_t.lane_width * 2:
            raise Z3EncodeError("simd extend requires lane_width doubled")
        x = encode_expr(expr.x, types, vars)
        if not isinstance(x, z3.BitVecRef):
            raise Z3EncodeError("simd extend requires bitvector encoding")
        lanes: list[z3.ExprRef] = []
        for i in reversed(range(dst_t.lanes)):
            off = i * src_t.lane_width
            hi = off + src_t.lane_width - 1
            lane = z3.Extract(hi, off, x)
            if isinstance(expr, SimdZExtLo):
                lanes.append(z3.ZeroExt(dst_t.lane_width - src_t.lane_width, lane))
            else:
                lanes.append(z3.SignExt(dst_t.lane_width - src_t.lane_width, lane))
        out = lanes[0]
        for part in lanes[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, SimdEq):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise Z3EncodeError("simd_eq requires simd operands")
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        bits: list[z3.ExprRef] = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            hi = off + t.lane_width - 1
            la = z3.Extract(hi, off, a)
            lb = z3.Extract(hi, off, b)
            bits.append(z3.If(la == lb, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)))
        if not bits:
            raise Z3EncodeError("simd lanes must be >= 1")
        out = bits[0]
        for part in bits[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, SimdNot):
        t = infer_type(expr.x, types)
        if not isinstance(t, SimdType):
            raise Z3EncodeError("simd_not requires simd operand")
        x = encode_expr(expr.x, types, vars)
        if not isinstance(x, z3.BitVecRef):
            raise Z3EncodeError("simd_not requires bitvector encoding")
        return ~x

    if isinstance(expr, (SimdAnd, SimdOr, SimdXor)):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise Z3EncodeError("simd bitwise requires simd operands")
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        if not isinstance(a, z3.BitVecRef) or not isinstance(b, z3.BitVecRef):
            raise Z3EncodeError("simd bitwise requires bitvector encoding")
        if isinstance(expr, SimdAnd):
            return a & b
        if isinstance(expr, SimdOr):
            return a | b
        return a ^ b

    if isinstance(expr, (SimdShl, SimdLShr, SimdAShr)):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise Z3EncodeError("simd shift requires simd lhs")
        a = encode_expr(expr.a, types, vars)
        sh = encode_expr(expr.sh, types, vars)
        if not isinstance(a, z3.BitVecRef) or not isinstance(sh, z3.BitVecRef):
            raise Z3EncodeError("simd shift expects bitvectors")
        shw = _bv_to_width(sh, t.lane_width)
        lanes: list[z3.ExprRef] = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            hi = off + t.lane_width - 1
            la = z3.Extract(hi, off, a)
            if isinstance(expr, SimdShl):
                lanes.append(la << shw)
            elif isinstance(expr, SimdLShr):
                lanes.append(z3.LShR(la, shw))
            else:
                lanes.append(la >> shw)
        if not lanes:
            raise Z3EncodeError("simd lanes must be >= 1")
        out = lanes[0]
        for part in lanes[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, (SimdUlt, SimdUle, SimdUgt, SimdUge)):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise Z3EncodeError("simd unsigned compare requires simd operands")
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        bits: list[z3.ExprRef] = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            hi = off + t.lane_width - 1
            la = z3.Extract(hi, off, a)
            lb = z3.Extract(hi, off, b)
            if isinstance(expr, SimdUlt):
                pred = z3.ULT(la, lb)
            elif isinstance(expr, SimdUle):
                pred = z3.ULE(la, lb)
            elif isinstance(expr, SimdUgt):
                pred = z3.UGT(la, lb)
            else:
                pred = z3.UGE(la, lb)
            bits.append(z3.If(pred, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)))
        if not bits:
            raise Z3EncodeError("simd lanes must be >= 1")
        out = bits[0]
        for part in bits[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, (SimdSlt, SimdSle, SimdSgt, SimdSge)):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise Z3EncodeError("simd signed compare requires simd operands")
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        bits: list[z3.ExprRef] = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            hi = off + t.lane_width - 1
            la = z3.Extract(hi, off, a)
            lb = z3.Extract(hi, off, b)
            if isinstance(expr, SimdSlt):
                pred = la < lb
            elif isinstance(expr, SimdSle):
                pred = la <= lb
            elif isinstance(expr, SimdSgt):
                pred = la > lb
            else:
                pred = la >= lb
            bits.append(z3.If(pred, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)))
        if not bits:
            raise Z3EncodeError("simd lanes must be >= 1")
        out = bits[0]
        for part in bits[1:]:
            out = z3.Concat(out, part)
        return out

    if isinstance(expr, Mux):
        cond = encode_expr(expr.cond, types, vars)
        a = encode_expr(expr.a, types, vars)
        b = encode_expr(expr.b, types, vars)
        return z3.If(cond, a, b)

    if isinstance(expr, Lut8):
        x = encode_expr(expr.x, types, vars)
        if not isinstance(x, z3.BitVecRef):
            raise Z3EncodeError("lut8 input must be bitvector")
        idx = _bv_to_width(x, 8)

        digest = hashlib.sha256(bytes(int(v) & 0xFF for v in expr.table)).hexdigest()
        name = f"__lut8_{digest[:16]}"
        arr = vars.get(name)
        if arr is None:
            key_sort = z3.BitVecSort(8)
            val_sort = z3.BitVecSort(8)
            out: z3.ArrayRef = z3.K(key_sort, z3.BitVecVal(0, 8))
            for i, v in enumerate(expr.table):
                out = z3.Store(out, z3.BitVecVal(i, 8), z3.BitVecVal(int(v) & 0xFF, 8))
            vars[name] = out
            arr = out
        if not isinstance(arr, z3.ArrayRef):
            raise Z3EncodeError("lut8 internal cache collision")
        sel = z3.Select(arr, idx)
        if not isinstance(sel, z3.BitVecRef) or sel.size() != 8:
            raise Z3EncodeError("lut8 select type mismatch")
        return sel

    if isinstance(expr, Slice):
        x = encode_expr(expr.x, types, vars)
        src_t = infer_type(expr.x, types)
        if isinstance(src_t, BoolType):
            raise Z3EncodeError("slice source must be bitvector")
        if isinstance(src_t, BitVecType):
            total = src_t.width
        else:
            assert isinstance(src_t, SimdType)
            total = src_t.total_width
        if expr.offset + expr.width > total:
            raise Z3EncodeError("slice out of bounds")
        if not isinstance(x, z3.BitVecRef):
            raise Z3EncodeError("slice source must be bitvector")
        hi = expr.offset + expr.width - 1
        part = z3.Extract(hi, expr.offset, x)
        if expr.width == 1:
            return part == z3.BitVecVal(1, 1)
        return part

    if isinstance(expr, Concat):
        parts: list[z3.ExprRef] = []
        for p in expr.parts:
            t = infer_type(p, types)
            z = encode_expr(p, types, vars)
            if isinstance(t, BoolType):
                parts.append(
                    z3.If(z, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1))
                    if isinstance(z, z3.BoolRef)
                    else z
                )
            else:
                if not isinstance(z, z3.BitVecRef):
                    raise Z3EncodeError("concat bitvector part expected")
                parts.append(z)
        if not parts:
            raise Z3EncodeError("empty concat is not supported")
        out = parts[0]
        for p in parts[1:]:
            out = z3.Concat(out, p)
        return out

    raise Z3EncodeError("unsupported expression kind")
