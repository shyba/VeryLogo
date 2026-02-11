from __future__ import annotations

import math
import struct
import ctypes
import ctypes.util
from dataclasses import dataclass
from typing import Any

from stc.tick_ir import (
    AShr,
    Add,
    And,
    Bitcast,
    BitTranspose,
    BitVecConst,
    BitVecType,
    BoolConst,
    BoolType,
    Concat,
    Div,
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
    Mul,
    Mux,
    Not,
    Or,
    Shl,
    Slice,
    Lut8,
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
    TickIR,
    Type,
    Uge,
    Ugt,
    Ule,
    Ult,
    Var,
    Xor,
)
from stc.tick_ir_validate import TickIRValidationError, type_equal, validate_tick_ir


@dataclass(frozen=True)
class TickState:
    state: dict[str, int | bool]


def _mask(width: int) -> int:
    return (1 << width) - 1


def _f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(bits) & 0xFFFFFFFF))[0]


def _f32_to_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _f64_from_bits(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(bits) & 0xFFFFFFFFFFFFFFFF))[0]


def _f64_to_bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


_libm = None
_libm_name = ctypes.util.find_library("m")
if _libm_name is not None:
    _libm = ctypes.CDLL(_libm_name)
    if hasattr(_libm, "fma"):
        _libm.fma.restype = ctypes.c_double
        _libm.fma.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]


def _f64_fma(a: float, b: float, c: float) -> float:
    if hasattr(math, "fma"):
        return float(math.fma(a, b, c))
    if _libm is not None and hasattr(_libm, "fma"):
        return float(_libm.fma(a, b, c))
    raise TickIRValidationError("f64 fma is not available in this runtime")


def _canonical_nan_bits(width: int) -> int:
    if width == 32:
        return 0x7FC00000
    return 0x7FF8000000000000


def _is_nan_bits(width: int, bits: int) -> bool:
    if width == 32:
        exp = (int(bits) >> 23) & 0xFF
        frac = int(bits) & 0x7FFFFF
        return exp == 0xFF and frac != 0
    exp = (int(bits) >> 52) & 0x7FF
    frac = int(bits) & 0xFFFFFFFFFFFFF
    return exp == 0x7FF and frac != 0


def _canonicalize_nan_bits(width: int, bits: int) -> int:
    if _is_nan_bits(width, bits):
        return _canonical_nan_bits(width)
    return int(bits)

_INFER_TYPE_CACHE: dict[int, dict[int, Type]] = {}


def infer_type(expr: Expr, ctx: dict[str, Type]) -> Type:
    ctx_key = id(ctx)
    cache = _INFER_TYPE_CACHE.get(ctx_key)
    if cache is None:
        cache = {}
        _INFER_TYPE_CACHE[ctx_key] = cache
    cache_key = id(expr)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    t = _infer_type_no_cache(expr, ctx)
    cache[cache_key] = t
    return t


def _infer_type_no_cache(expr: Expr, ctx: dict[str, Type]) -> Type:
    if isinstance(expr, BoolConst):
        return BoolType()
    if isinstance(expr, BitVecConst):
        return BitVecType(width=expr.width)
    if isinstance(expr, FloatConst):
        return FloatType(width=expr.width)
    if isinstance(expr, SimdConst):
        return SimdType(lane_width=expr.lane_width, lanes=expr.lanes)
    if isinstance(expr, Var):
        t = ctx.get(expr.name)
        if t is None:
            raise TickIRValidationError(f"unknown var {expr.name}")
        return t

    if isinstance(expr, Bitcast):
        src_t = infer_type(expr.x, ctx)
        dst_t = expr.to
        if isinstance(src_t, BoolType) or isinstance(dst_t, BoolType):
            raise TickIRValidationError("bitcast does not support bool")
        if isinstance(src_t, BitVecType):
            src_w = src_t.width
        elif isinstance(src_t, FloatType):
            src_w = src_t.width
        else:
            assert isinstance(src_t, SimdType)
            src_w = src_t.total_width
        if isinstance(dst_t, BitVecType):
            dst_w = dst_t.width
        elif isinstance(dst_t, FloatType):
            dst_w = dst_t.width
        else:
            assert isinstance(dst_t, SimdType)
            dst_w = dst_t.total_width
        if src_w != dst_w:
            raise TickIRValidationError("bitcast width mismatch")
        return dst_t

    from stc.tick_ir import Rotl, Rotr

    if isinstance(expr, (Rotl, Rotr)):
        x_t = infer_type(expr.x, ctx)
        sh_t = infer_type(expr.sh, ctx)
        if not isinstance(x_t, BitVecType):
            raise TickIRValidationError("rotate requires bitvec operand")
        if not isinstance(sh_t, BitVecType):
            raise TickIRValidationError("rotate shift amount must be bitvec")
        return x_t

    if isinstance(expr, BitTranspose):
        src_t = infer_type(expr.x, ctx)
        if not isinstance(src_t, SimdType):
            raise TickIRValidationError("bit_transpose input must be simd")
        if src_t.lane_width != expr.lane_width or src_t.lanes != expr.lanes:
            raise TickIRValidationError(
                f"bit_transpose layout mismatch: input is simd[{src_t.lane_width},{src_t.lanes}], "
                f"expected simd[{expr.lane_width},{expr.lanes}]"
            )
        return SimdType(lane_width=expr.lanes, lanes=expr.lane_width)

    if isinstance(expr, SimdSplat):
        dst_t = expr.to
        if not isinstance(dst_t, SimdType):
            raise TickIRValidationError("simd_splat target must be simd")
        src_t = infer_type(expr.x, ctx)
        if not (isinstance(src_t, BitVecType) and src_t.width == dst_t.lane_width):
            raise TickIRValidationError("simd_splat source width must match lane_width")
        return dst_t

    if isinstance(expr, SimdExtractLane):
        src_t = infer_type(expr.x, ctx)
        if not isinstance(src_t, SimdType):
            raise TickIRValidationError("simd_extract_lane requires simd source")
        if expr.lane >= src_t.lanes:
            raise TickIRValidationError("lane out of bounds")
        if src_t.lane_width == 1:
            return BoolType()
        return BitVecType(width=src_t.lane_width)

    if isinstance(expr, SimdInsertLane):
        src_t = infer_type(expr.x, ctx)
        if not isinstance(src_t, SimdType):
            raise TickIRValidationError("simd_insert_lane requires simd base")
        if expr.lane >= src_t.lanes:
            raise TickIRValidationError("lane out of bounds")
        v_t = infer_type(expr.value, ctx)
        if src_t.lane_width == 1:
            if not isinstance(v_t, BoolType):
                raise TickIRValidationError("insert value type mismatch")
        else:
            if not (isinstance(v_t, BitVecType) and v_t.width == src_t.lane_width):
                raise TickIRValidationError("insert value type mismatch")
        return src_t

    if isinstance(expr, SimdShuffle):
        src_t = infer_type(expr.x, ctx)
        if not isinstance(src_t, SimdType):
            raise TickIRValidationError("simd_shuffle requires simd source")
        if not expr.indices:
            raise TickIRValidationError("indices must be non-empty")
        if any(i < 0 or i >= src_t.lanes for i in expr.indices):
            raise TickIRValidationError("shuffle index out of bounds")
        return SimdType(lane_width=src_t.lane_width, lanes=len(expr.indices))

    if isinstance(expr, (FAdd, FMul, FSub, FDiv)):
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        if not (isinstance(a_t, FloatType) and type_equal(a_t, b_t)):
            raise TickIRValidationError("float op requires float operands")
        return a_t

    if isinstance(expr, (FSqrt, FNeg, FAbs)):
        x_t = infer_type(expr.x, ctx)
        if not isinstance(x_t, FloatType):
            raise TickIRValidationError("float op requires float operand")
        return x_t

    if isinstance(expr, (FEq, FLt, FLe, FNe)):
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        if not (isinstance(a_t, FloatType) and type_equal(a_t, b_t)):
            raise TickIRValidationError("float cmp requires float operands")
        return BoolType()

    if isinstance(expr, Not):
        t = infer_type(expr.x, ctx)
        if isinstance(t, FloatType):
            raise TickIRValidationError("not does not support float")
        return t

    if isinstance(
        expr,
        (And, Or, Xor, Add, Sub, Mul, Div, Shl, LShr, AShr, Eq, Ult, Ule, Ugt, Uge),
    ):
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        if not type_equal(a_t, b_t):
            raise TickIRValidationError("type mismatch")
        if isinstance(expr, (And, Or, Xor)) and isinstance(a_t, FloatType):
            raise TickIRValidationError("bitwise ops do not support float")
        if isinstance(expr, (Eq, Ult, Ule, Ugt, Uge)):
            return BoolType()
        if isinstance(expr, (Add, Sub, Mul, Div, Shl, LShr, AShr)) and not isinstance(
            a_t, BitVecType
        ):
            raise TickIRValidationError("bitvec op requires bitvec operands")
        return a_t

    if isinstance(expr, SimdAdd):
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        if not (isinstance(a_t, SimdType) and type_equal(a_t, b_t)):
            raise TickIRValidationError("simd_add requires simd operands")
        return a_t

    if isinstance(expr, SimdSub):
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        if not (isinstance(a_t, SimdType) and type_equal(a_t, b_t)):
            raise TickIRValidationError("simd_sub requires simd operands")
        return a_t

    if isinstance(expr, (SimdAddMasked, SimdSubMasked)):
        m_t = infer_type(expr.mask, ctx)
        p_t = infer_type(expr.passthru, ctx)
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        if not isinstance(m_t, SimdType) or m_t.lane_width != 1:
            raise TickIRValidationError("simd masked op requires simd mask")
        if not (
            isinstance(a_t, SimdType) and type_equal(a_t, b_t) and type_equal(a_t, p_t)
        ):
            raise TickIRValidationError(
                "simd masked op requires matching simd operands"
            )
        if m_t.lanes != a_t.lanes:
            raise TickIRValidationError("simd masked op mask lane mismatch")
        return a_t

    if isinstance(expr, (SimdFAdd, SimdFMul, SimdFSub, SimdFDiv)):
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        if not (isinstance(a_t, SimdType) and type_equal(a_t, b_t)):
            raise TickIRValidationError("simd float op requires simd operands")
        if a_t.lane_width not in {32, 64}:
            raise TickIRValidationError("simd float op requires lane_width in {32,64}")
        return a_t

    if isinstance(expr, SimdFFma):
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        c_t = infer_type(expr.c, ctx)
        if not (
            isinstance(a_t, SimdType) and type_equal(a_t, b_t) and type_equal(a_t, c_t)
        ):
            raise TickIRValidationError("simd fma requires simd operands")
        if a_t.lane_width not in {32, 64}:
            raise TickIRValidationError("simd fma requires lane_width in {32,64}")
        return a_t

    if isinstance(expr, (SimdFSqrt, SimdFNeg, SimdFAbs)):
        x_t = infer_type(expr.x, ctx)
        if not isinstance(x_t, SimdType):
            raise TickIRValidationError("simd float op requires simd operand")
        if x_t.lane_width not in {32, 64}:
            raise TickIRValidationError("simd float op requires lane_width in {32,64}")
        return x_t

    if isinstance(expr, (SimdFCmpEq, SimdFCmpLt, SimdFCmpLe, SimdFCmpNe)):
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        if not (isinstance(a_t, SimdType) and type_equal(a_t, b_t)):
            raise TickIRValidationError("simd float cmp requires simd operands")
        if a_t.lane_width not in {32, 64}:
            raise TickIRValidationError("simd float cmp requires lane_width in {32,64}")
        return SimdType(lane_width=1, lanes=a_t.lanes)

    if isinstance(expr, (SimdAddSatU, SimdSubSatU, SimdAddSatS, SimdSubSatS)):
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        if not (isinstance(a_t, SimdType) and type_equal(a_t, b_t)):
            raise TickIRValidationError("simd saturating op requires simd operands")
        return a_t

    if isinstance(expr, (SimdMulLo, SimdMulHiU, SimdMulHiS)):
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        if not (isinstance(a_t, SimdType) and type_equal(a_t, b_t)):
            raise TickIRValidationError("simd mul requires simd operands")
        return a_t

    if isinstance(expr, SimdMaddS16):
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        if not (isinstance(a_t, SimdType) and type_equal(a_t, b_t)):
            raise TickIRValidationError("simd_madd_s16 requires simd operands")
        if a_t.lane_width != 16:
            raise TickIRValidationError("simd_madd_s16 requires lane_width=16")
        if a_t.lanes % 2 != 0:
            raise TickIRValidationError("simd_madd_s16 requires even lanes")
        return SimdType(lane_width=32, lanes=a_t.lanes // 2)

    if isinstance(expr, (SimdUnpackLo, SimdUnpackHi)):
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        if not (isinstance(a_t, SimdType) and type_equal(a_t, b_t)):
            raise TickIRValidationError("simd unpack requires simd operands")
        if a_t.lanes % 2 != 0:
            raise TickIRValidationError("simd unpack requires even lanes")
        return a_t

    if isinstance(expr, (SimdPackSS16To8, SimdPackUS16To8)):
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        if not (isinstance(a_t, SimdType) and type_equal(a_t, b_t)):
            raise TickIRValidationError("simd pack requires simd operands")
        if a_t.lane_width != 16:
            raise TickIRValidationError("simd_pack_*16_to_8 requires lane_width=16")
        return SimdType(lane_width=8, lanes=a_t.lanes * 2)

    if isinstance(expr, SimdPackSS32To16):
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        if not (isinstance(a_t, SimdType) and type_equal(a_t, b_t)):
            raise TickIRValidationError("simd pack requires simd operands")
        if a_t.lane_width != 32:
            raise TickIRValidationError("simd_pack_ss32_to_16 requires lane_width=32")
        return SimdType(lane_width=16, lanes=a_t.lanes * 2)

    if isinstance(expr, SimdMaskExpand):
        if not isinstance(expr.to, SimdType):
            raise TickIRValidationError("simd_mask_expand target must be simd")
        src_t = infer_type(expr.x, ctx)
        if not (isinstance(src_t, SimdType) and src_t.lane_width == 1):
            raise TickIRValidationError("simd_mask_expand source must be simd mask")
        if src_t.lanes != expr.to.lanes:
            raise TickIRValidationError("simd_mask_expand lane mismatch")
        return expr.to

    if isinstance(expr, SimdMaskPack):
        src_t = infer_type(expr.x, ctx)
        if not isinstance(src_t, SimdType):
            raise TickIRValidationError("simd_mask_pack source must be simd")
        return SimdType(lane_width=1, lanes=src_t.lanes)

    if isinstance(expr, (SimdMinU, SimdMaxU, SimdMinS, SimdMaxS)):
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        if not (isinstance(a_t, SimdType) and type_equal(a_t, b_t)):
            raise TickIRValidationError("simd min/max requires simd operands")
        return a_t

    if isinstance(expr, SimdBlend):
        m_t = infer_type(expr.mask, ctx)
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        if not isinstance(m_t, SimdType) or m_t.lane_width != 1:
            raise TickIRValidationError("simd_blend mask must be simd mask")
        if not (isinstance(a_t, SimdType) and type_equal(a_t, b_t)):
            raise TickIRValidationError("simd_blend requires matching simd operands")
        if m_t.lanes != a_t.lanes:
            raise TickIRValidationError("simd_blend mask lane mismatch")
        return a_t

    if isinstance(expr, (SimdZExtLo, SimdSExtLo)):
        if not isinstance(expr.to, SimdType):
            raise TickIRValidationError("simd extend target must be simd")
        src_t = infer_type(expr.x, ctx)
        if not isinstance(src_t, SimdType):
            raise TickIRValidationError("simd extend source must be simd")
        dst_t = expr.to
        if dst_t.total_width != src_t.total_width:
            raise TickIRValidationError("simd extend total_width mismatch")
        if src_t.lanes % 2 != 0:
            raise TickIRValidationError("simd extend requires even lanes")
        if dst_t.lanes * dst_t.lane_width != src_t.total_width:
            raise TickIRValidationError("simd extend target width mismatch")
        if dst_t.lanes != src_t.lanes // 2:
            raise TickIRValidationError("simd extend requires lanes halved")
        if dst_t.lane_width != src_t.lane_width * 2:
            raise TickIRValidationError("simd extend requires lane_width doubled")
        return dst_t

    if isinstance(expr, SimdEq):
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        if not (isinstance(a_t, SimdType) and type_equal(a_t, b_t)):
            raise TickIRValidationError("simd_eq requires simd operands")
        return SimdType(lane_width=1, lanes=a_t.lanes)

    if isinstance(expr, SimdNot):
        a_t = infer_type(expr.x, ctx)
        if not isinstance(a_t, SimdType):
            raise TickIRValidationError("simd_not requires simd operand")
        return a_t

    if isinstance(expr, (SimdAnd, SimdOr, SimdXor)):
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        if not (isinstance(a_t, SimdType) and type_equal(a_t, b_t)):
            raise TickIRValidationError("simd bitwise requires simd operands")
        return a_t

    if isinstance(expr, (SimdShl, SimdLShr, SimdAShr)):
        a_t = infer_type(expr.a, ctx)
        sh_t = infer_type(expr.sh, ctx)
        if not isinstance(a_t, SimdType):
            raise TickIRValidationError("simd shift requires simd lhs")
        if not isinstance(sh_t, BitVecType):
            raise TickIRValidationError("simd shift amount must be bitvec")
        return a_t

    if isinstance(expr, (SimdUlt, SimdUle, SimdUgt, SimdUge)):
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        if not (isinstance(a_t, SimdType) and type_equal(a_t, b_t)):
            raise TickIRValidationError("simd unsigned compare requires simd operands")
        return SimdType(lane_width=1, lanes=a_t.lanes)

    if isinstance(expr, (SimdSlt, SimdSle, SimdSgt, SimdSge)):
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        if not (isinstance(a_t, SimdType) and type_equal(a_t, b_t)):
            raise TickIRValidationError("simd signed compare requires simd operands")
        return SimdType(lane_width=1, lanes=a_t.lanes)

    if isinstance(expr, Mux):
        c_t = infer_type(expr.cond, ctx)
        if not isinstance(c_t, BoolType):
            raise TickIRValidationError("mux condition must be bool")
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        if not type_equal(a_t, b_t):
            raise TickIRValidationError("mux branch type mismatch")
        return a_t

    if isinstance(expr, Concat):
        total = 0
        for part in expr.parts:
            t = infer_type(part, ctx)
            if isinstance(t, SimdType):
                raise TickIRValidationError("concat does not support simd parts")
            if isinstance(t, BoolType):
                total += 1
            else:
                assert isinstance(t, BitVecType)
                total += t.width
        return BitVecType(width=total)

    if isinstance(expr, Slice):
        t = infer_type(expr.x, ctx)
        if not isinstance(t, (BitVecType, SimdType)):
            raise TickIRValidationError("slice source must be bitvec")
        total = t.width if isinstance(t, BitVecType) else t.total_width
        if expr.offset + expr.width > total:
            raise TickIRValidationError("slice out of bounds")
        if expr.width == 1:
            return BoolType()
        return BitVecType(width=expr.width)

    if isinstance(expr, Lut8):
        x_t = infer_type(expr.x, ctx)
        if not isinstance(x_t, BitVecType) or x_t.width != 8:
            raise TickIRValidationError("lut8 input must be bitvec8")
        return BitVecType(width=8)

    from stc.tick_ir import TernaryLut

    if isinstance(expr, TernaryLut):
        a_t = infer_type(expr.a, ctx)
        b_t = infer_type(expr.b, ctx)
        c_t = infer_type(expr.c, ctx)
        if a_t != b_t or b_t != c_t:
            raise TickIRValidationError("TernaryLut inputs must have same type")
        return a_t

    raise TypeError("unsupported expression type")


def _as_bool(value: int | bool) -> bool:
    if isinstance(value, bool):
        return value
    return bool(value & 1)


def _as_int(value: int | bool) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    return int(value)


def eval_expr(
    expr: Expr, ctx_types: dict[str, Type], env: dict[str, int | bool]
) -> int | bool:
    if isinstance(expr, BoolConst):
        return expr.value
    if isinstance(expr, BitVecConst):
        return expr.value
    if isinstance(expr, FloatConst):
        return expr.bits
    if isinstance(expr, SimdConst):
        return expr.value
    if isinstance(expr, Var):
        if expr.name not in env:
            raise TickIRValidationError(f"missing value for {expr.name}")
        value = env[expr.name]
        t = ctx_types[expr.name]
        if isinstance(t, BoolType):
            return _as_bool(value)
        if isinstance(t, BitVecType):
            return _as_int(value) & _mask(t.width)
        if isinstance(t, FloatType):
            return _as_int(value) & _mask(t.width)
        if isinstance(t, SimdType):
            return _as_int(value) & _mask(t.total_width)
        raise TypeError("unknown type")

    if isinstance(expr, Bitcast):
        dst_t = expr.to
        if isinstance(dst_t, BoolType):
            raise TickIRValidationError("bitcast does not support bool")
        x = eval_expr(expr.x, ctx_types, env)
        if isinstance(dst_t, BitVecType):
            return _as_int(x) & _mask(dst_t.width)
        if isinstance(dst_t, FloatType):
            return _as_int(x) & _mask(dst_t.width)
        assert isinstance(dst_t, SimdType)
        return _as_int(x) & _mask(dst_t.total_width)

    from stc.tick_ir import Rotl, Rotr

    if isinstance(expr, (Rotl, Rotr)):
        x_t = infer_type(expr.x, ctx_types)
        assert isinstance(x_t, BitVecType)
        x_val = _as_int(eval_expr(expr.x, ctx_types, env)) & _mask(x_t.width)
        sh_val = _as_int(eval_expr(expr.sh, ctx_types, env))
        sh_val = sh_val % x_t.width
        if isinstance(expr, Rotl):
            result = ((x_val << sh_val) | (x_val >> (x_t.width - sh_val))) & _mask(
                x_t.width
            )
        else:
            result = ((x_val >> sh_val) | (x_val << (x_t.width - sh_val))) & _mask(
                x_t.width
            )
        return result

    if isinstance(expr, BitTranspose):
        x = _as_int(eval_expr(expr.x, ctx_types, env))
        lane_width = expr.lane_width
        lanes = expr.lanes
        total_bits = lane_width * lanes
        out = 0
        for lane in range(lanes):
            for bit in range(lane_width):
                src_pos = lane * lane_width + bit
                dst_pos = bit * lanes + lane
                if (x >> src_pos) & 1:
                    out |= 1 << dst_pos
        return out & _mask(total_bits)

    if isinstance(expr, SimdSplat):
        dst_t = expr.to
        assert isinstance(dst_t, SimdType)
        x = eval_expr(expr.x, ctx_types, env)
        lane_mask = _mask(dst_t.lane_width)
        lane = _as_int(x) & lane_mask
        out = 0
        for i in range(dst_t.lanes):
            out |= lane << (i * dst_t.lane_width)
        return out & _mask(dst_t.total_width)

    if isinstance(expr, SimdExtractLane):
        src_t = infer_type(expr.x, ctx_types)
        assert isinstance(src_t, SimdType)
        vv = _as_int(eval_expr(expr.x, ctx_types, env)) & _mask(src_t.total_width)
        off = expr.lane * src_t.lane_width
        lane = (vv >> off) & _mask(src_t.lane_width)
        if src_t.lane_width == 1:
            return bool(lane & 1)
        return lane

    if isinstance(expr, SimdInsertLane):
        src_t = infer_type(expr.x, ctx_types)
        assert isinstance(src_t, SimdType)
        base = _as_int(eval_expr(expr.x, ctx_types, env)) & _mask(src_t.total_width)
        v = eval_expr(expr.value, ctx_types, env)
        lane_mask = _mask(src_t.lane_width)
        off = expr.lane * src_t.lane_width
        if src_t.lane_width == 1:
            vv = 1 if _as_bool(v) else 0
        else:
            vv = _as_int(v) & lane_mask
        cleared = base & ~(lane_mask << off)
        return (cleared | (vv << off)) & _mask(src_t.total_width)

    if isinstance(expr, SimdShuffle):
        src_t = infer_type(expr.x, ctx_types)
        assert isinstance(src_t, SimdType)
        vv = _as_int(eval_expr(expr.x, ctx_types, env)) & _mask(src_t.total_width)
        out_lanes = len(expr.indices)
        out_t = SimdType(lane_width=src_t.lane_width, lanes=out_lanes)
        lane_mask = _mask(src_t.lane_width)
        out = 0
        for i, src_lane in enumerate(expr.indices):
            if src_lane < 0 or src_lane >= src_t.lanes:
                raise TickIRValidationError("shuffle index out of bounds")
            src_off = src_lane * src_t.lane_width
            lane_val = (vv >> src_off) & lane_mask
            out |= lane_val << (i * src_t.lane_width)
        return out & _mask(out_t.total_width)

    if isinstance(expr, (FAdd, FMul, FSub, FDiv)):
        t = infer_type(expr, ctx_types)
        assert isinstance(t, FloatType)
        aa = _as_int(eval_expr(expr.a, ctx_types, env)) & _mask(t.width)
        bb = _as_int(eval_expr(expr.b, ctx_types, env)) & _mask(t.width)
        aa = _canonicalize_nan_bits(t.width, aa) & _mask(t.width)
        bb = _canonicalize_nan_bits(t.width, bb) & _mask(t.width)
        if t.width == 32:
            a = _f32_from_bits(aa)
            b = _f32_from_bits(bb)
            if isinstance(expr, FAdd):
                r = a + b
            elif isinstance(expr, FSub):
                r = a - b
            elif isinstance(expr, FMul):
                r = a * b
            else:
                r = a / b
            out_bits = _f32_to_bits(r)
            return _canonicalize_nan_bits(32, out_bits) & _mask(32)
        a = _f64_from_bits(aa)
        b = _f64_from_bits(bb)
        if isinstance(expr, FAdd):
            r = a + b
        elif isinstance(expr, FSub):
            r = a - b
        elif isinstance(expr, FMul):
            r = a * b
        else:
            r = a / b
        out_bits = _f64_to_bits(r)
        return _canonicalize_nan_bits(64, out_bits) & _mask(64)

    if isinstance(expr, (FSqrt, FNeg, FAbs)):
        t = infer_type(expr, ctx_types)
        assert isinstance(t, FloatType)
        xx = _as_int(eval_expr(expr.x, ctx_types, env)) & _mask(t.width)
        xx = _canonicalize_nan_bits(t.width, xx) & _mask(t.width)
        if isinstance(expr, FNeg):
            sign = 1 << (t.width - 1)
            out_bits = int(xx) ^ sign
            return _canonicalize_nan_bits(t.width, out_bits) & _mask(t.width)
        if isinstance(expr, FAbs):
            sign = 1 << (t.width - 1)
            out_bits = int(xx) & ~sign
            return _canonicalize_nan_bits(t.width, out_bits) & _mask(t.width)
        if t.width == 32:
            x = _f32_from_bits(xx)
            try:
                out_bits = _f32_to_bits(math.sqrt(x))
            except ValueError:
                out_bits = _canonical_nan_bits(32)
            return _canonicalize_nan_bits(32, out_bits) & _mask(32)
        x = _f64_from_bits(xx)
        try:
            out_bits = _f64_to_bits(math.sqrt(x))
        except ValueError:
            out_bits = _canonical_nan_bits(64)
        return _canonicalize_nan_bits(64, out_bits) & _mask(64)

    if isinstance(expr, (FEq, FLt, FLe, FNe)):
        t = infer_type(expr.a, ctx_types)
        assert isinstance(t, FloatType)
        aa = _as_int(eval_expr(expr.a, ctx_types, env)) & _mask(t.width)
        bb = _as_int(eval_expr(expr.b, ctx_types, env)) & _mask(t.width)
        aa = _canonicalize_nan_bits(t.width, aa) & _mask(t.width)
        bb = _canonicalize_nan_bits(t.width, bb) & _mask(t.width)
        if t.width == 32:
            a = _f32_from_bits(aa)
            b = _f32_from_bits(bb)
        else:
            a = _f64_from_bits(aa)
            b = _f64_from_bits(bb)
        if isinstance(expr, FEq):
            return bool(a == b)
        if isinstance(expr, FLt):
            return bool(a < b)
        if isinstance(expr, FLe):
            return bool(a <= b)
        return bool(a != b)

    if isinstance(expr, (SimdFAdd, SimdFMul, SimdFSub, SimdFDiv)):
        t = infer_type(expr, ctx_types)
        assert isinstance(t, SimdType)
        aa = _as_int(eval_expr(expr.a, ctx_types, env)) & _mask(t.total_width)
        bb = _as_int(eval_expr(expr.b, ctx_types, env)) & _mask(t.total_width)
        out = 0
        lane_mask = _mask(t.lane_width)
        for i in range(t.lanes):
            a_bits = (aa >> (t.lane_width * i)) & lane_mask
            b_bits = (bb >> (t.lane_width * i)) & lane_mask
            a_bits = _canonicalize_nan_bits(t.lane_width, a_bits) & lane_mask
            b_bits = _canonicalize_nan_bits(t.lane_width, b_bits) & lane_mask
            if t.lane_width == 32:
                a = _f32_from_bits(a_bits)
                b = _f32_from_bits(b_bits)
                if isinstance(expr, SimdFAdd):
                    r = a + b
                elif isinstance(expr, SimdFSub):
                    r = a - b
                elif isinstance(expr, SimdFMul):
                    r = a * b
                else:
                    r = a / b
                r_bits = _canonicalize_nan_bits(32, _f32_to_bits(r)) & lane_mask
            else:
                a = _f64_from_bits(a_bits)
                b = _f64_from_bits(b_bits)
                if isinstance(expr, SimdFAdd):
                    r = a + b
                elif isinstance(expr, SimdFSub):
                    r = a - b
                elif isinstance(expr, SimdFMul):
                    r = a * b
                else:
                    r = a / b
                r_bits = _canonicalize_nan_bits(64, _f64_to_bits(r)) & lane_mask
            out |= (int(r_bits) & lane_mask) << (t.lane_width * i)
        return out & _mask(t.total_width)

    if isinstance(expr, SimdFFma):
        t = infer_type(expr, ctx_types)
        assert isinstance(t, SimdType)
        aa = _as_int(eval_expr(expr.a, ctx_types, env)) & _mask(t.total_width)
        bb = _as_int(eval_expr(expr.b, ctx_types, env)) & _mask(t.total_width)
        cc = _as_int(eval_expr(expr.c, ctx_types, env)) & _mask(t.total_width)
        out = 0
        lane_mask = _mask(t.lane_width)
        for i in range(t.lanes):
            a_bits = (aa >> (t.lane_width * i)) & lane_mask
            b_bits = (bb >> (t.lane_width * i)) & lane_mask
            c_bits = (cc >> (t.lane_width * i)) & lane_mask
            a_bits = _canonicalize_nan_bits(t.lane_width, a_bits) & lane_mask
            b_bits = _canonicalize_nan_bits(t.lane_width, b_bits) & lane_mask
            c_bits = _canonicalize_nan_bits(t.lane_width, c_bits) & lane_mask
            if t.lane_width == 32:
                a = _f32_from_bits(a_bits)
                b = _f32_from_bits(b_bits)
                c = _f32_from_bits(c_bits)
                r_bits = (
                    _canonicalize_nan_bits(32, _f32_to_bits((a * b) + c)) & lane_mask
                )
            else:
                a = _f64_from_bits(a_bits)
                b = _f64_from_bits(b_bits)
                c = _f64_from_bits(c_bits)
                r_bits = (
                    _canonicalize_nan_bits(64, _f64_to_bits(_f64_fma(a, b, c)))
                    & lane_mask
                )
            out |= (int(r_bits) & lane_mask) << (t.lane_width * i)
        return out & _mask(t.total_width)

    if isinstance(expr, (SimdFSqrt, SimdFNeg, SimdFAbs)):
        t = infer_type(expr, ctx_types)
        assert isinstance(t, SimdType)
        xx = _as_int(eval_expr(expr.x, ctx_types, env)) & _mask(t.total_width)
        out = 0
        lane_mask = _mask(t.lane_width)
        sign = 1 << (t.lane_width - 1)
        for i in range(t.lanes):
            bits = (xx >> (t.lane_width * i)) & lane_mask
            bits = _canonicalize_nan_bits(t.lane_width, bits) & lane_mask
            if isinstance(expr, SimdFNeg):
                out_bits = _canonicalize_nan_bits(t.lane_width, int(bits) ^ sign)
            elif isinstance(expr, SimdFAbs):
                out_bits = _canonicalize_nan_bits(t.lane_width, int(bits) & ~sign)
            else:
                if t.lane_width == 32:
                    x = _f32_from_bits(bits)
                    try:
                        out_bits = _f32_to_bits(math.sqrt(x))
                    except ValueError:
                        out_bits = _canonical_nan_bits(32)
                    out_bits = _canonicalize_nan_bits(32, out_bits)
                else:
                    x = _f64_from_bits(bits)
                    try:
                        out_bits = _f64_to_bits(math.sqrt(x))
                    except ValueError:
                        out_bits = _canonical_nan_bits(64)
                    out_bits = _canonicalize_nan_bits(64, out_bits)
            out |= (int(out_bits) & lane_mask) << (t.lane_width * i)
        return out & _mask(t.total_width)

    if isinstance(expr, (SimdFCmpEq, SimdFCmpLt, SimdFCmpLe, SimdFCmpNe)):
        at = infer_type(expr.a, ctx_types)
        assert isinstance(at, SimdType)
        aa = _as_int(eval_expr(expr.a, ctx_types, env)) & _mask(at.total_width)
        bb = _as_int(eval_expr(expr.b, ctx_types, env)) & _mask(at.total_width)
        out = 0
        lane_mask = _mask(at.lane_width)
        for i in range(at.lanes):
            a_bits = (aa >> (at.lane_width * i)) & lane_mask
            b_bits = (bb >> (at.lane_width * i)) & lane_mask
            a_bits = _canonicalize_nan_bits(at.lane_width, a_bits) & lane_mask
            b_bits = _canonicalize_nan_bits(at.lane_width, b_bits) & lane_mask
            if at.lane_width == 32:
                a = _f32_from_bits(a_bits)
                b = _f32_from_bits(b_bits)
            else:
                a = _f64_from_bits(a_bits)
                b = _f64_from_bits(b_bits)
            if isinstance(expr, SimdFCmpEq):
                ok = a == b
            elif isinstance(expr, SimdFCmpLt):
                ok = a < b
            elif isinstance(expr, SimdFCmpLe):
                ok = a <= b
            else:
                ok = a != b
            if ok:
                out |= 1 << i
        return out & _mask(at.lanes)

    if isinstance(expr, Not):
        x = eval_expr(expr.x, ctx_types, env)
        t = infer_type(expr.x, ctx_types)
        if isinstance(t, BoolType):
            return not _as_bool(x)
        if isinstance(t, BitVecType):
            return (~_as_int(x)) & _mask(t.width)
        if isinstance(t, SimdType):
            return (~_as_int(x)) & _mask(t.total_width)
        raise TypeError("unknown type")

    if isinstance(
        expr,
        (And, Or, Xor, Add, Sub, Mul, Div, Shl, LShr, AShr, Eq, Ult, Ule, Ugt, Uge),
    ):
        a = eval_expr(expr.a, ctx_types, env)
        b = eval_expr(expr.b, ctx_types, env)
        t = infer_type(expr.a, ctx_types)

        if isinstance(expr, (Eq, Ult, Ule, Ugt, Uge)):
            if isinstance(t, BoolType):
                if not isinstance(expr, Eq):
                    raise TickIRValidationError(
                        "unsigned compares require bitvec operands"
                    )
                return _as_bool(a) == _as_bool(b)
            if isinstance(t, BitVecType):
                aa = _as_int(a) & _mask(t.width)
                bb = _as_int(b) & _mask(t.width)
                if isinstance(expr, Eq):
                    return aa == bb
                if isinstance(expr, Ult):
                    return aa < bb
                if isinstance(expr, Ule):
                    return aa <= bb
                if isinstance(expr, Ugt):
                    return aa > bb
                return aa >= bb
            if isinstance(t, SimdType):
                m = _mask(t.total_width)
                if not isinstance(expr, Eq):
                    raise TickIRValidationError(
                        "unsigned compares require bitvec operands"
                    )
                return (_as_int(a) & m) == (_as_int(b) & m)
            raise TypeError("unknown type")

        if isinstance(t, BoolType):
            aa = _as_bool(a)
            bb = _as_bool(b)
            if isinstance(expr, And):
                return aa and bb
            if isinstance(expr, Or):
                return aa or bb
            if isinstance(expr, Xor):
                return aa ^ bb
            raise TickIRValidationError("add requires bitvec operands")

        if isinstance(t, BitVecType):
            aa = _as_int(a) & _mask(t.width)
            bb = _as_int(b) & _mask(t.width)
            if isinstance(expr, And):
                return aa & bb
            if isinstance(expr, Or):
                return aa | bb
            if isinstance(expr, Xor):
                return aa ^ bb
            if isinstance(expr, Add):
                return (aa + bb) & _mask(t.width)
            if isinstance(expr, Sub):
                return (aa - bb) & _mask(t.width)
            if isinstance(expr, Mul):
                return (aa * bb) & _mask(t.width)
            if isinstance(expr, Div):
                if bb == 0:
                    return 0
                return (aa // bb) & _mask(t.width)
            if isinstance(expr, Shl):
                return (aa << bb) & _mask(t.width)
            if isinstance(expr, LShr):
                return (aa >> bb) & _mask(t.width)
            if isinstance(expr, AShr):
                if bb >= t.width:
                    return _mask(t.width) if (aa & (1 << (t.width - 1))) else 0
                signed = aa
                if aa & (1 << (t.width - 1)):
                    signed = aa - (1 << t.width)
                return (signed >> bb) & _mask(t.width)
            raise TypeError("unreachable")

        if isinstance(t, SimdType):
            aa = _as_int(a) & _mask(t.total_width)
            bb = _as_int(b) & _mask(t.total_width)
            if isinstance(expr, And):
                return aa & bb
            if isinstance(expr, Or):
                return aa | bb
            if isinstance(expr, Xor):
                return aa ^ bb
            raise TickIRValidationError("add requires bitvec operands")

        raise TypeError("unknown type")

    if isinstance(expr, SimdAdd):
        t = infer_type(expr, ctx_types)
        assert isinstance(t, SimdType)
        aa = _as_int(eval_expr(expr.a, ctx_types, env)) & _mask(t.total_width)
        bb = _as_int(eval_expr(expr.b, ctx_types, env)) & _mask(t.total_width)
        out = 0
        lane_mask = _mask(t.lane_width)
        for i in range(t.lanes):
            off = i * t.lane_width
            la = (aa >> off) & lane_mask
            lb = (bb >> off) & lane_mask
            out |= ((la + lb) & lane_mask) << off
        return out & _mask(t.total_width)

    if isinstance(expr, SimdSub):
        t = infer_type(expr, ctx_types)
        assert isinstance(t, SimdType)
        aa = _as_int(eval_expr(expr.a, ctx_types, env)) & _mask(t.total_width)
        bb = _as_int(eval_expr(expr.b, ctx_types, env)) & _mask(t.total_width)
        out = 0
        lane_mask = _mask(t.lane_width)
        for i in range(t.lanes):
            off = i * t.lane_width
            la = (aa >> off) & lane_mask
            lb = (bb >> off) & lane_mask
            out |= ((la - lb) & lane_mask) << off
        return out & _mask(t.total_width)

    if isinstance(expr, (SimdAddMasked, SimdSubMasked)):
        t = infer_type(expr, ctx_types)
        assert isinstance(t, SimdType)
        mask_t = infer_type(expr.mask, ctx_types)
        assert isinstance(mask_t, SimdType)
        m = _as_int(eval_expr(expr.mask, ctx_types, env)) & _mask(mask_t.lanes)
        p = _as_int(eval_expr(expr.passthru, ctx_types, env)) & _mask(t.total_width)
        aa = _as_int(eval_expr(expr.a, ctx_types, env)) & _mask(t.total_width)
        bb = _as_int(eval_expr(expr.b, ctx_types, env)) & _mask(t.total_width)
        out = 0
        lane_mask = _mask(t.lane_width)
        for i in range(t.lanes):
            off = i * t.lane_width
            lp = (p >> off) & lane_mask
            la = (aa >> off) & lane_mask
            lb = (bb >> off) & lane_mask
            if (m >> i) & 1:
                v = (la + lb) if isinstance(expr, SimdAddMasked) else (la - lb)
                out |= (v & lane_mask) << off
            else:
                out |= lp << off
        return out & _mask(t.total_width)

    if isinstance(expr, SimdAddSatU):
        t = infer_type(expr, ctx_types)
        assert isinstance(t, SimdType)
        aa = _as_int(eval_expr(expr.a, ctx_types, env)) & _mask(t.total_width)
        bb = _as_int(eval_expr(expr.b, ctx_types, env)) & _mask(t.total_width)
        out = 0
        lane_mask = _mask(t.lane_width)
        lane_max = lane_mask
        for i in range(t.lanes):
            off = i * t.lane_width
            la = (aa >> off) & lane_mask
            lb = (bb >> off) & lane_mask
            s = la + lb
            out |= (lane_max if s > lane_max else s) << off
        return out & _mask(t.total_width)

    if isinstance(expr, SimdSubSatU):
        t = infer_type(expr, ctx_types)
        assert isinstance(t, SimdType)
        aa = _as_int(eval_expr(expr.a, ctx_types, env)) & _mask(t.total_width)
        bb = _as_int(eval_expr(expr.b, ctx_types, env)) & _mask(t.total_width)
        out = 0
        lane_mask = _mask(t.lane_width)
        for i in range(t.lanes):
            off = i * t.lane_width
            la = (aa >> off) & lane_mask
            lb = (bb >> off) & lane_mask
            out |= (0 if la < lb else (la - lb)) << off
        return out & _mask(t.total_width)

    if isinstance(expr, (SimdAddSatS, SimdSubSatS)):
        t = infer_type(expr, ctx_types)
        assert isinstance(t, SimdType)
        aa = _as_int(eval_expr(expr.a, ctx_types, env)) & _mask(t.total_width)
        bb = _as_int(eval_expr(expr.b, ctx_types, env)) & _mask(t.total_width)
        out = 0
        lane_mask = _mask(t.lane_width)
        lane_min = -(1 << (t.lane_width - 1))
        lane_max = (1 << (t.lane_width - 1)) - 1
        sign_bit = 1 << (t.lane_width - 1)
        for i in range(t.lanes):
            off = i * t.lane_width
            la = (aa >> off) & lane_mask
            lb = (bb >> off) & lane_mask
            if la & sign_bit:
                la = la - (1 << t.lane_width)
            if lb & sign_bit:
                lb = lb - (1 << t.lane_width)
            s = la + lb if isinstance(expr, SimdAddSatS) else la - lb
            if s > lane_max:
                s = lane_max
            if s < lane_min:
                s = lane_min
            out |= (s & lane_mask) << off
        return out & _mask(t.total_width)

    if isinstance(expr, (SimdMulLo, SimdMulHiU, SimdMulHiS)):
        t = infer_type(expr, ctx_types)
        assert isinstance(t, SimdType)
        aa = _as_int(eval_expr(expr.a, ctx_types, env)) & _mask(t.total_width)
        bb = _as_int(eval_expr(expr.b, ctx_types, env)) & _mask(t.total_width)
        out = 0
        lane_mask = _mask(t.lane_width)
        sign_bit = 1 << (t.lane_width - 1)
        for i in range(t.lanes):
            off = i * t.lane_width
            la = (aa >> off) & lane_mask
            lb = (bb >> off) & lane_mask
            if isinstance(expr, SimdMulLo):
                prod = (la * lb) & lane_mask
            else:
                if isinstance(expr, SimdMulHiS):
                    if la & sign_bit:
                        la = la - (1 << t.lane_width)
                    if lb & sign_bit:
                        lb = lb - (1 << t.lane_width)
                    prod_full = int(la) * int(lb)
                else:
                    prod_full = int(la) * int(lb)
                prod = (prod_full >> t.lane_width) & lane_mask
            out |= int(prod) << off
        return out & _mask(t.total_width)

    if isinstance(expr, SimdMaddS16):
        out_t = infer_type(expr, ctx_types)
        assert isinstance(out_t, SimdType)
        a_t = infer_type(expr.a, ctx_types)
        assert isinstance(a_t, SimdType)
        aa = _as_int(eval_expr(expr.a, ctx_types, env)) & _mask(a_t.total_width)
        bb = _as_int(eval_expr(expr.b, ctx_types, env)) & _mask(a_t.total_width)
        out = 0
        for i in range(out_t.lanes):
            a0 = (aa >> (16 * (2 * i))) & 0xFFFF
            a1 = (aa >> (16 * (2 * i + 1))) & 0xFFFF
            b0 = (bb >> (16 * (2 * i))) & 0xFFFF
            b1 = (bb >> (16 * (2 * i + 1))) & 0xFFFF
            if a0 & 0x8000:
                a0 = a0 - 0x10000
            if a1 & 0x8000:
                a1 = a1 - 0x10000
            if b0 & 0x8000:
                b0 = b0 - 0x10000
            if b1 & 0x8000:
                b1 = b1 - 0x10000
            s = int(a0) * int(b0) + int(a1) * int(b1)
            out |= (s & 0xFFFFFFFF) << (32 * i)
        return out & _mask(out_t.total_width)

    if isinstance(expr, (SimdUnpackLo, SimdUnpackHi)):
        t = infer_type(expr, ctx_types)
        assert isinstance(t, SimdType)
        aa = _as_int(eval_expr(expr.a, ctx_types, env)) & _mask(t.total_width)
        bb = _as_int(eval_expr(expr.b, ctx_types, env)) & _mask(t.total_width)
        lane_mask = _mask(t.lane_width)
        half = t.lanes // 2
        out = 0
        for i in range(half):
            src = i if isinstance(expr, SimdUnpackLo) else (i + half)
            la = (aa >> (t.lane_width * src)) & lane_mask
            lb = (bb >> (t.lane_width * src)) & lane_mask
            out |= int(la) << (t.lane_width * (2 * i))
            out |= int(lb) << (t.lane_width * (2 * i + 1))
        return out & _mask(t.total_width)

    if isinstance(expr, (SimdPackSS16To8, SimdPackUS16To8, SimdPackSS32To16)):
        out_t = infer_type(expr, ctx_types)
        assert isinstance(out_t, SimdType)
        a_t = infer_type(expr.a, ctx_types)
        assert isinstance(a_t, SimdType)
        aa = _as_int(eval_expr(expr.a, ctx_types, env)) & _mask(a_t.total_width)
        bb = _as_int(eval_expr(expr.b, ctx_types, env)) & _mask(a_t.total_width)
        out = 0
        if isinstance(expr, (SimdPackSS16To8, SimdPackUS16To8)):
            lane_in_w = 16
            lane_out_w = 8
            lanes_in = a_t.lanes
            for i in range(lanes_in * 2):
                src = aa if i < lanes_in else bb
                si = i if i < lanes_in else (i - lanes_in)
                v = (src >> (lane_in_w * si)) & 0xFFFF
                if v & 0x8000:
                    v = v - 0x10000
                if isinstance(expr, SimdPackUS16To8):
                    if v < 0:
                        v = 0
                    if v > 255:
                        v = 255
                else:
                    if v > 127:
                        v = 127
                    if v < -128:
                        v = -128
                out |= (int(v) & 0xFF) << (lane_out_w * i)
            return out & _mask(out_t.total_width)

        lane_in_w = 32
        lane_out_w = 16
        lanes_in = a_t.lanes
        for i in range(lanes_in * 2):
            src = aa if i < lanes_in else bb
            si = i if i < lanes_in else (i - lanes_in)
            v = (src >> (lane_in_w * si)) & 0xFFFFFFFF
            if v & 0x80000000:
                v = v - 0x100000000
            if v > 32767:
                v = 32767
            if v < -32768:
                v = -32768
            out |= (int(v) & 0xFFFF) << (lane_out_w * i)
        return out & _mask(out_t.total_width)

    if isinstance(expr, SimdMaskExpand):
        dst_t = infer_type(expr, ctx_types)
        assert isinstance(dst_t, SimdType)
        src_t = infer_type(expr.x, ctx_types)
        assert isinstance(src_t, SimdType)
        m = _as_int(eval_expr(expr.x, ctx_types, env)) & _mask(src_t.lanes)
        out = 0
        lane_ones = _mask(dst_t.lane_width)
        for i in range(dst_t.lanes):
            if (m >> i) & 1:
                out |= lane_ones << (dst_t.lane_width * i)
        return out & _mask(dst_t.total_width)

    if isinstance(expr, SimdMaskPack):
        src_t = infer_type(expr.x, ctx_types)
        assert isinstance(src_t, SimdType)
        aa = _as_int(eval_expr(expr.x, ctx_types, env)) & _mask(src_t.total_width)
        lane_mask = _mask(src_t.lane_width)
        out = 0
        for i in range(src_t.lanes):
            lane = (aa >> (src_t.lane_width * i)) & lane_mask
            if lane != 0:
                out |= 1 << i
        return out & _mask(src_t.lanes)

    if isinstance(expr, (SimdMinU, SimdMaxU, SimdMinS, SimdMaxS)):
        t = infer_type(expr, ctx_types)
        assert isinstance(t, SimdType)
        aa = _as_int(eval_expr(expr.a, ctx_types, env)) & _mask(t.total_width)
        bb = _as_int(eval_expr(expr.b, ctx_types, env)) & _mask(t.total_width)
        out = 0
        lane_mask = _mask(t.lane_width)
        sign_bit = 1 << (t.lane_width - 1)
        for i in range(t.lanes):
            off = i * t.lane_width
            la = (aa >> off) & lane_mask
            lb = (bb >> off) & lane_mask
            if isinstance(expr, (SimdMinS, SimdMaxS)):
                sa = la - (1 << t.lane_width) if (la & sign_bit) else la
                sb = lb - (1 << t.lane_width) if (lb & sign_bit) else lb
                pick_a = sa < sb if isinstance(expr, SimdMinS) else sa > sb
            else:
                pick_a = la < lb if isinstance(expr, SimdMinU) else la > lb
            out |= (la if pick_a else lb) << off
        return out & _mask(t.total_width)

    if isinstance(expr, SimdBlend):
        t = infer_type(expr, ctx_types)
        assert isinstance(t, SimdType)
        mask_t = infer_type(expr.mask, ctx_types)
        assert isinstance(mask_t, SimdType)
        m = _as_int(eval_expr(expr.mask, ctx_types, env)) & _mask(mask_t.lanes)
        aa = _as_int(eval_expr(expr.a, ctx_types, env)) & _mask(t.total_width)
        bb = _as_int(eval_expr(expr.b, ctx_types, env)) & _mask(t.total_width)
        out = 0
        lane_mask = _mask(t.lane_width)
        for i in range(t.lanes):
            off = i * t.lane_width
            la = (aa >> off) & lane_mask
            lb = (bb >> off) & lane_mask
            out |= (lb if ((m >> i) & 1) else la) << off
        return out & _mask(t.total_width)

    if isinstance(expr, (SimdZExtLo, SimdSExtLo)):
        dst_t = infer_type(expr, ctx_types)
        assert isinstance(dst_t, SimdType)
        src_t = infer_type(expr.x, ctx_types)
        assert isinstance(src_t, SimdType)
        x = _as_int(eval_expr(expr.x, ctx_types, env)) & _mask(src_t.total_width)
        in_mask = _mask(src_t.lane_width)
        out = 0
        sign_bit = 1 << (src_t.lane_width - 1)
        for i in range(dst_t.lanes):
            v = (x >> (src_t.lane_width * i)) & in_mask
            if isinstance(expr, SimdSExtLo) and (v & sign_bit):
                v = v - (1 << src_t.lane_width)
            out |= (int(v) & _mask(dst_t.lane_width)) << (dst_t.lane_width * i)
        return out & _mask(dst_t.total_width)

    if isinstance(expr, SimdEq):
        a_t = infer_type(expr.a, ctx_types)
        assert isinstance(a_t, SimdType)
        aa = _as_int(eval_expr(expr.a, ctx_types, env)) & _mask(a_t.total_width)
        bb = _as_int(eval_expr(expr.b, ctx_types, env)) & _mask(a_t.total_width)
        out = 0
        lane_mask = _mask(a_t.lane_width)
        for i in range(a_t.lanes):
            off = i * a_t.lane_width
            la = (aa >> off) & lane_mask
            lb = (bb >> off) & lane_mask
            if la == lb:
                out |= 1 << i
        return out & _mask(a_t.lanes)

    if isinstance(expr, SimdNot):
        t = infer_type(expr.x, ctx_types)
        assert isinstance(t, SimdType)
        aa = _as_int(eval_expr(expr.x, ctx_types, env)) & _mask(t.total_width)
        return (~aa) & _mask(t.total_width)

    if isinstance(expr, (SimdAnd, SimdOr, SimdXor)):
        t = infer_type(expr.a, ctx_types)
        assert isinstance(t, SimdType)
        aa = _as_int(eval_expr(expr.a, ctx_types, env)) & _mask(t.total_width)
        bb = _as_int(eval_expr(expr.b, ctx_types, env)) & _mask(t.total_width)
        if isinstance(expr, SimdAnd):
            return aa & bb
        if isinstance(expr, SimdOr):
            return aa | bb
        return aa ^ bb

    if isinstance(expr, (SimdShl, SimdLShr, SimdAShr)):
        a_t = infer_type(expr.a, ctx_types)
        assert isinstance(a_t, SimdType)
        aa = _as_int(eval_expr(expr.a, ctx_types, env)) & _mask(a_t.total_width)
        sh = _as_int(eval_expr(expr.sh, ctx_types, env))
        lane_mask = _mask(a_t.lane_width)
        out = 0
        for i in range(a_t.lanes):
            off = i * a_t.lane_width
            la = (aa >> off) & lane_mask
            if isinstance(expr, SimdShl):
                lv = (la << sh) & lane_mask
            elif isinstance(expr, SimdLShr):
                lv = (la >> sh) & lane_mask
            else:
                if sh >= a_t.lane_width:
                    lv = lane_mask if (la & (1 << (a_t.lane_width - 1))) else 0
                else:
                    signed = la
                    if la & (1 << (a_t.lane_width - 1)):
                        signed = la - (1 << a_t.lane_width)
                    lv = (signed >> sh) & lane_mask
            out |= lv << off
        return out & _mask(a_t.total_width)

    if isinstance(expr, (SimdUlt, SimdUle, SimdUgt, SimdUge)):
        a_t = infer_type(expr.a, ctx_types)
        assert isinstance(a_t, SimdType)
        aa = _as_int(eval_expr(expr.a, ctx_types, env)) & _mask(a_t.total_width)
        bb = _as_int(eval_expr(expr.b, ctx_types, env)) & _mask(a_t.total_width)
        out = 0
        lane_mask = _mask(a_t.lane_width)
        for i in range(a_t.lanes):
            off = i * a_t.lane_width
            la = (aa >> off) & lane_mask
            lb = (bb >> off) & lane_mask
            if isinstance(expr, SimdUlt):
                ok = la < lb
            elif isinstance(expr, SimdUle):
                ok = la <= lb
            elif isinstance(expr, SimdUgt):
                ok = la > lb
            else:
                ok = la >= lb
            if ok:
                out |= 1 << i
        return out & _mask(a_t.lanes)

    if isinstance(expr, (SimdSlt, SimdSle, SimdSgt, SimdSge)):
        a_t = infer_type(expr.a, ctx_types)
        assert isinstance(a_t, SimdType)
        aa = _as_int(eval_expr(expr.a, ctx_types, env)) & _mask(a_t.total_width)
        bb = _as_int(eval_expr(expr.b, ctx_types, env)) & _mask(a_t.total_width)
        out = 0
        lane_mask = _mask(a_t.lane_width)
        sign_bit = 1 << (a_t.lane_width - 1)
        for i in range(a_t.lanes):
            off = i * a_t.lane_width
            la = (aa >> off) & lane_mask
            lb = (bb >> off) & lane_mask
            if la & sign_bit:
                la = la - (1 << a_t.lane_width)
            if lb & sign_bit:
                lb = lb - (1 << a_t.lane_width)
            if isinstance(expr, SimdSlt):
                ok = la < lb
            elif isinstance(expr, SimdSle):
                ok = la <= lb
            elif isinstance(expr, SimdSgt):
                ok = la > lb
            else:
                ok = la >= lb
            if ok:
                out |= 1 << i
        return out & _mask(a_t.lanes)

    if isinstance(expr, Mux):
        cond = eval_expr(expr.cond, ctx_types, env)
        chosen = expr.a if _as_bool(cond) else expr.b
        return eval_expr(chosen, ctx_types, env)

    if isinstance(expr, Concat):
        value = 0
        for part in expr.parts:
            t = infer_type(part, ctx_types)
            if isinstance(t, SimdType):
                raise TickIRValidationError("concat does not support simd parts")
            part_value = eval_expr(part, ctx_types, env)
            if isinstance(t, BoolType):
                w = 1
                pv = 1 if _as_bool(part_value) else 0
            else:
                assert isinstance(t, BitVecType)
                w = t.width
                pv = _as_int(part_value) & _mask(w)
            value = (value << w) | pv
        return value

    if isinstance(expr, Slice):
        src_t = infer_type(expr.x, ctx_types)
        assert isinstance(src_t, (BitVecType, SimdType))
        total = src_t.width if isinstance(src_t, BitVecType) else src_t.total_width
        x = eval_expr(expr.x, ctx_types, env)
        v = (_as_int(x) >> expr.offset) & _mask(expr.width) & _mask(total)
        if expr.width == 1:
            return bool(v & 1)
        return v

    if isinstance(expr, Lut8):
        idx = _as_int(eval_expr(expr.x, ctx_types, env)) & 0xFF
        return int(expr.table[idx]) & 0xFF

    from stc.tick_ir import TernaryLut

    if isinstance(expr, TernaryLut):
        a_val = _as_int(eval_expr(expr.a, ctx_types, env))
        b_val = _as_int(eval_expr(expr.b, ctx_types, env))
        c_val = _as_int(eval_expr(expr.c, ctx_types, env))

        a_t = infer_type(expr.a, ctx_types)
        if isinstance(a_t, BoolType):
            width = 1
        elif isinstance(a_t, BitVecType):
            width = a_t.width
        elif isinstance(a_t, SimdType):
            width = a_t.total_width
        else:
            raise TypeError(f"TernaryLut input has unsupported type {a_t}")

        result = 0
        for bit in range(width):
            a_bit = (a_val >> bit) & 1
            b_bit = (b_val >> bit) & 1
            c_bit = (c_val >> bit) & 1
            idx = (a_bit << 2) | (b_bit << 1) | c_bit
            out_bit = (expr.imm8 >> idx) & 1
            result |= out_bit << bit

        return result

    raise TypeError("unsupported expression type")


def reset_state(ir: TickIR) -> TickState:
    validate_tick_ir(ir)
    ctx_types = {**ir.inputs, **ir.state}
    state_values: dict[str, int | bool] = {}
    for name, t in ir.state.items():
        expr = ir.reset_state[name]
        value = eval_expr(expr, ctx_types, {})
        if isinstance(t, BoolType):
            state_values[name] = _as_bool(value)
        else:
            if isinstance(t, BitVecType):
                state_values[name] = _as_int(value) & _mask(t.width)
            elif isinstance(t, SimdType):
                state_values[name] = _as_int(value) & _mask(t.total_width)
            else:
                raise TypeError("unknown type")
    return TickState(state=state_values)


def tick(
    ir: TickIR, cur: TickState, inputs: dict[str, int | bool]
) -> tuple[TickState, dict[str, int | bool]]:
    validate_tick_ir(ir)
    ctx_types = {**ir.inputs, **ir.state}
    env: dict[str, int | bool] = {**inputs, **cur.state}
    outputs: dict[str, int | bool] = {}
    for name, expr in ir.output_exprs.items():
        outputs[name] = eval_expr(expr, ctx_types, env)

    next_state: dict[str, int | bool] = {}
    for name, expr in ir.next_state.items():
        next_state[name] = eval_expr(expr, ctx_types, env)

    committed: dict[str, int | bool] = {}
    for name, t in ir.state.items():
        value = next_state[name]
        if isinstance(t, BoolType):
            committed[name] = _as_bool(value)
        elif isinstance(t, BitVecType):
            committed[name] = _as_int(value) & _mask(t.width)
        elif isinstance(t, SimdType):
            committed[name] = _as_int(value) & _mask(t.total_width)
        else:
            raise TypeError("unknown type")

    return TickState(state=committed), outputs
