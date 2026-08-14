from __future__ import annotations

from collections import Counter
import os
import subprocess
import tempfile
from pathlib import Path
import time

from stc.interp import infer_type
from stc.tick_ir import (
    AShr,
    Add,
    And,
    Bitcast,
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
    FloatConst,
    FloatType,
    GemmCall,
    LShr,
    Mux,
    Mul,
    Not,
    Or,
    Shl,
    Slice,
    Lut8,
    SimdAdd,
    SimdAddMasked,
    SimdAddSatS,
    SimdAddSatU,
    SimdAShr,
    SimdConst,
    SimdEq,
    SimdExtractLane,
    SimdInsertLane,
    SimdLShr,
    SimdMaddS16,
    SimdDotU8S8AccI32,
    SimdBlend,
    SimdMulHiS,
    SimdMulHiU,
    SimdMulLo,
    SimdNot,
    SimdPackSS16To8,
    SimdPackSS32To16,
    SimdPackUS16To8,
    SimdMaskExpand,
    SimdMaskPack,
    SimdMaxS,
    SimdMaxU,
    SimdMinS,
    SimdMinU,
    SimdSExtLo,
    SimdAnd,
    SimdOr,
    SimdShl,
    SimdShuffle,
    SimdSplat,
    SimdSub,
    SimdSubMasked,
    SimdSubSatS,
    SimdSubSatU,
    SimdType,
    SimdUnpackHi,
    SimdUnpackLo,
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
    SimdZExtLo,
    SimdSge,
    SimdSgt,
    SimdSle,
    SimdSlt,
    Sub,
    TickIR,
    SimdUge,
    SimdUgt,
    SimdUle,
    SimdUlt,
    SimdXor,
    Uge,
    Ugt,
    Ule,
    Ult,
    Var,
    Xor,
)
from stc.dead_state import remove_dead_state
from stc.hashcons import hashcons_tick_ir
from stc.reachability import ReachabilityError, constant_state_within_bound
from stc.replace import replace_vars


def _ones(t: BitVecType | SimdType) -> BitVecConst | SimdConst:
    if isinstance(t, BitVecType):
        return BitVecConst(width=t.width, value=(1 << t.width) - 1)
    return SimdConst(
        lane_width=t.lane_width, lanes=t.lanes, value=(1 << t.total_width) - 1
    )


def _zero(t: BitVecType | SimdType) -> BitVecConst | SimdConst:
    if isinstance(t, BitVecType):
        return BitVecConst(width=t.width, value=0)
    return SimdConst(lane_width=t.lane_width, lanes=t.lanes, value=0)


def _mask(width: int) -> int:
    return (1 << int(width)) - 1


def reduce_expr(
    expr: Expr,
    types: dict[str, BoolType | BitVecType | FloatType | SimdType],
    _cache: dict[int, Expr] | None = None,
) -> Expr:
    if _cache is not None:
        expr_id = id(expr)
        if expr_id in _cache:
            return _cache[expr_id]

    result = _reduce_expr_impl(expr, types, _cache)

    if _cache is not None:
        _cache[expr_id] = result
    return result


def _reduce_expr_impl(
    expr: Expr,
    types: dict[str, BoolType | BitVecType | FloatType | SimdType],
    _cache: dict[int, Expr] | None,
) -> Expr:
    if isinstance(expr, (BoolConst, BitVecConst, FloatConst, SimdConst, Var)):
        return expr

    if isinstance(expr, Bitcast):
        x = reduce_expr(expr.x, types, _cache)
        return Bitcast(to=expr.to, x=x)

    from stc.tick_ir import Rotl, Rotr

    if isinstance(expr, (Rotl, Rotr)):
        x = reduce_expr(expr.x, types, _cache)
        sh = reduce_expr(expr.sh, types, _cache)
        return expr.__class__(x=x, sh=sh)

    if isinstance(expr, Lut8):
        x = reduce_expr(expr.x, types, _cache)
        if isinstance(x, BitVecConst) and x.width == 8:
            return BitVecConst(width=8, value=int(expr.table[int(x.value) & 0xFF]))
        return Lut8(x=x, table=list(expr.table))

    from stc.tick_ir import TernaryLut

    if isinstance(expr, TernaryLut):
        a = reduce_expr(expr.a, types, _cache)
        b = reduce_expr(expr.b, types, _cache)
        c = reduce_expr(expr.c, types, _cache)
        return TernaryLut(a=a, b=b, c=c, imm8=expr.imm8)

    if isinstance(expr, Not):
        x = reduce_expr(expr.x, types, _cache)
        if isinstance(x, BoolConst):
            return BoolConst(value=not x.value)
        if isinstance(x, BitVecConst):
            return BitVecConst(width=x.width, value=~x.value)
        if isinstance(x, SimdConst):
            return SimdConst(lane_width=x.lane_width, lanes=x.lanes, value=~x.value)
        if isinstance(x, Not):
            return x.x
        return Not(x=x)

    if isinstance(expr, And):
        a = reduce_expr(expr.a, types, _cache)
        b = reduce_expr(expr.b, types, _cache)
        t = infer_type(a, types)
        if isinstance(t, BoolType):
            if isinstance(a, BoolConst) and isinstance(b, BoolConst):
                return BoolConst(value=a.value and b.value)
            if isinstance(a, BoolConst):
                return b if a.value else BoolConst(value=False)
            if isinstance(b, BoolConst):
                return a if b.value else BoolConst(value=False)
            if a == b:
                return a
            return And(a=a, b=b)
        assert isinstance(t, (BitVecType, SimdType))
        if isinstance(t, BitVecType):
            if isinstance(a, BitVecConst) and isinstance(b, BitVecConst):
                return BitVecConst(width=t.width, value=a.value & b.value)
        else:
            assert isinstance(t, SimdType)
            if isinstance(a, SimdConst) and isinstance(b, SimdConst):
                return SimdConst(
                    lane_width=t.lane_width, lanes=t.lanes, value=a.value & b.value
                )
        if isinstance(a, (BitVecConst, SimdConst)) and int(a.value) == 0:
            return _zero(t)
        if isinstance(b, (BitVecConst, SimdConst)) and int(b.value) == 0:
            return _zero(t)
        all1 = _ones(t)
        if isinstance(a, (BitVecConst, SimdConst)) and int(a.value) == int(all1.value):
            return b
        if isinstance(b, (BitVecConst, SimdConst)) and int(b.value) == int(all1.value):
            return a
        if a == b:
            return a
        return And(a=a, b=b)

    if isinstance(expr, Or):
        a = reduce_expr(expr.a, types, _cache)
        b = reduce_expr(expr.b, types, _cache)
        t = infer_type(a, types)
        if isinstance(t, BoolType):
            if isinstance(a, BoolConst) and isinstance(b, BoolConst):
                return BoolConst(value=a.value or b.value)
            if isinstance(a, BoolConst):
                return BoolConst(value=True) if a.value else b
            if isinstance(b, BoolConst):
                return BoolConst(value=True) if b.value else a
            if a == b:
                return a
            return Or(a=a, b=b)
        assert isinstance(t, (BitVecType, SimdType))
        if isinstance(t, BitVecType):
            if isinstance(a, BitVecConst) and isinstance(b, BitVecConst):
                return BitVecConst(width=t.width, value=a.value | b.value)
        else:
            assert isinstance(t, SimdType)
            if isinstance(a, SimdConst) and isinstance(b, SimdConst):
                return SimdConst(
                    lane_width=t.lane_width, lanes=t.lanes, value=a.value | b.value
                )
        all1 = _ones(t)
        if isinstance(a, (BitVecConst, SimdConst)) and int(a.value) == int(all1.value):
            return all1
        if isinstance(b, (BitVecConst, SimdConst)) and int(b.value) == int(all1.value):
            return all1
        if isinstance(a, (BitVecConst, SimdConst)) and int(a.value) == 0:
            return b
        if isinstance(b, (BitVecConst, SimdConst)) and int(b.value) == 0:
            return a
        if a == b:
            return a
        return Or(a=a, b=b)

    if isinstance(expr, Xor):
        a = reduce_expr(expr.a, types, _cache)
        b = reduce_expr(expr.b, types, _cache)
        t = infer_type(a, types)
        if isinstance(t, BoolType):
            if isinstance(a, BoolConst) and isinstance(b, BoolConst):
                return BoolConst(value=(a.value ^ b.value))
            if isinstance(a, BoolConst):
                return Not(x=b) if a.value else b
            if isinstance(b, BoolConst):
                return Not(x=a) if b.value else a
            if a == b:
                return BoolConst(value=False)
            return Xor(a=a, b=b)
        assert isinstance(t, (BitVecType, SimdType))
        if isinstance(t, SimdType):
            if isinstance(a, SimdConst) and isinstance(b, SimdConst):
                return SimdConst(
                    lane_width=t.lane_width, lanes=t.lanes, value=a.value ^ b.value
                )
            if isinstance(a, SimdConst) and int(a.value) == 0:
                return b
            if isinstance(b, SimdConst) and int(b.value) == 0:
                return a
            all1 = _ones(t)
            if isinstance(a, SimdConst) and int(a.value) == int(all1.value):
                return Not(x=b)
            if isinstance(b, SimdConst) and int(b.value) == int(all1.value):
                return Not(x=a)
            if a == b:
                return _zero(t)
            return Xor(a=a, b=b)

        if isinstance(a, BitVecConst) and isinstance(b, BitVecConst):
            return BitVecConst(width=t.width, value=a.value ^ b.value)

        def gather(x: Expr, out: list[Expr]) -> None:
            if isinstance(x, Xor):
                gather(x.a, out)
                gather(x.b, out)
            else:
                out.append(x)

        terms: list[Expr] = []
        gather(a, terms)
        gather(b, terms)

        c = 0
        nonconst: list[Expr] = []
        for term in terms:
            if isinstance(term, BitVecConst):
                c ^= int(term.value) & ((1 << t.width) - 1)
            else:
                nonconst.append(term)

        counts = Counter(nonconst)
        kept = [term for term, n in counts.items() if n % 2 == 1 and term != _zero(t)]
        kept_sorted = sorted(kept, key=repr)

        out_terms: list[Expr] = []
        if c != 0:
            out_terms.append(BitVecConst(width=t.width, value=c))
        out_terms.extend(kept_sorted)

        if not out_terms:
            return BitVecConst(width=t.width, value=0)
        if len(out_terms) == 1:
            return out_terms[0]
        cur = out_terms[0]
        for term in out_terms[1:]:
            cur = Xor(a=cur, b=term)
        return cur

    if isinstance(expr, Mux):
        cond = reduce_expr(expr.cond, types, _cache)
        a = reduce_expr(expr.a, types, _cache)
        b = reduce_expr(expr.b, types, _cache)
        if isinstance(cond, BoolConst):
            return a if cond.value else b
        if a == b:
            return a

        return Mux(cond=cond, a=a, b=b)

    if isinstance(expr, (FNeg, FAbs, FSqrt)):
        x = reduce_expr(expr.x, types, _cache)
        return expr.__class__(x=x)

    if isinstance(expr, (FAdd, FMul, FSub, FDiv, FEq, FLt, FLe, FNe)):
        a = reduce_expr(expr.a, types, _cache)
        b = reduce_expr(expr.b, types, _cache)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, Concat):
        parts = [reduce_expr(p, types, _cache) for p in expr.parts]
        return Concat(parts=parts)

    if isinstance(expr, Slice):
        x = reduce_expr(expr.x, types, _cache)
        return Slice(x=x, offset=expr.offset, width=expr.width)

    if isinstance(expr, (Add, Sub, Mul, Div, Shl, LShr, AShr, Eq, Ult, Ule, Ugt, Uge)):
        a = reduce_expr(expr.a, types, _cache)
        b = reduce_expr(expr.b, types, _cache)
        t = infer_type(a, types)
        if isinstance(expr, Eq):
            if a == b:
                return BoolConst(value=True)
            if isinstance(t, BoolType):
                if isinstance(a, BoolConst) and isinstance(b, BoolConst):
                    return BoolConst(value=bool(a.value) == bool(b.value))
                return Eq(a=a, b=b)
            if isinstance(t, BitVecType):
                if isinstance(a, BitVecConst) and isinstance(b, BitVecConst):
                    return BoolConst(
                        value=(int(a.value) & _mask(t.width))
                        == (int(b.value) & _mask(t.width))
                    )
                return Eq(a=a, b=b)
            if isinstance(t, SimdType):
                if isinstance(a, SimdConst) and isinstance(b, SimdConst):
                    return BoolConst(
                        value=(int(a.value) & _mask(t.total_width))
                        == (int(b.value) & _mask(t.total_width))
                    )
                return Eq(a=a, b=b)
            raise TypeError("unknown type")

        if isinstance(expr, (Ult, Ule, Ugt, Uge)):
            if a == b:
                if isinstance(expr, (Ule, Uge)):
                    return BoolConst(value=True)
                return BoolConst(value=False)
            if not isinstance(t, BitVecType):
                return expr.__class__(a=a, b=b)
            if isinstance(a, BitVecConst) and isinstance(b, BitVecConst):
                aa = int(a.value) & _mask(t.width)
                bb = int(b.value) & _mask(t.width)
                if isinstance(expr, Ult):
                    ok = aa < bb
                elif isinstance(expr, Ule):
                    ok = aa <= bb
                elif isinstance(expr, Ugt):
                    ok = aa > bb
                else:
                    ok = aa >= bb
                return BoolConst(value=ok)
            return expr.__class__(a=a, b=b)

        if not isinstance(t, BitVecType):
            return expr.__class__(a=a, b=b)
        if isinstance(a, BitVecConst) and isinstance(b, BitVecConst):
            aa = int(a.value) & _mask(t.width)
            bb = int(b.value) & _mask(t.width)
            if isinstance(expr, Add):
                return BitVecConst(width=t.width, value=aa + bb)
            if isinstance(expr, Sub):
                return BitVecConst(width=t.width, value=aa - bb)
            if isinstance(expr, Mul):
                return BitVecConst(width=t.width, value=aa * bb)
            if isinstance(expr, Div):
                return BitVecConst(width=t.width, value=0 if bb == 0 else aa // bb)
            if isinstance(expr, Shl):
                return BitVecConst(width=t.width, value=(aa << bb) & _mask(t.width))
            if isinstance(expr, LShr):
                return BitVecConst(width=t.width, value=(aa >> bb) & _mask(t.width))
            if isinstance(expr, AShr):
                if bb >= t.width:
                    return BitVecConst(
                        width=t.width,
                        value=_mask(t.width) if (aa & (1 << (t.width - 1))) else 0,
                    )
                signed = aa
                if aa & (1 << (t.width - 1)):
                    signed = aa - (1 << t.width)
                return BitVecConst(width=t.width, value=(signed >> bb) & _mask(t.width))
        return expr.__class__(a=a, b=b)

    if isinstance(expr, SimdAdd):
        a = reduce_expr(expr.a, types, _cache)
        b = reduce_expr(expr.b, types, _cache)
        return SimdAdd(a=a, b=b)

    if isinstance(expr, SimdSub):
        a = reduce_expr(expr.a, types, _cache)
        b = reduce_expr(expr.b, types, _cache)
        return SimdSub(a=a, b=b)

    if isinstance(expr, (SimdAddMasked, SimdSubMasked)):
        mask = reduce_expr(expr.mask, types, _cache)
        passthru = reduce_expr(expr.passthru, types, _cache)
        a = reduce_expr(expr.a, types, _cache)
        b = reduce_expr(expr.b, types, _cache)

        if isinstance(mask, SimdConst) and mask.lane_width == 1:
            if mask.value == 0:
                return passthru
            t = infer_type(a, types)
            if isinstance(t, SimdType) and mask.value == (1 << t.lanes) - 1:
                return (
                    SimdAdd(a=a, b=b)
                    if isinstance(expr, SimdAddMasked)
                    else SimdSub(a=a, b=b)
                )

        return expr.__class__(mask=mask, passthru=passthru, a=a, b=b)

    if isinstance(expr, (SimdFSqrt, SimdFNeg, SimdFAbs)):
        x = reduce_expr(expr.x, types, _cache)
        return expr.__class__(x=x)

    if isinstance(
        expr,
        (
            SimdFAdd,
            SimdFMul,
            SimdFSub,
            SimdFDiv,
            SimdFCmpEq,
            SimdFCmpLt,
            SimdFCmpLe,
            SimdFCmpNe,
        ),
    ):
        a = reduce_expr(expr.a, types, _cache)
        b = reduce_expr(expr.b, types, _cache)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, SimdFFma):
        a = reduce_expr(expr.a, types, _cache)
        b = reduce_expr(expr.b, types, _cache)
        c = reduce_expr(expr.c, types, _cache)
        return SimdFFma(a=a, b=b, c=c)

    if isinstance(expr, (SimdAddSatU, SimdSubSatU, SimdAddSatS, SimdSubSatS)):
        a = reduce_expr(expr.a, types, _cache)
        b = reduce_expr(expr.b, types, _cache)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, (SimdMulLo, SimdMulHiU, SimdMulHiS, SimdMaddS16)):
        a = reduce_expr(expr.a, types, _cache)
        b = reduce_expr(expr.b, types, _cache)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, SimdDotU8S8AccI32):
        a = reduce_expr(expr.a, types, _cache)
        b = reduce_expr(expr.b, types, _cache)
        acc = reduce_expr(expr.acc, types, _cache)
        return SimdDotU8S8AccI32(a=a, b=b, acc=acc)

    if isinstance(expr, GemmCall):
        return GemmCall(
            a=reduce_expr(expr.a, types, _cache),
            b=reduce_expr(expr.b, types, _cache),
            m=expr.m,
            n=expr.n,
            k=expr.k,
            a_width=expr.a_width,
            b_width=expr.b_width,
            acc_width=expr.acc_width,
            a_signed=expr.a_signed,
            b_signed=expr.b_signed,
            layout=expr.layout,
        )

    if isinstance(
        expr,
        (
            SimdUnpackLo,
            SimdUnpackHi,
            SimdPackSS16To8,
            SimdPackUS16To8,
            SimdPackSS32To16,
        ),
    ):
        a = reduce_expr(expr.a, types, _cache)
        b = reduce_expr(expr.b, types, _cache)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, SimdMaskExpand):
        x = reduce_expr(expr.x, types, _cache)
        return SimdMaskExpand(to=expr.to, x=x)

    if isinstance(expr, SimdMaskPack):
        x = reduce_expr(expr.x, types, _cache)
        if isinstance(x, SimdMaskExpand):
            return x.x
        return SimdMaskPack(x=x)

    if isinstance(expr, (SimdMinU, SimdMaxU, SimdMinS, SimdMaxS)):
        a = reduce_expr(expr.a, types, _cache)
        b = reduce_expr(expr.b, types, _cache)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, SimdBlend):
        m = reduce_expr(expr.mask, types, _cache)
        a = reduce_expr(expr.a, types, _cache)
        b = reduce_expr(expr.b, types, _cache)
        return SimdBlend(mask=m, a=a, b=b)

    if isinstance(expr, (SimdZExtLo, SimdSExtLo)):
        x = reduce_expr(expr.x, types, _cache)
        return expr.__class__(to=expr.to, x=x)

    if isinstance(expr, SimdEq):
        a = reduce_expr(expr.a, types, _cache)
        b = reduce_expr(expr.b, types, _cache)
        return SimdEq(a=a, b=b)

    if isinstance(expr, SimdNot):
        x = reduce_expr(expr.x, types, _cache)
        return SimdNot(x=x)

    if isinstance(expr, (SimdAnd, SimdOr, SimdXor)):
        a = reduce_expr(expr.a, types, _cache)
        b = reduce_expr(expr.b, types, _cache)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, SimdShl):
        a = reduce_expr(expr.a, types, _cache)
        sh = reduce_expr(expr.sh, types, _cache)
        return SimdShl(a=a, sh=sh)

    if isinstance(expr, SimdLShr):
        a = reduce_expr(expr.a, types, _cache)
        sh = reduce_expr(expr.sh, types, _cache)
        return SimdLShr(a=a, sh=sh)

    if isinstance(expr, SimdAShr):
        a = reduce_expr(expr.a, types, _cache)
        sh = reduce_expr(expr.sh, types, _cache)
        return SimdAShr(a=a, sh=sh)

    if isinstance(expr, (SimdUlt, SimdUle, SimdUgt, SimdUge)):
        a = reduce_expr(expr.a, types, _cache)
        b = reduce_expr(expr.b, types, _cache)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, (SimdSlt, SimdSle, SimdSgt, SimdSge)):
        a = reduce_expr(expr.a, types, _cache)
        b = reduce_expr(expr.b, types, _cache)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, SimdSplat):
        x = reduce_expr(expr.x, types, _cache)
        return SimdSplat(to=expr.to, x=x)

    if isinstance(expr, SimdExtractLane):
        x = reduce_expr(expr.x, types, _cache)
        return SimdExtractLane(x=x, lane=expr.lane)

    if isinstance(expr, SimdInsertLane):
        x = reduce_expr(expr.x, types, _cache)
        v = reduce_expr(expr.value, types, _cache)
        return SimdInsertLane(x=x, lane=expr.lane, value=v)

    if isinstance(expr, SimdShuffle):
        x = reduce_expr(expr.x, types, _cache)
        src_t = infer_type(x, types)
        assert isinstance(src_t, SimdType)
        indices = list(expr.indices)

        if len(indices) == src_t.lanes and indices == list(range(src_t.lanes)):
            return x

        if len(indices) == src_t.lanes and src_t.lanes % 2 == 0:
            half = src_t.lanes // 2
            dup_lo = [i for i in range(half) for _ in (0, 1)]
            if indices == dup_lo:
                return SimdUnpackLo(a=x, b=x)
            dup_hi = [i for i in range(half, src_t.lanes) for _ in (0, 1)]
            if indices == dup_hi:
                return SimdUnpackHi(a=x, b=x)

        return SimdShuffle(x=x, indices=indices)

    raise TypeError("unsupported expression type")


def reduce_tick_ir(ir: TickIR) -> TickIR:
    cache: dict[int, Expr] = {}
    types = {**ir.inputs, **ir.state}
    reduced_next = {k: reduce_expr(v, types, cache) for k, v in ir.next_state.items()}
    reduced_out = {k: reduce_expr(v, types, cache) for k, v in ir.output_exprs.items()}
    return TickIR(
        name=ir.name,
        inputs=dict(ir.inputs),
        outputs=dict(ir.outputs),
        state=dict(ir.state),
        reset_state=dict(ir.reset_state),
        next_state=reduced_next,
        output_exprs=reduced_out,
    )


def _find_rust_reduce_bin() -> Path | None:
    env = os.environ.get("STC_RUST_REDUCE_BIN")
    if env:
        p = Path(env)
        return p if p.exists() else None
    for candidate in [
        Path("rust/tick_reduce_rs/target/release/tick_reduce_rs"),
        Path("rust/tick_reduce_rs/target/debug/tick_reduce_rs"),
    ]:
        if candidate.exists():
            return candidate
    return None


def rust_reduce_tick_ir(ir: TickIR) -> TickIR | None:
    if os.environ.get("STC_RUST_REDUCE", "1").lower() in {"0", "false", "no"}:
        return None
    bin_path = _find_rust_reduce_bin()
    if bin_path is None:
        return None
    fmt_env = os.environ.get("STC_RUST_REDUCE_FORMAT")
    fmt = fmt_env.lower() if fmt_env else "bin"
    if fmt != "bin":
        return None
    try:
        from stc.tick_ir_bin2 import write_tick_ir_bin
    except Exception:
        return None
    timing_enabled = os.environ.get("STC_TIMING", "0").lower() not in {
        "0",
        "false",
        "no",
    }

    def _log(label: str, start: float) -> None:
        if timing_enabled:
            elapsed = time.perf_counter() - start
            print(f"[timing] rust_reduce:{label}: {elapsed:.3f}s", flush=True)

    with tempfile.TemporaryDirectory(prefix="stc_rust_reduce_") as td:
        td_path = Path(td)
        in_path = td_path / "in.bin"
        out_path = td_path / "out.bin"
        if timing_enabled:
            print("[timing] rust_reduce:start", flush=True)
        t0 = time.perf_counter()
        write_tick_ir_bin(ir, str(in_path))
        _log("write_input_bin", t0)
        try:
            t0 = time.perf_counter()
            subprocess.run(
                [
                    str(bin_path),
                    "--input",
                    str(in_path),
                    "--output",
                    str(out_path),
                    "--format",
                    fmt,
                ],
                check=True,
                capture_output=not timing_enabled,
                text=True,
            )
            _log("subprocess", t0)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        try:
            t0 = time.perf_counter()
            from stc.tick_ir_bin2 import read_tick_ir_bin

            reduced = read_tick_ir_bin(str(out_path))
            _log("read_output_bin", t0)
            return reduced
        except Exception:
            return None
    return None


def optimize_tick_ir(
    ir: TickIR,
    *,
    bound: int,
    autovec: bool = False,
    autovec_timeout_ms: int = 2000,
    superopt: bool = False,
    superopt_max_nodes: int = 6,
    superopt_timeout_ms: int = 2000,
    backend: str = "generic",
    ternary_mapping: bool | None = None,
    bounded_state_opt: bool = True,
    fuse_ticks: int = 1,
    fuse_mode: str = "final",
    fuse_input_policy: str = "shared",
    fuse_budget_max_nodes: int = 1 << 60,
    fuse_budget_max_depth: int = 1 << 60,
    fuse_budget_max_step_ms: int | None = None,
    arith_classify: bool = True,
    mul_div_max_width: int = 32,
) -> tuple[TickIR, dict[str, int] | None]:
    from stc.delay_lower import lower_delays
    from stc.tech import get_technology
    from stc.mapping.ternary import TernaryMappingPass
    from stc.passmgr import PassContext

    timing_enabled = os.environ.get("STC_TIMING", "0").lower() not in {
        "0",
        "false",
        "no",
    }

    def _log_timing(label: str, start: float) -> None:
        if timing_enabled:
            elapsed = time.perf_counter() - start
            print(f"[timing] optimize_tick_ir:{label}: {elapsed:.3f}s")

    t0 = time.perf_counter()
    ir = lower_delays(ir)
    _log_timing("lower_delays", t0)
    t0 = time.perf_counter()
    rust_reduced = rust_reduce_tick_ir(ir)
    ir = rust_reduced if rust_reduced is not None else reduce_tick_ir(ir)
    _log_timing("reduce_tick_ir", t0)
    if fuse_ticks is not None and fuse_ticks > 1:
        from stc.fuse_ticks import FuseBudget, fuse_ticks as _fuse_ticks

        t0 = time.perf_counter()
        ir = _fuse_ticks(
            ir,
            fuse_ticks,
            mode=fuse_mode,
            input_policy=fuse_input_policy,
            budget=FuseBudget(
                max_nodes=fuse_budget_max_nodes,
                max_depth=fuse_budget_max_depth,
                stop_on_budget_hit=True,
                max_step_ms=fuse_budget_max_step_ms,
            ),
        )
        _log_timing("fuse_ticks", t0)
        t0 = time.perf_counter()
        rust_reduced = rust_reduce_tick_ir(ir)
        ir = rust_reduced if rust_reduced is not None else reduce_tick_ir(ir)
        _log_timing("reduce_tick_ir_post_fuse", t0)

    tech = get_technology(backend)
    ctx = PassContext(
        technology=tech,
        cost_model=tech.cost_model(),
        depth_model=tech.depth_model(),
    )

    ternary_pass = TernaryMappingPass()
    should_run_ternary = (
        ternary_mapping if ternary_mapping is not None else ternary_pass.should_run(ctx)
    )
    if should_run_ternary:
        t0 = time.perf_counter()
        ir, _ = ternary_pass.run(ir, ctx)
        _log_timing("ternary_mapping", t0)
    if autovec:
        from stc.autovec_pass import autovectorize_tick_ir

        t0 = time.perf_counter()
        ir = autovectorize_tick_ir(ir, timeout_ms=autovec_timeout_ms)
        _log_timing("autovec", t0)
    if superopt:
        from stc.superopt import SuperoptError, superopt_expr

        t0 = time.perf_counter()
        types = {**ir.inputs, **ir.state}

        def go(e: Expr) -> Expr:
            try:
                return superopt_expr(
                    e,
                    types,
                    max_nodes=superopt_max_nodes,
                    timeout_ms=superopt_timeout_ms,
                )
            except (SuperoptError, Exception):
                return e

        ir = TickIR(
            name=ir.name,
            inputs=dict(ir.inputs),
            outputs=dict(ir.outputs),
            state=dict(ir.state),
            reset_state=dict(ir.reset_state),
            next_state={k: go(v) for k, v in ir.next_state.items()},
            output_exprs={k: go(v) for k, v in ir.output_exprs.items()},
        )
        _log_timing("superopt", t0)

    arith_report = None
    if arith_classify and os.environ.get("STC_ARITH_CLASSIFY", "1") not in {
        "0",
        "false",
        "no",
    }:
        from stc.tick_ir_classify_arith import classify_arithmetic

        t0 = time.perf_counter()
        ir, arith_report = classify_arithmetic(ir)
        _log_timing("arith_classify", t0)

    if not bounded_state_opt:
        if os.environ.get("STC_SKIP_HASHCONS", "0") not in {"0", "false", "no"}:
            return ir, arith_report
        t0 = time.perf_counter()
        ir = hashcons_tick_ir(ir)
        _log_timing("hashcons_tick_ir", t0)
        return ir, arith_report

    t0 = time.perf_counter()
    ir = remove_dead_state(ir, bound)
    _log_timing("remove_dead_state", t0)
    t0 = time.perf_counter()
    try:
        consts = constant_state_within_bound(ir, bound)
    except ReachabilityError:
        if os.environ.get("STC_SKIP_HASHCONS", "0") not in {"0", "false", "no"}:
            return ir, arith_report
        t1 = time.perf_counter()
        ir = hashcons_tick_ir(ir)
        _log_timing("hashcons_tick_ir", t1)
        return ir, arith_report
    _log_timing("constant_state_within_bound", t0)

    if not consts:
        if os.environ.get("STC_SKIP_HASHCONS", "0") not in {"0", "false", "no"}:
            return ir, arith_report
        t0 = time.perf_counter()
        ir = hashcons_tick_ir(ir)
        _log_timing("hashcons_tick_ir", t0)
        return ir, arith_report

    repl: dict[str, Expr] = {}
    for name, value in consts.items():
        t = ir.state[name]
        if isinstance(t, BoolType):
            repl[name] = BoolConst(value=bool(value))
        elif isinstance(t, BitVecType):
            repl[name] = BitVecConst(width=t.width, value=int(value))
        else:
            assert isinstance(t, SimdType)
            repl[name] = SimdConst(
                lane_width=t.lane_width,
                lanes=t.lanes,
                value=int(value),
            )

    new_output_exprs = {k: replace_vars(v, repl) for k, v in ir.output_exprs.items()}
    new_next_state = {k: replace_vars(v, repl) for k, v in ir.next_state.items()}

    removed = set(consts.keys())
    new_state = {k: v for k, v in ir.state.items() if k not in removed}
    new_reset = {k: v for k, v in ir.reset_state.items() if k not in removed}
    new_next = {k: v for k, v in new_next_state.items() if k not in removed}

    ir = TickIR(
        name=ir.name,
        inputs=dict(ir.inputs),
        outputs=dict(ir.outputs),
        state=new_state,
        reset_state=new_reset,
        next_state=new_next,
        output_exprs=new_output_exprs,
    )

    if os.environ.get("STC_SKIP_HASHCONS", "0") not in {"0", "false", "no"}:
        return ir, arith_report
    t0 = time.perf_counter()
    ir = hashcons_tick_ir(ir)
    _log_timing("hashcons_tick_ir", t0)
    return ir, arith_report


def fold_constants(ir: TickIR) -> tuple[TickIR, bool]:
    """Fold constants in IR. Returns (new_ir, changed)."""
    new_ir = reduce_tick_ir(ir)
    changed = new_ir != ir
    return new_ir, changed


def canonicalize_ir(ir: TickIR) -> tuple[TickIR, bool]:
    """Canonicalize IR (commutativity, associativity). Returns (new_ir, changed)."""
    new_ir = reduce_tick_ir(ir)
    changed = new_ir != ir
    return new_ir, changed


def eliminate_dead_state(ir: TickIR, bound: int = 100) -> TickIR:
    """Eliminate dead state variables. Wrapper for remove_dead_state."""
    return remove_dead_state(ir, bound)


from stc.passmgr import Pass, PassContext, PassMetrics, PassManager, PassSchedule
from stc.tech import Technology, get_technology


class ConstFoldPass(Pass):
    """Constant folding and simplification."""

    @property
    def name(self) -> str:
        return "const-fold"

    def run(self, ir: TickIR, ctx: PassContext) -> tuple[TickIR, PassMetrics]:
        new_ir, changed = fold_constants(ir)
        return new_ir, PassMetrics(changed=changed)


class CanonicalizePass(Pass):
    """Canonicalization (commutativity, associativity normalization)."""

    @property
    def name(self) -> str:
        return "canonicalize"

    def run(self, ir: TickIR, ctx: PassContext) -> tuple[TickIR, PassMetrics]:
        new_ir, changed = canonicalize_ir(ir)
        return new_ir, PassMetrics(changed=changed)


class DeadCodePass(Pass):
    """Dead code elimination."""

    def __init__(self, bound: int = 100):
        self.bound = bound

    @property
    def name(self) -> str:
        return "dce"

    def run(self, ir: TickIR, ctx: PassContext) -> tuple[TickIR, PassMetrics]:
        new_ir = eliminate_dead_state(ir, self.bound)
        changed = new_ir != ir
        return new_ir, PassMetrics(changed=changed)


def build_default_schedule(
    include_superopt: bool = False,
    use_ternary_mapping: bool = True,
    use_depth_aware_ternary: bool = False,
    depth_budget: int | None = None,
) -> PassSchedule:
    """Build the default optimization schedule.

    Args:
        include_superopt: Include superoptimization passes
        use_ternary_mapping: Include standard ternary mapping
        use_depth_aware_ternary: Include depth-aware ternary mapping (reduces depth)
        depth_budget: Maximum allowed circuit depth (enforced via DepthBudgetPass)
    """
    schedule = PassSchedule(max_iterations=50)
    schedule.add(CanonicalizePass())
    schedule.add(ConstFoldPass())
    schedule.add(DeadCodePass())

    if use_depth_aware_ternary:
        from stc.passes.depth_ternary import DepthAwareTernaryPass

        schedule.add(DepthAwareTernaryPass())

    if use_ternary_mapping and not use_depth_aware_ternary:
        from stc.mapping.ternary import TernaryMappingPass

        schedule.add(TernaryMappingPass())

    if depth_budget is not None:
        from stc.passes.depth_ternary import DepthBudgetPass

        schedule.add(DepthBudgetPass())

    if include_superopt:
        from stc.superopt import SuperoptPass

        schedule.add(SuperoptPass())

    return schedule


def optimize_tick_ir_passmanager(
    ir: TickIR,
    technology: str = "generic",
    include_superopt: bool = False,
    depth_budget: int | None = None,
    use_depth_aware_ternary: bool = False,
    verbosity: int = 0,
) -> TickIR:
    """Optimization entry point using pass manager architecture.

    Args:
        ir: Input TickIR to optimize
        technology: Target technology name
        include_superopt: Include superoptimization passes
        depth_budget: Maximum allowed circuit depth
        use_depth_aware_ternary: Use depth-aware ternary mapping to reduce depth
        verbosity: Logging verbosity level
    """
    tech = get_technology(technology)
    ctx = PassContext(
        technology=tech,
        cost_model=tech.cost_model(),
        depth_model=tech.depth_model(),
        depth_budget=depth_budget,
        verbosity=verbosity,
    )

    use_ternary = depth_budget is None
    schedule = build_default_schedule(
        include_superopt=include_superopt,
        use_ternary_mapping=use_ternary,
        use_depth_aware_ternary=use_depth_aware_ternary or depth_budget is not None,
        depth_budget=depth_budget,
    )
    mgr = PassManager(schedule)

    result_ir, metrics = mgr.run(ir, ctx)
    return result_ir
