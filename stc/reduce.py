from __future__ import annotations

from collections import Counter

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
    LShr,
    Mux,
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
    expr: Expr, types: dict[str, BoolType | BitVecType | FloatType | SimdType]
) -> Expr:
    if isinstance(expr, (BoolConst, BitVecConst, FloatConst, SimdConst, Var)):
        return expr

    if isinstance(expr, Bitcast):
        x = reduce_expr(expr.x, types)
        return Bitcast(to=expr.to, x=x)

    if isinstance(expr, Lut8):
        x = reduce_expr(expr.x, types)
        if isinstance(x, BitVecConst) and x.width == 8:
            return BitVecConst(width=8, value=int(expr.table[int(x.value) & 0xFF]))
        return Lut8(x=x, table=list(expr.table))

    if isinstance(expr, Not):
        x = reduce_expr(expr.x, types)
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
        a = reduce_expr(expr.a, types)
        b = reduce_expr(expr.b, types)
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
        a = reduce_expr(expr.a, types)
        b = reduce_expr(expr.b, types)
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
        a = reduce_expr(expr.a, types)
        b = reduce_expr(expr.b, types)
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
        cond = reduce_expr(expr.cond, types)
        a = reduce_expr(expr.a, types)
        b = reduce_expr(expr.b, types)
        if isinstance(cond, BoolConst):
            return a if cond.value else b
        if a == b:
            return a
        return Mux(cond=cond, a=a, b=b)

    if isinstance(expr, (FNeg, FAbs, FSqrt)):
        x = reduce_expr(expr.x, types)
        return expr.__class__(x=x)

    if isinstance(expr, (FAdd, FMul, FSub, FDiv, FEq, FLt, FLe, FNe)):
        a = reduce_expr(expr.a, types)
        b = reduce_expr(expr.b, types)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, Concat):
        parts = [reduce_expr(p, types) for p in expr.parts]
        return Concat(parts=parts)

    if isinstance(expr, Slice):
        x = reduce_expr(expr.x, types)
        return Slice(x=x, offset=expr.offset, width=expr.width)

    if isinstance(expr, (Add, Sub, Shl, LShr, AShr, Eq, Ult, Ule, Ugt, Uge)):
        a = reduce_expr(expr.a, types)
        b = reduce_expr(expr.b, types)
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
        a = reduce_expr(expr.a, types)
        b = reduce_expr(expr.b, types)
        return SimdAdd(a=a, b=b)

    if isinstance(expr, SimdSub):
        a = reduce_expr(expr.a, types)
        b = reduce_expr(expr.b, types)
        return SimdSub(a=a, b=b)

    if isinstance(expr, (SimdAddMasked, SimdSubMasked)):
        mask = reduce_expr(expr.mask, types)
        passthru = reduce_expr(expr.passthru, types)
        a = reduce_expr(expr.a, types)
        b = reduce_expr(expr.b, types)

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
        x = reduce_expr(expr.x, types)
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
        a = reduce_expr(expr.a, types)
        b = reduce_expr(expr.b, types)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, SimdFFma):
        a = reduce_expr(expr.a, types)
        b = reduce_expr(expr.b, types)
        c = reduce_expr(expr.c, types)
        return SimdFFma(a=a, b=b, c=c)

    if isinstance(expr, (SimdAddSatU, SimdSubSatU, SimdAddSatS, SimdSubSatS)):
        a = reduce_expr(expr.a, types)
        b = reduce_expr(expr.b, types)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, (SimdMulLo, SimdMulHiU, SimdMulHiS, SimdMaddS16)):
        a = reduce_expr(expr.a, types)
        b = reduce_expr(expr.b, types)
        return expr.__class__(a=a, b=b)

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
        a = reduce_expr(expr.a, types)
        b = reduce_expr(expr.b, types)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, SimdMaskExpand):
        x = reduce_expr(expr.x, types)
        return SimdMaskExpand(to=expr.to, x=x)

    if isinstance(expr, SimdMaskPack):
        x = reduce_expr(expr.x, types)
        if isinstance(x, SimdMaskExpand):
            return x.x
        return SimdMaskPack(x=x)

    if isinstance(expr, (SimdMinU, SimdMaxU, SimdMinS, SimdMaxS)):
        a = reduce_expr(expr.a, types)
        b = reduce_expr(expr.b, types)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, SimdBlend):
        m = reduce_expr(expr.mask, types)
        a = reduce_expr(expr.a, types)
        b = reduce_expr(expr.b, types)
        return SimdBlend(mask=m, a=a, b=b)

    if isinstance(expr, (SimdZExtLo, SimdSExtLo)):
        x = reduce_expr(expr.x, types)
        return expr.__class__(to=expr.to, x=x)

    if isinstance(expr, SimdEq):
        a = reduce_expr(expr.a, types)
        b = reduce_expr(expr.b, types)
        return SimdEq(a=a, b=b)

    if isinstance(expr, SimdNot):
        x = reduce_expr(expr.x, types)
        return SimdNot(x=x)

    if isinstance(expr, (SimdAnd, SimdOr, SimdXor)):
        a = reduce_expr(expr.a, types)
        b = reduce_expr(expr.b, types)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, SimdShl):
        a = reduce_expr(expr.a, types)
        sh = reduce_expr(expr.sh, types)
        return SimdShl(a=a, sh=sh)

    if isinstance(expr, SimdLShr):
        a = reduce_expr(expr.a, types)
        sh = reduce_expr(expr.sh, types)
        return SimdLShr(a=a, sh=sh)

    if isinstance(expr, SimdAShr):
        a = reduce_expr(expr.a, types)
        sh = reduce_expr(expr.sh, types)
        return SimdAShr(a=a, sh=sh)

    if isinstance(expr, (SimdUlt, SimdUle, SimdUgt, SimdUge)):
        a = reduce_expr(expr.a, types)
        b = reduce_expr(expr.b, types)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, (SimdSlt, SimdSle, SimdSgt, SimdSge)):
        a = reduce_expr(expr.a, types)
        b = reduce_expr(expr.b, types)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, SimdSplat):
        x = reduce_expr(expr.x, types)
        return SimdSplat(to=expr.to, x=x)

    if isinstance(expr, SimdExtractLane):
        x = reduce_expr(expr.x, types)
        return SimdExtractLane(x=x, lane=expr.lane)

    if isinstance(expr, SimdInsertLane):
        x = reduce_expr(expr.x, types)
        v = reduce_expr(expr.value, types)
        return SimdInsertLane(x=x, lane=expr.lane, value=v)

    if isinstance(expr, SimdShuffle):
        x = reduce_expr(expr.x, types)
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
    types = {**ir.inputs, **ir.state}
    reduced_next = {k: reduce_expr(v, types) for k, v in ir.next_state.items()}
    reduced_out = {k: reduce_expr(v, types) for k, v in ir.output_exprs.items()}
    return TickIR(
        name=ir.name,
        inputs=dict(ir.inputs),
        outputs=dict(ir.outputs),
        state=dict(ir.state),
        reset_state=dict(ir.reset_state),
        next_state=reduced_next,
        output_exprs=reduced_out,
    )


def optimize_tick_ir(
    ir: TickIR,
    *,
    bound: int,
    autovec: bool = False,
    autovec_timeout_ms: int = 200,
    superopt: bool = False,
    superopt_max_nodes: int = 6,
    superopt_timeout_ms: int = 200,
) -> TickIR:
    from stc.delay_lower import lower_delays

    ir = lower_delays(ir)
    ir = reduce_tick_ir(ir)
    if autovec:
        from stc.autovec_pass import autovectorize_tick_ir

        ir = autovectorize_tick_ir(ir, timeout_ms=autovec_timeout_ms)
    if superopt:
        from stc.superopt import SuperoptError, superopt_expr

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
    ir = remove_dead_state(ir, bound)
    try:
        consts = constant_state_within_bound(ir, bound)
    except ReachabilityError:
        return hashcons_tick_ir(ir)

    if not consts:
        return hashcons_tick_ir(ir)

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

    return hashcons_tick_ir(
        TickIR(
            name=ir.name,
            inputs=dict(ir.inputs),
            outputs=dict(ir.outputs),
            state=new_state,
            reset_state=new_reset,
            next_state=new_next,
            output_exprs=new_output_exprs,
        )
    )
