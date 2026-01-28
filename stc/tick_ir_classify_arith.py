from __future__ import annotations

from collections import Counter

from stc.interp import infer_type
from stc.tick_ir import (
    Add,
    AShr,
    BitVecConst,
    BitVecType,
    BoolConst,
    BoolType,
    Div,
    Expr,
    FloatType,
    LShr,
    Mul,
    Shl,
    SimdType,
    Sub,
    TickIR,
    Type,
    Var,
)


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _log2(n: int) -> int:
    assert _is_power_of_two(n)
    return n.bit_length() - 1


def _mask(width: int) -> int:
    return (1 << int(width)) - 1


def classify_arithmetic_expr(
    expr: Expr, types: dict[str, BoolType | BitVecType | FloatType | SimdType]
) -> Expr:
    from stc.tick_ir import FloatConst, SimdConst

    if isinstance(expr, (BoolConst, BitVecConst, FloatConst, SimdConst, Var)):
        return expr

    from stc.tick_ir import (
        And,
        Bitcast,
        BitTranspose,
        Concat,
        Eq,
        FAbs,
        FAdd,
        FDiv,
        FEq,
        FLe,
        FLt,
        FMul,
        FNe,
        FNeg,
        FSqrt,
        FSub,
        FloatConst,
        Lut8,
        Mux,
        Not,
        Or,
        Rotl,
        Rotr,
        SimdAShr,
        SimdAdd,
        SimdAddMasked,
        SimdAddSatS,
        SimdAddSatU,
        SimdAnd,
        SimdBlend,
        SimdConst,
        SimdEq,
        SimdExtractLane,
        SimdFAbs,
        SimdFAdd,
        SimdFCmpEq,
        SimdFCmpLe,
        SimdFCmpLt,
        SimdFCmpNe,
        SimdFDiv,
        SimdFFma,
        SimdFMul,
        SimdFNeg,
        SimdFSqrt,
        SimdFSub,
        SimdInsertLane,
        SimdLShr,
        SimdMaddS16,
        SimdMaskExpand,
        SimdMaskPack,
        SimdMaxS,
        SimdMaxU,
        SimdMinS,
        SimdMinU,
        SimdMulHiS,
        SimdMulHiU,
        SimdMulLo,
        SimdNot,
        SimdOr,
        SimdPackSS16To8,
        SimdPackSS32To16,
        SimdPackUS16To8,
        SimdSExtLo,
        SimdSge,
        SimdSgt,
        SimdShl,
        SimdShuffle,
        SimdSle,
        SimdSlt,
        SimdSplat,
        SimdSub,
        SimdSubMasked,
        SimdSubSatS,
        SimdSubSatU,
        SimdUnpackHi,
        SimdUnpackLo,
        SimdUge,
        SimdUgt,
        SimdUle,
        SimdUlt,
        SimdXor,
        SimdZExtLo,
        Slice,
        TernaryLut,
        Uge,
        Ugt,
        Ule,
        Ult,
        Xor,
    )

    if isinstance(expr, (Rotl, Rotr)):
        x = classify_arithmetic_expr(expr.x, types)
        sh = classify_arithmetic_expr(expr.sh, types)
        return expr.__class__(x=x, sh=sh)

    if isinstance(expr, Bitcast):
        x = classify_arithmetic_expr(expr.x, types)
        return Bitcast(to=expr.to, x=x)

    if isinstance(expr, BitTranspose):
        x = classify_arithmetic_expr(expr.x, types)
        return BitTranspose(
            x=x, lane_width=expr.lane_width, lanes=expr.lanes, row_width=expr.row_width
        )

    if isinstance(expr, Not):
        x = classify_arithmetic_expr(expr.x, types)
        return Not(x=x)

    if isinstance(expr, Lut8):
        x = classify_arithmetic_expr(expr.x, types)
        return Lut8(x=x, table=list(expr.table))

    if isinstance(expr, TernaryLut):
        a = classify_arithmetic_expr(expr.a, types)
        b = classify_arithmetic_expr(expr.b, types)
        c = classify_arithmetic_expr(expr.c, types)
        return TernaryLut(a=a, b=b, c=c, imm8=expr.imm8)

    if isinstance(expr, Mux):
        cond = classify_arithmetic_expr(expr.cond, types)
        a = classify_arithmetic_expr(expr.a, types)
        b = classify_arithmetic_expr(expr.b, types)
        return Mux(cond=cond, a=a, b=b)

    if isinstance(expr, Concat):
        parts = [classify_arithmetic_expr(p, types) for p in expr.parts]
        return Concat(parts=parts)

    if isinstance(expr, Slice):
        x = classify_arithmetic_expr(expr.x, types)
        return Slice(x=x, offset=expr.offset, width=expr.width)

    if isinstance(
        expr,
        (
            And,
            Or,
            Xor,
            Shl,
            LShr,
            AShr,
            Eq,
            Ult,
            Ule,
            Ugt,
            Uge,
            FAdd,
            FSub,
            FMul,
            FDiv,
            FEq,
            FLt,
            FLe,
            FNe,
        ),
    ):
        a = classify_arithmetic_expr(expr.a, types)
        b = classify_arithmetic_expr(expr.b, types)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, (FNeg, FAbs, FSqrt)):
        x = classify_arithmetic_expr(expr.x, types)
        return expr.__class__(x=x)

    if isinstance(expr, Add):
        a = classify_arithmetic_expr(expr.a, types)
        b = classify_arithmetic_expr(expr.b, types)

        t = infer_type(a, types)
        if not isinstance(t, BitVecType):
            return Add(a=a, b=b)

        if isinstance(b, BitVecConst) and b.value == 0:
            return a
        if isinstance(a, BitVecConst) and a.value == 0:
            return b

        return Add(a=a, b=b)

    if isinstance(expr, Sub):
        a = classify_arithmetic_expr(expr.a, types)
        b = classify_arithmetic_expr(expr.b, types)

        t = infer_type(a, types)
        if not isinstance(t, BitVecType):
            return Sub(a=a, b=b)

        if isinstance(b, BitVecConst) and b.value == 0:
            return a
        if a == b:
            return BitVecConst(width=t.width, value=0)

        return Sub(a=a, b=b)

    if isinstance(expr, Mul):
        a = classify_arithmetic_expr(expr.a, types)
        b = classify_arithmetic_expr(expr.b, types)

        t = infer_type(a, types)
        if not isinstance(t, BitVecType):
            return Mul(a=a, b=b)

        if isinstance(b, BitVecConst) and b.value == 0:
            return BitVecConst(width=t.width, value=0)
        if isinstance(a, BitVecConst) and a.value == 0:
            return BitVecConst(width=t.width, value=0)
        if isinstance(b, BitVecConst) and b.value == 1:
            return a
        if isinstance(a, BitVecConst) and a.value == 1:
            return b

        if isinstance(b, BitVecConst) and _is_power_of_two(b.value):
            shift_amount = _log2(b.value)
            return Shl(a=a, b=BitVecConst(width=t.width, value=shift_amount))
        if isinstance(a, BitVecConst) and _is_power_of_two(a.value):
            shift_amount = _log2(a.value)
            return Shl(a=b, b=BitVecConst(width=t.width, value=shift_amount))

        return Mul(a=a, b=b)

    if isinstance(expr, Div):
        a = classify_arithmetic_expr(expr.a, types)
        b = classify_arithmetic_expr(expr.b, types)

        t = infer_type(a, types)
        if not isinstance(t, BitVecType):
            return Div(a=a, b=b)

        if isinstance(b, BitVecConst) and b.value == 1:
            return a

        if isinstance(b, BitVecConst) and _is_power_of_two(b.value):
            shift_amount = _log2(b.value)
            return LShr(a=a, b=BitVecConst(width=t.width, value=shift_amount))

        return Div(a=a, b=b)

    if isinstance(expr, (SimdShl, SimdLShr, SimdAShr)):
        a = classify_arithmetic_expr(expr.a, types)
        sh = classify_arithmetic_expr(expr.sh, types)
        return expr.__class__(a=a, sh=sh)

    if isinstance(
        expr,
        (
            SimdAdd,
            SimdSub,
            SimdAnd,
            SimdOr,
            SimdXor,
            SimdEq,
            SimdUlt,
            SimdUle,
            SimdUgt,
            SimdUge,
            SimdSlt,
            SimdSle,
            SimdSgt,
            SimdSge,
            SimdMinU,
            SimdMaxU,
            SimdMinS,
            SimdMaxS,
            SimdMulLo,
            SimdMulHiU,
            SimdMulHiS,
            SimdFAdd,
            SimdFSub,
            SimdFMul,
            SimdFDiv,
            SimdFCmpEq,
            SimdFCmpLt,
            SimdFCmpLe,
            SimdFCmpNe,
        ),
    ):
        a = classify_arithmetic_expr(expr.a, types)
        b = classify_arithmetic_expr(expr.b, types)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, (SimdNot, SimdFNeg, SimdFAbs, SimdFSqrt)):
        x = classify_arithmetic_expr(expr.x, types)
        return expr.__class__(x=x)

    if isinstance(
        expr,
        (
            SimdAddMasked,
            SimdSubMasked,
            SimdAddSatU,
            SimdSubSatU,
            SimdAddSatS,
            SimdSubSatS,
        ),
    ):
        if isinstance(expr, (SimdAddMasked, SimdSubMasked)):
            mask = classify_arithmetic_expr(expr.mask, types)
            passthru = classify_arithmetic_expr(expr.passthru, types)
            a = classify_arithmetic_expr(expr.a, types)
            b = classify_arithmetic_expr(expr.b, types)
            return expr.__class__(mask=mask, passthru=passthru, a=a, b=b)
        else:
            a = classify_arithmetic_expr(expr.a, types)
            b = classify_arithmetic_expr(expr.b, types)
            return expr.__class__(a=a, b=b)

    if isinstance(expr, (SimdPackSS16To8, SimdPackUS16To8, SimdPackSS32To16)):
        a = classify_arithmetic_expr(expr.a, types)
        b = classify_arithmetic_expr(expr.b, types)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, (SimdUnpackHi, SimdUnpackLo)):
        a = classify_arithmetic_expr(expr.a, types)
        b = classify_arithmetic_expr(expr.b, types)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, SimdBlend):
        mask = classify_arithmetic_expr(expr.mask, types)
        a = classify_arithmetic_expr(expr.a, types)
        b = classify_arithmetic_expr(expr.b, types)
        return SimdBlend(mask=mask, a=a, b=b)

    if isinstance(expr, SimdMaddS16):
        a = classify_arithmetic_expr(expr.a, types)
        b = classify_arithmetic_expr(expr.b, types)
        return SimdMaddS16(a=a, b=b)

    if isinstance(expr, (SimdZExtLo, SimdSExtLo)):
        x = classify_arithmetic_expr(expr.x, types)
        return expr.__class__(x=x, to=expr.to)

    if isinstance(expr, (SimdMaskExpand, SimdMaskPack)):
        x = classify_arithmetic_expr(expr.x, types)
        return expr.__class__(x=x, to=expr.to)

    if isinstance(expr, SimdSplat):
        x = classify_arithmetic_expr(expr.x, types)
        return SimdSplat(x=x, to=expr.to)

    if isinstance(expr, SimdExtractLane):
        x = classify_arithmetic_expr(expr.x, types)
        return SimdExtractLane(x=x, lane=expr.lane)

    if isinstance(expr, SimdInsertLane):
        x = classify_arithmetic_expr(expr.x, types)
        value = classify_arithmetic_expr(expr.value, types)
        return SimdInsertLane(x=x, lane=expr.lane, value=value)

    if isinstance(expr, SimdShuffle):
        x = classify_arithmetic_expr(expr.x, types)
        return SimdShuffle(x=x, indices=list(expr.indices))

    if isinstance(expr, SimdFFma):
        a = classify_arithmetic_expr(expr.a, types)
        b = classify_arithmetic_expr(expr.b, types)
        c = classify_arithmetic_expr(expr.c, types)
        return SimdFFma(a=a, b=b, c=c)

    raise ValueError(f"unhandled expr kind: {type(expr).__name__}")


def _count_expr_kinds(expr: Expr) -> Counter[str]:
    counts: Counter[str] = Counter()
    counts[type(expr).__name__] += 1

    from stc.tick_ir import (
        And,
        Bitcast,
        BitTranspose,
        Concat,
        Eq,
        FAbs,
        FAdd,
        FDiv,
        FEq,
        FLe,
        FLt,
        FMul,
        FNe,
        FNeg,
        FSqrt,
        FSub,
        Lut8,
        Mux,
        Not,
        Or,
        Rotl,
        Rotr,
        SimdAShr,
        SimdAdd,
        SimdAddMasked,
        SimdAddSatS,
        SimdAddSatU,
        SimdAnd,
        SimdBlend,
        SimdEq,
        SimdExtractLane,
        SimdFAbs,
        SimdFAdd,
        SimdFCmpEq,
        SimdFCmpLe,
        SimdFCmpLt,
        SimdFCmpNe,
        SimdFDiv,
        SimdFFma,
        SimdFMul,
        SimdFNeg,
        SimdFSqrt,
        SimdFSub,
        SimdInsertLane,
        SimdLShr,
        SimdMaddS16,
        SimdMaskExpand,
        SimdMaskPack,
        SimdMaxS,
        SimdMaxU,
        SimdMinS,
        SimdMinU,
        SimdMulHiS,
        SimdMulHiU,
        SimdMulLo,
        SimdNot,
        SimdOr,
        SimdPackSS16To8,
        SimdPackSS32To16,
        SimdPackUS16To8,
        SimdSExtLo,
        SimdSge,
        SimdSgt,
        SimdShl,
        SimdShuffle,
        SimdSle,
        SimdSlt,
        SimdSplat,
        SimdSub,
        SimdSubMasked,
        SimdSubSatS,
        SimdSubSatU,
        SimdUge,
        SimdUgt,
        SimdUle,
        SimdUlt,
        SimdUnpackHi,
        SimdUnpackLo,
        SimdXor,
        SimdZExtLo,
        Slice,
        TernaryLut,
        Uge,
        Ugt,
        Ule,
        Ult,
        Xor,
    )

    if isinstance(expr, (Rotl, Rotr)):
        counts.update(_count_expr_kinds(expr.x))
        counts.update(_count_expr_kinds(expr.sh))
    elif isinstance(expr, (Not, Bitcast, BitTranspose, Lut8)):
        counts.update(_count_expr_kinds(expr.x))
    elif isinstance(expr, TernaryLut):
        counts.update(_count_expr_kinds(expr.a))
        counts.update(_count_expr_kinds(expr.b))
        counts.update(_count_expr_kinds(expr.c))
    elif isinstance(expr, Mux):
        counts.update(_count_expr_kinds(expr.cond))
        counts.update(_count_expr_kinds(expr.a))
        counts.update(_count_expr_kinds(expr.b))
    elif isinstance(expr, Concat):
        for p in expr.parts:
            counts.update(_count_expr_kinds(p))
    elif isinstance(expr, Slice):
        counts.update(_count_expr_kinds(expr.x))
    elif isinstance(expr, (SimdShl, SimdLShr, SimdAShr)):
        counts.update(_count_expr_kinds(expr.a))
        counts.update(_count_expr_kinds(expr.sh))
    elif isinstance(
        expr,
        (
            And,
            Or,
            Xor,
            Add,
            Sub,
            Mul,
            Div,
            Shl,
            LShr,
            AShr,
            Eq,
            Ult,
            Ule,
            Ugt,
            Uge,
            FAdd,
            FSub,
            FMul,
            FDiv,
            FEq,
            FLt,
            FLe,
            FNe,
            SimdAdd,
            SimdSub,
            SimdAnd,
            SimdOr,
            SimdXor,
            SimdEq,
            SimdUlt,
            SimdUle,
            SimdUgt,
            SimdUge,
            SimdSlt,
            SimdSle,
            SimdSgt,
            SimdSge,
            SimdMinU,
            SimdMaxU,
            SimdMinS,
            SimdMaxS,
            SimdMulLo,
            SimdMulHiU,
            SimdMulHiS,
            SimdFAdd,
            SimdFSub,
            SimdFMul,
            SimdFDiv,
            SimdFCmpEq,
            SimdFCmpLt,
            SimdFCmpLe,
            SimdFCmpNe,
            SimdUnpackHi,
            SimdUnpackLo,
            SimdPackSS16To8,
            SimdPackUS16To8,
            SimdPackSS32To16,
            SimdBlend,
            SimdMaddS16,
        ),
    ):
        counts.update(_count_expr_kinds(expr.a))
        counts.update(_count_expr_kinds(expr.b))
        if isinstance(expr, SimdBlend):
            counts.update(_count_expr_kinds(expr.mask))
    elif isinstance(expr, SimdShuffle):
        counts.update(_count_expr_kinds(expr.x))
    elif isinstance(expr, (FNeg, FAbs, FSqrt, SimdNot, SimdFNeg, SimdFAbs, SimdFSqrt)):
        counts.update(_count_expr_kinds(expr.x))
    elif isinstance(
        expr,
        (
            SimdAddMasked,
            SimdSubMasked,
            SimdAddSatU,
            SimdSubSatU,
            SimdAddSatS,
            SimdSubSatS,
        ),
    ):
        if isinstance(expr, (SimdAddMasked, SimdSubMasked)):
            counts.update(_count_expr_kinds(expr.mask))
            counts.update(_count_expr_kinds(expr.passthru))
            counts.update(_count_expr_kinds(expr.a))
            counts.update(_count_expr_kinds(expr.b))
        else:
            counts.update(_count_expr_kinds(expr.a))
            counts.update(_count_expr_kinds(expr.b))
    elif isinstance(expr, (SimdZExtLo, SimdSExtLo, SimdMaskExpand, SimdMaskPack)):
        counts.update(_count_expr_kinds(expr.x))
    elif isinstance(expr, SimdSplat):
        counts.update(_count_expr_kinds(expr.x))
    elif isinstance(expr, SimdExtractLane):
        counts.update(_count_expr_kinds(expr.x))
    elif isinstance(expr, SimdInsertLane):
        counts.update(_count_expr_kinds(expr.x))
        counts.update(_count_expr_kinds(expr.value))
    elif isinstance(expr, SimdFFma):
        counts.update(_count_expr_kinds(expr.a))
        counts.update(_count_expr_kinds(expr.b))
        counts.update(_count_expr_kinds(expr.c))

    return counts


def classify_arithmetic(ir: TickIR) -> tuple[TickIR, dict[str, int]]:
    types: dict[str, Type] = {}
    types.update(ir.inputs)
    types.update(ir.state)

    new_reset_state = {}
    for name, expr in ir.reset_state.items():
        new_reset_state[name] = classify_arithmetic_expr(expr, types)

    new_next_state = {}
    for name, expr in ir.next_state.items():
        new_next_state[name] = classify_arithmetic_expr(expr, types)

    new_output_exprs = {}
    for name, expr in ir.output_exprs.items():
        new_output_exprs[name] = classify_arithmetic_expr(expr, types)

    new_ir = TickIR(
        name=ir.name,
        inputs=dict(ir.inputs),
        outputs=dict(ir.outputs),
        state=dict(ir.state),
        reset_state=new_reset_state,
        next_state=new_next_state,
        output_exprs=new_output_exprs,
    )

    all_counts: Counter[str] = Counter()
    for expr in new_ir.next_state.values():
        all_counts.update(_count_expr_kinds(expr))
    for expr in new_ir.output_exprs.values():
        all_counts.update(_count_expr_kinds(expr))

    report = {
        "Add": all_counts["Add"],
        "Sub": all_counts["Sub"],
        "Mul": all_counts["Mul"],
        "Div": all_counts["Div"],
    }

    return new_ir, report
