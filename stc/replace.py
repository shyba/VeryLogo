from __future__ import annotations

from stc.tick_ir import (
    AShr,
    Add,
    And,
    Bitcast,
    BitVecConst,
    BoolConst,
    Concat,
    LShr,
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
    Mux,
    Not,
    Or,
    Shl,
    Slice,
    Lut8,
    SimdAdd,
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
    SimdAddMasked,
    SimdSubMasked,
    SimdMaxS,
    SimdMaxU,
    SimdMinS,
    SimdMinU,
    SimdSExtLo,
    SimdMulHiS,
    SimdMulHiU,
    SimdMulLo,
    SimdNot,
    SimdPackSS16To8,
    SimdPackSS32To16,
    SimdPackUS16To8,
    SimdMaskExpand,
    SimdMaskPack,
    SimdAnd,
    SimdOr,
    SimdShl,
    SimdShuffle,
    SimdSplat,
    SimdSub,
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
    Uge,
    Ugt,
    Ule,
    Ult,
    Var,
    Xor,
)


def replace_vars(expr: Expr, repl: dict[str, Expr]) -> Expr:
    if isinstance(expr, Var):
        return repl.get(expr.name, expr)
    if isinstance(expr, (BoolConst, BitVecConst, FloatConst, SimdConst)):
        return expr
    if isinstance(expr, Bitcast):
        return Bitcast(to=expr.to, x=replace_vars(expr.x, repl))
    if isinstance(expr, Not):
        return Not(x=replace_vars(expr.x, repl))
    if isinstance(
        expr, (And, Or, Xor, Add, Sub, Shl, LShr, AShr, Eq, Ult, Ule, Ugt, Uge)
    ):
        return expr.__class__(
            a=replace_vars(expr.a, repl),
            b=replace_vars(expr.b, repl),
        )
    if isinstance(expr, (FNeg, FAbs, FSqrt)):
        return expr.__class__(x=replace_vars(expr.x, repl))
    if isinstance(expr, (FAdd, FMul, FSub, FDiv, FEq, FLt, FLe, FNe)):
        return expr.__class__(
            a=replace_vars(expr.a, repl),
            b=replace_vars(expr.b, repl),
        )
    if isinstance(expr, SimdAdd):
        return SimdAdd(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdAddMasked):
        return SimdAddMasked(
            mask=replace_vars(expr.mask, repl),
            passthru=replace_vars(expr.passthru, repl),
            a=replace_vars(expr.a, repl),
            b=replace_vars(expr.b, repl),
        )
    if isinstance(expr, SimdAddSatU):
        return SimdAddSatU(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdAddSatS):
        return SimdAddSatS(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdSub):
        return SimdSub(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdSubMasked):
        return SimdSubMasked(
            mask=replace_vars(expr.mask, repl),
            passthru=replace_vars(expr.passthru, repl),
            a=replace_vars(expr.a, repl),
            b=replace_vars(expr.b, repl),
        )
    if isinstance(expr, SimdSubSatU):
        return SimdSubSatU(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdSubSatS):
        return SimdSubSatS(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, (SimdFSqrt, SimdFNeg, SimdFAbs)):
        return expr.__class__(x=replace_vars(expr.x, repl))
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
        return expr.__class__(
            a=replace_vars(expr.a, repl),
            b=replace_vars(expr.b, repl),
        )
    if isinstance(expr, SimdFFma):
        return SimdFFma(
            a=replace_vars(expr.a, repl),
            b=replace_vars(expr.b, repl),
            c=replace_vars(expr.c, repl),
        )
    if isinstance(expr, SimdMulLo):
        return SimdMulLo(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdMulHiU):
        return SimdMulHiU(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdMulHiS):
        return SimdMulHiS(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdMaddS16):
        return SimdMaddS16(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdUnpackLo):
        return SimdUnpackLo(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdUnpackHi):
        return SimdUnpackHi(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdPackSS16To8):
        return SimdPackSS16To8(
            a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl)
        )
    if isinstance(expr, SimdPackUS16To8):
        return SimdPackUS16To8(
            a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl)
        )
    if isinstance(expr, SimdPackSS32To16):
        return SimdPackSS32To16(
            a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl)
        )
    if isinstance(expr, SimdMaskExpand):
        return SimdMaskExpand(to=expr.to, x=replace_vars(expr.x, repl))
    if isinstance(expr, SimdMaskPack):
        return SimdMaskPack(x=replace_vars(expr.x, repl))
    if isinstance(expr, SimdMinU):
        return SimdMinU(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdMaxU):
        return SimdMaxU(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdMinS):
        return SimdMinS(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdMaxS):
        return SimdMaxS(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdBlend):
        return SimdBlend(
            mask=replace_vars(expr.mask, repl),
            a=replace_vars(expr.a, repl),
            b=replace_vars(expr.b, repl),
        )
    if isinstance(expr, SimdZExtLo):
        return SimdZExtLo(to=expr.to, x=replace_vars(expr.x, repl))
    if isinstance(expr, SimdSExtLo):
        return SimdSExtLo(to=expr.to, x=replace_vars(expr.x, repl))
    if isinstance(expr, SimdEq):
        return SimdEq(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdNot):
        return SimdNot(x=replace_vars(expr.x, repl))
    if isinstance(expr, SimdAnd):
        return SimdAnd(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdOr):
        return SimdOr(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdXor):
        return SimdXor(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdShl):
        return SimdShl(a=replace_vars(expr.a, repl), sh=replace_vars(expr.sh, repl))
    if isinstance(expr, SimdLShr):
        return SimdLShr(a=replace_vars(expr.a, repl), sh=replace_vars(expr.sh, repl))
    if isinstance(expr, SimdAShr):
        return SimdAShr(a=replace_vars(expr.a, repl), sh=replace_vars(expr.sh, repl))
    if isinstance(expr, SimdUlt):
        return SimdUlt(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdUle):
        return SimdUle(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdUgt):
        return SimdUgt(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdUge):
        return SimdUge(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdSlt):
        return SimdSlt(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdSle):
        return SimdSle(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdSgt):
        return SimdSgt(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdSge):
        return SimdSge(a=replace_vars(expr.a, repl), b=replace_vars(expr.b, repl))
    if isinstance(expr, SimdSplat):
        return SimdSplat(to=expr.to, x=replace_vars(expr.x, repl))
    if isinstance(expr, SimdExtractLane):
        return SimdExtractLane(x=replace_vars(expr.x, repl), lane=expr.lane)
    if isinstance(expr, SimdInsertLane):
        return SimdInsertLane(
            x=replace_vars(expr.x, repl),
            lane=expr.lane,
            value=replace_vars(expr.value, repl),
        )
    if isinstance(expr, SimdShuffle):
        return SimdShuffle(x=replace_vars(expr.x, repl), indices=list(expr.indices))
    if isinstance(expr, Mux):
        return Mux(
            cond=replace_vars(expr.cond, repl),
            a=replace_vars(expr.a, repl),
            b=replace_vars(expr.b, repl),
        )
    if isinstance(expr, Concat):
        return Concat(parts=[replace_vars(p, repl) for p in expr.parts])
    if isinstance(expr, Slice):
        return Slice(x=replace_vars(expr.x, repl), offset=expr.offset, width=expr.width)
    if isinstance(expr, Lut8):
        return Lut8(x=replace_vars(expr.x, repl), table=list(expr.table))
    raise TypeError("unsupported expression kind")
