from __future__ import annotations

from stc.autovec import autovectorize_expr
from stc.tick_ir import (
    AShr,
    Add,
    And,
    Bitcast,
    BoolConst,
    Concat,
    Eq,
    Expr,
    LShr,
    Mux,
    Not,
    Or,
    Shl,
    SimdAShr,
    SimdAdd,
    SimdAnd,
    SimdConst,
    SimdEq,
    SimdExtractLane,
    SimdInsertLane,
    SimdLShr,
    SimdNot,
    SimdOr,
    SimdShl,
    SimdShuffle,
    SimdSub,
    SimdSge,
    SimdSgt,
    SimdSle,
    SimdSlt,
    SimdUge,
    SimdUgt,
    SimdUle,
    SimdUlt,
    SimdSplat,
    SimdXor,
    Slice,
    Lut8,
    Sub,
    TickIR,
    Type,
    Uge,
    Ugt,
    Ule,
    Ult,
    Var,
    Xor,
    BitVecConst,
)


def _rewrite_expr(expr: Expr, types: dict[str, Type], *, timeout_ms: int) -> Expr:
    if isinstance(expr, (Var, BoolConst, BitVecConst, SimdConst)):
        return expr

    if isinstance(expr, Bitcast):
        return autovectorize_expr(
            Bitcast(to=expr.to, x=_rewrite_expr(expr.x, types, timeout_ms=timeout_ms)),
            types,
            timeout_ms=timeout_ms,
        )

    if isinstance(expr, Not):
        return autovectorize_expr(
            Not(x=_rewrite_expr(expr.x, types, timeout_ms=timeout_ms)),
            types,
            timeout_ms=timeout_ms,
        )

    if isinstance(
        expr, (And, Or, Xor, Add, Sub, Shl, LShr, AShr, Eq, Ult, Ule, Ugt, Uge)
    ):
        return autovectorize_expr(
            expr.__class__(
                a=_rewrite_expr(expr.a, types, timeout_ms=timeout_ms),
                b=_rewrite_expr(expr.b, types, timeout_ms=timeout_ms),
            ),
            types,
            timeout_ms=timeout_ms,
        )

    if isinstance(expr, Mux):
        return autovectorize_expr(
            Mux(
                cond=_rewrite_expr(expr.cond, types, timeout_ms=timeout_ms),
                a=_rewrite_expr(expr.a, types, timeout_ms=timeout_ms),
                b=_rewrite_expr(expr.b, types, timeout_ms=timeout_ms),
            ),
            types,
            timeout_ms=timeout_ms,
        )

    if isinstance(expr, Concat):
        return autovectorize_expr(
            Concat(
                parts=[
                    _rewrite_expr(p, types, timeout_ms=timeout_ms) for p in expr.parts
                ]
            ),
            types,
            timeout_ms=timeout_ms,
        )

    if isinstance(expr, Slice):
        return autovectorize_expr(
            Slice(
                x=_rewrite_expr(expr.x, types, timeout_ms=timeout_ms),
                offset=expr.offset,
                width=expr.width,
            ),
            types,
            timeout_ms=timeout_ms,
        )

    if isinstance(expr, Lut8):
        return Lut8(
            x=_rewrite_expr(expr.x, types, timeout_ms=timeout_ms),
            table=list(expr.table),
        )

    if isinstance(expr, SimdAdd):
        return autovectorize_expr(
            SimdAdd(
                a=_rewrite_expr(expr.a, types, timeout_ms=timeout_ms),
                b=_rewrite_expr(expr.b, types, timeout_ms=timeout_ms),
            ),
            types,
            timeout_ms=timeout_ms,
        )

    if isinstance(expr, SimdSub):
        return autovectorize_expr(
            SimdSub(
                a=_rewrite_expr(expr.a, types, timeout_ms=timeout_ms),
                b=_rewrite_expr(expr.b, types, timeout_ms=timeout_ms),
            ),
            types,
            timeout_ms=timeout_ms,
        )

    if isinstance(expr, SimdEq):
        return autovectorize_expr(
            SimdEq(
                a=_rewrite_expr(expr.a, types, timeout_ms=timeout_ms),
                b=_rewrite_expr(expr.b, types, timeout_ms=timeout_ms),
            ),
            types,
            timeout_ms=timeout_ms,
        )

    if isinstance(expr, SimdNot):
        return autovectorize_expr(
            SimdNot(x=_rewrite_expr(expr.x, types, timeout_ms=timeout_ms)),
            types,
            timeout_ms=timeout_ms,
        )

    if isinstance(expr, (SimdAnd, SimdOr, SimdXor)):
        return autovectorize_expr(
            expr.__class__(
                a=_rewrite_expr(expr.a, types, timeout_ms=timeout_ms),
                b=_rewrite_expr(expr.b, types, timeout_ms=timeout_ms),
            ),
            types,
            timeout_ms=timeout_ms,
        )

    if isinstance(expr, (SimdShl, SimdLShr, SimdAShr)):
        return autovectorize_expr(
            expr.__class__(
                a=_rewrite_expr(expr.a, types, timeout_ms=timeout_ms),
                sh=_rewrite_expr(expr.sh, types, timeout_ms=timeout_ms),
            ),
            types,
            timeout_ms=timeout_ms,
        )

    if isinstance(expr, (SimdUlt, SimdUle, SimdUgt, SimdUge)):
        return autovectorize_expr(
            expr.__class__(
                a=_rewrite_expr(expr.a, types, timeout_ms=timeout_ms),
                b=_rewrite_expr(expr.b, types, timeout_ms=timeout_ms),
            ),
            types,
            timeout_ms=timeout_ms,
        )

    if isinstance(expr, (SimdSlt, SimdSle, SimdSgt, SimdSge)):
        return autovectorize_expr(
            expr.__class__(
                a=_rewrite_expr(expr.a, types, timeout_ms=timeout_ms),
                b=_rewrite_expr(expr.b, types, timeout_ms=timeout_ms),
            ),
            types,
            timeout_ms=timeout_ms,
        )

    if isinstance(expr, SimdSplat):
        return autovectorize_expr(
            SimdSplat(
                to=expr.to, x=_rewrite_expr(expr.x, types, timeout_ms=timeout_ms)
            ),
            types,
            timeout_ms=timeout_ms,
        )

    if isinstance(expr, SimdExtractLane):
        return autovectorize_expr(
            SimdExtractLane(
                x=_rewrite_expr(expr.x, types, timeout_ms=timeout_ms), lane=expr.lane
            ),
            types,
            timeout_ms=timeout_ms,
        )

    if isinstance(expr, SimdInsertLane):
        return autovectorize_expr(
            SimdInsertLane(
                x=_rewrite_expr(expr.x, types, timeout_ms=timeout_ms),
                lane=expr.lane,
                value=_rewrite_expr(expr.value, types, timeout_ms=timeout_ms),
            ),
            types,
            timeout_ms=timeout_ms,
        )

    if isinstance(expr, SimdShuffle):
        return autovectorize_expr(
            SimdShuffle(
                x=_rewrite_expr(expr.x, types, timeout_ms=timeout_ms),
                indices=list(expr.indices),
            ),
            types,
            timeout_ms=timeout_ms,
        )

    raise TypeError("unsupported expression kind")


def autovectorize_tick_ir(ir: TickIR, *, timeout_ms: int = 200) -> TickIR:
    types: dict[str, Type] = {**ir.inputs, **ir.state}
    out_exprs = {
        k: _rewrite_expr(v, types, timeout_ms=timeout_ms)
        for k, v in ir.output_exprs.items()
    }
    next_state = {
        k: _rewrite_expr(v, types, timeout_ms=timeout_ms)
        for k, v in ir.next_state.items()
    }
    return TickIR(
        name=ir.name,
        inputs=dict(ir.inputs),
        outputs=dict(ir.outputs),
        state=dict(ir.state),
        reset_state=dict(ir.reset_state),
        next_state=next_state,
        output_exprs=out_exprs,
    )
