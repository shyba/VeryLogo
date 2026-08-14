"""Target-independent lowering helpers for shaped integer GEMM calls."""

from __future__ import annotations

from dataclasses import fields, replace as dc_replace

from stc.tick_ir import (
    Add,
    BitVecConst,
    Concat,
    Expr,
    GemmCall,
    BoolConst,
    Mul,
    Mux,
    Slice,
    Sub,
    TickIR,
    Xor,
)


class GemmLoweringError(ValueError):
    """The requested generic GEMM expansion is outside its safe bound."""


def _extend(x: Expr, width: int, target: int, *, signed: bool) -> Expr:
    if width == target:
        return x
    if width < 1 or target < width:
        raise GemmLoweringError("invalid GEMM element extension")
    pad = target - width
    if signed:
        sign = Slice(x=x, offset=width - 1, width=1)
        pad_expr: Expr = sign
        if pad > 1:
            pad_expr = Concat(parts=tuple(sign for _ in range(pad)))
        return Concat(parts=(pad_expr, x))
    return Concat(parts=(BitVecConst(width=pad, value=0), x))


def _magnitude(x: Expr, width: int, target: int, *, signed: bool) -> tuple[Expr, Expr]:
    raw = _extend(x, width, target, signed=signed)
    if not signed:
        return raw, BoolConst(value=False)
    sign: Expr = Slice(x=x, offset=width - 1, width=1)
    neg = Sub(a=BitVecConst(width=target, value=0), b=raw)
    return Mux(cond=sign, a=neg, b=raw), sign


def expand_gemm_call(expr: GemmCall, *, max_macs: int = 4096) -> Expr:
    """Expand a shaped GEMM into fixed-width scalar Tick-IR.

    This is the portable/reference path.  Large GEMMs intentionally remain as
    ``GemmCall`` nodes so a target lowering can consume the shape without
    constructing an enormous expression tree.
    """

    macs = expr.m * expr.n * expr.k
    if macs > max_macs:
        raise GemmLoweringError(
            f"generic GEMM expansion would create {macs} MACs (limit {max_macs})"
        )

    result: list[Expr] = []
    zero = BitVecConst(width=expr.acc_width, value=0)
    for i in range(expr.m):
        for j in range(expr.n):
            acc: Expr = zero
            for q in range(expr.k):
                a_raw = Slice(
                    x=expr.a,
                    offset=(i * expr.k + q) * expr.a_width,
                    width=expr.a_width,
                )
                b_raw = Slice(
                    x=expr.b,
                    offset=(q * expr.n + j) * expr.b_width,
                    width=expr.b_width,
                )
                a_mag, a_sign = _magnitude(
                    a_raw, expr.a_width, expr.acc_width, signed=expr.a_signed
                )
                b_mag, b_sign = _magnitude(
                    b_raw, expr.b_width, expr.acc_width, signed=expr.b_signed
                )
                product: Expr = Mul(a=a_mag, b=b_mag)
                neg = Xor(a=a_sign, b=b_sign)
                product = Mux(
                    cond=neg,
                    a=Sub(a=BitVecConst(width=expr.acc_width, value=0), b=product),
                    b=product,
                )
                acc = Add(a=acc, b=product)
            result.append(acc)

    return result[0] if len(result) == 1 else Concat(parts=tuple(reversed(result)))


def expand_gemm_expr(expr: Expr, *, max_macs: int = 4096) -> Expr:
    """Recursively expand every GEMM node in an expression tree."""

    if isinstance(expr, GemmCall):
        shaped = dc_replace(
            expr,
            a=expand_gemm_expr(expr.a, max_macs=max_macs),
            b=expand_gemm_expr(expr.b, max_macs=max_macs),
        )
        from stc.tech import get_technology

        # Route the generic choice through the technology dispatcher so this
        # path exercises the same target-selection seam as x86-gemm.
        return get_technology("generic").lower_expr(shaped, target="scalar")

    updates: dict[str, object] = {}
    changed = False
    for field in fields(expr):
        value = getattr(expr, field.name)
        if isinstance(value, tuple):
            new_value = tuple(
                expand_gemm_expr(v, max_macs=max_macs) if isinstance(v, Expr) else v
                for v in value
            )
            changed |= new_value != value
            updates[field.name] = new_value
        elif isinstance(value, list):
            new_value = [
                expand_gemm_expr(v, max_macs=max_macs) if isinstance(v, Expr) else v
                for v in value
            ]
            changed |= new_value != value
            updates[field.name] = new_value
        elif isinstance(value, Expr):
            new_value = expand_gemm_expr(value, max_macs=max_macs)
            changed |= new_value != value
            updates[field.name] = new_value
    return dc_replace(expr, **updates) if changed else expr


def expand_gemm_calls(ir: TickIR, *, max_macs: int = 4096) -> TickIR:
    """Apply portable GEMM expansion to all expressions in a TickIR module."""

    return TickIR(
        name=ir.name,
        inputs=dict(ir.inputs),
        outputs=dict(ir.outputs),
        state=dict(ir.state),
        reset_state={
            name: expand_gemm_expr(expr, max_macs=max_macs)
            for name, expr in ir.reset_state.items()
        },
        next_state={
            name: expand_gemm_expr(expr, max_macs=max_macs)
            for name, expr in ir.next_state.items()
        },
        output_exprs={
            name: expand_gemm_expr(expr, max_macs=max_macs)
            for name, expr in ir.output_exprs.items()
        },
    )


def find_gemm_call(expr: Expr) -> GemmCall | None:
    """Find the first shaped GEMM node in an expression tree."""

    if isinstance(expr, GemmCall):
        return expr
    for field in fields(expr):
        value = getattr(expr, field.name)
        values = value if isinstance(value, (tuple, list)) else (value,)
        for child in values:
            if isinstance(child, Expr):
                found = find_gemm_call(child)
                if found is not None:
                    return found
    return None
