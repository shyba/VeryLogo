from __future__ import annotations

from stc.interp import infer_type
from stc.tick_ir import (
    Add,
    And,
    Bitcast,
    BitVecConst,
    BoolConst,
    Concat,
    Expr,
    Mux,
    Not,
    Or,
    SimdConst,
    SimdAdd,
    SimdBlend,
    SimdAnd,
    SimdEq,
    SimdMaxS,
    SimdMaxU,
    SimdMinS,
    SimdMinU,
    SimdNot,
    SimdOr,
    SimdShuffle,
    SimdSub,
    SimdType,
    SimdUge,
    SimdUgt,
    SimdUle,
    SimdUlt,
    SimdXor,
    Slice,
    Lut8,
    Sub,
    Type,
    Var,
    Xor,
)


def expr_cost(expr: Expr, types: dict[str, Type]) -> int:
    if isinstance(expr, (Var, BoolConst, BitVecConst, SimdConst)):
        return 0

    if isinstance(expr, Bitcast):
        return expr_cost(expr.x, types)

    if isinstance(expr, Not):
        t = infer_type(expr.x, types)
        op = 2 if isinstance(t, SimdType) else 1
        return op + expr_cost(expr.x, types)

    if isinstance(expr, (SimdNot, SimdAnd, SimdOr, SimdXor)):
        if isinstance(expr, SimdNot):
            return 1 + expr_cost(expr.x, types)
        return 1 + expr_cost(expr.a, types) + expr_cost(expr.b, types)

    if isinstance(expr, (SimdAdd, SimdSub, SimdMinU, SimdMaxU, SimdMinS, SimdMaxS)):
        return 1 + expr_cost(expr.a, types) + expr_cost(expr.b, types)

    if isinstance(expr, (SimdEq, SimdUlt, SimdUle, SimdUgt, SimdUge)):
        return 1 + expr_cost(expr.a, types) + expr_cost(expr.b, types)

    if isinstance(expr, SimdBlend):
        return (
            2
            + expr_cost(expr.mask, types)
            + expr_cost(expr.a, types)
            + expr_cost(expr.b, types)
        )

    if isinstance(expr, SimdShuffle):
        return 3 + expr_cost(expr.x, types)

    if isinstance(expr, (And, Or, Xor, Add, Sub)):
        t = infer_type(expr.a, types)
        op = 2 if isinstance(t, SimdType) and isinstance(expr, (And, Or, Xor)) else 1
        return op + expr_cost(expr.a, types) + expr_cost(expr.b, types)

    if isinstance(expr, Mux):
        return (
            1
            + expr_cost(expr.cond, types)
            + expr_cost(expr.a, types)
            + expr_cost(expr.b, types)
        )

    if isinstance(expr, Concat):
        return 1 + sum(expr_cost(p, types) for p in expr.parts)

    if isinstance(expr, Slice):
        return 1 + expr_cost(expr.x, types)

    if isinstance(expr, Lut8):
        return 1 + expr_cost(expr.x, types)

    return 1
