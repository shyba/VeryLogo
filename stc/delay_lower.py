from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from stc.interp import infer_type
from stc.tick_ir import (
    BitVecConst,
    BitVecType,
    BoolConst,
    BoolType,
    Delay,
    EXPR_CLASSES,
    Expr,
    SimdConst,
    SimdType,
    TickIR,
    Type,
    Var,
)


@dataclass(frozen=True)
class DelayLowerError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def _zero(t: Type) -> Expr:
    if isinstance(t, BoolType):
        return BoolConst(value=False)
    if isinstance(t, BitVecType):
        return BitVecConst(width=t.width, value=0)
    assert isinstance(t, SimdType)
    return SimdConst(lane_width=t.lane_width, lanes=t.lanes, value=0)


def lower_delays(ir: TickIR) -> TickIR:
    types: dict[str, Type] = {**ir.inputs, **ir.state}
    state = dict(ir.state)
    reset_state = dict(ir.reset_state)
    next_state = dict(ir.next_state)

    cache: dict[tuple[str, int, str], str] = {}
    next_id = 0

    def lower_expr(expr: Expr) -> Expr:
        nonlocal next_id
        if isinstance(expr, Delay):
            if expr.ticks < 1:
                raise DelayLowerError("delay ticks must be >= 1")
            x = lower_expr(expr.x)
            t = infer_type(x, types)
            key = (repr(x), int(expr.ticks), repr(t))
            existing = cache.get(key)
            if existing is not None:
                return Var(existing)

            base = next_id
            next_id += 1
            prev = None
            for i in range(int(expr.ticks)):
                name = f"__delay_{base}_{i}"
                state[name] = t
                reset_state[name] = _zero(t)
                if i == 0:
                    next_state[name] = x
                else:
                    assert prev is not None
                    next_state[name] = Var(prev)
                types[name] = t
                prev = name
            assert prev is not None
            cache[key] = prev
            return Var(prev)

        if isinstance(expr, (Var, BoolConst, BitVecConst, SimdConst)):
            return expr

        if not is_dataclass(expr):
            raise DelayLowerError("unsupported expression kind")

        kwargs: dict[str, Any] = {}
        for f in fields(expr):
            v = getattr(expr, f.name)
            if isinstance(v, list):
                out: list[Any] = []
                for item in v:
                    out.append(
                        lower_expr(item) if isinstance(item, EXPR_CLASSES) else item
                    )
                kwargs[f.name] = out
            elif isinstance(v, EXPR_CLASSES):
                kwargs[f.name] = lower_expr(v)
            else:
                kwargs[f.name] = v
        return expr.__class__(**kwargs)

    output_exprs = {k: lower_expr(v) for k, v in ir.output_exprs.items()}

    for name in list(next_state.keys()):
        next_state[name] = lower_expr(next_state[name])

    return TickIR(
        name=ir.name,
        inputs=dict(ir.inputs),
        outputs=dict(ir.outputs),
        state=state,
        reset_state=reset_state,
        next_state=next_state,
        output_exprs=output_exprs,
    )
