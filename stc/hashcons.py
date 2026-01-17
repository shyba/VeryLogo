from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from stc.tick_ir import BitVecType, BoolType, EXPR_CLASSES, Expr, SimdType, TickIR, Type


def _key_type(t: Type) -> tuple[Any, ...]:
    if isinstance(t, BoolType):
        return ("bool",)
    if isinstance(t, BitVecType):
        return ("bitvec", t.width)
    assert isinstance(t, SimdType)
    return ("simd", t.lane_width, t.lanes)


def _key_value(v: object) -> tuple[Any, ...]:
    if v is None:
        return ("none",)
    if isinstance(v, (bool, int, str)):
        return ("val", v)
    if isinstance(v, (BoolType, BitVecType, SimdType)):
        return ("type",) + _key_type(v)
    if isinstance(v, EXPR_CLASSES):
        return _key_expr(v)
    if isinstance(v, (list, tuple)):
        return ("seq", tuple(_key_value(x) for x in v))
    raise TypeError("unsupported key value")


def _key_expr(expr: Expr) -> tuple[Any, ...]:
    if not is_dataclass(expr):
        raise TypeError("expr must be dataclass")
    items: list[tuple[Any, ...]] = []
    for f in fields(expr):
        items.append((f.name, _key_value(getattr(expr, f.name))))
    return (expr.__class__.__name__, tuple(items))


def hashcons_expr(expr: Expr, memo: dict[tuple[Any, ...], Expr]) -> Expr:
    if isinstance(expr, (BoolType, BitVecType, SimdType)):
        raise TypeError("expected expression")
    if not is_dataclass(expr):
        raise TypeError("expr must be dataclass")

    kwargs: dict[str, object] = {}
    for f in fields(expr):
        v = getattr(expr, f.name)
        if isinstance(v, EXPR_CLASSES):
            kwargs[f.name] = hashcons_expr(v, memo)
        elif isinstance(v, list):
            out: list[object] = []
            for item in v:
                if isinstance(item, EXPR_CLASSES):
                    out.append(hashcons_expr(item, memo))
                else:
                    out.append(item)
            kwargs[f.name] = out
        else:
            kwargs[f.name] = v

    rebuilt = expr.__class__(**kwargs)
    k = _key_expr(rebuilt)
    prev = memo.get(k)
    if prev is not None:
        return prev
    memo[k] = rebuilt
    return rebuilt


def hashcons_tick_ir(ir: TickIR) -> TickIR:
    memo: dict[tuple[Any, ...], Expr] = {}
    return TickIR(
        name=ir.name,
        inputs=dict(ir.inputs),
        outputs=dict(ir.outputs),
        state=dict(ir.state),
        reset_state={k: hashcons_expr(v, memo) for k, v in ir.reset_state.items()},
        next_state={k: hashcons_expr(v, memo) for k, v in ir.next_state.items()},
        output_exprs={k: hashcons_expr(v, memo) for k, v in ir.output_exprs.items()},
    )
