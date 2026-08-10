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
        return ("expr", id(v))
    if isinstance(v, (list, tuple)):
        return ("seq", tuple(_key_value(x) for x in v))
    raise TypeError("unsupported key value")


_FIELDS_CACHE: dict[type, tuple] = {}


def _get_fields(cls: type) -> tuple:
    cached = _FIELDS_CACHE.get(cls)
    if cached is not None:
        return cached
    cached = tuple(fields(cls))
    _FIELDS_CACHE[cls] = cached
    return cached


def _key_expr(expr: Expr) -> tuple[Any, ...]:
    if not is_dataclass(expr):
        raise TypeError("expr must be dataclass")
    items: list[tuple[Any, ...]] = []
    for f in _get_fields(expr.__class__):
        items.append((f.name, _key_value(getattr(expr, f.name))))
    return (expr.__class__.__name__, tuple(items))


def _iter_child_exprs(expr: Expr) -> list[Expr]:
    children: list[Expr] = []
    for f in _get_fields(expr.__class__):
        v = getattr(expr, f.name)
        if isinstance(v, EXPR_CLASSES):
            children.append(v)
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, EXPR_CLASSES):
                    children.append(item)
    return children


def _postorder_exprs(roots: list[Expr]) -> list[Expr]:
    order: list[Expr] = []
    seen: set[int] = set()
    done: set[int] = set()
    stack: list[tuple[Expr, bool]] = [(r, False) for r in roots]
    while stack:
        expr, expanded = stack.pop()
        eid = id(expr)
        if eid in done:
            continue
        if not expanded:
            if eid in seen:
                continue
            seen.add(eid)
            stack.append((expr, True))
            for child in _iter_child_exprs(expr):
                stack.append((child, False))
        else:
            done.add(eid)
            order.append(expr)
    return order


def _key_value_fast(v: object, canon_no: dict[int, int]) -> tuple[Any, ...]:
    if v is None:
        return ("none",)
    if isinstance(v, (bool, int, str)):
        return ("val", v)
    if isinstance(v, (BoolType, BitVecType, SimdType)):
        return ("type",) + _key_type(v)
    if isinstance(v, EXPR_CLASSES):
        return ("expr", canon_no[id(v)])
    if isinstance(v, (list, tuple)):
        out: list[Any] = []
        for item in v:
            if isinstance(item, EXPR_CLASSES):
                out.append(("expr", canon_no[id(item)]))
            else:
                out.append(_key_value_fast(item, canon_no))
        return ("seq", tuple(out))
    raise TypeError("unsupported key value")


def _key_expr_fast(expr: Expr, canon_no: dict[int, int]) -> tuple[Any, ...]:
    if not is_dataclass(expr):
        raise TypeError("expr must be dataclass")
    items: list[tuple[Any, ...]] = []
    for f in _get_fields(expr.__class__):
        items.append((f.name, _key_value_fast(getattr(expr, f.name), canon_no)))
    return (expr.__class__.__name__, tuple(items))


def _rebuild_expr(expr: Expr, canon_map: dict[int, Expr]) -> Expr:
    kwargs: dict[str, object] = {}
    for f in _get_fields(expr.__class__):
        v = getattr(expr, f.name)
        if isinstance(v, EXPR_CLASSES):
            kwargs[f.name] = canon_map[id(v)]
        elif isinstance(v, (list, tuple)):
            out: list[object] = []
            for item in v:
                if isinstance(item, EXPR_CLASSES):
                    out.append(canon_map[id(item)])
                else:
                    out.append(item)
            kwargs[f.name] = out
        else:
            kwargs[f.name] = v
    return expr.__class__(**kwargs)


def hashcons_tick_ir(ir: TickIR) -> TickIR:
    memo: dict[tuple[Any, ...], Expr] = {}
    roots: list[Expr] = (
        list(ir.reset_state.values())
        + list(ir.next_state.values())
        + list(ir.output_exprs.values())
    )
    order = _postorder_exprs(roots)
    canon_map: dict[int, Expr] = {}
    canon_no: dict[int, int] = {}
    next_no = 0
    for expr in order:
        key = _key_expr_fast(expr, canon_no)
        prev = memo.get(key)
        if prev is not None:
            canon = prev
        else:
            canon = _rebuild_expr(expr, canon_map)
            memo[key] = canon
            canon_no[id(canon)] = next_no
            next_no += 1
        canon_map[id(expr)] = canon
        canon_no[id(expr)] = canon_no[id(canon)]
    return TickIR(
        name=ir.name,
        inputs=dict(ir.inputs),
        outputs=dict(ir.outputs),
        state=dict(ir.state),
        reset_state={k: canon_map[id(v)] for k, v in ir.reset_state.items()},
        next_state={k: canon_map[id(v)] for k, v in ir.next_state.items()},
        output_exprs={k: canon_map[id(v)] for k, v in ir.output_exprs.items()},
    )
