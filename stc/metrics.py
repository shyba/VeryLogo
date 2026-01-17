from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass

from stc.tick_ir import (
    BitVecType,
    BitVecConst,
    BoolConst,
    BoolType,
    Expr,
    EXPR_CLASSES,
    SimdConst,
    SimdType,
    TickIR,
    Var,
)


@dataclass(frozen=True)
class IRMetrics:
    state_regs: int
    state_bits: int
    ops_total: int
    expr_nodes_total: int
    expr_depth_max: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _expr_children(expr: Expr) -> list[Expr]:
    if not is_dataclass(expr):
        return []
    out: list[Expr] = []
    for f in fields(expr):
        v = getattr(expr, f.name)
        if isinstance(v, EXPR_CLASSES):
            out.append(v)
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, EXPR_CLASSES):
                    out.append(item)
    return out


def _count_ops(expr: Expr) -> int:
    if isinstance(expr, (Var, BoolConst, BitVecConst, SimdConst)):
        return 0
    children = _expr_children(expr)
    if not children:
        return 0
    return 1 + sum(_count_ops(c) for c in children)


def _count_nodes(expr: Expr) -> int:
    total = 1
    for child in _expr_children(expr):
        total += _count_nodes(child)
    return total


def _depth(expr: Expr) -> int:
    children = _expr_children(expr)
    if not children:
        return 1
    return 1 + max(_depth(c) for c in children)


def compute_metrics(ir: TickIR) -> IRMetrics:
    state_regs = len(ir.state)
    state_bits = 0
    for t in ir.state.values():
        if isinstance(t, BoolType):
            state_bits += 1
        elif isinstance(t, BitVecType):
            state_bits += t.width
        else:
            assert isinstance(t, SimdType)
            state_bits += t.total_width

    exprs = list(ir.next_state.values()) + list(ir.output_exprs.values())
    ops_total = sum(_count_ops(e) for e in exprs)
    expr_nodes_total = sum(_count_nodes(e) for e in exprs)
    expr_depth_max = max((_depth(e) for e in exprs), default=0)
    return IRMetrics(
        state_regs=state_regs,
        state_bits=state_bits,
        ops_total=ops_total,
        expr_nodes_total=expr_nodes_total,
        expr_depth_max=expr_depth_max,
    )
