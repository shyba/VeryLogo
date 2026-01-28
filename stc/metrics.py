from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import TYPE_CHECKING

from stc.tick_ir import (
    And,
    BitVecType,
    BitVecConst,
    BoolConst,
    BoolType,
    Expr,
    EXPR_CLASSES,
    Not,
    SimdConst,
    SimdType,
    TickIR,
    Var,
)

if TYPE_CHECKING:
    from stc.tech import DepthModel


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


def _get_op_name(expr: Expr) -> str:
    """Get operation name from expression type."""
    if isinstance(expr, And):
        return "and"
    elif isinstance(expr, Not):
        return "not"
    elif isinstance(expr, (Var, BoolConst, BitVecConst, SimdConst)):
        return "const"
    else:
        return expr.__class__.__name__.lower()


def compute_depth(expr: Expr, depth_model: DepthModel) -> int:
    """Compute expression depth using provided depth model."""
    children = _expr_children(expr)
    if not children:
        op_name = _get_op_name(expr)
        return depth_model.op_depth(op_name)

    child_depths = [compute_depth(c, depth_model) for c in children]
    op_name = _get_op_name(expr)
    return depth_model.op_depth(op_name) + max(child_depths)


def and_depth(expr: Expr) -> int:
    """Compute AND-depth (only AND gates count, for FHE/MPC)."""
    if isinstance(expr, And):
        children = _expr_children(expr)
        child_depths = [and_depth(c) for c in children]
        return 1 + max(child_depths)

    children = _expr_children(expr)
    if not children:
        return 0

    return max((and_depth(c) for c in children), default=0)


def multiplicative_depth(expr: Expr) -> int:
    """Compute multiplicative depth (AND gates only, for FHE)."""
    return and_depth(expr)


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

    # These metrics are frequently used on large DAGs (shared subexpressions),
    # so avoid recursive re-traversal by memoizing on object identity.
    seen: set[int] = set()
    depth_by_id: dict[int, int] = {}
    ops_total = 0
    expr_nodes_total = 0
    expr_depth_max = 0

    def _iter_children(e: Expr) -> list[Expr]:
        return _expr_children(e)

    for root in exprs:
        stack: list[tuple[Expr, int]] = [(root, 0)]
        while stack:
            e, phase = stack.pop()
            eid = id(e)
            if phase == 0:
                if eid in seen:
                    continue
                seen.add(eid)
                expr_nodes_total += 1
                stack.append((e, 1))
                for c in _iter_children(e):
                    stack.append((c, 0))
            else:
                children = _iter_children(e)
                if not children:
                    d = 1
                else:
                    d = 1 + max(depth_by_id[id(c)] for c in children)
                    if not isinstance(e, (Var, BoolConst, BitVecConst, SimdConst)):
                        ops_total += 1
                depth_by_id[eid] = d
                if d > expr_depth_max:
                    expr_depth_max = d

    return IRMetrics(
        state_regs=state_regs,
        state_bits=state_bits,
        ops_total=ops_total,
        expr_nodes_total=expr_nodes_total,
        expr_depth_max=expr_depth_max,
    )
