from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from stc.tick_ir import (
    Bitcast,
    BitVecConst,
    BoolConst,
    EXPR_CLASSES,
    Expr,
    FloatConst,
    SimdConst,
    SimdType,
    TickIR,
    Type,
    Var,
)


@dataclass(frozen=True)
class TickIRValidationError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def iter_vars(expr: Expr) -> Iterable[str]:
    if isinstance(expr, Var):
        yield expr.name
        return
    if isinstance(expr, (BoolConst, BitVecConst, FloatConst, SimdConst)):
        return
    if isinstance(expr, Bitcast):
        yield from iter_vars(expr.x)
        return

    child_exprs: list[Expr] = []
    for field_name in getattr(expr, "__dataclass_fields__", {}):
        value = getattr(expr, field_name)
        if isinstance(value, (tuple, list)):
            child_exprs.extend([v for v in value if isinstance(v, EXPR_CLASSES)])
        elif isinstance(value, EXPR_CLASSES):
            child_exprs.append(value)

    for child in child_exprs:
        yield from iter_vars(child)


def _is_const(expr: Expr) -> bool:
    return isinstance(expr, (BoolConst, BitVecConst, FloatConst, SimdConst))


def validate_tick_ir(ir: TickIR) -> None:
    if not ir.name:
        raise TickIRValidationError("Tick-IR name is empty")

    if set(ir.output_exprs.keys()) != set(ir.outputs.keys()):
        raise TickIRValidationError("output_exprs keys must match outputs")

    if set(ir.next_state.keys()) != set(ir.state.keys()):
        raise TickIRValidationError("next_state keys must match state")

    if set(ir.reset_state.keys()) != set(ir.state.keys()):
        raise TickIRValidationError("reset_state keys must match state")

    for reg, expr in ir.reset_state.items():
        if not _is_const(expr):
            raise TickIRValidationError(f"reset_state for {reg} must be constant")

    allowed = set(ir.inputs.keys()) | set(ir.state.keys())

    for name, expr in list(ir.next_state.items()) + list(ir.output_exprs.items()):
        for var_name in iter_vars(expr):
            if var_name not in allowed:
                raise TickIRValidationError(f"unknown var {var_name} in {name}")


def type_equal(a: Type, b: Type) -> bool:
    if isinstance(a, type(b)) and isinstance(b, type(a)):
        if isinstance(a, SimdType):
            return a.lane_width == b.lane_width and a.lanes == b.lanes
        if hasattr(a, "width") and hasattr(b, "width"):
            return int(a.width) == int(b.width)
        return True
    return False
