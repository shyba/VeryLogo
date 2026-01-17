from __future__ import annotations

from dataclasses import dataclass

from stc.interp import infer_type
from stc.tick_ir import (
    Bitcast,
    BitVecConst,
    BitVecType,
    BoolType,
    Concat,
    EXPR_CLASSES,
    Expr,
    SimdConst,
    SimdType,
    TickIR,
    Type,
    Var,
    Slice,
)


@dataclass(frozen=True)
class InferSimdError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def _collect_slices(expr: Expr, out: dict[str, list[tuple[int, int]]]) -> None:
    if isinstance(expr, Slice) and isinstance(expr.x, Var):
        if expr.width > 1:
            out.setdefault(expr.x.name, []).append((expr.offset, expr.width))
    for field_name in getattr(expr, "__dataclass_fields__", {}):
        v = getattr(expr, field_name)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, EXPR_CLASSES):
                    _collect_slices(item, out)
        elif isinstance(v, EXPR_CLASSES):
            _collect_slices(v, out)


def _infer_simd_for_var(
    name: str, t: Type, slices: list[tuple[int, int]]
) -> SimdType | None:
    if not isinstance(t, BitVecType):
        return None
    if not slices:
        return None
    widths = {w for _, w in slices}
    if len(widths) != 1:
        return None
    lane_width = next(iter(widths))
    if lane_width < 1:
        return None
    if t.width % lane_width != 0:
        return None
    lanes = t.width // lane_width
    if lanes < 2:
        return None
    for off, w in slices:
        if w != lane_width:
            return None
        if off % lane_width != 0:
            return None
        if off + w > t.width:
            return None
    return SimdType(lane_width=lane_width, lanes=lanes)


def _infer_output_simd(
    output_t: BitVecType, expr: Expr, types: dict[str, Type]
) -> SimdType | None:
    if not isinstance(expr, Concat):
        return None
    parts = expr.parts
    if not parts:
        return None
    widths = []
    for p in parts:
        try:
            t = infer_type(p, types)
        except Exception:
            return None
        if isinstance(t, BitVecType):
            widths.append(t.width)
        elif isinstance(t, BoolType):
            widths.append(1)
        else:
            return None
    if len(set(widths)) != 1:
        return None
    lane_width = widths[0]
    if output_t.width != lane_width * len(parts):
        return None
    if len(parts) < 2:
        return None
    return SimdType(lane_width=lane_width, lanes=len(parts))


def _wrap_to_type(expr: Expr, want: Type) -> Expr:
    if isinstance(want, BitVecType):
        return expr
    assert isinstance(want, SimdType)
    return Bitcast(to=want, x=expr)


def infer_simd_types(ir: TickIR) -> TickIR:
    slice_uses: dict[str, list[tuple[int, int]]] = {}
    for e in list(ir.output_exprs.values()) + list(ir.next_state.values()):
        _collect_slices(e, slice_uses)

    inputs = dict(ir.inputs)
    outputs = dict(ir.outputs)
    state = dict(ir.state)

    for name, t in list(inputs.items()):
        simd = _infer_simd_for_var(name, t, slice_uses.get(name, []))
        if simd is not None:
            inputs[name] = simd

    for name, t in list(state.items()):
        simd = _infer_simd_for_var(name, t, slice_uses.get(name, []))
        if simd is not None:
            state[name] = simd

    pre_types: dict[str, Type] = {**inputs, **state}
    for name, t in list(outputs.items()):
        if not isinstance(t, BitVecType):
            continue
        simd = _infer_output_simd(t, ir.output_exprs[name], pre_types)
        if simd is not None:
            outputs[name] = simd

    reset_state: dict[str, Expr] = {}
    for reg, t in state.items():
        expr = ir.reset_state[reg]
        if isinstance(t, SimdType):
            if isinstance(expr, BitVecConst):
                reset_state[reg] = SimdConst(
                    lane_width=t.lane_width, lanes=t.lanes, value=expr.value
                )
            elif isinstance(expr, SimdConst):
                reset_state[reg] = expr
            else:
                raise InferSimdError("simd reset_state must be constant bitvector")
        else:
            reset_state[reg] = expr

    next_state: dict[str, Expr] = {}
    for reg, expr in ir.next_state.items():
        want = state[reg]
        next_state[reg] = _wrap_to_type(expr, want)

    output_exprs: dict[str, Expr] = {}
    for name, expr in ir.output_exprs.items():
        want = outputs[name]
        output_exprs[name] = _wrap_to_type(expr, want)

    return TickIR(
        name=ir.name,
        inputs=inputs,
        outputs=outputs,
        state=state,
        reset_state=reset_state,
        next_state=next_state,
        output_exprs=output_exprs,
    )
