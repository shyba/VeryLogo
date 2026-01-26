from __future__ import annotations

from stc.tick_ir import (
    Add,
    And,
    AShr,
    BitVecConst,
    BitVecType,
    BoolType,
    Concat,
    Expr,
    FloatType,
    LShr,
    Mux,
    Or,
    Rotl,
    Rotr,
    Shl,
    SimdType,
    Slice,
    Sub,
    TickIR,
    Type,
    Var,
)
from stc.interp import infer_type


def _is_const(expr: Expr) -> bool:
    from stc.tick_ir import BitVecConst, BoolConst, FloatConst, SimdConst

    return isinstance(expr, (BoolConst, BitVecConst, FloatConst, SimdConst))


def _extract_const_value(expr: Expr) -> int | None:
    from stc.tick_ir import BitVecConst

    if isinstance(expr, BitVecConst):
        return expr.value
    return None


def _simplify_expr(expr: Expr, types: dict[str, Type]) -> Expr:
    from stc.tick_ir import (
        BitVecConst,
        BoolConst,
        FloatConst,
        SimdConst,
        TernaryLut,
    )

    if isinstance(expr, Var):
        return expr

    if isinstance(expr, (BoolConst, BitVecConst, FloatConst, SimdConst)):
        return expr

    if isinstance(expr, (And, Or)):
        a = _simplify_expr(expr.a, types)
        b = _simplify_expr(expr.b, types)

        if isinstance(a, BitVecConst) and isinstance(b, BitVecConst):
            t = infer_type(a, types)
            if isinstance(t, BitVecType):
                mask = (1 << t.width) - 1
                if isinstance(expr, And):
                    return BitVecConst(width=t.width, value=(a.value & b.value) & mask)
                return BitVecConst(width=t.width, value=(a.value | b.value) & mask)

        return expr.__class__(a=a, b=b)

    if isinstance(expr, (Add, Sub, Shl, LShr, AShr)):
        a = _simplify_expr(expr.a, types)
        b = _simplify_expr(expr.b, types)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, Mux):
        cond = _simplify_expr(expr.cond, types)
        a = _simplify_expr(expr.a, types)
        b = _simplify_expr(expr.b, types)
        return Mux(cond=cond, a=a, b=b)

    if isinstance(expr, Concat):
        parts = [_simplify_expr(p, types) for p in expr.parts]
        return Concat(parts=parts)

    if isinstance(expr, Slice):
        x = _simplify_expr(expr.x, types)
        return Slice(x=x, offset=expr.offset, width=expr.width)

    if isinstance(expr, TernaryLut):
        a = _simplify_expr(expr.a, types)
        b = _simplify_expr(expr.b, types)
        c = _simplify_expr(expr.c, types)
        return TernaryLut(a=a, b=b, c=c, imm8=expr.imm8)

    return expr


def _recognize_rotate(expr: Expr, types: dict[str, Type]) -> Expr | None:
    if not isinstance(expr, Or):
        return None

    left_shift = None
    right_shift = None

    if isinstance(expr.a, Shl):
        left_shift = expr.a
    if isinstance(expr.b, Shl):
        left_shift = expr.b

    if isinstance(expr.a, LShr):
        right_shift = expr.a
    if isinstance(expr.b, LShr):
        right_shift = expr.b

    if left_shift is None or right_shift is None:
        return None

    if left_shift.a != right_shift.a:
        return None

    x = left_shift.a
    x_type = infer_type(x, types)
    if not isinstance(x_type, BitVecType):
        return None

    width = x_type.width

    left_amt = _extract_const_value(left_shift.b)
    right_amt = _extract_const_value(right_shift.b)

    if left_amt is not None and right_amt is not None:
        if left_amt + right_amt == width:
            return Rotl(x=x, sh=left_shift.b)

    if isinstance(right_shift.b, Sub):
        if isinstance(right_shift.b.a, BitVecConst):
            if right_shift.b.a.value == width and right_shift.b.b == left_shift.b:
                return Rotl(x=x, sh=left_shift.b)

    if isinstance(left_shift.b, Sub):
        if isinstance(left_shift.b.a, BitVecConst):
            if left_shift.b.a.value == width and left_shift.b.b == right_shift.b:
                return Rotr(x=x, sh=right_shift.b)

    return None


def _normalize_concat_slice(expr: Expr, types: dict[str, Type]) -> Expr:
    if isinstance(expr, Concat):
        parts = [_normalize_concat_slice(p, types) for p in expr.parts]

        flattened = []
        for part in parts:
            if isinstance(part, Concat):
                flattened.extend(part.parts)
            else:
                flattened.append(part)

        if len(flattened) == 1:
            return flattened[0]

        return Concat(parts=flattened)

    if isinstance(expr, Slice):
        x = _normalize_concat_slice(expr.x, types)

        if isinstance(x, Concat):
            part_widths = []
            for part in x.parts:
                part_type = infer_type(part, types)
                if isinstance(part_type, BoolType):
                    part_widths.append(1)
                elif isinstance(part_type, BitVecType):
                    part_widths.append(part_type.width)
                else:
                    return Slice(x=x, offset=expr.offset, width=expr.width)

            total_width = sum(part_widths)
            slice_start = expr.offset
            slice_end = expr.offset + expr.width

            parts_consumed = []
            current_pos = 0
            for i in range(len(x.parts) - 1, -1, -1):
                part = x.parts[i]
                part_width = part_widths[i]
                part_start = current_pos
                part_end = current_pos + part_width

                if slice_end <= part_start:
                    current_pos = part_end
                    continue

                if slice_start >= part_end:
                    current_pos = part_end
                    continue

                local_start = max(0, slice_start - part_start)
                local_end = min(part_width, slice_end - part_start)
                local_width = local_end - local_start

                if local_start == 0 and local_width == part_width:
                    parts_consumed.insert(0, part)
                else:
                    parts_consumed.insert(
                        0, Slice(x=part, offset=local_start, width=local_width)
                    )

                current_pos = part_end

            if len(parts_consumed) == 1:
                return parts_consumed[0]
            if len(parts_consumed) > 1:
                return Concat(parts=parts_consumed)

        return Slice(x=x, offset=expr.offset, width=expr.width)

    if isinstance(expr, (And, Or, Add, Sub, Shl, LShr, AShr)):
        a = _normalize_concat_slice(expr.a, types)
        b = _normalize_concat_slice(expr.b, types)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, Mux):
        cond = _normalize_concat_slice(expr.cond, types)
        a = _normalize_concat_slice(expr.a, types)
        b = _normalize_concat_slice(expr.b, types)
        return Mux(cond=cond, a=a, b=b)

    return expr


def _normalize_masks(expr: Expr, types: dict[str, Type]) -> Expr:
    if isinstance(expr, And):
        a = _normalize_masks(expr.a, types)
        b = _normalize_masks(expr.b, types)

        if isinstance(a, BitVecConst):
            a, b = b, a

        if isinstance(b, BitVecConst):
            b_val = b.value
            if b_val == 0:
                return b
            t = infer_type(a, types)
            if isinstance(t, BitVecType):
                mask = (1 << t.width) - 1
                if b_val == mask:
                    return a

        return And(a=a, b=b)

    if isinstance(expr, Or):
        a = _normalize_masks(expr.a, types)
        b = _normalize_masks(expr.b, types)
        return Or(a=a, b=b)

    if isinstance(expr, (Add, Sub, Shl, LShr, AShr)):
        a = _normalize_masks(expr.a, types)
        b = _normalize_masks(expr.b, types)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, Mux):
        cond = _normalize_masks(expr.cond, types)
        a = _normalize_masks(expr.a, types)
        b = _normalize_masks(expr.b, types)
        return Mux(cond=cond, a=a, b=b)

    if isinstance(expr, Concat):
        parts = [_normalize_masks(p, types) for p in expr.parts]
        return Concat(parts=parts)

    if isinstance(expr, Slice):
        x = _normalize_masks(expr.x, types)
        return Slice(x=x, offset=expr.offset, width=expr.width)

    return expr


def _normalize_shifts(expr: Expr, types: dict[str, Type]) -> Expr:
    if isinstance(expr, (Shl, LShr, AShr)):
        a = _normalize_shifts(expr.a, types)
        b = _normalize_shifts(expr.b, types)

        shift_amt = _extract_const_value(b)
        if shift_amt is not None:
            a_type = infer_type(a, types)
            if isinstance(a_type, BitVecType):
                width = a_type.width
                if shift_amt >= width:
                    if isinstance(expr, (Shl, LShr)):
                        return BitVecConst(width=width, value=0)
                    if isinstance(a, BitVecConst):
                        sign_bit = (a.value >> (width - 1)) & 1
                        if sign_bit:
                            return BitVecConst(width=width, value=(1 << width) - 1)
                        return BitVecConst(width=width, value=0)

        return expr.__class__(a=a, b=b)

    if isinstance(expr, (And, Or, Add, Sub)):
        a = _normalize_shifts(expr.a, types)
        b = _normalize_shifts(expr.b, types)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, Mux):
        cond = _normalize_shifts(expr.cond, types)
        a = _normalize_shifts(expr.a, types)
        b = _normalize_shifts(expr.b, types)
        return Mux(cond=cond, a=a, b=b)

    if isinstance(expr, Concat):
        parts = [_normalize_shifts(p, types) for p in expr.parts]
        return Concat(parts=parts)

    if isinstance(expr, Slice):
        x = _normalize_shifts(expr.x, types)
        return Slice(x=x, offset=expr.offset, width=expr.width)

    return expr


def _normalize_mux_trees(expr: Expr, types: dict[str, Type]) -> Expr:
    if isinstance(expr, Mux):
        cond = _normalize_mux_trees(expr.cond, types)
        a = _normalize_mux_trees(expr.a, types)
        b = _normalize_mux_trees(expr.b, types)

        if a == b:
            return a

        return Mux(cond=cond, a=a, b=b)

    if isinstance(expr, (And, Or, Add, Sub, Shl, LShr, AShr)):
        a = _normalize_mux_trees(expr.a, types)
        b = _normalize_mux_trees(expr.b, types)
        return expr.__class__(a=a, b=b)

    if isinstance(expr, Concat):
        parts = [_normalize_mux_trees(p, types) for p in expr.parts]
        return Concat(parts=parts)

    if isinstance(expr, Slice):
        x = _normalize_mux_trees(expr.x, types)
        return Slice(x=x, offset=expr.offset, width=expr.width)

    return expr


def normalize_expr(expr: Expr, types: dict[str, Type]) -> Expr:
    expr = _simplify_expr(expr, types)

    rotate = _recognize_rotate(expr, types)
    if rotate is not None:
        return rotate

    expr = _normalize_concat_slice(expr, types)
    expr = _normalize_masks(expr, types)
    expr = _normalize_shifts(expr, types)
    expr = _normalize_mux_trees(expr, types)

    return expr


def normalize_tick_ir(ir: TickIR) -> TickIR:
    types = {**ir.inputs, **ir.state}

    normalized_next = {}
    for name, expr in ir.next_state.items():
        normalized_next[name] = normalize_expr(expr, types)

    normalized_output = {}
    for name, expr in ir.output_exprs.items():
        normalized_output[name] = normalize_expr(expr, types)

    return TickIR(
        name=ir.name,
        inputs=dict(ir.inputs),
        outputs=dict(ir.outputs),
        state=dict(ir.state),
        reset_state=dict(ir.reset_state),
        next_state=normalized_next,
        output_exprs=normalized_output,
    )
