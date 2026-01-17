from __future__ import annotations

from dataclasses import dataclass

import z3

from stc.interp import infer_type
from stc.tick_ir import (
    Add,
    Sub,
    And,
    Bitcast,
    BitVecType,
    Concat,
    Eq,
    Expr,
    Mux,
    Or,
    SimdAdd,
    SimdBlend,
    SimdMaxU,
    SimdMinU,
    SimdSub,
    SimdEq,
    SimdType,
    SimdUge,
    SimdUgt,
    SimdUle,
    SimdUlt,
    Slice,
    Type,
    Uge,
    Ugt,
    Ule,
    Ult,
    Var,
    Xor,
)
from stc.tick_ir_validate import type_equal
from stc.z3_encode import encode_expr


@dataclass(frozen=True)
class AutovecError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def _equiv(spec: Expr, cand: Expr, types: dict[str, Type], *, timeout_ms: int) -> bool:
    z3_vars: dict[str, z3.ExprRef] = {}
    spec_z = encode_expr(spec, types, z3_vars)
    cand_z = encode_expr(cand, types, z3_vars)
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(spec_z != cand_z)
    return solver.check() == z3.unsat


def _bitcast_var(expr: Expr, total_width: int) -> str | None:
    raise RuntimeError("use _src_var")


def _src_var(expr: Expr, total_width: int, types: dict[str, Type]) -> str | None:
    if isinstance(expr, Var):
        t = types.get(expr.name)
        if isinstance(t, SimdType) and t.total_width == total_width:
            return expr.name
        return None

    if isinstance(expr, Bitcast):
        if not isinstance(expr.to, BitVecType) or expr.to.width != total_width:
            return None
        if not isinstance(expr.x, Var):
            return None
        t = types.get(expr.x.name)
        if isinstance(t, SimdType) and t.total_width == total_width:
            return expr.x.name
        return None

    return None


def _match_lane_binop(
    lane_expr: Expr,
    *,
    op: type,
    lane_width: int,
    lane_offset: int,
    total_width: int,
    types: dict[str, Type],
) -> tuple[str, str] | None:
    if not isinstance(lane_expr, op):
        return None
    a = lane_expr.a
    b = lane_expr.b
    if not (isinstance(a, Slice) and isinstance(b, Slice)):
        return None
    if not (a.offset == lane_offset and b.offset == lane_offset):
        return None
    if not (a.width == lane_width and b.width == lane_width):
        return None

    a_src = _src_var(a.x, total_width, types)
    b_src = _src_var(b.x, total_width, types)
    if a_src is None or b_src is None:
        return None
    return (a_src, b_src)


def autovectorize_expr(
    spec: Expr, types: dict[str, Type], *, timeout_ms: int = 2000
) -> Expr:
    out_t = infer_type(spec, types)
    if not isinstance(out_t, SimdType):
        return spec

    if not isinstance(spec, Bitcast):
        return spec
    if not isinstance(spec.x, Concat):
        return spec
    if not type_equal(spec.to, out_t):
        return spec

    lane_width = out_t.lane_width
    lanes = out_t.lanes
    total_width = out_t.total_width

    if len(spec.x.parts) != lanes:
        return spec

    for op, make in [
        (Add, lambda a, b: SimdAdd(a=Var(name=a), b=Var(name=b))),
        (Sub, lambda a, b: SimdSub(a=Var(name=a), b=Var(name=b))),
        (Xor, lambda a, b: Xor(a=Var(name=a), b=Var(name=b))),
        (And, lambda a, b: And(a=Var(name=a), b=Var(name=b))),
        (Or, lambda a, b: Or(a=Var(name=a), b=Var(name=b))),
    ]:
        pair: tuple[str, str] | None = None
        ok = True
        for idx, lane_expr in enumerate(spec.x.parts):
            lane = lanes - 1 - idx
            lane_offset = lane * lane_width
            m = _match_lane_binop(
                lane_expr,
                op=op,
                lane_width=lane_width,
                lane_offset=lane_offset,
                total_width=total_width,
                types=types,
            )
            if m is None:
                ok = False
                break
            if pair is None:
                pair = m
            elif pair != m:
                ok = False
                break

        if ok and pair is not None:
            cand = make(pair[0], pair[1])
            if _equiv(spec, cand, types, timeout_ms=timeout_ms):
                return cand

    if lane_width == 1:
        pair = None
        src_lane_width = None
        cmp_cls = None
        ok = True
        for idx, lane_expr in enumerate(spec.x.parts):
            if not isinstance(lane_expr, (Eq, Ult, Ule, Ugt, Uge)):
                ok = False
                break
            if cmp_cls is None:
                cmp_cls = lane_expr.__class__
            elif lane_expr.__class__ is not cmp_cls:
                ok = False
                break
            a = lane_expr.a
            b = lane_expr.b
            if not (isinstance(a, Slice) and isinstance(b, Slice)):
                ok = False
                break
            if src_lane_width is None:
                src_lane_width = a.width
            if not (a.width == src_lane_width and b.width == src_lane_width):
                ok = False
                break
            lane = lanes - 1 - idx
            lane_offset = lane * src_lane_width
            if not (a.offset == lane_offset and b.offset == lane_offset):
                ok = False
                break

            src_total_width = src_lane_width * lanes
            a_src = _src_var(a.x, src_total_width, types)
            b_src = _src_var(b.x, src_total_width, types)
            if a_src is None or b_src is None:
                ok = False
                break
            cur = (a_src, b_src)
            if pair is None:
                pair = cur
            elif pair != cur:
                ok = False
                break

        if ok and pair is not None and src_lane_width is not None:
            if cmp_cls is Eq:
                cand = SimdEq(a=Var(name=pair[0]), b=Var(name=pair[1]))
            elif cmp_cls is Ult:
                cand = SimdUlt(a=Var(name=pair[0]), b=Var(name=pair[1]))
            elif cmp_cls is Ule:
                cand = SimdUle(a=Var(name=pair[0]), b=Var(name=pair[1]))
            elif cmp_cls is Ugt:
                cand = SimdUgt(a=Var(name=pair[0]), b=Var(name=pair[1]))
            else:
                cand = SimdUge(a=Var(name=pair[0]), b=Var(name=pair[1]))
            if _equiv(spec, cand, types, timeout_ms=timeout_ms):
                return cand

    if lane_width > 1:
        pair = None
        ok = True
        swapped = None
        for idx, lane_expr in enumerate(spec.x.parts):
            if not isinstance(lane_expr, Mux):
                ok = False
                break
            lane = lanes - 1 - idx
            lane_offset = lane * lane_width
            m = _match_lane_binop(
                lane_expr.cond,
                op=Ult,
                lane_width=lane_width,
                lane_offset=lane_offset,
                total_width=total_width,
                types=types,
            )
            if m is None:
                ok = False
                break
            if pair is None:
                pair = m
            elif pair != m:
                ok = False
                break

            a_src = (
                _src_var(lane_expr.a.x, total_width, types)
                if isinstance(lane_expr.a, Slice)
                else None
            )
            b_src = (
                _src_var(lane_expr.b.x, total_width, types)
                if isinstance(lane_expr.b, Slice)
                else None
            )
            if (
                a_src is None
                or b_src is None
                or not (
                    lane_expr.a.offset == lane_offset
                    and lane_expr.a.width == lane_width
                )
                or not (
                    lane_expr.b.offset == lane_offset
                    and lane_expr.b.width == lane_width
                )
            ):
                ok = False
                break

            if (a_src, b_src) == pair:
                cur_swapped = False
            elif (b_src, a_src) == pair:
                cur_swapped = True
            else:
                ok = False
                break

            if swapped is None:
                swapped = cur_swapped
            elif swapped != cur_swapped:
                ok = False
                break

        if ok and pair is not None and swapped is not None:
            a_src, b_src = pair
            cand = (
                SimdMaxU(a=Var(name=a_src), b=Var(name=b_src))
                if swapped
                else SimdMinU(a=Var(name=a_src), b=Var(name=b_src))
            )
            if _equiv(spec, cand, types, timeout_ms=timeout_ms):
                return cand

        cmp_pair = None
        branch_pair = None
        ok = True
        swapped = None
        for idx, lane_expr in enumerate(spec.x.parts):
            if not isinstance(lane_expr, Mux):
                ok = False
                break
            lane = lanes - 1 - idx
            lane_offset = lane * lane_width
            m = _match_lane_binop(
                lane_expr.cond,
                op=Eq,
                lane_width=lane_width,
                lane_offset=lane_offset,
                total_width=total_width,
                types=types,
            )
            if m is None:
                ok = False
                break
            if cmp_pair is None:
                cmp_pair = m
            elif cmp_pair != m:
                ok = False
                break

            if not isinstance(lane_expr.a, Slice) or not isinstance(lane_expr.b, Slice):
                ok = False
                break
            if not (
                lane_expr.a.offset == lane_offset
                and lane_expr.a.width == lane_width
                and lane_expr.b.offset == lane_offset
                and lane_expr.b.width == lane_width
            ):
                ok = False
                break

            a_src = _src_var(lane_expr.a.x, total_width, types)
            b_src = _src_var(lane_expr.b.x, total_width, types)
            if a_src is None or b_src is None:
                ok = False
                break

            if branch_pair is None:
                branch_pair = (a_src, b_src)
                swapped = False
            elif branch_pair == (a_src, b_src):
                pass
            elif branch_pair == (b_src, a_src):
                if swapped is False:
                    swapped = True
            else:
                ok = False
                break

            if swapped:
                ok = False
                break

        if ok and cmp_pair is not None and branch_pair is not None:
            ca, cb = cmp_pair
            xa, xb = branch_pair
            cand = SimdBlend(
                mask=SimdEq(a=Var(name=ca), b=Var(name=cb)),
                a=Var(name=xb),
                b=Var(name=xa),
            )
            if _equiv(spec, cand, types, timeout_ms=timeout_ms):
                return cand

    return spec
