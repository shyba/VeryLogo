from __future__ import annotations

from dataclasses import dataclass

import z3

from stc.interp import eval_expr
from stc.interp import infer_type
from stc.cost import expr_cost
from stc.tick_ir import (
    Add,
    And,
    Bitcast,
    BitVecType,
    BoolType,
    Expr,
    Not,
    Or,
    Sub,
    SimdAdd,
    SimdBlend,
    SimdEq,
    SimdUge,
    SimdUgt,
    SimdUle,
    SimdUlt,
    SimdType,
    Type,
    Var,
    Xor,
    SimdSub,
    SimdAnd,
    SimdOr,
    SimdXor,
    SimdNot,
    SimdMinU,
    SimdMaxU,
    SimdMinS,
    SimdMaxS,
)
from stc.tick_ir_validate import iter_vars, type_equal
from stc.z3_encode import encode_expr


@dataclass(frozen=True)
class SuperoptError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class SuperoptStats:
    candidates_checked: int = 0
    solver_checks: int = 0
    counterexamples: int = 0


def _equiv(
    spec: Expr,
    cand: Expr,
    types: dict[str, Type],
    *,
    timeout_ms: int,
) -> bool:
    z3_vars: dict[str, z3.ExprRef] = {}
    spec_z = encode_expr(spec, types, z3_vars)
    cand_z = encode_expr(cand, types, z3_vars)
    diff = spec_z != cand_z
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(diff)
    return solver.check() == z3.unsat


def _cex(
    spec: Expr,
    cand: Expr,
    types: dict[str, Type],
    used_vars: list[str],
    *,
    timeout_ms: int,
) -> dict[str, int | bool] | None:
    z3_vars: dict[str, z3.ExprRef] = {}
    spec_z = encode_expr(spec, types, z3_vars)
    cand_z = encode_expr(cand, types, z3_vars)
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(spec_z != cand_z)
    r = solver.check()
    if r == z3.unsat:
        return None
    if r == z3.unknown:
        raise SuperoptError("z3 returned unknown")
    model = solver.model()
    env: dict[str, int | bool] = {}
    for name in used_vars:
        z = z3_vars.get(name)
        if z is None:
            raise SuperoptError("model missing var")
        v = model.eval(z, model_completion=True)
        t = types[name]
        if isinstance(t, BoolType):
            env[name] = bool(z3.is_true(v))
        else:
            assert isinstance(v, z3.BitVecNumRef)
            env[name] = int(v.as_long())
    return env


def superopt_expr(
    spec: Expr,
    types: dict[str, Type],
    *,
    max_nodes: int = 6,
    constants: list[Expr] | None = None,
    timeout_ms: int = 2000,
    max_candidates_per_size: int = 5000,
    max_total_candidates: int = 20000,
    use_cegis: bool = True,
    max_counterexamples: int = 8,
    stats: SuperoptStats | None = None,
) -> Expr:
    if max_nodes < 1:
        raise SuperoptError("max_nodes must be >= 1")
    if max_candidates_per_size < 1:
        raise SuperoptError("max_candidates_per_size must be >= 1")
    if max_total_candidates < 1:
        raise SuperoptError("max_total_candidates must be >= 1")
    if max_counterexamples < 1:
        raise SuperoptError("max_counterexamples must be >= 1")

    out_t = infer_type(spec, types)
    used_vars = sorted(set(iter_vars(spec)))
    for name in used_vars:
        if name not in types:
            raise SuperoptError(f"unknown var {name}")

    if constants is None:
        constants = []

    def supports_superopt_t(t: Type) -> bool:
        return isinstance(t, (BoolType, BitVecType, SimdType))

    def type_width(t: Type) -> int | None:
        if isinstance(t, BitVecType):
            return t.width
        if isinstance(t, SimdType):
            return t.total_width
        return None

    def allow_bitcast(src_t: Type, dst_t: Type) -> bool:
        if not (
            isinstance(src_t, (BitVecType, SimdType))
            and isinstance(dst_t, (BitVecType, SimdType))
        ):
            return False
        if isinstance(src_t, BitVecType) and isinstance(dst_t, BitVecType):
            return False
        if isinstance(src_t, SimdType) and isinstance(dst_t, SimdType):
            return False
        return type_width(src_t) == type_width(dst_t)

    allowed_types: set[Type] = set()
    if supports_superopt_t(out_t):
        allowed_types.add(out_t)
    for name in used_vars:
        t = types[name]
        if supports_superopt_t(t):
            allowed_types.add(t)

    for t in list(allowed_types):
        if isinstance(t, SimdType):
            allowed_types.add(SimdType(lane_width=1, lanes=t.lanes))

    base_by_type: dict[Type, list[Expr]] = {t: [] for t in allowed_types}
    for name in used_vars:
        t = types[name]
        if t in base_by_type:
            base_by_type[t].append(Var(name=name))
    for c in constants:
        try:
            ct = infer_type(c, types)
        except Exception:
            continue
        if ct in base_by_type:
            base_by_type[ct].append(c)

    if not any(v for v in base_by_type.values()):
        raise SuperoptError("no base terms for any relevant type")

    by_size: dict[tuple[Type, int], list[Expr]] = {}
    for t, base in base_by_type.items():
        if base:
            by_size[(t, 1)] = sorted(set(base), key=repr)

    checked = 0
    counterexamples: list[dict[str, int | bool]] = []
    spec_vals: list[object] = []

    best = None
    best_cost = None

    for cand in by_size.get((out_t, 1), []):
        if stats is not None:
            stats.candidates_checked += 1
        checked += 1
        if checked > max_total_candidates:
            raise SuperoptError("candidate cap exceeded")
        if use_cegis and counterexamples:
            ok = True
            for ce, sv in zip(counterexamples, spec_vals, strict=True):
                if eval_expr(cand, types, ce) != sv:
                    ok = False
                    break
            if not ok:
                continue
        if stats is not None:
            stats.solver_checks += 1
        ce = _cex(spec, cand, types, used_vars, timeout_ms=timeout_ms)
        if ce is None:
            c = expr_cost(cand, types)
            if best is None or best_cost is None or c < best_cost:
                best = cand
                best_cost = c
        elif use_cegis and len(counterexamples) < max_counterexamples:
            counterexamples.append(ce)
            spec_vals.append(eval_expr(spec, types, ce))
            if stats is not None:
                stats.counterexamples = len(counterexamples)

    if best is not None:
        return best

    for size in range(2, max_nodes + 1):
        for t in allowed_types:
            cand_set: set[Expr] = set()

            for x in by_size.get((t, size - 1), []):
                if isinstance(t, (BoolType, BitVecType)):
                    cand_set.add(Not(x=x))
                elif isinstance(t, SimdType):
                    cand_set.add(SimdNot(x=x))

            for src_t in allowed_types:
                if allow_bitcast(src_t, t):
                    for x in by_size.get((src_t, size - 1), []):
                        cand_set.add(Bitcast(to=t, x=x))

            for left_size in range(1, size - 1):
                right_size = size - 1 - left_size

                xs = by_size.get((t, left_size), [])
                ys = by_size.get((t, right_size), [])
                if xs and ys:
                    lim = min(128, max_candidates_per_size)
                    xs = sorted(xs, key=lambda e: (expr_cost(e, types), repr(e)))[:lim]
                    ys = sorted(ys, key=lambda e: (expr_cost(e, types), repr(e)))[:lim]
                    for a in xs:
                        for b in ys:
                            if isinstance(t, BoolType):
                                cand_set.add(And(a=a, b=b))
                                cand_set.add(Or(a=a, b=b))
                                cand_set.add(Xor(a=a, b=b))
                            elif isinstance(t, BitVecType):
                                cand_set.add(And(a=a, b=b))
                                cand_set.add(Or(a=a, b=b))
                                cand_set.add(Xor(a=a, b=b))
                                cand_set.add(Add(a=a, b=b))
                                cand_set.add(Sub(a=a, b=b))
                            else:
                                assert isinstance(t, SimdType)
                                cand_set.add(SimdAnd(a=a, b=b))
                                cand_set.add(SimdOr(a=a, b=b))
                                cand_set.add(SimdXor(a=a, b=b))
                                if t.lane_width != 1:
                                    cand_set.add(SimdAdd(a=a, b=b))
                                    cand_set.add(SimdSub(a=a, b=b))
                                    cand_set.add(SimdMinU(a=a, b=b))
                                    cand_set.add(SimdMaxU(a=a, b=b))
                                    cand_set.add(SimdMinS(a=a, b=b))
                                    cand_set.add(SimdMaxS(a=a, b=b))

                if isinstance(t, SimdType) and t.lane_width == 1:
                    for src_t in allowed_types:
                        if (
                            isinstance(src_t, SimdType)
                            and src_t.lane_width != 1
                            and src_t.lanes == t.lanes
                        ):
                            xs2 = by_size.get((src_t, left_size), [])
                            ys2 = by_size.get((src_t, right_size), [])
                            if xs2 and ys2:
                                lim = min(128, max_candidates_per_size)
                                xs2 = sorted(
                                    xs2, key=lambda e: (expr_cost(e, types), repr(e))
                                )[:lim]
                                ys2 = sorted(
                                    ys2, key=lambda e: (expr_cost(e, types), repr(e))
                                )[:lim]
                                for a in xs2:
                                    for b in ys2:
                                        cand_set.add(SimdEq(a=a, b=b))
                                        cand_set.add(SimdUlt(a=a, b=b))
                                        cand_set.add(SimdUle(a=a, b=b))
                                        cand_set.add(SimdUgt(a=a, b=b))
                                        cand_set.add(SimdUge(a=a, b=b))

            if isinstance(t, SimdType):
                mask_t = SimdType(lane_width=1, lanes=t.lanes)
                if mask_t in allowed_types and size >= 4:
                    for mask_size in range(1, size - 2):
                        for a_size in range(1, size - 1 - mask_size):
                            b_size = size - 1 - mask_size - a_size
                            if b_size < 1:
                                continue
                            ms = by_size.get((mask_t, mask_size), [])
                            as_ = by_size.get((t, a_size), [])
                            bs = by_size.get((t, b_size), [])
                            if not (ms and as_ and bs):
                                continue
                            lim = min(64, max_candidates_per_size)
                            ms = sorted(
                                ms, key=lambda e: (expr_cost(e, types), repr(e))
                            )[:lim]
                            as_ = sorted(
                                as_, key=lambda e: (expr_cost(e, types), repr(e))
                            )[:lim]
                            bs = sorted(
                                bs, key=lambda e: (expr_cost(e, types), repr(e))
                            )[:lim]
                            for m0 in ms:
                                for a0 in as_:
                                    for b0 in bs:
                                        cand_set.add(SimdBlend(mask=m0, a=a0, b=b0))

            typed: list[Expr] = []
            for c in cand_set:
                try:
                    ct = infer_type(c, types)
                except Exception:
                    continue
                if type_equal(ct, t):
                    typed.append(c)

            typed_sorted = sorted(
                set(typed), key=lambda e: (expr_cost(e, types), repr(e))
            )[:max_candidates_per_size]
            if typed_sorted:
                by_size[(t, size)] = typed_sorted

        best = None
        best_cost = None
        for cand in by_size.get((out_t, size), []):
            if stats is not None:
                stats.candidates_checked += 1
            checked += 1
            if checked > max_total_candidates:
                raise SuperoptError("candidate cap exceeded")
            if use_cegis and counterexamples:
                ok = True
                for ce, sv in zip(counterexamples, spec_vals, strict=True):
                    if eval_expr(cand, types, ce) != sv:
                        ok = False
                        break
                if not ok:
                    continue
            if stats is not None:
                stats.solver_checks += 1
            ce = _cex(spec, cand, types, used_vars, timeout_ms=timeout_ms)
            if ce is None:
                c = expr_cost(cand, types)
                if best is None or best_cost is None or c < best_cost:
                    best = cand
                    best_cost = c
            elif use_cegis and len(counterexamples) < max_counterexamples:
                counterexamples.append(ce)
                spec_vals.append(eval_expr(spec, types, ce))
                if stats is not None:
                    stats.counterexamples = len(counterexamples)

        if best is not None:
            return best

    raise SuperoptError("no equivalent program found within max_nodes")
