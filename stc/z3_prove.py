from __future__ import annotations

from dataclasses import dataclass

import z3

from stc.tick_ir import Expr, Type
from stc.z3_encode import encode_expr
from stc.z3_util import CpuBudget, check_with_budget, make_solver


@dataclass(frozen=True)
class Z3ProveError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def prove_equiv(
    spec: Expr, cand: Expr, types: dict[str, Type], *, timeout_ms: int = 200
) -> bool:
    z3_vars: dict[str, z3.ExprRef] = {}
    spec_z = encode_expr(spec, types, z3_vars)
    cand_z = encode_expr(cand, types, z3_vars)
    solver = make_solver()
    solver.add(spec_z != cand_z)
    r = check_with_budget(solver, CpuBudget(timeout_ms))
    if r == z3.unknown:
        raise Z3ProveError("z3 returned unknown")
    return r == z3.unsat


def prove_not_equiv(
    spec: Expr, cand: Expr, types: dict[str, Type], *, timeout_ms: int = 200
) -> bool:
    z3_vars: dict[str, z3.ExprRef] = {}
    spec_z = encode_expr(spec, types, z3_vars)
    cand_z = encode_expr(cand, types, z3_vars)
    solver = make_solver()
    solver.add(spec_z != cand_z)
    r = check_with_budget(solver, CpuBudget(timeout_ms))
    if r == z3.unknown:
        raise Z3ProveError("z3 returned unknown")
    return r == z3.sat
