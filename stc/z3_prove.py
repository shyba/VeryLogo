from __future__ import annotations

from dataclasses import dataclass

import z3

from stc.tick_ir import Expr, Type
from stc.z3_encode import encode_expr


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
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(spec_z != cand_z)
    r = solver.check()
    if r == z3.unknown:
        raise Z3ProveError("z3 returned unknown")
    return r == z3.unsat


def prove_not_equiv(
    spec: Expr, cand: Expr, types: dict[str, Type], *, timeout_ms: int = 200
) -> bool:
    z3_vars: dict[str, z3.ExprRef] = {}
    spec_z = encode_expr(spec, types, z3_vars)
    cand_z = encode_expr(cand, types, z3_vars)
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(spec_z != cand_z)
    r = solver.check()
    if r == z3.unknown:
        raise Z3ProveError("z3 returned unknown")
    return r == z3.sat
