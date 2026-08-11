from __future__ import annotations

from dataclasses import dataclass

import z3

from stc.interp import eval_expr, reset_state
from stc.replace import replace_vars
from stc.tick_ir import BitVecType, BoolType, SimdType, TickIR, Var
from stc.tick_ir_validate import validate_tick_ir
from stc.z3_encode import encode_expr
from stc.z3_util import CpuBudget, check_with_budget, make_solver


@dataclass(frozen=True)
class ReachabilityError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def constant_state_within_bound(
    ir: TickIR, bound: int, *, timeout_ms: int = 200
) -> dict[str, int | bool]:
    validate_tick_ir(ir)
    if bound < 0:
        raise ReachabilityError("bound must be >= 0")
    if not ir.state:
        return {}

    if bound == 0:
        init = reset_state(ir)
        return dict(init.state)

    types: dict[str, BoolType | BitVecType | SimdType] = {}
    for t in range(bound + 1):
        for name, ty in ir.state.items():
            types[f"{name}__st_{t}"] = ty
    for t in range(bound):
        for name, ty in ir.inputs.items():
            types[f"{name}__in_{t}"] = ty

    solver = make_solver()
    budget = CpuBudget(timeout_ms)

    init = reset_state(ir)
    for name, ty in ir.state.items():
        lhs = encode_expr(Var(f"{name}__st_0"), types)
        rhs = encode_expr(ir.reset_state[name], {})
        solver.add(lhs == rhs)

    for t in range(bound):
        repl: dict[str, Var] = {}
        for name in ir.state:
            repl[name] = Var(f"{name}__st_{t}")
        for name in ir.inputs:
            repl[name] = Var(f"{name}__in_{t}")

        for name in ir.state:
            lhs = encode_expr(Var(f"{name}__st_{t + 1}"), types)
            rhs_expr = replace_vars(ir.next_state[name], repl)
            rhs = encode_expr(rhs_expr, types)
            solver.add(lhs == rhs)

    constants: dict[str, int | bool] = {}

    for name, ty in ir.state.items():
        solver.push()
        diffs: list[z3.BoolRef] = []
        s0 = encode_expr(Var(f"{name}__st_0"), types)
        for t in range(1, bound + 1):
            st = encode_expr(Var(f"{name}__st_{t}"), types)
            diffs.append(st != s0)
        solver.add(z3.Or(diffs))
        res = check_with_budget(solver, budget)
        solver.pop()
        if res == z3.unknown:
            continue
        if res == z3.unsat:
            v = eval_expr(ir.reset_state[name], {}, {})
            if isinstance(ty, BoolType):
                constants[name] = bool(v)
            else:
                constants[name] = int(v)
    return constants
