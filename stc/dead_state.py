from __future__ import annotations

from dataclasses import dataclass

from stc.tick_ir import Expr, TickIR, Var
from stc.tick_ir_validate import iter_vars, validate_tick_ir


@dataclass(frozen=True)
class DeadStateError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def _deps_in_state(expr: Expr, state_names: set[str]) -> set[str]:
    return {v for v in iter_vars(expr) if v in state_names}


def live_state_within_bound(ir: TickIR, bound: int) -> set[str]:
    validate_tick_ir(ir)
    if bound < 1:
        raise DeadStateError("bound must be >= 1")

    state_names = set(ir.state.keys())
    live: set[str] = set()

    for expr in ir.output_exprs.values():
        live |= _deps_in_state(expr, state_names)

    for _ in range(bound - 1):
        expanded: set[str] = set(live)
        for reg in live:
            expanded |= _deps_in_state(ir.next_state[reg], state_names)
        live = expanded

    return live


def remove_dead_state(ir: TickIR, bound: int) -> TickIR:
    validate_tick_ir(ir)
    live = live_state_within_bound(ir, bound)
    if live == set(ir.state.keys()):
        return ir

    state = {k: v for k, v in ir.state.items() if k in live}
    reset_state = {k: v for k, v in ir.reset_state.items() if k in live}
    next_state = {k: v for k, v in ir.next_state.items() if k in live}
    return TickIR(
        name=ir.name,
        inputs=dict(ir.inputs),
        outputs=dict(ir.outputs),
        state=state,
        reset_state=reset_state,
        next_state=next_state,
        output_exprs=dict(ir.output_exprs),
    )
