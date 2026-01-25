from __future__ import annotations

from dataclasses import dataclass

from stc.interp import eval_expr
from stc.tick_ir import TickIR
from stc.tick_ir_validate import validate_tick_ir


@dataclass(frozen=True)
class TickStep:
    outputs: dict[str, int | bool]
    next_state: dict[str, int | bool]


def step_tickir(ir: TickIR, *, state: dict[str, int | bool], inputs: dict[str, int | bool]) -> TickStep:
    """One-cycle TickIR step with synchronous reset semantics.

    If there is an input named `reset` and it is true, next_state is taken from
    `reset_state` (matching the behavior of `$sdff` in Yosys-flattened designs).
    """
    validate_tick_ir(ir)
    types = {**ir.inputs, **ir.state}

    env = {**state, **inputs}
    outs = {k: eval_expr(ir.output_exprs[k], types, env) for k in ir.outputs}

    rst = bool(inputs.get("reset", False))
    if "reset" in ir.inputs and rst:
        nxt = {k: eval_expr(ir.reset_state[k], types, env) for k in ir.state}
    else:
        nxt = {k: eval_expr(ir.next_state[k], types, env) for k in ir.state}

    return TickStep(outputs=outs, next_state=nxt)

