from __future__ import annotations

from dataclasses import dataclass

from stc.interp import eval_expr
from stc.reduce import reduce_expr
from stc.replace import replace_vars
from stc.tick_ir import BitVecConst, BitVecType, BoolConst, BoolType, Expr, TickIR, Type
from stc.tick_ir_validate import iter_vars, validate_tick_ir


@dataclass(frozen=True)
class ConstStateSchedule:
    const_state_order: list[str]
    runtime_state_order: list[str]
    reset_vals: dict[str, int | bool]
    const_state_steps: list[dict[str, int | bool]]
    const_state_final: dict[str, int | bool]


def compute_const_state_schedule(ir: TickIR, *, steps: int) -> ConstStateSchedule:
    validate_tick_ir(ir)
    if int(steps) < 1:
        raise ValueError("steps must be >= 1")

    ctx_types: dict[str, Type] = {**ir.inputs, **ir.state}
    state_order = sorted(ir.state.keys())

    reset_vals: dict[str, int | bool] = {}
    for name in state_order:
        reset_vals[name] = eval_expr(ir.reset_state[name], ctx_types, {})

    const_names: set[str] = set(state_order)
    changed = True
    while changed:
        changed = False
        for name in list(const_names):
            used = set(iter_vars(ir.next_state[name]))
            if not used.issubset(const_names):
                const_names.remove(name)
                changed = True

    const_state_order = [n for n in state_order if n in const_names]
    runtime_state_order = [n for n in state_order if n not in const_names]

    cur = {n: reset_vals[n] for n in const_state_order}
    const_state_steps: list[dict[str, int | bool]] = []
    for _ in range(int(steps)):
        const_state_steps.append(dict(cur))
        nxt: dict[str, int | bool] = {}
        for n in const_state_order:
            nxt[n] = eval_expr(ir.next_state[n], ctx_types, cur)
        cur = nxt

    return ConstStateSchedule(
        const_state_order=const_state_order,
        runtime_state_order=runtime_state_order,
        reset_vals=reset_vals,
        const_state_steps=const_state_steps,
        const_state_final=dict(cur),
    )


def _const_expr(t: Type, value: int | bool) -> Expr:
    if isinstance(t, BoolType):
        return BoolConst(value=bool(value))
    if isinstance(t, BitVecType):
        return BitVecConst(width=t.width, value=int(value))
    raise TypeError("const state schedule supports only bool and bitvec types")


def const_state_repl_for_step(
    ir: TickIR, schedule: ConstStateSchedule, *, step_index: int
) -> dict[str, Expr]:
    if step_index < 0 or step_index >= len(schedule.const_state_steps):
        raise ValueError("step_index out of range")
    repl: dict[str, Expr] = {}
    vals = schedule.const_state_steps[step_index]
    for name, value in vals.items():
        repl[name] = _const_expr(ir.state[name], value)
    return repl


def specialize_expr(expr: Expr, types: dict[str, Type], repl: dict[str, Expr]) -> Expr:
    if not repl:
        return expr
    return reduce_expr(replace_vars(expr, repl), types)


@dataclass(frozen=True)
class SpecializedSteps:
    output_exprs: list[dict[str, Expr]]
    next_state_exprs: list[dict[str, Expr]]
    schedule: ConstStateSchedule


def specialize_tick_ir_steps(ir: TickIR, *, steps: int) -> SpecializedSteps:
    schedule = compute_const_state_schedule(ir, steps=int(steps))
    ctx_types: dict[str, Type] = {**ir.inputs, **ir.state}

    outs: list[dict[str, Expr]] = []
    nxts: list[dict[str, Expr]] = []
    for step_index in range(int(steps)):
        repl = const_state_repl_for_step(ir, schedule, step_index=step_index)
        step_out: dict[str, Expr] = {}
        for name, expr in ir.output_exprs.items():
            step_out[name] = specialize_expr(expr, ctx_types, repl)
        outs.append(step_out)

        step_nxt: dict[str, Expr] = {}
        for name in schedule.runtime_state_order:
            step_nxt[name] = specialize_expr(ir.next_state[name], ctx_types, repl)
        nxts.append(step_nxt)

    return SpecializedSteps(output_exprs=outs, next_state_exprs=nxts, schedule=schedule)
