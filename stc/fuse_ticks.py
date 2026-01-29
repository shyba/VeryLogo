from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import time

from stc.metrics import compute_metrics
from stc.reduce import reduce_tick_ir
from stc.hashcons import hashcons_tick_ir
from stc.tick_ir import BoolType, EXPR_CLASSES, Expr, Mux, TickIR, Type, Var


def _estimate_fused_nodes(ir: TickIR, steps: int) -> int:
    """Estimate node count after fusion without building the IR.

    Conservative estimate to avoid OOM: assumes exponential growth
    with reuse capping at cubic growth for deep fusion.
    """
    base_metrics = compute_metrics(ir)
    base_nodes = base_metrics.expr_nodes_total

    estimated = min(base_nodes * steps, base_nodes ** min(steps, 3))
    return estimated


@dataclass(frozen=True)
class FuseBudget:
    max_nodes: int
    max_depth: int
    stop_on_budget_hit: bool = True
    max_step_ms: int | None = None


def _subst_expr(expr: Expr, repl: dict[str, Expr], memo: dict[Expr, Expr]) -> Expr:
    cached = memo.get(expr)
    if cached is not None:
        return cached

    if isinstance(expr, Var):
        out = repl.get(expr.name, expr)
        memo[expr] = out
        return out

    if not is_dataclass(expr):
        memo[expr] = expr
        return expr

    changed = False
    kwargs: dict[str, object] = {}
    for f in fields(expr):
        v = getattr(expr, f.name)
        if isinstance(v, EXPR_CLASSES):
            nv = _subst_expr(v, repl, memo)
            changed |= nv is not v
            kwargs[f.name] = nv
            continue
        if isinstance(v, (list, tuple)):
            new_items = []
            any_expr = False
            any_changed = False
            for item in v:
                if isinstance(item, EXPR_CLASSES):
                    any_expr = True
                    nitem = _subst_expr(item, repl, memo)
                    any_changed |= nitem is not item
                    new_items.append(nitem)
                else:
                    new_items.append(item)
            if any_expr and any_changed:
                changed = True
                kwargs[f.name] = type(v)(new_items)
            else:
                kwargs[f.name] = v
            continue
        kwargs[f.name] = v

    out = expr if not changed else expr.__class__(**kwargs)  # type: ignore[arg-type]
    memo[expr] = out
    return out


def fuse_ticks(
    ir: TickIR,
    n: int,
    *,
    mode: str = "final",  # "final" | "all" | "state-only"
    input_policy: str = "shared",  # "shared" | "replicate"
    budget: FuseBudget | None = None,
) -> TickIR:
    """Compose the TickIR transition function for N ticks (generic sequential fusion).

    This operates at the TickIR level (pure expression DAGs), before lowering to
    CircuitState/scheduling/emit. It is generic for arbitrary sequential designs:
    multi-bit state is supported (BoolType, BitVecType, SimdType, ...).

    Notes:
    - Output expressions are stored as a dict in TickIR. For mode="all", fused
      outputs are emitted with deterministic suffixed names: f"{name}__t{k}".
    - For input_policy="replicate", per-tick input Vars are introduced as
      f"{name}__t{k}". The original input Vars are also retained (unused) to
      preserve compatibility with existing APIs that expect base input names.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return ir
    if mode not in {"final", "all", "state-only"}:
        raise ValueError(f"invalid mode: {mode}")
    if input_policy not in {"shared", "replicate"}:
        raise ValueError(f"invalid input_policy: {input_policy}")

    if budget is None:
        budget = FuseBudget(
            max_nodes=1 << 60, max_depth=1 << 60, stop_on_budget_hit=True
        )

    # Fast fail: if the caller's budget cannot even cover the current IR size,
    # do not attempt fusion (which necessarily builds at least the 1-step
    # transition and may be extremely expensive for large sequential designs).
    if budget.stop_on_budget_hit and n > 1:
        base_metrics = compute_metrics(
            TickIR(
                name=ir.name,
                inputs=dict(ir.inputs),
                outputs=dict(ir.outputs),
                state=dict(ir.state),
                reset_state=dict(ir.reset_state),
                next_state=dict(ir.next_state),
                output_exprs=dict(ir.output_exprs),
            )
        )
        if (
            base_metrics.expr_nodes_total >= budget.max_nodes
            or base_metrics.expr_depth_max >= budget.max_depth
        ):
            return ir

    expanded_inputs: dict[str, Type] = dict(ir.inputs)
    if input_policy == "replicate":
        for k in range(n):
            for name, ty in ir.inputs.items():
                expanded_inputs[f"{name}__t{k}"] = ty

    cur_state_exprs: dict[str, Expr] = {name: Var(name) for name in ir.state}

    has_sync_reset = "reset" in ir.inputs and isinstance(
        ir.inputs.get("reset"), BoolType
    )

    best_state_exprs = dict(cur_state_exprs)
    best_output_state_exprs = dict(cur_state_exprs)
    best_steps = 0
    states_before: list[dict[str, Expr]] = []

    for k in range(n):
        if budget.stop_on_budget_hit and k > 0:
            estimated = _estimate_fused_nodes(ir, k + 1)
            if estimated > budget.max_nodes * 1.5:
                break

        step_t0 = time.perf_counter()
        state_before = dict(cur_state_exprs)
        if input_policy == "shared":
            input_repl: dict[str, Expr] = {}
        else:
            input_repl = {name: Var(f"{name}__t{k}") for name in ir.inputs}

        repl: dict[str, Expr] = {}
        repl.update(cur_state_exprs)
        repl.update(input_repl)

        memo: dict[Expr, Expr] = {}
        if has_sync_reset:
            reset_var_name = "reset" if input_policy == "shared" else f"reset__t{k}"
            reset_var = Var(reset_var_name)
            next_state_exprs = {
                name: _subst_expr(
                    Mux(cond=reset_var, a=ir.reset_state[name], b=ir.next_state[name]),
                    repl,
                    memo,
                )
                for name in ir.state
            }
        else:
            next_state_exprs = {
                name: _subst_expr(expr, repl, memo)
                for name, expr in ir.next_state.items()
            }

        reduced_step = hashcons_tick_ir(
            TickIR(
                name=ir.name,
                inputs=expanded_inputs,
                outputs=dict(ir.outputs),
                state=dict(ir.state),
                reset_state=dict(ir.reset_state),
                next_state=next_state_exprs,
                output_exprs={},
            )
        )
        cur_state_exprs = dict(reduced_step.next_state)
        states_before.append(state_before)

        if budget.max_step_ms is not None:
            dt_ms = (time.perf_counter() - step_t0) * 1000.0
            if dt_ms > budget.max_step_ms:
                if budget.stop_on_budget_hit:
                    break

        metrics = compute_metrics(
            TickIR(
                name=ir.name,
                inputs=expanded_inputs,
                outputs={},
                state=dict(ir.state),
                reset_state={},
                next_state=cur_state_exprs,
                output_exprs={},
            )
        )
        if (
            metrics.expr_nodes_total > budget.max_nodes
            or metrics.expr_depth_max > budget.max_depth
        ):
            if budget.stop_on_budget_hit:
                break
        best_state_exprs = dict(cur_state_exprs)
        best_output_state_exprs = state_before
        best_steps = k + 1

    # If we couldn't fuse even one step under budget/time limits, return the
    # original IR unchanged (PR1/PR2 callers treat fusion as an optional pass).
    if best_steps == 0:
        return ir

    fused_next_state = dict(best_state_exprs)

    if mode == "state-only":
        fused_outputs = dict(ir.output_exprs)
    elif mode == "final":
        k_last = best_steps - 1
        input_repl = (
            {}
            if input_policy == "shared"
            else {name: Var(f"{name}__t{k_last}") for name in ir.inputs}
        )
        # TickIR outputs are computed from S_k (pre-state) for tick k.
        repl = {**best_output_state_exprs, **input_repl}
        memo = {}
        fused_outputs = {
            name: _subst_expr(expr, repl, memo)
            for name, expr in ir.output_exprs.items()
        }
    else:
        # mode == "all"
        fused_outputs = {}
        for k in range(best_steps):
            input_repl = (
                {}
                if input_policy == "shared"
                else {name: Var(f"{name}__t{k}") for name in ir.inputs}
            )
            repl = {**states_before[k], **input_repl}
            memo = {}
            fused_outputs.update(
                {
                    f"{name}__t{k}": _subst_expr(expr, repl, memo)
                    for name, expr in ir.output_exprs.items()
                }
            )

    fused = TickIR(
        name=f"{ir.name}__fused{best_steps}",
        inputs=expanded_inputs,
        outputs=(
            {k: v for k, v in ir.outputs.items()}
            if mode != "all"
            else {
                f"{name}__t{k}": ty
                for k in range(best_steps)
                for name, ty in ir.outputs.items()
            }
        ),
        state=dict(ir.state),
        reset_state=dict(ir.reset_state),
        next_state=fused_next_state,
        output_exprs=fused_outputs,
    )
    return reduce_tick_ir(fused)
