from __future__ import annotations


import argparse
import json
from pathlib import Path

from stc.tick_ir import TickIR, Expr, Var
from stc.tick_ir_to_packed_circuit_state import (
    PackedLoweringError,
    lower_tick_ir_to_packed_circuit_state,
)
from stc.metrics import _count_nodes, _expr_children


def _load_tick_ir(path: Path) -> TickIR:
    data = json.loads(path.read_text(encoding="utf-8"))
    return TickIR.from_dict(data)


def _save_tick_ir(ir: TickIR, path: Path) -> None:
    path.write_text(json.dumps(ir.to_dict(), indent=2), encoding="utf-8")


def _try_lowering(ir: TickIR) -> tuple[bool, str]:
    """
    Attempt packed lowering and return (success, error_msg).
    Returns (True, "") if lowering succeeds, (False, error_msg) if it fails.
    """
    try:
        lower_tick_ir_to_packed_circuit_state(ir)
        return (True, "")
    except PackedLoweringError as e:
        return (False, str(e))


def _count_total_nodes(ir: TickIR) -> int:
    """Count total expression nodes in outputs and next_state."""
    total = 0
    for expr in ir.output_exprs.values():
        total += _count_nodes(expr)
    for expr in ir.next_state.values():
        total += _count_nodes(expr)
    return total


def _simplify_expr(expr: Expr, keep_vars: set[str]) -> Expr | None:
    """
    Try to simplify an expression by replacing it with a simpler form.
    Returns None if no simplification is possible.
    Returns simplified expression if reduction succeeds.
    """
    if isinstance(expr, Var):
        if expr.name in keep_vars:
            return None
        from stc.tick_ir import BoolConst, BitVecConst
        from stc.interp import infer_type
        from stc.tick_ir import BoolType, BitVecType

        typ = infer_type(expr, {expr.name: infer_type(expr, {})})
        if isinstance(typ, BoolType):
            return BoolConst(value=False)
        elif isinstance(typ, BitVecType):
            return BitVecConst(width=typ.width, value=0)
        return None

    children = _expr_children(expr)
    if not children:
        return None

    for i, child in enumerate(children):
        simplified = _simplify_expr(child, keep_vars)
        if simplified is None:
            continue

        new_children = list(children)
        new_children[i] = simplified

        try:
            new_expr = (
                type(expr)(*new_children)
                if len(new_children) > 0
                else type(expr)(
                    **{
                        f.name: (
                            new_children[j]
                            if f.name
                            in [
                                ff.name
                                for ff in expr.__dataclass_fields__.values()
                                if isinstance(
                                    getattr(expr, ff.name), Expr.__class__.__mro__
                                )
                            ][: len(new_children)]
                            else getattr(expr, f.name)
                        )
                        for j, f in enumerate(expr.__dataclass_fields__.values())
                    }
                )
            )
            return new_expr
        except (TypeError, ValueError):
            pass

    return None


def _minimize_outputs(ir: TickIR, target_error: str) -> TickIR:
    """Try removing outputs one by one to minimize the TickIR."""
    output_order = sorted(ir.outputs.keys())

    for name in output_order:
        if len(ir.outputs) <= 1:
            break

        new_outputs = {k: v for k, v in ir.outputs.items() if k != name}
        new_output_exprs = {k: v for k, v in ir.output_exprs.items() if k != name}

        candidate = TickIR(
            name=ir.name,
            inputs=ir.inputs,
            outputs=new_outputs,
            state=ir.state,
            reset_state=ir.reset_state,
            next_state=ir.next_state,
            output_exprs=new_output_exprs,
        )

        success, error_msg = _try_lowering(candidate)
        if not success and target_error in error_msg:
            ir = candidate

    return ir


def _minimize_state(ir: TickIR, target_error: str) -> TickIR:
    """Try removing state variables one by one to minimize the TickIR."""
    state_order = sorted(ir.state.keys())

    for name in state_order:
        if len(ir.state) == 0:
            break

        new_state = {k: v for k, v in ir.state.items() if k != name}
        new_reset_state = {k: v for k, v in ir.reset_state.items() if k != name}
        new_next_state = {k: v for k, v in ir.next_state.items() if k != name}

        used_in_outputs = any(
            _uses_var(expr, name) for expr in ir.output_exprs.values()
        )
        used_in_next_state = any(
            _uses_var(expr, name) for expr in new_next_state.values()
        )

        if used_in_outputs or used_in_next_state:
            continue

        candidate = TickIR(
            name=ir.name,
            inputs=ir.inputs,
            outputs=ir.outputs,
            state=new_state,
            reset_state=new_reset_state,
            next_state=new_next_state,
            output_exprs=ir.output_exprs,
        )

        success, error_msg = _try_lowering(candidate)
        if not success and target_error in error_msg:
            ir = candidate

    return ir


def _uses_var(expr: Expr, var_name: str) -> bool:
    """Check if an expression uses a given variable."""
    if isinstance(expr, Var) and expr.name == var_name:
        return True
    for child in _expr_children(expr):
        if _uses_var(child, var_name):
            return True
    return False


def _minimize_inputs(ir: TickIR, target_error: str) -> TickIR:
    """Try removing inputs one by one to minimize the TickIR."""
    input_order = sorted(ir.inputs.keys())

    for name in input_order:
        used_in_outputs = any(
            _uses_var(expr, name) for expr in ir.output_exprs.values()
        )
        used_in_next_state = any(
            _uses_var(expr, name) for expr in ir.next_state.values()
        )

        if not used_in_outputs and not used_in_next_state:
            new_inputs = {k: v for k, v in ir.inputs.items() if k != name}

            candidate = TickIR(
                name=ir.name,
                inputs=new_inputs,
                outputs=ir.outputs,
                state=ir.state,
                reset_state=ir.reset_state,
                next_state=ir.next_state,
                output_exprs=ir.output_exprs,
            )

            success, error_msg = _try_lowering(candidate)
            if not success and target_error in error_msg:
                ir = candidate

    return ir


def _minimize_expressions(
    ir: TickIR, target_error: str, max_passes: int = 10
) -> TickIR:
    """
    Try simplifying expressions within outputs and next_state.
    Uses a greedy approach: for each expression node, try replacing it with a constant.
    """
    required_vars = set(ir.inputs.keys()) | set(ir.state.keys())

    for pass_num in range(max_passes):
        made_progress = False

        for out_name in sorted(ir.output_exprs.keys()):
            expr = ir.output_exprs[out_name]
            simplified = _try_simplify_recursive(expr, required_vars)

            if simplified is not None and simplified != expr:
                candidate_output_exprs = dict(ir.output_exprs)
                candidate_output_exprs[out_name] = simplified

                candidate = TickIR(
                    name=ir.name,
                    inputs=ir.inputs,
                    outputs=ir.outputs,
                    state=ir.state,
                    reset_state=ir.reset_state,
                    next_state=ir.next_state,
                    output_exprs=candidate_output_exprs,
                )

                success, error_msg = _try_lowering(candidate)
                if not success and target_error in error_msg:
                    ir = candidate
                    made_progress = True
                    break

        if made_progress:
            continue

        for state_name in sorted(ir.next_state.keys()):
            expr = ir.next_state[state_name]
            simplified = _try_simplify_recursive(expr, required_vars)

            if simplified is not None and simplified != expr:
                candidate_next_state = dict(ir.next_state)
                candidate_next_state[state_name] = simplified

                candidate = TickIR(
                    name=ir.name,
                    inputs=ir.inputs,
                    outputs=ir.outputs,
                    state=ir.state,
                    reset_state=ir.reset_state,
                    next_state=candidate_next_state,
                    output_exprs=ir.output_exprs,
                )

                success, error_msg = _try_lowering(candidate)
                if not success and target_error in error_msg:
                    ir = candidate
                    made_progress = True
                    break

        if not made_progress:
            break

    return ir


def _try_simplify_recursive(expr: Expr, keep_vars: set[str]) -> Expr | None:
    """
    Recursively try to simplify subexpressions.
    Returns None if no simplification found, otherwise returns simplified expr.
    """
    children = _expr_children(expr)

    for i, child in enumerate(children):
        child_simplified = _try_replace_with_zero(child, keep_vars)
        if child_simplified is not None:
            new_children = list(children)
            new_children[i] = child_simplified
            try:
                return _rebuild_expr(expr, new_children)
            except (TypeError, ValueError):
                pass

        recursive_simplified = _try_simplify_recursive(child, keep_vars)
        if recursive_simplified is not None:
            new_children = list(children)
            new_children[i] = recursive_simplified
            try:
                return _rebuild_expr(expr, new_children)
            except (TypeError, ValueError):
                pass

    return None


def _try_replace_with_zero(expr: Expr, keep_vars: set[str]) -> Expr | None:
    """Try replacing an expression with a zero constant of the same type."""
    if isinstance(expr, Var) and expr.name in keep_vars:
        return None

    from stc.interp import infer_type
    from stc.tick_ir import BoolType, BitVecType, BoolConst, BitVecConst

    try:
        typ = infer_type(expr, {})
        if isinstance(typ, BoolType):
            return BoolConst(value=False)
        elif isinstance(typ, BitVecType):
            return BitVecConst(width=typ.width, value=0)
    except Exception:
        pass

    return None


def _rebuild_expr(original: Expr, new_children: list[Expr]) -> Expr:
    """Rebuild an expression with new children."""
    from dataclasses import fields

    field_list = [f for f in fields(original)]
    expr_fields = [
        f
        for f in field_list
        if isinstance(getattr(original, f.name), Expr.__class__.__mro__)
    ]

    if len(new_children) != len(expr_fields):
        kwargs = {}
        child_idx = 0
        for f in field_list:
            val = getattr(original, f.name)
            if isinstance(val, Expr.__class__.__mro__):
                if child_idx < len(new_children):
                    kwargs[f.name] = new_children[child_idx]
                    child_idx += 1
                else:
                    kwargs[f.name] = val
            else:
                kwargs[f.name] = val
        return type(original)(**kwargs)

    return type(original)(*new_children)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Minimize a TickIR that fails packed lowering to smallest reproducible test case"
    )
    p.add_argument(
        "--tick-ir",
        type=Path,
        required=True,
        help="Path to TickIR JSON that fails packed lowering",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to write minimized TickIR JSON",
    )
    p.add_argument(
        "--max-passes",
        type=int,
        default=3,
        help="Maximum number of expression simplification passes (default: 3)",
    )
    args = p.parse_args()

    ir = _load_tick_ir(args.tick_ir)

    success, error_msg = _try_lowering(ir)
    if success:
        raise SystemExit("Input TickIR does not fail packed lowering")

    print(f"Original TickIR fails with: {error_msg}")
    original_nodes = _count_total_nodes(ir)
    print(f"Original expression node count: {original_nodes}")
    print()

    target_error = error_msg.split(":")[0] if ":" in error_msg else error_msg

    print("Phase 1: Removing outputs...")
    ir = _minimize_outputs(ir, target_error)
    print(f"  Outputs remaining: {len(ir.outputs)}")
    print(f"  Expression nodes: {_count_total_nodes(ir)}")

    print("Phase 2: Removing unused state variables...")
    ir = _minimize_state(ir, target_error)
    print(f"  State variables remaining: {len(ir.state)}")
    print(f"  Expression nodes: {_count_total_nodes(ir)}")

    print("Phase 3: Removing unused inputs...")
    ir = _minimize_inputs(ir, target_error)
    print(f"  Inputs remaining: {len(ir.inputs)}")
    print(f"  Expression nodes: {_count_total_nodes(ir)}")

    print(f"Phase 4: Simplifying expressions (max {args.max_passes} passes)...")
    ir = _minimize_expressions(ir, target_error, max_passes=args.max_passes)
    final_nodes = _count_total_nodes(ir)
    print(f"  Expression nodes: {final_nodes}")

    _save_tick_ir(ir, args.out)

    success, final_error = _try_lowering(ir)
    if success:
        print()
        print("WARNING: Minimized TickIR no longer fails packed lowering!")
        raise SystemExit(1)

    reduction_ratio = (
        (1.0 - final_nodes / original_nodes) * 100 if original_nodes > 0 else 0
    )

    print()
    print("=" * 60)
    print("Minimization complete")
    print("=" * 60)
    print(f"Original expression nodes: {original_nodes}")
    print(f"Final expression nodes: {final_nodes}")
    print(f"Reduction: {reduction_ratio:.1f}%")
    print(f"Preserved error: {final_error}")
    print(f"Output written to: {args.out}")


if __name__ == "__main__":
    main()
