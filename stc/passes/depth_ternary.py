from __future__ import annotations

from dataclasses import replace
from typing import Callable

from stc.passmgr import Pass, PassContext, PassMetrics
from stc.tick_ir import (
    TickIR,
    Expr,
    And,
    Or,
    Xor,
    Not,
    Var,
    BoolConst,
    BitVecConst,
    SimdConst,
    TernaryLut,
)
from stc.mapping.ternary import extract_3input_cone, Cone3


def _expr_depth(expr: Expr, cache: dict[int, int] | None = None) -> int:
    if cache is None:
        cache = {}

    eid = id(expr)
    if eid in cache:
        return cache[eid]

    if isinstance(expr, (Var, BoolConst, BitVecConst, SimdConst)):
        cache[eid] = 0
        return 0

    if isinstance(expr, TernaryLut):
        a_depth = _expr_depth(expr.a, cache)
        b_depth = _expr_depth(expr.b, cache)
        c_depth = _expr_depth(expr.c, cache)
        depth = 1 + max(a_depth, b_depth, c_depth)
        cache[eid] = depth
        return depth

    if isinstance(expr, Not):
        depth = 1 + _expr_depth(expr.x, cache)
        cache[eid] = depth
        return depth

    if isinstance(expr, (And, Or, Xor)):
        a_depth = _expr_depth(expr.a, cache)
        b_depth = _expr_depth(expr.b, cache)
        depth = 1 + max(a_depth, b_depth)
        cache[eid] = depth
        return depth

    cache[eid] = 1
    return 1


def _is_on_critical_path(
    expr: Expr, max_depth: int, depth_cache: dict[int, int]
) -> bool:
    return _expr_depth(expr, depth_cache) == max_depth


def _collect_critical_path_exprs(
    expr: Expr, max_depth: int, depth_cache: dict[int, int], result: set[int]
) -> None:
    eid = id(expr)
    if eid in result:
        return

    current_depth = _expr_depth(expr, depth_cache)
    if current_depth != max_depth:
        return

    result.add(eid)

    if isinstance(expr, (Var, BoolConst, BitVecConst, SimdConst)):
        return

    if isinstance(expr, TernaryLut):
        for child in [expr.a, expr.b, expr.c]:
            child_depth = _expr_depth(child, depth_cache)
            if child_depth == max_depth - 1:
                _collect_critical_path_exprs(child, max_depth - 1, depth_cache, result)
        return

    if isinstance(expr, Not):
        child_depth = _expr_depth(expr.x, depth_cache)
        if child_depth == max_depth - 1:
            _collect_critical_path_exprs(expr.x, max_depth - 1, depth_cache, result)
        return

    if isinstance(expr, (And, Or, Xor)):
        a_depth = _expr_depth(expr.a, depth_cache)
        b_depth = _expr_depth(expr.b, depth_cache)
        if a_depth == max_depth - 1:
            _collect_critical_path_exprs(expr.a, max_depth - 1, depth_cache, result)
        if b_depth == max_depth - 1:
            _collect_critical_path_exprs(expr.b, max_depth - 1, depth_cache, result)


def map_critical_path_to_ternary(
    expr: Expr, depth_cache: dict[int, int] | None = None
) -> tuple[Expr, bool]:
    if depth_cache is None:
        depth_cache = {}

    eid = id(expr)
    current_depth = _expr_depth(expr, depth_cache)

    if isinstance(expr, (Var, BoolConst, BitVecConst, SimdConst)):
        return expr, False

    if isinstance(expr, TernaryLut):
        a_new, a_changed = map_critical_path_to_ternary(expr.a, depth_cache)
        b_new, b_changed = map_critical_path_to_ternary(expr.b, depth_cache)
        c_new, c_changed = map_critical_path_to_ternary(expr.c, depth_cache)
        if a_changed or b_changed or c_changed:
            return TernaryLut(a=a_new, b=b_new, c=c_new, imm8=expr.imm8), True
        return expr, False

    if isinstance(expr, Not):
        x_new, x_changed = map_critical_path_to_ternary(expr.x, depth_cache)
        if x_changed:
            return Not(x=x_new), True
        return expr, False

    if isinstance(expr, (And, Or, Xor)):
        cone = extract_3input_cone(expr, max_depth=4)
        if cone is not None and cone.gate_count >= 2:
            cone_depth = _expr_depth(expr, {})
            lut_depth = 1 + max(
                _expr_depth(cone.inputs[0], {}),
                _expr_depth(cone.inputs[1], {}),
                _expr_depth(cone.inputs[2], {}),
            )
            if lut_depth < cone_depth:
                return cone.to_ternary_lut(), True

        a_new, a_changed = map_critical_path_to_ternary(expr.a, depth_cache)
        b_new, b_changed = map_critical_path_to_ternary(expr.b, depth_cache)
        if a_changed or b_changed:
            return type(expr)(a=a_new, b=b_new), True
        return expr, False

    return expr, False


class DepthAwareTernaryPass(Pass):
    """Map critical path operations to ternary gates to reduce depth.

    This pass identifies nodes on the critical path and preferentially
    applies ternary mapping to reduce overall circuit depth.
    """

    def __init__(self, min_depth_reduction: int = 1):
        self.min_depth_reduction = min_depth_reduction

    @property
    def name(self) -> str:
        return "depth-ternary"

    def should_run(self, ctx: PassContext) -> bool:
        prims = {p.name for p in ctx.technology.primitives()}
        return "lop3" in prims or "vpternlog" in prims

    def run(self, ir: TickIR, ctx: PassContext) -> tuple[TickIR, PassMetrics]:
        changed = False
        expressions_modified = 0
        total_depth_reduction = 0

        depth_cache: dict[int, int] = {}

        def compute_ir_depth(tick_ir: TickIR) -> int:
            max_depth = 0
            for expr in list(tick_ir.output_exprs.values()) + list(
                tick_ir.next_state.values()
            ):
                d = _expr_depth(expr, {})
                if d > max_depth:
                    max_depth = d
            return max_depth

        original_depth = compute_ir_depth(ir)

        new_output_exprs = {}
        for name, expr in ir.output_exprs.items():
            old_depth = _expr_depth(expr, {})
            new_expr, was_changed = map_critical_path_to_ternary(expr, {})
            new_depth = _expr_depth(new_expr, {})

            if was_changed and new_depth < old_depth:
                new_output_exprs[name] = new_expr
                changed = True
                expressions_modified += 1
                total_depth_reduction += old_depth - new_depth
            else:
                new_output_exprs[name] = expr

        new_next_state = {}
        for name, expr in ir.next_state.items():
            old_depth = _expr_depth(expr, {})
            new_expr, was_changed = map_critical_path_to_ternary(expr, {})
            new_depth = _expr_depth(new_expr, {})

            if was_changed and new_depth < old_depth:
                new_next_state[name] = new_expr
                changed = True
                expressions_modified += 1
                total_depth_reduction += old_depth - new_depth
            else:
                new_next_state[name] = expr

        if changed:
            new_ir = replace(
                ir, output_exprs=new_output_exprs, next_state=new_next_state
            )
        else:
            new_ir = ir

        return new_ir, PassMetrics(
            changed=changed,
            expressions_modified=expressions_modified,
            depth_delta=-total_depth_reduction,
        )


def map_circuit_critical_path_to_ternary(
    circuit: "CircuitState",
) -> "CircuitState":
    """Map critical path operations in a CircuitState to ternary gates to reduce depth.

    Identifies 3-input cones on the critical path and replaces them with
    TernaryLut gates where doing so reduces depth.

    Args:
        circuit: Input CircuitState to optimize

    Returns:
        Optimized CircuitState with critical path operations mapped to ternary.
    """
    from stc.circuit_synth import CircuitState

    critical_nodes = circuit.critical_path_nodes()
    node_depths = circuit.node_depths()
    original_depth = circuit.depth

    if not critical_nodes:
        return circuit

    new_gates = list(circuit.gates)
    changed = False

    for g_idx in range(len(circuit.gates)):
        full_idx = circuit.input_bits + g_idx
        if full_idx not in critical_nodes:
            continue

        gate = circuit.gates[g_idx]
        if len(gate) == 5:
            continue

        op, left, right = gate
        if op not in ("xor", "and", "or"):
            continue

        candidate_inputs = set()

        def collect_cone(idx: int, depth_limit: int, collected: set[int]) -> bool:
            if idx < circuit.input_bits:
                candidate_inputs.add(idx)
                return True
            if len(candidate_inputs) > 3:
                return False

            g = idx - circuit.input_bits
            if g < 0 or g >= len(circuit.gates):
                candidate_inputs.add(idx)
                return True

            gate_inner = circuit.gates[g]
            if len(gate_inner) == 5:
                candidate_inputs.add(idx)
                return True

            inner_op, inner_left, inner_right = gate_inner
            if inner_op not in ("xor", "and", "or"):
                candidate_inputs.add(idx)
                return True

            if depth_limit <= 0:
                candidate_inputs.add(idx)
                return True

            collected.add(idx)
            if not collect_cone(inner_left, depth_limit - 1, collected):
                return False
            if inner_op != "not":
                if not collect_cone(inner_right, depth_limit - 1, collected):
                    return False
            return len(candidate_inputs) <= 3

        collected: set[int] = set()
        collected.add(full_idx)
        collect_cone(left, 2, collected)
        if op != "not":
            collect_cone(right, 2, collected)

        if len(candidate_inputs) == 3 and len(collected) >= 2:
            input_list = sorted(candidate_inputs)

            def eval_gate_cone(
                idx: int, vals: dict[int, int], cone_gates: set[int]
            ) -> int:
                if idx in vals:
                    return vals[idx]
                if idx < circuit.input_bits:
                    return 0
                g = idx - circuit.input_bits
                if g >= len(circuit.gates):
                    return 0
                gate_inner = circuit.gates[g]
                if len(gate_inner) == 5:
                    _, a, b, c, imm8 = gate_inner
                    va = eval_gate_cone(a, vals, cone_gates)
                    vb = eval_gate_cone(b, vals, cone_gates)
                    vc = eval_gate_cone(c, vals, cone_gates)
                    tidx = (va << 2) | (vb << 1) | vc
                    return (imm8 >> tidx) & 1
                inner_op, inner_left, inner_right = gate_inner
                if inner_op == "xor":
                    return eval_gate_cone(
                        inner_left, vals, cone_gates
                    ) ^ eval_gate_cone(inner_right, vals, cone_gates)
                if inner_op == "and":
                    return eval_gate_cone(
                        inner_left, vals, cone_gates
                    ) & eval_gate_cone(inner_right, vals, cone_gates)
                if inner_op == "or":
                    return eval_gate_cone(
                        inner_left, vals, cone_gates
                    ) | eval_gate_cone(inner_right, vals, cone_gates)
                if inner_op == "not":
                    return 1 - eval_gate_cone(inner_left, vals, cone_gates)
                if inner_op == "const":
                    return inner_left & 1
                return 0

            imm8 = 0
            for i in range(8):
                a_val = (i >> 2) & 1
                b_val = (i >> 1) & 1
                c_val = i & 1
                vals = {
                    input_list[0]: a_val,
                    input_list[1]: b_val,
                    input_list[2]: c_val,
                }
                result = eval_gate_cone(full_idx, vals, collected)
                if result:
                    imm8 |= 1 << i

            max_input_depth = max(node_depths.get(inp, 0) for inp in input_list)
            new_depth = max_input_depth + 1
            current_gate_depth = node_depths.get(full_idx, 0)

            if new_depth < current_gate_depth:
                new_gates[g_idx] = (
                    "ternary",
                    input_list[0],
                    input_list[1],
                    input_list[2],
                    imm8,
                )
                changed = True

    if not changed:
        return circuit

    result = CircuitState(
        input_bits=circuit.input_bits,
        output_bits=circuit.output_bits,
        gates=new_gates,
        outputs=list(circuit.outputs),
        gate_count=len(new_gates),
    )

    return result.eliminate_dead_code()


class DepthBudgetPass(Pass):
    """Reject optimizations that exceed depth budget.

    This pass acts as a filter that checks if the current IR's depth
    exceeds the configured depth budget and reverts to the best-so-far
    if the budget is violated.
    """

    @property
    def name(self) -> str:
        return "depth-budget"

    def should_run(self, ctx: PassContext) -> bool:
        return ctx.depth_budget is not None

    def run(self, ir: TickIR, ctx: PassContext) -> tuple[TickIR, PassMetrics]:
        if ctx.depth_budget is None:
            return ir, PassMetrics(changed=False)

        from stc.metrics import compute_ir_depth

        current_depth = compute_ir_depth(ir, ctx.depth_model)

        if current_depth > ctx.depth_budget:
            if ctx.best_ir is not None:
                return ctx.best_ir, PassMetrics(changed=True, depth_delta=0)
            return ir, PassMetrics(changed=False)

        return ir, PassMetrics(changed=False)
