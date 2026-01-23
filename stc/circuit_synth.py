from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import z3

from stc.tick_ir import And, Expr, Not, Slice, Var, Xor, BitVecConst


@dataclass
class SynthesisResult:
    circuit: list[Expr]
    gate_count: int
    xor_count: int
    and_count: int


def synthesize_single_output(
    table: Sequence[int],
    input_bits: int = 8,
    max_gates: int = 50,
    timeout_ms: int = 30000,
) -> tuple[Expr, int] | None:
    """
    Synthesize a boolean circuit for a single-output function.
    Uses iterative deepening to find smallest circuit.
    """
    start_time = time.time()
    deadline = start_time + timeout_ms / 1000.0

    time_per_attempt = max(timeout_ms // (max_gates + 1), 500)
    for num_gates in range(max_gates + 1):
        if time.time() > deadline:
            return None
        result = _synthesize_single_fixed_size(
            table, input_bits, num_gates, time_per_attempt
        )
        if result is not None:
            return result
    return None


def synthesize_single_output_binary_search(
    table: Sequence[int],
    input_bits: int = 8,
    min_gates: int = 0,
    max_gates: int = 100,
    timeout_ms_per_attempt: int = 30000,
) -> tuple[Expr, int] | None:
    """
    Synthesize using binary search to find minimum gates quickly.

    First verifies a solution exists at max_gates, then binary searches
    to find the minimum.
    """
    result_at_max = _synthesize_single_fixed_size(
        table, input_bits, max_gates, timeout_ms_per_attempt
    )
    if result_at_max is None:
        return None

    best = result_at_max
    best_gates = max_gates

    lo, hi = min_gates, max_gates - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        result = _synthesize_single_fixed_size(
            table, input_bits, mid, timeout_ms_per_attempt
        )
        if result is not None:
            best = result
            best_gates = mid
            hi = mid - 1
        else:
            lo = mid + 1

    return best


def _synthesize_single_fixed_size(
    table: Sequence[int],
    input_bits: int,
    num_gates: int,
    timeout_ms: int,
) -> tuple[Expr, int] | None:
    num_entries = 1 << input_bits
    assert len(table) == num_entries

    target = sum((table[i] & 1) << i for i in range(num_entries))

    input_patterns = []
    for bit in range(input_bits):
        pattern = sum(((j >> bit) & 1) << j for j in range(num_entries))
        input_patterns.append(pattern)

    if num_gates == 0:
        for bit in range(input_bits):
            if input_patterns[bit] == target:
                return (Slice(x=Var("x"), offset=bit, width=1), 0)
            if input_patterns[bit] ^ ((1 << num_entries) - 1) == target:
                return (Not(x=Slice(x=Var("x"), offset=bit, width=1)), 0)
        return None

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    OP_XOR = 0
    OP_AND = 1
    OP_OR = 2

    gate_op = [z3.Int(f"op_{g}") for g in range(num_gates)]
    gate_left = [z3.Int(f"left_{g}") for g in range(num_gates)]
    gate_right = [z3.Int(f"right_{g}") for g in range(num_gates)]
    gate_val = [z3.BitVec(f"val_{g}", num_entries) for g in range(num_gates)]

    for g in range(num_gates):
        solver.add(gate_op[g] >= 0, gate_op[g] <= 2)
        num_avail = input_bits + g
        solver.add(gate_left[g] >= 0, gate_left[g] < num_avail)
        solver.add(gate_right[g] >= 0, gate_right[g] < num_avail)
        solver.add(gate_left[g] <= gate_right[g])

    def get_node_val(idx: int, g: int) -> z3.BitVecRef:
        if idx < input_bits:
            return z3.BitVecVal(input_patterns[idx], num_entries)
        else:
            return gate_val[idx - input_bits]

    for g in range(num_gates):
        num_avail = input_bits + g

        left_val = z3.BitVec(f"lv_{g}", num_entries)
        right_val = z3.BitVec(f"rv_{g}", num_entries)

        for i in range(num_avail):
            nv = get_node_val(i, g)
            solver.add(z3.Implies(gate_left[g] == i, left_val == nv))
            solver.add(z3.Implies(gate_right[g] == i, right_val == nv))

        solver.add(
            z3.Implies(gate_op[g] == OP_XOR, gate_val[g] == (left_val ^ right_val))
        )
        solver.add(
            z3.Implies(gate_op[g] == OP_AND, gate_val[g] == (left_val & right_val))
        )
        solver.add(
            z3.Implies(gate_op[g] == OP_OR, gate_val[g] == (left_val | right_val))
        )

    output_sel = z3.Int("out")
    total_nodes = input_bits + num_gates
    solver.add(output_sel >= 0, output_sel < total_nodes)

    output_val = z3.BitVec("out_val", num_entries)
    for i in range(total_nodes):
        nv = get_node_val(i, num_gates)
        solver.add(z3.Implies(output_sel == i, output_val == nv))

    target_bv = z3.BitVecVal(target, num_entries)
    inv_target = z3.BitVecVal(target ^ ((1 << num_entries) - 1), num_entries)
    invert_output = z3.Bool("inv")
    solver.add(
        z3.Or(
            z3.And(z3.Not(invert_output), output_val == target_bv),
            z3.And(invert_output, output_val == inv_target),
        )
    )

    if solver.check() != z3.sat:
        return None

    model = solver.model()

    def get_int(v: z3.ExprRef) -> int:
        return model.eval(v, model_completion=True).as_long()

    def get_bool(v: z3.ExprRef) -> bool:
        return z3.is_true(model.eval(v, model_completion=True))

    node_exprs: list[Expr] = [
        Slice(x=Var("x"), offset=i, width=1) for i in range(input_bits)
    ]
    actual_gates = 0

    for g in range(num_gates):
        op = get_int(gate_op[g])
        li = get_int(gate_left[g])
        ri = get_int(gate_right[g])
        actual_gates += 1

        if op == OP_XOR:
            expr = Xor(a=node_exprs[li], b=node_exprs[ri])
        elif op == OP_AND:
            expr = And(a=node_exprs[li], b=node_exprs[ri])
        else:
            expr = Not(x=And(a=Not(x=node_exprs[li]), b=Not(x=node_exprs[ri])))
        node_exprs.append(expr)

    out_idx = get_int(output_sel)
    result_expr = node_exprs[out_idx]
    if get_bool(invert_output):
        result_expr = Not(x=result_expr)

    return (result_expr, actual_gates)


def synthesize_multi_output(
    table: Sequence[int],
    input_bits: int = 8,
    output_bits: int = 8,
    max_gates: int = 150,
    timeout_ms: int = 60000,
) -> SynthesisResult | None:
    """
    Synthesize each output bit independently, then try to merge.
    """
    num_entries = 1 << input_bits
    assert len(table) == num_entries

    output_exprs = []
    total_gates = 0

    for out_bit in range(output_bits):
        bit_table = [(table[i] >> out_bit) & 1 for i in range(num_entries)]
        result = synthesize_single_output(
            bit_table,
            input_bits,
            max_gates=max_gates // output_bits,
            timeout_ms=timeout_ms // output_bits,
        )
        if result is None:
            return None
        expr, gates = result
        output_exprs.append(expr)
        total_gates += gates

    xor_count = _count_ops(output_exprs, Xor)
    and_count = _count_ops(output_exprs, And)

    return SynthesisResult(
        circuit=output_exprs,
        gate_count=total_gates,
        xor_count=xor_count,
        and_count=and_count,
    )


def synthesize_multi_output_shared(
    table: Sequence[int],
    input_bits: int = 8,
    output_bits: int = 8,
    max_gates: int = 200,
    timeout_ms: int = 300000,
    min_gates: int | None = None,
) -> SynthesisResult | None:
    """
    Synthesize a multi-output circuit with shared gates.

    Uses iterative deepening to find the smallest circuit where gates
    can be shared across multiple outputs.

    Args:
        table: Lookup table mapping input values to output values
        input_bits: Number of input bits
        output_bits: Number of output bits
        max_gates: Maximum gates to search for
        timeout_ms: Total timeout in milliseconds
        min_gates: Optional minimum gates to start search from (skip smaller sizes)

    Returns:
        SynthesisResult if successful, None if no circuit found within limits
    """
    start_time = time.time()
    deadline = start_time + timeout_ms / 1000.0

    start_gates = min_gates if min_gates is not None else 0
    range_size = max_gates - start_gates + 1

    base_time = max(timeout_ms // (range_size * 2), 500)

    for num_gates in range(start_gates, max_gates + 1):
        if time.time() > deadline:
            return None
        time_for_this = min(base_time * (1 + num_gates // 10), timeout_ms // 4)
        result = _synthesize_multi_fixed_size(
            table, input_bits, output_bits, num_gates, time_for_this
        )
        if result is not None:
            return result
    return None


def _synthesize_multi_fixed_size(
    table: Sequence[int],
    input_bits: int,
    output_bits: int,
    num_gates: int,
    timeout_ms: int,
) -> SynthesisResult | None:
    """
    Try to synthesize a multi-output circuit with exactly num_gates gates.
    """
    num_entries = 1 << input_bits
    assert len(table) == num_entries

    targets = []
    for out_bit in range(output_bits):
        target = sum(((table[i] >> out_bit) & 1) << i for i in range(num_entries))
        targets.append(target)

    input_patterns = []
    for bit in range(input_bits):
        pattern = sum(((j >> bit) & 1) << j for j in range(num_entries))
        input_patterns.append(pattern)

    if num_gates == 0:
        output_exprs = []
        for out_bit in range(output_bits):
            target = targets[out_bit]
            found = False
            for bit in range(input_bits):
                if input_patterns[bit] == target:
                    output_exprs.append(Slice(x=Var("x"), offset=bit, width=1))
                    found = True
                    break
                if input_patterns[bit] ^ ((1 << num_entries) - 1) == target:
                    output_exprs.append(Not(x=Slice(x=Var("x"), offset=bit, width=1)))
                    found = True
                    break
            if not found:
                return None
        return SynthesisResult(
            circuit=output_exprs, gate_count=0, xor_count=0, and_count=0
        )

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    OP_XOR = 0
    OP_AND = 1
    OP_OR = 2

    gate_op = [z3.Int(f"op_{g}") for g in range(num_gates)]
    gate_left = [z3.Int(f"left_{g}") for g in range(num_gates)]
    gate_right = [z3.Int(f"right_{g}") for g in range(num_gates)]
    gate_val = [z3.BitVec(f"val_{g}", num_entries) for g in range(num_gates)]

    for g in range(num_gates):
        solver.add(gate_op[g] >= 0, gate_op[g] <= 2)
        num_avail = input_bits + g
        solver.add(gate_left[g] >= 0, gate_left[g] < num_avail)
        solver.add(gate_right[g] >= 0, gate_right[g] < num_avail)
        solver.add(gate_left[g] <= gate_right[g])

    def get_node_val(idx: int, g: int) -> z3.BitVecRef:
        if idx < input_bits:
            return z3.BitVecVal(input_patterns[idx], num_entries)
        else:
            return gate_val[idx - input_bits]

    for g in range(num_gates):
        num_avail = input_bits + g

        left_val = z3.BitVec(f"lv_{g}", num_entries)
        right_val = z3.BitVec(f"rv_{g}", num_entries)

        for i in range(num_avail):
            nv = get_node_val(i, g)
            solver.add(z3.Implies(gate_left[g] == i, left_val == nv))
            solver.add(z3.Implies(gate_right[g] == i, right_val == nv))

        solver.add(
            z3.Implies(gate_op[g] == OP_XOR, gate_val[g] == (left_val ^ right_val))
        )
        solver.add(
            z3.Implies(gate_op[g] == OP_AND, gate_val[g] == (left_val & right_val))
        )
        solver.add(
            z3.Implies(gate_op[g] == OP_OR, gate_val[g] == (left_val | right_val))
        )

    output_sels = [z3.Int(f"out_{o}") for o in range(output_bits)]
    invert_outputs = [z3.Bool(f"inv_{o}") for o in range(output_bits)]
    total_nodes = input_bits + num_gates

    for out_bit in range(output_bits):
        solver.add(output_sels[out_bit] >= 0, output_sels[out_bit] < total_nodes)

        output_val = z3.BitVec(f"out_val_{out_bit}", num_entries)
        for i in range(total_nodes):
            nv = get_node_val(i, num_gates)
            solver.add(z3.Implies(output_sels[out_bit] == i, output_val == nv))

        target_bv = z3.BitVecVal(targets[out_bit], num_entries)
        inv_target = z3.BitVecVal(
            targets[out_bit] ^ ((1 << num_entries) - 1), num_entries
        )
        solver.add(
            z3.Or(
                z3.And(z3.Not(invert_outputs[out_bit]), output_val == target_bv),
                z3.And(invert_outputs[out_bit], output_val == inv_target),
            )
        )

    if solver.check() != z3.sat:
        return None

    model = solver.model()

    def get_int(v: z3.ExprRef) -> int:
        return model.eval(v, model_completion=True).as_long()

    def get_bool(v: z3.ExprRef) -> bool:
        return z3.is_true(model.eval(v, model_completion=True))

    node_exprs: list[Expr] = [
        Slice(x=Var("x"), offset=i, width=1) for i in range(input_bits)
    ]
    xor_count = 0
    and_count = 0

    for g in range(num_gates):
        op = get_int(gate_op[g])
        li = get_int(gate_left[g])
        ri = get_int(gate_right[g])

        if op == OP_XOR:
            expr = Xor(a=node_exprs[li], b=node_exprs[ri])
            xor_count += 1
        elif op == OP_AND:
            expr = And(a=node_exprs[li], b=node_exprs[ri])
            and_count += 1
        else:
            expr = Not(x=And(a=Not(x=node_exprs[li]), b=Not(x=node_exprs[ri])))
            and_count += 1
        node_exprs.append(expr)

    output_exprs = []
    for out_bit in range(output_bits):
        out_idx = get_int(output_sels[out_bit])
        result_expr = node_exprs[out_idx]
        if get_bool(invert_outputs[out_bit]):
            result_expr = Not(x=result_expr)
        output_exprs.append(result_expr)

    return SynthesisResult(
        circuit=output_exprs,
        gate_count=num_gates,
        xor_count=xor_count,
        and_count=and_count,
    )


def synthesize_single_output_greedy(
    table: Sequence[int],
    input_bits: int = 8,
    max_gates: int = 200,
    max_values: int = 10000,
    timeout_ms: int = 60000,
) -> tuple[Expr, int] | None:
    """
    Synthesize a circuit using greedy BFS construction with pruning.

    Builds up a set of reachable values by adding gates that combine
    existing values. Uses Hamming distance scoring to prune the search space.
    """
    num_entries = 1 << input_bits
    assert len(table) == num_entries

    start_time = time.time()
    deadline = start_time + timeout_ms / 1000.0

    target = sum((table[i] & 1) << i for i in range(num_entries))
    inv_target = target ^ ((1 << num_entries) - 1)
    all_ones = (1 << num_entries) - 1

    def popcount(x: int) -> int:
        return bin(x).count("1")

    def hamming_to_target(val: int) -> int:
        return min(popcount(val ^ target), popcount(val ^ inv_target))

    input_patterns = []
    for bit in range(input_bits):
        pattern = sum(((j >> bit) & 1) << j for j in range(num_entries))
        input_patterns.append(pattern)

    values: dict[int, tuple[Expr, int]] = {}
    for bit in range(input_bits):
        pattern = input_patterns[bit]
        expr = Slice(x=Var("x"), offset=bit, width=1)
        values[pattern] = (expr, 0)
        values[pattern ^ all_ones] = (Not(x=expr), 0)

    if target in values:
        return values[target]
    if inv_target in values:
        expr, gates = values[inv_target]
        return (Not(x=expr) if not isinstance(expr, Not) else expr.x, gates)

    for depth in range(max_gates):
        if time.time() > deadline:
            return None
        new_values: dict[int, tuple[Expr, int, int]] = {}
        value_list = list(values.keys())

        for i, v1 in enumerate(value_list):
            if i % 100 == 0 and time.time() > deadline:
                return None
            for v2 in value_list[i + 1 :]:
                xor_val = v1 ^ v2
                and_val = v1 & v2
                or_val = v1 | v2

                for val, op_name in [
                    (xor_val, "xor"),
                    (and_val, "and"),
                    (or_val, "or"),
                ]:
                    if val in values or val in new_values:
                        continue
                    if val == 0 or val == all_ones:
                        continue

                    expr1, g1 = values[v1]
                    expr2, g2 = values[v2]
                    new_gates = max(g1, g2) + 1

                    if op_name == "xor":
                        new_expr = Xor(a=expr1, b=expr2)
                    elif op_name == "and":
                        new_expr = And(a=expr1, b=expr2)
                    else:
                        new_expr = Not(x=And(a=Not(x=expr1), b=Not(x=expr2)))

                    ham = hamming_to_target(val)
                    new_values[val] = (new_expr, new_gates, ham)

                    if val == target:
                        return (new_expr, new_gates)
                    if val == inv_target:
                        return (Not(x=new_expr), new_gates)

        if not new_values:
            break

        sorted_new = sorted(new_values.items(), key=lambda x: x[1][2])
        for val, (expr, g, _) in sorted_new[:max_values]:
            if len(values) < max_values:
                values[val] = (expr, g)

    return None


def synthesize_single_output_cegar(
    table: Sequence[int],
    input_bits: int = 8,
    max_gates: int = 100,
    timeout_ms_per_round: int = 10000,
    max_rounds: int = 100,
    timeout_ms_total: int = 300000,
) -> tuple[Expr, int] | None:
    """
    Synthesize using CEGAR (Counter-Example Guided Abstraction Refinement).

    Start with a small set of input-output constraints, find a circuit,
    then verify against full table. Add counter-examples as needed.
    """
    num_entries = 1 << input_bits
    assert len(table) == num_entries

    start_time = time.time()
    deadline = start_time + timeout_ms_total / 1000.0

    def evaluate_circuit(
        node_exprs: list[int], out_idx: int, invert: bool, inp: int
    ) -> int:
        val = node_exprs[out_idx]
        return val ^ 1 if invert else val

    input_patterns = []
    for bit in range(input_bits):
        pattern = sum(((j >> bit) & 1) << j for j in range(num_entries))
        input_patterns.append(pattern)

    constrained_inputs = [0, 1, 2, 255]

    for num_gates in range(max_gates + 1):
        if time.time() > deadline:
            return None
        for _ in range(max_rounds):
            if time.time() > deadline:
                return None
            target = sum((table[i] & 1) << i for i in constrained_inputs)
            target_compact = sum(
                ((table[constrained_inputs[j]] & 1) << j)
                for j in range(len(constrained_inputs))
            )

            if num_gates == 0:
                for bit in range(input_bits):
                    pattern_compact = sum(
                        (((constrained_inputs[j] >> bit) & 1) << j)
                        for j in range(len(constrained_inputs))
                    )
                    all_ones_compact = (1 << len(constrained_inputs)) - 1
                    if pattern_compact == target_compact:
                        full_pattern = input_patterns[bit]
                        full_target = sum(
                            (table[i] & 1) << i for i in range(num_entries)
                        )
                        if full_pattern == full_target:
                            return (Slice(x=Var("x"), offset=bit, width=1), 0)
                    if pattern_compact ^ all_ones_compact == target_compact:
                        full_pattern = input_patterns[bit] ^ ((1 << num_entries) - 1)
                        full_target = sum(
                            (table[i] & 1) << i for i in range(num_entries)
                        )
                        if full_pattern == full_target:
                            return (Not(x=Slice(x=Var("x"), offset=bit, width=1)), 0)
                break

            solver = z3.Solver()
            solver.set("timeout", timeout_ms_per_round)

            OP_XOR, OP_AND, OP_OR = 0, 1, 2
            nc = len(constrained_inputs)

            gate_op = [z3.Int(f"op_{g}") for g in range(num_gates)]
            gate_left = [z3.Int(f"left_{g}") for g in range(num_gates)]
            gate_right = [z3.Int(f"right_{g}") for g in range(num_gates)]
            gate_val = [z3.BitVec(f"val_{g}", nc) for g in range(num_gates)]

            for g in range(num_gates):
                solver.add(gate_op[g] >= 0, gate_op[g] <= 2)
                num_avail = input_bits + g
                solver.add(gate_left[g] >= 0, gate_left[g] < num_avail)
                solver.add(gate_right[g] >= 0, gate_right[g] < num_avail)
                solver.add(gate_left[g] <= gate_right[g])

            compact_input_patterns = []
            for bit in range(input_bits):
                pattern = sum(
                    (((constrained_inputs[j] >> bit) & 1) << j) for j in range(nc)
                )
                compact_input_patterns.append(pattern)

            def get_node_val(idx: int) -> z3.BitVecRef:
                if idx < input_bits:
                    return z3.BitVecVal(compact_input_patterns[idx], nc)
                return gate_val[idx - input_bits]

            for g in range(num_gates):
                num_avail = input_bits + g
                left_val = z3.BitVec(f"lv_{g}", nc)
                right_val = z3.BitVec(f"rv_{g}", nc)

                for i in range(num_avail):
                    nv = get_node_val(i)
                    solver.add(z3.Implies(gate_left[g] == i, left_val == nv))
                    solver.add(z3.Implies(gate_right[g] == i, right_val == nv))

                solver.add(
                    z3.Implies(
                        gate_op[g] == OP_XOR, gate_val[g] == (left_val ^ right_val)
                    )
                )
                solver.add(
                    z3.Implies(
                        gate_op[g] == OP_AND, gate_val[g] == (left_val & right_val)
                    )
                )
                solver.add(
                    z3.Implies(
                        gate_op[g] == OP_OR, gate_val[g] == (left_val | right_val)
                    )
                )

            output_sel = z3.Int("out")
            total_nodes = input_bits + num_gates
            solver.add(output_sel >= 0, output_sel < total_nodes)

            output_val = z3.BitVec("out_val", nc)
            for i in range(total_nodes):
                solver.add(z3.Implies(output_sel == i, output_val == get_node_val(i)))

            target_bv = z3.BitVecVal(target_compact, nc)
            inv_target_bv = z3.BitVecVal(target_compact ^ ((1 << nc) - 1), nc)
            invert_output = z3.Bool("inv")
            solver.add(
                z3.Or(
                    z3.And(z3.Not(invert_output), output_val == target_bv),
                    z3.And(invert_output, output_val == inv_target_bv),
                )
            )

            if solver.check() != z3.sat:
                break

            model = solver.model()

            def get_int(v: z3.ExprRef) -> int:
                return model.eval(v, model_completion=True).as_long()

            def get_bool(v: z3.ExprRef) -> bool:
                return z3.is_true(model.eval(v, model_completion=True))

            ops = [get_int(gate_op[g]) for g in range(num_gates)]
            lefts = [get_int(gate_left[g]) for g in range(num_gates)]
            rights = [get_int(gate_right[g]) for g in range(num_gates)]
            out_idx = get_int(output_sel)
            invert = get_bool(invert_output)

            def eval_full(inp: int) -> int:
                node_vals = [(inp >> b) & 1 for b in range(input_bits)]
                for g in range(num_gates):
                    l, r = node_vals[lefts[g]], node_vals[rights[g]]
                    if ops[g] == OP_XOR:
                        node_vals.append(l ^ r)
                    elif ops[g] == OP_AND:
                        node_vals.append(l & r)
                    else:
                        node_vals.append(1 - ((1 - l) & (1 - r)))
                result = node_vals[out_idx]
                return result ^ 1 if invert else result

            counter_example = None
            for i in range(num_entries):
                if i in constrained_inputs:
                    continue
                expected = table[i] & 1
                got = eval_full(i)
                if got != expected:
                    counter_example = i
                    break

            if counter_example is None:
                node_exprs: list[Expr] = [
                    Slice(x=Var("x"), offset=b, width=1) for b in range(input_bits)
                ]
                for g in range(num_gates):
                    if ops[g] == OP_XOR:
                        expr = Xor(a=node_exprs[lefts[g]], b=node_exprs[rights[g]])
                    elif ops[g] == OP_AND:
                        expr = And(a=node_exprs[lefts[g]], b=node_exprs[rights[g]])
                    else:
                        expr = Not(
                            x=And(
                                a=Not(x=node_exprs[lefts[g]]),
                                b=Not(x=node_exprs[rights[g]]),
                            )
                        )
                    node_exprs.append(expr)

                result_expr = node_exprs[out_idx]
                if invert:
                    result_expr = Not(x=result_expr)
                return (result_expr, num_gates)

            constrained_inputs.append(counter_example)

    return None


def synthesize_via_anf(
    table: Sequence[int],
    input_bits: int = 8,
) -> tuple[Expr, int]:
    """
    Synthesize a circuit using Algebraic Normal Form (ANF) decomposition.

    This always produces a working circuit by computing the ANF (Zhegalkin polynomial)
    of the boolean function and building a circuit from it. The circuit is not optimal
    but is guaranteed correct.

    Returns (expr, gate_count) - always succeeds.
    """
    num_entries = 1 << input_bits
    assert len(table) == num_entries

    anf = _compute_anf(table, input_bits)

    input_exprs = [Slice(x=Var("x"), offset=i, width=1) for i in range(input_bits)]

    term_cache: dict[int, Expr] = {}
    for i in range(input_bits):
        term_cache[1 << i] = input_exprs[i]

    result_expr: Expr | None = None
    gate_count = 0

    for term_mask in range(num_entries):
        if not anf[term_mask]:
            continue

        if term_mask == 0:
            term_expr = BitVecConst(width=1, value=1)
        else:
            term_expr = _build_and_term(term_mask, input_exprs, term_cache)
            if term_mask not in term_cache:
                bits_in_term = bin(term_mask).count("1")
                gate_count += bits_in_term - 1
                term_cache[term_mask] = term_expr

        if result_expr is None:
            result_expr = term_expr
        else:
            result_expr = Xor(a=result_expr, b=term_expr)
            gate_count += 1

    if result_expr is None:
        result_expr = BitVecConst(width=1, value=0)

    return (result_expr, gate_count)


def _compute_anf(table: Sequence[int], input_bits: int) -> list[int]:
    """Compute ANF coefficients using Möbius transform."""
    num_entries = 1 << input_bits
    anf = [table[i] & 1 for i in range(num_entries)]

    for i in range(input_bits):
        step = 1 << i
        for j in range(0, num_entries, step * 2):
            for k in range(step):
                anf[j + k + step] ^= anf[j + k]

    return anf


def _build_and_term(mask: int, input_exprs: list[Expr], cache: dict[int, Expr]) -> Expr:
    """Build an AND term for the given variable mask."""
    if mask in cache:
        return cache[mask]

    bits = []
    m = mask
    while m:
        bit = m & -m
        bits.append(bit)
        m &= m - 1

    if len(bits) == 1:
        return input_exprs[bits[0].bit_length() - 1]

    mid = len(bits) // 2
    left_mask = sum(bits[:mid])
    right_mask = sum(bits[mid:])

    left_expr = _build_and_term(left_mask, input_exprs, cache)
    right_expr = _build_and_term(right_mask, input_exprs, cache)

    result = And(a=left_expr, b=right_expr)
    cache[mask] = result
    return result


def synthesize_multi_output_anf(
    table: Sequence[int],
    input_bits: int = 8,
    output_bits: int = 8,
    optimize_xor: bool = False,
) -> SynthesisResult:
    """
    Synthesize a multi-output circuit using ANF decomposition.

    Always succeeds - produces a working (but not optimal) circuit.
    If optimize_xor=True, attempts to share XOR subexpressions across outputs.
    """
    num_entries = 1 << input_bits
    assert len(table) == num_entries

    input_exprs = [Slice(x=Var("x"), offset=i, width=1) for i in range(input_bits)]
    term_cache: dict[int, Expr] = {}
    for i in range(input_bits):
        term_cache[1 << i] = input_exprs[i]

    all_anfs = []
    for out_bit in range(output_bits):
        bit_table = [(table[i] >> out_bit) & 1 for i in range(num_entries)]
        anf = _compute_anf(bit_table, input_bits)
        term_set = frozenset(i for i in range(num_entries) if anf[i])
        all_anfs.append(term_set)

    total_ands = 0
    for term_mask in set().union(*all_anfs):
        if term_mask == 0:
            continue
        if term_mask not in term_cache:
            term_expr = _build_and_term(term_mask, input_exprs, term_cache)
            term_cache[term_mask] = term_expr
            bits_in_term = bin(term_mask).count("1")
            total_ands += bits_in_term - 1

    if optimize_xor:
        output_exprs, total_xors = _build_xor_trees_optimized(
            all_anfs, term_cache, num_entries
        )
    else:
        output_exprs = []
        total_xors = 0
        for term_set in all_anfs:
            result_expr: Expr | None = None
            for term_mask in sorted(term_set):
                if term_mask == 0:
                    term_expr = BitVecConst(width=1, value=1)
                else:
                    term_expr = term_cache[term_mask]

                if result_expr is None:
                    result_expr = term_expr
                else:
                    result_expr = Xor(a=result_expr, b=term_expr)
                    total_xors += 1

            if result_expr is None:
                result_expr = BitVecConst(width=1, value=0)
            output_exprs.append(result_expr)

    return SynthesisResult(
        circuit=output_exprs,
        gate_count=total_xors + total_ands,
        xor_count=total_xors,
        and_count=total_ands,
    )


def _build_xor_trees_optimized(
    all_anfs: list[frozenset[int]],
    term_cache: dict[int, Expr],
    num_entries: int,
) -> tuple[list[Expr], int]:
    """Build XOR trees with shared subexpressions."""
    xor_cache: dict[frozenset[int], Expr] = {}
    total_xors = 0

    for term_mask in term_cache:
        xor_cache[frozenset([term_mask])] = term_cache[term_mask]
    xor_cache[frozenset([0])] = BitVecConst(width=1, value=1)

    def build_xor_tree(term_set: frozenset[int]) -> Expr:
        nonlocal total_xors

        if term_set in xor_cache:
            return xor_cache[term_set]

        if len(term_set) == 0:
            return BitVecConst(width=1, value=0)

        terms = sorted(term_set)
        mid = len(terms) // 2
        left_set = frozenset(terms[:mid])
        right_set = frozenset(terms[mid:])

        left_expr = build_xor_tree(left_set)
        right_expr = build_xor_tree(right_set)

        result = Xor(a=left_expr, b=right_expr)
        total_xors += 1
        xor_cache[term_set] = result
        return result

    output_exprs = []
    for term_set in all_anfs:
        expr = build_xor_tree(term_set)
        output_exprs.append(expr)

    return output_exprs, total_xors


def _count_ops(exprs: list[Expr], op_type: type) -> int:
    count = 0
    seen: set[int] = set()

    def visit(e: Expr) -> None:
        nonlocal count
        eid = id(e)
        if eid in seen:
            return
        seen.add(eid)
        if isinstance(e, op_type):
            count += 1
        if isinstance(e, (Xor, And)):
            visit(e.a)
            visit(e.b)
        elif isinstance(e, Not):
            visit(e.x)

    for expr in exprs:
        visit(expr)
    return count


@dataclass
class CircuitState:
    """Serializable circuit state for incremental optimization."""

    input_bits: int
    output_bits: int
    gates: list[tuple[str, int, int]]
    outputs: list[tuple[int, bool]]
    gate_count: int

    def to_dict(self) -> dict:
        return {
            "input_bits": self.input_bits,
            "output_bits": self.output_bits,
            "gates": self.gates,
            "outputs": self.outputs,
            "gate_count": self.gate_count,
        }

    @staticmethod
    def from_dict(d: dict) -> "CircuitState":
        return CircuitState(
            input_bits=d["input_bits"],
            output_bits=d["output_bits"],
            gates=[tuple(g) for g in d["gates"]],
            outputs=[tuple(o) for o in d["outputs"]],
            gate_count=d["gate_count"],
        )

    def evaluate(self, x: int) -> int:
        """Evaluate circuit on input x, return output value."""
        node_vals = [(x >> i) & 1 for i in range(self.input_bits)]

        for op, left, right in self.gates:
            if left >= len(node_vals):
                raise IndexError(f"Invalid left index {left}, only {len(node_vals)} nodes")
            if op not in ("const", "not") and right >= len(node_vals):
                raise IndexError(f"Invalid right index {right}, only {len(node_vals)} nodes")

            if op == "xor":
                node_vals.append(node_vals[left] ^ node_vals[right])
            elif op == "and":
                node_vals.append(node_vals[left] & node_vals[right])
            elif op == "or":
                node_vals.append(node_vals[left] | node_vals[right])
            elif op == "not":
                node_vals.append(node_vals[left] ^ 1)
            elif op == "const":
                node_vals.append(left & 1)

        result = 0
        for bit, (idx, invert) in enumerate(self.outputs):
            val = node_vals[idx]
            if invert:
                val ^= 1
            result |= val << bit

        return result

    def to_exprs(self) -> list[Expr]:
        """Convert back to Tick-IR expressions."""
        node_exprs: list[Expr] = [
            Slice(x=Var("x"), offset=i, width=1) for i in range(self.input_bits)
        ]

        for op, left, right in self.gates:
            if op == "xor":
                expr = Xor(a=node_exprs[left], b=node_exprs[right])
            elif op == "and":
                expr = And(a=node_exprs[left], b=node_exprs[right])
            elif op == "not":
                expr = Not(x=node_exprs[left])
            elif op == "const":
                expr = BitVecConst(width=right, value=left)
            else:
                expr = Not(x=And(a=Not(x=node_exprs[left]), b=Not(x=node_exprs[right])))
            node_exprs.append(expr)

        output_exprs = []
        for idx, invert in self.outputs:
            expr = node_exprs[idx]
            if invert:
                expr = Not(x=expr)
            output_exprs.append(expr)

        return output_exprs


def circuit_to_state(
    exprs: list[Expr], input_bits: int, output_bits: int
) -> CircuitState:
    """Convert Tick-IR expressions to CircuitState."""
    node_to_idx: dict[int, int] = {}
    gates: list[tuple[str, int, int]] = []

    def process_expr(e: Expr) -> int:
        """Process expression and return node index."""
        eid = id(e)
        if eid in node_to_idx:
            return node_to_idx[eid]

        if isinstance(e, Slice) and isinstance(e.x, Var) and e.x.name == "x":
            node_to_idx[eid] = e.offset
            return e.offset

        if isinstance(e, BitVecConst):
            new_idx = input_bits + len(gates)
            gates.append(("const", e.value, e.width))
            node_to_idx[eid] = new_idx
            return new_idx

        if isinstance(e, Not):
            inner_idx = process_expr(e.x)
            new_idx = input_bits + len(gates)
            gates.append(("not", inner_idx, 0))
            node_to_idx[eid] = new_idx
            return new_idx

        if isinstance(e, Xor):
            left_idx = process_expr(e.a)
            right_idx = process_expr(e.b)
            new_idx = input_bits + len(gates)
            gates.append(("xor", left_idx, right_idx))
            node_to_idx[eid] = new_idx
            return new_idx

        if isinstance(e, And):
            left_idx = process_expr(e.a)
            right_idx = process_expr(e.b)
            new_idx = input_bits + len(gates)
            gates.append(("and", left_idx, right_idx))
            node_to_idx[eid] = new_idx
            return new_idx

        raise ValueError(f"Unknown expression type: {type(e)}")

    outputs = []
    for expr in exprs:
        idx = process_expr(expr)
        outputs.append((idx, False))

    return CircuitState(
        input_bits=input_bits,
        output_bits=output_bits,
        gates=gates,
        outputs=outputs,
        gate_count=len(gates),
    )


class IncrementalOptimizer:
    """Incrementally optimize a circuit, allowing save/resume."""

    def __init__(
        self,
        table: Sequence[int],
        input_bits: int = 8,
        output_bits: int = 8,
        initial_state: CircuitState | None = None,
    ):
        self.table = list(table)
        self.input_bits = input_bits
        self.output_bits = output_bits
        self.num_entries = 1 << input_bits

        if initial_state is not None:
            self.best_state = initial_state
        else:
            result = synthesize_multi_output_anf(
                table, input_bits, output_bits, optimize_xor=True
            )
            self.best_state = circuit_to_state(result.circuit, input_bits, output_bits)

        self.iterations = 0
        self.improvements = 0

    def get_gate_count(self) -> int:
        return self.best_state.gate_count

    def get_state(self) -> CircuitState:
        return self.best_state

    def verify(self) -> bool:
        """Verify current circuit is correct."""
        for i in range(self.num_entries):
            expected = self.table[i] & ((1 << self.output_bits) - 1)
            got = self.best_state.evaluate(i)
            if got != expected:
                return False
        return True

    def optimize_step(self, timeout_ms: int = 5000) -> bool:
        """
        Try one optimization step. Returns True if improvement found.
        """
        self.iterations += 1

        improved = self._try_remove_unused_gates()
        if improved:
            self.improvements += 1
            return True

        improved = self._try_merge_duplicate_gates()
        if improved:
            self.improvements += 1
            return True

        improved = self._try_simplify_xor_chains(timeout_ms)
        if improved:
            self.improvements += 1
            return True

        improved = self._try_value_based_merge()
        if improved:
            self.improvements += 1
            return True

        improved = self._try_and_factoring()
        if improved:
            self.improvements += 1
            return True

        improved = self._try_random_restructure()
        if improved:
            self.improvements += 1
            return True

        return False

    def _try_random_restructure(self) -> bool:
        """Randomly restructure part of the circuit to escape local minima."""
        import random

        gates = list(self.best_state.gates)
        if len(gates) < 10:
            return False

        xor_indices = [i for i, (op, _, _) in enumerate(gates) if op == "xor"]
        if len(xor_indices) < 2:
            return False

        idx1, idx2 = random.sample(xor_indices, 2)
        op1, l1, r1 = gates[idx1]
        op2, l2, r2 = gates[idx2]

        full1 = self.input_bits + idx1
        full2 = self.input_bits + idx2

        use_count = {}
        for i, (op, left, right) in enumerate(gates):
            use_count[left] = use_count.get(left, 0) + 1
            if op != "const":
                use_count[right] = use_count.get(right, 0) + 1
        for out_idx, _ in self.best_state.outputs:
            use_count[out_idx] = use_count.get(out_idx, 0) + 1

        if use_count.get(full1, 0) != 1 or use_count.get(full2, 0) != 1:
            return False

        shared = None
        if l1 == l2:
            shared = l1
            new_xor = (r1, r2)
        elif l1 == r2:
            shared = l1
            new_xor = (r1, l2)
        elif r1 == l2:
            shared = r1
            new_xor = (l1, r2)
        elif r1 == r2:
            shared = r1
            new_xor = (l1, l2)

        if shared is None:
            return False

        new_gates = list(gates)
        inner_idx = self.input_bits + len(new_gates)
        new_gates.append(("xor", new_xor[0], new_xor[1]))

        outer_idx = self.input_bits + len(new_gates)
        new_gates.append(("xor", shared, inner_idx))

        new_gates[idx1] = ("const", 0, 0)
        new_gates[idx2] = ("const", 0, 0)

        new_outputs = []
        for out_idx, inv in self.best_state.outputs:
            if out_idx == full1 or out_idx == full2:
                new_outputs.append((outer_idx, inv))
            else:
                new_outputs.append((out_idx, inv))

        test_state = CircuitState(
            input_bits=self.input_bits,
            output_bits=self.output_bits,
            gates=new_gates,
            outputs=new_outputs,
            gate_count=len(new_gates),
        )

        for i in range(min(16, self.num_entries)):
            if test_state.evaluate(i) != self.table[i]:
                return False

        self.best_state = test_state
        self._try_remove_unused_gates()

        if not self.verify():
            return False

        return True

    def optimize(
        self,
        max_iterations: int = 100,
        timeout_ms_per_step: int = 5000,
        target_gates: int | None = None,
        callback: callable | None = None,
    ) -> CircuitState:
        """
        Run optimization loop.

        Args:
            max_iterations: Maximum optimization steps
            timeout_ms_per_step: Timeout per optimization attempt
            target_gates: Stop if gate count reaches this target
            callback: Called after each iteration with (iteration, gate_count, improved)
        """
        for i in range(max_iterations):
            if target_gates and self.best_state.gate_count <= target_gates:
                break

            improved = self.optimize_step(timeout_ms_per_step)

            if callback:
                callback(i, self.best_state.gate_count, improved)

        return self.best_state

    def anneal(
        self,
        max_iterations: int = 1000,
        initial_temp: float = 10.0,
        cooling_rate: float = 0.995,
        timeout_seconds: float = 300.0,
        callback: callable | None = None,
    ) -> CircuitState:
        """
        Simulated annealing optimization.

        Randomly mutates circuit and accepts worse solutions with decreasing probability.
        """
        import math
        import random

        deadline = time.time() + timeout_seconds

        current_state = self.best_state
        current_cost = current_state.gate_count
        best_cost = current_cost
        temp = initial_temp

        for i in range(max_iterations):
            if time.time() > deadline:
                break
            mutated = self._mutate_circuit(current_state)
            if mutated is None:
                temp *= cooling_rate
                continue

            new_cost = mutated.gate_count

            if not self._quick_verify(mutated):
                temp *= cooling_rate
                continue

            delta = new_cost - current_cost

            if delta < 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                current_state = mutated
                current_cost = new_cost

                self.best_state = mutated
                cleaned = self._try_remove_unused_gates()
                if cleaned:
                    self._try_merge_duplicate_gates()
                    self._try_value_based_merge()

                if not self._quick_verify(self.best_state):
                    self.best_state = current_state
                    current_state = current_state
                    current_cost = best_cost
                    temp *= cooling_rate
                    continue

                actual_cost = self.best_state.gate_count
                current_state = self.best_state
                current_cost = actual_cost

                if actual_cost < best_cost:
                    if best_cost - actual_cost > 50:
                        if not self.verify():
                            self.best_state = current_state
                            current_cost = best_cost
                            temp *= cooling_rate
                            continue

                    best_cost = actual_cost
                    self.improvements += 1

                    if callback:
                        callback(i, best_cost, True)
                elif callback and i % 100 == 0:
                    callback(i, best_cost, False)

            temp *= cooling_rate
            self.iterations += 1

        return self.best_state

    def _mutate_circuit(self, state: "CircuitState") -> "CircuitState | None":
        """Create a mutation by restructuring or merging."""
        import random

        if random.random() < 0.7:
            result = self._mutate_restructure_xor(state)
            if result is not None:
                return result

        gates = list(state.gates)
        outputs = list(state.outputs)

        gate_values = {}
        for i in range(self.input_bits):
            pattern = sum(((j >> i) & 1) << j for j in range(self.num_entries))
            gate_values[i] = pattern

        try:
            for inp in range(self.num_entries):
                node_vals = [(inp >> i) & 1 for i in range(self.input_bits)]

                for g_idx, (op, left, right) in enumerate(gates):
                    if left >= len(node_vals):
                        return None
                    if op not in ("const", "not") and right >= len(node_vals):
                        return None

                    if op == "xor":
                        val = node_vals[left] ^ node_vals[right]
                    elif op == "and":
                        val = node_vals[left] & node_vals[right]
                    elif op == "or":
                        val = node_vals[left] | node_vals[right]
                    elif op == "not":
                        val = node_vals[left] ^ 1
                    elif op == "const":
                        val = left & 1
                    else:
                        val = 0
                    node_vals.append(val)

                    full_idx = self.input_bits + g_idx
                    if full_idx not in gate_values:
                        gate_values[full_idx] = 0
                    gate_values[full_idx] |= val << inp
        except (IndexError, KeyError):
            return None

        value_to_nodes: dict[int, list[int]] = {}
        for idx, val in gate_values.items():
            if val not in value_to_nodes:
                value_to_nodes[val] = []
            value_to_nodes[val].append(idx)

        duplicates = [(v, nodes) for v, nodes in value_to_nodes.items() if len(nodes) > 1]
        if not duplicates:
            return None

        _, equiv_nodes = random.choice(duplicates)
        equiv_nodes = sorted(equiv_nodes)

        keep_idx = equiv_nodes[0]
        remove_idx = random.choice(equiv_nodes[1:])

        if remove_idx < self.input_bits:
            return None

        new_gates = list(gates)
        new_outputs = []

        for i, (op, left, right) in enumerate(new_gates):
            new_left = keep_idx if left == remove_idx else left
            new_right = keep_idx if right == remove_idx else right
            if new_left != left or new_right != right:
                new_gates[i] = (op, new_left, new_right)

        for out_idx, inv in outputs:
            new_idx = keep_idx if out_idx == remove_idx else out_idx
            new_outputs.append((new_idx, inv))

        return CircuitState(
            input_bits=state.input_bits,
            output_bits=state.output_bits,
            gates=new_gates,
            outputs=new_outputs,
            gate_count=len(new_gates),
        )

    def _collect_xor_leaves(self, gates: list, idx: int, depth: int = 0) -> list[int]:
        """Collect all leaf nodes of an XOR tree."""
        if depth > 20:
            return [idx]
        if idx < self.input_bits:
            return [idx]
        gate_idx = idx - self.input_bits
        if gate_idx >= len(gates):
            return [idx]
        op, left, right = gates[gate_idx]
        if op != "xor":
            return [idx]
        return self._collect_xor_leaves(gates, left, depth + 1) + self._collect_xor_leaves(
            gates, right, depth + 1
        )

    def _build_xor_tree(
        self, gates: list, leaves: list[int], start_idx: int
    ) -> tuple[int, list]:
        """Build a balanced XOR tree from leaves, return root index and new gates."""
        if len(leaves) == 0:
            new_idx = start_idx + len(gates)
            return new_idx, gates + [("const", 0, 0)]
        if len(leaves) == 1:
            return leaves[0], gates

        import random

        random.shuffle(leaves)

        new_gates = list(gates)
        while len(leaves) > 1:
            new_leaves = []
            for i in range(0, len(leaves) - 1, 2):
                new_idx = start_idx + len(new_gates)
                new_gates.append(("xor", leaves[i], leaves[i + 1]))
                new_leaves.append(new_idx)
            if len(leaves) % 2 == 1:
                new_leaves.append(leaves[-1])
            leaves = new_leaves

        return leaves[0], new_gates

    def _mutate_restructure_xor(self, state: "CircuitState") -> "CircuitState | None":
        """Restructure an XOR tree by rebuilding with different association."""
        import random

        gates = list(state.gates)
        outputs = list(state.outputs)

        xor_outputs = []
        for i, (out_idx, inv) in enumerate(outputs):
            if out_idx >= self.input_bits:
                gate_idx = out_idx - self.input_bits
                if gate_idx < len(gates) and gates[gate_idx][0] == "xor":
                    xor_outputs.append(i)

        if not xor_outputs:
            return None

        target_out = random.choice(xor_outputs)
        out_idx, inv = outputs[target_out]

        leaves = self._collect_xor_leaves(gates, out_idx)
        if len(leaves) < 3:
            return None

        new_root, new_gates = self._build_xor_tree(gates, leaves, self.input_bits)

        new_outputs = list(outputs)
        new_outputs[target_out] = (new_root, inv)

        return CircuitState(
            input_bits=state.input_bits,
            output_bits=state.output_bits,
            gates=new_gates,
            outputs=new_outputs,
            gate_count=len(new_gates),
        )

    def _quick_verify(self, state: "CircuitState") -> bool:
        """Quick verification on all inputs."""
        try:
            for i in range(self.num_entries):
                if state.evaluate(i) != self.table[i]:
                    return False
            return True
        except (IndexError, KeyError):
            return False

    def _try_remove_unused_gates(self) -> bool:
        """Remove gates that aren't used by any output."""
        used = set()

        def mark_used(idx: int) -> None:
            if idx < self.input_bits:
                return
            gate_idx = idx - self.input_bits
            if gate_idx in used:
                return
            used.add(gate_idx)
            _, left, right = self.best_state.gates[gate_idx]
            mark_used(left)
            mark_used(right)

        for idx, _ in self.best_state.outputs:
            mark_used(idx)

        if len(used) == len(self.best_state.gates):
            return False

        old_to_new = {}
        new_gates = []
        for i in range(self.input_bits):
            old_to_new[i] = i

        for old_idx in sorted(used):
            op, left, right = self.best_state.gates[old_idx]
            new_idx = self.input_bits + len(new_gates)
            old_to_new[self.input_bits + old_idx] = new_idx
            new_gates.append((op, old_to_new[left], old_to_new[right]))

        new_outputs = [(old_to_new[idx], inv) for idx, inv in self.best_state.outputs]

        self.best_state = CircuitState(
            input_bits=self.input_bits,
            output_bits=self.output_bits,
            gates=new_gates,
            outputs=new_outputs,
            gate_count=len(new_gates),
        )
        return True

    def _try_merge_duplicate_gates(self) -> bool:
        """Merge gates that compute the same value."""
        gate_to_first: dict[tuple[str, int, int], int] = {}
        remap: dict[int, int] = {}

        for i in range(self.input_bits):
            remap[i] = i

        changed = False
        for i, (op, left, right) in enumerate(self.best_state.gates):
            left = remap.get(left, left)
            right = remap.get(right, right)

            if op in ("xor", "and", "or") and left > right:
                left, right = right, left

            key = (op, left, right)
            old_idx = self.input_bits + i

            if key in gate_to_first:
                remap[old_idx] = gate_to_first[key]
                changed = True
            else:
                gate_to_first[key] = old_idx
                remap[old_idx] = old_idx

        if not changed:
            return False

        used = set()
        for idx, _ in self.best_state.outputs:
            idx = remap.get(idx, idx)
            if idx >= self.input_bits:
                used.add(idx - self.input_bits)

        def mark_deps(gate_idx: int) -> None:
            if gate_idx in used:
                return
            used.add(gate_idx)
            _, left, right = self.best_state.gates[gate_idx]
            left = remap.get(left, left)
            right = remap.get(right, right)
            if left >= self.input_bits:
                mark_deps(left - self.input_bits)
            if right >= self.input_bits:
                mark_deps(right - self.input_bits)

        for idx, _ in self.best_state.outputs:
            idx = remap.get(idx, idx)
            if idx >= self.input_bits:
                mark_deps(idx - self.input_bits)

        old_to_new = {i: i for i in range(self.input_bits)}
        new_gates = []

        for old_idx in sorted(used):
            op, left, right = self.best_state.gates[old_idx]
            left = remap.get(left, left)
            right = remap.get(right, right)
            left = old_to_new.get(left, left)
            right = old_to_new.get(right, right)

            new_idx = self.input_bits + len(new_gates)
            old_to_new[self.input_bits + old_idx] = new_idx
            new_gates.append((op, left, right))

        new_outputs = []
        for idx, inv in self.best_state.outputs:
            idx = remap.get(idx, idx)
            idx = old_to_new.get(idx, idx)
            new_outputs.append((idx, inv))

        self.best_state = CircuitState(
            input_bits=self.input_bits,
            output_bits=self.output_bits,
            gates=new_gates,
            outputs=new_outputs,
            gate_count=len(new_gates),
        )
        return True

    def _try_simplify_xor_chains(self, timeout_ms: int) -> bool:
        """Try to find algebraic simplifications."""
        improved = False
        new_gates = list(self.best_state.gates)

        for gate_idx in range(len(new_gates)):
            op, left, right = new_gates[gate_idx]

            if op == "xor" and left == right:
                new_gates[gate_idx] = ("const", 0, 1)
                improved = True

            if op == "and" and left == right:
                pass

        if improved:
            self.best_state = CircuitState(
                input_bits=self.input_bits,
                output_bits=self.output_bits,
                gates=new_gates,
                outputs=self.best_state.outputs,
                gate_count=len(new_gates),
            )
            self._try_remove_unused_gates()
            self._try_merge_duplicate_gates()

        return improved

    def _try_value_based_merge(self) -> bool:
        """Merge gates that compute the same function (same truth table)."""
        gate_values: dict[int, int] = {}

        for i in range(self.input_bits):
            pattern = sum(((j >> i) & 1) << j for j in range(self.num_entries))
            gate_values[i] = pattern

        for inp in range(self.num_entries):
            node_vals = [(inp >> i) & 1 for i in range(self.input_bits)]

            for g_idx, (op, left, right) in enumerate(self.best_state.gates):
                if left >= len(node_vals) or (right >= len(node_vals) and op not in ("const", "not")):
                    return False

                if op == "xor":
                    val = node_vals[left] ^ node_vals[right]
                elif op == "and":
                    val = node_vals[left] & node_vals[right]
                elif op == "or":
                    val = node_vals[left] | node_vals[right]
                elif op == "not":
                    val = node_vals[left] ^ 1
                elif op == "const":
                    val = left & 1
                else:
                    val = 0
                node_vals.append(val)

                full_idx = self.input_bits + g_idx
                if full_idx not in gate_values:
                    gate_values[full_idx] = 0
                gate_values[full_idx] |= val << inp

        value_to_first: dict[int, int] = {}
        remap: dict[int, int] = {}

        for i in range(self.input_bits):
            val = gate_values[i]
            value_to_first[val] = i
            remap[i] = i

        changed = False
        for g_idx in range(len(self.best_state.gates)):
            full_idx = self.input_bits + g_idx
            val = gate_values[full_idx]

            if val in value_to_first:
                existing = value_to_first[val]
                if existing != full_idx:
                    remap[full_idx] = existing
                    changed = True
                else:
                    remap[full_idx] = full_idx
            else:
                value_to_first[val] = full_idx
                remap[full_idx] = full_idx

        if not changed:
            return False

        used = set()

        def mark_used(idx: int) -> None:
            idx = remap.get(idx, idx)
            if idx < self.input_bits:
                return
            gate_idx = idx - self.input_bits
            if gate_idx in used:
                return
            used.add(gate_idx)
            _, left, right = self.best_state.gates[gate_idx]
            mark_used(left)
            mark_used(right)

        for idx, _ in self.best_state.outputs:
            mark_used(idx)

        old_to_new = {i: i for i in range(self.input_bits)}
        new_gates = []

        for old_idx in sorted(used):
            op, left, right = self.best_state.gates[old_idx]
            left = remap.get(left, left)
            right = remap.get(right, right)
            left = old_to_new.get(left, left)
            right = old_to_new.get(right, right)

            new_idx = self.input_bits + len(new_gates)
            old_to_new[self.input_bits + old_idx] = new_idx
            new_gates.append((op, left, right))

        new_outputs = []
        for idx, inv in self.best_state.outputs:
            idx = remap.get(idx, idx)
            idx = old_to_new.get(idx, idx)
            new_outputs.append((idx, inv))

        self.best_state = CircuitState(
            input_bits=self.input_bits,
            output_bits=self.output_bits,
            gates=new_gates,
            outputs=new_outputs,
            gate_count=len(new_gates),
        )
        return True

    def _try_and_factoring(self) -> bool:
        """Apply (A AND B) XOR (A AND C) = A AND (B XOR C) when it saves gates."""
        gates = list(self.best_state.gates)

        and_gates = {}
        for idx, (op, left, right) in enumerate(gates):
            if op == "and":
                and_gates[self.input_bits + idx] = (left, right)

        use_count = {}
        for idx, (op, left, right) in enumerate(gates):
            use_count[left] = use_count.get(left, 0) + 1
            if op != "const":
                use_count[right] = use_count.get(right, 0) + 1
        for out_idx, _ in self.best_state.outputs:
            use_count[out_idx] = use_count.get(out_idx, 0) + 1

        for idx, (op, left, right) in enumerate(gates):
            if op != "xor":
                continue
            if left not in and_gates or right not in and_gates:
                continue

            l_a, l_b = and_gates[left]
            r_a, r_b = and_gates[right]

            shared = None
            other_l = other_r = None
            if l_a == r_a:
                shared, other_l, other_r = l_a, l_b, r_b
            elif l_a == r_b:
                shared, other_l, other_r = l_a, l_b, r_a
            elif l_b == r_a:
                shared, other_l, other_r = l_b, l_a, r_b
            elif l_b == r_b:
                shared, other_l, other_r = l_b, l_a, r_a

            if shared is None:
                continue

            left_only_user = use_count.get(left, 0) == 1
            right_only_user = use_count.get(right, 0) == 1

            if not (left_only_user and right_only_user):
                continue

            xor_idx = self.input_bits + len(gates)
            gates.append(("xor", other_l, other_r))

            and_idx = self.input_bits + len(gates)
            gates.append(("and", shared, xor_idx))

            left_gate_idx = left - self.input_bits
            right_gate_idx = right - self.input_bits
            gates[left_gate_idx] = ("const", 0, 0)
            gates[right_gate_idx] = ("const", 0, 0)
            gates[idx] = ("const", 0, 0)

            new_outputs = []
            for out_idx, inv in self.best_state.outputs:
                if out_idx == self.input_bits + idx:
                    new_outputs.append((and_idx, inv))
                else:
                    new_outputs.append((out_idx, inv))

            self.best_state = CircuitState(
                input_bits=self.input_bits,
                output_bits=self.output_bits,
                gates=gates,
                outputs=new_outputs,
                gate_count=len(gates),
            )

            self._try_remove_unused_gates()
            return True

        return False

    def _try_local_resynthesize(self, timeout_ms: int = 5000) -> bool:
        """Try to resynthesize individual output bits with fewer gates."""
        import random

        output_gates = []
        for bit in range(self.output_bits):
            used = set()

            def count_gates(idx: int) -> None:
                if idx < self.input_bits:
                    return
                gate_idx = idx - self.input_bits
                if gate_idx in used:
                    return
                used.add(gate_idx)
                _, left, right = self.best_state.gates[gate_idx]
                count_gates(left)
                count_gates(right)

            out_idx, _ = self.best_state.outputs[bit]
            count_gates(out_idx)
            output_gates.append((len(used), bit))

        output_gates.sort(reverse=True)

        for _, bit in output_gates[:4]:
            out_table = [(self.table[i] >> bit) & 1 for i in range(self.num_entries)]

            used = set()

            def mark_used(idx: int) -> None:
                if idx < self.input_bits:
                    return
                gate_idx = idx - self.input_bits
                if gate_idx in used:
                    return
                used.add(gate_idx)
                _, left, right = self.best_state.gates[gate_idx]
                mark_used(left)
                mark_used(right)

            out_idx, _ = self.best_state.outputs[bit]
            mark_used(out_idx)
            current_gates = len(used)

            if current_gates <= 5:
                continue

            target = current_gates - 1
            result = synthesize_single_output(
                out_table, self.input_bits, max_gates=target, timeout_ms=timeout_ms
            )

            if result is not None:
                new_expr, new_gates = result
                if new_gates < current_gates:
                    self._rebuild_with_new_output(bit, new_expr)
                    return True

        return False

    def _rebuild_with_new_output(self, bit: int, new_expr: Expr) -> None:
        """Rebuild circuit with a new expression for one output bit."""
        from stc.tick_ir import And, Xor, Slice, Var, BitVecConst

        old_gates = list(self.best_state.gates)
        old_outputs = list(self.best_state.outputs)

        new_gates = []
        expr_to_idx: dict[int, int] = {}

        for i in range(self.input_bits):
            pass

        def add_expr(expr: Expr) -> int:
            expr_id = id(expr)
            if expr_id in expr_to_idx:
                return expr_to_idx[expr_id]

            if isinstance(expr, Slice) and isinstance(expr.x, Var):
                idx = expr.offset
                expr_to_idx[expr_id] = idx
                return idx
            elif isinstance(expr, BitVecConst):
                idx = self.input_bits + len(new_gates)
                new_gates.append(("const", expr.value & 1, 1))
                expr_to_idx[expr_id] = idx
                return idx
            elif isinstance(expr, Xor):
                left_idx = add_expr(expr.a)
                right_idx = add_expr(expr.b)
                idx = self.input_bits + len(new_gates)
                new_gates.append(("xor", left_idx, right_idx))
                expr_to_idx[expr_id] = idx
                return idx
            elif isinstance(expr, And):
                left_idx = add_expr(expr.a)
                right_idx = add_expr(expr.b)
                idx = self.input_bits + len(new_gates)
                new_gates.append(("and", left_idx, right_idx))
                expr_to_idx[expr_id] = idx
                return idx
            else:
                raise ValueError(f"Unsupported expression type: {type(expr)}")

        new_out_idx = add_expr(new_expr)

        for b in range(self.output_bits):
            if b == bit:
                continue
            old_idx, inv = old_outputs[b]
            if old_idx < self.input_bits:
                continue

            def copy_gate(gate_idx: int) -> int:
                full_old = self.input_bits + gate_idx
                if full_old in expr_to_idx:
                    return expr_to_idx[full_old]

                op, left, right = old_gates[gate_idx]

                if left >= self.input_bits:
                    left = copy_gate(left - self.input_bits)
                if right >= self.input_bits and op != "const":
                    right = copy_gate(right - self.input_bits)

                new_idx = self.input_bits + len(new_gates)
                new_gates.append((op, left, right))
                expr_to_idx[full_old] = new_idx
                return new_idx

            gate_idx = old_idx - self.input_bits
            copy_gate(gate_idx)

        new_outputs = []
        for b in range(self.output_bits):
            if b == bit:
                new_outputs.append((new_out_idx, False))
            else:
                old_idx, inv = old_outputs[b]
                if old_idx < self.input_bits:
                    new_outputs.append((old_idx, inv))
                else:
                    new_idx = expr_to_idx.get(old_idx, old_idx)
                    new_outputs.append((new_idx, inv))

        self.best_state = CircuitState(
            input_bits=self.input_bits,
            output_bits=self.output_bits,
            gates=new_gates,
            outputs=new_outputs,
            gate_count=len(new_gates),
        )

        self._try_remove_unused_gates()
        self._try_merge_duplicate_gates()

    def save(self, filepath: str) -> None:
        """Save current state to JSON file."""
        import json

        data = {
            "state": self.best_state.to_dict(),
            "iterations": self.iterations,
            "improvements": self.improvements,
            "table": self.table,
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "IncrementalOptimizer":
        """Load optimizer from JSON file."""
        import json

        with open(filepath, "r") as f:
            data = json.load(f)

        state = CircuitState.from_dict(data["state"])
        opt = cls(
            table=data["table"],
            input_bits=state.input_bits,
            output_bits=state.output_bits,
            initial_state=state,
        )
        opt.iterations = data["iterations"]
        opt.improvements = data["improvements"]
        return opt

    def try_sat_optimization(self, timeout_ms: int = 30000) -> bool:
        """Try to find a smaller circuit using SAT solver."""
        current_gates = self.best_state.gate_count

        if current_gates <= 100:
            return False

        target_gates = current_gates - 1
        result = _try_resynthesize(
            self.table,
            self.input_bits,
            self.output_bits,
            target_gates,
            timeout_ms,
        )

        if result is not None:
            self.best_state = result
            return True

        return False


def _try_resynthesize(
    table: Sequence[int],
    input_bits: int,
    output_bits: int,
    max_gates: int,
    timeout_ms: int,
) -> CircuitState | None:
    """Try to synthesize a circuit with at most max_gates gates."""
    result = _synthesize_multi_fixed_size(
        table, input_bits, output_bits, max_gates, timeout_ms
    )

    if result is None:
        return None

    gates = []
    for g in range(result.gate_count):
        op = "xor"
        gates.append((op, 0, 0))

    exprs = result.circuit
    return circuit_to_state(exprs, input_bits, output_bits)
