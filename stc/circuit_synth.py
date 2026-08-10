from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence, Union

import z3

from stc.tick_ir import And, Expr, Not, Or, Slice, TernaryLut, Var, Xor, BitVecConst

Gate = Union[tuple[str, int, int], tuple[str, int, int, int, int]]


@dataclass
class SynthesisResult:
    circuit: list[Expr]
    gate_count: int
    xor_count: int
    and_count: int


def _synth_start() -> tuple[float, float]:
    return time.process_time(), time.time()


def _synth_deadline(start_cpu: float, timeout_ms: int) -> float:
    return start_cpu + timeout_ms / 1000.0


def _synth_wall_for_cpu(ms: int, start_cpu: float, start_wall: float) -> int:
    cpu_elapsed = time.process_time() - start_cpu
    wall_elapsed = time.time() - start_wall
    load_factor = wall_elapsed / max(cpu_elapsed, 1e-6)
    return int(ms * min(max(load_factor, 1.0), 16.0))


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
    start_cpu, start_wall = _synth_start()
    deadline = _synth_deadline(start_cpu, timeout_ms)

    time_per_attempt = max(timeout_ms // (max_gates + 1), 3000)
    for num_gates in range(max_gates + 1):
        if time.process_time() > deadline:
            return None
        result = _synthesize_single_fixed_size(
            table,
            input_bits,
            num_gates,
            _synth_wall_for_cpu(time_per_attempt, start_cpu, start_wall),
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
    start_cpu, start_wall = _synth_start()

    result_at_max = _synthesize_single_fixed_size(
        table,
        input_bits,
        max_gates,
        _synth_wall_for_cpu(timeout_ms_per_attempt, start_cpu, start_wall),
    )
    if result_at_max is None:
        return None

    best = result_at_max
    best_gates = max_gates

    lo, hi = min_gates, max_gates - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        result = _synthesize_single_fixed_size(
            table,
            input_bits,
            mid,
            _synth_wall_for_cpu(timeout_ms_per_attempt, start_cpu, start_wall),
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
    start_cpu, start_wall = _synth_start()
    deadline = _synth_deadline(start_cpu, timeout_ms)

    start_gates = min_gates if min_gates is not None else 0
    range_size = max_gates - start_gates + 1

    base_time = max(timeout_ms // (range_size * 2), 500)

    for num_gates in range(start_gates, max_gates + 1):
        if time.process_time() > deadline:
            return None
        time_for_this = min(base_time * (1 + num_gates // 10), timeout_ms // 4)
        result = _synthesize_multi_fixed_size(
            table,
            input_bits,
            output_bits,
            num_gates,
            _synth_wall_for_cpu(time_for_this, start_cpu, start_wall),
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

    start_cpu, _ = _synth_start()
    deadline = _synth_deadline(start_cpu, timeout_ms)

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
        if time.process_time() > deadline:
            return None
        new_values: dict[int, tuple[Expr, int, int]] = {}
        value_list = list(values.keys())

        for i, v1 in enumerate(value_list):
            if i % 100 == 0 and time.process_time() > deadline:
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

    start_cpu, _ = _synth_start()
    deadline = _synth_deadline(start_cpu, timeout_ms_total)

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
        if time.process_time() > deadline:
            return None
        for _ in range(max_rounds):
            if time.process_time() > deadline:
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
    gates: list[Gate]
    outputs: list[tuple[int, bool]]
    gate_count: int

    @property
    def and_count(self) -> int:
        count = 0
        for gate in self.gates:
            if len(gate) == 3:
                op, _, _ = gate
                if op == "and":
                    count += 1
            elif len(gate) == 5:
                count += 1
        return count

    @property
    def xor_count(self) -> int:
        count = 0
        for gate in self.gates:
            if len(gate) == 3:
                op, _, _ = gate
                if op == "xor":
                    count += 1
            elif len(gate) == 5:
                count += 1
        return count

    @property
    def not_count(self) -> int:
        count = 0
        for gate in self.gates:
            if len(gate) == 3:
                op, _, _ = gate
                if op == "not":
                    count += 1
            elif len(gate) == 5:
                count += 1
        return count

    def weighted_cost(self, and_weight: float = 1.0, xor_weight: float = 1.0) -> float:
        """
        Compute weighted gate cost.

        Different contexts have different gate costs:
        - FHE: AND gates require expensive bootstrapping, XOR is nearly free (and_weight=100+)
        - MPC: AND gates require communication rounds (and_weight=10-100)
        - Hardware area: Similar cost (and_weight=1.5, xor_weight=1.0)
        - Software bitslice: Equal cost (and_weight=1.0, xor_weight=1.0)

        Args:
            and_weight: Cost multiplier for AND gates
            xor_weight: Cost multiplier for XOR gates

        Returns:
            Weighted total cost
        """
        return and_weight * self.and_count + xor_weight * self.xor_count

    def backend_cost(self, technology: "Technology | None" = None) -> float:
        """Compute cost using backend-specific gate weights.

        On PTX/AVX-512, ternary gates cost 1 (native lop3/vpternlog).
        On other backends, ternary gates cost 3 (decomposition to binary gates).

        Args:
            technology: Target technology for cost model. If None, uses worst-case
                        (ternary = 3).

        Returns:
            Total cost with backend-specific weights.
        """
        from stc.tech import Technology

        if technology is None:
            ternary_cost = 3.0
        else:
            prims = {p.name for p in technology.primitives()}
            ternary_cost = 1.0 if ("lop3" in prims or "vpternlog" in prims) else 3.0

        cost = 0.0
        for gate in self.gates:
            if len(gate) == 5:
                cost += ternary_cost
            else:
                op = gate[0]
                if op == "not":
                    cost += 0.0
                elif op == "const":
                    cost += 0.0
                else:
                    cost += 1.0
        return cost

    def backend_depth(self, technology: "Technology | None" = None) -> int:
        """Compute depth using backend-specific latencies.

        Uses technology.depth_model() if available to get per-op latencies.
        On backends without native ternary support, ternary gates add 2 depth
        (decomposition to 2 binary gates in series).

        Args:
            technology: Target technology for depth model. If None, uses default
                        unit depth model.

        Returns:
            Critical path depth with backend-specific latencies.
        """
        from stc.tech import Technology

        if technology is not None:
            depth_model = technology.depth_model()
            prims = {p.name for p in technology.primitives()}
            has_ternary = "lop3" in prims or "vpternlog" in prims
        else:
            depth_model = None
            has_ternary = False

        node_depth: dict[int, int] = {}
        for i in range(self.input_bits):
            node_depth[i] = 0

        for g_idx, gate in enumerate(self.gates):
            full_idx = self.input_bits + g_idx
            if len(gate) == 5:
                _, a, b, c, _ = gate
                a_depth = node_depth.get(a, 0)
                b_depth = node_depth.get(b, 0)
                c_depth = node_depth.get(c, 0)
                base_depth = max(a_depth, b_depth, c_depth)
                if has_ternary:
                    node_depth[full_idx] = base_depth + 1
                else:
                    node_depth[full_idx] = base_depth + 2
            else:
                op, left, right = gate
                if op == "const":
                    node_depth[full_idx] = 0
                elif op == "not":
                    base_depth = node_depth.get(left, 0)
                    if depth_model is not None and depth_model.is_free("not"):
                        node_depth[full_idx] = base_depth
                    else:
                        node_depth[full_idx] = base_depth + 1
                else:
                    left_depth = node_depth.get(left, 0)
                    right_depth = node_depth.get(right, 0)
                    base_depth = max(left_depth, right_depth)
                    if depth_model is not None:
                        node_depth[full_idx] = base_depth + depth_model.op_depth(op)
                    else:
                        node_depth[full_idx] = base_depth + 1

        if not self.outputs:
            return 0
        return max(node_depth.get(idx, 0) for idx, _ in self.outputs)

    @property
    def depth(self) -> int:
        node_depth: dict[int, int] = {}
        for i in range(self.input_bits):
            node_depth[i] = 0

        for g_idx, gate in enumerate(self.gates):
            full_idx = self.input_bits + g_idx
            if len(gate) == 5:
                _, a, b, c, _ = gate
                a_depth = node_depth.get(a, 0)
                b_depth = node_depth.get(b, 0)
                c_depth = node_depth.get(c, 0)
                node_depth[full_idx] = max(a_depth, b_depth, c_depth) + 1
            else:
                op, left, right = gate
                if op == "const":
                    node_depth[full_idx] = 0
                elif op == "not":
                    node_depth[full_idx] = node_depth.get(left, 0) + 1
                else:
                    left_depth = node_depth.get(left, 0)
                    right_depth = node_depth.get(right, 0)
                    node_depth[full_idx] = max(left_depth, right_depth) + 1

        if not self.outputs:
            return 0
        return max(node_depth.get(idx, 0) for idx, _ in self.outputs)

    @property
    def multiplicative_depth(self) -> int:
        node_depth: dict[int, int] = {}
        for i in range(self.input_bits):
            node_depth[i] = 0

        for g_idx, gate in enumerate(self.gates):
            full_idx = self.input_bits + g_idx
            if len(gate) == 5:
                _, a, b, c, _ = gate
                a_depth = node_depth.get(a, 0)
                b_depth = node_depth.get(b, 0)
                c_depth = node_depth.get(c, 0)
                node_depth[full_idx] = max(a_depth, b_depth, c_depth) + 1
            else:
                op, left, right = gate
                if op == "const":
                    node_depth[full_idx] = 0
                elif op == "not":
                    node_depth[full_idx] = node_depth.get(left, 0)
                elif op == "and":
                    left_depth = node_depth.get(left, 0)
                    right_depth = node_depth.get(right, 0)
                    node_depth[full_idx] = max(left_depth, right_depth) + 1
                else:
                    left_depth = node_depth.get(left, 0)
                    right_depth = node_depth.get(right, 0)
                    node_depth[full_idx] = max(left_depth, right_depth)

        if not self.outputs:
            return 0
        return max(node_depth.get(idx, 0) for idx, _ in self.outputs)

    def critical_path_nodes(self) -> set[int]:
        """Find all nodes that lie on a critical (maximum depth) path.

        Returns:
            Set of node indices (including inputs) on critical paths.
        """
        node_depth: dict[int, int] = {}
        for i in range(self.input_bits):
            node_depth[i] = 0

        for g_idx, gate in enumerate(self.gates):
            full_idx = self.input_bits + g_idx
            if len(gate) == 5:
                _, a, b, c, _ = gate
                a_depth = node_depth.get(a, 0)
                b_depth = node_depth.get(b, 0)
                c_depth = node_depth.get(c, 0)
                node_depth[full_idx] = max(a_depth, b_depth, c_depth) + 1
            else:
                op, left, right = gate
                if op == "const":
                    node_depth[full_idx] = 0
                elif op == "not":
                    node_depth[full_idx] = node_depth.get(left, 0) + 1
                else:
                    left_depth = node_depth.get(left, 0)
                    right_depth = node_depth.get(right, 0)
                    node_depth[full_idx] = max(left_depth, right_depth) + 1

        if not self.outputs:
            return set()

        max_depth = max(node_depth.get(idx, 0) for idx, _ in self.outputs)
        critical_nodes: set[int] = set()

        def trace_critical(idx: int, target_depth: int) -> None:
            if idx in critical_nodes:
                return
            if node_depth.get(idx, 0) != target_depth:
                return
            critical_nodes.add(idx)

            if idx < self.input_bits:
                return

            g_idx = idx - self.input_bits
            if g_idx < 0 or g_idx >= len(self.gates):
                return

            gate = self.gates[g_idx]
            if len(gate) == 5:
                _, a, b, c, _ = gate
                for child in [a, b, c]:
                    if node_depth.get(child, 0) == target_depth - 1:
                        trace_critical(child, target_depth - 1)
            else:
                op, left, right = gate
                if op == "const":
                    pass
                elif op == "not":
                    if node_depth.get(left, 0) == target_depth - 1:
                        trace_critical(left, target_depth - 1)
                else:
                    if node_depth.get(left, 0) == target_depth - 1:
                        trace_critical(left, target_depth - 1)
                    if node_depth.get(right, 0) == target_depth - 1:
                        trace_critical(right, target_depth - 1)

        for out_idx, _ in self.outputs:
            if node_depth.get(out_idx, 0) == max_depth:
                trace_critical(out_idx, max_depth)

        return critical_nodes

    def node_depths(self) -> dict[int, int]:
        """Compute depth of each node in the circuit.

        Returns:
            Dict mapping node index to its depth.
        """
        node_depth: dict[int, int] = {}
        for i in range(self.input_bits):
            node_depth[i] = 0

        for g_idx, gate in enumerate(self.gates):
            full_idx = self.input_bits + g_idx
            if len(gate) == 5:
                _, a, b, c, _ = gate
                a_depth = node_depth.get(a, 0)
                b_depth = node_depth.get(b, 0)
                c_depth = node_depth.get(c, 0)
                node_depth[full_idx] = max(a_depth, b_depth, c_depth) + 1
            else:
                op, left, right = gate
                if op == "const":
                    node_depth[full_idx] = 0
                elif op == "not":
                    node_depth[full_idx] = node_depth.get(left, 0) + 1
                else:
                    left_depth = node_depth.get(left, 0)
                    right_depth = node_depth.get(right, 0)
                    node_depth[full_idx] = max(left_depth, right_depth) + 1

        return node_depth

    def optimize_for_depth(self, max_depth_increase: int = 0) -> "CircuitState":
        """Optimize circuit with depth constraints.

        Applies optimizations that may reduce gate count while respecting
        a maximum allowed depth increase.

        Args:
            max_depth_increase: Maximum allowed increase in circuit depth (0 = no increase)

        Returns:
            Optimized CircuitState that respects depth constraint.
        """
        original_depth = self.depth
        max_allowed_depth = original_depth + max_depth_increase

        current = self

        candidate = current.eliminate_dead_code()
        if candidate.depth <= max_allowed_depth:
            current = candidate

        candidate = current.eliminate_common_subexpressions()
        if candidate.depth <= max_allowed_depth:
            current = candidate

        candidate = current.apply_algebraic_rewrites()
        if candidate.depth <= max_allowed_depth:
            current = candidate

        candidate = current.flatten_xor_trees()
        if candidate.depth <= max_allowed_depth:
            current = candidate

        return current

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

    @staticmethod
    def remap_gates(gates: list[Gate], mapping: dict[int, int]) -> list[Gate]:
        """Remap node indices in a list of gates using the provided mapping.

        Args:
            gates: List of gates to remap
            mapping: Dict mapping old indices to new indices

        Returns:
            New list of gates with remapped indices
        """
        result: list[Gate] = []
        for g in gates:
            if len(g) == 5:
                op, a, b, c, imm8 = g
                result.append(
                    (op, mapping.get(a, a), mapping.get(b, b), mapping.get(c, c), imm8)
                )
            else:
                op, left, right = g
                new_left = mapping.get(left, left)
                new_right = (
                    mapping.get(right, right) if op not in ("const", "not") else right
                )
                result.append((op, new_left, new_right))
        return result

    def evaluate(self, x: int) -> int:
        """Evaluate circuit on input x, return output value."""
        node_vals = [(x >> i) & 1 for i in range(self.input_bits)]

        for gate in self.gates:
            if len(gate) == 5:
                _, a, b, c, imm8 = gate
                if a >= len(node_vals):
                    raise IndexError(
                        f"Invalid a index {a}, only {len(node_vals)} nodes"
                    )
                if b >= len(node_vals):
                    raise IndexError(
                        f"Invalid b index {b}, only {len(node_vals)} nodes"
                    )
                if c >= len(node_vals):
                    raise IndexError(
                        f"Invalid c index {c}, only {len(node_vals)} nodes"
                    )
                va = node_vals[a]
                vb = node_vals[b]
                vc = node_vals[c]
                idx = (va << 2) | (vb << 1) | vc
                node_vals.append((imm8 >> idx) & 1)
            else:
                op, left, right = gate
                if left >= len(node_vals):
                    raise IndexError(
                        f"Invalid left index {left}, only {len(node_vals)} nodes"
                    )
                if op not in ("const", "not") and right >= len(node_vals):
                    raise IndexError(
                        f"Invalid right index {right}, only {len(node_vals)} nodes"
                    )

                if op == "xor":
                    node_vals.append(node_vals[left] ^ node_vals[right])
                elif op == "and":
                    node_vals.append(node_vals[left] & node_vals[right])
                elif op == "or":
                    node_vals.append(node_vals[left] | node_vals[right])
                elif op == "andn":
                    node_vals.append((~node_vals[left]) & node_vals[right] & 1)
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

        for gate in self.gates:
            if len(gate) == 5:
                raise NotImplementedError(
                    "Conversion of ternary gates to expressions not yet supported"
                )
            else:
                op, left, right = gate
                if op == "xor":
                    expr = Xor(a=node_exprs[left], b=node_exprs[right])
                elif op == "and":
                    expr = And(a=node_exprs[left], b=node_exprs[right])
                elif op == "not":
                    expr = Not(x=node_exprs[left])
                elif op == "const":
                    expr = BitVecConst(width=right, value=left)
                else:
                    expr = Not(
                        x=And(a=Not(x=node_exprs[left]), b=Not(x=node_exprs[right]))
                    )
                node_exprs.append(expr)

        output_exprs = []
        for idx, invert in self.outputs:
            expr = node_exprs[idx]
            if invert:
                expr = Not(x=expr)
            output_exprs.append(expr)

        return output_exprs

    def eliminate_dead_code(self) -> "CircuitState":
        """Remove gates not in any output cone."""
        used: set[int] = set()

        def mark_used(idx: int) -> None:
            if idx < self.input_bits:
                return
            gate_idx = idx - self.input_bits
            if gate_idx < 0 or gate_idx >= len(self.gates):
                return
            if gate_idx in used:
                return
            used.add(gate_idx)
            gate = self.gates[gate_idx]
            if len(gate) == 5:
                _, a, b, c, _ = gate
                mark_used(a)
                mark_used(b)
                mark_used(c)
            else:
                op, left, right = gate
                mark_used(left)
                if op not in ("const", "not"):
                    mark_used(right)

        for idx, _ in self.outputs:
            mark_used(idx)

        if len(used) == len(self.gates):
            return CircuitState(
                input_bits=self.input_bits,
                output_bits=self.output_bits,
                gates=list(self.gates),
                outputs=list(self.outputs),
                gate_count=self.gate_count,
            )

        old_to_new: dict[int, int] = {}
        for i in range(self.input_bits):
            old_to_new[i] = i

        new_gates: list[Gate] = []
        for old_idx in sorted(used):
            gate = self.gates[old_idx]
            new_idx = self.input_bits + len(new_gates)
            old_to_new[self.input_bits + old_idx] = new_idx
            if len(gate) == 5:
                _, a, b, c, imm8 = gate
                new_a = old_to_new.get(a, a)
                new_b = old_to_new.get(b, b)
                new_c = old_to_new.get(c, c)
                new_gates.append(("ternary", new_a, new_b, new_c, imm8))
            else:
                op, left, right = gate
                new_left = old_to_new.get(left, left)
                new_right = (
                    old_to_new.get(right, right)
                    if op not in ("const", "not")
                    else right
                )
                new_gates.append((op, new_left, new_right))

        new_outputs = [(old_to_new[idx], inv) for idx, inv in self.outputs]

        return CircuitState(
            input_bits=self.input_bits,
            output_bits=self.output_bits,
            gates=new_gates,
            outputs=new_outputs,
            gate_count=len(new_gates),
        )

    def eliminate_common_subexpressions(self) -> "CircuitState":
        seen: dict[tuple, int] = {}
        remap: dict[int, int] = {}

        for i in range(self.input_bits):
            remap[i] = i

        for g_idx, gate in enumerate(self.gates):
            full_idx = self.input_bits + g_idx

            if len(gate) == 5:
                _, a, b, c, imm8 = gate
                new_a = remap.get(a, a)
                new_b = remap.get(b, b)
                new_c = remap.get(c, c)
                key = ("ternary", new_a, new_b, new_c, imm8)
            else:
                op, left, right = gate
                new_left = remap.get(left, left)
                new_right = remap.get(right, right)

                if op in ("xor", "and", "or") and new_left > new_right:
                    new_left, new_right = new_right, new_left

                key = (op, new_left, new_right)

            if key in seen:
                remap[full_idx] = seen[key]
            else:
                seen[key] = full_idx
                remap[full_idx] = full_idx

        used: set[int] = set()

        def mark_used(idx: int) -> None:
            if idx < self.input_bits:
                return
            gate_idx = idx - self.input_bits
            if gate_idx < 0 or gate_idx >= len(self.gates):
                return
            if gate_idx in used:
                return
            used.add(gate_idx)
            gate = self.gates[gate_idx]
            if len(gate) == 5:
                _, a, b, c, _ = gate
                mark_used(remap.get(a, a))
                mark_used(remap.get(b, b))
                mark_used(remap.get(c, c))
            else:
                op, left, right = gate
                mark_used(remap.get(left, left))
                if op not in ("const", "not"):
                    mark_used(remap.get(right, right))

        for out_idx, _ in self.outputs:
            mark_used(remap.get(out_idx, out_idx))

        old_to_new: dict[int, int] = {i: i for i in range(self.input_bits)}
        new_gates: list[Gate] = []

        for old_gate_idx in sorted(used):
            full_old_idx = self.input_bits + old_gate_idx

            if remap.get(full_old_idx, full_old_idx) != full_old_idx:
                continue

            gate = self.gates[old_gate_idx]
            new_idx = self.input_bits + len(new_gates)
            old_to_new[full_old_idx] = new_idx

            if len(gate) == 5:
                _, a, b, c, imm8 = gate
                new_a = old_to_new.get(remap.get(a, a), remap.get(a, a))
                new_b = old_to_new.get(remap.get(b, b), remap.get(b, b))
                new_c = old_to_new.get(remap.get(c, c), remap.get(c, c))
                new_gates.append(("ternary", new_a, new_b, new_c, imm8))
            else:
                op, left, right = gate
                new_left = old_to_new.get(remap.get(left, left), remap.get(left, left))
                new_right = old_to_new.get(
                    remap.get(right, right), remap.get(right, right)
                )
                new_gates.append((op, new_left, new_right))

        for old_gate_idx in sorted(used):
            full_old_idx = self.input_bits + old_gate_idx
            canonical = remap.get(full_old_idx, full_old_idx)
            if canonical != full_old_idx and canonical in old_to_new:
                old_to_new[full_old_idx] = old_to_new[canonical]

        new_outputs: list[tuple[int, bool]] = []
        for out_idx, inv in self.outputs:
            canonical = remap.get(out_idx, out_idx)
            new_idx = old_to_new.get(canonical, canonical)
            new_outputs.append((new_idx, inv))

        return CircuitState(
            input_bits=self.input_bits,
            output_bits=self.output_bits,
            gates=new_gates,
            outputs=new_outputs,
            gate_count=len(new_gates),
        )

    def apply_algebraic_rewrites(self) -> "CircuitState":
        """
        Apply algebraic identity rewrites until fixed point.

        XOR identities:
        - x ^ x -> 0 (constant)
        - x ^ 0 -> x (propagate)
        - (a ^ b) ^ b -> a (cancel)
        - (a ^ b) ^ a -> b (cancel)

        AND identities:
        - x & x -> x
        - x & 0 -> 0
        - x & 1 -> x

        Returns a new CircuitState with simplifications applied.
        """
        gates = list(self.gates)
        outputs = list(self.outputs)

        def find_const_gate(value: int) -> int | None:
            for g_idx, gate in enumerate(gates):
                if len(gate) == 3:
                    op, left, right = gate
                    if op == "const" and (left & 1) == value:
                        return self.input_bits + g_idx
            return None

        def ensure_const_gate(value: int) -> int:
            existing = find_const_gate(value)
            if existing is not None:
                return existing
            new_idx = self.input_bits + len(gates)
            gates.append(("const", value, 1))
            return new_idx

        def get_gate_def(idx: int) -> tuple[str, int, int] | None:
            if idx < self.input_bits:
                return None
            gate_idx = idx - self.input_bits
            if gate_idx < 0 or gate_idx >= len(gates):
                return None
            gate = gates[gate_idx]
            if len(gate) == 5:
                return None
            return gate

        remap: dict[int, int] = {}

        def resolve(idx: int) -> int:
            while idx in remap:
                idx = remap[idx]
            return idx

        changed = True
        while changed:
            changed = False

            for g_idx in range(len(gates)):
                gate = gates[g_idx]
                full_idx = self.input_bits + g_idx

                if full_idx in remap:
                    continue

                if len(gate) == 5:
                    _, a, b, c, imm8 = gate
                    a = resolve(a)
                    b = resolve(b)
                    c = resolve(c)
                    gates[g_idx] = ("ternary", a, b, c, imm8)
                    continue

                op, left, right = gate
                left = resolve(left)
                right = resolve(right)
                gates[g_idx] = (op, left, right)

                if op == "xor":
                    if left == right:
                        const_0 = ensure_const_gate(0)
                        remap[full_idx] = const_0
                        changed = True
                        continue

                    left_def = get_gate_def(left)
                    if left_def is not None and left_def[0] == "const":
                        if (left_def[1] & 1) == 0:
                            remap[full_idx] = right
                            changed = True
                            continue

                    right_def = get_gate_def(right)
                    if right_def is not None and right_def[0] == "const":
                        if (right_def[1] & 1) == 0:
                            remap[full_idx] = left
                            changed = True
                            continue

                    left_def = get_gate_def(left)
                    if left_def is not None and left_def[0] == "xor":
                        inner_a = resolve(left_def[1])
                        inner_b = resolve(left_def[2])
                        if inner_b == right:
                            remap[full_idx] = inner_a
                            changed = True
                            continue
                        if inner_a == right:
                            remap[full_idx] = inner_b
                            changed = True
                            continue

                    right_def = get_gate_def(right)
                    if right_def is not None and right_def[0] == "xor":
                        inner_a = resolve(right_def[1])
                        inner_b = resolve(right_def[2])
                        if inner_b == left:
                            remap[full_idx] = inner_a
                            changed = True
                            continue
                        if inner_a == left:
                            remap[full_idx] = inner_b
                            changed = True
                            continue

                elif op == "and":
                    if left == right:
                        remap[full_idx] = left
                        changed = True
                        continue

                    left_def = get_gate_def(left)
                    if left_def is not None and left_def[0] == "const":
                        if (left_def[1] & 1) == 0:
                            const_0 = ensure_const_gate(0)
                            remap[full_idx] = const_0
                            changed = True
                            continue
                        else:
                            remap[full_idx] = right
                            changed = True
                            continue

                    right_def = get_gate_def(right)
                    if right_def is not None and right_def[0] == "const":
                        if (right_def[1] & 1) == 0:
                            const_0 = ensure_const_gate(0)
                            remap[full_idx] = const_0
                            changed = True
                            continue
                        else:
                            remap[full_idx] = left
                            changed = True
                            continue

        new_outputs = []
        for idx, inv in outputs:
            new_outputs.append((resolve(idx), inv))

        used: set[int] = set()

        def mark_used(idx: int) -> None:
            idx = resolve(idx)
            if idx < self.input_bits:
                return
            gate_idx = idx - self.input_bits
            if gate_idx in used or gate_idx >= len(gates):
                return
            used.add(gate_idx)
            op, left, right = gates[gate_idx]
            mark_used(left)
            if op not in ("const", "not"):
                mark_used(right)

        for idx, _ in new_outputs:
            mark_used(idx)

        old_to_new: dict[int, int] = {}
        for i in range(self.input_bits):
            old_to_new[i] = i

        final_gates: list[tuple[str, int, int]] = []
        for old_idx in sorted(used):
            op, left, right = gates[old_idx]
            left = resolve(left)
            right = resolve(right)
            new_left = old_to_new.get(left, left)
            new_right = (
                old_to_new.get(right, right) if op not in ("const", "not") else right
            )
            new_idx = self.input_bits + len(final_gates)
            old_to_new[self.input_bits + old_idx] = new_idx
            final_gates.append((op, new_left, new_right))

        final_outputs = []
        for idx, inv in new_outputs:
            idx = resolve(idx)
            final_outputs.append((old_to_new.get(idx, idx), inv))

        return CircuitState(
            input_bits=self.input_bits,
            output_bits=self.output_bits,
            gates=final_gates,
            outputs=final_outputs,
            gate_count=len(final_gates),
        )

    def flatten_xor_trees(self) -> "CircuitState":
        """
        Flatten XOR trees into balanced binary trees for lower depth.

        For each XOR gate, collects all leaf inputs through XOR chains.
        Duplicate leaves cancel (x ^ x = 0). Rebuilds XORs as balanced trees.
        Does not flatten through AND gates.
        """
        gate_refs: dict[int, int] = {}
        for gate in self.gates:
            if len(gate) == 5:
                _, a, b, c, _ = gate
                gate_refs[a] = gate_refs.get(a, 0) + 1
                gate_refs[b] = gate_refs.get(b, 0) + 1
                gate_refs[c] = gate_refs.get(c, 0) + 1
            else:
                op, left, right = gate
                gate_refs[left] = gate_refs.get(left, 0) + 1
                if op not in ("const", "not"):
                    gate_refs[right] = gate_refs.get(right, 0) + 1
        for out_idx, _ in self.outputs:
            gate_refs[out_idx] = gate_refs.get(out_idx, 0) + 1

        def get_gate_def(idx: int) -> tuple[str, int, int] | None:
            if idx < self.input_bits:
                return None
            gate_idx = idx - self.input_bits
            if gate_idx < 0 or gate_idx >= len(self.gates):
                return None
            gate = self.gates[gate_idx]
            if len(gate) == 5:
                return None
            return gate

        def is_xor_only_used_by(idx: int, parent_idx: int) -> bool:
            g = get_gate_def(idx)
            if g is None or g[0] != "xor":
                return False
            return gate_refs.get(idx, 0) == 1

        def collect_xor_leaves(idx: int, parent_idx: int | None = None) -> list[int]:
            g = get_gate_def(idx)
            if g is None:
                return [idx]
            op, left, right = g
            if op != "xor":
                return [idx]
            if parent_idx is not None and not is_xor_only_used_by(idx, parent_idx):
                return [idx]
            left_leaves = collect_xor_leaves(left, idx)
            right_leaves = collect_xor_leaves(right, idx)
            return left_leaves + right_leaves

        def simplify_leaves(leaves: list[int]) -> list[int]:
            counts: dict[int, int] = {}
            for leaf in leaves:
                counts[leaf] = counts.get(leaf, 0) + 1
            return [leaf for leaf, count in sorted(counts.items()) if count % 2 == 1]

        new_gates: list[Gate] = []
        old_to_new: dict[int, int] = {i: i for i in range(self.input_bits)}
        const_zero_idx: int | None = None

        def ensure_const_zero() -> int:
            nonlocal const_zero_idx
            if const_zero_idx is not None:
                return const_zero_idx
            const_zero_idx = self.input_bits + len(new_gates)
            new_gates.append(("const", 0, 1))
            return const_zero_idx

        def build_balanced_xor(leaves: list[int]) -> int:
            if len(leaves) == 0:
                return ensure_const_zero()
            if len(leaves) == 1:
                return leaves[0]
            mid = len(leaves) // 2
            left_result = build_balanced_xor(leaves[:mid])
            right_result = build_balanced_xor(leaves[mid:])
            new_idx = self.input_bits + len(new_gates)
            new_gates.append(("xor", left_result, right_result))
            return new_idx

        processed_as_xor_tree: set[int] = set()

        def process_gate(old_full_idx: int) -> int:
            if old_full_idx in old_to_new:
                return old_to_new[old_full_idx]

            if old_full_idx in processed_as_xor_tree:
                return old_to_new[old_full_idx]

            g = get_gate_def(old_full_idx)
            if g is None:
                old_to_new[old_full_idx] = old_full_idx
                return old_full_idx

            op, left, right = g

            if op == "xor":
                old_leaves = collect_xor_leaves(old_full_idx, None)
                for leaf_idx in old_leaves:
                    if leaf_idx >= self.input_bits:
                        process_gate(leaf_idx)

                new_leaves = [old_to_new.get(l, l) for l in old_leaves]
                simplified = simplify_leaves(new_leaves)
                result = build_balanced_xor(simplified)
                old_to_new[old_full_idx] = result

                for leaf_idx in old_leaves:
                    if leaf_idx >= self.input_bits and leaf_idx != old_full_idx:
                        lg = get_gate_def(leaf_idx)
                        if lg is not None and lg[0] == "xor":
                            processed_as_xor_tree.add(leaf_idx)
                            if leaf_idx not in old_to_new:
                                old_to_new[leaf_idx] = result

                return result

            if op == "const":
                new_idx = self.input_bits + len(new_gates)
                new_gates.append(("const", left, right))
                old_to_new[old_full_idx] = new_idx
                return new_idx

            if op == "not":
                new_left = process_gate(left)
                new_idx = self.input_bits + len(new_gates)
                new_gates.append(("not", new_left, 0))
                old_to_new[old_full_idx] = new_idx
                return new_idx

            new_left = process_gate(left)
            new_right = process_gate(right)
            new_idx = self.input_bits + len(new_gates)
            new_gates.append((op, new_left, new_right))
            old_to_new[old_full_idx] = new_idx
            return new_idx

        for out_idx, _ in self.outputs:
            process_gate(out_idx)

        new_outputs = [(old_to_new.get(idx, idx), inv) for idx, inv in self.outputs]

        result = CircuitState(
            input_bits=self.input_bits,
            output_bits=self.output_bits,
            gates=new_gates,
            outputs=new_outputs,
            gate_count=len(new_gates),
        )
        return result.eliminate_dead_code()

    def to_slp(self) -> str:
        """
        Export circuit in standard Straight-Line Program (SLP) format.

        Format:
            t0 = x0 ^ x1
            t1 = x2 & t0
            ...
            y0 = t42
            y1 = ~t50

        Where:
            - x0, x1, ... are inputs
            - t0, t1, ... are intermediate gates
            - y0, y1, ... are outputs (with ~ prefix if inverted)
            - ^ is XOR, & is AND, | is OR, ~ is NOT
        """
        lines = []

        def node_name(idx: int) -> str:
            if idx < self.input_bits:
                return f"x{idx}"
            else:
                return f"t{idx - self.input_bits}"

        for g_idx, gate in enumerate(self.gates):
            gate_name = f"t{g_idx}"
            if len(gate) == 5:
                _, a, b, c, imm8 = gate
                lines.append(
                    f"{gate_name} = ternary({node_name(a)}, {node_name(b)}, "
                    f"{node_name(c)}, 0x{imm8:02x})"
                )
            else:
                op, left, right = gate
                if op == "xor":
                    lines.append(
                        f"{gate_name} = {node_name(left)} ^ {node_name(right)}"
                    )
                elif op == "and":
                    lines.append(
                        f"{gate_name} = {node_name(left)} & {node_name(right)}"
                    )
                elif op == "or":
                    lines.append(
                        f"{gate_name} = {node_name(left)} | {node_name(right)}"
                    )
                elif op == "not":
                    lines.append(f"{gate_name} = ~{node_name(left)}")
                elif op == "const":
                    lines.append(f"{gate_name} = {left & 1}")

        for out_idx, (idx, invert) in enumerate(self.outputs):
            output_name = f"y{out_idx}"
            ref = node_name(idx)
            if invert:
                lines.append(f"{output_name} = ~{ref}")
            else:
                lines.append(f"{output_name} = {ref}")

        return "\n".join(lines)

    def optimize_linear_layers(self) -> "CircuitState":
        """Replace XOR regions with BP-optimized versions.

        Optimizes the pure linear layers at the top (before first AND) and
        bottom (outputs) of the circuit. The interleaved middle section
        with mixed AND/XOR gates is preserved unchanged.

        Returns:
            A new CircuitState with optimized linear regions.
        """
        from stc.linear_opt import LinearCone

        and_gate_indices: list[int] = []
        for g_idx, gate in enumerate(self.gates):
            if len(gate) == 5:
                and_gate_indices.append(self.input_bits + g_idx)
            elif gate[0] == "and":
                and_gate_indices.append(self.input_bits + g_idx)

        if not and_gate_indices:
            all_xors = [
                self.input_bits + g_idx
                for g_idx, gate in enumerate(self.gates)
                if len(gate) == 3 and gate[0] == "xor"
            ]
            if not all_xors:
                return CircuitState(
                    input_bits=self.input_bits,
                    output_bits=self.output_bits,
                    gates=list(self.gates),
                    outputs=list(self.outputs),
                    gate_count=self.gate_count,
                )

            output_xors = [
                idx
                for idx, _ in self.outputs
                if idx >= self.input_bits
                and len(self.gates[idx - self.input_bits]) == 3
                and self.gates[idx - self.input_bits][0] == "xor"
            ]
            if not output_xors:
                return CircuitState(
                    input_bits=self.input_bits,
                    output_bits=self.output_bits,
                    gates=list(self.gates),
                    outputs=list(self.outputs),
                    gate_count=self.gate_count,
                )

            cone = LinearCone.from_circuit(self, output_xors, set())
            optimized_gates = cone.to_xor_circuit_optimized()
            optimized_output_signals = cone.get_output_signals_optimized()

            new_gates: list[tuple[str, int, int]] = []
            old_to_new: dict[int, int] = {i: i for i in range(self.input_bits)}

            opt_gate_to_new: dict[int, int] = {}
            for local_idx, (op, left, right) in enumerate(optimized_gates):
                if left < len(cone.inputs):
                    new_left = cone.inputs[left]
                else:
                    new_left = opt_gate_to_new[left]
                if right < len(cone.inputs):
                    new_right = cone.inputs[right]
                else:
                    new_right = opt_gate_to_new[right]
                new_idx = self.input_bits + len(new_gates)
                new_gates.append((op, new_left, new_right))
                opt_gate_to_new[len(cone.inputs) + local_idx] = new_idx

            for i, out_idx in enumerate(output_xors):
                local_signal = optimized_output_signals[i]
                if local_signal == -1:
                    row = cone.matrix[i]
                    if row == 0:
                        pass
                    elif bin(row).count("1") == 1:
                        bit_pos = row.bit_length() - 1
                        old_to_new[out_idx] = cone.inputs[bit_pos]
                elif local_signal < len(cone.inputs):
                    old_to_new[out_idx] = cone.inputs[local_signal]
                else:
                    old_to_new[out_idx] = opt_gate_to_new.get(
                        local_signal, local_signal
                    )

            new_outputs: list[tuple[int, bool]] = []
            for out_idx, inv in self.outputs:
                new_idx = old_to_new.get(out_idx, out_idx)
                new_outputs.append((new_idx, inv))

            result = CircuitState(
                input_bits=self.input_bits,
                output_bits=self.output_bits,
                gates=new_gates,
                outputs=new_outputs,
                gate_count=len(new_gates),
            )
            return result.eliminate_dead_code()

        stop_at = set(and_gate_indices)

        and_inputs: list[int] = []
        for and_idx in and_gate_indices:
            gate_idx = and_idx - self.input_bits
            gate = self.gates[gate_idx]
            if len(gate) == 5:
                _, a, b, c, _ = gate
                if a not in stop_at and a >= self.input_bits:
                    and_inputs.append(a)
                if b not in stop_at and b >= self.input_bits:
                    and_inputs.append(b)
                if c not in stop_at and c >= self.input_bits:
                    and_inputs.append(c)
            else:
                _, left, right = gate
                if left not in stop_at and left >= self.input_bits:
                    and_inputs.append(left)
                if right not in stop_at and right >= self.input_bits:
                    and_inputs.append(right)

        and_input_xors = [
            idx
            for idx in and_inputs
            if len(self.gates[idx - self.input_bits]) == 3
            and self.gates[idx - self.input_bits][0] == "xor"
        ]

        output_indices = [idx for idx, _ in self.outputs]
        post_and_xors = [
            idx
            for idx in output_indices
            if idx >= self.input_bits
            and idx not in stop_at
            and len(self.gates[idx - self.input_bits]) == 3
            and self.gates[idx - self.input_bits][0] == "xor"
        ]

        new_gates: list[Gate] = []
        old_to_new: dict[int, int] = {i: i for i in range(self.input_bits)}

        for g_idx, gate in enumerate(self.gates):
            full_idx = self.input_bits + g_idx
            if len(gate) == 5:
                _, a, b, c, imm8 = gate
                new_a = old_to_new.get(a, a)
                new_b = old_to_new.get(b, b)
                new_c = old_to_new.get(c, c)
                new_idx = self.input_bits + len(new_gates)
                new_gates.append(("ternary", new_a, new_b, new_c, imm8))
                old_to_new[full_idx] = new_idx
            else:
                op, left, right = gate
                if op in ("const", "not", "or"):
                    if op == "not":
                        new_left = old_to_new.get(left, left)
                        new_idx = self.input_bits + len(new_gates)
                        new_gates.append(("not", new_left, 0))
                        old_to_new[full_idx] = new_idx
                    elif op == "const":
                        new_idx = self.input_bits + len(new_gates)
                        new_gates.append(("const", left, right))
                        old_to_new[full_idx] = new_idx
                    elif op == "or":
                        new_left = old_to_new.get(left, left)
                        new_right = old_to_new.get(right, right)
                        new_idx = self.input_bits + len(new_gates)
                        new_gates.append(("or", new_left, new_right))
                        old_to_new[full_idx] = new_idx

        if and_input_xors:
            pre_cone = LinearCone.from_circuit(self, and_input_xors, stop_at)
            upstream_ands = set(pre_cone.inputs) & set(and_gate_indices)

            if upstream_ands:
                return CircuitState(
                    input_bits=self.input_bits,
                    output_bits=self.output_bits,
                    gates=list(self.gates),
                    outputs=list(self.outputs),
                    gate_count=self.gate_count,
                )

            updated_pre_inputs = []
            for inp in pre_cone.inputs:
                updated_pre_inputs.append(old_to_new.get(inp, inp))

            pre_gates = pre_cone.to_xor_circuit_optimized()
            pre_output_signals = pre_cone.get_output_signals_optimized()

            opt_gate_to_new: dict[int, int] = {}
            for local_idx, (op, left, right) in enumerate(pre_gates):
                if left < len(updated_pre_inputs):
                    new_left = updated_pre_inputs[left]
                else:
                    new_left = opt_gate_to_new[left]
                if right < len(updated_pre_inputs):
                    new_right = updated_pre_inputs[right]
                else:
                    new_right = opt_gate_to_new[right]
                new_idx = self.input_bits + len(new_gates)
                new_gates.append((op, new_left, new_right))
                opt_gate_to_new[len(updated_pre_inputs) + local_idx] = new_idx

            for i, out_idx in enumerate(and_input_xors):
                local_signal = pre_output_signals[i]
                if local_signal == -1:
                    row = pre_cone.matrix[i]
                    if row == 0:
                        pass
                    elif bin(row).count("1") == 1:
                        bit_pos = row.bit_length() - 1
                        old_to_new[out_idx] = updated_pre_inputs[bit_pos]
                elif local_signal < len(updated_pre_inputs):
                    old_to_new[out_idx] = updated_pre_inputs[local_signal]
                else:
                    old_to_new[out_idx] = opt_gate_to_new.get(
                        local_signal, local_signal
                    )

        for and_idx in sorted(and_gate_indices):
            gate_idx = and_idx - self.input_bits
            _, left, right = self.gates[gate_idx]
            new_left = old_to_new.get(left, left)
            new_right = old_to_new.get(right, right)
            new_idx = self.input_bits + len(new_gates)
            new_gates.append(("and", new_left, new_right))
            old_to_new[and_idx] = new_idx

        if post_and_xors:
            post_cone = LinearCone.from_circuit(self, post_and_xors, stop_at)

            updated_inputs = []
            for inp in post_cone.inputs:
                updated_inputs.append(old_to_new.get(inp, inp))

            post_gates = post_cone.to_xor_circuit_optimized()
            post_output_signals = post_cone.get_output_signals_optimized()

            opt_gate_to_new_post: dict[int, int] = {}
            for local_idx, (op, left, right) in enumerate(post_gates):
                if left < len(updated_inputs):
                    new_left = updated_inputs[left]
                else:
                    new_left = opt_gate_to_new_post[left]
                if right < len(updated_inputs):
                    new_right = updated_inputs[right]
                else:
                    new_right = opt_gate_to_new_post[right]
                new_idx = self.input_bits + len(new_gates)
                new_gates.append((op, new_left, new_right))
                opt_gate_to_new_post[len(updated_inputs) + local_idx] = new_idx

            for i, out_idx in enumerate(post_and_xors):
                local_signal = post_output_signals[i]
                if local_signal == -1:
                    row = post_cone.matrix[i]
                    if row == 0:
                        pass
                    elif bin(row).count("1") == 1:
                        bit_pos = row.bit_length() - 1
                        old_to_new[out_idx] = updated_inputs[bit_pos]
                elif local_signal < len(updated_inputs):
                    old_to_new[out_idx] = updated_inputs[local_signal]
                else:
                    old_to_new[out_idx] = opt_gate_to_new_post.get(
                        local_signal, local_signal
                    )

        new_outputs: list[tuple[int, bool]] = []
        for out_idx, inv in self.outputs:
            new_idx = old_to_new.get(out_idx, out_idx)
            new_outputs.append((new_idx, inv))

        result = CircuitState(
            input_bits=self.input_bits,
            output_bits=self.output_bits,
            gates=new_gates,
            outputs=new_outputs,
            gate_count=len(new_gates),
        )

        return result.eliminate_dead_code()

    def try_local_rewrites(self, max_window: int = 3) -> "CircuitState":
        """
        Apply window-based local rewriting to reduce gate count.

        For each gate, considers windows of 2 and 3 gates, computes the truth table
        for the window, and enumerates all smaller implementations. If a smaller
        implementation matches the truth table, it replaces the window.

        Args:
            max_window: Maximum window size (2 or 3)

        Returns:
            A new CircuitState with local rewrites applied.
        """
        current = CircuitState(
            input_bits=self.input_bits,
            output_bits=self.output_bits,
            gates=list(self.gates),
            outputs=list(self.outputs),
            gate_count=self.gate_count,
        )

        changed = True
        while changed:
            changed = False
            result = current._try_single_local_rewrite(max_window)
            if result is not None:
                current = result
                changed = True

        return current

    def _try_single_local_rewrite(self, max_window: int) -> "CircuitState | None":
        """Try a single local rewrite pass. Returns new state if improvement found."""
        for gate_idx in range(len(self.gates)):
            for window_size in range(2, max_window + 1):
                result = self._try_rewrite_window(gate_idx, window_size)
                if result is not None:
                    return result
        return None

    def _try_rewrite_window(
        self, root_gate_idx: int, window_size: int
    ) -> "CircuitState | None":
        """
        Try to rewrite a window of gates rooted at root_gate_idx.

        Returns a new CircuitState if a smaller implementation was found, None otherwise.
        """
        root_idx = self.input_bits + root_gate_idx
        if root_gate_idx >= len(self.gates):
            return None

        window_gates = self._collect_window(root_gate_idx, window_size)
        if len(window_gates) < 2:
            return None

        if not self._can_replace_window(root_gate_idx, window_gates):
            return None

        window_inputs = self._get_window_inputs(window_gates)
        if len(window_inputs) > 4:
            return None

        truth_table = self._compute_window_truth_table(
            root_idx, window_gates, window_inputs
        )

        replacement = self._find_smaller_implementation(
            truth_table, window_inputs, len(window_gates)
        )
        if replacement is None:
            return None

        return self._apply_replacement(
            root_gate_idx, window_gates, window_inputs, replacement
        )

    def _can_replace_window(self, root_gate_idx: int, window_gates: list[int]) -> bool:
        """
        Check if the window can be safely replaced.

        Returns True only if no gate outside the window references any non-root gate
        in the window. This ensures we can replace the entire window with a single
        output without breaking any dependencies.
        """
        window_gate_set = set(window_gates)
        window_full = set(self.input_bits + g for g in window_gates)
        root_full = self.input_bits + root_gate_idx

        for gate_idx, gate in enumerate(self.gates):
            if gate_idx in window_gate_set:
                continue

            if len(gate) == 5:
                _, a, b, c, _ = gate
                if a in window_full and a != root_full:
                    return False
                if b in window_full and b != root_full:
                    return False
                if c in window_full and c != root_full:
                    return False
            else:
                op, left, right = gate
                if left in window_full and left != root_full:
                    return False
                if (
                    op not in ("const", "not")
                    and right in window_full
                    and right != root_full
                ):
                    return False

        for out_idx, _ in self.outputs:
            if out_idx in window_full and out_idx != root_full:
                return False

        return True

    def _collect_window(self, root_gate_idx: int, max_size: int) -> list[int]:
        """
        Collect gate indices that form a window rooted at root_gate_idx.

        Returns gate indices (not full indices) in the window.
        """
        if root_gate_idx >= len(self.gates):
            return []

        window = []
        to_visit = [root_gate_idx]
        visited = set()

        while to_visit and len(window) < max_size:
            gate_idx = to_visit.pop(0)
            if gate_idx in visited:
                continue
            if gate_idx >= len(self.gates):
                continue
            visited.add(gate_idx)
            window.append(gate_idx)

            gate = self.gates[gate_idx]
            if len(gate) == 5:
                _, a, b, c, _ = gate
                for idx in [a, b, c]:
                    if idx >= self.input_bits:
                        child_idx = idx - self.input_bits
                        if child_idx not in visited and child_idx < len(self.gates):
                            to_visit.append(child_idx)
            else:
                op, left, right = gate
                if left >= self.input_bits:
                    child_idx = left - self.input_bits
                    if child_idx not in visited and child_idx < len(self.gates):
                        to_visit.append(child_idx)
                if op not in ("const", "not") and right >= self.input_bits:
                    child_idx = right - self.input_bits
                    if child_idx not in visited and child_idx < len(self.gates):
                        to_visit.append(child_idx)

        return window

    def _get_window_inputs(self, window_gates: list[int]) -> list[int]:
        """
        Get the inputs to the window (indices outside the window).

        Returns list of node indices that are inputs to the window.
        """
        window_full = set(self.input_bits + g for g in window_gates)
        inputs = set()

        for gate_idx in window_gates:
            gate = self.gates[gate_idx]
            if len(gate) == 5:
                _, a, b, c, _ = gate
                if a not in window_full:
                    inputs.add(a)
                if b not in window_full:
                    inputs.add(b)
                if c not in window_full:
                    inputs.add(c)
            else:
                op, left, right = gate
                if left not in window_full:
                    inputs.add(left)
                if op not in ("const", "not") and right not in window_full:
                    inputs.add(right)

        return sorted(inputs)

    def _compute_window_truth_table(
        self, root_idx: int, window_gates: list[int], window_inputs: list[int]
    ) -> list[int]:
        """
        Compute the truth table for the window output given window inputs.

        Returns a list of output bits for each combination of window input values.
        """
        num_window_inputs = len(window_inputs)
        table = []

        for val in range(1 << num_window_inputs):
            input_vals = {}
            for i, inp_idx in enumerate(window_inputs):
                input_vals[inp_idx] = (val >> i) & 1

            gate_vals = {}
            for gate_idx in sorted(window_gates):
                gate = self.gates[gate_idx]
                full_idx = self.input_bits + gate_idx

                if len(gate) == 5:
                    _, a, b, c, imm8 = gate
                    a_val = input_vals.get(a) or gate_vals.get(a, 0)
                    b_val = input_vals.get(b) or gate_vals.get(b, 0)
                    c_val = input_vals.get(c) or gate_vals.get(c, 0)
                    idx = (a_val << 2) | (b_val << 1) | c_val
                    gate_vals[full_idx] = (imm8 >> idx) & 1
                else:
                    op, left, right = gate

                    left_val = input_vals.get(left) or gate_vals.get(left, 0)
                    right_val = input_vals.get(right) or gate_vals.get(right, 0)

                    if op == "xor":
                        gate_vals[full_idx] = left_val ^ right_val
                    elif op == "and":
                        gate_vals[full_idx] = left_val & right_val
                    elif op == "or":
                        gate_vals[full_idx] = left_val | right_val
                    elif op == "not":
                        gate_vals[full_idx] = left_val ^ 1
                    elif op == "const":
                        gate_vals[full_idx] = left & 1
                    else:
                        gate_vals[full_idx] = 0

            table.append(gate_vals.get(root_idx, 0))

        return table

    def _find_smaller_implementation(
        self, truth_table: list[int], inputs: list[int], current_gates: int
    ) -> list[tuple[str, int, int]] | None:
        """
        Find a smaller implementation that produces the same truth table.

        Returns list of gates for the replacement, or None if no improvement.
        """
        num_inputs = len(inputs)

        for i, inp in enumerate(inputs):
            if all(truth_table[v] == ((v >> i) & 1) for v in range(len(truth_table))):
                return []
            if all(
                truth_table[v] == (1 - ((v >> i) & 1)) for v in range(len(truth_table))
            ):
                return [("not", i, 0)]

        if current_gates <= 1:
            return None

        for i in range(num_inputs):
            for j in range(i, num_inputs):
                for op in ["xor", "and", "or"]:
                    matches = True
                    for v in range(len(truth_table)):
                        a = (v >> i) & 1
                        b = (v >> j) & 1
                        if op == "xor":
                            expected = a ^ b
                        elif op == "and":
                            expected = a & b
                        else:
                            expected = a | b
                        if truth_table[v] != expected:
                            matches = False
                            break
                    if matches:
                        return [(op, i, j)]

        if current_gates <= 2:
            return None

        for i in range(num_inputs):
            for j in range(i, num_inputs):
                for op1 in ["xor", "and", "or"]:
                    for k in range(num_inputs):
                        for op2 in ["xor", "and", "or"]:
                            matches = True
                            for v in range(len(truth_table)):
                                a = (v >> i) & 1
                                b = (v >> j) & 1
                                if op1 == "xor":
                                    intermediate = a ^ b
                                elif op1 == "and":
                                    intermediate = a & b
                                else:
                                    intermediate = a | b

                                c = (v >> k) & 1

                                if op2 == "xor":
                                    expected = intermediate ^ c
                                elif op2 == "and":
                                    expected = intermediate & c
                                else:
                                    expected = intermediate | c

                                if truth_table[v] != expected:
                                    matches = False
                                    break

                            if matches:
                                return [(op1, i, j), (op2, num_inputs, k)]

        return None

    def _apply_replacement(
        self,
        root_gate_idx: int,
        window_gates: list[int],
        window_inputs: list[int],
        replacement: list[tuple[str, int, int]],
    ) -> "CircuitState":
        """
        Apply a replacement to the circuit, substituting the window with replacement gates.

        The approach is to rebuild the circuit preserving topological order:
        1. Identify the earliest position in window_gates where we can insert replacement
        2. Build new gate list, skipping window gates and inserting replacement at that position
        3. Update all references accordingly
        """
        root_full = self.input_bits + root_gate_idx
        window_gate_set = set(window_gates)

        min_window_gate = min(window_gates)
        earliest_position = min_window_gate

        local_to_full = {i: idx for i, idx in enumerate(window_inputs)}

        old_to_new: dict[int, int] = {i: i for i in range(self.input_bits)}
        new_gates: list[Gate] = []

        replacement_inserted = False
        result_idx = None

        for old_gate_idx, gate in enumerate(self.gates):
            if old_gate_idx == earliest_position and not replacement_inserted:
                for rep_op, rep_left, rep_right in replacement:
                    new_left = local_to_full.get(rep_left, rep_left)
                    new_right = local_to_full.get(rep_right, rep_right)
                    if rep_left >= len(window_inputs):
                        rep_offset = rep_left - len(window_inputs)
                        new_left = (
                            self.input_bits
                            + len(new_gates)
                            - len(replacement)
                            + rep_offset
                            + 1
                        )
                    if rep_right >= len(window_inputs):
                        rep_offset = rep_right - len(window_inputs)
                        new_right = (
                            self.input_bits
                            + len(new_gates)
                            - len(replacement)
                            + rep_offset
                            + 1
                        )

                    new_idx = self.input_bits + len(new_gates)
                    new_gates.append((rep_op, new_left, new_right))
                    local_to_full[len(window_inputs) + len(new_gates) - 1] = new_idx
                    result_idx = new_idx

                replacement_inserted = True

                if not replacement:
                    result_idx = window_inputs[0] if window_inputs else 0

            if old_gate_idx in window_gate_set:
                old_full = self.input_bits + old_gate_idx
                if old_full == root_full:
                    old_to_new[old_full] = result_idx
                continue

            new_idx = self.input_bits + len(new_gates)
            old_to_new[self.input_bits + old_gate_idx] = new_idx

            if len(gate) == 5:
                _, a, b, c, imm8 = gate
                new_a = old_to_new.get(a, a)
                new_b = old_to_new.get(b, b)
                new_c = old_to_new.get(c, c)
                new_gates.append(("ternary", new_a, new_b, new_c, imm8))
            else:
                op, left, right = gate
                new_left = old_to_new.get(left, left)
                new_right = (
                    old_to_new.get(right, right)
                    if op not in ("const", "not")
                    else right
                )
                new_gates.append((op, new_left, new_right))

        if not replacement_inserted:
            for rep_op, rep_left, rep_right in replacement:
                new_left = local_to_full.get(rep_left, rep_left)
                new_right = local_to_full.get(rep_right, rep_right)

                new_idx = self.input_bits + len(new_gates)
                new_gates.append((rep_op, new_left, new_right))
                result_idx = new_idx

            if not replacement:
                result_idx = window_inputs[0] if window_inputs else 0

        old_to_new[root_full] = result_idx

        new_outputs = []
        for out_idx, inv in self.outputs:
            new_out = old_to_new.get(out_idx, out_idx)
            new_outputs.append((new_out, inv))

        return CircuitState(
            input_bits=self.input_bits,
            output_bits=self.output_bits,
            gates=new_gates,
            outputs=new_outputs,
            gate_count=len(new_gates),
        )

    def sat_window_resynthesis(
        self,
        max_window_inputs: int = 6,
        timeout_per_window: int = 5000,
        max_iterations: int = 1000,
        checkpoint_file: str | None = None,
        and_weight: float = 1.0,
        xor_weight: float = 1.0,
    ) -> "CircuitState":
        """Repeatedly try to improve circuit via window resynthesis.

        Loop:
        1. For each gate (round-robin), try to extract a window
        2. If window exists and has <= max_window_inputs:
           a. Try to synthesize with fewer gates than current window
           b. If success, splice the optimized window
        3. Continue until max_iterations or no improvement possible

        Args:
            max_window_inputs: Max inputs for windows (default 6, since 2^6=64 combos)
            timeout_per_window: SAT solver timeout in ms per window
            max_iterations: Max iterations through all gates
            checkpoint_file: If provided, save best circuit periodically
            and_weight: Weight for AND gates in cost function (default 1.0)
            xor_weight: Weight for XOR gates in cost function (default 1.0)

        Returns:
            Optimized CircuitState

        Note: When and_weight > xor_weight, optimizer prioritizes reducing AND gates.
              This is useful for FHE (and_weight=100+) or MPC (and_weight=10+).
        """
        import json

        from stc.window_synth import (
            compute_fanouts,
            extract_window,
            splice_window,
            synthesize_exact,
        )

        current = CircuitState(
            input_bits=self.input_bits,
            output_bits=self.output_bits,
            gates=list(self.gates),
            outputs=list(self.outputs),
            gate_count=self.gate_count,
        )

        best_gate_count = current.gate_count
        improvements_this_round = 0
        total_improvements = 0
        checkpoint_interval = 10

        for iteration in range(max_iterations):
            improved_in_iteration = False

            for gate_idx in range(len(current.gates)):
                root_idx = current.input_bits + gate_idx

                window = extract_window(current, root_idx, max_inputs=max_window_inputs)
                if window is None:
                    continue

                current_window_gates = len(window.internal)
                if current_window_gates <= 1:
                    continue

                target_gates = current_window_gates - 1
                new_gates = synthesize_exact(
                    window.truth_tables,
                    n_inputs=len(window.inputs),
                    max_gates=target_gates,
                    timeout_ms=timeout_per_window,
                    and_weight=and_weight,
                    xor_weight=xor_weight,
                )

                if new_gates is None:
                    continue

                actual_new_gates = len([g for g in new_gates if g[0] != "wire"])
                if actual_new_gates >= current_window_gates:
                    continue

                try:
                    new_state = splice_window(current, window, new_gates)

                    old_cost = current.weighted_cost(and_weight, xor_weight)
                    new_cost = new_state.weighted_cost(and_weight, xor_weight)
                    if new_cost < old_cost:
                        current = new_state
                        improvements_this_round += 1
                        total_improvements += 1
                        improved_in_iteration = True

                        if (
                            checkpoint_file is not None
                            and total_improvements % checkpoint_interval == 0
                        ):
                            with open(checkpoint_file, "w") as f:
                                json.dump(current.to_dict(), f)

                        break
                except (IndexError, KeyError):
                    continue

            if not improved_in_iteration:
                break

            if current.gate_count < best_gate_count:
                best_gate_count = current.gate_count

        if checkpoint_file is not None and total_improvements > 0:
            with open(checkpoint_file, "w") as f:
                json.dump(current.to_dict(), f)

        return current


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

        if isinstance(e, Or):
            left_idx = process_expr(e.a)
            right_idx = process_expr(e.b)
            new_idx = input_bits + len(gates)
            gates.append(("or", left_idx, right_idx))
            node_to_idx[eid] = new_idx
            return new_idx

        if isinstance(e, TernaryLut):
            a_idx = process_expr(e.a)
            b_idx = process_expr(e.b)
            c_idx = process_expr(e.c)
            new_idx = input_bits + len(gates)
            gates.append(("ternary", a_idx, b_idx, c_idx, e.imm8))
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
        seed: int | None = None,
    ):
        self.table = list(table)
        self.input_bits = input_bits
        self.output_bits = output_bits
        self.num_entries = 1 << input_bits
        self.seed = seed

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

        xor_indices = [
            i for i, gate in enumerate(gates) if len(gate) == 3 and gate[0] == "xor"
        ]
        if len(xor_indices) < 2:
            return False

        idx1, idx2 = random.sample(xor_indices, 2)
        op1, l1, r1 = gates[idx1]
        op2, l2, r2 = gates[idx2]

        full1 = self.input_bits + idx1
        full2 = self.input_bits + idx2

        use_count = {}
        for i, gate in enumerate(gates):
            if len(gate) == 5:
                _, a, b, c, _ = gate
                use_count[a] = use_count.get(a, 0) + 1
                use_count[b] = use_count.get(b, 0) + 1
                use_count[c] = use_count.get(c, 0) + 1
            else:
                op, left, right = gate
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

    def _log_transformation(
        self,
        log_file: str | None,
        transformation: str,
        gate_count: int,
        effective_seed: int | None = None,
    ) -> None:
        if log_file is None:
            return
        import json

        entry = {
            "timestamp": time.time(),
            "iteration": self.iterations,
            "transformation": transformation,
            "gate_count": gate_count,
            "and_count": self.best_state.and_count,
            "xor_count": self.best_state.xor_count,
        }
        if effective_seed is not None:
            entry["seed"] = effective_seed
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _maybe_checkpoint(
        self,
        checkpoint_file: str | None,
        checkpoint_interval: int | None,
        iteration: int,
    ) -> None:
        if checkpoint_file is None or checkpoint_interval is None:
            return
        if checkpoint_interval <= 0:
            return
        if iteration % checkpoint_interval == 0:
            self.save(checkpoint_file)

    def optimize(
        self,
        max_iterations: int = 100,
        timeout_ms_per_step: int = 5000,
        target_gates: int | None = None,
        callback: callable | None = None,
        seed: int | None = None,
        log_file: str | None = None,
        checkpoint_interval: int | None = None,
        checkpoint_file: str | None = None,
    ) -> CircuitState:
        """
        Run optimization loop.

        Args:
            max_iterations: Maximum optimization steps
            timeout_ms_per_step: Timeout per optimization attempt
            target_gates: Stop if gate count reaches this target
            callback: Called after each iteration with (iteration, gate_count, improved)
            seed: Random seed for reproducibility (overrides __init__ seed if provided)
            log_file: Path to file for logging accepted transformations (JSON lines)
            checkpoint_interval: Save checkpoint every N iterations
            checkpoint_file: Path to checkpoint file
        """
        import random

        effective_seed = seed if seed is not None else self.seed
        if effective_seed is not None:
            random.seed(effective_seed)

        if log_file is not None:
            self._log_transformation(
                log_file, "optimize_start", self.best_state.gate_count, effective_seed
            )

        for i in range(max_iterations):
            if target_gates and self.best_state.gate_count <= target_gates:
                break

            old_gate_count = self.best_state.gate_count
            improved = self.optimize_step(timeout_ms_per_step)

            if improved:
                self._log_transformation(
                    log_file, "optimize_step", self.best_state.gate_count
                )

            if callback:
                callback(i, self.best_state.gate_count, improved)

            self._maybe_checkpoint(checkpoint_file, checkpoint_interval, i)

        if checkpoint_file:
            self.save(checkpoint_file)

        return self.best_state

    def anneal(
        self,
        max_iterations: int = 1000,
        initial_temp: float = 10.0,
        cooling_rate: float = 0.995,
        timeout_seconds: float = 300.0,
        callback: callable | None = None,
        seed: int | None = None,
        log_file: str | None = None,
        checkpoint_interval: int | None = None,
        checkpoint_file: str | None = None,
    ) -> CircuitState:
        """
        Simulated annealing optimization.

        Randomly mutates circuit and accepts worse solutions with decreasing probability.

        Args:
            max_iterations: Maximum annealing iterations
            initial_temp: Starting temperature
            cooling_rate: Temperature multiplier per iteration (< 1.0)
            timeout_seconds: Maximum wall-clock time
            callback: Called on improvements with (iteration, gate_count, improved)
            seed: Random seed for reproducibility (overrides __init__ seed if provided)
            log_file: Path to file for logging accepted transformations (JSON lines)
            checkpoint_interval: Save checkpoint every N iterations
            checkpoint_file: Path to checkpoint file
        """
        import math
        import random

        effective_seed = seed if seed is not None else self.seed
        if effective_seed is not None:
            random.seed(effective_seed)

        if log_file is not None:
            self._log_transformation(
                log_file, "anneal_start", self.best_state.gate_count, effective_seed
            )

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

                    self._log_transformation(log_file, "anneal_accept", best_cost)

                    if callback:
                        callback(i, best_cost, True)
                elif callback and i % 100 == 0:
                    callback(i, best_cost, False)

            temp *= cooling_rate
            self.iterations += 1

            self._maybe_checkpoint(checkpoint_file, checkpoint_interval, i)

        if checkpoint_file:
            self.save(checkpoint_file)

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

                for g_idx, gate in enumerate(gates):
                    if len(gate) == 5:
                        _, a, b, c, imm8 = gate
                        if a >= len(node_vals):
                            return None
                        if b >= len(node_vals):
                            return None
                        if c >= len(node_vals):
                            return None
                        va = node_vals[a]
                        vb = node_vals[b]
                        vc = node_vals[c]
                        idx = (va << 2) | (vb << 1) | vc
                        val = (imm8 >> idx) & 1
                    else:
                        op, left, right = gate
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

        duplicates = [
            (v, nodes) for v, nodes in value_to_nodes.items() if len(nodes) > 1
        ]
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

        for i, gate in enumerate(new_gates):
            if len(gate) == 5:
                op, a, b, c, imm8 = gate
                new_a = keep_idx if a == remove_idx else a
                new_b = keep_idx if b == remove_idx else b
                new_c = keep_idx if c == remove_idx else c
                if new_a != a or new_b != b or new_c != c:
                    new_gates[i] = (op, new_a, new_b, new_c, imm8)
            else:
                op, left, right = gate
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
        return self._collect_xor_leaves(
            gates, left, depth + 1
        ) + self._collect_xor_leaves(gates, right, depth + 1)

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
            if gate_idx < 0 or gate_idx >= len(self.best_state.gates):
                return
            if gate_idx in used:
                return
            used.add(gate_idx)
            op, left, right = self.best_state.gates[gate_idx]
            mark_used(left)
            if op not in ("const", "not"):
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
        gate_to_first: dict[tuple, int] = {}
        remap: dict[int, int] = {}

        for i in range(self.input_bits):
            remap[i] = i

        changed = False
        for i, gate in enumerate(self.best_state.gates):
            old_idx = self.input_bits + i

            if len(gate) == 5:
                op, a, b, c, imm8 = gate
                new_a = remap.get(a, a)
                new_b = remap.get(b, b)
                new_c = remap.get(c, c)
                key = (op, new_a, new_b, new_c, imm8)
            else:
                op, left, right = gate
                left = remap.get(left, left)
                right = remap.get(right, right)

                if op in ("xor", "and", "or") and left > right:
                    left, right = right, left

                key = (op, left, right)

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
            gate = self.best_state.gates[gate_idx]
            if len(gate) == 5:
                _, a, b, c, _ = gate
                a = remap.get(a, a)
                b = remap.get(b, b)
                c = remap.get(c, c)
                if a >= self.input_bits:
                    mark_deps(a - self.input_bits)
                if b >= self.input_bits:
                    mark_deps(b - self.input_bits)
                if c >= self.input_bits:
                    mark_deps(c - self.input_bits)
            else:
                _, left, right = gate
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
            gate = self.best_state.gates[old_idx]
            new_idx = self.input_bits + len(new_gates)
            old_to_new[self.input_bits + old_idx] = new_idx

            if len(gate) == 5:
                op, a, b, c, imm8 = gate
                a = remap.get(a, a)
                b = remap.get(b, b)
                c = remap.get(c, c)
                a = old_to_new.get(a, a)
                b = old_to_new.get(b, b)
                c = old_to_new.get(c, c)
                new_gates.append((op, a, b, c, imm8))
            else:
                op, left, right = gate
                left = remap.get(left, left)
                right = remap.get(right, right)
                left = old_to_new.get(left, left)
                right = old_to_new.get(right, right)
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
            gate = new_gates[gate_idx]
            if len(gate) == 5:
                continue
            op, left, right = gate

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

            for g_idx, gate in enumerate(self.best_state.gates):
                if len(gate) == 5:
                    _, a, b, c, imm8 = gate
                    if (
                        a >= len(node_vals)
                        or b >= len(node_vals)
                        or c >= len(node_vals)
                    ):
                        return False
                    va = node_vals[a]
                    vb = node_vals[b]
                    vc = node_vals[c]
                    idx = (va << 2) | (vb << 1) | vc
                    val = (imm8 >> idx) & 1
                else:
                    op, left, right = gate
                    if left >= len(node_vals) or (
                        right >= len(node_vals) and op not in ("const", "not")
                    ):
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
            gate = self.best_state.gates[gate_idx]
            if len(gate) == 5:
                _, a, b, c, _ = gate
                mark_used(a)
                mark_used(b)
                mark_used(c)
            else:
                _, left, right = gate
                mark_used(left)
                mark_used(right)

        for idx, _ in self.best_state.outputs:
            mark_used(idx)

        old_to_new = {i: i for i in range(self.input_bits)}
        new_gates = []

        for old_idx in sorted(used):
            gate = self.best_state.gates[old_idx]
            new_idx = self.input_bits + len(new_gates)
            old_to_new[self.input_bits + old_idx] = new_idx

            if len(gate) == 5:
                op, a, b, c, imm8 = gate
                a = remap.get(a, a)
                b = remap.get(b, b)
                c = remap.get(c, c)
                a = old_to_new.get(a, a)
                b = old_to_new.get(b, b)
                c = old_to_new.get(c, c)
                new_gates.append((op, a, b, c, imm8))
            else:
                op, left, right = gate
                left = remap.get(left, left)
                right = remap.get(right, right)
                left = old_to_new.get(left, left)
                right = old_to_new.get(right, right)
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
        for idx, gate in enumerate(gates):
            if len(gate) == 3 and gate[0] == "and":
                _, left, right = gate
                and_gates[self.input_bits + idx] = (left, right)

        use_count = {}
        for idx, gate in enumerate(gates):
            if len(gate) == 5:
                _, a, b, c, _ = gate
                use_count[a] = use_count.get(a, 0) + 1
                use_count[b] = use_count.get(b, 0) + 1
                use_count[c] = use_count.get(c, 0) + 1
            else:
                op, left, right = gate
                use_count[left] = use_count.get(left, 0) + 1
                if op != "const":
                    use_count[right] = use_count.get(right, 0) + 1
        for out_idx, _ in self.best_state.outputs:
            use_count[out_idx] = use_count.get(out_idx, 0) + 1

        for idx, gate in enumerate(gates):
            if len(gate) != 3:
                continue
            op, left, right = gate
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
            "seed": self.seed,
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
            seed=data.get("seed"),
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
