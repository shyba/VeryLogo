from __future__ import annotations

from dataclasses import dataclass

import z3

from stc.circuit_synth import CircuitState


def compute_fanouts(state: CircuitState) -> dict[int, list[int]]:
    """For each node, list of gates that use it as input."""
    fanouts: dict[int, list[int]] = {}
    total_nodes = state.input_bits + len(state.gates)

    for i in range(total_nodes):
        fanouts[i] = []

    for g_idx, (op, left, right) in enumerate(state.gates):
        full_idx = state.input_bits + g_idx
        fanouts[left].append(full_idx)
        if op not in ("const", "not"):
            fanouts[right].append(full_idx)

    for idx, _ in state.outputs:
        if idx not in fanouts:
            fanouts[idx] = []

    return fanouts


def compute_depths(state: CircuitState) -> list[int]:
    """Depth of each node from inputs. Inputs have depth 0."""
    depths: list[int] = []

    for _ in range(state.input_bits):
        depths.append(0)

    for g_idx, (op, left, right) in enumerate(state.gates):
        if op == "const":
            depths.append(0)
        elif op == "not":
            depths.append(depths[left] + 1)
        else:
            left_depth = depths[left]
            right_depth = depths[right]
            depths.append(max(left_depth, right_depth) + 1)

    return depths


def compute_mffc(
    state: CircuitState, node: int, fanouts: dict[int, list[int]]
) -> set[int]:
    """Maximum fanout-free cone rooted at node.

    MFFC = all nodes reachable backward from node that have
    no fanout outside the cone (except through node).
    """
    if node < state.input_bits:
        return set()

    output_nodes: set[int] = {idx for idx, _ in state.outputs}

    mffc: set[int] = {node}
    worklist = [node]
    processed: set[int] = set()

    while worklist:
        current = worklist.pop()
        if current in processed:
            continue
        processed.add(current)

        if current < state.input_bits:
            continue

        gate_idx = current - state.input_bits
        if gate_idx < 0 or gate_idx >= len(state.gates):
            continue

        op, left, right = state.gates[gate_idx]

        inputs_to_check = [left]
        if op not in ("const", "not"):
            inputs_to_check.append(right)

        for inp in inputs_to_check:
            if inp < state.input_bits:
                continue

            if inp in output_nodes and inp != node:
                continue

            all_fanouts_in_mffc = all(
                fo in mffc or fo == node for fo in fanouts.get(inp, [])
            )

            if all_fanouts_in_mffc:
                if inp not in mffc:
                    mffc.add(inp)
                    worklist.append(inp)

    return mffc


@dataclass
class Window:
    inputs: list[int]
    outputs: list[int]
    internal: list[int]
    truth_tables: list[int]


def extract_window(
    state: CircuitState, root: int, max_inputs: int = 6
) -> Window | None:
    """Extract resynthesizable window rooted at given node.

    1. Start with MFFC of root
    2. Expand to include related gates if under input limit
    3. Compute truth table for each output
    4. Return None if too many inputs
    """
    if root < state.input_bits:
        return None

    fanouts = compute_fanouts(state)
    mffc = compute_mffc(state, root, fanouts)

    if not mffc:
        return None

    internal = sorted(mffc)

    window_inputs: set[int] = set()
    for node_idx in internal:
        if node_idx < state.input_bits:
            continue
        gate_idx = node_idx - state.input_bits
        if gate_idx < 0 or gate_idx >= len(state.gates):
            continue

        op, left, right = state.gates[gate_idx]

        if left not in mffc:
            window_inputs.add(left)
        if op not in ("const", "not") and right not in mffc:
            window_inputs.add(right)

    if len(window_inputs) > max_inputs:
        return None

    inputs_list = sorted(window_inputs)

    outputs_list = [root]

    truth_tables = []
    for output in outputs_list:
        tt = compute_truth_table(state, output, inputs_list, mffc)
        truth_tables.append(tt)

    return Window(
        inputs=inputs_list,
        outputs=outputs_list,
        internal=internal,
        truth_tables=truth_tables,
    )


def compute_truth_table(
    state: CircuitState, output: int, inputs: list[int], window_nodes: set[int] = None
) -> int:
    """Compute 2^k bit truth table for output in terms of inputs.

    For k inputs, evaluate all 2^k input combinations and pack into integer.
    """
    k = len(inputs)
    num_combinations = 1 << k

    if window_nodes is None:
        window_nodes = _collect_cone(state, output)

    input_to_bit: dict[int, int] = {inp: i for i, inp in enumerate(inputs)}

    result = 0

    for combo in range(num_combinations):
        node_vals: dict[int, int] = {}

        for inp in inputs:
            bit_idx = input_to_bit[inp]
            node_vals[inp] = (combo >> bit_idx) & 1

        for node_idx in sorted(window_nodes):
            if node_idx in node_vals:
                continue

            if node_idx < state.input_bits:
                if node_idx not in node_vals:
                    node_vals[node_idx] = 0
                continue

            gate_idx = node_idx - state.input_bits
            if gate_idx < 0 or gate_idx >= len(state.gates):
                continue

            op, left, right = state.gates[gate_idx]

            left_val = node_vals.get(left, 0)

            if op == "const":
                node_vals[node_idx] = left & 1
            elif op == "not":
                node_vals[node_idx] = left_val ^ 1
            elif op == "xor":
                right_val = node_vals.get(right, 0)
                node_vals[node_idx] = left_val ^ right_val
            elif op == "and":
                right_val = node_vals.get(right, 0)
                node_vals[node_idx] = left_val & right_val
            elif op == "or":
                right_val = node_vals.get(right, 0)
                node_vals[node_idx] = left_val | right_val
            else:
                right_val = node_vals.get(right, 0)
                node_vals[node_idx] = 1 - ((1 - left_val) & (1 - right_val))

        output_val = node_vals.get(output, 0)
        result |= output_val << combo

    return result


def _collect_cone(state: CircuitState, output: int) -> set[int]:
    """Collect all nodes in the transitive fanin cone of output."""
    cone: set[int] = set()
    worklist = [output]

    while worklist:
        node = worklist.pop()
        if node in cone:
            continue
        cone.add(node)

        if node < state.input_bits:
            continue

        gate_idx = node - state.input_bits
        if gate_idx < 0 or gate_idx >= len(state.gates):
            continue

        op, left, right = state.gates[gate_idx]
        worklist.append(left)
        if op not in ("const", "not"):
            worklist.append(right)

    return cone


def synthesize_exact(
    truth_tables: list[int],
    n_inputs: int,
    max_gates: int,
    gate_types: list[str] | None = None,
    timeout_ms: int = 10000,
) -> list[tuple[str, int, int]] | None:
    """Find minimum circuit implementing truth tables using SAT/Z3.

    Uses iterative deepening: tries g=0, g=1, ... until SAT or g > max_gates.

    Args:
        truth_tables: List of truth tables (one per output), each as a bitvector
                      where bit i is the output when the input is i.
        n_inputs: Number of input bits.
        max_gates: Maximum number of gates to search for.
        gate_types: List of allowed gate types. Defaults to ["xor", "and"].
        timeout_ms: Timeout in milliseconds for each gate count attempt.

    Returns:
        List of gates [(op, left, right), ...] or None if no solution found.
        Indices 0..n_inputs-1 are inputs; indices n_inputs..n_inputs+g-1 are gates.
    """
    if gate_types is None:
        gate_types = ["xor", "and"]

    time_per_attempt = max(timeout_ms // (max_gates + 1), 100)

    for num_gates in range(max_gates + 1):
        result = _try_synth_with_g_gates(
            truth_tables, n_inputs, num_gates, gate_types, time_per_attempt
        )
        if result is not None:
            return result

    return None


def _try_synth_with_g_gates(
    truth_tables: list[int],
    n_inputs: int,
    g: int,
    gate_types: list[str],
    timeout_ms: int,
) -> list[tuple[str, int, int]] | None:
    """Try to synthesize with exactly g gates."""
    num_entries = 1 << n_inputs
    num_outputs = len(truth_tables)

    input_patterns = []
    for bit in range(n_inputs):
        pattern = sum(((j >> bit) & 1) << j for j in range(num_entries))
        input_patterns.append(pattern)

    if g == 0:
        output_indices = []
        for target in truth_tables:
            found_idx = None
            for inp in range(n_inputs):
                if input_patterns[inp] == target:
                    found_idx = inp
                    break
            if found_idx is None:
                return None
            output_indices.append(found_idx)
        if len(output_indices) == 1 and output_indices[0] == 0:
            return []
        return [("wire", i, 0) for i in output_indices]

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    OP_XOR = 0
    OP_AND = 1
    OP_OR = 2

    type_to_op = {"xor": OP_XOR, "and": OP_AND, "or": OP_OR}
    allowed_ops = [type_to_op[t] for t in gate_types if t in type_to_op]

    gate_op = [z3.Int(f"op_{i}") for i in range(g)]
    gate_left = [z3.Int(f"left_{i}") for i in range(g)]
    gate_right = [z3.Int(f"right_{i}") for i in range(g)]
    gate_val = [z3.BitVec(f"val_{i}", num_entries) for i in range(g)]

    for i in range(g):
        if len(allowed_ops) == 1:
            solver.add(gate_op[i] == allowed_ops[0])
        else:
            solver.add(z3.Or([gate_op[i] == op for op in allowed_ops]))

        num_avail = n_inputs + i
        solver.add(gate_left[i] >= 0, gate_left[i] < num_avail)
        solver.add(gate_right[i] >= 0, gate_right[i] < num_avail)
        solver.add(gate_left[i] <= gate_right[i])

    def get_node_val(idx: int, g_limit: int) -> z3.BitVecRef:
        if idx < n_inputs:
            return z3.BitVecVal(input_patterns[idx], num_entries)
        else:
            return gate_val[idx - n_inputs]

    for i in range(g):
        num_avail = n_inputs + i
        left_val = z3.BitVec(f"lv_{i}", num_entries)
        right_val = z3.BitVec(f"rv_{i}", num_entries)

        for j in range(num_avail):
            nv = get_node_val(j, i)
            solver.add(z3.Implies(gate_left[i] == j, left_val == nv))
            solver.add(z3.Implies(gate_right[i] == j, right_val == nv))

        solver.add(
            z3.Implies(gate_op[i] == OP_XOR, gate_val[i] == (left_val ^ right_val))
        )
        solver.add(
            z3.Implies(gate_op[i] == OP_AND, gate_val[i] == (left_val & right_val))
        )
        solver.add(
            z3.Implies(gate_op[i] == OP_OR, gate_val[i] == (left_val | right_val))
        )

    output_sels = [z3.Int(f"out_{o}") for o in range(num_outputs)]
    total_nodes = n_inputs + g

    for out_idx in range(num_outputs):
        solver.add(output_sels[out_idx] >= 0, output_sels[out_idx] < total_nodes)

        output_val = z3.BitVec(f"out_val_{out_idx}", num_entries)
        for j in range(total_nodes):
            nv = get_node_val(j, g)
            solver.add(z3.Implies(output_sels[out_idx] == j, output_val == nv))

        target_bv = z3.BitVecVal(truth_tables[out_idx], num_entries)
        solver.add(output_val == target_bv)

    if solver.check() != z3.sat:
        return None

    model = solver.model()

    def get_int(v: z3.ExprRef) -> int:
        return model.eval(v, model_completion=True).as_long()

    op_to_str = {OP_XOR: "xor", OP_AND: "and", OP_OR: "or"}

    result_gates = []
    for i in range(g):
        op = get_int(gate_op[i])
        left = get_int(gate_left[i])
        right = get_int(gate_right[i])
        result_gates.append((op_to_str[op], left, right))

    return result_gates


def splice_window(
    state: CircuitState,
    window: Window,
    new_gates: list[tuple[str, int, int]],
) -> CircuitState:
    """Replace window's internal gates with new implementation.

    1. Add replacement gates for the window
    2. Map window outputs to new outputs
    3. Process non-internal gates with updated references
    4. Renumber everything, clean up dead code

    Args:
        state: The original circuit state.
        window: The window to replace.
        new_gates: The new gate implementation. Indices 0..len(window.inputs)-1
                   refer to window inputs; higher indices refer to gates within
                   new_gates.

    Returns:
        A new CircuitState with the window replaced.
    """
    internal_set = set(window.internal)

    old_to_new: dict[int, int] = {}
    for i in range(state.input_bits):
        old_to_new[i] = i

    new_gates_list: list[tuple[str, int, int]] = []

    local_to_full: dict[int, int] = {}
    for i, inp_idx in enumerate(window.inputs):
        local_to_full[i] = inp_idx

    wire_gates = [g for g in new_gates if g[0] == "wire"]
    real_gates = [g for g in new_gates if g[0] != "wire"]

    for local_idx, (op, left, right) in enumerate(real_gates):
        if left < len(window.inputs):
            new_left = local_to_full[left]
        else:
            new_left = local_to_full[left]

        if op not in ("const", "not"):
            if right < len(window.inputs):
                new_right = local_to_full[right]
            else:
                new_right = local_to_full[right]
        else:
            new_right = right

        new_idx = state.input_bits + len(new_gates_list)
        new_gates_list.append((op, new_left, new_right))
        local_to_full[len(window.inputs) + local_idx] = new_idx

    if wire_gates:
        final_new_output = local_to_full[wire_gates[0][1]]
    elif real_gates:
        final_new_output = local_to_full[len(window.inputs) + len(real_gates) - 1]
    elif window.inputs:
        final_new_output = local_to_full[0]
    else:
        final_new_output = 0

    for old_out in window.outputs:
        old_to_new[old_out] = final_new_output

    for old_gate_idx in range(len(state.gates)):
        full_idx = state.input_bits + old_gate_idx

        if full_idx in internal_set:
            continue

        op, left, right = state.gates[old_gate_idx]
        new_left = old_to_new.get(left, left)
        if op not in ("const", "not"):
            new_right = old_to_new.get(right, right)
        else:
            new_right = right

        new_idx = state.input_bits + len(new_gates_list)
        new_gates_list.append((op, new_left, new_right))
        old_to_new[full_idx] = new_idx

    new_outputs: list[tuple[int, bool]] = []
    for out_idx, inv in state.outputs:
        new_out = old_to_new.get(out_idx, out_idx)
        new_outputs.append((new_out, inv))

    result = CircuitState(
        input_bits=state.input_bits,
        output_bits=state.output_bits,
        gates=new_gates_list,
        outputs=new_outputs,
        gate_count=len(new_gates_list),
    )

    return result.eliminate_dead_code()
