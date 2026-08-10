"""Gate-level ternary synthesis for CircuitState.

This pass identifies 3-input boolean cones in the gate graph and replaces
them with single ternary gates using VPTERNLOG/lop3 imm8 encoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from stc.circuit_synth import CircuitState


@dataclass
class TernarySynthStats:
    """Statistics from ternary synthesis pass."""

    gates_before: int
    gates_after: int
    ternary_gates_created: int
    patterns_found: dict[str, int]


def compute_imm8(op1: str, op2: str, negate_result: bool = False) -> int:
    """Compute imm8 for nested 2-input operations.

    Supported patterns:
    - op1(op2(a, b), c) where op1, op2 in {and, or, xor}
    - op1(a, op2(b, c)) where op1, op2 in {and, or, xor}

    Args:
        op1: Outer operation ("and", "or", "xor")
        op2: Inner operation ("and", "or", "xor")
        negate_result: If True, negate the final result

    Returns:
        8-bit immediate for VPTERNLOG instruction
    """
    ops = {
        "and": lambda x, y: x & y,
        "or": lambda x, y: x | y,
        "xor": lambda x, y: x ^ y,
    }

    imm8 = 0
    for i in range(8):
        a = (i >> 2) & 1
        b = (i >> 1) & 1
        c = i & 1

        inner = ops[op2](a, b)
        result = ops[op1](inner, c)

        if negate_result:
            result = 1 - result

        if result:
            imm8 |= 1 << i

    return imm8


def compute_imm8_with_nots(
    op1: str, op2: str, not_a: bool, not_b: bool, not_c: bool, not_result: bool
) -> int:
    """Compute imm8 with optional input/output negations.

    Pattern: [NOT] op1([NOT]a_input, op2([NOT]b_input, [NOT]c_input))
    """
    ops = {
        "and": lambda x, y: x & y,
        "or": lambda x, y: x | y,
        "xor": lambda x, y: x ^ y,
    }

    imm8 = 0
    for i in range(8):
        a = (i >> 2) & 1
        b = (i >> 1) & 1
        c = i & 1

        if not_a:
            a = 1 - a
        if not_b:
            b = 1 - b
        if not_c:
            c = 1 - c

        inner = ops[op2](b, c)
        result = ops[op1](a, inner)

        if not_result:
            result = 1 - result

        if result:
            imm8 |= 1 << i

    return imm8


@dataclass
class GateInfo:
    """Information about a gate for pattern matching."""

    op: str
    inputs: tuple[int, ...]
    imm8: int | None = None
    fanout: int = 0


def build_gate_info(gates: Sequence[tuple], input_bits: int) -> list[GateInfo]:
    """Build GateInfo list with fanout counts.

    Args:
        gates: List of gate tuples from CircuitState
        input_bits: Number of input bits (gates are indexed starting at input_bits)

    Returns:
        List of GateInfo, one per gate
    """
    infos = []
    fanout_counts = [0] * (input_bits + len(gates))

    for gate in gates:
        op = gate[0]
        if op == "const":
            inputs = ()
        elif op == "not":
            inputs = (gate[1],)
        elif op == "ternary":
            inputs = (gate[1], gate[2], gate[3])
        elif op == "andn":
            inputs = (gate[1], gate[2])
        else:
            inputs = (gate[1], gate[2])

        for inp in inputs:
            if inp < len(fanout_counts):
                fanout_counts[inp] += 1

        imm8 = gate[4] if op == "ternary" else None
        infos.append(GateInfo(op=op, inputs=inputs, imm8=imm8))

    for i, info in enumerate(infos):
        info.fanout = fanout_counts[input_bits + i]

    return infos


def find_ternary_patterns(
    gates: Sequence[tuple],
    gate_infos: list[GateInfo],
    input_bits: int,
) -> list[tuple[int, int, int, int, int, int]]:
    """Find gate pairs that can be merged into ternary operations.

    Returns list of (outer_gate_idx, inner_gate_idx, a, b, c, imm8) tuples.

    Pattern: outer_op(inner_gate_result, other_input)
    where inner_gate = inner_op(input1, input2)

    Only matches if inner_gate has fanout == 1 (used only by outer).
    """
    patterns = []

    for outer_idx, outer_info in enumerate(gate_infos):
        if outer_info.op not in ("and", "or", "xor"):
            continue

        outer_in0, outer_in1 = outer_info.inputs

        for inner_pos, (inner_idx, other_idx) in enumerate(
            [
                (outer_in0, outer_in1),
                (outer_in1, outer_in0),
            ]
        ):
            if inner_idx < input_bits:
                continue

            inner_gate_idx = inner_idx - input_bits
            if inner_gate_idx >= len(gate_infos):
                continue

            inner_info = gate_infos[inner_gate_idx]

            if inner_info.op not in ("and", "or", "xor"):
                continue
            if inner_info.fanout != 1:
                continue

            inner_in0, inner_in1 = inner_info.inputs

            if inner_pos == 0:
                imm8 = compute_imm8(outer_info.op, inner_info.op)
            else:
                imm8 = compute_imm8_swapped(outer_info.op, inner_info.op)

            patterns.append(
                (
                    outer_idx,
                    inner_gate_idx,
                    inner_in0,
                    inner_in1,
                    other_idx,
                    imm8,
                )
            )

    return patterns


def compute_imm8_swapped(op1: str, op2: str) -> int:
    """Compute imm8 for op1(c, op2(a, b)) pattern."""
    ops = {
        "and": lambda x, y: x & y,
        "or": lambda x, y: x | y,
        "xor": lambda x, y: x ^ y,
    }

    imm8 = 0
    for i in range(8):
        a = (i >> 2) & 1
        b = (i >> 1) & 1
        c = i & 1

        inner = ops[op2](a, b)
        result = ops[op1](c, inner)

        if result:
            imm8 |= 1 << i

    return imm8


def apply_ternary_synthesis(
    circuit: CircuitState,
) -> tuple[CircuitState, TernarySynthStats]:
    """Apply ternary synthesis to a CircuitState using batch processing.

    This implementation uses a batch algorithm to achieve O(N²) complexity:
    1. Build gate info once with fanout counts
    2. Find all valid ternary patterns upfront
    3. Select non-conflicting patterns greedily
    4. Apply entire batch in single reconstruction pass

    Args:
        circuit: Input CircuitState with and/or/xor/not gates

    Returns:
        (new_circuit, stats) tuple
    """
    gates = list(circuit.gates)
    input_bits = circuit.input_bits
    output_bits = circuit.output_bits
    outputs = list(circuit.outputs)

    gates_before = len(gates)
    ternary_created = 0
    pattern_counts: dict[str, int] = {}

    gate_infos = build_gate_info(gates, input_bits)
    all_patterns = find_ternary_patterns(gates, gate_infos, input_bits)

    if not all_patterns:
        stats = TernarySynthStats(
            gates_before=gates_before,
            gates_after=gates_before,
            ternary_gates_created=0,
            patterns_found={},
        )
        return circuit, stats

    used_gates: set[int] = set()
    selected_patterns: list[tuple[int, int, int, int, int, int]] = []

    for pattern in all_patterns:
        outer_idx, inner_idx, a, b, c, imm8 = pattern
        if outer_idx in used_gates or inner_idx in used_gates:
            continue
        selected_patterns.append(pattern)
        used_gates.add(outer_idx)
        used_gates.add(inner_idx)

        inner_op = gate_infos[inner_idx].op
        outer_op = gate_infos[outer_idx].op
        pattern_key = f"{outer_op}_{inner_op}"
        pattern_counts[pattern_key] = pattern_counts.get(pattern_key, 0) + 1

    if not selected_patterns:
        stats = TernarySynthStats(
            gates_before=gates_before,
            gates_after=gates_before,
            ternary_gates_created=0,
            patterns_found={},
        )
        return circuit, stats

    merged_gates: set[int] = set()
    replacement_map: dict[int, tuple[str, int, int, int, int]] = {}

    for outer_idx, inner_idx, a, b, c, imm8 in selected_patterns:
        merged_gates.add(inner_idx)
        replacement_map[outer_idx] = ("ternary", a, b, c, imm8)
        ternary_created += 1

    new_gates = []
    idx_remap: dict[int, int] = {}

    for i, gate in enumerate(gates):
        if i in merged_gates:
            continue

        old_node = input_bits + i
        new_idx = input_bits + len(new_gates)
        idx_remap[old_node] = new_idx

        if i in replacement_map:
            op, a, b, c, imm8 = replacement_map[i]
            a_new = idx_remap.get(a, a)
            b_new = idx_remap.get(b, b)
            c_new = idx_remap.get(c, c)
            new_gates.append(("ternary", a_new, b_new, c_new, imm8))
        else:
            remapped_gate = _remap_gate_inputs(gate, idx_remap, input_bits)
            new_gates.append(remapped_gate)

    for outer_idx, inner_idx, _, _, _, _ in selected_patterns:
        inner_node = input_bits + inner_idx
        outer_node = input_bits + outer_idx
        if outer_node in idx_remap:
            idx_remap[inner_node] = idx_remap[outer_node]

    new_outputs = []
    for out_idx, invert in outputs:
        if out_idx >= input_bits:
            new_out_idx = idx_remap.get(out_idx, out_idx)
        else:
            new_out_idx = out_idx
        new_outputs.append((new_out_idx, invert))

    new_circuit = CircuitState(
        input_bits=input_bits,
        output_bits=output_bits,
        gates=new_gates,
        outputs=new_outputs,
        gate_count=len(new_gates),
    )

    stats = TernarySynthStats(
        gates_before=gates_before,
        gates_after=len(new_gates),
        ternary_gates_created=ternary_created,
        patterns_found=pattern_counts,
    )

    return new_circuit, stats


def find_andnot_patterns(
    gates: Sequence[tuple],
    gate_infos: list[GateInfo],
    input_bits: int,
) -> list[tuple[int, int, int, int]]:
    """Find AND(NOT(x), y) patterns that can use ANDNOT.

    Returns list of (and_gate_idx, not_gate_idx, x, y) tuples.
    x is the input to NOT, y is the other AND input.
    """
    patterns = []

    for and_idx, info in enumerate(gate_infos):
        if info.op != "and":
            continue

        in0, in1 = info.inputs

        for not_idx_candidate, other in [(in0, in1), (in1, in0)]:
            if not_idx_candidate < input_bits:
                continue

            not_gate_idx = not_idx_candidate - input_bits
            if not_gate_idx >= len(gate_infos):
                continue

            not_info = gate_infos[not_gate_idx]
            if not_info.op != "not":
                continue
            if not_info.fanout != 1:
                continue

            x = not_info.inputs[0]
            y = other
            patterns.append((and_idx, not_gate_idx, x, y))

    return patterns


def apply_andnot_optimization(circuit: CircuitState) -> tuple[CircuitState, dict]:
    """Replace AND(NOT(x), y) with ANDNOT(x, y).

    Returns (optimized_circuit, stats).
    """
    gates = list(circuit.gates)
    input_bits = circuit.input_bits

    andnot_count = 0
    idx_remap = {}

    changed = True
    while changed:
        changed = False
        gate_infos = build_gate_info(gates, input_bits)
        patterns = find_andnot_patterns(gates, gate_infos, input_bits)

        if not patterns:
            break

        and_idx, not_idx, x, y = patterns[0]

        new_gates = []
        pass_remap = {}

        for i, gate in enumerate(gates):
            if i == not_idx:
                continue
            elif i == and_idx:
                new_idx = input_bits + len(new_gates)
                x_new = pass_remap.get(x, x) if x >= input_bits else x
                y_new = pass_remap.get(y, y) if y >= input_bits else y
                new_gates.append(("andn", x_new, y_new))
                pass_remap[input_bits + i] = new_idx
                andnot_count += 1
            else:
                new_idx = input_bits + len(new_gates)
                new_gate = _remap_gate_inputs(gate, pass_remap, input_bits)
                new_gates.append(new_gate)
                pass_remap[input_bits + i] = new_idx

        for old_idx in list(idx_remap.keys()):
            idx_remap[old_idx] = pass_remap.get(idx_remap[old_idx], idx_remap[old_idx])
        for old_idx, new_idx in pass_remap.items():
            if old_idx not in idx_remap:
                idx_remap[old_idx] = new_idx

        gates = new_gates
        changed = True

    outputs = list(circuit.outputs)
    new_outputs = []
    for out_idx, invert in outputs:
        if out_idx >= input_bits:
            new_out_idx = idx_remap.get(out_idx, out_idx)
        else:
            new_out_idx = out_idx
        new_outputs.append((new_out_idx, invert))

    new_circuit = CircuitState(
        input_bits=input_bits,
        output_bits=circuit.output_bits,
        gates=gates,
        outputs=new_outputs,
        gate_count=len(gates),
    )

    return new_circuit, {"andnot_created": andnot_count}


def _remap_gate_inputs(gate: tuple, idx_remap: dict, input_bits: int) -> tuple:
    """Remap gate inputs through index mapping."""
    op = gate[0]
    if op == "const":
        return gate
    elif op == "not":
        inp = gate[1]
        new_inp = idx_remap.get(inp, inp) if inp >= input_bits else inp
        return ("not", new_inp, 0)
    elif op == "ternary":
        a, b, c = gate[1], gate[2], gate[3]
        new_a = idx_remap.get(a, a) if a >= input_bits else a
        new_b = idx_remap.get(b, b) if b >= input_bits else b
        new_c = idx_remap.get(c, c) if c >= input_bits else c
        return ("ternary", new_a, new_b, new_c, gate[4])
    elif op == "andn":
        a, b = gate[1], gate[2]
        new_a = idx_remap.get(a, a) if a >= input_bits else a
        new_b = idx_remap.get(b, b) if b >= input_bits else b
        return ("andn", new_a, new_b)
    else:
        in0, in1 = gate[1], gate[2]
        new_in0 = idx_remap.get(in0, in0) if in0 >= input_bits else in0
        new_in1 = idx_remap.get(in1, in1) if in1 >= input_bits else in1
        return (op, new_in0, new_in1)


def eliminate_double_nots(circuit: CircuitState) -> tuple[CircuitState, dict]:
    """Eliminate NOT(NOT(x)) -> x patterns.

    Returns (optimized_circuit, stats).
    """
    gates = list(circuit.gates)
    input_bits = circuit.input_bits
    outputs = list(circuit.outputs)

    eliminated = 0
    idx_remap = {}
    changed = True

    while changed:
        changed = False
        gate_infos = build_gate_info(gates, input_bits)

        found_pattern = None
        for i, info in enumerate(gate_infos):
            if info.op != "not":
                continue

            inp = info.inputs[0]
            if inp < input_bits:
                continue

            inner_idx = inp - input_bits
            if inner_idx >= len(gate_infos):
                continue

            inner_info = gate_infos[inner_idx]
            if inner_info.op != "not":
                continue

            if inner_info.fanout == 1:
                found_pattern = (i, inner_idx, inner_info.inputs[0])
                break

        if found_pattern is None:
            break

        outer_idx, inner_idx, x = found_pattern

        new_gates = []
        pass_remap = {}

        for i, gate in enumerate(gates):
            if i == inner_idx or i == outer_idx:
                continue
            else:
                new_idx = input_bits + len(new_gates)
                new_gate = _remap_gate_inputs(gate, pass_remap, input_bits)
                new_gates.append(new_gate)
                pass_remap[input_bits + i] = new_idx

        pass_remap[input_bits + outer_idx] = x

        for old_idx in list(idx_remap.keys()):
            idx_remap[old_idx] = pass_remap.get(idx_remap[old_idx], idx_remap[old_idx])
        for old_idx, new_idx in pass_remap.items():
            if old_idx not in idx_remap:
                idx_remap[old_idx] = new_idx

        new_outputs = []
        for out_idx, invert in outputs:
            if out_idx >= input_bits:
                new_out_idx = pass_remap.get(out_idx, out_idx)
            else:
                new_out_idx = out_idx
            new_outputs.append((new_out_idx, invert))

        gates = new_gates
        outputs = new_outputs
        eliminated += 2
        changed = True

    new_circuit = CircuitState(
        input_bits=input_bits,
        output_bits=circuit.output_bits,
        gates=gates,
        outputs=outputs,
        gate_count=len(gates),
    )

    return new_circuit, {"double_nots_eliminated": eliminated}


def eliminate_dead_gates(circuit: CircuitState) -> tuple[CircuitState, dict]:
    """Remove gates with fanout=0 that aren't circuit outputs.

    Returns (optimized_circuit, stats).
    """
    gates = list(circuit.gates)
    input_bits = circuit.input_bits
    outputs = list(circuit.outputs)

    gate_infos = build_gate_info(gates, input_bits)
    output_gate_indices = {idx - input_bits for idx, _ in outputs if idx >= input_bits}

    live_gates: set[int] = set()
    worklist = list(output_gate_indices)

    while worklist:
        gi = worklist.pop()
        if gi < 0 or gi >= len(gate_infos) or gi in live_gates:
            continue
        live_gates.add(gi)

        for inp in gate_infos[gi].inputs:
            if inp >= input_bits:
                worklist.append(inp - input_bits)

    dead_count = len(gates) - len(live_gates)

    if dead_count == 0:
        return circuit, {"dead_gates_removed": 0}

    new_gates = []
    idx_remap: dict[int, int] = {}

    for i, gate in enumerate(gates):
        if i not in live_gates:
            continue

        old_node = input_bits + i
        new_idx = input_bits + len(new_gates)
        idx_remap[old_node] = new_idx

        remapped_gate = _remap_gate_inputs(gate, idx_remap, input_bits)
        new_gates.append(remapped_gate)

    new_outputs = []
    for out_idx, invert in outputs:
        if out_idx >= input_bits:
            new_out_idx = idx_remap.get(out_idx, out_idx)
        else:
            new_out_idx = out_idx
        new_outputs.append((new_out_idx, invert))

    new_circuit = CircuitState(
        input_bits=input_bits,
        output_bits=circuit.output_bits,
        gates=new_gates,
        outputs=new_outputs,
        gate_count=len(new_gates),
    )

    return new_circuit, {"dead_gates_removed": dead_count}
