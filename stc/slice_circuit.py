from __future__ import annotations

from dataclasses import dataclass

from stc.circuit_synth import CircuitState


@dataclass(frozen=True)
class CircuitSlice:
    block_index: int
    input_offset: int
    input_bits: int
    output_indices: list[int]
    output_offset: int | None
    circuit: CircuitState


def _gate_operands(gate) -> list[int]:
    op = gate[0]
    if op == "ternary":
        return [int(gate[1]), int(gate[2]), int(gate[3])]
    if op in ("const",):
        return []
    if op in ("not", "shl", "lshr"):
        return [int(gate[1])]
    return [int(gate[1]), int(gate[2])]


def compute_node_dependencies(circuit: CircuitState) -> list[set[int]]:
    """Return dependency set for each node index (inputs + gates)."""
    input_bits = circuit.input_bits
    deps: list[set[int]] = [set([i]) for i in range(input_bits)]
    for g in circuit.gates:
        op = g[0]
        if op == "const":
            deps.append(set())
            continue
        operands = _gate_operands(g)
        acc: set[int] = set()
        for o in operands:
            if o < input_bits:
                acc.add(o)
            else:
                acc |= deps[o]
        deps.append(acc)
    return deps


def slice_circuit_by_block_size(
    circuit: CircuitState, block_size: int
) -> list[CircuitSlice] | None:
    """
    Slice a bit-level circuit into independent blocks of size `block_size`.

    Returns None if any output depends on inputs from multiple blocks.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if circuit.input_bits % block_size != 0:
        raise ValueError("block_size must divide input_bits")

    deps = compute_node_dependencies(circuit)
    outputs_by_block: dict[int, list[int]] = {}

    for out_idx, (node_idx, _inv) in enumerate(circuit.outputs):
        dep = deps[node_idx]
        if not dep:
            block_idx = 0
        else:
            min_dep = min(dep)
            max_dep = max(dep)
            block_idx = min_dep // block_size
            if min_dep < block_idx * block_size or max_dep >= (block_idx + 1) * block_size:
                return None
        outputs_by_block.setdefault(block_idx, []).append(out_idx)

    slices: list[CircuitSlice] = []
    input_blocks = circuit.input_bits // block_size

    for block_idx in range(input_blocks):
        if block_idx not in outputs_by_block:
            continue
        output_indices = sorted(outputs_by_block[block_idx])
        block_start = block_idx * block_size
        block_end = block_start + block_size - 1

        # Collect required gates via DFS from outputs.
        required_gates: set[int] = set()
        stack = [circuit.outputs[i][0] for i in output_indices]
        while stack:
            node = stack.pop()
            if node < circuit.input_bits:
                continue
            gate_idx = node - circuit.input_bits
            if gate_idx in required_gates:
                continue
            required_gates.add(gate_idx)
            gate = circuit.gates[gate_idx]
            for o in _gate_operands(gate):
                stack.append(o)

        gate_list = sorted(required_gates)
        gate_old_to_new = {gi: idx for idx, gi in enumerate(gate_list)}

        def map_operand(o: int) -> int:
            if o < circuit.input_bits:
                if o < block_start or o > block_end:
                    raise ValueError("operand outside block")
                return o - block_start
            return block_size + gate_old_to_new[o - circuit.input_bits]

        new_gates = []
        for gi in gate_list:
            gate = circuit.gates[gi]
            op = gate[0]
            if op == "const":
                new_gates.append((op, int(gate[1]), int(gate[2])))
            elif op == "ternary":
                new_gates.append(
                    (
                        op,
                        map_operand(int(gate[1])),
                        map_operand(int(gate[2])),
                        map_operand(int(gate[3])),
                        int(gate[4]),
                    )
                )
            elif op in ("not", "shl", "lshr"):
                if op in ("shl", "lshr") and len(gate) == 4:
                    new_gates.append(
                        (op, map_operand(int(gate[1])), int(gate[2]), int(gate[3]))
                    )
                elif op in ("shl", "lshr"):
                    new_gates.append((op, map_operand(int(gate[1])), int(gate[2])))
                else:
                    new_gates.append((op, map_operand(int(gate[1])), int(gate[2])))
            else:
                new_gates.append(
                    (op, map_operand(int(gate[1])), map_operand(int(gate[2])))
                )

        new_outputs = []
        for out_idx in output_indices:
            node_idx, inv = circuit.outputs[out_idx]
            if node_idx < circuit.input_bits:
                new_node = node_idx - block_start
            else:
                new_node = block_size + gate_old_to_new[node_idx - circuit.input_bits]
            new_outputs.append((new_node, inv))

        output_offset = None
        if output_indices:
            start = output_indices[0]
            if output_indices == list(range(start, start + len(output_indices))):
                output_offset = start

        slice_circuit = CircuitState(
            input_bits=block_size,
            output_bits=len(output_indices),
            gates=new_gates,
            outputs=new_outputs,
            gate_count=len(new_gates),
        )
        slices.append(
            CircuitSlice(
                block_index=block_idx,
                input_offset=block_start,
                input_bits=block_size,
                output_indices=output_indices,
                output_offset=output_offset,
                circuit=slice_circuit,
            )
        )

    return slices
