"""Lowering from CircuitState to Machine IR."""

from __future__ import annotations

from stc.circuit_synth import CircuitState
from stc.mir import (
    Binary,
    Const,
    Copy,
    Load,
    MIRFunction,
    Mux,
    Store,
    Ternary,
    Unary,
    VReg,
)
from stc.sched.regalloc import RegAllocation
from stc.sched.schedule import Schedule


def circuit_to_mir(
    circuit: CircuitState,
    schedule: Schedule,
    allocation: RegAllocation,
) -> MIRFunction:
    """Lower CircuitState to Machine IR.

    Args:
        circuit: Gate-level circuit representation
        schedule: Gate scheduling (order of execution)
        allocation: Register allocation result

    Returns:
        MIRFunction with MIR instructions
    """
    input_bits = circuit.input_bits
    gates = list(circuit.gates)
    outputs = list(circuit.outputs)

    input_regs = [VReg(i) for i in range(input_bits)]
    output_regs = [VReg(outputs[i][0]) for i in range(len(outputs))]

    instructions: list = []
    num_virtual_regs = input_bits + len(gates)

    spill_map: dict[int, int] = {}
    for i, node in enumerate(allocation.spills):
        spill_map[node] = i

    load_nodes = {node for node, _, _ in allocation.loads}
    for node in load_nodes:
        if node in spill_map:
            dst = VReg(node)
            slot = spill_map[node]
            instructions.append(Load(dst=dst, slot=slot))

    gates_by_cycle: dict[int, list[int]] = {}
    for g_idx, cycle in schedule.gate_cycle.items():
        if cycle not in gates_by_cycle:
            gates_by_cycle[cycle] = []
        gates_by_cycle[cycle].append(g_idx)

    for cycle in sorted(gates_by_cycle.keys()):
        for g_idx in gates_by_cycle[cycle]:
            node_idx = input_bits + g_idx
            gate = gates[g_idx]
            dst = VReg(node_idx)

            if gate[0] == "and":
                _, a, b = gate
                instructions.append(Binary(dst=dst, op="and", a=VReg(a), b=VReg(b)))
            elif gate[0] == "or":
                _, a, b = gate
                instructions.append(Binary(dst=dst, op="or", a=VReg(a), b=VReg(b)))
            elif gate[0] == "xor":
                _, a, b = gate
                instructions.append(Binary(dst=dst, op="xor", a=VReg(a), b=VReg(b)))
            elif gate[0] == "not":
                _, a = gate
                instructions.append(Unary(dst=dst, op="not", a=VReg(a)))
            elif gate[0] == "ternary":
                _, a, b, c, imm8 = gate
                instructions.append(
                    Ternary(dst=dst, a=VReg(a), b=VReg(b), c=VReg(c), imm8=imm8)
                )
            elif gate[0] == "mux":
                _, sel, a, b = gate
                instructions.append(
                    Mux(dst=dst, select=VReg(sel), a=VReg(a), b=VReg(b))
                )
            elif gate[0] == "const":
                _, value = gate
                instructions.append(Const(dst=dst, value=value))
            elif gate[0] == "copy":
                _, src = gate
                instructions.append(Copy(dst=dst, src=VReg(src)))
            else:
                raise ValueError(f"Unknown gate type: {gate[0]}")

    store_nodes = {node for node, _, _ in allocation.stores}
    for node in store_nodes:
        if node in spill_map:
            src = VReg(node)
            slot = spill_map[node]
            instructions.append(Store(src=src, slot=slot, dst=None))

    return MIRFunction(
        input_regs=input_regs,
        output_regs=output_regs,
        instructions=instructions,
        num_virtual_regs=num_virtual_regs,
    )
