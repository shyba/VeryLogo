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


def _record_use(use_sites: dict[int, list[int]], reg: VReg, idx: int) -> None:
    use_sites.setdefault(reg.id, []).append(idx)


def _compute_use_sites(
    instructions: list, output_regs: list[VReg]
) -> dict[int, list[int]]:
    use_sites: dict[int, list[int]] = {}
    for idx, inst in enumerate(instructions):
        if isinstance(inst, Binary):
            if isinstance(inst.a, VReg):
                _record_use(use_sites, inst.a, idx)
            if isinstance(inst.b, VReg):
                _record_use(use_sites, inst.b, idx)
        elif isinstance(inst, Unary):
            if isinstance(inst.a, VReg):
                _record_use(use_sites, inst.a, idx)
        elif isinstance(inst, Ternary):
            if isinstance(inst.a, VReg):
                _record_use(use_sites, inst.a, idx)
            if isinstance(inst.b, VReg):
                _record_use(use_sites, inst.b, idx)
            if isinstance(inst.c, VReg):
                _record_use(use_sites, inst.c, idx)
        elif isinstance(inst, Mux):
            if isinstance(inst.select, VReg):
                _record_use(use_sites, inst.select, idx)
            if isinstance(inst.a, VReg):
                _record_use(use_sites, inst.a, idx)
            if isinstance(inst.b, VReg):
                _record_use(use_sites, inst.b, idx)
        elif isinstance(inst, Copy):
            if isinstance(inst.src, VReg):
                _record_use(use_sites, inst.src, idx)
        elif isinstance(inst, Store):
            if isinstance(inst.src, VReg):
                _record_use(use_sites, inst.src, idx)

    out_idx = len(instructions)
    for out in output_regs:
        _record_use(use_sites, out, out_idx)
    return use_sites


def _canonicalize_boolean_patterns(instructions: list, output_regs: list[VReg]) -> list:
    """Apply local MIR boolean peepholes.

    Current rewrite:
    - `not(x)` consumed once by `and` => `andn(x, y)` and drop the `not`.
    """
    use_sites = _compute_use_sites(instructions, output_regs)
    not_alias: dict[int, VReg] = {}
    skip_not_indices: set[int] = set()

    for idx, inst in enumerate(instructions):
        if not isinstance(inst, Unary) or inst.op != "not":
            continue
        if not isinstance(inst.dst, VReg) or not isinstance(inst.a, VReg):
            continue
        uses = use_sites.get(inst.dst.id, [])
        if len(uses) != 1:
            continue
        use_idx = uses[0]
        if use_idx >= len(instructions):
            continue
        user = instructions[use_idx]
        if not (isinstance(user, Binary) and user.op == "and"):
            continue
        not_alias[inst.dst.id] = inst.a
        skip_not_indices.add(idx)

    if not not_alias:
        return instructions

    rewritten: list = []
    for idx, inst in enumerate(instructions):
        if idx in skip_not_indices:
            continue
        if isinstance(inst, Binary) and inst.op == "and":
            if isinstance(inst.a, VReg) and inst.a.id in not_alias:
                rewritten.append(
                    Binary(dst=inst.dst, op="andn", a=not_alias[inst.a.id], b=inst.b)
                )
                continue
            if isinstance(inst.b, VReg) and inst.b.id in not_alias:
                rewritten.append(
                    Binary(dst=inst.dst, op="andn", a=not_alias[inst.b.id], b=inst.a)
                )
                continue
        rewritten.append(inst)
    return rewritten


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

    gates_by_cycle: dict[int, list[int]] = {}
    for g_idx, cycle in schedule.gate_cycle.items():
        if cycle not in gates_by_cycle:
            gates_by_cycle[cycle] = []
        gates_by_cycle[cycle].append(g_idx)

    stores_by_cycle: dict[int, list[tuple[int, int, int]]] = {}
    for node, reg, cycle in allocation.stores:
        stores_by_cycle.setdefault(cycle, []).append((node, reg, cycle))

    loads_by_cycle: dict[int, list[tuple[int, int, int]]] = {}
    for node, reg, cycle in allocation.loads:
        loads_by_cycle.setdefault(cycle, []).append((node, reg, cycle))

    all_cycles = sorted(
        set(gates_by_cycle) | set(loads_by_cycle) | set(stores_by_cycle)
    )
    for cycle in all_cycles:
        if cycle in stores_by_cycle:
            for node, _reg, _ in sorted(
                stores_by_cycle[cycle], key=lambda x: (x[0], x[1])
            ):
                if node in spill_map:
                    src = VReg(node)
                    slot = spill_map[node]
                    instructions.append(Store(src=src, slot=slot, dst=None))

        if cycle in loads_by_cycle:
            for node, _reg, _ in sorted(
                loads_by_cycle[cycle], key=lambda x: (x[0], x[1])
            ):
                if node in spill_map:
                    dst = VReg(node)
                    slot = spill_map[node]
                    instructions.append(Load(dst=dst, slot=slot))

        if cycle not in gates_by_cycle:
            continue
        for g_idx in sorted(gates_by_cycle[cycle]):
            node_idx = input_bits + g_idx
            gate = gates[g_idx]
            dst = VReg(node_idx)

            if gate[0] == "and":
                _, a, b = gate
                instructions.append(Binary(dst=dst, op="and", a=VReg(a), b=VReg(b)))
            elif gate[0] in {"andn", "andnot"}:
                _, a, b = gate
                instructions.append(Binary(dst=dst, op="andn", a=VReg(a), b=VReg(b)))
            elif gate[0] == "or":
                _, a, b = gate
                instructions.append(Binary(dst=dst, op="or", a=VReg(a), b=VReg(b)))
            elif gate[0] == "xor":
                _, a, b = gate
                instructions.append(Binary(dst=dst, op="xor", a=VReg(a), b=VReg(b)))
            elif gate[0] == "not":
                a = gate[1]
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
                value = gate[1]
                instructions.append(Const(dst=dst, value=value))
            elif gate[0] == "copy":
                _, src = gate
                instructions.append(Copy(dst=dst, src=VReg(src)))
            else:
                raise ValueError(f"Unknown gate type: {gate[0]}")

    instructions = _canonicalize_boolean_patterns(instructions, output_regs)

    return MIRFunction(
        input_regs=input_regs,
        output_regs=output_regs,
        instructions=instructions,
        num_virtual_regs=num_virtual_regs,
    )
