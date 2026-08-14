"""CircuitState adapter for the explicit floor-planner v2 path."""

from __future__ import annotations

from stc.circuit_synth import CircuitState
from stc.sched.cpu_model import CpuModel
from stc.sched.emit_x86_asm import emit_x86_64_asm
from stc.sched.floor_planner import (
    FloorOp,
    FloorPlannerError,
    FloorProgram,
    FloorSchedule,
    plan_floor,
)
from stc.sched.x86_encodings import x86_encoding


_GATE_ASM = {
    "and": "vpandq",
    "andn": "vpandnq",
    "andnot": "vpandnq",
    "or": "vporq",
    "xor": "vpxorq",
}


def circuit_to_floor_program(circuit: CircuitState) -> FloorProgram:
    """Lower the direct boolean gate subset to floor IR.

    ``not``, ``mux``, and ``ternary`` use VPTERNLOG's destructive destination;
    the planner therefore adds a tied-input constraint and can reject graphs
    where the source value must remain live after the operation.
    """

    operations: list[FloorOp] = []
    for gate_id, gate in enumerate(circuit.gates):
        if not gate:
            raise FloorPlannerError(f"gate {gate_id} is empty")
        op = gate[0]
        if op in _GATE_ASM and len(gate) == 3:
            _, left, right = gate
            operations.append(
                FloorOp(
                    id=gate_id,
                    family="bitwise",
                    inputs=(left, right),
                    output=circuit.input_bits + gate_id,
                    operands="v,v,v",
                    asm_mnemonic=_GATE_ASM[op],
                    encoding=x86_encoding(_GATE_ASM[op]),
                )
            )
            continue
        if op == "not" and len(gate) == 2:
            inputs = (gate[1],) * 3
            operations.append(
                FloorOp(
                    id=gate_id,
                    family="ternary",
                    inputs=inputs,
                    output=circuit.input_bits + gate_id,
                    operands="v,v,v",
                    asm_mnemonic="vpternlogq",
                    immediate=0x01,
                    tied_input=0,
                    encoding=x86_encoding("vpternlogq"),
                )
            )
            continue
        if op == "mux" and len(gate) == 4:
            _, select, left, right = gate
            operations.append(
                FloorOp(
                    id=gate_id,
                    family="ternary",
                    inputs=(select, left, right),
                    output=circuit.input_bits + gate_id,
                    operands="v,v,v",
                    asm_mnemonic="vpternlogq",
                    immediate=0xCA,
                    tied_input=0,
                    encoding=x86_encoding("vpternlogq"),
                )
            )
            continue
        if op == "ternary" and len(gate) == 5:
            _, left, middle, right, immediate = gate
            operations.append(
                FloorOp(
                    id=gate_id,
                    family="ternary",
                    inputs=(left, middle, right),
                    output=circuit.input_bits + gate_id,
                    operands="v,v,v",
                    asm_mnemonic="vpternlogq",
                    immediate=immediate,
                    tied_input=0,
                    encoding=x86_encoding("vpternlogq"),
                )
            )
            continue
        if op in {"const", "copy"}:
            raise FloorPlannerError(
                f"gate {gate_id} ({op!r}) is outside floor-planner v2; "
                "constant and copy expansions need explicit machine forms"
            )
        raise FloorPlannerError(f"gate {gate_id} ({op!r}) is outside floor-planner v2")

    outputs: list[int] = []
    for output_id, inverted in circuit.outputs:
        if inverted:
            raise FloorPlannerError(
                "inverted circuit outputs need an explicit v2 instruction expansion"
            )
        outputs.append(output_id)
    return FloorProgram(
        inputs=tuple(range(circuit.input_bits)),
        outputs=tuple(outputs),
        operations=tuple(operations),
    )


def plan_circuit_floor(
    circuit: CircuitState,
    cpu: CpuModel,
    *,
    register_file: str | None = None,
) -> tuple[FloorProgram, FloorSchedule]:
    """Lower and schedule a CircuitState without C/GCC code generation."""

    program = circuit_to_floor_program(circuit)
    return program, plan_floor(program, cpu, register_file=register_file)


def emit_circuit_x86_64_asm(
    circuit: CircuitState,
    cpu: CpuModel,
    *,
    function_name: str = "floor_fn",
    register_file: str | None = None,
) -> str:
    """Emit a supported CircuitState as direct GNU-as x86-64 text."""

    program, schedule = plan_circuit_floor(circuit, cpu, register_file=register_file)
    return emit_x86_64_asm(program, schedule, function_name=function_name)
