"""CircuitState adapter for the explicit floor-planner v1 path."""

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


_GATE_ASM = {
    "and": "vpandq",
    "andn": "vpandnq",
    "andnot": "vpandnq",
    "or": "vporq",
    "xor": "vpxorq",
}


def circuit_to_floor_program(circuit: CircuitState) -> FloorProgram:
    """Lower the direct three-input boolean gate subset to floor IR."""

    operations: list[FloorOp] = []
    for gate_id, gate in enumerate(circuit.gates):
        if not gate:
            raise FloorPlannerError(f"gate {gate_id} is empty")
        op = gate[0]
        if op not in _GATE_ASM or len(gate) != 3:
            raise FloorPlannerError(
                f"gate {gate_id} ({op!r}) is outside floor-planner v1; "
                "only binary bitwise gates are supported"
            )
        _, left, right = gate
        output = circuit.input_bits + gate_id
        operations.append(
            FloorOp(
                id=gate_id,
                family="bitwise",
                inputs=(left, right),
                output=output,
                operands="v,v,v",
                asm_mnemonic=_GATE_ASM[op],
            )
        )

    outputs: list[int] = []
    for output_id, inverted in circuit.outputs:
        if inverted:
            raise FloorPlannerError(
                "inverted circuit outputs need an explicit v1 instruction expansion"
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

    program, schedule = plan_circuit_floor(
        circuit, cpu, register_file=register_file
    )
    return emit_x86_64_asm(program, schedule, function_name=function_name)

