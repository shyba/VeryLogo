"""Lower the explicit boolean subset of MIR to the floor-planner v1 IR."""

from __future__ import annotations

from stc.mir import Binary, MIRFunction, Mux, Ternary, Unary, VReg
from stc.sched.floor_planner import FloorOp, FloorPlannerError, FloorProgram, FloorSchedule, plan_floor
from stc.sched.cpu_model import CpuModel
from stc.sched.emit_x86_asm import emit_x86_64_asm


_BITWISE_ASM = {
    "and": "vpandq",
    "or": "vporq",
    "xor": "vpxorq",
    "andn": "vpandnq",
}


def mir_to_floor_program(mir: MIRFunction) -> FloorProgram:
    """Lower the supported boolean MIR instructions to machine-planner nodes.

    The function rejects operations whose register constraints or instruction
    expansions are not represented by floor-planner v1.  Rejecting them here
    is important: passing them through would produce a schedule that cannot be
    emitted without a compiler taking over again.
    """

    inputs = tuple(_vreg_id(reg, "input") for reg in mir.input_regs)
    outputs = tuple(_vreg_id(reg, "output") for reg in mir.output_regs)
    operations: list[FloorOp] = []
    for op_id, instruction in enumerate(mir.instructions):
        if isinstance(instruction, Binary):
            if instruction.op not in _BITWISE_ASM:
                raise FloorPlannerError(
                    f"MIR binary operation {instruction.op!r} is outside floor-planner v1"
                )
            dst = _vreg_id(instruction.dst, f"instruction {op_id} destination")
            left = _vreg_id(instruction.a, f"instruction {op_id} input")
            right = _vreg_id(instruction.b, f"instruction {op_id} input")
            operations.append(
                FloorOp(
                    id=op_id,
                    family="bitwise",
                    inputs=(left, right),
                    output=dst,
                    operands="v,v,v",
                    asm_mnemonic=_BITWISE_ASM[instruction.op],
                )
            )
            continue
        if isinstance(instruction, Unary) and instruction.op == "not":
            dst = _vreg_id(instruction.dst, f"instruction {op_id} destination")
            value = _vreg_id(instruction.a, f"instruction {op_id} input")
            operations.append(
                FloorOp(
                    id=op_id,
                    family="ternary",
                    inputs=(value, value, value),
                    output=dst,
                    operands="v,v,v",
                    asm_mnemonic="vpternlogq",
                    immediate=0x01,
                    tied_input=0,
                )
            )
            continue
        if isinstance(instruction, Ternary):
            dst = _vreg_id(instruction.dst, f"instruction {op_id} destination")
            values = tuple(
                _vreg_id(value, f"instruction {op_id} input")
                for value in (instruction.a, instruction.b, instruction.c)
            )
            operations.append(
                FloorOp(
                    id=op_id,
                    family="ternary",
                    inputs=values,
                    output=dst,
                    operands="v,v,v",
                    asm_mnemonic="vpternlogq",
                    immediate=instruction.imm8,
                    tied_input=0,
                )
            )
            continue
        if isinstance(instruction, Mux):
            dst = _vreg_id(instruction.dst, f"instruction {op_id} destination")
            values = tuple(
                _vreg_id(value, f"instruction {op_id} input")
                for value in (instruction.select, instruction.a, instruction.b)
            )
            operations.append(
                FloorOp(
                    id=op_id,
                    family="ternary",
                    inputs=values,
                    output=dst,
                    operands="v,v,v",
                    asm_mnemonic="vpternlogq",
                    immediate=0xCA,
                    tied_input=0,
                )
            )
            continue
        raise FloorPlannerError(
            f"MIR instruction {op_id} ({type(instruction).__name__}) "
            "is outside floor-planner v1"
        )
    return FloorProgram(inputs=inputs, outputs=outputs, operations=tuple(operations))


def plan_mir_floor(
    mir: MIRFunction,
    cpu: CpuModel,
    *,
    register_file: str | None = None,
) -> tuple[FloorProgram, FloorSchedule]:
    """Lower and plan a MIR function without invoking C code generation."""

    program = mir_to_floor_program(mir)
    return program, plan_floor(program, cpu, register_file=register_file)


def emit_mir_x86_64_asm(
    mir: MIRFunction,
    cpu: CpuModel,
    *,
    function_name: str = "floor_fn",
    register_file: str | None = None,
) -> str:
    """Lower, schedule, and emit a MIR boolean function as GNU-as text."""

    program, schedule = plan_mir_floor(
        mir, cpu, register_file=register_file
    )
    return emit_x86_64_asm(program, schedule, function_name=function_name)


def _vreg_id(value: object, role: str) -> int:
    if not isinstance(value, VReg):
        raise FloorPlannerError(f"{role} must be a virtual register, got {value!r}")
    return value.id
