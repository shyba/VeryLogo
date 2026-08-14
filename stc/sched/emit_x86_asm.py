"""Direct GNU-as x86-64 emission for the floor-planner v1 IR.

This emitter intentionally returns assembly text rather than C or intrinsics.
The system assembler is the only tool needed to turn the result into an
object; GCC is not part of this path.  v1 supports non-destructive three-input
integer vector operations (the common AVX-512 bitwise forms).  Destructive or
multi-instruction forms must declare a target-specific expansion before they
are accepted.
"""

from __future__ import annotations

from stc.sched.floor_planner import FloorOp, FloorPlannerError, FloorProgram, FloorSchedule


_DEFAULT_MNEMONICS = {
    "and": "vpandq",
    "or": "vporq",
    "xor": "vpxorq",
    "andn": "vpandnq",
}


def emit_x86_64_asm(
    program: FloorProgram,
    schedule: FloorSchedule,
    *,
    function_name: str = "floor_fn",
) -> str:
    """Emit a SysV x86-64 function over 512-bit vector input/output arrays.

    The generated function has the ABI ``void f(const void *in, void *out)``.
    Each program input is one 64-byte vector at ``in + index * 64`` and each
    program output is stored at the corresponding output index.  Input and
    output values remain in the register file selected by the planner.
    """

    if not function_name.isidentifier():
        raise FloorPlannerError(f"invalid assembly function name {function_name!r}")
    if schedule.registers.register_file.lower() not in {"zmm", "zmm512", "vector"}:
        raise FloorPlannerError(
            "x86-64 AVX-512 emitter requires a zmm register file, got "
            f"{schedule.registers.register_file!r}"
        )

    assignment = schedule.registers.value_to_register
    lines = [
        ".text",
        f".globl {function_name}",
        f".type {function_name}, @function",
        f"{function_name}:",
    ]

    for input_index, value in enumerate(program.inputs):
        lines.append(
            f"    vmovdqu64 {input_index * 64}(%rdi), %zmm{_reg(assignment, value)}"
        )

    operations = sorted(
        program.operations,
        key=lambda op: (schedule.placement(op.id).cycle, op.id),
    )
    for op in operations:
        placement = schedule.placement(op.id)
        dst = _reg(assignment, op.output)
        sources = [_reg(assignment, value) for value in op.inputs]
        lines.extend(_emit_operation(op, dst, sources))

    for output_index, value in enumerate(program.outputs):
        lines.append(
            f"    vmovdqu64 %zmm{_reg(assignment, value)}, "
            f"{output_index * 64}(%rsi)"
        )

    lines.extend(
        [
            "    vzeroupper",
            "    ret",
            f".size {function_name}, .-{function_name}",
            ".section .note.GNU-stack,\"\",@progbits",
            "",
        ]
    )
    return "\n".join(lines)


def _reg(assignment: dict[int, int] | object, value: int) -> int:
    try:
        register = assignment[value]  # type: ignore[index]
    except (KeyError, TypeError) as exc:
        raise FloorPlannerError(f"value {value} has no physical register") from exc
    if not isinstance(register, int) or register < 0 or register > 31:
        raise FloorPlannerError(f"invalid zmm register assignment {register!r}")
    return register


def _emit_operation(op: FloorOp, dst: int, sources: list[int]) -> list[str]:
    mnemonic = op.asm_mnemonic or _DEFAULT_MNEMONICS.get(op.family)
    if mnemonic is None:
        raise FloorPlannerError(
            f"operation {op.id} ({op.family}) has no direct x86-64 mnemonic"
        )
    mnemonic = mnemonic.lower()

    if len(sources) == 2:
        # AT&T reverses the two Intel source operands.  This also gives
        # vpandnq the intended ``(~sources[0]) & sources[1]`` semantics.
        left, right = sources
        return [f"    {mnemonic} %zmm{right}, %zmm{left}, %zmm{dst}"]

    if len(sources) == 3:
        if mnemonic != "vpternlogq":
            raise FloorPlannerError(
                f"operation {op.id} has unsupported three-input opcode {mnemonic!r}"
            )
        if op.immediate is None:
            raise FloorPlannerError(f"operation {op.id} needs a ternary immediate")
        # GNU as exposes VPTERNLOG's destructive destination as the last
        # vector operand.  A separate copy would be an extra scheduled op, so
        # v1 requires the first logical input to share the destination register.
        if sources[0] != dst:
            raise FloorPlannerError(
                f"operation {op.id} requires a tied ternary destination in v1"
            )
        return [
            f"    vpternlogq ${op.immediate}, %zmm{sources[2]}, "
            f"%zmm{sources[1]}, %zmm{dst}"
        ]

    raise FloorPlannerError(
        f"operation {op.id} has {len(sources)} inputs; direct x86 v1 supports two or three"
    )
