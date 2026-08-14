"""Direct GNU-as x86-64 emission for the floor-planner v2 IR.

This emitter intentionally returns assembly text rather than C or intrinsics.
The system assembler is the only tool needed to turn the result into an
object; GCC is not part of this path.  The emitter validates concrete
encodings, ISA requirements, and optional memory placements before emitting
non-destructive integer bitwise forms and destructive ``VPTERNLOG`` forms.
"""

from __future__ import annotations

from stc.sched.cpu_model import CpuModelError, InstructionEncoding
from stc.sched.floor_planner import (
    FloorOp,
    FloorPlacement,
    FloorPlannerError,
    FloorProgram,
    FloorSchedule,
)
from stc.sched.x86_encodings import x86_encoding, x86_memory_encoding


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

    load_placements = tuple(
        placement
        for placement in schedule.memory_placements
        if placement.kind == "load"
    )
    if load_placements:
        load_accesses = sorted(
            load_placements, key=lambda placement: (placement.cycle, placement.slot)
        )
        for placement in load_accesses:
            if placement.value not in assignment:
                continue
            encoding = _validate_x86_memory_encoding("load", placement, schedule)
            lines.append(f"    # floor load cycle {placement.cycle}")
            lines.append(
                f"    {encoding.mnemonic} {placement.slot * 64}(%rdi), "
                f"%zmm{_reg(assignment, placement.value)}"
            )
    else:
        encoding = _validate_x86_memory_encoding("load", None, schedule)
        for input_index, value in enumerate(program.inputs):
            if value not in assignment:
                continue
            lines.append("    # floor load cycle 0")
            lines.append(
                f"    {encoding.mnemonic} {input_index * 64}(%rdi), "
                f"%zmm{_reg(assignment, value)}"
            )

    operations = sorted(
        program.operations,
        key=lambda op: (schedule.placement(op.id).cycle, op.id),
    )
    for op in operations:
        placement = schedule.placement(op.id)
        encoding = _validate_x86_encoding(op, placement, schedule)
        dst = _reg(assignment, op.output)
        sources = [_reg(assignment, value) for value in op.inputs]
        lines.append(f"    # floor core cycle {placement.cycle}")
        lines.extend(_emit_operation(op, dst, sources, encoding.mnemonic))

    store_placements = tuple(
        placement
        for placement in schedule.memory_placements
        if placement.kind == "store"
    )
    if store_placements:
        store_accesses = sorted(
            store_placements, key=lambda placement: (placement.cycle, placement.slot)
        )
        for placement in store_accesses:
            output_index = placement.slot
            value = placement.value
            encoding = _validate_x86_memory_encoding("store", placement, schedule)
            lines.append(f"    # floor store cycle {placement.cycle}")
            lines.append(
                f"    {encoding.mnemonic} %zmm{_reg(assignment, value)}, "
                f"{output_index * 64}(%rsi)"
            )
    else:
        encoding = _validate_x86_memory_encoding("store", None, schedule)
        for output_index, value in enumerate(program.outputs):
            lines.append(f"    # floor store cycle {schedule.total_cycles}")
            lines.append(
                f"    {encoding.mnemonic} %zmm{_reg(assignment, value)}, "
                f"{output_index * 64}(%rsi)"
            )

    lines.extend(
        [
            "    vzeroupper",
            "    ret",
            f".size {function_name}, .-{function_name}",
            '.section .note.GNU-stack,"",@progbits',
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


def _validate_x86_encoding(
    op: FloorOp, placement: FloorPlacement, schedule: FloorSchedule
) -> InstructionEncoding:
    """Validate the concrete opcode against the scheduled machine form."""

    form = placement.form
    requested = placement.encoding
    mnemonic = op.asm_mnemonic or (
        requested.mnemonic if requested is not None else None
    )
    if mnemonic is None:
        mnemonic = _DEFAULT_MNEMONICS.get(op.family)
    if mnemonic is None:
        raise FloorPlannerError(
            f"operation {op.id} ({op.family}) has no direct x86-64 encoding"
        )
    try:
        concrete = x86_encoding(mnemonic)
    except CpuModelError as exc:
        raise FloorPlannerError(str(exc)) from exc
    if concrete.family.casefold() != form.family.casefold():
        raise FloorPlannerError(
            f"operation {op.id} emits {concrete.mnemonic}, but the scheduled form "
            f"is {form.family}"
        )
    if concrete.operands is not None and _normalize_operands(
        concrete.operands
    ) != _normalize_operands(form.operands):
        raise FloorPlannerError(
            f"operation {op.id} emits {concrete.mnemonic}, but its operand form "
            f"is {form.operands}"
        )
    if concrete.source_count != len(op.inputs):
        raise FloorPlannerError(
            f"operation {op.id} emits {concrete.mnemonic} with the wrong source count"
        )
    if concrete.tied_input != op.tied_input:
        raise FloorPlannerError(
            f"operation {op.id} emits {concrete.mnemonic} with the wrong tied input"
        )
    if concrete.requires_immediate != (op.immediate is not None):
        raise FloorPlannerError(
            f"operation {op.id} emits {concrete.mnemonic} with the wrong immediate contract"
        )
    if requested is not None and requested != concrete:
        raise FloorPlannerError(
            f"operation {op.id} placement encoding does not match emitted {concrete.mnemonic}"
        )
    missing = sorted(
        feature
        for feature in concrete.required_features
        if feature not in schedule.cpu_features
    )
    if missing:
        raise FloorPlannerError(
            f"operation {op.id} emits {concrete.mnemonic} without ISA features: "
            f"{', '.join(missing)}"
        )
    return concrete


def _normalize_operands(value: str) -> str:
    return ",".join(part.strip() for part in value.split(","))


def _validate_x86_memory_encoding(
    kind: str, placement: object, schedule: FloorSchedule
) -> InstructionEncoding:
    concrete = x86_memory_encoding(kind)
    requested = getattr(placement, "encoding", None)
    if requested is not None and requested != concrete:
        raise FloorPlannerError(
            f"memory {kind} placement encoding does not match {concrete.mnemonic}"
        )
    form = getattr(placement, "form", None)
    if form is not None:
        if form.family.casefold() != concrete.family.casefold():
            raise FloorPlannerError(
                f"memory {kind} emits {concrete.mnemonic}, but the scheduled form "
                f"is {form.family}"
            )
        if form.operands != concrete.operands:
            raise FloorPlannerError(
                f"memory {kind} emits {concrete.mnemonic} with the wrong operands"
            )
    missing = sorted(
        feature
        for feature in concrete.required_features
        if feature not in schedule.cpu_features
    )
    if missing:
        raise FloorPlannerError(
            f"memory {kind} emits {concrete.mnemonic} without ISA features: "
            f"{', '.join(missing)}"
        )
    return concrete


def _emit_operation(
    op: FloorOp, dst: int, sources: list[int], mnemonic: str
) -> list[str]:
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
        # v2 requires the first logical input to share the destination register.
        if op.tied_input != 0 or sources[0] != dst:
            raise FloorPlannerError(
                f"operation {op.id} requires a tied ternary destination in v2"
            )
        return [
            f"    vpternlogq ${op.immediate}, %zmm{sources[2]}, "
            f"%zmm{sources[1]}, %zmm{dst}"
        ]

    raise FloorPlannerError(
        f"operation {op.id} has {len(sources)} inputs; direct x86 v2 supports two or three"
    )
