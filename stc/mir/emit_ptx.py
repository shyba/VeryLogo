"""PTX code emission from Machine IR."""

from __future__ import annotations

from stc.mir import (
    Binary,
    Const,
    Copy,
    Load,
    MIRFunction,
    Mux,
    PReg,
    Store,
    Ternary,
    Unary,
    VReg,
)
from stc.sched.regalloc import RegAllocation


def emit_ptx(
    mir: MIRFunction,
    allocation: RegAllocation,
    function_name: str = "circuit",
    sm: str = "sm_61",
) -> str:
    """Emit PTX assembly from MIR.

    Args:
        mir: Machine IR representation
        allocation: Physical register allocation
        function_name: Name of generated kernel
        sm: Target compute capability

    Returns:
        PTX assembly code
    """
    lines: list[str] = []
    lines.append(f".version 6.0")
    lines.append(f".target {sm}")
    lines.append(f".address_size 64")
    lines.append("")
    lines.append(f".visible .entry {function_name}(")
    lines.append("    .param .u64 .ptr .align 4 in_ptr,")
    lines.append("    .param .u64 .ptr .align 4 out_ptr")
    lines.append(") {")

    num_physical_regs = max(allocation.reg_assignment.values(), default=0) + 1
    num_input_regs = len(mir.input_regs)
    max_reg = max(num_input_regs - 1, num_physical_regs - 1)

    if max_reg >= 0:
        lines.append(f"    .reg .b32 %r<{max_reg + 1}>;")

    lines.append("    .reg .u64 %in_addr;")
    lines.append("    .reg .u64 %out_addr;")

    num_spills = len(allocation.spills)
    if num_spills > 0:
        lines.append(f"    .reg .b32 %stack<{num_spills}>;")

    lines.append("")
    lines.append("    ld.param.u64 %in_addr, [in_ptr];")
    lines.append("    ld.param.u64 %out_addr, [out_ptr];")

    vreg_to_preg: dict[int, int] = {}
    for vreg_id, preg_id in allocation.reg_assignment.items():
        vreg_to_preg[vreg_id] = preg_id

    spill_slots: dict[int, int] = {}
    for i, node in enumerate(allocation.spills):
        spill_slots[node] = i

    for i, vreg in enumerate(mir.input_regs):
        offset = i * 4
        lines.append(f"    ld.global.u32 %r{i}, [%in_addr + {offset}];")

    for inst in mir.instructions:
        code = _emit_inst_ptx(inst, vreg_to_preg, spill_slots)
        if code:
            for line in code:
                lines.append(f"    {line}")

    for i, vreg in enumerate(mir.output_regs):
        offset = i * 4
        if isinstance(vreg, VReg):
            preg = vreg_to_preg.get(vreg.id, vreg.id)
            lines.append(f"    st.global.u32 [%out_addr + {offset}], %r{preg};")
        else:
            reg_name = _reg_name_ptx(vreg, vreg_to_preg)
            lines.append(f"    st.global.u32 [%out_addr + {offset}], {reg_name};")

    lines.append("    ret;")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _reg_name_ptx(reg: VReg | PReg, vreg_to_preg: dict[int, int]) -> str:
    """Get the physical register name for PTX."""
    if isinstance(reg, PReg):
        return reg.name
    preg = vreg_to_preg.get(reg.id, reg.id)
    return f"%r{preg}"


def _emit_inst_ptx(
    inst,
    vreg_to_preg: dict[int, int],
    spill_slots: dict[int, int],
) -> list[str]:
    """Emit a single MIR instruction as PTX assembly."""
    lines: list[str] = []

    if isinstance(inst, Binary):
        dst = _reg_name_ptx(inst.dst, vreg_to_preg)
        a = _reg_name_ptx(inst.a, vreg_to_preg)
        b = _reg_name_ptx(inst.b, vreg_to_preg)
        if inst.op == "and":
            lines.append(f"and.b32 {dst}, {a}, {b};")
        elif inst.op == "or":
            lines.append(f"or.b32 {dst}, {a}, {b};")
        elif inst.op == "xor":
            lines.append(f"xor.b32 {dst}, {a}, {b};")
        elif inst.op == "andn":
            # andn(a, b) == (~a) & b; encode as a single ternary op.
            lines.append(f"lop3.b32 {dst}, {a}, {b}, {a}, 12;")
        else:
            raise ValueError(f"Unsupported PTX binary op: {inst.op}")

    elif isinstance(inst, Unary):
        dst = _reg_name_ptx(inst.dst, vreg_to_preg)
        a = _reg_name_ptx(inst.a, vreg_to_preg)
        if inst.op == "not":
            lines.append(f"not.b32 {dst}, {a};")
        else:
            raise ValueError(f"Unsupported PTX unary op: {inst.op}")

    elif isinstance(inst, Ternary):
        dst = _reg_name_ptx(inst.dst, vreg_to_preg)
        a = _reg_name_ptx(inst.a, vreg_to_preg)
        b = _reg_name_ptx(inst.b, vreg_to_preg)
        c = _reg_name_ptx(inst.c, vreg_to_preg)
        lines.append(f"lop3.b32 {dst}, {a}, {b}, {c}, {inst.imm8};")

    elif isinstance(inst, Mux):
        dst = _reg_name_ptx(inst.dst, vreg_to_preg)
        sel = _reg_name_ptx(inst.select, vreg_to_preg)
        a = _reg_name_ptx(inst.a, vreg_to_preg)
        b = _reg_name_ptx(inst.b, vreg_to_preg)
        lines.append(f"lop3.b32 {dst}, {sel}, {a}, {b}, 0xCA;")

    elif isinstance(inst, Copy):
        dst = _reg_name_ptx(inst.dst, vreg_to_preg)
        src = _reg_name_ptx(inst.src, vreg_to_preg)
        if dst != src:
            lines.append(f"mov.b32 {dst}, {src};")

    elif isinstance(inst, Const):
        dst = _reg_name_ptx(inst.dst, vreg_to_preg)
        if inst.value == 0:
            lines.append(f"mov.b32 {dst}, 0;")
        else:
            lines.append(f"mov.b32 {dst}, 0xFFFFFFFF;")

    elif isinstance(inst, Load):
        dst = _reg_name_ptx(inst.dst, vreg_to_preg)
        lines.append(f"mov.b32 {dst}, %stack{inst.slot};")

    elif isinstance(inst, Store):
        src = _reg_name_ptx(inst.src, vreg_to_preg)
        lines.append(f"mov.b32 %stack{inst.slot}, {src};")

    return lines
