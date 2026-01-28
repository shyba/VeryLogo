"""AVX-512 code emission from Machine IR."""

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


def emit_avx512(
    mir: MIRFunction,
    allocation: RegAllocation,
    function_name: str = "circuit",
) -> str:
    """Emit AVX-512 C code from MIR.

    Args:
        mir: Machine IR representation
        allocation: Physical register allocation
        function_name: Name of generated function

    Returns:
        C code with AVX-512 intrinsics
    """
    lines: list[str] = []
    lines.append("#include <immintrin.h>")
    lines.append("")
    lines.append(f"void {function_name}(__m512i* in, __m512i* out) {{")

    num_input_regs = len(mir.input_regs)
    num_physical_regs = max(allocation.reg_assignment.values(), default=0) + 1
    max_reg = max(num_input_regs - 1, num_physical_regs - 1)

    for r in range(max_reg + 1):
        lines.append(f"    __m512i r{r};")

    needs_ones = _needs_ones(mir)
    if needs_ones:
        lines.append("    __m512i ones = _mm512_set1_epi32(-1);")

    num_spills = len(allocation.spills)
    for slot in range(num_spills):
        lines.append(f"    __m512i stack{slot};")

    vreg_to_preg: dict[int, int] = {}
    for vreg_id, preg_id in allocation.reg_assignment.items():
        vreg_to_preg[vreg_id] = preg_id

    spill_slots: dict[int, int] = {}
    for i, node in enumerate(allocation.spills):
        spill_slots[node] = i

    for i, vreg in enumerate(mir.input_regs):
        lines.append(f"    r{i} = in[{i}];")

    for inst in mir.instructions:
        code = _emit_inst(inst, vreg_to_preg, spill_slots)
        if code:
            lines.append(f"    {code}")

    for i, vreg in enumerate(mir.output_regs):
        if isinstance(vreg, VReg):
            preg = vreg_to_preg.get(vreg.id, vreg.id)
            lines.append(f"    out[{i}] = r{preg};")
        else:
            lines.append(f"    out[{i}] = {_reg_name(vreg, vreg_to_preg)};")

    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _needs_ones(mir: MIRFunction) -> bool:
    """Check if the ones constant is needed."""
    for inst in mir.instructions:
        if isinstance(inst, Unary) and inst.op == "not":
            return True
        if isinstance(inst, Const) and inst.value == 1:
            return True
    return False


def _reg_name(reg: VReg | PReg, vreg_to_preg: dict[int, int]) -> str:
    """Get the physical register name for a register."""
    if isinstance(reg, PReg):
        return reg.name
    preg = vreg_to_preg.get(reg.id, reg.id)
    return f"r{preg}"


def _emit_inst(
    inst,
    vreg_to_preg: dict[int, int],
    spill_slots: dict[int, int],
) -> str:
    """Emit a single MIR instruction as AVX-512 C code."""
    if isinstance(inst, Binary):
        dst = _reg_name(inst.dst, vreg_to_preg)
        a = _reg_name(inst.a, vreg_to_preg)
        b = _reg_name(inst.b, vreg_to_preg)
        if inst.op == "and":
            return f"{dst} = _mm512_and_si512({a}, {b});"
        elif inst.op == "or":
            return f"{dst} = _mm512_or_si512({a}, {b});"
        elif inst.op == "xor":
            return f"{dst} = _mm512_xor_si512({a}, {b});"
        elif inst.op == "andn":
            return f"{dst} = _mm512_andnot_si512({a}, {b});"
        else:
            return f"{dst} = _mm512_setzero_si512();"

    elif isinstance(inst, Unary):
        dst = _reg_name(inst.dst, vreg_to_preg)
        a = _reg_name(inst.a, vreg_to_preg)
        if inst.op == "not":
            return f"{dst} = _mm512_xor_si512({a}, ones);"
        else:
            return f"{dst} = {a};"

    elif isinstance(inst, Ternary):
        dst = _reg_name(inst.dst, vreg_to_preg)
        a = _reg_name(inst.a, vreg_to_preg)
        b = _reg_name(inst.b, vreg_to_preg)
        c = _reg_name(inst.c, vreg_to_preg)
        return f"{dst} = _mm512_ternarylogic_epi32({a}, {b}, {c}, {inst.imm8});"

    elif isinstance(inst, Mux):
        dst = _reg_name(inst.dst, vreg_to_preg)
        sel = _reg_name(inst.select, vreg_to_preg)
        a = _reg_name(inst.a, vreg_to_preg)
        b = _reg_name(inst.b, vreg_to_preg)
        return f"{dst} = _mm512_ternarylogic_epi32({sel}, {a}, {b}, 0xCA);"

    elif isinstance(inst, Copy):
        dst = _reg_name(inst.dst, vreg_to_preg)
        src = _reg_name(inst.src, vreg_to_preg)
        return f"{dst} = {src};"

    elif isinstance(inst, Const):
        dst = _reg_name(inst.dst, vreg_to_preg)
        if inst.value == 0:
            return f"{dst} = _mm512_setzero_si512();"
        else:
            return f"{dst} = ones;"

    elif isinstance(inst, Load):
        dst = _reg_name(inst.dst, vreg_to_preg)
        return f"{dst} = stack{inst.slot};"

    elif isinstance(inst, Store):
        src = _reg_name(inst.src, vreg_to_preg)
        return f"stack{inst.slot} = {src};"

    return ""
