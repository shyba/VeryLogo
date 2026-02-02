"""
Complete SPIR-V Vulkan compute shader emitter from MIR.
"""

from __future__ import annotations

from stc.mir import MIRFunction, Binary, Unary, Ternary, Copy, Const, Load, Store, VReg
from stc.sched.regalloc import RegAllocation


def emit_vulkan_compute(
    mir: MIRFunction,
    allocation: RegAllocation,
    outputs=None,
    *,
    local_size: tuple[int, int, int] = (64, 1, 1),
) -> str:
    """
    Generate complete Vulkan compute shader from MIR.

    Args:
        mir: Machine IR function
        allocation: Register allocation
        outputs: List of (node_idx, inverted) tuples for output inversions
    """

    lines = []
    id_counter = 1

    def nid():
        nonlocal id_counter
        result = f"%{id_counter}"
        id_counter += 1
        return result

    # IDs for key types/vars
    main_id = nid()
    gid_var_id = nid()
    in_buf_id = nid()
    out_buf_id = nid()

    # Header
    lines.append("; SPIR-V")
    lines.append("; Version: 1.3")
    lines.append("; Generator: VeryLogo Vulkan Backend")
    lines.append("; Bound: 1000")
    lines.append("; Schema: 0")
    lines.append("")
    lines.append("OpCapability Shader")
    lines.append("OpMemoryModel Logical GLSL450")
    lines.append(f'OpEntryPoint GLCompute {main_id} "main" {gid_var_id}')
    lines.append(
        f"OpExecutionMode {main_id} LocalSize {local_size[0]} {local_size[1]} {local_size[2]}"
    )
    lines.append("")

    # Decorations
    arr_id = nid()
    in_struct_id = nid()
    out_struct_id = nid()

    lines.append(f"OpDecorate {gid_var_id} BuiltIn GlobalInvocationId")
    lines.append(f"OpDecorate {arr_id} ArrayStride 4")
    lines.append(f"OpMemberDecorate {in_struct_id} 0 Offset 0")
    lines.append(f"OpDecorate {in_struct_id} BufferBlock")
    lines.append(f"OpDecorate {in_buf_id} DescriptorSet 0")
    lines.append(f"OpDecorate {in_buf_id} Binding 0")
    lines.append(f"OpMemberDecorate {out_struct_id} 0 Offset 0")
    lines.append(f"OpDecorate {out_struct_id} BufferBlock")
    lines.append(f"OpDecorate {out_buf_id} DescriptorSet 0")
    lines.append(f"OpDecorate {out_buf_id} Binding 1")
    lines.append("")

    # Types
    void_id = nid()
    uint_id = nid()
    int_id = nid()
    v3uint_id = nid()
    ptr_in_v3uint = nid()
    ptr_unif_uint = nid()
    ptr_unif_in = nid()
    ptr_unif_out = nid()
    func_type = nid()

    lines.append(f"{void_id} = OpTypeVoid")
    lines.append(f"{uint_id} = OpTypeInt 32 0")
    lines.append(f"{int_id} = OpTypeInt 32 1")
    lines.append(f"{v3uint_id} = OpTypeVector {uint_id} 3")
    lines.append(f"{ptr_in_v3uint} = OpTypePointer Input {v3uint_id}")
    lines.append(f"{ptr_unif_uint} = OpTypePointer Uniform {uint_id}")
    lines.append(f"{arr_id} = OpTypeRuntimeArray {uint_id}")
    lines.append(f"{in_struct_id} = OpTypeStruct {arr_id}")
    lines.append(f"{out_struct_id} = OpTypeStruct {arr_id}")
    lines.append(f"{ptr_unif_in} = OpTypePointer Uniform {in_struct_id}")
    lines.append(f"{ptr_unif_out} = OpTypePointer Uniform {out_struct_id}")
    lines.append(f"{func_type} = OpTypeFunction {void_id}")
    lines.append("")

    # Constants
    const_0 = nid()
    const_8 = nid()
    int_0 = nid()
    const_ffff = nid()

    lines.append(f"{const_0} = OpConstant {uint_id} 0")
    lines.append(f"{const_8} = OpConstant {uint_id} 8")
    lines.append(f"{int_0} = OpConstant {int_id} 0")
    lines.append(f"{const_ffff} = OpConstant {uint_id} 4294967295")

    const_map = {0: const_0, 0xFFFFFFFF: const_ffff}
    for i in range(1, max(len(mir.input_regs), len(mir.output_regs))):
        const_map[i] = nid()
        lines.append(f"{const_map[i]} = OpConstant {uint_id} {i}")
    lines.append("")

    # Global variables
    lines.append(f"{gid_var_id} = OpVariable {ptr_in_v3uint} Input")
    lines.append(f"{in_buf_id} = OpVariable {ptr_unif_in} Uniform")
    lines.append(f"{out_buf_id} = OpVariable {ptr_unif_out} Uniform")
    lines.append("")

    # Main function
    lines.append(f"{main_id} = OpFunction {void_id} None {func_type}")
    entry_id = nid()
    lines.append(f"{entry_id} = OpLabel")
    lines.append("")

    # Get thread ID and base offset
    gid_val = nid()
    tid = nid()
    base_off = nid()
    lines.append(f"{gid_val} = OpLoad {v3uint_id} {gid_var_id}")
    lines.append(f"{tid} = OpCompositeExtract {uint_id} {gid_val} 0")
    lines.append(f"{base_off} = OpIMul {uint_id} {tid} {const_8}")
    lines.append("")

    # Load inputs and map to MIR input_regs
    reg_map = {}
    for i, reg in enumerate(mir.input_regs):
        off_id = nid()
        ptr_id = nid()
        val_id = nid()

        const_i = const_map.get(i, const_0)
        lines.append(f"{off_id} = OpIAdd {uint_id} {base_off} {const_i}")
        lines.append(
            f"{ptr_id} = OpAccessChain {ptr_unif_uint} {in_buf_id} {int_0} {off_id}"
        )
        lines.append(f"{val_id} = OpLoad {uint_id} {ptr_id}")

        reg_map[f"v{reg.id}"] = val_id

    lines.append("")

    def ternary_decompose(dst_id, a_id, b_id, c_id, imm8: int):
        # Build OR of minterms where imm8 has bit set.
        terms = []
        for i in range(8):
            if (imm8 >> i) & 1 == 0:
                continue
            a_sel = (i >> 2) & 1
            b_sel = (i >> 1) & 1
            c_sel = i & 1

            def sel(x_id, bit):
                if bit:
                    return x_id
                inv = nid()
                lines.append(f"{inv} = OpNot {uint_id} {x_id}")
                return inv

            a_term = sel(a_id, a_sel)
            b_term = sel(b_id, b_sel)
            c_term = sel(c_id, c_sel)
            t1 = nid()
            t2 = nid()
            lines.append(f"{t1} = OpBitwiseAnd {uint_id} {a_term} {b_term}")
            lines.append(f"{t2} = OpBitwiseAnd {uint_id} {t1} {c_term}")
            terms.append(t2)

        if not terms:
            lines.append(f"{dst_id} = OpCopyObject {uint_id} {const_0}")
            return
        acc = terms[0]
        for t in terms[1:]:
            tmp = nid()
            lines.append(f"{tmp} = OpBitwiseOr {uint_id} {acc} {t}")
            acc = tmp
        lines.append(f"{dst_id} = OpCopyObject {uint_id} {acc}")

    # Emit circuit operations
    for inst in mir.instructions:
        if isinstance(inst, Load):
            # Spill load - skip for now
            continue
        if isinstance(inst, Store):
            # Spill store - skip for now
            continue

        if inst.dst:
            dst_id = nid()
            reg_map[f"v{inst.dst.id}"] = dst_id

            if isinstance(inst, Binary):
                a_id = reg_map.get(f"v{inst.a.id}", const_0)
                b_id = reg_map.get(f"v{inst.b.id}", const_0)

                if inst.op == "and":
                    lines.append(f"{dst_id} = OpBitwiseAnd {uint_id} {a_id} {b_id}")
                elif inst.op == "xor":
                    lines.append(f"{dst_id} = OpBitwiseXor {uint_id} {a_id} {b_id}")
                elif inst.op == "or":
                    lines.append(f"{dst_id} = OpBitwiseOr {uint_id} {a_id} {b_id}")

            elif isinstance(inst, Ternary):
                a_id = reg_map.get(f"v{inst.a.id}", const_0)
                b_id = reg_map.get(f"v{inst.b.id}", const_0)
                c_id = reg_map.get(f"v{inst.c.id}", const_0)
                ternary_decompose(dst_id, a_id, b_id, c_id, inst.imm8)

            elif isinstance(inst, Unary):
                a_id = reg_map.get(f"v{inst.a.id}", const_0)
                if inst.op == "not":
                    lines.append(f"{dst_id} = OpNot {uint_id} {a_id}")

            elif isinstance(inst, Copy):
                src_id = reg_map.get(f"v{inst.src.id}", const_0)
                lines.append(f"{dst_id} = OpCopyObject {uint_id} {src_id}")

            elif isinstance(inst, Const):
                const_val_id = const_map.get(inst.value, const_0)
                lines.append(f"{dst_id} = OpCopyObject {uint_id} {const_val_id}")

    lines.append("")

    # Store outputs
    for i, reg in enumerate(mir.output_regs):
        val_id = reg_map.get(f"v{reg.id}", const_0)

        # Apply inversion if needed
        if outputs and i < len(outputs):
            _, inverted = outputs[i]
            if inverted:
                inverted_id = nid()
                lines.append(
                    f"{inverted_id} = OpBitwiseXor {uint_id} {val_id} {const_ffff}"
                )
                val_id = inverted_id

        off_id = nid()
        ptr_id = nid()

        const_i = const_map.get(i, const_0)
        lines.append(f"{off_id} = OpIAdd {uint_id} {base_off} {const_i}")
        lines.append(
            f"{ptr_id} = OpAccessChain {ptr_unif_uint} {out_buf_id} {int_0} {off_id}"
        )
        lines.append(f"OpStore {ptr_id} {val_id}")

    lines.append("")
    lines.append("OpReturn")
    lines.append("OpFunctionEnd")

    return "\n".join(lines)
