"""Complete SPIR-V Vulkan compute shader emitter from MIR."""

from __future__ import annotations

from stc.mir import MIRFunction, Binary, Unary, Ternary, Copy, Const, Load, Store
from stc.sched.regalloc import RegAllocation


def emit_vulkan_compute(
    mir: MIRFunction,
    allocation: RegAllocation,
    outputs=None,
    *,
    local_size: tuple[int, int, int] = (64, 1, 1),
    input_stride: int | None = None,
    output_stride: int | None = None,
    output_base_offset: int = 0,
    tid_linear_mode: str = "full",
    step_count: int = 1,
    loop_state_words: int = 0,
    loop_state_input_offset: int = 0,
    loop_state_output_offset: int = 0,
    loop_output_state_only: bool = False,
) -> str:
    """Generate complete Vulkan compute shader from MIR."""

    lines: list[str] = []
    id_counter = 1

    def nid() -> str:
        nonlocal id_counter
        result = f"%{id_counter}"
        id_counter += 1
        return result

    if tid_linear_mode not in {"full", "xy", "xonly"}:
        raise ValueError(
            f"invalid tid_linear_mode={tid_linear_mode!r}, expected one of: "
            "'full', 'xy', 'xonly'"
        )
    if output_base_offset < 0:
        raise ValueError("output_base_offset must be >= 0")
    if tid_linear_mode == "xonly" and (local_size[1] != 1 or local_size[2] != 1):
        raise ValueError(
            "tid_linear_mode='xonly' requires LocalSize Y=1 and Z=1 "
            f"(got {local_size})"
        )
    if tid_linear_mode == "xy" and local_size[2] != 1:
        raise ValueError(
            "tid_linear_mode='xy' requires LocalSize Z=1 " f"(got {local_size})"
        )
    if step_count < 1:
        raise ValueError("step_count must be >= 1")

    loop_enabled = step_count > 1 and loop_state_words > 0
    if loop_enabled:
        if loop_state_input_offset < 0 or loop_state_output_offset < 0:
            raise ValueError("loop state offsets must be >= 0")
        if loop_state_input_offset + loop_state_words > len(mir.input_regs):
            raise ValueError("loop state input range exceeds MIR inputs")
        if loop_state_output_offset + loop_state_words > len(mir.output_regs):
            raise ValueError("loop state output range exceeds MIR outputs")

    main_id = nid()
    gid_var_id = nid()
    num_groups_var_id = nid()
    in_buf_id = nid()
    out_buf_id = nid()

    lines.append("; SPIR-V")
    lines.append("; Version: 1.3")
    lines.append("; Generator: VeryLogo Vulkan Backend")
    lines.append(f"; TID linear mode: {tid_linear_mode}")
    if loop_enabled:
        lines.append(
            f"; Step loop enabled: steps={step_count} state_words={loop_state_words}"
        )
    bound_idx = len(lines)
    lines.append("; Bound: 1000")
    lines.append("; Schema: 0")
    lines.append("")
    lines.append("OpCapability Shader")
    lines.append('OpExtension "SPV_KHR_storage_buffer_storage_class"')
    lines.append("OpMemoryModel Logical GLSL450")
    lines.append(
        f'OpEntryPoint GLCompute {main_id} "main" {gid_var_id} {num_groups_var_id}'
    )
    lines.append(
        f"OpExecutionMode {main_id} LocalSize {local_size[0]} {local_size[1]} {local_size[2]}"
    )
    lines.append("")

    arr_id = nid()
    in_struct_id = nid()
    out_struct_id = nid()

    lines.append(f"OpDecorate {gid_var_id} BuiltIn GlobalInvocationId")
    lines.append(f"OpDecorate {num_groups_var_id} BuiltIn NumWorkgroups")
    lines.append(f"OpDecorate {arr_id} ArrayStride 4")
    lines.append(f"OpMemberDecorate {in_struct_id} 0 Offset 0")
    lines.append(f"OpDecorate {in_struct_id} Block")
    lines.append(f"OpDecorate {in_buf_id} DescriptorSet 0")
    lines.append(f"OpDecorate {in_buf_id} Binding 0")
    lines.append(f"OpMemberDecorate {out_struct_id} 0 Offset 0")
    lines.append(f"OpDecorate {out_struct_id} Block")
    lines.append(f"OpDecorate {out_buf_id} DescriptorSet 0")
    lines.append(f"OpDecorate {out_buf_id} Binding 1")
    lines.append("")

    void_id = nid()
    bool_id = nid()
    uint_id = nid()
    int_id = nid()
    v3uint_id = nid()
    ptr_in_v3uint = nid()
    ptr_storage_uint = nid()
    ptr_storage_in = nid()
    ptr_storage_out = nid()
    ptr_func_uint = nid()
    func_type = nid()

    lines.append(f"{void_id} = OpTypeVoid")
    lines.append(f"{bool_id} = OpTypeBool")
    lines.append(f"{uint_id} = OpTypeInt 32 0")
    lines.append(f"{int_id} = OpTypeInt 32 1")
    lines.append(f"{v3uint_id} = OpTypeVector {uint_id} 3")
    lines.append(f"{ptr_in_v3uint} = OpTypePointer Input {v3uint_id}")
    lines.append(f"{ptr_storage_uint} = OpTypePointer StorageBuffer {uint_id}")
    lines.append(f"{arr_id} = OpTypeRuntimeArray {uint_id}")
    lines.append(f"{in_struct_id} = OpTypeStruct {arr_id}")
    lines.append(f"{out_struct_id} = OpTypeStruct {arr_id}")
    lines.append(f"{ptr_storage_in} = OpTypePointer StorageBuffer {in_struct_id}")
    lines.append(f"{ptr_storage_out} = OpTypePointer StorageBuffer {out_struct_id}")
    lines.append(f"{ptr_func_uint} = OpTypePointer Function {uint_id}")
    lines.append(f"{func_type} = OpTypeFunction {void_id}")
    lines.append("")

    input_stride_val = len(mir.input_regs) if input_stride is None else input_stride
    output_stride_val = len(mir.output_regs) if output_stride is None else output_stride
    input_stride_val = max(input_stride_val, 1)
    output_stride_val = max(output_stride_val, 1)

    const_0 = nid()
    int_0 = nid()
    const_ffff = nid()
    lines.append(f"{const_0} = OpConstant {uint_id} 0")
    lines.append(f"{int_0} = OpConstant {int_id} 0")
    lines.append(f"{const_ffff} = OpConstant {uint_id} 4294967295")

    max_const_index = max(
        len(mir.input_regs), len(mir.output_regs), loop_state_words, 2
    )
    const_map = {0: const_0, 0xFFFFFFFF: const_ffff}
    for i in range(1, max_const_index):
        const_map[i] = nid()
        lines.append(f"{const_map[i]} = OpConstant {uint_id} {i}")
    for value in (
        input_stride_val,
        output_stride_val,
        output_base_offset,
        local_size[0],
        local_size[1],
        local_size[2],
        step_count,
    ):
        if value not in const_map:
            const_map[value] = nid()
            lines.append(f"{const_map[value]} = OpConstant {uint_id} {value}")
    lines.append("")

    lines.append(f"{gid_var_id} = OpVariable {ptr_in_v3uint} Input")
    lines.append(f"{num_groups_var_id} = OpVariable {ptr_in_v3uint} Input")
    lines.append(f"{in_buf_id} = OpVariable {ptr_storage_in} StorageBuffer")
    lines.append(f"{out_buf_id} = OpVariable {ptr_storage_out} StorageBuffer")
    lines.append("")

    lines.append(f"{main_id} = OpFunction {void_id} None {func_type}")
    entry_id = nid()
    lines.append(f"{entry_id} = OpLabel")
    lines.append("")

    spill_slots = sorted(
        {inst.slot for inst in mir.instructions if isinstance(inst, (Load, Store))}
    )
    spill_ptrs: dict[int, str] = {}
    for slot in spill_slots:
        slot_ptr = nid()
        spill_ptrs[slot] = slot_ptr
        lines.append(f"{slot_ptr} = OpVariable {ptr_func_uint} Function")
    state_ptrs: list[str] = []
    loop_out_ptrs: list[str] = []
    counter_ptr: str | None = None
    if loop_enabled:
        for _ in range(loop_state_words):
            ptr = nid()
            state_ptrs.append(ptr)
            lines.append(f"{ptr} = OpVariable {ptr_func_uint} Function")
        if not loop_output_state_only:
            for _ in mir.output_regs:
                ptr = nid()
                loop_out_ptrs.append(ptr)
                lines.append(f"{ptr} = OpVariable {ptr_func_uint} Function")
        counter_ptr = nid()
        lines.append(f"{counter_ptr} = OpVariable {ptr_func_uint} Function")
    if spill_slots or loop_enabled:
        lines.append("")

    gid_val = nid()
    local_x = const_map[local_size[0]]
    local_y = const_map[local_size[1]]
    tid_sum1 = nid()
    base_in = nid()
    base_out = nid()

    lines.append(f"{gid_val} = OpLoad {v3uint_id} {gid_var_id}")
    tid_x = nid()
    lines.append(f"{tid_x} = OpCompositeExtract {uint_id} {gid_val} 0")
    if tid_linear_mode == "xonly":
        lines.append(f"{tid_sum1} = OpCopyObject {uint_id} {tid_x}")
    elif tid_linear_mode == "xy":
        num_groups_val = nid()
        tid_y = nid()
        num_x = nid()
        grid_x = nid()
        tid_y_mul = nid()
        lines.append(f"{num_groups_val} = OpLoad {v3uint_id} {num_groups_var_id}")
        lines.append(f"{tid_y} = OpCompositeExtract {uint_id} {gid_val} 1")
        lines.append(f"{num_x} = OpCompositeExtract {uint_id} {num_groups_val} 0")
        lines.append(f"{grid_x} = OpIMul {uint_id} {num_x} {local_x}")
        lines.append(f"{tid_y_mul} = OpIMul {uint_id} {tid_y} {grid_x}")
        lines.append(f"{tid_sum1} = OpIAdd {uint_id} {tid_x} {tid_y_mul}")
    else:
        num_groups_val = nid()
        tid_y = nid()
        tid_z = nid()
        num_x = nid()
        num_y = nid()
        grid_x = nid()
        grid_y = nid()
        grid_xy = nid()
        tid_y_mul = nid()
        tid_z_mul = nid()
        tid_sum0 = nid()
        lines.append(f"{num_groups_val} = OpLoad {v3uint_id} {num_groups_var_id}")
        lines.append(f"{tid_y} = OpCompositeExtract {uint_id} {gid_val} 1")
        lines.append(f"{tid_z} = OpCompositeExtract {uint_id} {gid_val} 2")
        lines.append(f"{num_x} = OpCompositeExtract {uint_id} {num_groups_val} 0")
        lines.append(f"{num_y} = OpCompositeExtract {uint_id} {num_groups_val} 1")
        lines.append(f"{grid_x} = OpIMul {uint_id} {num_x} {local_x}")
        lines.append(f"{grid_y} = OpIMul {uint_id} {num_y} {local_y}")
        lines.append(f"{grid_xy} = OpIMul {uint_id} {grid_x} {grid_y}")
        lines.append(f"{tid_y_mul} = OpIMul {uint_id} {tid_y} {grid_x}")
        lines.append(f"{tid_z_mul} = OpIMul {uint_id} {tid_z} {grid_xy}")
        lines.append(f"{tid_sum0} = OpIAdd {uint_id} {tid_x} {tid_y_mul}")
        lines.append(f"{tid_sum1} = OpIAdd {uint_id} {tid_sum0} {tid_z_mul}")
    lines.append(
        f"{base_in} = OpIMul {uint_id} {tid_sum1} {const_map[input_stride_val]}"
    )
    lines.append(
        f"{base_out} = OpIMul {uint_id} {tid_sum1} {const_map[output_stride_val]}"
    )
    lines.append("")

    def emit_load_input_word(index: int) -> str:
        off_id = nid()
        ptr_id = nid()
        val_id = nid()
        const_i = const_map.get(index, const_0)
        lines.append(f"{off_id} = OpIAdd {uint_id} {base_in} {const_i}")
        lines.append(
            f"{ptr_id} = OpAccessChain {ptr_storage_uint} {in_buf_id} {int_0} {off_id}"
        )
        lines.append(f"{val_id} = OpLoad {uint_id} {ptr_id}")
        return val_id

    base_reg_map: dict[str, str] = {}
    for i, reg in enumerate(mir.input_regs):
        base_reg_map[f"v{reg.id}"] = emit_load_input_word(i)
    lines.append("")

    def _imm8(func) -> int:
        out = 0
        for i in range(8):
            a = (i >> 2) & 1
            b = (i >> 1) & 1
            c = i & 1
            if func(a, b, c):
                out |= 1 << i
        return out

    IMM_A = _imm8(lambda a, b, c: a)
    IMM_B = _imm8(lambda a, b, c: b)
    IMM_C = _imm8(lambda a, b, c: c)
    IMM_NOT_A = _imm8(lambda a, b, c: 1 - a)
    IMM_NOT_B = _imm8(lambda a, b, c: 1 - b)
    IMM_NOT_C = _imm8(lambda a, b, c: 1 - c)
    IMM_AND_AB = _imm8(lambda a, b, c: a & b)
    IMM_AND_AC = _imm8(lambda a, b, c: a & c)
    IMM_AND_BC = _imm8(lambda a, b, c: b & c)
    IMM_OR_AB = _imm8(lambda a, b, c: a | b)
    IMM_OR_AC = _imm8(lambda a, b, c: a | c)
    IMM_OR_BC = _imm8(lambda a, b, c: b | c)
    IMM_XOR_AB = _imm8(lambda a, b, c: a ^ b)
    IMM_XOR_AC = _imm8(lambda a, b, c: a ^ c)
    IMM_XOR_BC = _imm8(lambda a, b, c: b ^ c)
    IMM_XOR_ABC = _imm8(lambda a, b, c: a ^ b ^ c)
    IMM_MAJ = _imm8(lambda a, b, c: (a & b) | (a & c) | (b & c))
    IMM_MUX_A_B_C = _imm8(lambda a, b, c: (a & b) | ((1 - a) & c))
    IMM_MUX_B_A_C = _imm8(lambda a, b, c: (b & a) | ((1 - b) & c))
    IMM_MUX_C_A_B = _imm8(lambda a, b, c: (c & a) | ((1 - c) & b))

    def ternary_decompose(
        dst_id: str, a_id: str, b_id: str, c_id: str, imm8: int
    ) -> None:
        if imm8 == 0:
            lines.append(f"{dst_id} = OpCopyObject {uint_id} {const_0}")
            return
        if imm8 == 0xFF:
            lines.append(f"{dst_id} = OpCopyObject {uint_id} {const_ffff}")
            return
        if imm8 == IMM_A:
            lines.append(f"{dst_id} = OpCopyObject {uint_id} {a_id}")
            return
        if imm8 == IMM_B:
            lines.append(f"{dst_id} = OpCopyObject {uint_id} {b_id}")
            return
        if imm8 == IMM_C:
            lines.append(f"{dst_id} = OpCopyObject {uint_id} {c_id}")
            return
        if imm8 == IMM_NOT_A:
            lines.append(f"{dst_id} = OpNot {uint_id} {a_id}")
            return
        if imm8 == IMM_NOT_B:
            lines.append(f"{dst_id} = OpNot {uint_id} {b_id}")
            return
        if imm8 == IMM_NOT_C:
            lines.append(f"{dst_id} = OpNot {uint_id} {c_id}")
            return
        if imm8 == IMM_AND_AB:
            lines.append(f"{dst_id} = OpBitwiseAnd {uint_id} {a_id} {b_id}")
            return
        if imm8 == IMM_AND_AC:
            lines.append(f"{dst_id} = OpBitwiseAnd {uint_id} {a_id} {c_id}")
            return
        if imm8 == IMM_AND_BC:
            lines.append(f"{dst_id} = OpBitwiseAnd {uint_id} {b_id} {c_id}")
            return
        if imm8 == IMM_OR_AB:
            lines.append(f"{dst_id} = OpBitwiseOr {uint_id} {a_id} {b_id}")
            return
        if imm8 == IMM_OR_AC:
            lines.append(f"{dst_id} = OpBitwiseOr {uint_id} {a_id} {c_id}")
            return
        if imm8 == IMM_OR_BC:
            lines.append(f"{dst_id} = OpBitwiseOr {uint_id} {b_id} {c_id}")
            return
        if imm8 == IMM_XOR_AB:
            lines.append(f"{dst_id} = OpBitwiseXor {uint_id} {a_id} {b_id}")
            return
        if imm8 == IMM_XOR_AC:
            lines.append(f"{dst_id} = OpBitwiseXor {uint_id} {a_id} {c_id}")
            return
        if imm8 == IMM_XOR_BC:
            lines.append(f"{dst_id} = OpBitwiseXor {uint_id} {b_id} {c_id}")
            return
        if imm8 == IMM_XOR_ABC:
            t0 = nid()
            lines.append(f"{t0} = OpBitwiseXor {uint_id} {a_id} {b_id}")
            lines.append(f"{dst_id} = OpBitwiseXor {uint_id} {t0} {c_id}")
            return
        if imm8 == IMM_MAJ:
            t0 = nid()
            t1 = nid()
            t2 = nid()
            t3 = nid()
            lines.append(f"{t0} = OpBitwiseAnd {uint_id} {a_id} {b_id}")
            lines.append(f"{t1} = OpBitwiseAnd {uint_id} {a_id} {c_id}")
            lines.append(f"{t2} = OpBitwiseAnd {uint_id} {b_id} {c_id}")
            lines.append(f"{t3} = OpBitwiseOr {uint_id} {t0} {t1}")
            lines.append(f"{dst_id} = OpBitwiseOr {uint_id} {t3} {t2}")
            return
        if imm8 in (IMM_MUX_A_B_C, IMM_MUX_B_A_C, IMM_MUX_C_A_B):
            if imm8 == IMM_MUX_A_B_C:
                sel_id, t_id, f_id = a_id, b_id, c_id
            elif imm8 == IMM_MUX_B_A_C:
                sel_id, t_id, f_id = b_id, a_id, c_id
            else:
                sel_id, t_id, f_id = c_id, a_id, b_id
            nsel = nid()
            t0 = nid()
            t1 = nid()
            lines.append(f"{nsel} = OpNot {uint_id} {sel_id}")
            lines.append(f"{t0} = OpBitwiseAnd {uint_id} {sel_id} {t_id}")
            lines.append(f"{t1} = OpBitwiseAnd {uint_id} {nsel} {f_id}")
            lines.append(f"{dst_id} = OpBitwiseOr {uint_id} {t0} {t1}")
            return

        terms: list[str] = []
        for i in range(8):
            if ((imm8 >> i) & 1) == 0:
                continue
            a_sel = (i >> 2) & 1
            b_sel = (i >> 1) & 1
            c_sel = i & 1

            def sel(x_id: str, bit: int) -> str:
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

    def apply_output_inversion(value_id: str, out_idx: int) -> str:
        if outputs and out_idx < len(outputs):
            _, inverted = outputs[out_idx]
            if inverted:
                inverted_id = nid()
                lines.append(
                    f"{inverted_id} = OpBitwiseXor {uint_id} {value_id} {const_ffff}"
                )
                return inverted_id
        return value_id

    def emit_store_output_word(value_id: str, out_idx: int) -> None:
        off_id = nid()
        ptr_id = nid()
        rel_off = nid()
        const_i = const_map.get(out_idx, const_0)
        lines.append(
            f"{rel_off} = OpIAdd {uint_id} {const_i} {const_map[output_base_offset]}"
        )
        lines.append(f"{off_id} = OpIAdd {uint_id} {base_out} {rel_off}")
        lines.append(
            f"{ptr_id} = OpAccessChain {ptr_storage_uint} {out_buf_id} {int_0} {off_id}"
        )
        lines.append(f"OpStore {ptr_id} {value_id}")

    def emit_load_output_word(out_idx: int) -> str:
        off_id = nid()
        ptr_id = nid()
        val_id = nid()
        rel_off = nid()
        const_i = const_map.get(out_idx, const_0)
        lines.append(
            f"{rel_off} = OpIAdd {uint_id} {const_i} {const_map[output_base_offset]}"
        )
        lines.append(f"{off_id} = OpIAdd {uint_id} {base_out} {rel_off}")
        lines.append(
            f"{ptr_id} = OpAccessChain {ptr_storage_uint} {out_buf_id} {int_0} {off_id}"
        )
        lines.append(f"{val_id} = OpLoad {uint_id} {ptr_id}")
        return val_id

    def emit_mir_body(start_reg_map: dict[str, str]) -> dict[str, str]:
        reg_map = dict(start_reg_map)
        for inst in mir.instructions:
            if isinstance(inst, Load):
                slot_ptr = spill_ptrs.get(inst.slot)
                dst_id = nid()
                reg_map[f"v{inst.dst.id}"] = dst_id
                if slot_ptr is None:
                    lines.append(f"{dst_id} = OpCopyObject {uint_id} {const_0}")
                else:
                    lines.append(f"{dst_id} = OpLoad {uint_id} {slot_ptr}")
                continue
            if isinstance(inst, Store):
                slot_ptr = spill_ptrs.get(inst.slot)
                if slot_ptr is not None:
                    src_id = reg_map.get(f"v{inst.src.id}", const_0)
                    lines.append(f"OpStore {slot_ptr} {src_id}")
                continue
            if not inst.dst:
                continue

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
                elif inst.op == "andn":
                    inv_id = nid()
                    lines.append(f"{inv_id} = OpNot {uint_id} {a_id}")
                    lines.append(f"{dst_id} = OpBitwiseAnd {uint_id} {inv_id} {b_id}")
                else:
                    raise ValueError(f"unsupported MIR binary op for SPIR-V: {inst.op}")
            elif isinstance(inst, Ternary):
                a_id = reg_map.get(f"v{inst.a.id}", const_0)
                b_id = reg_map.get(f"v{inst.b.id}", const_0)
                c_id = reg_map.get(f"v{inst.c.id}", const_0)
                ternary_decompose(dst_id, a_id, b_id, c_id, inst.imm8)
            elif isinstance(inst, Unary):
                a_id = reg_map.get(f"v{inst.a.id}", const_0)
                if inst.op == "not":
                    lines.append(f"{dst_id} = OpNot {uint_id} {a_id}")
                else:
                    raise ValueError(f"unsupported MIR unary op for SPIR-V: {inst.op}")
            elif isinstance(inst, Copy):
                src_id = reg_map.get(f"v{inst.src.id}", const_0)
                lines.append(f"{dst_id} = OpCopyObject {uint_id} {src_id}")
            elif isinstance(inst, Const):
                if inst.value == 1:
                    const_val_id = const_ffff
                elif inst.value == 0:
                    const_val_id = const_0
                else:
                    const_val_id = const_map.get(inst.value, const_0)
                lines.append(f"{dst_id} = OpCopyObject {uint_id} {const_val_id}")
        return reg_map

    if not loop_enabled:
        reg_map = emit_mir_body(base_reg_map)
        lines.append("")
        for i, reg in enumerate(mir.output_regs):
            val_id = reg_map.get(f"v{reg.id}", const_0)
            val_id = apply_output_inversion(val_id, i)
            emit_store_output_word(val_id, i)
    else:
        assert counter_ptr is not None

        for i in range(loop_state_words):
            input_idx = loop_state_input_offset + i
            input_reg = mir.input_regs[input_idx]
            state_init = base_reg_map.get(f"v{input_reg.id}", const_0)
            lines.append(f"OpStore {state_ptrs[i]} {state_init}")
        lines.append(f"OpStore {counter_ptr} {const_0}")
        lines.append("")

        loop_header = nid()
        loop_body = nid()
        loop_continue = nid()
        loop_merge = nid()

        lines.append(f"OpBranch {loop_header}")
        lines.append(f"{loop_header} = OpLabel")
        ctr_val = nid()
        cond = nid()
        lines.append(f"{ctr_val} = OpLoad {uint_id} {counter_ptr}")
        lines.append(
            f"{cond} = OpULessThan {bool_id} {ctr_val} {const_map[step_count]}"
        )
        lines.append(f"OpLoopMerge {loop_merge} {loop_continue} None")
        lines.append(f"OpBranchConditional {cond} {loop_body} {loop_merge}")
        lines.append(f"{loop_body} = OpLabel")

        step_reg_map = dict(base_reg_map)
        for i in range(loop_state_words):
            load_id = nid()
            lines.append(f"{load_id} = OpLoad {uint_id} {state_ptrs[i]}")
            input_idx = loop_state_input_offset + i
            input_reg = mir.input_regs[input_idx]
            step_reg_map[f"v{input_reg.id}"] = load_id

        step_reg_map = emit_mir_body(step_reg_map)
        lines.append("")

        if loop_output_state_only:
            for i in range(loop_state_words):
                out_idx = loop_state_output_offset + i
                out_reg = mir.output_regs[out_idx]
                val_id = step_reg_map.get(f"v{out_reg.id}", const_0)
                val_id = apply_output_inversion(val_id, out_idx)
                lines.append(f"OpStore {state_ptrs[i]} {val_id}")
        else:
            for i, out_reg in enumerate(mir.output_regs):
                val_id = step_reg_map.get(f"v{out_reg.id}", const_0)
                val_id = apply_output_inversion(val_id, i)
                lines.append(f"OpStore {loop_out_ptrs[i]} {val_id}")
            for i in range(loop_state_words):
                out_idx = loop_state_output_offset + i
                out_reg = mir.output_regs[out_idx]
                val_id = step_reg_map.get(f"v{out_reg.id}", const_0)
                val_id = apply_output_inversion(val_id, out_idx)
                lines.append(f"OpStore {state_ptrs[i]} {val_id}")

        lines.append(f"OpBranch {loop_continue}")
        lines.append(f"{loop_continue} = OpLabel")
        ctr_next = nid()
        lines.append(f"{ctr_next} = OpIAdd {uint_id} {ctr_val} {const_map[1]}")
        lines.append(f"OpStore {counter_ptr} {ctr_next}")
        lines.append(f"OpBranch {loop_header}")
        lines.append(f"{loop_merge} = OpLabel")
        lines.append("")

        if loop_output_state_only:
            for i in range(loop_state_words):
                load_id = nid()
                lines.append(f"{load_id} = OpLoad {uint_id} {state_ptrs[i]}")
                emit_store_output_word(load_id, i)
        else:
            for i in range(len(mir.output_regs)):
                load_id = nid()
                lines.append(f"{load_id} = OpLoad {uint_id} {loop_out_ptrs[i]}")
                emit_store_output_word(load_id, i)

    lines.append("")
    lines.append("OpReturn")
    lines.append("OpFunctionEnd")

    lines[bound_idx] = f"; Bound: {id_counter}"
    return "\n".join(lines)
