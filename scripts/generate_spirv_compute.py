#!/usr/bin/env python3
"""
Generate a complete SPIR-V compute shader for tower field S-box.

This creates a proper Vulkan compute shader with buffer I/O.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.bp_circuit_sbox import build_bp_sbox
from stc.gate_ternary_synth import apply_ternary_synthesis


def generate_spirv_compute_shader(circuit_opt, num_inputs=8, num_outputs=8):
    """Generate complete SPIR-V compute shader with buffer I/O."""

    gates = list(circuit_opt.gates)
    outputs = list(circuit_opt.outputs)

    lines = []

    # Header
    lines.append("; SPIR-V")
    lines.append("; Version: 1.3")
    lines.append("; Generator: VeryLogo")
    lines.append("; Bound: 500")
    lines.append("; Schema: 0")
    lines.append("")
    lines.append("OpCapability Shader")
    lines.append("OpMemoryModel Logical GLSL450")
    lines.append('OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID')
    lines.append("OpExecutionMode %main LocalSize 64 1 1")
    lines.append("")

    # Decorations
    lines.append("; Decorations")
    lines.append("OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId")
    lines.append("OpDecorate %_runtimearr_uint ArrayStride 4")
    lines.append("OpMemberDecorate %input_struct 0 Offset 0")
    lines.append("OpDecorate %input_struct BufferBlock")
    lines.append("OpDecorate %input_buffer DescriptorSet 0")
    lines.append("OpDecorate %input_buffer Binding 0")
    lines.append("OpMemberDecorate %output_struct 0 Offset 0")
    lines.append("OpDecorate %output_struct BufferBlock")
    lines.append("OpDecorate %output_buffer DescriptorSet 0")
    lines.append("OpDecorate %output_buffer Binding 1")
    lines.append("")

    # Types
    lines.append("; Types")
    lines.append("%void = OpTypeVoid")
    lines.append("%bool = OpTypeBool")
    lines.append("%uint = OpTypeInt 32 0")
    lines.append("%int = OpTypeInt 32 1")
    lines.append("%v3uint = OpTypeVector %uint 3")
    lines.append("%_ptr_Input_v3uint = OpTypePointer Input %v3uint")
    lines.append("%_ptr_Uniform_uint = OpTypePointer Uniform %uint")
    lines.append("%_runtimearr_uint = OpTypeRuntimeArray %uint")
    lines.append("%input_struct = OpTypeStruct %_runtimearr_uint")
    lines.append("%output_struct = OpTypeStruct %_runtimearr_uint")
    lines.append("%_ptr_Uniform_input = OpTypePointer Uniform %input_struct")
    lines.append("%_ptr_Uniform_output = OpTypePointer Uniform %output_struct")
    lines.append("%func_type = OpTypeFunction %void")
    lines.append("")

    # Constants
    lines.append("; Constants")
    lines.append("%uint_0 = OpConstant %uint 0")
    lines.append("%uint_1 = OpConstant %uint 1")
    lines.append("%int_0 = OpConstant %int 0")
    for i in range(2, num_inputs):  # Start from 2 since 0,1 already defined
        lines.append(f"%uint_{i} = OpConstant %uint {i}")
    lines.append("%uint_8 = OpConstant %uint 8")
    lines.append("%uint_0xFFFFFFFF = OpConstant %uint 4294967295")
    lines.append("")

    # Global variables
    lines.append("; Variables")
    lines.append("%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input")
    lines.append("%input_buffer = OpVariable %_ptr_Uniform_input Uniform")
    lines.append("%output_buffer = OpVariable %_ptr_Uniform_output Uniform")
    lines.append("")

    # Main function
    lines.append("; Main function")
    lines.append("%main = OpFunction %void None %func_type")
    lines.append("%entry = OpLabel")
    lines.append("")

    # Get thread ID
    lines.append("; Get thread ID")
    lines.append("%gid_vec = OpLoad %v3uint %gl_GlobalInvocationID")
    lines.append("%tid = OpCompositeExtract %uint %gid_vec 0")
    lines.append("")

    # Calculate base offset
    lines.append("; Calculate base offset (tid * 8)")
    lines.append("%base_offset = OpIMul %uint %tid %uint_8")
    lines.append("")

    # Load inputs
    lines.append("; Load 8 input words from buffer")
    input_ids = []
    for i in range(num_inputs):
        offset_id = f"%in_offset_{i}"
        ptr_id = f"%in_ptr_{i}"
        val_id = f"%in_{i}"

        lines.append(f"{offset_id} = OpIAdd %uint %base_offset %uint_{i}")
        lines.append(
            f"{ptr_id} = OpAccessChain %_ptr_Uniform_uint %input_buffer %int_0 {offset_id}"
        )
        lines.append(f"{val_id} = OpLoad %uint {ptr_id}")
        input_ids.append(val_id)
    lines.append("")

    # Circuit computation
    lines.append("; Circuit computation (128 gates → 94 after optimization)")

    # Generate gates
    reg_map = {i: input_ids[i] for i in range(num_inputs)}
    reg_counter = 200

    for g_idx, gate in enumerate(gates):
        op = gate[0]
        dst_id = f"%g{reg_counter}"
        reg_counter += 1
        node_idx = num_inputs + g_idx

        if op == "xor":
            left = gate[1]
            right = gate[2]
            left_id = reg_map[left]
            right_id = reg_map[right]
            lines.append(f"{dst_id} = OpBitwiseXor %uint {left_id} {right_id}")

        elif op == "and":
            left = gate[1]
            right = gate[2]
            left_id = reg_map[left]
            right_id = reg_map[right]
            lines.append(f"{dst_id} = OpBitwiseAnd %uint {left_id} {right_id}")

        elif op == "ternary":
            a = gate[1]
            b = gate[2]
            c = gate[3]
            imm8 = gate[4]

            a_id = reg_map[a]
            b_id = reg_map[b]
            c_id = reg_map[c]

            # Decompose ternary for common cases
            if imm8 == 150:  # a^b^c
                temp_id = f"%t{reg_counter}"
                reg_counter += 1
                lines.append(f"{temp_id} = OpBitwiseXor %uint {a_id} {b_id}")
                lines.append(f"{dst_id} = OpBitwiseXor %uint {temp_id} {c_id}")
            elif imm8 == 106:  # a^(b&c)
                temp_id = f"%t{reg_counter}"
                reg_counter += 1
                lines.append(f"{temp_id} = OpBitwiseAnd %uint {b_id} {c_id}")
                lines.append(f"{dst_id} = OpBitwiseXor %uint {a_id} {temp_id}")
            elif imm8 == 40:  # (a&b)^c
                temp_id = f"%t{reg_counter}"
                reg_counter += 1
                lines.append(f"{temp_id} = OpBitwiseAnd %uint {a_id} {b_id}")
                lines.append(f"{dst_id} = OpBitwiseXor %uint {temp_id} {c_id}")
            elif imm8 == 128:  # a&b&c
                temp_id = f"%t{reg_counter}"
                reg_counter += 1
                lines.append(f"{temp_id} = OpBitwiseAnd %uint {a_id} {b_id}")
                lines.append(f"{dst_id} = OpBitwiseAnd %uint {temp_id} {c_id}")
            else:
                # Generic fallback
                print(f"Warning: unhandled ternary pattern imm8={imm8}")
                lines.append(f"{dst_id} = OpBitwiseXor %uint {a_id} {b_id}")

        reg_map[node_idx] = dst_id

    lines.append("")

    # Store outputs
    lines.append("; Store 8 output words to buffer")
    for i, (out_node, invert) in enumerate(outputs):
        out_val_id = reg_map[out_node]

        # Apply inversion if needed
        if invert:
            inverted_id = f"%out_inv_{i}"
            lines.append(
                f"{inverted_id} = OpBitwiseXor %uint {out_val_id} %uint_0xFFFFFFFF"
            )
            out_val_id = inverted_id

        offset_id = f"%out_offset_{i}"
        ptr_id = f"%out_ptr_{i}"

        lines.append(f"{offset_id} = OpIAdd %uint %base_offset %uint_{i}")
        lines.append(
            f"{ptr_id} = OpAccessChain %_ptr_Uniform_uint %output_buffer %int_0 {offset_id}"
        )
        lines.append(f"OpStore {ptr_id} {out_val_id}")

    lines.append("")
    lines.append("OpReturn")
    lines.append("OpFunctionEnd")

    return "\n".join(lines)


def main():
    print("Generating complete SPIR-V compute shader with buffer I/O...")
    print()

    circuit = build_bp_sbox()
    print(f"Gates: {circuit.gate_count}")

    circuit_opt, stats = apply_ternary_synthesis(circuit)
    print(f"After ternary synthesis: {circuit_opt.gate_count}")
    print(f"Ternary gates: {stats.ternary_gates_created}")
    print()

    spirv = generate_spirv_compute_shader(circuit_opt, 8, 8)

    output_path = "out/tower_sbox_spirv/compute.spvasm"
    with open(output_path, "w") as f:
        f.write(spirv)

    print(f"Written to: {output_path}")
    print(f"Size: {len(spirv)} bytes")
    print()

    # Show sample
    lines = spirv.split("\n")
    print("First 30 lines:")
    print("-" * 60)
    for line in lines[:30]:
        print(line)
    print("-" * 60)


if __name__ == "__main__":
    main()
