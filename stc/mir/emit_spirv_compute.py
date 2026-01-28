"""
SPIR-V Vulkan compute shader emitter for VeryLogo MIR.
"""

from __future__ import annotations

from stc.mir import MIRFunction
from stc.sched.regalloc import RegAllocation
from stc.mir.emit_spirv import SPIRVEmitter


def emit_vulkan_compute_shader(
    mir: MIRFunction,
    allocation: RegAllocation,
    workgroup_size_x: int = 64,
    workgroup_size_y: int = 1,
    workgroup_size_z: int = 1,
) -> str:
    """
    Emit a complete Vulkan compute shader from MIR.

    Each thread processes one instance of the circuit on 32-bit data planes.

    Args:
        mir: Machine IR function
        allocation: Register allocation
        workgroup_size_x/y/z: Compute shader workgroup dimensions

    Returns:
        SPIR-V assembly for a complete compute shader
    """
    lines = []

    # Header
    lines.append("; SPIR-V")
    lines.append("; Version: 1.3")
    lines.append("; Generator: VeryLogo Vulkan Backend")
    lines.append("; Bound: 2000")  # Will update
    lines.append("; Schema: 0")
    lines.append("")

    # Capabilities
    lines.append("OpCapability Shader")
    lines.append("")

    # Memory model
    lines.append("OpMemoryModel Logical GLSL450")
    lines.append("")

    # Entry point (function %1 named "main")
    lines.append('OpEntryPoint GLCompute %1 "main" %gl_GlobalInvocationID')
    lines.append(
        f"OpExecutionMode %1 LocalSize {workgroup_size_x} {workgroup_size_y} {workgroup_size_z}"
    )
    lines.append("")

    # Decorations
    lines.append("; Decorations")
    lines.append("OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId")
    lines.append("OpDecorate %_arr_uint_32 ArrayStride 4")
    lines.append("OpMemberDecorate %_struct_input 0 Offset 0")
    lines.append("OpDecorate %_struct_input BufferBlock")
    lines.append("OpDecorate %input_buffer DescriptorSet 0")
    lines.append("OpDecorate %input_buffer Binding 0")
    lines.append("OpMemberDecorate %_struct_output 0 Offset 0")
    lines.append("OpDecorate %_struct_output BufferBlock")
    lines.append("OpDecorate %output_buffer DescriptorSet 0")
    lines.append("OpDecorate %output_buffer Binding 1")
    lines.append("")

    # Type declarations
    lines.append("; Types")
    lines.append("%void = OpTypeVoid")
    lines.append("%bool = OpTypeBool")
    lines.append("%uint = OpTypeInt 32 0")
    lines.append("%v3uint = OpTypeVector %uint 3")
    lines.append("%_ptr_Input_v3uint = OpTypePointer Input %v3uint")
    lines.append("%_ptr_Function_uint = OpTypePointer Function %uint")
    lines.append("%_ptr_Uniform_uint = OpTypePointer Uniform %uint")

    # Arrays for storage buffers
    num_inputs = len(mir.input_regs)
    num_outputs = len(mir.output_regs)
    lines.append(f"%uint_32 = OpConstant %uint 32")
    lines.append(f"%_arr_uint_32 = OpTypeRuntimeArray %uint")

    # Structs for storage buffers
    lines.append("%_struct_input = OpTypeStruct %_arr_uint_32")
    lines.append("%_struct_output = OpTypeStruct %_arr_uint_32")
    lines.append("%_ptr_Uniform_struct_input = OpTypePointer Uniform %_struct_input")
    lines.append("%_ptr_Uniform_struct_output = OpTypePointer Uniform %_struct_output")
    lines.append("%func_type = OpTypeFunction %void")
    lines.append("")

    # Constants
    lines.append("; Constants")
    lines.append("%uint_0 = OpConstant %uint 0")
    lines.append("%uint_1 = OpConstant %uint 1")
    for i in range(max(num_inputs, num_outputs)):
        lines.append(f"%uint_{i} = OpConstant %uint {i}")
    lines.append("")

    # Global variables
    lines.append("; Global variables")
    lines.append("%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input")
    lines.append("%input_buffer = OpVariable %_ptr_Uniform_struct_input Uniform")
    lines.append("%output_buffer = OpVariable %_ptr_Uniform_struct_output Uniform")
    lines.append("")

    # Main function
    lines.append("; Main compute shader function")
    lines.append("%1 = OpFunction %void None %func_type")
    lines.append("%entry = OpLabel")
    lines.append("")

    # Get thread ID
    lines.append("; Get global invocation ID")
    lines.append("%gid_vec = OpLoad %v3uint %gl_GlobalInvocationID")
    lines.append("%tid = OpCompositeExtract %uint %gid_vec 0")
    lines.append("")

    # Calculate base offset for this thread
    lines.append("; Calculate buffer offset (tid * num_words)")
    lines.append(f"%tid_mul_{num_inputs} = OpIMul %uint %tid %uint_{num_inputs}")
    lines.append("")

    # Load inputs
    lines.append("; Load inputs from buffer")
    reg_ids = {}
    for i, reg in enumerate(mir.input_regs):
        offset_id = f"%offset_in_{i}"
        ptr_id = f"%ptr_in_{i}"
        val_id = f"%val_in_{i}"

        lines.append(f"{offset_id} = OpIAdd %uint %tid_mul_{num_inputs} %uint_{i}")
        lines.append(
            f"{ptr_id} = OpAccessChain %_ptr_Uniform_uint %input_buffer %uint_0 {offset_id}"
        )
        lines.append(f"{val_id} = OpLoad %uint {ptr_id}")

        # Map MIR register to SPIR-V ID
        reg_key = f"v{reg.id}"
        reg_ids[reg_key] = val_id

    lines.append("")

    # Circuit computation (simplified - just use the input as output for now)
    lines.append("; Circuit computation")
    lines.append("; (Circuit operations would go here)")
    lines.append("")

    # Store outputs
    lines.append("; Store outputs to buffer")
    lines.append(f"%tid_mul_out = OpIMul %uint %tid %uint_{num_outputs}")
    for i, reg in enumerate(mir.output_regs):
        offset_id = f"%offset_out_{i}"
        ptr_id = f"%ptr_out_{i}"
        # For now, just store input values (placeholder)
        val_id = f"%val_in_{min(i, num_inputs - 1)}"

        lines.append(f"{offset_id} = OpIAdd %uint %tid_mul_out %uint_{i}")
        lines.append(
            f"{ptr_id} = OpAccessChain %_ptr_Uniform_uint %output_buffer %uint_0 {offset_id}"
        )
        lines.append(f"OpStore {ptr_id} {val_id}")

    lines.append("")
    lines.append("OpReturn")
    lines.append("OpFunctionEnd")

    return "\n".join(lines)


def emit_vulkan_compute_minimal(num_inputs: int, num_outputs: int) -> str:
    """Generate minimal valid Vulkan compute shader for testing."""
    return f"""; SPIR-V
; Version: 1.3
; Generator: VeryLogo
; Bound: 100
; Schema: 0
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID
               OpExecutionMode %main LocalSize 64 1 1
               OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
               OpDecorate %_runtimearr_uint ArrayStride 4
               OpMemberDecorate %_struct_7 0 Offset 0
               OpDecorate %_struct_7 BufferBlock
               OpDecorate %8 DescriptorSet 0
               OpDecorate %8 Binding 0
               OpMemberDecorate %_struct_9 0 Offset 0
               OpDecorate %_struct_9 BufferBlock
               OpDecorate %10 DescriptorSet 0
               OpDecorate %10 Binding 1
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
%_runtimearr_uint = OpTypeRuntimeArray %uint
  %_struct_7 = OpTypeStruct %_runtimearr_uint
%_ptr_Uniform__struct_7 = OpTypePointer Uniform %_struct_7
          %8 = OpVariable %_ptr_Uniform__struct_7 Uniform
  %_struct_9 = OpTypeStruct %_runtimearr_uint
%_ptr_Uniform__struct_9 = OpTypePointer Uniform %_struct_9
         %10 = OpVariable %_ptr_Uniform__struct_9 Uniform
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
     %uint_0 = OpConstant %uint 0
       %main = OpFunction %void None %3
          %5 = OpLabel
         %13 = OpLoad %v3uint %gl_GlobalInvocationID
         %14 = OpCompositeExtract %uint %13 0
         %20 = OpAccessChain %_ptr_Uniform_uint %8 %int_0 %14
         %21 = OpLoad %uint %20
         %23 = OpAccessChain %_ptr_Uniform_uint %10 %int_0 %14
               OpStore %23 %21
               OpReturn
               OpFunctionEnd
"""
