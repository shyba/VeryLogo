"""
Vulkan compute shader backend for VeryLogo.

Generates SPIR-V assembly for Vulkan compute shaders.
"""

from __future__ import annotations

from stc.circuit_synth import CircuitState
from stc.sched.schedule import Schedule
from stc.sched.regalloc import RegAllocation
from stc.mir.lower import circuit_to_mir
from stc.mir.emit_spirv import emit_spirv


def wrap_spirv_as_compute_shader(
    spirv_func: str, num_inputs: int, num_outputs: int, workgroup_size: int = 256
) -> str:
    """
    Wrap SPIR-V circuit function as a Vulkan compute shader.

    Args:
        spirv_func: SPIR-V assembly for the circuit function
        num_inputs: Number of 32-bit input words
        num_outputs: Number of 32-bit output words
        workgroup_size: Compute shader workgroup size

    Returns:
        Complete SPIR-V assembly for compute shader
    """
    lines = []

    # Extract header from spirv_func
    func_lines = spirv_func.split("\n")
    header_end = 0
    for i, line in enumerate(func_lines):
        if line.startswith("; Bound:"):
            header_end = i + 1
            break

    # Start with header
    lines.extend(func_lines[:header_end])
    lines.append("")

    # Entry point and execution mode
    lines.append("; Entry point")
    lines.append('OpEntryPoint GLCompute %1 "main"')
    lines.append(f"OpExecutionMode %1 LocalSize {workgroup_size} 1 1")
    lines.append("")

    # Decorations for storage buffers
    lines.append("; Decorations")
    lines.append("OpDecorate %2 DescriptorSet 0")
    lines.append("OpDecorate %2 Binding 0")
    lines.append("OpDecorate %3 DescriptorSet 0")
    lines.append("OpDecorate %3 Binding 1")
    lines.append("")

    # Type declarations
    lines.append("; Types")
    lines.append("%4 = OpTypeInt 32 0")  # uint32
    lines.append("%5 = OpTypeVoid")
    lines.append("%6 = OpTypeFunction %5")  # void()

    # Array types for storage buffers
    lines.append(f"%7 = OpTypeArray %4 %8")  # uint32[] (runtime array)
    lines.append("%8 = OpConstant %4 0")  # Size constant

    # Struct types for storage buffers
    lines.append("%9 = OpTypeStruct %7")  # struct { uint32[] }
    lines.append("%10 = OpTypePointer StorageBuffer %9")  # ptr to struct
    lines.append("%11 = OpTypePointer StorageBuffer %4")  # ptr to uint32

    # Storage buffer variables
    lines.append("")
    lines.append("; Storage buffers")
    lines.append("%2 = OpVariable %10 StorageBuffer")  # Input buffer
    lines.append("%3 = OpVariable %10 StorageBuffer")  # Output buffer
    lines.append("")

    # Main compute shader function
    lines.append("; Main compute shader")
    lines.append("%1 = OpFunction %5 None %6")
    lines.append("%12 = OpLabel")
    lines.append("")

    # Get global invocation ID
    lines.append("; Get thread ID")
    lines.append("%13 = OpLoad %4 %gl_GlobalInvocationID")  # Simplified
    lines.append("")

    # Load inputs from buffer
    lines.append("; Load inputs")
    for i in range(num_inputs):
        base_offset = f"%{20 + i}"
        ptr_id = f"%{30 + i}"
        val_id = f"%{40 + i}"
        lines.append(f"{base_offset} = OpConstant %4 {i}")
        lines.append(f"{ptr_id} = OpAccessChain %11 %2 {base_offset}")
        lines.append(f"{val_id} = OpLoad %4 {ptr_id}")

    lines.append("")

    # TODO: Call circuit function (embed circuit logic here)
    # For now, just copy the circuit function body
    lines.append("; Circuit computation")
    lines.append("; (Circuit function body would be inlined here)")
    lines.append("")

    # Store outputs to buffer
    lines.append("; Store outputs")
    for i in range(num_outputs):
        base_offset = f"%{50 + i}"
        ptr_id = f"%{60 + i}"
        val_id = f"%{40 + i}"  # Use computed values
        lines.append(f"{base_offset} = OpConstant %4 {i}")
        lines.append(f"{ptr_id} = OpAccessChain %11 %3 {base_offset}")
        lines.append(f"OpStore {ptr_id} {val_id}")

    lines.append("")
    lines.append("OpReturn")
    lines.append("OpFunctionEnd")

    return "\n".join(lines)


def emit_vulkan_compute_shader(
    circuit: CircuitState,
    schedule: Schedule,
    allocation: RegAllocation,
    workgroup_size: int = 256,
) -> str:
    """
    Generate Vulkan compute shader from circuit.

    Args:
        circuit: Circuit to compile
        schedule: Instruction schedule
        allocation: Register allocation
        workgroup_size: Workgroup size for compute shader

    Returns:
        SPIR-V assembly text
    """
    # Lower to MIR
    mir = circuit_to_mir(circuit, schedule, allocation)

    # Generate SPIR-V for circuit function
    spirv_func = emit_spirv(mir, allocation, "circuit")

    # Wrap as compute shader
    return wrap_spirv_as_compute_shader(
        spirv_func, circuit.input_bits, circuit.output_bits, workgroup_size
    )
