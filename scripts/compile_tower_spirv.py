#!/usr/bin/env python3
"""
Compile tower field S-box to SPIR-V for Vulkan.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.bp_circuit_sbox import build_bp_sbox
from stc.gate_ternary_synth import apply_ternary_synthesis
from stc.sched.list_scheduler import list_schedule
from stc.sched.liveness import compute_live_ranges
from stc.sched.regalloc import allocate_registers
from stc.backend_sched import TARGETS
from stc.mir.lower import circuit_to_mir
from stc.mir.emit_spirv import emit_spirv


def main():
    print("=" * 60)
    print("Tower Field S-box - SPIR-V/Vulkan Compilation")
    print("=" * 60)
    print()

    # Build BP128 circuit
    print("1. Building BP128 S-box circuit...")
    circuit = build_bp_sbox()
    print(f"   Gates before optimization: {circuit.gate_count}")
    print(f"   ({circuit.and_count} AND, {circuit.xor_count} XOR)")
    print()

    # Apply gate optimizations
    print("2. Applying ternary synthesis...")
    circuit_opt, ternary_stats = apply_ternary_synthesis(circuit)
    print(f"   Gates after optimization: {circuit_opt.gate_count}")
    print(f"   Ternary gates created: {ternary_stats.ternary_gates_created}")
    reduction = 100 - circuit_opt.gate_count * 100 / circuit.gate_count
    print(f"   Reduction: {reduction:.1f}%")
    print()

    # Schedule and allocate
    print("3. Scheduling and register allocation...")
    target = TARGETS["avx512"]
    schedule = list_schedule(
        list(circuit_opt.gates),
        circuit_opt.input_bits,
        list(circuit_opt.outputs),
        target,
    )

    max_cycle = max(schedule.gate_cycle.values()) if schedule.gate_cycle else 0
    print(f"   Scheduled in {max_cycle + 1} cycles")

    live_ranges = compute_live_ranges(
        schedule,
        list(circuit_opt.gates),
        circuit_opt.input_bits,
        list(circuit_opt.outputs),
    )

    allocation = allocate_registers(
        live_ranges,
        schedule,
        target.registers,
        gates=list(circuit_opt.gates),
        input_bits=circuit_opt.input_bits,
        outputs=list(circuit_opt.outputs),
    )
    max_reg = (
        max(allocation.reg_assignment.values()) if allocation.reg_assignment else 0
    )
    print(f"   Registers used: {max_reg + 1}")
    print(f"   Spills: {allocation.num_spills}")
    print()

    # Lower to MIR
    print("4. Lowering to Machine IR...")
    mir = circuit_to_mir(circuit_opt, schedule, allocation)
    print(f"   {len(mir.instructions)} MIR instructions")

    ternary_count = sum(
        1 for inst in mir.instructions if inst.__class__.__name__ == "Ternary"
    )
    print(f"   Ternary operations: {ternary_count}")
    print()

    # Generate SPIR-V
    print("5. Generating SPIR-V assembly...")
    spirv = emit_spirv(mir, allocation, "circuit")
    print(f"   Generated {len(spirv)} bytes")

    # Count operations
    op_bitwise_and = spirv.count("OpBitwiseAnd")
    op_bitwise_or = spirv.count("OpBitwiseOr")
    op_bitwise_xor = spirv.count("OpBitwiseXor")
    op_not = spirv.count("OpNot")

    print(f"   OpBitwiseAnd: {op_bitwise_and}")
    print(f"   OpBitwiseXor: {op_bitwise_xor}")
    print(f"   OpBitwiseOr: {op_bitwise_or}")
    print(f"   OpNot: {op_not}")
    print()

    # Write output
    output_path = "out/tower_sbox_spirv/circuit.spvasm"
    os.makedirs("out/tower_sbox_spirv", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(spirv)
    print(f"6. Written to: {output_path}")
    print()

    # Show sample
    print("Sample SPIR-V assembly (first 30 lines):")
    print("-" * 60)
    for line in spirv.split("\n")[:30]:
        print(line)
    print("-" * 60)
    print()

    print("✓ SPIR-V compilation complete!")
    print()
    print("Next steps:")
    print("  - Assemble to binary: spirv-as circuit.spvasm -o circuit.spv")
    print("  - Validate: spirv-val circuit.spv")
    print("  - Optimize: spirv-opt -O circuit.spv -o circuit_opt.spv")
    print("  - Use in Vulkan compute shaders for cross-platform GPU execution")


if __name__ == "__main__":
    main()
