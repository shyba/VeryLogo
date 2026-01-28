#!/usr/bin/env python3
"""
Compile tower field S-box to PTX from Verilog pipeline output.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stc.tick_ir_to_circuit_state import lower_tick_ir_to_circuit_state
from stc.tick_ir import TickIR
from stc.backend_sched import generate_scheduled_code


def main():
    # Load the reduced tick IR
    print("Loading reduced TickIR...")
    with open("out/tower_sbox_ptx/reduced_tick_ir.json") as f:
        ir_dict = json.load(f)

    ir = TickIR.from_dict(ir_dict)
    print(f"  Loaded TickIR")
    print()

    # Lower to CircuitState
    print("Lowering to CircuitState...")
    circuit, layout = lower_tick_ir_to_circuit_state(ir)
    print(
        f"  Gates before optimization: {circuit.gate_count} ({circuit.and_count} AND, {circuit.xor_count} XOR)"
    )

    # Apply gate optimizations
    print()
    print("Applying gate optimizations...")
    from stc.gate_ternary_synth import apply_ternary_synthesis

    circuit_opt, ternary_stats = apply_ternary_synthesis(circuit)
    print(f"  Gates after ternary synth: {circuit_opt.gate_count}")
    print(f"  Ternary gates created: {ternary_stats.ternary_gates_created}")
    print()

    # Generate PTX code
    print("Generating PTX code...")
    code = generate_scheduled_code(circuit_opt, target="ptx", scheduler="list")
    print(f"  Generated: {len(code)} bytes")

    # Count lop3 instructions
    lop3_count = code.count("lop3.b32")
    print(f"  lop3.b32 instructions: {lop3_count}")
    print()

    # Write output
    output_path = "out/tower_sbox_ptx/circuit.ptx"
    with open(output_path, "w") as f:
        f.write(code)
    print(f"Written to: {output_path}")


if __name__ == "__main__":
    main()
