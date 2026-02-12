#!/usr/bin/env python3
"""
Compile tower field S-box to PTX without ternary synthesis.

This is useful for comparing pure boolean scheduling vs lop3-heavy versions.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stc.tick_ir_to_circuit_state import lower_tick_ir_to_circuit_state
from stc.tick_ir_bin2 import read_tick_ir_bin
from stc.backend_sched import generate_scheduled_code


def main() -> None:
    # Load the reduced tick IR
    print("Loading reduced TickIR...")
    ir = read_tick_ir_bin("out/tower_sbox_ptx/reduced_tick_ir.bin")
    print("  Loaded TickIR")
    print()

    # Lower to CircuitState
    print("Lowering to CircuitState...")
    circuit, _layout = lower_tick_ir_to_circuit_state(ir)
    print(
        f"  Gates: {circuit.gate_count} ({circuit.and_count} AND, {circuit.xor_count} XOR)"
    )
    print()

    # Generate PTX code (no ternary synthesis)
    print("Generating PTX code (no ternary)...")
    code = generate_scheduled_code(circuit, target="ptx_legacy", scheduler="list")
    print(f"  Generated: {len(code)} bytes")

    lop3_count = code.count("lop3.b32")
    print(f"  lop3.b32 instructions: {lop3_count}")
    print()

    output_path = "out/tower_sbox_ptx/circuit_no_ternary.ptx"
    with open(output_path, "w") as f:
        f.write(code)
    print(f"Written to: {output_path}")


if __name__ == "__main__":
    main()
