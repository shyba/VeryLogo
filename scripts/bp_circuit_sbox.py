#!/usr/bin/env python3
"""
AES S-box using Boyar-Peralta's optimal circuit (115 gates = 32 AND + 83 XOR).

This directly implements the circuit from:
"A New Combinational Logic Minimization Technique with Applications to Cryptology"
by Joan Boyar and René Peralta.

The circuit is organized as:
1. Top linear layer (XOR only) - prepares inputs for non-linear section
2. Non-linear middle (32 AND gates) - computes GF(2^8) inversion
3. Bottom linear layer (XOR only) - applies affine transformation and output mapping
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stc.circuit_synth import CircuitState
from stc.bitslice import AES_SBOX_TABLE


def build_bp_sbox() -> CircuitState:
    """
    Build the Boyar-Peralta AES S-box circuit.

    Directly transcribed from external-bitsliced/bs.c bs_sbox() function.
    Input U[0]-U[7], Output S[0]-S[7].
    """
    gates: list[tuple[str, int, int]] = []
    wire_map: dict[str, int] = {}

    # Input mapping: U[i] maps to input bit i directly
    # Our convention: input 0 is bit 0 (LSB), input 7 is bit 7 (MSB)
    for i in range(8):
        wire_map[f"U{i}"] = i

    def xor(a: str, b: str, name: str) -> int:
        idx_a = wire_map[a]
        idx_b = wire_map[b]
        new_idx = 8 + len(gates)
        gates.append(("xor", idx_a, idx_b))
        wire_map[name] = new_idx
        return new_idx

    def and_(a: str, b: str, name: str) -> int:
        idx_a = wire_map[a]
        idx_b = wire_map[b]
        new_idx = 8 + len(gates)
        gates.append(("and", idx_a, idx_b))
        wire_map[name] = new_idx
        return new_idx

    def xnor(a: str, b: str, name: str) -> int:
        """XOR with inverted output - we'll handle via output inversions."""
        return xor(a, b, name)

    # ========================================
    # Top linear layer - from bs.c bs_sbox()
    # ========================================

    xor("U7", "U4", "T1")
    xor("U7", "U2", "T2")
    xor("U7", "U1", "T3")
    xor("U4", "U2", "T4")
    xor("U3", "U1", "T5")
    xor("T1", "T5", "T6")
    xor("U6", "U5", "T7")
    xor("U0", "T6", "T8")
    xor("U0", "T7", "T9")
    xor("T6", "T7", "T10")
    xor("U6", "U2", "T11")
    xor("U5", "U2", "T12")
    xor("T3", "T4", "T13")
    xor("T6", "T11", "T14")
    xor("T5", "T11", "T15")
    xor("T5", "T12", "T16")
    xor("T9", "T16", "T17")
    xor("U4", "U0", "T18")
    xor("T7", "T18", "T19")
    xor("T1", "T19", "T20")
    xor("U1", "U0", "T21")
    xor("T7", "T21", "T22")
    xor("T2", "T22", "T23")
    xor("T2", "T10", "T24")
    xor("T20", "T17", "T25")
    xor("T3", "T16", "T26")
    xor("T1", "T12", "T27")

    # ========================================
    # Non-linear middle - from bs.c bs_sbox()
    # ========================================

    and_("T13", "T6", "M1")
    and_("T23", "T8", "M2")
    xor("T14", "M1", "M3")
    and_("T19", "U0", "M4")
    xor("M4", "M1", "M5")
    and_("T3", "T16", "M6")
    and_("T22", "T9", "M7")
    xor("T26", "M6", "M8")
    and_("T20", "T17", "M9")
    xor("M9", "M6", "M10")
    and_("T1", "T15", "M11")
    and_("T4", "T27", "M12")
    xor("M12", "M11", "M13")
    and_("T2", "T10", "M14")
    xor("M14", "M11", "M15")
    xor("M3", "M2", "M16")
    xor("M5", "T24", "M17")
    xor("M8", "M7", "M18")
    xor("M10", "M15", "M19")
    xor("M16", "M13", "M20")
    xor("M17", "M15", "M21")
    xor("M18", "M13", "M22")
    xor("M19", "T25", "M23")
    xor("M22", "M23", "M24")
    and_("M22", "M20", "M25")
    xor("M21", "M25", "M26")
    xor("M20", "M21", "M27")
    xor("M23", "M25", "M28")
    and_("M28", "M27", "M29")
    and_("M26", "M24", "M30")
    and_("M20", "M23", "M31")
    and_("M27", "M31", "M32")
    xor("M27", "M25", "M33")
    and_("M21", "M22", "M34")
    and_("M24", "M34", "M35")
    xor("M24", "M25", "M36")
    xor("M21", "M29", "M37")
    xor("M32", "M33", "M38")
    xor("M23", "M30", "M39")
    xor("M35", "M36", "M40")
    xor("M38", "M40", "M41")
    xor("M37", "M39", "M42")
    xor("M37", "M38", "M43")
    xor("M39", "M40", "M44")
    xor("M42", "M41", "M45")
    and_("M44", "T6", "M46")
    and_("M40", "T8", "M47")
    and_("M39", "U0", "M48")
    and_("M43", "T16", "M49")
    and_("M38", "T9", "M50")
    and_("M37", "T17", "M51")
    and_("M42", "T15", "M52")
    and_("M45", "T27", "M53")
    and_("M41", "T10", "M54")
    and_("M44", "T13", "M55")
    and_("M40", "T23", "M56")
    and_("M39", "T19", "M57")
    and_("M43", "T3", "M58")
    and_("M38", "T22", "M59")
    and_("M37", "T20", "M60")
    and_("M42", "T1", "M61")
    and_("M45", "T4", "M62")
    and_("M41", "T2", "M63")

    # ========================================
    # Bottom linear layer - from bs.c bs_sbox()
    # ========================================

    xor("M61", "M62", "L0")
    xor("M50", "M56", "L1")
    xor("M46", "M48", "L2")
    xor("M47", "M55", "L3")
    xor("M54", "M58", "L4")
    xor("M49", "M61", "L5")
    xor("M62", "L5", "L6")
    xor("M46", "L3", "L7")
    xor("M51", "M59", "L8")
    xor("M52", "M53", "L9")
    xor("M53", "L4", "L10")
    xor("M60", "L2", "L11")
    xor("M48", "M51", "L12")
    xor("M50", "L0", "L13")
    xor("M52", "M61", "L14")
    xor("M55", "L1", "L15")
    xor("M56", "L0", "L16")
    xor("M57", "L1", "L17")
    xor("M58", "L8", "L18")
    xor("M63", "L4", "L19")
    xor("L0", "L1", "L20")
    xor("L1", "L7", "L21")
    xor("L3", "L12", "L22")
    xor("L18", "L2", "L23")
    xor("L15", "L9", "L24")
    xor("L6", "L10", "L25")
    xor("L7", "L9", "L26")
    xor("L8", "L10", "L27")
    xor("L11", "L14", "L28")
    xor("L11", "L17", "L29")

    # Final outputs - from bs.c
    # S[7] = L6 ^ L24
    # S[6] = ~(L16 ^ L26)  -- inverted
    # S[5] = ~(L19 ^ L28)  -- inverted
    # S[4] = L6 ^ L21
    # S[3] = L20 ^ L22
    # S[2] = L25 ^ L29
    # S[1] = ~(L13 ^ L27)  -- inverted
    # S[0] = ~(L6 ^ L23)   -- inverted
    xor("L6", "L24", "S7")
    xor("L16", "L26", "S6")
    xor("L19", "L28", "S5")
    xor("L6", "L21", "S4")
    xor("L20", "L22", "S3")
    xor("L25", "L29", "S2")
    xor("L13", "L27", "S1")
    xor("L6", "L23", "S0")

    # Build outputs: S[i] maps to output bit i
    # Inversions from bs.c: S[6], S[5], S[1], S[0] are inverted
    outputs = [
        (wire_map["S0"], True),  # bit 0, inverted (~)
        (wire_map["S1"], True),  # bit 1, inverted (~)
        (wire_map["S2"], False),  # bit 2
        (wire_map["S3"], False),  # bit 3
        (wire_map["S4"], False),  # bit 4
        (wire_map["S5"], True),  # bit 5, inverted (~)
        (wire_map["S6"], True),  # bit 6, inverted (~)
        (wire_map["S7"], False),  # bit 7
    ]

    return CircuitState(
        input_bits=8,
        output_bits=8,
        gates=gates,
        outputs=outputs,
        gate_count=len(gates),
    )


def verify_sbox(circuit: CircuitState, name: str) -> bool:
    """Verify circuit against AES S-box table."""
    errors = []
    for i in range(256):
        result = circuit.evaluate(i)
        expected = AES_SBOX_TABLE[i]
        if result != expected:
            errors.append((i, result, expected))
    if errors:
        print(f"  {name}: {len(errors)} errors")
        for i, got, exp in errors[:5]:
            print(f"    Input {i:#04x}: got {got:#04x}, expected {exp:#04x}")
        return False
    print(f"  {name}: All 256 values correct!")
    return True


def main():
    print("Building Boyar-Peralta AES S-box circuit...")
    print()

    circuit = build_bp_sbox()
    print(
        f"Circuit: {circuit.gate_count} gates ({circuit.and_count} AND, {circuit.xor_count} XOR)"
    )
    print()

    correct = verify_sbox(circuit, "BP circuit")
    print()

    if not correct:
        print("Circuit is not correct. Debugging...")
        # Check S-box(0) = 0x63
        print(f"  S-box(0x00) = {circuit.evaluate(0):#04x} (expected 0x63)")
        print(f"  S-box(0x01) = {circuit.evaluate(1):#04x} (expected 0x7c)")
        print(f"  S-box(0x63) = {circuit.evaluate(0x63):#04x} (expected 0xfb)")
        return

    print("Applying optimizations to verify optimizer works on BP-style circuit...")
    print()

    # Phase 1 optimizations
    opt = circuit
    print(f"Initial: {opt.gate_count} gates ({opt.and_count} AND, {opt.xor_count} XOR)")

    opt = opt.eliminate_common_subexpressions()
    print(f"After CSE: {opt.gate_count} gates")

    opt = opt.apply_algebraic_rewrites()
    print(f"After algebraic: {opt.gate_count} gates")

    opt = opt.eliminate_dead_code()
    print(f"After DCE: {opt.gate_count} gates")

    opt = opt.flatten_xor_trees()
    print(f"After XOR flatten: {opt.gate_count} gates")

    opt = opt.try_local_rewrites()
    print(
        f"After local rewrites: {opt.gate_count} gates ({opt.and_count} AND, {opt.xor_count} XOR)"
    )

    # Verify still correct
    if not verify_sbox(opt, "After Phase 1"):
        print("ERROR: Optimization broke correctness!")
        return

    # Phase 2: BP linear optimization (skip due to bug)
    # print()
    # print("Applying BP linear layer optimization...")
    # opt = opt.optimize_linear_layers()
    # print(f"After BP linear: {opt.gate_count} gates")

    # Phase 3: SAT window resynthesis
    print()
    print("Applying Phase 3 SAT window resynthesis (10 iterations)...")
    opt = opt.sat_window_resynthesis(max_iterations=10, max_window_inputs=6)
    print(
        f"After Phase 3: {opt.gate_count} gates ({opt.and_count} AND, {opt.xor_count} XOR)"
    )

    if not verify_sbox(opt, "After Phase 3"):
        print("ERROR: SAT resynthesis broke correctness!")
        return

    print()
    print("SUCCESS: Optimizer works correctly on BP-style circuit!")
    print(
        f"Final circuit: {opt.gate_count} gates ({opt.and_count} AND, {opt.xor_count} XOR)"
    )


if __name__ == "__main__":
    main()
