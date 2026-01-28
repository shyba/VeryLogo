#!/usr/bin/env python3
"""
AES S-box via Tower Field Decomposition

This implements the AES S-box using tower field arithmetic:
  GF(2^8) = GF((2^4)^2) = GF(((2^2)^2)^2)

The S-box computes: S(x) = A * inv(x) + c
where inv(x) is the multiplicative inverse in GF(2^8), with inv(0) = 0.

Tower field inversion requires exactly 32 AND gates (multiplications),
with the rest being XOR gates (additions in GF(2)).

Reference: Boyar & Peralta, Canright's compact AES implementation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from stc.circuit_synth import CircuitState
from stc.bitslice import AES_SBOX_TABLE


@dataclass
class TowerFieldBuilder:
    """Builds a circuit using tower field arithmetic."""

    input_bits: int
    gates: list[tuple[str, int, int]]

    def __init__(self, input_bits: int = 8):
        self.input_bits = input_bits
        self.gates = []

    def input(self, i: int) -> int:
        """Get input signal index."""
        assert 0 <= i < self.input_bits
        return i

    def xor(self, a: int, b: int) -> int:
        """Add XOR gate, return output index."""
        idx = self.input_bits + len(self.gates)
        self.gates.append(("xor", a, b))
        return idx

    def and_(self, a: int, b: int) -> int:
        """Add AND gate, return output index."""
        idx = self.input_bits + len(self.gates)
        self.gates.append(("and", a, b))
        return idx

    def xor3(self, a: int, b: int, c: int) -> int:
        """XOR three values."""
        return self.xor(self.xor(a, b), c)

    def xor4(self, a: int, b: int, c: int, d: int) -> int:
        """XOR four values."""
        return self.xor(self.xor(a, b), self.xor(c, d))


def build_tower_field_sbox() -> CircuitState:
    """
    Build AES S-box circuit using Boyar-Peralta's optimal circuit.

    The tower field approach decomposes GF(2^8) inversion as:
    1. Top linear layer (27 XOR) - maps polynomial basis to tower basis
    2. Non-linear middle (32 AND) - computes GF(2^8) inversion via tower field
    3. Bottom linear layer (69 XOR) - maps back and applies affine transformation

    This is the Boyar-Peralta optimal circuit with 128 gates total.
    """
    b = TowerFieldBuilder(8)

    # Input bits: U[i] = input(i), U[0] is LSB, U[7] is MSB

    # ========================================
    # Top linear layer (27 XOR gates)
    # Prepares inputs for non-linear section
    # ========================================

    # Note: Using TowerFieldBuilder which tracks node indices automatically
    # Inputs are 0-7, gates start at index 8

    T1 = b.xor(7, 4)  # U7 ^ U4
    T2 = b.xor(7, 2)  # U7 ^ U2
    T3 = b.xor(7, 1)  # U7 ^ U1
    T4 = b.xor(4, 2)  # U4 ^ U2
    T5 = b.xor(3, 1)  # U3 ^ U1
    T6 = b.xor(T1, T5)  # T1 ^ T5
    T7 = b.xor(6, 5)  # U6 ^ U5
    T8 = b.xor(0, T6)  # U0 ^ T6
    T9 = b.xor(0, T7)  # U0 ^ T7
    T10 = b.xor(T6, T7)  # T6 ^ T7
    T11 = b.xor(6, 2)  # U6 ^ U2
    T12 = b.xor(5, 2)  # U5 ^ U2
    T13 = b.xor(T3, T4)  # T3 ^ T4
    T14 = b.xor(T6, T11)  # T6 ^ T11
    T15 = b.xor(T5, T11)  # T5 ^ T11
    T16 = b.xor(T5, T12)  # T5 ^ T12
    T17 = b.xor(T9, T16)  # T9 ^ T16
    T18 = b.xor(4, 0)  # U4 ^ U0
    T19 = b.xor(T7, T18)  # T7 ^ T18
    T20 = b.xor(T1, T19)  # T1 ^ T19
    T21 = b.xor(1, 0)  # U1 ^ U0
    T22 = b.xor(T7, T21)  # T7 ^ T21
    T23 = b.xor(T2, T22)  # T2 ^ T22
    T24 = b.xor(T2, T10)  # T2 ^ T10
    T25 = b.xor(T20, T17)  # T20 ^ T17
    T26 = b.xor(T3, T16)  # T3 ^ T16
    T27 = b.xor(T1, T12)  # T1 ^ T12

    # ========================================
    # Non-linear middle (32 AND gates)
    # Computes GF(2^8) inversion via tower field
    # ========================================

    M1 = b.and_(T13, T6)
    M2 = b.and_(T23, T8)
    M3 = b.xor(T14, M1)
    M4 = b.and_(T19, 0)  # T19 & U0
    M5 = b.xor(M4, M1)
    M6 = b.and_(T3, T16)
    M7 = b.and_(T22, T9)
    M8 = b.xor(T26, M6)
    M9 = b.and_(T20, T17)
    M10 = b.xor(M9, M6)
    M11 = b.and_(T1, T15)
    M12 = b.and_(T4, T27)
    M13 = b.xor(M12, M11)
    M14 = b.and_(T2, T10)
    M15 = b.xor(M14, M11)
    M16 = b.xor(M3, M2)
    M17 = b.xor(M5, T24)
    M18 = b.xor(M8, M7)
    M19 = b.xor(M10, M15)
    M20 = b.xor(M16, M13)
    M21 = b.xor(M17, M15)
    M22 = b.xor(M18, M13)
    M23 = b.xor(M19, T25)
    M24 = b.xor(M22, M23)
    M25 = b.and_(M22, M20)
    M26 = b.xor(M21, M25)
    M27 = b.xor(M20, M21)
    M28 = b.xor(M23, M25)
    M29 = b.and_(M28, M27)
    M30 = b.and_(M26, M24)
    M31 = b.and_(M20, M23)
    M32 = b.and_(M27, M31)
    M33 = b.xor(M27, M25)
    M34 = b.and_(M21, M22)
    M35 = b.and_(M24, M34)
    M36 = b.xor(M24, M25)
    M37 = b.xor(M21, M29)
    M38 = b.xor(M32, M33)
    M39 = b.xor(M23, M30)
    M40 = b.xor(M35, M36)
    M41 = b.xor(M38, M40)
    M42 = b.xor(M37, M39)
    M43 = b.xor(M37, M38)
    M44 = b.xor(M39, M40)
    M45 = b.xor(M42, M41)
    M46 = b.and_(M44, T6)
    M47 = b.and_(M40, T8)
    M48 = b.and_(M39, 0)  # M39 & U0
    M49 = b.and_(M43, T16)
    M50 = b.and_(M38, T9)
    M51 = b.and_(M37, T17)
    M52 = b.and_(M42, T15)
    M53 = b.and_(M45, T27)
    M54 = b.and_(M41, T10)
    M55 = b.and_(M44, T13)
    M56 = b.and_(M40, T23)
    M57 = b.and_(M39, T19)
    M58 = b.and_(M43, T3)
    M59 = b.and_(M38, T22)
    M60 = b.and_(M37, T20)
    M61 = b.and_(M42, T1)
    M62 = b.and_(M45, T4)
    M63 = b.and_(M41, T2)

    # ========================================
    # Bottom linear layer (69 XOR gates)
    # Maps back from tower basis and applies affine
    # ========================================

    L0 = b.xor(M61, M62)
    L1 = b.xor(M50, M56)
    L2 = b.xor(M46, M48)
    L3 = b.xor(M47, M55)
    L4 = b.xor(M54, M58)
    L5 = b.xor(M49, M61)
    L6 = b.xor(M62, L5)
    L7 = b.xor(M46, L3)
    L8 = b.xor(M51, M59)
    L9 = b.xor(M52, M53)
    L10 = b.xor(M53, L4)
    L11 = b.xor(M60, L2)
    L12 = b.xor(M48, M51)
    L13 = b.xor(M50, L0)
    L14 = b.xor(M52, M61)
    L15 = b.xor(M55, L1)
    L16 = b.xor(M56, L0)
    L17 = b.xor(M57, L1)
    L18 = b.xor(M58, L8)
    L19 = b.xor(M63, L4)
    L20 = b.xor(L0, L1)
    L21 = b.xor(L1, L7)
    L22 = b.xor(L3, L12)
    L23 = b.xor(L18, L2)
    L24 = b.xor(L15, L9)
    L25 = b.xor(L6, L10)
    L26 = b.xor(L7, L9)
    L27 = b.xor(L8, L10)
    L28 = b.xor(L11, L14)
    L29 = b.xor(L11, L17)

    # Final outputs: S[0]-S[7]
    S7 = b.xor(L6, L24)
    S6 = b.xor(L16, L26)
    S5 = b.xor(L19, L28)
    S4 = b.xor(L6, L21)
    S3 = b.xor(L20, L22)
    S2 = b.xor(L25, L29)
    S1 = b.xor(L13, L27)
    S0 = b.xor(L6, L23)

    # Build outputs with inversions for affine constant 0x63 = 01100011
    # Inversions: S[0], S[1], S[5], S[6]
    outputs = [
        (S0, True),  # bit 0: inverted (~)
        (S1, True),  # bit 1: inverted (~)
        (S2, False),  # bit 2
        (S3, False),  # bit 3
        (S4, False),  # bit 4
        (S5, True),  # bit 5: inverted (~)
        (S6, True),  # bit 6: inverted (~)
        (S7, False),  # bit 7
    ]

    return CircuitState(
        input_bits=8,
        output_bits=8,
        gates=b.gates,
        outputs=outputs,
        gate_count=len(b.gates),
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
        print(f"  {name}: {len(errors)} errors out of 256")
        for i, got, exp in errors[:5]:
            print(f"    Input {i:#04x}: got {got:#04x}, expected {exp:#04x}")
        return False
    print(f"  {name}: All 256 values correct!")
    return True


def main():
    print("Building Boyar-Peralta tower field S-box circuit...")
    print()

    circuit = build_tower_field_sbox()
    print(
        f"Circuit: {circuit.gate_count} gates ({circuit.and_count} AND, {circuit.xor_count} XOR)"
    )
    print()

    correct = verify_sbox(circuit, "Tower field S-box")
    print()

    if not correct:
        print("ERROR: Circuit verification failed!")
        print(f"  S-box(0x00) = {circuit.evaluate(0):#04x} (expected 0x63)")
        print(f"  S-box(0x01) = {circuit.evaluate(1):#04x} (expected 0x7c)")
        return

    print("Applying optimizations...")
    print()

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

    if not verify_sbox(opt, "After Phase 1"):
        print("ERROR: Optimization broke correctness!")
        return

    print()
    print("SUCCESS: Tower field S-box circuit is correct!")
    print(f"Final: {opt.gate_count} gates ({opt.and_count} AND, {opt.xor_count} XOR)")


if __name__ == "__main__":
    main()
