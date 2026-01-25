#!/usr/bin/env python3
"""
Simple verification of ternary circuit - compare Python evaluation
to manually computed expected values.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stc.bitslice import AES_SBOX_TABLE
from stc.circuit_synth import CircuitState


def main():
    # Load circuit
    circuit_file = "out/sbox_anf_optimized.json"
    with open(circuit_file) as f:
        data = json.load(f)
    circuit = CircuitState.from_dict(data)

    ternary_count = sum(1 for g in circuit.gates if len(g) == 5)
    print(f"Circuit: {circuit.gate_count} gates ({ternary_count} ternary)")

    # Test a few values with Python evaluation
    print("\nPython evaluation test:")
    errors = 0
    for i in range(256):
        result = circuit.evaluate(i)
        expected = AES_SBOX_TABLE[i]
        if result != expected:
            if errors < 10:
                print(f"  S-box[{i}] = 0x{result:02x}, expected 0x{expected:02x}")
            errors += 1

    if errors == 0:
        print("  All 256 values correct!")
    else:
        print(f"  {errors} errors")

    # Print first few gates to understand structure
    print("\nFirst 10 gates:")
    for i, gate in enumerate(circuit.gates[:10]):
        if len(gate) == 5:
            op, a, b, c, imm8 = gate
            print(f"  {i}: ternary({a}, {b}, {c}, 0x{imm8:02x})")
        else:
            op, left, right = gate
            print(f"  {i}: {op}({left}, {right})")

    # Print first ternary gate and verify its truth table
    for i, gate in enumerate(circuit.gates):
        if len(gate) == 5:
            op, a, b, c, imm8 = gate
            print(f"\nFirst ternary gate (gate {i}):")
            print(f"  ternary(a={a}, b={b}, c={c}, imm8=0x{imm8:02x})")
            print(f"  imm8 binary: {bin(imm8)}")

            # Show truth table
            print("  Truth table:")
            print("    i | a b c | imm8[i]")
            print("   ---|-------|--------")
            for j in range(8):
                a_val = (j >> 2) & 1
                b_val = (j >> 1) & 1
                c_val = j & 1
                result = (imm8 >> j) & 1
                print(f"    {j} | {a_val} {b_val} {c_val} |   {result}")

            # Now show what CircuitState.evaluate does
            print("\n  CircuitState evaluation (idx = (va<<2)|(vb<<1)|vc):")
            print("    va vb vc | idx | imm8[idx]")
            print("   ---------|-----|----------")
            for va in range(2):
                for vb in range(2):
                    for vc in range(2):
                        idx = (va << 2) | (vb << 1) | vc
                        result = (imm8 >> idx) & 1
                        print(f"    {va}  {vb}  {vc}  | {idx:3} |    {result}")

            # Now show what AVX intrinsic does (index = (c<<2)|(b<<1)|a)
            print("\n  AVX intrinsic (index = (c<<2)|(b<<1)|a):")
            print("    a  b  c  | idx | imm8[idx]")
            print("   ---------|-----|----------")
            for va in range(2):
                for vb in range(2):
                    for vc in range(2):
                        idx_avx = (vc << 2) | (vb << 1) | va
                        result = (imm8 >> idx_avx) & 1
                        print(f"    {va}  {vb}  {vc}  | {idx_avx:3} |    {result}")

            break

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
