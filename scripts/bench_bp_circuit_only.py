#!/usr/bin/env python3
"""
Circuit-only benchmark (no transpose overhead).
This measures just the boolean operations.
"""

import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.bp_circuit_sbox import build_bp_sbox


def generate_circuit_only_c(circuit) -> str:
    """Generate AVX2 C code for circuit-only benchmark (no transpose)."""
    lines = []
    lines.append("#include <immintrin.h>")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("// STC-generated bitsliced AES S-box circuit (circuit only)")
    lines.append(
        f"// {circuit.gate_count} gates ({circuit.and_count} AND, {circuit.xor_count} XOR)"
    )
    lines.append("")
    lines.append("// Takes 8 bit planes as input, returns 8 bit planes as output")
    lines.append("void sbox_circuit(__m256i* U, __m256i* S) {")
    lines.append("  __m256i b0 = U[0], b1 = U[1], b2 = U[2], b3 = U[3];")
    lines.append("  __m256i b4 = U[4], b5 = U[5], b6 = U[6], b7 = U[7];")
    lines.append("")

    # Generate gate operations
    for g_idx, (op, left, right) in enumerate(circuit.gates):
        g_name = f"g{g_idx}"
        left_name = f"b{left}" if left < 8 else f"g{left - 8}"
        right_name = f"b{right}" if right < 8 else f"g{right - 8}"

        if op == "xor":
            lines.append(
                f"  __m256i {g_name} = _mm256_xor_si256({left_name}, {right_name});"
            )
        elif op == "and":
            lines.append(
                f"  __m256i {g_name} = _mm256_and_si256({left_name}, {right_name});"
            )
        elif op == "not":
            lines.append(
                f"  __m256i {g_name} = _mm256_xor_si256({left_name}, _mm256_set1_epi32(-1));"
            )
    lines.append("")

    # Generate output collection with inversions
    lines.append("  // Collect outputs")
    for bit, (idx, invert) in enumerate(circuit.outputs):
        out_idx = idx
        out_name = f"b{out_idx}" if out_idx < 8 else f"g{out_idx - 8}"
        if invert:
            lines.append(
                f"  S[{bit}] = _mm256_xor_si256({out_name}, _mm256_set1_epi32(-1));"
            )
        else:
            lines.append(f"  S[{bit}] = {out_name};")

    lines.append("}")
    return "\n".join(lines)


def generate_reference_circuit_c() -> str:
    """Generate reference BP circuit (circuit only)."""
    return """#include <immintrin.h>
#include <stdint.h>

void sbox_ref_circuit(__m256i* U, __m256i* S) {
  // Top linear layer
  __m256i T1 = _mm256_xor_si256(U[7], U[4]);
  __m256i T2 = _mm256_xor_si256(U[7], U[2]);
  __m256i T3 = _mm256_xor_si256(U[7], U[1]);
  __m256i T4 = _mm256_xor_si256(U[4], U[2]);
  __m256i T5 = _mm256_xor_si256(U[3], U[1]);
  __m256i T6 = _mm256_xor_si256(T1, T5);
  __m256i T7 = _mm256_xor_si256(U[6], U[5]);
  __m256i T8 = _mm256_xor_si256(U[0], T6);
  __m256i T9 = _mm256_xor_si256(U[0], T7);
  __m256i T10 = _mm256_xor_si256(T6, T7);
  __m256i T11 = _mm256_xor_si256(U[6], U[2]);
  __m256i T12 = _mm256_xor_si256(U[5], U[2]);
  __m256i T13 = _mm256_xor_si256(T3, T4);
  __m256i T14 = _mm256_xor_si256(T6, T11);
  __m256i T15 = _mm256_xor_si256(T5, T11);
  __m256i T16 = _mm256_xor_si256(T5, T12);
  __m256i T17 = _mm256_xor_si256(T9, T16);
  __m256i T18 = _mm256_xor_si256(U[4], U[0]);
  __m256i T19 = _mm256_xor_si256(T7, T18);
  __m256i T20 = _mm256_xor_si256(T1, T19);
  __m256i T21 = _mm256_xor_si256(U[1], U[0]);
  __m256i T22 = _mm256_xor_si256(T7, T21);
  __m256i T23 = _mm256_xor_si256(T2, T22);
  __m256i T24 = _mm256_xor_si256(T2, T10);
  __m256i T25 = _mm256_xor_si256(T20, T17);
  __m256i T26 = _mm256_xor_si256(T3, T16);
  __m256i T27 = _mm256_xor_si256(T1, T12);

  // Non-linear middle
  __m256i M1 = _mm256_and_si256(T13, T6);
  __m256i M2 = _mm256_and_si256(T23, T8);
  __m256i M3 = _mm256_xor_si256(T14, M1);
  __m256i M4 = _mm256_and_si256(T19, U[0]);
  __m256i M5 = _mm256_xor_si256(M4, M1);
  __m256i M6 = _mm256_and_si256(T3, T16);
  __m256i M7 = _mm256_and_si256(T22, T9);
  __m256i M8 = _mm256_xor_si256(T26, M6);
  __m256i M9 = _mm256_and_si256(T20, T17);
  __m256i M10 = _mm256_xor_si256(M9, M6);
  __m256i M11 = _mm256_and_si256(T1, T15);
  __m256i M12 = _mm256_and_si256(T4, T27);
  __m256i M13 = _mm256_xor_si256(M12, M11);
  __m256i M14 = _mm256_and_si256(T2, T10);
  __m256i M15 = _mm256_xor_si256(M14, M11);
  __m256i M16 = _mm256_xor_si256(M3, M2);
  __m256i M17 = _mm256_xor_si256(M5, T24);
  __m256i M18 = _mm256_xor_si256(M8, M7);
  __m256i M19 = _mm256_xor_si256(M10, M15);
  __m256i M20 = _mm256_xor_si256(M16, M13);
  __m256i M21 = _mm256_xor_si256(M17, M15);
  __m256i M22 = _mm256_xor_si256(M18, M13);
  __m256i M23 = _mm256_xor_si256(M19, T25);
  __m256i M24 = _mm256_xor_si256(M22, M23);
  __m256i M25 = _mm256_and_si256(M22, M20);
  __m256i M26 = _mm256_xor_si256(M21, M25);
  __m256i M27 = _mm256_xor_si256(M20, M21);
  __m256i M28 = _mm256_xor_si256(M23, M25);
  __m256i M29 = _mm256_and_si256(M28, M27);
  __m256i M30 = _mm256_and_si256(M26, M24);
  __m256i M31 = _mm256_and_si256(M20, M23);
  __m256i M32 = _mm256_and_si256(M27, M31);
  __m256i M33 = _mm256_xor_si256(M27, M25);
  __m256i M34 = _mm256_and_si256(M21, M22);
  __m256i M35 = _mm256_and_si256(M24, M34);
  __m256i M36 = _mm256_xor_si256(M24, M25);
  __m256i M37 = _mm256_xor_si256(M21, M29);
  __m256i M38 = _mm256_xor_si256(M32, M33);
  __m256i M39 = _mm256_xor_si256(M23, M30);
  __m256i M40 = _mm256_xor_si256(M35, M36);
  __m256i M41 = _mm256_xor_si256(M38, M40);
  __m256i M42 = _mm256_xor_si256(M37, M39);
  __m256i M43 = _mm256_xor_si256(M37, M38);
  __m256i M44 = _mm256_xor_si256(M39, M40);
  __m256i M45 = _mm256_xor_si256(M42, M41);

  __m256i M46 = _mm256_and_si256(M44, T6);
  __m256i M47 = _mm256_and_si256(M40, T8);
  __m256i M48 = _mm256_and_si256(M39, U[0]);
  __m256i M49 = _mm256_and_si256(M43, T16);
  __m256i M50 = _mm256_and_si256(M38, T9);
  __m256i M51 = _mm256_and_si256(M37, T17);
  __m256i M52 = _mm256_and_si256(M42, T15);
  __m256i M53 = _mm256_and_si256(M45, T27);
  __m256i M54 = _mm256_and_si256(M41, T10);
  __m256i M55 = _mm256_and_si256(M44, T13);
  __m256i M56 = _mm256_and_si256(M40, T23);
  __m256i M57 = _mm256_and_si256(M39, T19);
  __m256i M58 = _mm256_and_si256(M43, T3);
  __m256i M59 = _mm256_and_si256(M38, T22);
  __m256i M60 = _mm256_and_si256(M37, T20);
  __m256i M61 = _mm256_and_si256(M42, T1);
  __m256i M62 = _mm256_and_si256(M45, T4);
  __m256i M63 = _mm256_and_si256(M41, T2);

  // Bottom linear layer
  __m256i L0 = _mm256_xor_si256(M61, M62);
  __m256i L1 = _mm256_xor_si256(M50, M56);
  __m256i L2 = _mm256_xor_si256(M46, M48);
  __m256i L3 = _mm256_xor_si256(M47, M55);
  __m256i L4 = _mm256_xor_si256(M54, M58);
  __m256i L5 = _mm256_xor_si256(M49, M61);
  __m256i L6 = _mm256_xor_si256(M62, L5);
  __m256i L7 = _mm256_xor_si256(M46, L3);
  __m256i L8 = _mm256_xor_si256(M51, M59);
  __m256i L9 = _mm256_xor_si256(M52, M53);
  __m256i L10 = _mm256_xor_si256(M53, L4);
  __m256i L11 = _mm256_xor_si256(M60, L2);
  __m256i L12 = _mm256_xor_si256(M48, M51);
  __m256i L13 = _mm256_xor_si256(M50, L0);
  __m256i L14 = _mm256_xor_si256(M52, M61);
  __m256i L15 = _mm256_xor_si256(M55, L1);
  __m256i L16 = _mm256_xor_si256(M56, L0);
  __m256i L17 = _mm256_xor_si256(M57, L1);
  __m256i L18 = _mm256_xor_si256(M58, L8);
  __m256i L19 = _mm256_xor_si256(M63, L4);
  __m256i L20 = _mm256_xor_si256(L0, L1);
  __m256i L21 = _mm256_xor_si256(L1, L7);
  __m256i L22 = _mm256_xor_si256(L3, L12);
  __m256i L23 = _mm256_xor_si256(L18, L2);
  __m256i L24 = _mm256_xor_si256(L15, L9);
  __m256i L25 = _mm256_xor_si256(L6, L10);
  __m256i L26 = _mm256_xor_si256(L7, L9);
  __m256i L27 = _mm256_xor_si256(L8, L10);
  __m256i L28 = _mm256_xor_si256(L11, L14);
  __m256i L29 = _mm256_xor_si256(L11, L17);

  __m256i ones = _mm256_set1_epi32(-1);
  S[7] = _mm256_xor_si256(L6, L24);
  S[6] = _mm256_xor_si256(_mm256_xor_si256(L16, L26), ones);
  S[5] = _mm256_xor_si256(_mm256_xor_si256(L19, L28), ones);
  S[4] = _mm256_xor_si256(L6, L21);
  S[3] = _mm256_xor_si256(L20, L22);
  S[2] = _mm256_xor_si256(L25, L29);
  S[1] = _mm256_xor_si256(_mm256_xor_si256(L13, L27), ones);
  S[0] = _mm256_xor_si256(_mm256_xor_si256(L6, L23), ones);
}
"""


def generate_main_c() -> str:
    return """#include <stdio.h>
#include <time.h>
#include <immintrin.h>

void sbox_circuit(__m256i* U, __m256i* S);
void sbox_ref_circuit(__m256i* U, __m256i* S);

#define ITERATIONS 100000000

int main() {
    struct timespec start, end;
    double elapsed;

    __m256i U[8], S[8];
    for (int i = 0; i < 8; i++)
        U[i] = _mm256_set1_epi32(i * 0x11111111);

    printf("Circuit-Only Benchmark (no transpose)\\n");
    printf("======================================\\n\\n");
    printf("Iterations: %d x 256 values\\n\\n", ITERATIONS);

    // Warm up
    for (int i = 0; i < 10000; i++) {
        sbox_circuit(U, S);
        sbox_ref_circuit(U, S);
    }

    // Benchmark STC-generated
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++)
        sbox_circuit(U, S);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double stc_ns = elapsed * 1e9 / (ITERATIONS * 256.0);
    double stc_rate = ITERATIONS * 256.0 / elapsed / 1e9;
    printf("STC-generated: %.3f ns/eval, %.1fB evals/sec\\n", stc_ns, stc_rate);

    // Benchmark Reference
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++)
        sbox_ref_circuit(U, S);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double ref_ns = elapsed * 1e9 / (ITERATIONS * 256.0);
    double ref_rate = ITERATIONS * 256.0 / elapsed / 1e9;
    printf("Reference:     %.3f ns/eval, %.1fB evals/sec\\n", ref_ns, ref_rate);

    printf("\\nRatio: %.2fx\\n", stc_ns / ref_ns);
    printf("\\nNote: benchmark.md shows 0.028 ns/eval, 35.5B evals/sec\\n");

    return 0;
}
"""


def main():
    print("Building BP circuit...")
    circuit = build_bp_sbox()
    print(
        f"Circuit: {circuit.gate_count} gates ({circuit.and_count} AND, {circuit.xor_count} XOR)"
    )
    print()

    print("Generating circuit-only C code...")
    stc_code = generate_circuit_only_c(circuit)
    ref_code = generate_reference_circuit_c()
    main_code = generate_main_c()

    with tempfile.TemporaryDirectory() as tmpdir:
        stc_path = os.path.join(tmpdir, "sbox_stc.c")
        ref_path = os.path.join(tmpdir, "sbox_ref.c")
        main_path = os.path.join(tmpdir, "main.c")
        exe_path = os.path.join(tmpdir, "bench")

        with open(stc_path, "w") as f:
            f.write(stc_code)
        with open(ref_path, "w") as f:
            f.write(ref_code)
        with open(main_path, "w") as f:
            f.write(main_code)

        print("Compiling...")
        cmd = [
            "cc",
            "-O3",
            "-march=native",
            "-mavx2",
            "-o",
            exe_path,
            stc_path,
            ref_path,
            main_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Compilation failed: {result.stderr}")
            return 1
        print()

        print("Running benchmark...")
        print()
        result = subprocess.run([exe_path], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
