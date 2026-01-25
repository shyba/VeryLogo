#!/usr/bin/env python3
"""
Benchmark the BP circuit through the STC compilation pipeline vs. hand-written C.

This validates that:
1. The BP circuit from scripts/bp_circuit_sbox.py is correct
2. STC-generated AVX2 code achieves the same performance as hand-written C
"""

import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.bp_circuit_sbox import build_bp_sbox
from stc.bitslice import AES_SBOX_TABLE


def generate_avx2_c(circuit) -> str:
    """Generate AVX2 C code from CircuitState."""
    lines = []
    lines.append("#include <immintrin.h>")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("// STC-generated bitsliced AES S-box circuit")
    lines.append(
        f"// {circuit.gate_count} gates ({circuit.and_count} AND, {circuit.xor_count} XOR)"
    )
    lines.append("")

    lines.append("void sbox_stc(const uint8_t* input, uint8_t* output) {")

    # Generate bit plane extraction from input (transpose bytes to bit planes)
    lines.append("  // Transpose 32 input bytes to 8 bit planes")
    lines.append("  uint64_t planes[8][4];")
    lines.append("  for (int w = 0; w < 4; w++) {")
    lines.append("    for (int bit = 0; bit < 8; bit++) planes[bit][w] = 0;")
    lines.append("  }")
    lines.append("  for (int i = 0; i < 32; i++) {")
    lines.append("    uint8_t byte = input[i];")
    lines.append("    int word = i / 64;")
    lines.append("    int shift = i % 64;")
    lines.append("    for (int bit = 0; bit < 8; bit++) {")
    lines.append("      if (byte & (1 << bit))")
    lines.append("        planes[bit][word] |= (1ULL << shift);")
    lines.append("    }")
    lines.append("  }")
    lines.append("")

    # Load input bit planes into AVX2 registers
    for i in range(8):
        lines.append(
            f"  __m256i b{i} = _mm256_set_epi64x(planes[{i}][3], planes[{i}][2], planes[{i}][1], planes[{i}][0]);"
        )
    lines.append("")

    # Generate gate operations
    lines.append("  // Execute circuit")
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
    lines.append("  // Collect output bit planes")
    out_names = []
    for bit, (idx, invert) in enumerate(circuit.outputs):
        out_idx = idx
        out_name = f"b{out_idx}" if out_idx < 8 else f"g{out_idx - 8}"
        if invert:
            inv_name = f"out{bit}"
            lines.append(
                f"  __m256i {inv_name} = _mm256_xor_si256({out_name}, _mm256_set1_epi32(-1));"
            )
            out_names.append(inv_name)
        else:
            out_names.append(out_name)
    lines.append("")

    # Transpose bit planes back to bytes
    lines.append("  // Transpose bit planes back to output bytes")
    lines.append("  union { __m256i v; uint64_t u64[4]; } out_planes[8];")
    for i, name in enumerate(out_names):
        lines.append(f"  out_planes[{i}].v = {name};")

    lines.append("  for (int i = 0; i < 32; i++) {")
    lines.append("    int word = i / 64;")
    lines.append("    int shift = i % 64;")
    lines.append("    uint8_t byte = 0;")
    lines.append("    for (int bit = 0; bit < 8; bit++) {")
    lines.append("      if (out_planes[bit].u64[word] & (1ULL << shift))")
    lines.append("        byte |= (1 << bit);")
    lines.append("    }")
    lines.append("    output[i] = byte;")
    lines.append("  }")
    lines.append("}")

    return "\n".join(lines)


def generate_reference_c() -> str:
    """Generate reference BP circuit C code (hand-transcribed from bs.c)."""
    return """#include <immintrin.h>
#include <stdint.h>

// Reference hand-written Boyar-Peralta AES S-box
// Directly from external-bitsliced/bs.c

void sbox_ref(const uint8_t* input, uint8_t* output) {
  // Transpose 32 input bytes to 8 bit planes
  uint64_t planes[8][4];
  for (int w = 0; w < 4; w++) {
    for (int bit = 0; bit < 8; bit++) planes[bit][w] = 0;
  }
  for (int i = 0; i < 32; i++) {
    uint8_t byte = input[i];
    int word = i / 64;
    int shift = i % 64;
    for (int bit = 0; bit < 8; bit++) {
      if (byte & (1 << bit))
        planes[bit][word] |= (1ULL << shift);
    }
  }

  __m256i U[8];
  for (int i = 0; i < 8; i++)
    U[i] = _mm256_set_epi64x(planes[i][3], planes[i][2], planes[i][1], planes[i][0]);

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

  // Output with inversions
  __m256i ones = _mm256_set1_epi32(-1);
  __m256i S7 = _mm256_xor_si256(L6, L24);
  __m256i S6 = _mm256_xor_si256(_mm256_xor_si256(L16, L26), ones);
  __m256i S5 = _mm256_xor_si256(_mm256_xor_si256(L19, L28), ones);
  __m256i S4 = _mm256_xor_si256(L6, L21);
  __m256i S3 = _mm256_xor_si256(L20, L22);
  __m256i S2 = _mm256_xor_si256(L25, L29);
  __m256i S1 = _mm256_xor_si256(_mm256_xor_si256(L13, L27), ones);
  __m256i S0 = _mm256_xor_si256(_mm256_xor_si256(L6, L23), ones);

  // Transpose back to bytes
  union { __m256i v; uint64_t u64[4]; } out_planes[8];
  out_planes[0].v = S0;
  out_planes[1].v = S1;
  out_planes[2].v = S2;
  out_planes[3].v = S3;
  out_planes[4].v = S4;
  out_planes[5].v = S5;
  out_planes[6].v = S6;
  out_planes[7].v = S7;

  for (int i = 0; i < 32; i++) {
    int word = i / 64;
    int shift = i % 64;
    uint8_t byte = 0;
    for (int bit = 0; bit < 8; bit++) {
      if (out_planes[bit].u64[word] & (1ULL << shift))
        byte |= (1 << bit);
    }
    output[i] = byte;
  }
}
"""


def generate_main_c() -> str:
    """Generate benchmark main() function."""
    return """#include <stdio.h>
#include <time.h>
#include <string.h>

void sbox_stc(const unsigned char* input, unsigned char* output);
void sbox_ref(const unsigned char* input, unsigned char* output);

static const unsigned char SBOX[256] = {
  0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
  0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
  0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
  0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
  0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
  0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
  0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
  0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
  0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
  0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
  0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
  0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
  0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
  0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
  0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
  0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

int verify(void (*sbox_fn)(const unsigned char*, unsigned char*), const char* name) {
    unsigned char input[32], output[32];
    int errors = 0;

    // Test all 256 values by running 8 batches of 32
    for (int batch = 0; batch < 8; batch++) {
        for (int i = 0; i < 32; i++) {
            input[i] = (batch * 32 + i) & 0xFF;
        }
        sbox_fn(input, output);
        for (int i = 0; i < 32; i++) {
            int idx = (batch * 32 + i) & 0xFF;
            if (output[i] != SBOX[idx]) {
                if (errors < 5)
                    printf("%s: input %d expected 0x%02x got 0x%02x\\n",
                           name, idx, SBOX[idx], output[i]);
                errors++;
            }
        }
    }

    if (errors == 0) {
        printf("%s: All 256 values correct!\\n", name);
        return 1;
    } else {
        printf("%s: %d errors\\n", name, errors);
        return 0;
    }
}

#define ITERATIONS 10000000

int main() {
    struct timespec start, end;
    double elapsed;

    unsigned char input[32], output[32];
    for (int i = 0; i < 32; i++) input[i] = i;

    printf("STC Pipeline vs. Reference Benchmark\\n");
    printf("=====================================\\n\\n");

    // Verify correctness
    printf("Verification:\\n");
    int stc_ok = verify(sbox_stc, "STC-generated");
    int ref_ok = verify(sbox_ref, "Reference");
    printf("\\n");

    if (!stc_ok || !ref_ok) {
        printf("VERIFICATION FAILED!\\n");
        return 1;
    }

    printf("Performance (iterations: %d x 32 values):\\n", ITERATIONS);

    // Warm up
    for (int i = 0; i < 1000; i++) {
        sbox_stc(input, output);
        sbox_ref(input, output);
    }

    // Benchmark STC-generated
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++)
        sbox_stc(input, output);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double stc_ns = elapsed * 1e9 / (ITERATIONS * 32.0);
    double stc_rate = ITERATIONS * 32.0 / elapsed / 1e6;
    printf("  STC-generated: %.1f ns/eval, %.1fM evals/sec\\n", stc_ns, stc_rate);

    // Benchmark Reference
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++)
        sbox_ref(input, output);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double ref_ns = elapsed * 1e9 / (ITERATIONS * 32.0);
    double ref_rate = ITERATIONS * 32.0 / elapsed / 1e6;
    printf("  Reference:     %.1f ns/eval, %.1fM evals/sec\\n", ref_ns, ref_rate);

    printf("\\nRatio: %.2fx\\n", stc_ns / ref_ns);

    return 0;
}
"""


def main():
    print("Building BP circuit from scripts/bp_circuit_sbox.py...")
    circuit = build_bp_sbox()
    print(
        f"Circuit: {circuit.gate_count} gates ({circuit.and_count} AND, {circuit.xor_count} XOR)"
    )
    print()

    # Verify circuit correctness first
    print("Verifying circuit correctness...")
    errors = 0
    for i in range(256):
        result = circuit.evaluate(i)
        if result != AES_SBOX_TABLE[i]:
            errors += 1
            if errors <= 5:
                print(
                    f"  Error: input {i:#04x} -> {result:#04x}, expected {AES_SBOX_TABLE[i]:#04x}"
                )
    if errors:
        print(f"  {errors} errors total - FAILED!")
        return 1
    print("  All 256 values correct!")
    print()

    # Generate C code
    print("Generating AVX2 C code...")
    stc_code = generate_avx2_c(circuit)
    ref_code = generate_reference_c()
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

        # Compile
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
            print(f"Compilation failed:")
            print(result.stderr)
            return 1
        print()

        # Run benchmark
        print("Running benchmark...")
        print()
        result = subprocess.run([exe_path], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
