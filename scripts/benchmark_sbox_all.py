#!/usr/bin/env python3
"""
Comprehensive S-box implementation benchmark.

Compares:
1. AES-NI hardware instruction (if available) - full AES round
2. Simple C table lookup - cache-based, potential timing side-channel
3. Boyar-Peralta bitsliced (~115 gates) - optimal known circuit
4. Our ANF bitsliced uint64 (1145 gates)
5. Our ANF bitsliced SSE2 (16 parallel)
6. Our ANF bitsliced AVX2 (32 parallel)
7. Python evaluation (reference)

The Boyar-Peralta circuit is the optimal known circuit for AES S-box.
Our ANF-synthesized circuit (1145 gates) is a baseline for comparison.

Usage:
    python scripts/benchmark_sbox_all.py
    python scripts/benchmark_sbox_all.py --iterations 10000000
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.setrecursionlimit(10000)

from stc.bitslice import AES_SBOX_TABLE
from stc.bitslice_codegen import (
    AVX2_CONFIG,
    SSE2_CONFIG,
    UINT64_CONFIG,
    generate_bitslice_c,
)
from stc.circuit_synth import IncrementalOptimizer


BENCHMARK_TEMPLATE = """
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
{extra_includes}

// AES S-box table
static const uint8_t SBOX[256] = {{
{sbox_table}
}};

// ============== Simple Table Lookup ==============
static inline uint8_t sbox_table(uint8_t x) {{
    return SBOX[x];
}}

void bench_table(int iterations) {{
    volatile uint8_t result = 0;
    clock_t start = clock();
    for (int iter = 0; iter < iterations; iter++) {{
        for (int i = 0; i < 256; i++) {{
            result = sbox_table((uint8_t)i);
        }}
    }}
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double evals = (double)iterations * 256;
    printf("Table lookup:      %8.1f ns/eval, %10.0f evals/sec\\n",
           (elapsed / evals) * 1e9, evals / elapsed);
    (void)result;
}}

{aesni_code}

{boyar_peralta_code}

{bitslice_uint64_code}

{bitslice_sse2_code}

{bitslice_avx2_code}

int main(int argc, char** argv) {{
    int iterations = 1000000;
    if (argc > 1) iterations = atoi(argv[1]);

    printf("S-box Implementation Benchmark\\n");
    printf("==============================\\n");
    printf("Iterations: %d (x 256 values each)\\n\\n", iterations);

    bench_table(iterations);
{aesni_bench_call}
{bp_avx2_bench_call}
    bench_bp_u64_circuit(iterations);
    bench_boyar_peralta(iterations);
    bench_bitslice_u64(iterations);
{sse2_bench_call}
{avx2_bench_call}

    return 0;
}}
"""

AESNI_CODE = """
#ifdef __AES__
#include <wmmintrin.h>

// AES-NI SubBytes via AESENC with zero round key
// Note: This computes full AES round, we extract SubBytes+ShiftRows+MixColumns
// For pure SubBytes, we'd need inverse MixColumns which adds overhead
void bench_aesni_round(int iterations) {
    __m128i state = _mm_setzero_si128();
    __m128i key = _mm_setzero_si128();
    clock_t start = clock();
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < 16; i++) {  // 16 rounds to match 256 byte-evals
            state = _mm_aesenc_si128(state, key);
        }
    }
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    // Each AESENC processes 16 bytes, so 16 rounds = 256 bytes
    double evals = (double)iterations * 256;
    printf("AES-NI (full rnd): %8.1f ns/eval, %10.0f evals/sec\\n",
           (elapsed / evals) * 1e9, evals / elapsed);
    // Prevent optimization
    volatile __m128i sink = state;
    (void)sink;
}
#endif
"""

AESNI_BENCH_CALL = """
#ifdef __AES__
    bench_aesni_round(iterations);
#endif
"""

BOYAR_PERALTA_CODE = """
// Boyar-Peralta AES S-box circuit (~115 gates)
// From: https://github.com/conorpp/bitsliced-aes
// Based on: "A New Combinational Logic Minimization Technique" by Boyar & Peralta

// AVX2 version for 256 parallel evaluations
#ifdef __AVX2__
static inline void bs_sbox_bp_avx2(__m256i U[8]) {
    __m256i T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16;
    __m256i T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27;
    __m256i M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12,M13,M14,M15;
    __m256i M16,M17,M18,M19,M20,M21,M22,M23,M24,M25,M26,M27,M28,M29;
    __m256i M30,M31,M32,M33,M34,M35,M36,M37,M38,M39,M40,M41,M42,M43;
    __m256i M44,M45,M46,M47,M48,M49,M50,M51,M52,M53,M54,M55,M56,M57;
    __m256i M58,M59,M60,M61,M62,M63;
    __m256i L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12,L13,L14;
    __m256i L15,L16,L17,L18,L19,L20,L21,L22,L23,L24,L25,L26,L27,L28,L29;
    __m256i S[8];
    __m256i ones = _mm256_set1_epi32(-1);
    #define XOR(a,b) _mm256_xor_si256(a,b)
    #define AND(a,b) _mm256_and_si256(a,b)
    #define NOT(a) _mm256_xor_si256(a, ones)
    T1 = XOR(U[7], U[4]); T2 = XOR(U[7], U[2]); T3 = XOR(U[7], U[1]); T4 = XOR(U[4], U[2]);
    T5 = XOR(U[3], U[1]); T6 = XOR(T1, T5); T7 = XOR(U[6], U[5]); T8 = XOR(U[0], T6);
    T9 = XOR(U[0], T7); T10 = XOR(T6, T7); T11 = XOR(U[6], U[2]); T12 = XOR(U[5], U[2]);
    T13 = XOR(T3, T4); T14 = XOR(T6, T11); T15 = XOR(T5, T11); T16 = XOR(T5, T12);
    T17 = XOR(T9, T16); T18 = XOR(U[4], U[0]); T19 = XOR(T7, T18); T20 = XOR(T1, T19);
    T21 = XOR(U[1], U[0]); T22 = XOR(T7, T21); T23 = XOR(T2, T22); T24 = XOR(T2, T10);
    T25 = XOR(T20, T17); T26 = XOR(T3, T16); T27 = XOR(T1, T12);
    M1 = AND(T13, T6); M2 = AND(T23, T8); M3 = XOR(T14, M1); M4 = AND(T19, U[0]); M5 = XOR(M4, M1);
    M6 = AND(T3, T16); M7 = AND(T22, T9); M8 = XOR(T26, M6); M9 = AND(T20, T17); M10 = XOR(M9, M6);
    M11 = AND(T1, T15); M12 = AND(T4, T27); M13 = XOR(M12, M11); M14 = AND(T2, T10); M15 = XOR(M14, M11);
    M16 = XOR(M3, M2); M17 = XOR(M5, T24); M18 = XOR(M8, M7); M19 = XOR(M10, M15);
    M20 = XOR(M16, M13); M21 = XOR(M17, M15); M22 = XOR(M18, M13); M23 = XOR(M19, T25);
    M24 = XOR(M22, M23); M25 = AND(M22, M20); M26 = XOR(M21, M25); M27 = XOR(M20, M21);
    M28 = XOR(M23, M25); M29 = AND(M28, M27); M30 = AND(M26, M24); M31 = AND(M20, M23);
    M32 = AND(M27, M31); M33 = XOR(M27, M25); M34 = AND(M21, M22); M35 = AND(M24, M34);
    M36 = XOR(M24, M25); M37 = XOR(M21, M29); M38 = XOR(M32, M33); M39 = XOR(M23, M30);
    M40 = XOR(M35, M36); M41 = XOR(M38, M40); M42 = XOR(M37, M39); M43 = XOR(M37, M38);
    M44 = XOR(M39, M40); M45 = XOR(M42, M41);
    M46 = AND(M44, T6); M47 = AND(M40, T8); M48 = AND(M39, U[0]); M49 = AND(M43, T16);
    M50 = AND(M38, T9); M51 = AND(M37, T17); M52 = AND(M42, T15); M53 = AND(M45, T27);
    M54 = AND(M41, T10); M55 = AND(M44, T13); M56 = AND(M40, T23); M57 = AND(M39, T19);
    M58 = AND(M43, T3); M59 = AND(M38, T22); M60 = AND(M37, T20); M61 = AND(M42, T1);
    M62 = AND(M45, T4); M63 = AND(M41, T2);
    L0 = XOR(M61, M62); L1 = XOR(M50, M56); L2 = XOR(M46, M48); L3 = XOR(M47, M55);
    L4 = XOR(M54, M58); L5 = XOR(M49, M61); L6 = XOR(M62, L5); L7 = XOR(M46, L3);
    L8 = XOR(M51, M59); L9 = XOR(M52, M53); L10 = XOR(M53, L4); L11 = XOR(M60, L2);
    L12 = XOR(M48, M51); L13 = XOR(M50, L0); L14 = XOR(M52, M61); L15 = XOR(M55, L1);
    L16 = XOR(M56, L0); L17 = XOR(M57, L1); L18 = XOR(M58, L8); L19 = XOR(M63, L4);
    L20 = XOR(L0, L1); L21 = XOR(L1, L7); L22 = XOR(L3, L12); L23 = XOR(L18, L2);
    L24 = XOR(L15, L9); L25 = XOR(L6, L10); L26 = XOR(L7, L9); L27 = XOR(L8, L10);
    L28 = XOR(L11, L14); L29 = XOR(L11, L17);
    S[7] = XOR(L6, L24); S[6] = NOT(XOR(L16, L26)); S[5] = NOT(XOR(L19, L28)); S[4] = XOR(L6, L21);
    S[3] = XOR(L20, L22); S[2] = XOR(L25, L29); S[1] = NOT(XOR(L13, L27)); S[0] = NOT(XOR(L6, L23));
    #undef XOR
    #undef AND
    #undef NOT
    U[0] = S[0]; U[1] = S[1]; U[2] = S[2]; U[3] = S[3];
    U[4] = S[4]; U[5] = S[5]; U[6] = S[6]; U[7] = S[7];
}

void bench_bp_avx2_circuit(int iterations) {
    volatile __m256i planes[8];
    __m256i work[8];
    for (int i = 0; i < 8; i++) planes[i] = _mm256_set1_epi64x(0x123 + i);

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < 8; j++) work[j] = planes[j];
        bs_sbox_bp_avx2(work);
        planes[0] = work[0];
    }
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double elapsed = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    double evals = (double)iterations * 256;
    printf("BP AVX2 (circuit): %8.3f ns/eval, %10.0f evals/sec\\n",
           (elapsed / evals) * 1e9, evals / elapsed);
}
#endif

// uint64 version for 64 parallel evaluations
static inline void bs_sbox_bp(uint64_t U[8]) {
    uint64_t T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16;
    uint64_t T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27;
    uint64_t M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12,M13,M14,M15;
    uint64_t M16,M17,M18,M19,M20,M21,M22,M23,M24,M25,M26,M27,M28,M29;
    uint64_t M30,M31,M32,M33,M34,M35,M36,M37,M38,M39,M40,M41,M42,M43;
    uint64_t M44,M45,M46,M47,M48,M49,M50,M51,M52,M53,M54,M55,M56,M57;
    uint64_t M58,M59,M60,M61,M62,M63;
    uint64_t L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12,L13,L14;
    uint64_t L15,L16,L17,L18,L19,L20,L21,L22,L23,L24,L25,L26,L27,L28,L29;
    uint64_t S[8];

    T1 = U[7] ^ U[4];
    T2 = U[7] ^ U[2];
    T3 = U[7] ^ U[1];
    T4 = U[4] ^ U[2];
    T5 = U[3] ^ U[1];
    T6 = T1 ^ T5;
    T7 = U[6] ^ U[5];
    T8 = U[0] ^ T6;
    T9 = U[0] ^ T7;
    T10 = T6 ^ T7;
    T11 = U[6] ^ U[2];
    T12 = U[5] ^ U[2];
    T13 = T3 ^ T4;
    T14 = T6 ^ T11;
    T15 = T5 ^ T11;
    T16 = T5 ^ T12;
    T17 = T9 ^ T16;
    T18 = U[4] ^ U[0];
    T19 = T7 ^ T18;
    T20 = T1 ^ T19;
    T21 = U[1] ^ U[0];
    T22 = T7 ^ T21;
    T23 = T2 ^ T22;
    T24 = T2 ^ T10;
    T25 = T20 ^ T17;
    T26 = T3 ^ T16;
    T27 = T1 ^ T12;
    M1 = T13 & T6;
    M2 = T23 & T8;
    M3 = T14 ^ M1;
    M4 = T19 & U[0];
    M5 = M4 ^ M1;
    M6 = T3 & T16;
    M7 = T22 & T9;
    M8 = T26 ^ M6;
    M9 = T20 & T17;
    M10 = M9 ^ M6;
    M11 = T1 & T15;
    M12 = T4 & T27;
    M13 = M12 ^ M11;
    M14 = T2 & T10;
    M15 = M14 ^ M11;
    M16 = M3 ^ M2;
    M17 = M5 ^ T24;
    M18 = M8 ^ M7;
    M19 = M10 ^ M15;
    M20 = M16 ^ M13;
    M21 = M17 ^ M15;
    M22 = M18 ^ M13;
    M23 = M19 ^ T25;
    M24 = M22 ^ M23;
    M25 = M22 & M20;
    M26 = M21 ^ M25;
    M27 = M20 ^ M21;
    M28 = M23 ^ M25;
    M29 = M28 & M27;
    M30 = M26 & M24;
    M31 = M20 & M23;
    M32 = M27 & M31;
    M33 = M27 ^ M25;
    M34 = M21 & M22;
    M35 = M24 & M34;
    M36 = M24 ^ M25;
    M37 = M21 ^ M29;
    M38 = M32 ^ M33;
    M39 = M23 ^ M30;
    M40 = M35 ^ M36;
    M41 = M38 ^ M40;
    M42 = M37 ^ M39;
    M43 = M37 ^ M38;
    M44 = M39 ^ M40;
    M45 = M42 ^ M41;
    M46 = M44 & T6;
    M47 = M40 & T8;
    M48 = M39 & U[0];
    M49 = M43 & T16;
    M50 = M38 & T9;
    M51 = M37 & T17;
    M52 = M42 & T15;
    M53 = M45 & T27;
    M54 = M41 & T10;
    M55 = M44 & T13;
    M56 = M40 & T23;
    M57 = M39 & T19;
    M58 = M43 & T3;
    M59 = M38 & T22;
    M60 = M37 & T20;
    M61 = M42 & T1;
    M62 = M45 & T4;
    M63 = M41 & T2;
    L0 = M61 ^ M62;
    L1 = M50 ^ M56;
    L2 = M46 ^ M48;
    L3 = M47 ^ M55;
    L4 = M54 ^ M58;
    L5 = M49 ^ M61;
    L6 = M62 ^ L5;
    L7 = M46 ^ L3;
    L8 = M51 ^ M59;
    L9 = M52 ^ M53;
    L10 = M53 ^ L4;
    L11 = M60 ^ L2;
    L12 = M48 ^ M51;
    L13 = M50 ^ L0;
    L14 = M52 ^ M61;
    L15 = M55 ^ L1;
    L16 = M56 ^ L0;
    L17 = M57 ^ L1;
    L18 = M58 ^ L8;
    L19 = M63 ^ L4;
    L20 = L0 ^ L1;
    L21 = L1 ^ L7;
    L22 = L3 ^ L12;
    L23 = L18 ^ L2;
    L24 = L15 ^ L9;
    L25 = L6 ^ L10;
    L26 = L7 ^ L9;
    L27 = L8 ^ L10;
    L28 = L11 ^ L14;
    L29 = L11 ^ L17;
    S[7] = L6 ^ L24;
    S[6] = ~(L16 ^ L26);
    S[5] = ~(L19 ^ L28);
    S[4] = L6 ^ L21;
    S[3] = L20 ^ L22;
    S[2] = L25 ^ L29;
    S[1] = ~(L13 ^ L27);
    S[0] = ~(L6 ^ L23);

    U[0] = S[0]; U[1] = S[1]; U[2] = S[2]; U[3] = S[3];
    U[4] = S[4]; U[5] = S[5]; U[6] = S[6]; U[7] = S[7];
}

static inline void bp_transpose_to_planes(const uint8_t* input, uint64_t planes[8]) {
    for (int bit = 0; bit < 8; bit++) planes[bit] = 0;
    for (int i = 0; i < 64; i++) {
        uint8_t byte = input[i];
        for (int bit = 0; bit < 8; bit++) {
            if (byte & (1 << bit))
                planes[bit] |= (1ULL << i);
        }
    }
}

static inline void bp_transpose_from_planes(uint64_t planes[8], uint8_t* output) {
    for (int i = 0; i < 64; i++) {
        uint8_t byte = 0;
        for (int bit = 0; bit < 8; bit++) {
            if (planes[bit] & (1ULL << i))
                byte |= (1 << bit);
        }
        output[i] = byte;
    }
}

void bench_bp_u64_circuit(int iterations) {
    volatile uint64_t planes[8] = {0x123, 0x456, 0x789, 0xabc, 0xdef, 0x111, 0x222, 0x333};
    uint64_t work[8];

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < 8; j++) work[j] = planes[j];
        bs_sbox_bp(work);
        planes[0] = work[0];
    }
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double elapsed = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    double evals = (double)iterations * 64;
    printf("BP u64 (circuit):  %8.3f ns/eval, %10.0f evals/sec\\n",
           (elapsed / evals) * 1e9, evals / elapsed);
}

void bench_boyar_peralta(int iterations) {
    volatile uint8_t input[64];
    volatile uint8_t output[64];
    uint64_t planes[8];
    for (int i = 0; i < 64; i++) input[i] = i;

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    int total_batches = iterations * 4;
    for (int batch = 0; batch < total_batches; batch++) {
        bp_transpose_to_planes((uint8_t*)input, planes);
        bs_sbox_bp(planes);
        bp_transpose_from_planes(planes, (uint8_t*)output);
        input[0] = output[0];
    }
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double elapsed = (ts_end.tv_sec - ts_start.tv_sec) +
                     (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    double evals = (double)total_batches * 64;
    printf("BP u64 (w/xpose):  %8.1f ns/eval, %10.0f evals/sec\\n",
           (elapsed / evals) * 1e9, evals / elapsed);
}
"""


def generate_sbox_table_c() -> str:
    """Generate C array initializer for S-box table."""
    lines = []
    for i in range(0, 256, 16):
        row = ", ".join(
            f"0x{AES_SBOX_TABLE[j]:02x}" for j in range(i, min(i + 16, 256))
        )
        lines.append(f"    {row},")
    return "\n".join(lines)


def generate_bitslice_bench_u64(state) -> str:
    """Generate uint64 bitslice benchmark code."""
    code = generate_bitslice_c(state, UINT64_CONFIG, "sbox_bitslice_u64")

    bench = """
void bench_bitslice_u64(int iterations) {
    uint8_t input[8];
    uint8_t output[8];
    for (int i = 0; i < 8; i++) input[i] = i;

    clock_t start = clock();
    for (int iter = 0; iter < iterations; iter++) {
        for (int batch = 0; batch < 32; batch++) {  // 32 batches x 8 = 256
            sbox_bitslice_u64(input, output);
            input[0] = output[0];
        }
    }
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double evals = (double)iterations * 256;
    printf("Bitslice uint64:   %8.1f ns/eval, %10.0f evals/sec\\n",
           (elapsed / evals) * 1e9, evals / elapsed);
}
"""
    return code + "\n" + bench


def generate_bitslice_bench_sse2(state) -> str:
    """Generate SSE2 bitslice benchmark code."""
    code = generate_bitslice_c(state, SSE2_CONFIG, "sbox_bitslice_sse2")

    bench = """
void bench_bitslice_sse2(int iterations) {
    uint8_t input[16];
    uint8_t output[16];
    for (int i = 0; i < 16; i++) input[i] = i;

    clock_t start = clock();
    for (int iter = 0; iter < iterations; iter++) {
        for (int batch = 0; batch < 16; batch++) {  // 16 batches x 16 = 256
            sbox_bitslice_sse2(input, output);
            input[0] = output[0];
        }
    }
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double evals = (double)iterations * 256;
    printf("Bitslice SSE2:     %8.1f ns/eval, %10.0f evals/sec\\n",
           (elapsed / evals) * 1e9, evals / elapsed);
}
"""
    return code + "\n" + bench


def generate_bitslice_bench_avx2(state) -> str:
    """Generate AVX2 bitslice benchmark code."""
    code = generate_bitslice_c(state, AVX2_CONFIG, "sbox_bitslice_avx2")

    bench = """
#ifdef __AVX2__
void bench_bitslice_avx2(int iterations) {
    uint8_t input[32];
    uint8_t output[32];
    for (int i = 0; i < 32; i++) input[i] = i;

    clock_t start = clock();
    for (int iter = 0; iter < iterations; iter++) {
        for (int batch = 0; batch < 8; batch++) {  // 8 batches x 32 = 256
            sbox_bitslice_avx2(input, output);
            input[0] = output[0];
        }
    }
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double evals = (double)iterations * 256;
    printf("Bitslice AVX2:     %8.1f ns/eval, %10.0f evals/sec\\n",
           (elapsed / evals) * 1e9, evals / elapsed);
}
#endif
"""
    return code + "\n" + bench


def run_python_benchmark(opt: IncrementalOptimizer, iterations: int) -> None:
    """Run Python reference benchmark."""
    state = opt.best_state

    start = time.perf_counter()
    for _ in range(iterations):
        for i in range(256):
            state.evaluate(i)
    elapsed = time.perf_counter() - start

    evals = iterations * 256
    ns_per = (elapsed / evals) * 1e9
    per_sec = evals / elapsed
    print(f"Python evaluate:   {ns_per:8.1f} ns/eval, {per_sec:10.0f} evals/sec")


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive S-box benchmark")
    parser.add_argument("--iterations", type=int, default=1000000)
    parser.add_argument("--load", type=str, help="Load circuit from JSON")
    parser.add_argument(
        "--python-only", action="store_true", help="Only run Python benchmark"
    )
    args = parser.parse_args()

    if args.load:
        print(f"Loading circuit from {args.load}...")
        opt = IncrementalOptimizer.load(args.load)
    else:
        print("Generating ANF circuit...")
        opt = IncrementalOptimizer(AES_SBOX_TABLE, input_bits=8, output_bits=8)

    state = opt.best_state
    print(f"Circuit: {state.gate_count} gates")
    print()

    if args.python_only:
        print("S-box Implementation Benchmark (Python only)")
        print("=" * 45)
        print(f"Iterations: {args.iterations} (x 256 values each)")
        print()
        run_python_benchmark(opt, min(args.iterations, 1000))
        return

    sbox_table = generate_sbox_table_c()
    bitslice_u64 = generate_bitslice_bench_u64(state)
    bitslice_sse2 = generate_bitslice_bench_sse2(state)
    bitslice_avx2 = generate_bitslice_bench_avx2(state)

    has_avx2 = (
        subprocess.run(
            ["grep", "-q", "avx2", "/proc/cpuinfo"], capture_output=True
        ).returncode
        == 0
    )

    has_aesni = (
        subprocess.run(
            ["grep", "-q", "aes", "/proc/cpuinfo"], capture_output=True
        ).returncode
        == 0
    )

    avx2_bench_call = "    bench_bitslice_avx2(iterations);" if has_avx2 else ""
    bp_avx2_bench_call = "    bench_bp_avx2_circuit(iterations);" if has_avx2 else ""
    sse2_bench_call = "    bench_bitslice_sse2(iterations);"
    aesni_bench_call = AESNI_BENCH_CALL if has_aesni else ""
    extra_includes = "#include <immintrin.h>" if has_avx2 else "#include <emmintrin.h>"

    code = BENCHMARK_TEMPLATE.format(
        sbox_table=sbox_table,
        aesni_code=AESNI_CODE if has_aesni else "",
        boyar_peralta_code=BOYAR_PERALTA_CODE,
        bitslice_uint64_code=bitslice_u64,
        bitslice_sse2_code=bitslice_sse2,
        bitslice_avx2_code=bitslice_avx2 if has_avx2 else "",
        aesni_bench_call=aesni_bench_call,
        bp_avx2_bench_call=bp_avx2_bench_call,
        sse2_bench_call=sse2_bench_call,
        avx2_bench_call=avx2_bench_call,
        extra_includes=extra_includes,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        c_file = os.path.join(tmpdir, "bench.c")
        exe_file = os.path.join(tmpdir, "bench")

        with open(c_file, "w") as f:
            f.write(code)

        cflags = ["-O3", "-march=native"]
        if has_aesni:
            cflags.append("-maes")

        compile_cmd = ["gcc"] + cflags + ["-o", exe_file, c_file]
        print(f"Compiling with: {' '.join(compile_cmd)}")

        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Compilation failed:\n{result.stderr}")
            with open("debug_bench.c", "w") as f:
                f.write(code)
            print("Saved to debug_bench.c")
            return

        print()
        result = subprocess.run(
            [exe_file, str(args.iterations)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        print()
        print("Python reference (slower, for comparison):")
        run_python_benchmark(opt, min(args.iterations // 1000, 100))


if __name__ == "__main__":
    main()
