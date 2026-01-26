#!/usr/bin/env python3
"""
Benchmark the ternary-optimized S-box circuit against other implementations.

Compares:
1. BP AVX2 (115 gates) - optimal known circuit
2. Ternary AVX-512 (828 gates, 349 vpternlogd) - our optimized circuit
3. Original ANF AVX2 (1145 gates) - baseline
4. Table lookup
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stc.bitslice import AES_SBOX_TABLE
from stc.circuit_synth import CircuitState


def load_optimized_circuit():
    """Load the ternary-optimized circuit from JSON."""
    circuit_file = "out/sbox_anf_optimized.json"
    if not os.path.exists(circuit_file):
        print(f"Error: {circuit_file} not found. Run depth_aware_avx512_sbox.py first.")
        sys.exit(1)

    with open(circuit_file) as f:
        data = json.load(f)
    return CircuitState.from_dict(data)


def generate_ternary_avx512_code(circuit: CircuitState, func_name: str) -> str:
    """Generate AVX-512 code with vpternlogd for ternary gates."""
    lines = []
    lines.append(f"static inline void {func_name}(__m512i* in, __m512i* out) {{")

    # Count max temp vars needed
    num_gates = len(circuit.gates)
    lines.append(f"    __m512i t[{num_gates}];")
    lines.append("    __m512i ones = _mm512_set1_epi64(-1);")
    lines.append("")

    input_bits = circuit.input_bits

    def get_reg(idx):
        if idx < input_bits:
            return f"in[{idx}]"
        return f"t[{idx - input_bits}]"

    for g_idx, gate in enumerate(circuit.gates):
        if len(gate) == 5:
            # Ternary gate: ("ternary", a, b, c, imm8)
            _, a, b, c, imm8 = gate
            # Match the repository's ternary truth-table convention:
            # imm8 bit i corresponds to i = (a<<2)|(b<<1)|c, which is also the
            # operand order used by the AVX-512 vpternlog* intrinsics here.
            lines.append(
                f"    t[{g_idx}] = _mm512_ternarylogic_epi64({get_reg(a)}, {get_reg(b)}, {get_reg(c)}, 0x{imm8:02x});"
            )
        else:
            op, left, right = gate
            if op == "xor":
                lines.append(
                    f"    t[{g_idx}] = _mm512_xor_si512({get_reg(left)}, {get_reg(right)});"
                )
            elif op == "and":
                lines.append(
                    f"    t[{g_idx}] = _mm512_and_si512({get_reg(left)}, {get_reg(right)});"
                )
            elif op == "or":
                lines.append(
                    f"    t[{g_idx}] = _mm512_or_si512({get_reg(left)}, {get_reg(right)});"
                )
            elif op == "not":
                lines.append(
                    f"    t[{g_idx}] = _mm512_xor_si512({get_reg(left)}, ones);"
                )
            elif op == "const":
                if left == 0:
                    lines.append(f"    t[{g_idx}] = _mm512_setzero_si512();")
                else:
                    lines.append(f"    t[{g_idx}] = ones;")

    lines.append("")
    for i, (out_idx, inverted) in enumerate(circuit.outputs):
        if inverted:
            lines.append(f"    out[{i}] = _mm512_xor_si512({get_reg(out_idx)}, ones);")
        else:
            lines.append(f"    out[{i}] = {get_reg(out_idx)};")

    lines.append("}")
    return "\n".join(lines)


def generate_benchmark_code(circuit: CircuitState) -> str:
    """Generate complete benchmark code."""

    ternary_code = generate_ternary_avx512_code(circuit, "sbox_ternary_avx512")
    ternary_count = sum(1 for g in circuit.gates if len(g) == 5)
    binary_count = len(circuit.gates) - ternary_count

    sbox_table = ",\n    ".join(
        ", ".join(f"0x{AES_SBOX_TABLE[i+j]:02x}" for j in range(16))
        for i in range(0, 256, 16)
    )

    code = f"""
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>

static const uint8_t SBOX[256] = {{
    {sbox_table}
}};

// ============== Table Lookup ==============
void bench_table(int iterations) {{
    volatile uint8_t result = 0;
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    for (int iter = 0; iter < iterations; iter++) {{
        for (int i = 0; i < 256; i++) {{
            result = SBOX[(uint8_t)i];
        }}
    }}
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double elapsed = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    double evals = (double)iterations * 256;
    printf("Table lookup:           %8.2f ns/eval, %10.1fM evals/sec\\n",
           (elapsed / evals) * 1e9, evals / elapsed / 1e6);
}}

// ============== Boyar-Peralta AVX2 (~115 gates) ==============
static inline void bs_sbox_bp_avx2(__m256i U[8]) {{
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
}}

void bench_bp_avx2(int iterations) {{
    volatile __m256i planes[8];
    __m256i work[8];
    for (int i = 0; i < 8; i++) planes[i] = _mm256_set1_epi64x(0x123 + i);

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    for (int i = 0; i < iterations; i++) {{
        for (int j = 0; j < 8; j++) work[j] = planes[j];
        bs_sbox_bp_avx2(work);
        planes[0] = work[0];
    }}
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double elapsed = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    double evals = (double)iterations * 256;  // 256 parallel evals per call
    printf("BP AVX2 (115 gates):    %8.3f ns/eval, %10.1fM evals/sec\\n",
           (elapsed / evals) * 1e9, evals / elapsed / 1e6);
}}

// ============== Ternary AVX-512 ({circuit.gate_count} gates, {ternary_count} vpternlogd) ==============
{ternary_code}

void bench_ternary_avx512(int iterations) {{
    volatile __m512i planes[8];
    __m512i in[8], out[8];
    for (int i = 0; i < 8; i++) planes[i] = _mm512_set1_epi64(0x123 + i);

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    for (int i = 0; i < iterations; i++) {{
        for (int j = 0; j < 8; j++) in[j] = planes[j];
        sbox_ternary_avx512(in, out);
        planes[0] = out[0];
    }}
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double elapsed = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    double evals = (double)iterations * 512;  // 512 parallel evals per call (8x64-bit lanes)
    printf("Ternary AVX-512 ({circuit.gate_count}g): %8.3f ns/eval, %10.1fM evals/sec\\n",
           (elapsed / evals) * 1e9, evals / elapsed / 1e6);
}}

// ============== Verification ==============
// Verify the first gate computes correctly
void debug_first_gate() {{
    // Gate 0 is const(1,1), so let's check a ternary gate
    // Find first ternary gate and check it manually
    printf("DEBUG: Testing ternary gate 56: ternary(a=10, b=61, c=62, imm8=0x6c)\\n");
    printf("       imm8=0x6c = 0b01101100\\n");
    printf("       Truth table (imm8 bit i = F(a,b,c) where i=(a<<2)|(b<<1)|c):\\n");
    for (int i = 0; i < 8; i++) {{
        int a_val = (i >> 2) & 1;
        int b_val = (i >> 1) & 1;
        int c_val = i & 1;
        int result = (0x6c >> i) & 1;
        printf("         i=%d: a=%d b=%d c=%d -> %d\\n", i, a_val, b_val, c_val, result);
    }}
}}

// Simple scalar verification - evaluate one input at a time
int verify_ternary_scalar() {{
    int errors = 0;
    debug_first_gate();

    printf("\\nVerifying all 256 values...\\n");
    for (int input_val = 0; input_val < 256; input_val++) {{
        // Set up bit planes for single evaluation
        __m512i in[8], out[8];
        for (int bit = 0; bit < 8; bit++) {{
            uint64_t plane = (input_val >> bit) & 1 ? 0xFFFFFFFFFFFFFFFFULL : 0;
            in[bit] = _mm512_set1_epi64(plane);
        }}

        // Run S-box
        sbox_ternary_avx512(in, out);

        // Extract result (all bits should be the same)
        uint8_t result = 0;
        for (int bit = 0; bit < 8; bit++) {{
            uint64_t plane = _mm_cvtsi128_si64(_mm512_castsi512_si128(out[bit]));
            if (plane & 1) result |= (1 << bit);
        }}

        if (result != SBOX[input_val]) {{
            if (errors < 5) {{
                printf("ERROR: S-box[%d] (input 0x%02x) = 0x%02x, expected 0x%02x\\n",
                       input_val, input_val, result, SBOX[input_val]);
                // Print input bits
                printf("       input bits: ");
                for (int b = 7; b >= 0; b--) printf("%d", (input_val >> b) & 1);
                printf("\\n");
            }}
            errors++;
        }}
    }}
    return errors;
}}

int main(int argc, char** argv) {{
    int iterations = 10000000;
    if (argc > 1) iterations = atoi(argv[1]);

    printf("S-box Implementation Benchmark (Ternary Optimized)\\n");
    printf("==================================================\\n");
    printf("Iterations: %d\\n", iterations);
    printf("Circuit: {circuit.gate_count} gates ({binary_count} binary + {ternary_count} vpternlogd)\\n\\n");

    // Verify first
    printf("Verifying ternary circuit... ");
    int errors = verify_ternary_scalar();
    if (errors == 0) {{
        printf("PASS\\n\\n");
    }} else {{
        printf("FAIL (%d errors)\\n\\n", errors);
    }}

    printf("Benchmarks:\\n");
    bench_table(iterations);
    bench_bp_avx2(iterations);
    bench_ternary_avx512(iterations);

    printf("\\nComparison:\\n");
    printf("  BP optimal:  115 gates (hand-optimized, ~30 years of research)\\n");
    printf("  Ours:        {circuit.gate_count} gates ({ternary_count} ternary) - automated synthesis\\n");
    printf("  Gap:         %.1fx gate count\\n", {circuit.gate_count} / 115.0);

    return errors > 0 ? 1 : 0;
}}
"""
    return code


def main():
    parser = argparse.ArgumentParser(description="Benchmark ternary-optimized S-box")
    parser.add_argument("--iterations", type=int, default=10000000)
    args = parser.parse_args()

    print("Loading ternary-optimized circuit...")
    circuit = load_optimized_circuit()

    ternary_count = sum(1 for g in circuit.gates if len(g) == 5)
    print(f"Circuit: {circuit.gate_count} gates ({ternary_count} ternary)")

    # Check for AVX-512 support
    try:
        result = subprocess.run(
            ["grep", "-q", "avx512f", "/proc/cpuinfo"], capture_output=True
        )
        has_avx512 = result.returncode == 0
    except Exception:
        has_avx512 = False

    if not has_avx512:
        print("\nWarning: AVX-512 not available on this system")
        print("Generating code anyway for inspection...")

    print("\nGenerating benchmark code...")
    code = generate_benchmark_code(circuit)

    with tempfile.TemporaryDirectory() as tmpdir:
        c_file = os.path.join(tmpdir, "bench_ternary.c")
        exe_file = os.path.join(tmpdir, "bench_ternary")

        with open(c_file, "w") as f:
            f.write(code)

        # Also save for debugging
        with open("out/bench_ternary.c", "w") as f:
            f.write(code)
        print("Saved benchmark code to out/bench_ternary.c")

        cflags = ["-O3", "-march=native", "-mavx512f"]
        compile_cmd = ["gcc"] + cflags + ["-o", exe_file, c_file]
        print(f"Compiling: {' '.join(compile_cmd)}")

        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Compilation failed:\n{result.stderr}")
            return 1

        print("\nRunning benchmark...\n")
        result = subprocess.run(
            [exe_file, str(args.iterations)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
