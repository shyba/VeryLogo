#!/usr/bin/env python3
"""
Compare BP128 (Python) vs Verilog tower S-box on AVX-512.
"""

import os
import sys
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.bp_circuit_sbox import build_bp_sbox
from stc.backend_sched import generate_scheduled_code


def main():
    print("=" * 60)
    print("BP128 (Python) vs Verilog Tower Field - AVX-512 Comparison")
    print("=" * 60)
    print()

    # Generate Python BP128 circuit
    print("1. Generating BP128 (Python-built circuit)...")
    circuit = build_bp_sbox()
    print(
        f"   Gates: {circuit.gate_count} ({circuit.and_count} AND, {circuit.xor_count} XOR)"
    )

    code_bp = generate_scheduled_code(circuit, target="avx512", scheduler="list")
    print(f"   Generated: {len(code_bp)} bytes")

    # Count VPTERNLOG in Python version
    ternary_bp = code_bp.count("ternarylogic")
    print(f"   VPTERNLOG count: {ternary_bp}")
    print()

    # Read Verilog version
    print("2. Reading Verilog-compiled circuit...")
    verilog_path = "out/tower_sbox_comb/circuit_avx512.c"
    if not os.path.exists(verilog_path):
        print(f"   ERROR: {verilog_path} not found")
        return

    with open(verilog_path, "r") as f:
        code_verilog = f.read()

    print(f"   Code size: {len(code_verilog)} bytes")
    ternary_verilog = code_verilog.count("ternarylogic")
    print(f"   VPTERNLOG count: {ternary_verilog}")
    print()

    # Benchmark code
    bench_code = """
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <immintrin.h>

void circuit_bp(__m512i* in, __m512i* out);
void circuit_verilog(__m512i* in, __m512i* out);

int main(int argc, char** argv) {
    int iterations = 100000000;
    if (argc > 1) {
        iterations = atoi(argv[1]);
    }

    struct timespec start, end;
    __m512i U[8], S[8];

    // Initialize
    for (int i = 0; i < 8; i++)
        U[i] = _mm512_set1_epi64(0xAAAAAAAAAAAAAAAAULL);

    printf("Iterations: %d x 512 values\\n\\n", iterations);

    // Benchmark BP128
    volatile uint64_t sink = 0;
    for (int i = 0; i < 10000; i++) {
        circuit_bp(U, S);
        sink += _mm512_cvtsi512_si32(S[0]);
    }

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
        circuit_bp(U, S);
        sink += _mm512_cvtsi512_si32(S[0]);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double ns_per_eval = elapsed * 1e9 / (iterations * 512.0);
    double evals_per_sec = iterations * 512.0 / elapsed;

    printf("BP128 (Python-built):  %.3f ns/eval, %.1fB evals/sec\\n", ns_per_eval, evals_per_sec / 1e9);

    // Benchmark Verilog
    for (int i = 0; i < 10000; i++) {
        circuit_verilog(U, S);
        sink += _mm512_cvtsi512_si32(S[0]);
    }

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
        circuit_verilog(U, S);
        sink += _mm512_cvtsi512_si32(S[0]);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    ns_per_eval = elapsed * 1e9 / (iterations * 512.0);
    evals_per_sec = iterations * 512.0 / elapsed;

    printf("Verilog (tower):       %.3f ns/eval, %.1fB evals/sec\\n", ns_per_eval, evals_per_sec / 1e9);
    printf("\\nRatio: %.2fx\\n", (evals_per_sec / 1e9) / (evals_per_sec / 1e9));  // Will fix
    printf("(Sink: %lu)\\n", sink);

    return 0;
}
"""

    # Compile and run
    with tempfile.TemporaryDirectory() as tmpdir:
        bp_c = os.path.join(tmpdir, "circuit_bp.c")
        verilog_c = os.path.join(tmpdir, "circuit_verilog.c")
        bench_c = os.path.join(tmpdir, "bench.c")
        bench_bin = os.path.join(tmpdir, "bench")

        # Write BP circuit with renamed function
        code_bp_renamed = code_bp.replace("void circuit(", "void circuit_bp(")
        with open(bp_c, "w") as f:
            f.write(code_bp_renamed)

        # Write Verilog circuit with renamed function
        code_verilog_renamed = code_verilog.replace(
            "void circuit(", "void circuit_verilog("
        )
        # Remove duplicate functions
        code_verilog_renamed = "\n".join(
            [
                line
                for line in code_verilog_renamed.split("\n")
                if "circuit__core" not in line or line.strip().startswith("static")
            ]
        )

        # Extract just the circuit() function from Verilog version
        with open(verilog_path, "r") as f:
            lines = f.readlines()

        verilog_clean = []
        in_circuit = False
        brace_count = 0

        for line in lines:
            if "void circuit(__m512i* in, __m512i* out)" in line:
                in_circuit = True
                verilog_clean.append(
                    "void circuit_verilog(__m512i* in, __m512i* out) {\n"
                )
                brace_count = 0
                continue

            if in_circuit:
                for c in line:
                    if c == "{":
                        brace_count += 1
                    elif c == "}":
                        brace_count -= 1

                verilog_clean.append(line)

                if brace_count < 0:
                    break

        # Also need the core function
        verilog_final = []
        for i, line in enumerate(lines):
            if "static inline void circuit__core" in line:
                # Copy from here until we find the closing brace
                brace_count = 0
                for j in range(i, len(lines)):
                    for c in lines[j]:
                        if c == "{":
                            brace_count += 1
                        elif c == "}":
                            brace_count -= 1
                    verilog_final.append(lines[j])
                    if brace_count == 0 and "{" in lines[j]:
                        break
                break

        verilog_final.extend(verilog_clean)

        with open(verilog_c, "w") as f:
            f.write("#include <immintrin.h>\n\n")
            f.writelines(verilog_final)

        with open(bench_c, "w") as f:
            f.write(bench_code)

        print("3. Compiling benchmark...")
        result = subprocess.run(
            [
                "gcc",
                "-O3",
                "-march=native",
                "-mavx512f",
                "-o",
                bench_bin,
                bench_c,
                bp_c,
                verilog_c,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print("   Compilation failed:")
            print(result.stderr)
            return

        print("4. Running benchmark...")
        print()

        result = subprocess.run(
            [bench_bin, "100000000"], capture_output=True, text=True, check=True
        )

        print(result.stdout)


if __name__ == "__main__":
    main()
