#!/usr/bin/env python3
"""
Benchmark the 128-gate Boyar-Peralta tower field optimized S-box.

Compares:
1. Our ANF circuit (1145 gates)
2. ANF + Ternary optimized (994 gates)
3. BP tower field optimized (128 gates)
4. Reference BP from external-bitsliced (~115 gates, hand-tuned)
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stc.bitslice import AES_SBOX_TABLE
from stc.circuit_synth import IncrementalOptimizer, CircuitState
from stc.bitslice_codegen import generate_bitslice_c, AVX2_CONFIG
from scripts.bp_circuit_sbox import build_bp_sbox


def main():
    print("=" * 70)
    print("AES S-box Benchmark: ANF vs BP Tower Field")
    print("=" * 70)

    # Generate circuits
    print("\n1. Building circuits...")

    # ANF circuit
    opt = IncrementalOptimizer(AES_SBOX_TABLE, input_bits=8, output_bits=8)
    anf_state = opt.best_state
    print(
        f"   ANF circuit: {anf_state.gate_count} gates ({anf_state.and_count} AND, {anf_state.xor_count} XOR)"
    )

    # BP tower field circuit
    bp_state = build_bp_sbox()
    print(
        f"   BP tower:    {bp_state.gate_count} gates ({bp_state.and_count} AND, {bp_state.xor_count} XOR)"
    )

    # Verify BP is correct
    errors = 0
    for i in range(256):
        if bp_state.evaluate(i) != AES_SBOX_TABLE[i]:
            errors += 1
    print(f"   BP verified: {'✅ Correct' if errors == 0 else f'❌ {errors} errors'}")

    # Generate benchmark code
    print("\n2. Generating benchmark code...")

    # Generate code - need to avoid duplicate helper functions
    anf_code = generate_bitslice_c(anf_state, AVX2_CONFIG, "sbox_anf_avx2")
    bp_code_full = generate_bitslice_c(bp_state, AVX2_CONFIG, "sbox_bp_avx2")

    # Extract just the function from bp_code (skip helper functions)
    bp_lines = bp_code_full.split("\n")
    bp_func_start = None
    for i, line in enumerate(bp_lines):
        if "void sbox_bp_avx2" in line:
            bp_func_start = i
            break
    bp_code = "\n".join(bp_lines[bp_func_start:]) if bp_func_start else bp_code_full

    bench_code = f"""
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <immintrin.h>

// ANF circuit ({anf_state.gate_count} gates)
{anf_code}

// BP tower field circuit ({bp_state.gate_count} gates)
{bp_code}

int main(int argc, char** argv) {{
    int iterations = 1000000;
    if (argc > 1) iterations = atoi(argv[1]);

    printf("AES S-box Benchmark: ANF vs BP Tower Field\\n");
    printf("==========================================\\n");
    printf("Iterations: %d\\n\\n", iterations);

    volatile uint8_t input[32];
    volatile uint8_t output[32];
    for (int i = 0; i < 32; i++) input[i] = i;

    struct timespec ts_start, ts_end;

    // Benchmark ANF AVX2
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    for (int iter = 0; iter < iterations; iter++) {{
        sbox_anf_avx2((uint8_t*)input, (uint8_t*)output);
        input[0] = output[0];
    }}
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    double elapsed_anf = (ts_end.tv_sec - ts_start.tv_sec) +
                         (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    double evals_anf = (double)iterations * 32;
    double ns_anf = (elapsed_anf / evals_anf) * 1e9;
    printf("ANF AVX2 ({anf_state.gate_count} gates):\\n");
    printf("  %.3f ns/eval, %.0f evals/sec\\n\\n", ns_anf, evals_anf / elapsed_anf);

    // Reset
    for (int i = 0; i < 32; i++) input[i] = i;

    // Benchmark BP AVX2
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    for (int iter = 0; iter < iterations; iter++) {{
        sbox_bp_avx2((uint8_t*)input, (uint8_t*)output);
        input[0] = output[0];
    }}
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    double elapsed_bp = (ts_end.tv_sec - ts_start.tv_sec) +
                        (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    double evals_bp = (double)iterations * 32;
    double ns_bp = (elapsed_bp / evals_bp) * 1e9;
    printf("BP Tower ({bp_state.gate_count} gates):\\n");
    printf("  %.3f ns/eval, %.0f evals/sec\\n\\n", ns_bp, evals_bp / elapsed_bp);

    // Comparison
    printf("Comparison:\\n");
    printf("  Gate reduction: {anf_state.gate_count} -> {bp_state.gate_count} (%.1f%% fewer)\\n",
           (1.0 - (double){bp_state.gate_count} / {anf_state.gate_count}) * 100);
    printf("  Speedup: %.2fx\\n", ns_anf / ns_bp);

    return 0;
}}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        c_file = Path(tmpdir) / "bench.c"
        exe_file = Path(tmpdir) / "bench"

        c_file.write_text(bench_code)

        print("3. Compiling...")
        result = subprocess.run(
            ["gcc", "-O3", "-march=native", "-o", str(exe_file), str(c_file)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Compilation failed: {result.stderr}")
            Path("debug_bp_bench.c").write_text(bench_code)
            return

        print("4. Running benchmark...\n")
        result = subprocess.run(
            [str(exe_file), "1000000"], capture_output=True, text=True, timeout=60
        )
        print(result.stdout)

    print("=" * 70)
    print("Summary:")
    print(
        f"  ANF:      {anf_state.gate_count} gates ({anf_state.and_count} AND + {anf_state.xor_count} XOR)"
    )
    print(
        f"  BP Tower: {bp_state.gate_count} gates ({bp_state.and_count} AND + {bp_state.xor_count} XOR)"
    )
    print(
        f"  Reduction: {anf_state.gate_count - bp_state.gate_count} gates ({(anf_state.gate_count - bp_state.gate_count) / anf_state.gate_count * 100:.1f}%)"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
