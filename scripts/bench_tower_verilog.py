#!/usr/bin/env python3
"""
Benchmark the Verilog-compiled tower field S-box.
"""

import os
import sys
import subprocess
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stc.bitslice import AES_SBOX_TABLE


def benchmark_verilog_sbox(iterations: int = 10_000_000):
    """Benchmark the AVX-512 compiled Verilog tower field S-box."""

    # C benchmark code
    c_code = """
#include <immintrin.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

#include "circuit_avx512.c"

static inline uint64_t get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(int argc, char** argv) {
    int iterations = 10000000;
    if (argc > 1) {
        iterations = atoi(argv[1]);
    }

    // Allocate aligned buffers
    __m512i* in_io = aligned_alloc(64, 9 * sizeof(__m512i));
    __m512i* out_io = aligned_alloc(64, 8 * sizeof(__m512i));
    __m512i* state_in = aligned_alloc(64, 8 * sizeof(__m512i));
    __m512i* state_out = aligned_alloc(64, 8 * sizeof(__m512i));

    // Initialize with test pattern
    for (int i = 0; i < 9; i++) in_io[i] = _mm512_set1_epi8(0);
    for (int i = 0; i < 8; i++) state_in[i] = _mm512_set1_epi8(0);

    // Warmup
    for (int i = 0; i < 1000; i++) {
        circuit_steps_shared(in_io, out_io, state_in, state_out, 1);
    }

    // Benchmark
    uint64_t start = get_time_ns();
    for (int i = 0; i < iterations; i++) {
        circuit_steps_shared(in_io, out_io, state_in, state_out, 1);
    }
    uint64_t end = get_time_ns();

    uint64_t total_ns = end - start;
    double ns_per_eval = (double)total_ns / iterations;
    double evals_per_sec = 1e9 / ns_per_eval;

    printf("Verilog Tower Field S-box (AVX-512):\\n");
    printf("  Iterations: %d\\n", iterations);
    printf("  Total time: %.3f ms\\n", total_ns / 1e6);
    printf("  Time per eval: %.3f ns\\n", ns_per_eval);
    printf("  Throughput: %.1f M evals/sec\\n", evals_per_sec / 1e6);

    free(in_io);
    free(out_io);
    free(state_in);
    free(state_out);
    return 0;
}
"""

    # Write benchmark to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        bench_c = os.path.join(tmpdir, "bench.c")
        bench_bin = os.path.join(tmpdir, "bench")

        with open(bench_c, "w") as f:
            f.write(c_code)

        # Copy circuit to temp dir
        circuit_path = "out/tower_sbox_bitsliced/circuit_avx512.c"
        if not os.path.exists(circuit_path):
            print(f"ERROR: {circuit_path} not found. Run compilation first.")
            return

        subprocess.run(
            ["cp", circuit_path, os.path.join(tmpdir, "circuit_avx512.c")], check=True
        )

        # Compile
        print("Compiling benchmark...")
        result = subprocess.run(
            [
                "gcc",
                "-O3",
                "-march=native",
                "-mavx512f",
                "-o",
                bench_bin,
                bench_c,
                "-I",
                tmpdir,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print("Compilation failed:")
            print(result.stderr)
            return

        print()

        # Run benchmark
        result = subprocess.run(
            [bench_bin, str(iterations)], capture_output=True, text=True, check=True
        )

        print(result.stdout)


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
    print(f"Benchmarking Verilog-compiled tower field S-box ({iters:,} iterations)...")
    print()
    benchmark_verilog_sbox(iters)
