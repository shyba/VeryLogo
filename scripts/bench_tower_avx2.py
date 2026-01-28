#!/usr/bin/env python3
"""
Benchmark the AVX2-compiled Verilog tower field S-box (circuit-only).
"""

import os
import sys
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def benchmark_avx2(iterations: int = 100_000_000):
    """Benchmark AVX2 version."""

    # C benchmark code
    c_code = """
#include <immintrin.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

void circuit(__m256i* in, __m256i* out);

int main(int argc, char** argv) {
    int iterations = 100000000;
    if (argc > 1) {
        iterations = atoi(argv[1]);
    }

    struct timespec start, end;
    __m256i U[8], S[8];

    // Initialize
    for (int i = 0; i < 8; i++)
        U[i] = _mm256_set1_epi32(0xAAAAAAAA);

    // Warmup
    volatile uint64_t sink = 0;
    for (int i = 0; i < 10000; i++) {
        circuit(U, S);
        sink += _mm256_extract_epi32(S[0], 0);
    }

    // Benchmark
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
        circuit(U, S);
        sink += _mm256_extract_epi32(S[0], 0);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double ns_per_eval = elapsed * 1e9 / (iterations * 256.0);
    double evals_per_sec = iterations * 256.0 / elapsed;

    printf("Verilog Tower Field S-box (AVX2):\\n");
    printf("  Gates: 132 -> 98 (ternary synthesis)\\n");
    printf("  Iterations: %d x 256 values\\n", iterations);
    printf("  Time per eval: %.3f ns\\n", ns_per_eval);
    printf("  Throughput: %.1fB evals/sec\\n", evals_per_sec / 1e9);
    printf("  (Sink: %lu)\\n", sink);

    return 0;
}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        bench_c = os.path.join(tmpdir, "bench.c")
        bench_bin = os.path.join(tmpdir, "bench")

        with open(bench_c, "w") as f:
            f.write(c_code)

        # Copy circuit
        circuit_path = "out/tower_sbox_avx2/circuit_avx2.c"
        if not os.path.exists(circuit_path):
            print(f"ERROR: {circuit_path} not found")
            return

        subprocess.run(
            ["cp", circuit_path, os.path.join(tmpdir, "circuit_avx2.c")], check=True
        )

        # Compile
        print("Compiling AVX2 version...")
        result = subprocess.run(
            [
                "gcc",
                "-O3",
                "-march=native",
                "-mavx2",
                "-o",
                bench_bin,
                bench_c,
                os.path.join(tmpdir, "circuit_avx2.c"),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print("Compilation failed:")
            print(result.stderr)
            return

        print("Running benchmark...")
        print()

        # Run
        result = subprocess.run(
            [bench_bin, str(iterations)], capture_output=True, text=True, check=True
        )

        print(result.stdout)


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000_000
    benchmark_avx2(iters)
