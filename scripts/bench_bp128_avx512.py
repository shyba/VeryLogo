#!/usr/bin/env python3
"""
Benchmark the BP128 circuit compiled to AVX-512.
"""

import os
import sys
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.bp_circuit_sbox import build_bp_sbox
from stc.backend_sched import generate_scheduled_code


def benchmark_bp128_avx512(iterations: int = 100_000_000):
    """Benchmark BP128 with AVX-512 backend."""

    print("Generating BP128 circuit...")
    circuit = build_bp_sbox()
    print(
        f"  {circuit.gate_count} gates ({circuit.and_count} AND, {circuit.xor_count} XOR)"
    )
    print()

    print("Compiling to AVX-512...")
    code = generate_scheduled_code(circuit, target="avx512", scheduler="list")
    print(f"  Generated {len(code)} bytes of C code")
    print()

    # C benchmark wrapper
    bench_code = """
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <immintrin.h>

void circuit(__m512i* in, __m512i* out);

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

    // Warmup
    volatile uint64_t sink = 0;
    for (int i = 0; i < 10000; i++) {
        circuit(U, S);
        sink += _mm512_cvtsi512_si32(S[0]);
    }

    // Benchmark
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
        circuit(U, S);
        sink += _mm512_cvtsi512_si32(S[0]);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double ns_per_eval = elapsed * 1e9 / (iterations * 512.0);
    double evals_per_sec = iterations * 512.0 / elapsed;

    printf("BP128 (AVX-512):\\n");
    printf("  Iterations: %d x 512 values\\n", iterations);
    printf("  Time per eval: %.3f ns\\n", ns_per_eval);
    printf("  Throughput: %.1fB evals/sec\\n", evals_per_sec / 1e9);
    printf("  (Sink: %lu)\\n", sink);

    return 0;
}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        circuit_c = os.path.join(tmpdir, "circuit.c")
        bench_c = os.path.join(tmpdir, "bench.c")
        bench_bin = os.path.join(tmpdir, "bench")

        with open(circuit_c, "w") as f:
            f.write(code)

        with open(bench_c, "w") as f:
            f.write(bench_code)

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
                circuit_c,
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
    benchmark_bp128_avx512(iters)
