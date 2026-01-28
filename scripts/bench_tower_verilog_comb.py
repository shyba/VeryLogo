#!/usr/bin/env python3
"""
Benchmark the combinational Verilog tower field S-box (circuit-only, no state overhead).
"""

import os
import sys
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def benchmark_verilog_sbox_combinational(iterations: int = 100_000_000):
    """Benchmark the AVX-512 compiled combinational Verilog tower field S-box."""

    # C benchmark code - circuit only
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
    int iterations = 100000000;
    if (argc > 1) {
        iterations = atoi(argv[1]);
    }

    // Allocate aligned buffers
    __m512i* in_buf = aligned_alloc(64, 8 * sizeof(__m512i));
    __m512i* out_buf = aligned_alloc(64, 8 * sizeof(__m512i));

    // Initialize with test pattern
    for (int i = 0; i < 8; i++) {
        in_buf[i] = _mm512_set1_epi64(0xAAAAAAAAAAAAAAAAULL);
    }

    // Warmup
    volatile uint64_t sink = 0;
    for (int i = 0; i < 10000; i++) {
        circuit(in_buf, out_buf);
        sink += _mm512_cvtsi512_si32(out_buf[0]);
    }

    // Benchmark - call circuit() directly (no state management)
    uint64_t start = get_time_ns();
    for (int i = 0; i < iterations; i++) {
        circuit(in_buf, out_buf);
        sink += _mm512_cvtsi512_si32(out_buf[0]);
    }
    uint64_t end = get_time_ns();

    uint64_t total_ns = end - start;
    double ns_per_eval = (double)total_ns / iterations;
    double evals_per_sec = 1e9 / ns_per_eval;

    printf("Verilog Tower Field S-box (AVX-512, combinational):\\n");
    printf("  Gates: 132 -> 98 (25.8%% reduction via ternary synthesis)\\n");
    printf("  VPTERNLOG instructions: 34\\n");
    printf("  State bits: 0 (purely combinational)\\n");
    printf("  Iterations: %d\\n", iterations);
    printf("  Total time: %.3f ms\\n", total_ns / 1e6);
    printf("  Time per eval: %.3f ns\\n", ns_per_eval);
    printf("  Throughput: %.1f B evals/sec\\n", evals_per_sec / 1e9);
    printf("  (Sink: %lu)\\n", sink);

    free(in_buf);
    free(out_buf);
    return 0;
}
"""

    # Write and compile benchmark
    with tempfile.TemporaryDirectory() as tmpdir:
        bench_c = os.path.join(tmpdir, "bench.c")
        bench_bin = os.path.join(tmpdir, "bench")

        with open(bench_c, "w") as f:
            f.write(c_code)

        # Copy circuit
        circuit_path = "out/tower_sbox_comb/circuit_avx512.c"
        if not os.path.exists(circuit_path):
            print(f"ERROR: {circuit_path} not found")
            return

        subprocess.run(
            ["cp", circuit_path, os.path.join(tmpdir, "circuit_avx512.c")], check=True
        )

        # Compile with aggressive optimization
        print("Compiling...")
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

        print("Running circuit-only benchmark...")
        print()

        # Run
        result = subprocess.run(
            [bench_bin, str(iterations)], capture_output=True, text=True, check=True
        )

        print(result.stdout)


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000_000
    benchmark_verilog_sbox_combinational(iters)
