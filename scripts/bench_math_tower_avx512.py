#!/usr/bin/env python3
import subprocess
import sys
import os

os.chdir("/home/user/repos/VeryLogo")

code = """
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "/home/user/repos/VeryLogo/out/test_fixed/circuit_avx512.c"

int main() {
    const int REPS = 10000000;
    __m512i in[8], out[8];

    for (int i = 0; i < 8; i++) {
        in[i] = _mm512_set1_epi64(0x0123456789abcdefULL);
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int rep = 0; rep < REPS; rep++) {
        circuit(in, out);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double evals_per_sec = (REPS * 64.0) / elapsed;

    printf("Mathematical tower field S-box (compiled via Yosys ABC)\\n");
    printf("Reps: %d x 64 lanes = %d total evaluations\\n", REPS, REPS * 64);
    printf("Time: %.3f seconds\\n", elapsed);
    printf("Throughput: %.2f billion S-box evals/sec\\n", evals_per_sec / 1e9);

    return 0;
}
"""

with open("/tmp/bench_math.cpp", "w") as f:
    f.write(code)

print("Compiling...")
subprocess.run(
    [
        "g++",
        "-O3",
        "-march=native",
        "-mavx512f",
        "-o",
        "/tmp/bench_math",
        "/tmp/bench_math.cpp",
    ],
    check=True,
)

print("Running benchmark...")
subprocess.run(["/tmp/bench_math"])
