#!/usr/bin/env python3
"""
Generate bitsliced SIMD code for AES S-box circuit.

Usage:
    # Generate AVX2 code with test harness
    python scripts/generate_bitslice.py --avx2 --test --output sbox_avx2.c

    # Generate portable uint64 code
    python scripts/generate_bitslice.py --uint64 --output sbox_u64.c

    # Generate benchmark
    python scripts/generate_bitslice.py --avx2 --bench --output bench_avx2.c

    # Compile and run test (requires gcc with AVX2 support)
    python scripts/generate_bitslice.py --avx2 --test --compile --run
"""
import argparse
import os
import subprocess
import sys
import tempfile

sys.setrecursionlimit(10000)

from stc.bitslice import AES_SBOX_TABLE
from stc.bitslice_codegen import (
    AVX2_CONFIG,
    AVX512_CONFIG,
    SSE2_CONFIG,
    UINT64_CONFIG,
    generate_benchmark,
    generate_bitslice_c,
    generate_test_harness,
)
from stc.circuit_synth import IncrementalOptimizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate bitsliced S-box code")
    parser.add_argument("--avx2", action="store_true", help="Generate AVX2 code")
    parser.add_argument("--avx512", action="store_true", help="Generate AVX-512 code")
    parser.add_argument("--sse2", action="store_true", help="Generate SSE2 code")
    parser.add_argument(
        "--uint64", action="store_true", help="Generate portable uint64 code"
    )
    parser.add_argument("--test", action="store_true", help="Generate test harness")
    parser.add_argument("--bench", action="store_true", help="Generate benchmark")
    parser.add_argument("--output", "-o", type=str, help="Output file")
    parser.add_argument("--load", type=str, help="Load circuit from JSON")
    parser.add_argument(
        "--compile", action="store_true", help="Compile the generated code"
    )
    parser.add_argument("--run", action="store_true", help="Run compiled code")
    args = parser.parse_args()

    configs = []
    if args.avx2:
        configs.append(("avx2", AVX2_CONFIG, "-mavx2"))
    if args.avx512:
        configs.append(("avx512", AVX512_CONFIG, "-mavx512f"))
    if args.sse2:
        configs.append(("sse2", SSE2_CONFIG, "-msse2"))
    if args.uint64:
        configs.append(("uint64", UINT64_CONFIG, ""))

    if not configs:
        configs.append(("uint64", UINT64_CONFIG, ""))

    if args.load:
        print(f"Loading circuit from {args.load}...")
        opt = IncrementalOptimizer.load(args.load)
    else:
        print("Generating ANF circuit...")
        opt = IncrementalOptimizer(AES_SBOX_TABLE, input_bits=8, output_bits=8)

    state = opt.best_state
    print(f"Circuit: {state.gate_count} gates")

    for name, config, cflags in configs:
        func_name = f"aes_sbox_{name}"

        if args.test:
            code = generate_test_harness(state, AES_SBOX_TABLE, config, func_name)
        elif args.bench:
            code = generate_benchmark(state, config, func_name)
        else:
            code = generate_bitslice_c(state, config, func_name)

        if args.output:
            output_file = args.output
            if len(configs) > 1:
                base, ext = os.path.splitext(args.output)
                output_file = f"{base}_{name}{ext}"
        else:
            if args.test:
                output_file = f"sbox_{name}_test.c"
            elif args.bench:
                output_file = f"sbox_{name}_bench.c"
            else:
                output_file = f"sbox_{name}.c"

        with open(output_file, "w") as f:
            f.write(code)
        print(f"Generated {output_file} ({len(code)} bytes)")

        if args.compile or args.run:
            exe_file = output_file.replace(".c", "")
            compile_cmd = ["gcc", "-O3", "-o", exe_file, output_file]
            if cflags:
                compile_cmd.insert(2, cflags)

            print(f"Compiling: {' '.join(compile_cmd)}")
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Compilation failed:\n{result.stderr}")
                continue
            print(f"Compiled to {exe_file}")

            if args.run:
                print(f"Running {exe_file}...")
                result = subprocess.run(
                    [f"./{exe_file}"], capture_output=True, text=True
                )
                print(result.stdout)
                if result.stderr:
                    print(result.stderr)


if __name__ == "__main__":
    main()
