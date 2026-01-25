#!/usr/bin/env python3
"""
Benchmark scheduled circuit WITHOUT transpose overhead.

Measures pure gate evaluation throughput to compare with hand-optimized baselines.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bp_circuit_sbox import build_bp_sbox
from stc.sched import AVX2, list_schedule, compute_depth


def generate_circuit_only_code(gates, input_bits, outputs, schedule):
    """Generate AVX2 code that operates on pre-transposed data."""
    lines = []
    lines.append("#include <immintrin.h>")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("// Circuit-only: assumes input already in bit-plane form")
    lines.append("static inline void sbox_circuit_only(__m256i planes[8]) {")
    lines.append("    __m256i ones = _mm256_set1_epi32(-1);")

    def get_reg(idx):
        if idx < input_bits:
            return f"planes[{idx}]"
        return f"t{idx - input_bits}"

    total_cycles = schedule.total_cycles
    gates_by_cycle = {}
    for g_idx, cycle in schedule.gate_cycle.items():
        if cycle not in gates_by_cycle:
            gates_by_cycle[cycle] = []
        gates_by_cycle[cycle].append(g_idx)

    for cycle in range(total_cycles):
        cycle_gates = gates_by_cycle.get(cycle, [])
        for g_idx in sorted(cycle_gates):
            gate = gates[g_idx]
            op = gate[0]
            left = gate[1]
            right = gate[2] if len(gate) > 2 else 0
            reg = f"t{g_idx}"

            if op == "xor":
                lines.append(
                    f"    __m256i {reg} = _mm256_xor_si256({get_reg(left)}, {get_reg(right)});"
                )
            elif op == "and":
                lines.append(
                    f"    __m256i {reg} = _mm256_and_si256({get_reg(left)}, {get_reg(right)});"
                )
            elif op == "or":
                lines.append(
                    f"    __m256i {reg} = _mm256_or_si256({get_reg(left)}, {get_reg(right)});"
                )
            elif op == "not":
                lines.append(
                    f"    __m256i {reg} = _mm256_xor_si256({get_reg(left)}, ones);"
                )
            elif op == "andn":
                lines.append(
                    f"    __m256i {reg} = _mm256_andnot_si256({get_reg(left)}, {get_reg(right)});"
                )
            elif op == "const":
                if left == 0:
                    lines.append(f"    __m256i {reg} = _mm256_setzero_si256();")
                else:
                    lines.append(f"    __m256i {reg} = ones;")

    lines.append("")
    lines.append("    // Write outputs back to planes")
    for i, (out_idx, inv) in enumerate(outputs):
        if inv:
            lines.append(
                f"    planes[{i}] = _mm256_xor_si256({get_reg(out_idx)}, ones);"
            )
        else:
            lines.append(f"    planes[{i}] = {get_reg(out_idx)};")

    lines.append("}")
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("Scheduled Circuit-Only Benchmark (No Transpose)")
    print("=" * 70)

    circuit = build_bp_sbox()
    gates = list(circuit.gates)
    input_bits = circuit.input_bits
    outputs = list(circuit.outputs)

    print(f"\nCircuit: {len(gates)} gates, depth {compute_depth(gates, input_bits)}")

    schedule = list_schedule(gates, input_bits, outputs, AVX2, "slack")
    stats = schedule.compute_stats(gates, input_bits)
    print(f"Schedule: {stats.total_cycles} cycles")

    circuit_code = generate_circuit_only_code(gates, input_bits, outputs, schedule)

    bench_code = f"""
#include <stdio.h>
#include <stdint.h>
#include <time.h>

{circuit_code}

int main() {{
    volatile __m256i planes[8];
    __m256i work[8];

    // Initialize with some data
    for (int i = 0; i < 8; i++) {{
        planes[i] = _mm256_set1_epi64x(0x123456789ABCDEF0ULL + i);
    }}

    int iterations = 10000000;

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    for (int iter = 0; iter < iterations; iter++) {{
        for (int j = 0; j < 8; j++) work[j] = planes[j];
        sbox_circuit_only(work);
        planes[0] = work[0];  // Prevent optimization
    }}

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double elapsed = (ts_end.tv_sec - ts_start.tv_sec) +
                     (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    // Each call processes 256 S-boxes (256-bit / 1-bit per S-box input)
    double evals = (double)iterations * 256;
    double ns_per_eval = (elapsed / evals) * 1e9;
    double evals_per_sec = evals / elapsed;

    printf("Circuit-only benchmark (no transpose):\\n");
    printf("  Iterations: %d\\n", iterations);
    printf("  Evals: %.0f\\n", evals);
    printf("  Time: %.3f sec\\n", elapsed);
    printf("  Throughput: %.3f ns/eval\\n", ns_per_eval);
    printf("  Throughput: %.2e evals/sec\\n", evals_per_sec);

    return 0;
}}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        c_file = Path(tmpdir) / "bench.c"
        exe_file = Path(tmpdir) / "bench"

        c_file.write_text(bench_code)

        result = subprocess.run(
            ["gcc", "-O3", "-march=native", "-o", str(exe_file), str(c_file)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Compilation failed: {result.stderr}")
            return

        result = subprocess.run(
            [str(exe_file)], capture_output=True, text=True, timeout=60
        )
        print(result.stdout)

    print("Comparison:")
    print("  Hand-optimized BP AVX2: 0.028 ns/eval (from benchmark.md)")
    print("  Our scheduled circuit:  see above")


if __name__ == "__main__":
    main()
