#!/usr/bin/env python3
"""
Verify correctness and compute throughput of scheduled S-box.

1. Generate code from schedule
2. Compile and run
3. Verify against reference S-box table
4. Measure actual throughput
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bp_circuit_sbox import build_bp_sbox
from stc.bitslice import AES_SBOX_TABLE
from stc.sched import (
    AVX2,
    PTX,
    list_schedule,
    pipelined_schedule,
    compute_depth,
)


def generate_scheduled_avx2_code(gates, input_bits, outputs, schedule):
    """Generate AVX2 C code following the schedule order."""
    lines = []
    lines.append("#include <immintrin.h>")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("void sbox_scheduled_avx2(const uint8_t* input, uint8_t* output) {")
    lines.append("    __m256i planes[8];")
    lines.append("    __m256i ones = _mm256_set1_epi32(-1);")
    lines.append("")

    lines.append("    // Transpose input to bit planes")
    lines.append("    for (int bit = 0; bit < 8; bit++) {")
    lines.append("        uint32_t plane = 0;")
    lines.append("        for (int i = 0; i < 32; i++) {")
    lines.append("            if (input[i] & (1 << bit)) plane |= (1U << i);")
    lines.append("        }")
    lines.append("        planes[bit] = _mm256_set1_epi32(plane);")
    lines.append("    }")
    lines.append("")

    total_cycles = schedule.total_cycles

    lines.append(
        f"    // Evaluate circuit ({len(gates)} gates, {total_cycles} scheduled cycles)"
    )

    def get_reg(idx):
        if idx < input_bits:
            return f"planes[{idx}]"
        return f"t{idx - input_bits}"

    gates_by_cycle = {}
    for g_idx, cycle in schedule.gate_cycle.items():
        if cycle not in gates_by_cycle:
            gates_by_cycle[cycle] = []
        gates_by_cycle[cycle].append(g_idx)

    for cycle in range(total_cycles):
        cycle_gates = gates_by_cycle.get(cycle, [])
        if cycle_gates:
            lines.append(f"    // Cycle {cycle}: {len(cycle_gates)} gates")
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
    lines.append("    // Store outputs")
    lines.append("    __m256i out_planes[8];")
    for i, (out_idx, inv) in enumerate(outputs):
        if inv:
            lines.append(
                f"    out_planes[{i}] = _mm256_xor_si256({get_reg(out_idx)}, ones);"
            )
        else:
            lines.append(f"    out_planes[{i}] = {get_reg(out_idx)};")

    lines.append("")
    lines.append("    // Transpose output from bit planes")
    lines.append("    for (int i = 0; i < 32; i++) {")
    lines.append("        uint8_t byte = 0;")
    lines.append("        for (int bit = 0; bit < 8; bit++) {")
    lines.append(
        "            uint32_t plane = _mm256_extract_epi32(out_planes[bit], 0);"
    )
    lines.append("            if (plane & (1U << i)) byte |= (1 << bit);")
    lines.append("        }")
    lines.append("        output[i] = byte;")
    lines.append("    }")
    lines.append("}")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("Scheduled S-box Verification")
    print("=" * 70)

    print("\n1. Building BP tower field S-box circuit...")
    circuit = build_bp_sbox()
    gates = list(circuit.gates)
    input_bits = circuit.input_bits
    outputs = list(circuit.outputs)

    print(f"   Gates: {len(gates)} ({circuit.and_count} AND, {circuit.xor_count} XOR)")
    print(f"   Depth: {compute_depth(gates, input_bits)}")

    print("\n2. Verifying circuit correctness (Python)...")
    errors = 0
    for i in range(256):
        result = circuit.evaluate(i)
        expected = AES_SBOX_TABLE[i]
        if result != expected:
            errors += 1
            if errors <= 5:
                print(f"   ERROR: S-box[{i}] = {result}, expected {expected}")

    if errors == 0:
        print("   ✅ Circuit correct for all 256 inputs")
    else:
        print(f"   ❌ {errors} errors")
        return

    print("\n3. Scheduling for AVX2...")
    schedule = list_schedule(gates, input_bits, outputs, AVX2, "slack")
    stats = schedule.compute_stats(gates, input_bits)

    print(f"   Cycles: {stats.total_cycles}")
    print(f"   Max live: {stats.max_live}")
    print(f"   Avg parallelism: {stats.avg_parallelism:.1f} gates/cycle")

    validation_errors = schedule.validate(gates, input_bits, AVX2.latencies)
    if validation_errors:
        print(f"   ❌ Schedule invalid: {validation_errors[:3]}")
        return
    print("   ✅ Schedule valid (dependencies respected)")

    print("\n4. Generating scheduled AVX2 code...")
    scheduled_code = generate_scheduled_avx2_code(gates, input_bits, outputs, schedule)

    print("\n5. Compiling and verifying...")

    bench_code = f"""
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

{scheduled_code}

static const uint8_t EXPECTED_SBOX[256] = {{
    {", ".join(f"0x{v:02x}" for v in AES_SBOX_TABLE)}
}};

int verify_correctness() {{
    uint8_t input[32];
    uint8_t output[32];
    int errors = 0;

    // Test all 256 values (8 batches of 32)
    for (int batch = 0; batch < 8; batch++) {{
        for (int i = 0; i < 32; i++) {{
            input[i] = batch * 32 + i;
        }}
        sbox_scheduled_avx2(input, output);
        for (int i = 0; i < 32; i++) {{
            int idx = batch * 32 + i;
            if (output[i] != EXPECTED_SBOX[idx]) {{
                if (errors < 10) {{
                    printf("ERROR: S-box[%d] = 0x%02x, expected 0x%02x\\n",
                           idx, output[i], EXPECTED_SBOX[idx]);
                }}
                errors++;
            }}
        }}
    }}
    return errors;
}}

double benchmark(int iterations) {{
    volatile uint8_t input[32];
    volatile uint8_t output[32];
    for (int i = 0; i < 32; i++) input[i] = i;

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    for (int iter = 0; iter < iterations; iter++) {{
        sbox_scheduled_avx2((uint8_t*)input, (uint8_t*)output);
        input[0] = output[0];  // Prevent optimization
    }}

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double elapsed = (ts_end.tv_sec - ts_start.tv_sec) +
                     (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    return elapsed;
}}

int main() {{
    printf("Verifying correctness...\\n");
    int errors = verify_correctness();
    if (errors == 0) {{
        printf("✅ All 256 S-box values correct\\n\\n");
    }} else {{
        printf("❌ %d errors\\n", errors);
        return 1;
    }}

    printf("Benchmarking throughput...\\n");
    int iterations = 1000000;
    double elapsed = benchmark(iterations);

    double evals = (double)iterations * 32;  // 32 parallel per call
    double ns_per_eval = (elapsed / evals) * 1e9;
    double evals_per_sec = evals / elapsed;

    printf("  Iterations: %d\\n", iterations);
    printf("  Total evals: %.0f\\n", evals);
    printf("  Time: %.3f sec\\n", elapsed);
    printf("  Throughput: %.3f ns/eval\\n", ns_per_eval);
    printf("  Throughput: %.2e evals/sec\\n", evals_per_sec);

    // Theoretical analysis
    printf("\\nTheoretical analysis:\\n");
    printf("  Schedule cycles: {stats.total_cycles}\\n");
    printf("  Circuit depth: {compute_depth(gates, input_bits)}\\n");
    printf("  Gates: {len(gates)}\\n");
    printf("  Parallel width: 32 (AVX2 256-bit / 8 bits)\\n");

    return 0;
}}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        c_file = Path(tmpdir) / "verify.c"
        exe_file = Path(tmpdir) / "verify"

        c_file.write_text(bench_code)

        result = subprocess.run(
            ["gcc", "-O3", "-march=native", "-o", str(exe_file), str(c_file)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"   ❌ Compilation failed: {result.stderr}")
            Path("debug_scheduled.c").write_text(bench_code)
            print("   Saved to debug_scheduled.c")
            return

        print("   ✅ Compiled successfully")

        print("\n6. Running verification and benchmark...\n")
        result = subprocess.run(
            [str(exe_file)], capture_output=True, text=True, timeout=60
        )
        print(result.stdout)

        if result.returncode != 0:
            print(f"   ❌ Verification failed")
            return

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Circuit: {len(gates)} gates, depth {compute_depth(gates, input_bits)}")
    print(f"Schedule: {stats.total_cycles} cycles, {stats.max_live} max live")
    print(f"Target: AVX2 (16 registers, 256-bit SIMD)")
    print(f"Register pressure: {stats.max_live} live > 16 registers = SPILLS NEEDED")


if __name__ == "__main__":
    main()
