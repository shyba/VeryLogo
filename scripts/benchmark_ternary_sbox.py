#!/usr/bin/env python3
"""
Benchmark AES S-box with and without ternary optimization.

Compares:
1. Original ANF circuit (1145 gates) - AVX2 baseline
2. Ternary-optimized circuit (993 gates) - AVX-512 vpternlogd
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stc.bitslice import AES_SBOX_TABLE
from stc.circuit_synth import IncrementalOptimizer, CircuitState
from stc.bitslice_codegen import generate_bitslice_c, AVX2_CONFIG


def find_3input_cones(state: CircuitState):
    """Find 3-input cones that can be replaced with ternary ops."""
    gates = list(state.gates)
    input_bits = state.input_bits

    use_count = {}
    for idx, (op, left, right) in enumerate(gates):
        use_count[left] = use_count.get(left, 0) + 1
        if op not in ("const", "not"):
            use_count[right] = use_count.get(right, 0) + 1
    for out_idx, _ in state.outputs:
        use_count[out_idx] = use_count.get(out_idx, 0) + 1

    cones = []
    visited_gates = set()

    for root_idx, (root_op, root_left, root_right) in enumerate(gates):
        if root_op not in ("xor", "and", "or"):
            continue

        full_root = input_bits + root_idx

        def collect_cone(idx, depth=0):
            if idx < input_bits:
                return ({idx}, set())
            gate_idx = idx - input_bits
            if gate_idx >= len(gates):
                return ({idx}, set())
            op, left, right = gates[gate_idx]
            if op == "const":
                return ({idx}, set())
            if use_count.get(idx, 0) > 1 and idx != full_root:
                return ({idx}, set())
            if depth > 1:
                return ({idx}, set())
            left_leaves, left_internal = collect_cone(left, depth + 1)
            right_leaves, right_internal = set(), set()
            if op not in ("not",):
                right_leaves, right_internal = collect_cone(right, depth + 1)
            return (
                left_leaves | right_leaves,
                left_internal | right_internal | {gate_idx},
            )

        leaves, internal_gates = collect_cone(full_root)

        if len(leaves) == 3 and len(internal_gates) >= 2:
            if not any(g in visited_gates for g in internal_gates):
                cones.append(
                    {
                        "root": root_idx,
                        "leaves": sorted(leaves),
                        "internal_gates": internal_gates,
                    }
                )
                visited_gates.update(internal_gates)

    return cones


def compute_truth_table(state: CircuitState, root_idx: int, leaves: list) -> int:
    """Compute the 8-bit truth table for a 3-input cone."""
    gates = state.gates
    input_bits = state.input_bits

    def eval_at(assignments: dict) -> int:
        node_vals = {}
        for i in range(input_bits):
            node_vals[i] = assignments.get(i, 0)

        for idx, (op, left, right) in enumerate(gates):
            full_idx = input_bits + idx
            l_val = node_vals.get(left, assignments.get(left, 0))
            r_val = node_vals.get(right, assignments.get(right, 0)) if right >= 0 else 0

            if op == "xor":
                node_vals[full_idx] = l_val ^ r_val
            elif op == "and":
                node_vals[full_idx] = l_val & r_val
            elif op == "or":
                node_vals[full_idx] = l_val | r_val
            elif op == "not":
                node_vals[full_idx] = 1 - l_val
            elif op == "const":
                node_vals[full_idx] = left
            else:
                node_vals[full_idx] = 0

            if idx == root_idx:
                return node_vals[full_idx]
        return 0

    imm8 = 0
    for i in range(8):
        assignments = {
            leaves[0]: (i >> 0) & 1,
            leaves[1]: (i >> 1) & 1,
            leaves[2]: (i >> 2) & 1,
        }
        if eval_at(assignments):
            imm8 |= 1 << i
    return imm8


def generate_avx512_ternary_code(state: CircuitState, cones: list) -> str:
    """Generate AVX-512 code with vpternlogd for ternary ops."""
    lines = []
    lines.append("#include <immintrin.h>")
    lines.append("#include <stdint.h>")
    lines.append("")

    # Build mapping of which gates are ternary roots
    ternary_roots = {}
    for cone in cones:
        imm8 = compute_truth_table(state, cone["root"], cone["leaves"])
        ternary_roots[cone["root"]] = {
            "leaves": cone["leaves"],
            "imm8": imm8,
        }

    lines.append("void sbox_avx512_ternary(const uint8_t* input, uint8_t* output) {")
    lines.append("    __m512i planes[8];")
    lines.append("    __m512i ones = _mm512_set1_epi32(-1);")
    lines.append("")

    # Transpose
    lines.append("    // Transpose to bit planes")
    lines.append("    for (int bit = 0; bit < 8; bit++) {")
    lines.append("        uint64_t plane = 0;")
    lines.append("        for (int i = 0; i < 64; i++) {")
    lines.append("            if (input[i] & (1 << bit)) plane |= (1ULL << i);")
    lines.append("        }")
    lines.append("        planes[bit] = _mm512_set1_epi64(plane);")
    lines.append("    }")
    lines.append("")

    # Evaluate - generate all gates, replace ternary roots
    gates = state.gates
    input_bits = state.input_bits

    def get_reg(i):
        if i < input_bits:
            return f"planes[{i}]"
        return f"t{i - input_bits}"

    lines.append("    // Evaluate circuit")
    ternary_used = 0
    for idx, gate in enumerate(gates):
        op = gate[0]
        left = gate[1] if len(gate) > 1 else 0
        right = gate[2] if len(gate) > 2 else 0
        reg = f"t{idx}"

        if idx in ternary_roots:
            # Replace with ternary instruction
            t = ternary_roots[idx]
            a, b, c = t["leaves"]
            imm8 = t["imm8"]
            lines.append(
                f"    __m512i {reg} = _mm512_ternarylogic_epi64({get_reg(a)}, {get_reg(b)}, {get_reg(c)}, 0x{imm8:02x});"
            )
            ternary_used += 1
        elif op == "const":
            if left == 0:
                lines.append(f"    __m512i {reg} = _mm512_setzero_si512();")
            else:
                lines.append(f"    __m512i {reg} = ones;")
        elif op == "xor":
            lines.append(
                f"    __m512i {reg} = _mm512_xor_si512({get_reg(left)}, {get_reg(right)});"
            )
        elif op == "and":
            lines.append(
                f"    __m512i {reg} = _mm512_and_si512({get_reg(left)}, {get_reg(right)});"
            )
        elif op == "or":
            lines.append(
                f"    __m512i {reg} = _mm512_or_si512({get_reg(left)}, {get_reg(right)});"
            )
        elif op == "not":
            lines.append(
                f"    __m512i {reg} = _mm512_xor_si512({get_reg(left)}, ones);"
            )

    lines.append("")

    # Store outputs
    lines.append("    // Store outputs")
    lines.append("    __m512i out_planes[8];")
    for i, (out_idx, inv) in enumerate(state.outputs):
        if inv:
            lines.append(
                f"    out_planes[{i}] = _mm512_xor_si512({get_reg(out_idx)}, ones);"
            )
        else:
            lines.append(f"    out_planes[{i}] = {get_reg(out_idx)};")

    lines.append("")

    # Transpose output
    lines.append("    // Transpose output")
    lines.append("    uint64_t pdata[8];")
    lines.append("    for (int bit = 0; bit < 8; bit++) {")
    lines.append("        _mm512_storeu_si512((__m512i*)&pdata[bit], out_planes[bit]);")
    lines.append("    }")
    lines.append("    for (int i = 0; i < 64; i++) {")
    lines.append("        uint8_t byte = 0;")
    lines.append("        for (int bit = 0; bit < 8; bit++) {")
    lines.append("            if (pdata[bit] & (1ULL << i)) byte |= (1 << bit);")
    lines.append("        }")
    lines.append("        output[i] = byte;")
    lines.append("    }")
    lines.append("}")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("AES S-box Benchmark: Original vs Ternary Optimized")
    print("=" * 70)

    # Generate circuit
    print("\n1. Generating circuits...")
    opt = IncrementalOptimizer(AES_SBOX_TABLE, input_bits=8, output_bits=8)
    state = opt.best_state
    print(f"   Original: {state.gate_count} gates")

    # Find ternary cones
    cones = find_3input_cones(state)
    active_gates = state.gate_count - len(cones)
    print(f"   Ternary-optimized: {active_gates} gates ({len(cones)} ternary ops)")

    # Check for AVX-512 support
    try:
        result = subprocess.run(
            ["grep", "-q", "avx512", "/proc/cpuinfo"], capture_output=True
        )
        has_avx512 = result.returncode == 0
    except Exception:
        has_avx512 = False

    if not has_avx512:
        print("\n⚠️  AVX-512 not available on this system")
        print("   Running gate count comparison only")

        print(f"\nResults:")
        print(f"   Original circuit:    {state.gate_count} gates")
        print(f"   Ternary optimized:   {active_gates} gates")
        print(
            f"   Reduction:           {state.gate_count - active_gates} gates ({(state.gate_count - active_gates) / state.gate_count * 100:.1f}%)"
        )
        print(
            f"   Target (1000 gates): {'✅ ACHIEVED' if active_gates < 1000 else '❌ NOT MET'}"
        )
        return

    print("\n2. Generating benchmark code...")

    # Generate AVX2 baseline
    avx2_code = generate_bitslice_c(state, AVX2_CONFIG, "sbox_avx2")

    # Generate AVX-512 ternary
    avx512_code = generate_avx512_ternary_code(state, cones)

    # Full benchmark code
    bench_code = f"""
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

// AVX2 baseline ({state.gate_count} gates)
{avx2_code}

// AVX-512 with vpternlogd ({active_gates} ops, {len(cones)} ternary)
{avx512_code}

int main(int argc, char** argv) {{
    int iterations = 1000000;
    if (argc > 1) iterations = atoi(argv[1]);

    printf("AES S-box Benchmark\\n");
    printf("==================\\n");
    printf("Iterations: %d\\n\\n", iterations);

    volatile uint8_t input[64];
    volatile uint8_t output[64];
    for (int i = 0; i < 64; i++) input[i] = i;

    // Benchmark AVX2 baseline
    struct timespec ts_start, ts_end;

    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    for (int iter = 0; iter < iterations; iter++) {{
        sbox_avx2((uint8_t*)input, (uint8_t*)output);
        input[0] = output[0];
    }}
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    double elapsed_avx2 = (ts_end.tv_sec - ts_start.tv_sec) +
                          (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    double evals_avx2 = (double)iterations * 32;  // 32 parallel evals
    printf("AVX2 baseline ({state.gate_count} gates):\\n");
    printf("  %.3f ns/eval, %.0f evals/sec\\n\\n",
           (elapsed_avx2 / evals_avx2) * 1e9, evals_avx2 / elapsed_avx2);

    // Reset input
    for (int i = 0; i < 64; i++) input[i] = i;

    // Benchmark AVX-512 ternary
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    for (int iter = 0; iter < iterations; iter++) {{
        sbox_avx512_ternary((uint8_t*)input, (uint8_t*)output);
        input[0] = output[0];
    }}
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    double elapsed_avx512 = (ts_end.tv_sec - ts_start.tv_sec) +
                            (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    double evals_avx512 = (double)iterations * 64;  // 64 parallel evals
    printf("AVX-512 ternary ({active_gates} ops, {len(cones)} vpternlogd):\\n");
    printf("  %.3f ns/eval, %.0f evals/sec\\n\\n",
           (elapsed_avx512 / evals_avx512) * 1e9, evals_avx512 / elapsed_avx512);

    // Speedup
    double ns_avx2 = (elapsed_avx2 / evals_avx2) * 1e9;
    double ns_avx512 = (elapsed_avx512 / evals_avx512) * 1e9;
    printf("Speedup: %.2fx\\n", ns_avx2 / ns_avx512);

    return 0;
}}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        c_file = Path(tmpdir) / "bench.c"
        exe_file = Path(tmpdir) / "bench"

        c_file.write_text(bench_code)

        print("3. Compiling benchmark...")
        compile_cmd = [
            "gcc",
            "-O3",
            "-march=native",
            "-mavx512f",
            "-o",
            str(exe_file),
            str(c_file),
        ]
        result = subprocess.run(compile_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Compilation failed: {result.stderr}")
            # Save for debugging
            debug_file = Path("debug_ternary_bench.c")
            debug_file.write_text(bench_code)
            print(f"Saved to {debug_file}")
            return

        print("4. Running benchmark...\n")
        result = subprocess.run(
            [str(exe_file), "1000000"], capture_output=True, text=True, timeout=60
        )
        print(result.stdout)

        if result.stderr:
            print(result.stderr)

    print("=" * 70)
    print("Gate count comparison:")
    print(f"  Original:  {state.gate_count} gates")
    print(f"  Optimized: {active_gates} gates")
    print(
        f"  Reduction: {state.gate_count - active_gates} gates ({(state.gate_count - active_gates) / state.gate_count * 100:.1f}%)"
    )
    print(
        f"  Target:    {'✅ < 1000 achieved' if active_gates < 1000 else '❌ Still above 1000'}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
