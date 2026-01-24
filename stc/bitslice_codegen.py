"""
Generate bitsliced SIMD code from CircuitState.

Converts a boolean circuit (XOR/AND gates) into SIMD code that processes
multiple evaluations in parallel using bitslicing.

For a circuit with N input bits and M output bits:
- Input: K parallel values (K = SIMD width in bits)
- Transpose: K values -> N SIMD registers (bit planes)
- Evaluate: Execute gates on SIMD registers
- Transpose: M SIMD registers -> K output values
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from stc.circuit_synth import CircuitState


@dataclass
class BitsliceConfig:
    simd_width: int
    register_type: str
    xor_op: str
    and_op: str
    or_op: str
    not_op: str
    zero_const: str
    ones_const: str
    include: str
    name_suffix: str = ""


AVX2_CONFIG = BitsliceConfig(
    simd_width=256,
    register_type="__m256i",
    xor_op="_mm256_xor_si256",
    and_op="_mm256_and_si256",
    or_op="_mm256_or_si256",
    not_op="_mm256_xor_si256({}, _mm256_set1_epi32(-1))",
    zero_const="_mm256_setzero_si256()",
    ones_const="_mm256_set1_epi32(-1)",
    include="#include <immintrin.h>",
    name_suffix="_avx2",
)

SSE2_CONFIG = BitsliceConfig(
    simd_width=128,
    register_type="__m128i",
    xor_op="_mm_xor_si128",
    and_op="_mm_and_si128",
    or_op="_mm_or_si128",
    not_op="_mm_xor_si128({}, _mm_set1_epi32(-1))",
    zero_const="_mm_setzero_si128()",
    ones_const="_mm_set1_epi32(-1)",
    include="#include <emmintrin.h>",
    name_suffix="_sse2",
)

AVX512_CONFIG = BitsliceConfig(
    simd_width=512,
    register_type="__m512i",
    xor_op="_mm512_xor_si512",
    and_op="_mm512_and_si512",
    or_op="_mm512_or_si512",
    not_op="_mm512_xor_si512({}, _mm512_set1_epi32(-1))",
    zero_const="_mm512_setzero_si512()",
    ones_const="_mm512_set1_epi32(-1)",
    include="#include <immintrin.h>",
    name_suffix="_avx512",
)

UINT64_CONFIG = BitsliceConfig(
    simd_width=64,
    register_type="uint64_t",
    xor_op="({} ^ {})",
    and_op="({} & {})",
    or_op="({} | {})",
    not_op="(~{})",
    zero_const="0ULL",
    ones_const="0xFFFFFFFFFFFFFFFFULL",
    include="#include <stdint.h>",
    name_suffix="_u64",
)


def generate_bitslice_c(
    state: CircuitState,
    config: BitsliceConfig,
    function_name: str = "sbox_bitslice",
) -> str:
    """Generate C code for bitsliced circuit evaluation."""
    lines = []
    lines.append(config.include)
    lines.append("#include <stdint.h>")
    lines.append("")

    parallel = config.simd_width // 8
    reg = config.register_type

    lines.append(f"// Bitsliced circuit: {state.gate_count} gates")
    lines.append(f"// Processes {parallel} evaluations in parallel")
    lines.append(f"// Input: {state.input_bits} bits, Output: {state.output_bits} bits")
    lines.append("")

    _emit_transpose_functions(lines, config)

    lines.append(f"void {function_name}(const uint8_t* input, uint8_t* output) {{")

    for i in range(state.input_bits):
        lines.append(f"  {reg} b{i};")

    lines.append("")
    lines.append("  // Transpose input bytes to bit planes")
    lines.append(
        f"  bit_transpose_to_planes{config.name_suffix}(input, {', '.join(f'&b{i}' for i in range(state.input_bits))});"
    )
    lines.append("")

    lines.append("  // Execute circuit")
    node_names = [f"b{i}" for i in range(state.input_bits)]

    for idx, (op, left, right) in enumerate(state.gates):
        gate_name = f"g{idx}"
        left_name = node_names[left]

        if op == "xor":
            right_name = node_names[right]
            if config.simd_width == 64:
                expr = config.xor_op.format(left_name, right_name)
            else:
                expr = f"{config.xor_op}({left_name}, {right_name})"
        elif op == "and":
            right_name = node_names[right]
            if config.simd_width == 64:
                expr = config.and_op.format(left_name, right_name)
            else:
                expr = f"{config.and_op}({left_name}, {right_name})"
        elif op == "or":
            right_name = node_names[right]
            if config.simd_width == 64:
                expr = config.or_op.format(left_name, right_name)
            else:
                expr = f"{config.or_op}({left_name}, {right_name})"
        elif op == "not":
            if config.simd_width == 64:
                expr = config.not_op.format(left_name)
            else:
                expr = config.not_op.format(left_name)
        elif op == "const":
            val = left & 1
            expr = config.ones_const if val else config.zero_const
        else:
            expr = config.zero_const

        lines.append(f"  {reg} {gate_name} = {expr};")
        node_names.append(gate_name)

    lines.append("")
    lines.append("  // Collect output bit planes")

    output_names = []
    for i, (out_idx, inv) in enumerate(state.outputs):
        out_name = node_names[out_idx]
        if inv:
            if config.simd_width == 64:
                lines.append(f"  {reg} out{i} = {config.not_op.format(out_name)};")
            else:
                lines.append(f"  {reg} out{i} = {config.not_op.format(out_name)};")
            output_names.append(f"out{i}")
        else:
            output_names.append(out_name)

    lines.append("")
    lines.append("  // Transpose bit planes back to output bytes")
    lines.append(
        f"  bit_transpose_from_planes{config.name_suffix}({', '.join(output_names)}, output);"
    )

    lines.append("}")

    return "\n".join(lines)


def _emit_transpose_functions(lines: list[str], config: BitsliceConfig) -> None:
    """Emit bit transpose helper functions."""
    parallel = config.simd_width // 8
    reg = config.register_type
    suffix = config.name_suffix

    lines.append(f"// Transpose {parallel} input bytes to 8 bit planes")
    if config.simd_width == 64:
        lines.append(
            f"static inline void bit_transpose_to_planes{suffix}(const uint8_t* input, "
            + ", ".join(f"{reg}* b{i}" for i in range(8))
            + ") {"
        )
        lines.append("  uint64_t planes[8] = {0};")
        lines.append(f"  for (int i = 0; i < {parallel}; i++) {{")
        lines.append("    uint8_t byte = input[i];")
        lines.append("    for (int bit = 0; bit < 8; bit++) {")
        lines.append("      if (byte & (1 << bit))")
        lines.append("        planes[bit] |= (1ULL << i);")
        lines.append("    }")
        lines.append("  }")
        for i in range(8):
            lines.append(f"  *b{i} = planes[{i}];")
        lines.append("}")
    else:
        lines.append(
            f"static inline void bit_transpose_to_planes{suffix}(const uint8_t* input, "
            + ", ".join(f"{reg}* b{i}" for i in range(8))
            + ") {"
        )
        lines.append(f"  uint64_t planes[8][{config.simd_width // 64}];")
        lines.append(f"  for (int w = 0; w < {config.simd_width // 64}; w++) {{")
        lines.append("    for (int bit = 0; bit < 8; bit++) planes[bit][w] = 0;")
        lines.append("  }")
        lines.append(f"  for (int i = 0; i < {parallel}; i++) {{")
        lines.append("    uint8_t byte = input[i];")
        lines.append("    int word = i / 64;")
        lines.append("    int shift = i % 64;")
        lines.append("    for (int bit = 0; bit < 8; bit++) {")
        lines.append("      if (byte & (1 << bit))")
        lines.append("        planes[bit][word] |= (1ULL << shift);")
        lines.append("    }")
        lines.append("  }")
        for i in range(8):
            if config.simd_width == 128:
                lines.append(
                    f"  *b{i} = _mm_set_epi64x(planes[{i}][1], planes[{i}][0]);"
                )
            elif config.simd_width == 256:
                lines.append(
                    f"  *b{i} = _mm256_set_epi64x(planes[{i}][3], planes[{i}][2], planes[{i}][1], planes[{i}][0]);"
                )
            elif config.simd_width == 512:
                lines.append(
                    f"  *b{i} = _mm512_set_epi64(planes[{i}][7], planes[{i}][6], planes[{i}][5], planes[{i}][4], planes[{i}][3], planes[{i}][2], planes[{i}][1], planes[{i}][0]);"
                )
        lines.append("}")

    lines.append("")
    lines.append(f"// Transpose 8 bit planes back to {parallel} output bytes")

    if config.simd_width == 64:
        lines.append(
            f"static inline void bit_transpose_from_planes{suffix}("
            + ", ".join(f"{reg} b{i}" for i in range(8))
            + ", uint8_t* output) {"
        )
        lines.append(f"  for (int i = 0; i < {parallel}; i++) {{")
        lines.append("    uint8_t byte = 0;")
        lines.append("    for (int bit = 0; bit < 8; bit++) {")
        bits = ", ".join(f"b{i}" for i in range(8))
        lines.append(f"      uint64_t planes[8] = {{{bits}}};")
        lines.append("      if (planes[bit] & (1ULL << i))")
        lines.append("        byte |= (1 << bit);")
        lines.append("    }")
        lines.append("    output[i] = byte;")
        lines.append("  }")
        lines.append("}")
    else:
        lines.append(
            f"static inline void bit_transpose_from_planes{suffix}("
            + ", ".join(f"{reg} b{i}" for i in range(8))
            + ", uint8_t* output) {"
        )
        lines.append(
            f"  union {{ {reg} v; uint64_t u64[{config.simd_width // 64}]; }} planes[8];"
        )
        for i in range(8):
            lines.append(f"  planes[{i}].v = b{i};")
        lines.append(f"  for (int i = 0; i < {parallel}; i++) {{")
        lines.append("    int word = i / 64;")
        lines.append("    int shift = i % 64;")
        lines.append("    uint8_t byte = 0;")
        lines.append("    for (int bit = 0; bit < 8; bit++) {")
        lines.append("      if (planes[bit].u64[word] & (1ULL << shift))")
        lines.append("        byte |= (1 << bit);")
        lines.append("    }")
        lines.append("    output[i] = byte;")
        lines.append("  }")
        lines.append("}")

    lines.append("")


def generate_test_harness(
    state: CircuitState,
    table: Sequence[int],
    config: BitsliceConfig,
    function_name: str = "sbox_bitslice",
) -> str:
    """Generate a test harness that verifies the bitsliced implementation."""
    parallel = config.simd_width // 8
    lines = []

    lines.append("#include <stdio.h>")
    lines.append("#include <string.h>")
    lines.append("")

    circuit_code = generate_bitslice_c(state, config, function_name)
    lines.append(circuit_code)
    lines.append("")

    lines.append("// Reference S-box table")
    lines.append(f"static const uint8_t SBOX[256] = {{")
    for i in range(0, 256, 16):
        row = ", ".join(f"0x{table[j]:02x}" for j in range(i, min(i + 16, 256)))
        lines.append(f"  {row},")
    lines.append("};")
    lines.append("")

    lines.append("int main(void) {")
    lines.append(f"  uint8_t input[{parallel}];")
    lines.append(f"  uint8_t output[{parallel}];")
    lines.append("  int errors = 0;")
    lines.append("")
    lines.append("  // Test all 256 S-box entries")
    lines.append(f"  for (int base = 0; base < 256; base += {parallel}) {{")
    lines.append(f"    for (int i = 0; i < {parallel}; i++) {{")
    lines.append(f"      input[i] = (base + i) < 256 ? (base + i) : 0;")
    lines.append("    }")
    lines.append("")
    lines.append(f"    {function_name}(input, output);")
    lines.append("")
    lines.append(f"    for (int i = 0; i < {parallel} && (base + i) < 256; i++) {{")
    lines.append("      uint8_t expected = SBOX[input[i]];")
    lines.append("      if (output[i] != expected) {")
    lines.append('        printf("FAIL: input=%d, expected=0x%02x, got=0x%02x\\n",')
    lines.append("               input[i], expected, output[i]);")
    lines.append("        errors++;")
    lines.append("      }")
    lines.append("    }")
    lines.append("  }")
    lines.append("")
    lines.append("  if (errors == 0) {")
    lines.append('    printf("PASS: All 256 S-box entries correct\\n");')
    lines.append(
        f'    printf("Circuit: {state.gate_count} gates, {parallel} parallel evaluations\\n");'
    )
    lines.append("    return 0;")
    lines.append("  } else {")
    lines.append('    printf("FAIL: %d errors\\n", errors);')
    lines.append("    return 1;")
    lines.append("  }")
    lines.append("}")

    return "\n".join(lines)


def generate_benchmark(
    state: CircuitState,
    config: BitsliceConfig,
    function_name: str = "sbox_bitslice",
) -> str:
    """Generate a benchmark that measures throughput."""
    parallel = config.simd_width // 8
    lines = []

    lines.append("#include <stdio.h>")
    lines.append("#include <string.h>")
    lines.append("#include <time.h>")
    lines.append("")

    circuit_code = generate_bitslice_c(state, config, function_name)
    lines.append(circuit_code)
    lines.append("")

    lines.append("#define ITERATIONS 1000000")
    lines.append("")
    lines.append("int main(void) {")
    lines.append(f"  uint8_t input[{parallel}];")
    lines.append(f"  uint8_t output[{parallel}];")
    lines.append("")
    lines.append(f"  for (int i = 0; i < {parallel}; i++) input[i] = i;")
    lines.append("")
    lines.append("  clock_t start = clock();")
    lines.append("  for (int iter = 0; iter < ITERATIONS; iter++) {")
    lines.append(f"    {function_name}(input, output);")
    lines.append("    input[0] = output[0];")
    lines.append("  }")
    lines.append("  clock_t end = clock();")
    lines.append("")
    lines.append("  double elapsed = (double)(end - start) / CLOCKS_PER_SEC;")
    lines.append(f"  double evals = (double)ITERATIONS * {parallel};")
    lines.append("  double evals_per_sec = evals / elapsed;")
    lines.append("  double ns_per_eval = (elapsed / evals) * 1e9;")
    lines.append("")
    lines.append('  printf("Bitsliced S-box benchmark:\\n");')
    lines.append(f'  printf("  Circuit: {state.gate_count} gates\\n");')
    lines.append(f'  printf("  Parallel: {parallel} evaluations\\n");')
    lines.append('  printf("  Iterations: %d\\n", ITERATIONS);')
    lines.append('  printf("  Total evaluations: %.0f\\n", evals);')
    lines.append('  printf("  Elapsed: %.3f s\\n", elapsed);')
    lines.append('  printf("  Throughput: %.0f evals/sec\\n", evals_per_sec);')
    lines.append('  printf("  Latency: %.1f ns/eval\\n", ns_per_eval);')
    lines.append("")
    lines.append("  return 0;")
    lines.append("}")

    return "\n".join(lines)
