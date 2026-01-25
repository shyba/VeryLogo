#!/usr/bin/env python3
"""
End-to-end test: emit AVX2 code for BP circuit, compile, verify correctness.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bp_circuit_sbox import build_bp_sbox
from stc.bitslice import AES_SBOX_TABLE
from stc.sched import list_schedule, AVX2, AVX512
from stc.sched.liveness import compute_live_ranges
from stc.sched.regalloc import allocate_registers
from stc.sched.emit import AVX2Emitter


def main():
    print("=" * 70)
    print("End-to-End Emitter Test: BP S-box → AVX2 Code → Verify")
    print("=" * 70)

    print("\n1. Building BP circuit...")
    circuit = build_bp_sbox()
    gates = list(circuit.gates)
    input_bits = circuit.input_bits
    outputs = list(circuit.outputs)
    print(f"   {len(gates)} gates ({circuit.and_count} AND, {circuit.xor_count} XOR)")

    print("\n2. Scheduling for AVX-512 (no spills)...")
    schedule = list_schedule(gates, input_bits, outputs, AVX512, "slack")
    print(f"   {schedule.total_cycles} cycles")

    print("\n3. Allocating registers...")
    live_ranges = compute_live_ranges(schedule, gates, input_bits, outputs)
    allocation = allocate_registers(live_ranges, schedule, AVX512.registers)
    print(f"   {AVX512.registers} registers, {allocation.num_spills} spills")

    print("\n4. Emitting AVX2 code...")
    emitter = AVX2Emitter()
    code = emitter.emit(schedule, allocation, gates, input_bits, outputs, "sbox_bp")
    print(f"   Generated {len(code)} bytes of C code")

    test_harness = f"""
#include <stdio.h>
#include <stdint.h>
#include <string.h>

{code}

static const uint8_t EXPECTED[256] = {{
    {", ".join(f"0x{v:02x}" for v in AES_SBOX_TABLE)}
}};

int main() {{
    __m256i in[8], out[8];
    uint8_t input[32], output[32];
    int errors = 0;

    for (int batch = 0; batch < 8; batch++) {{
        // Set up input
        for (int i = 0; i < 32; i++) {{
            input[i] = batch * 32 + i;
        }}

        // Transpose to bit planes
        for (int bit = 0; bit < 8; bit++) {{
            uint32_t plane = 0;
            for (int i = 0; i < 32; i++) {{
                if (input[i] & (1 << bit)) plane |= (1U << i);
            }}
            in[bit] = _mm256_set1_epi32(plane);
        }}

        // Run S-box
        sbox_bp(in, out);

        // Transpose from bit planes
        for (int i = 0; i < 32; i++) {{
            uint8_t byte = 0;
            for (int bit = 0; bit < 8; bit++) {{
                uint32_t plane = _mm256_extract_epi32(out[bit], 0);
                if (plane & (1U << i)) byte |= (1 << bit);
            }}
            output[i] = byte;
        }}

        // Verify
        for (int i = 0; i < 32; i++) {{
            int idx = batch * 32 + i;
            if (output[i] != EXPECTED[idx]) {{
                if (errors < 10) {{
                    printf("ERROR: S-box[%d] = 0x%02x, expected 0x%02x\\n",
                           idx, output[i], EXPECTED[idx]);
                }}
                errors++;
            }}
        }}
    }}

    if (errors == 0) {{
        printf("✅ All 256 S-box values correct!\\n");
        return 0;
    }} else {{
        printf("❌ %d errors\\n", errors);
        return 1;
    }}
}}
"""

    print("\n5. Compiling...")
    with tempfile.TemporaryDirectory() as tmpdir:
        c_file = Path(tmpdir) / "test.c"
        exe_file = Path(tmpdir) / "test"

        c_file.write_text(test_harness)

        result = subprocess.run(
            ["gcc", "-O3", "-march=native", "-o", str(exe_file), str(c_file)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"   ❌ Compilation failed:")
            print(result.stderr[:500])
            Path("debug_emitted.c").write_text(test_harness)
            print("   Saved to debug_emitted.c")
            return 1

        print("   ✅ Compiled successfully")

        print("\n6. Running verification...")
        result = subprocess.run([str(exe_file)], capture_output=True, text=True)
        print(f"   {result.stdout.strip()}")

        if result.returncode != 0:
            return 1

    print("\n" + "=" * 70)
    print("SUCCESS: End-to-end pipeline working!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
