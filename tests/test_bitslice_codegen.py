import os
import subprocess
import sys
import tempfile
import unittest

sys.setrecursionlimit(10000)

from stc.bitslice import AES_SBOX_TABLE
from stc.bitslice_codegen import (
    UINT64_CONFIG,
    generate_bitslice_c,
    generate_test_harness,
)
from stc.circuit_synth import IncrementalOptimizer


class TestBitsliceCodegen(unittest.TestCase):
    def test_generate_bitslice_c(self) -> None:
        """Test that bitslice C code can be generated."""
        opt = IncrementalOptimizer(AES_SBOX_TABLE, input_bits=8, output_bits=8)
        code = generate_bitslice_c(opt.best_state, UINT64_CONFIG, "test_sbox")

        self.assertIn("void test_sbox", code)
        self.assertIn("uint64_t", code)
        self.assertIn("// Execute circuit", code)
        self.assertIn(f"// Bitsliced circuit: {opt.best_state.gate_count} gates", code)

    def test_generate_test_harness(self) -> None:
        """Test that test harness can be generated."""
        opt = IncrementalOptimizer(AES_SBOX_TABLE, input_bits=8, output_bits=8)
        code = generate_test_harness(
            opt.best_state, AES_SBOX_TABLE, UINT64_CONFIG, "test_sbox"
        )

        self.assertIn("int main(void)", code)
        self.assertIn("SBOX[256]", code)
        self.assertIn("PASS:", code)
        self.assertIn("FAIL:", code)

    @unittest.skipUnless(
        subprocess.run(["which", "gcc"], capture_output=True).returncode == 0,
        "gcc not available",
    )
    def test_compile_and_run_uint64(self) -> None:
        """Test that generated code compiles and runs correctly."""
        opt = IncrementalOptimizer(AES_SBOX_TABLE, input_bits=8, output_bits=8)
        code = generate_test_harness(
            opt.best_state, AES_SBOX_TABLE, UINT64_CONFIG, "aes_sbox_uint64"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            c_file = os.path.join(tmpdir, "test.c")
            exe_file = os.path.join(tmpdir, "test")

            with open(c_file, "w") as f:
                f.write(code)

            compile_result = subprocess.run(
                ["gcc", "-O2", "-o", exe_file, c_file],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                compile_result.returncode,
                0,
                f"Compilation failed: {compile_result.stderr}",
            )

            run_result = subprocess.run(
                [exe_file], capture_output=True, text=True, timeout=30
            )
            self.assertEqual(run_result.returncode, 0, f"Test failed: {run_result.stdout}")
            self.assertIn("PASS", run_result.stdout)


class TestBitsliceCorrectness(unittest.TestCase):
    def test_circuit_matches_table(self) -> None:
        """Verify circuit produces correct S-box values in Python."""
        opt = IncrementalOptimizer(AES_SBOX_TABLE, input_bits=8, output_bits=8)

        for i in range(256):
            got = opt.best_state.evaluate(i)
            self.assertEqual(
                got, AES_SBOX_TABLE[i], f"Mismatch at input {i}: got {got}, expected {AES_SBOX_TABLE[i]}"
            )
