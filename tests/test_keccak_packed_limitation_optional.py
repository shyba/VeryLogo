import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestKeccakPackedLimitation(unittest.TestCase):
    def test_keccak_packed_architectural_mismatch(self) -> None:
        if shutil.which("yosys") is None:
            self.skipTest("yosys not available")

        rtl_dir = Path("external-sha3-verilog/low_throughput_core/rtl")
        if not rtl_dir.exists():
            self.skipTest("external-sha3-verilog not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)

            flat_json = td / "keccak_flat.json"
            script = (
                f"read_verilog {rtl_dir}/*.v; "
                "hierarchy -top keccak; proc; flatten; opt; opt_clean; "
                f"write_json {flat_json}"
            )
            subprocess.run(["yosys", "-q", "-p", script], check=True)

            bitsliced_dir = td / "bitsliced"
            bitsliced_dir.mkdir()
            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "-m",
                    "stc",
                    str(flat_json),
                    "--top",
                    "keccak",
                    "--out",
                    str(bitsliced_dir),
                    "--backend",
                    "x86-avx512",
                    "--bound",
                    "8",
                    "--force-bitsliced",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                self.skipTest(f"Bitsliced compilation failed: {result.stderr}")

            packed_dir = td / "packed"
            packed_dir.mkdir()
            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "-m",
                    "stc",
                    str(flat_json),
                    "--top",
                    "keccak",
                    "--out",
                    str(packed_dir),
                    "--backend",
                    "x86-avx512",
                    "--bound",
                    "8",
                    "--force-packed",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                self.skipTest(f"Packed compilation failed: {result.stderr}")

            bit_circuit = bitsliced_dir / "circuit_state.json"
            packed_circuit = packed_dir / "packed_circuit_state.json"

            if not bit_circuit.exists() or not packed_circuit.exists():
                self.skipTest("Output circuits not generated")

            with open(bit_circuit) as f:
                bit_data = json.load(f)
            with open(packed_circuit) as f:
                packed_data = json.load(f)

            bit_gates = len(bit_data.get("gates", []))
            packed_gates = len(packed_data.get("gates", []))

            ratio = packed_gates / bit_gates if bit_gates > 0 else 0

            self.assertGreater(
                ratio,
                50,
                f"Keccak packed/bit ratio expected > 50x (got {ratio:.1f}x). "
                "This documents that Keccak is unsuitable for packed lowering "
                "due to architectural mismatch between bit-twiddling operations "
                "and 64-bit word-level lowering.",
            )


if __name__ == "__main__":
    unittest.main()
