import os
import shutil
import unittest
from pathlib import Path


class TestSha3VerilogAvx512Optional(unittest.TestCase):
    def test_sha3_512_matches_hashlib(self):
        if os.environ.get("STC_RUN_SHA3_VERILOG", "") not in {"1", "true", "TRUE"}:
            self.skipTest("set STC_RUN_SHA3_VERILOG=1 to enable")
        if not Path("/proc/cpuinfo").exists():
            self.skipTest("no /proc/cpuinfo")
        if "avx512f" not in Path("/proc/cpuinfo").read_text(errors="ignore"):
            self.skipTest("CPU lacks AVX-512")
        if shutil.which("yosys") is None:
            self.skipTest("yosys not present")
        if shutil.which(os.environ.get("CC", "cc")) is None:
            self.skipTest("C compiler not present")
        if not Path("external-sha3-verilog/low_throughput_core/rtl").exists():
            self.skipTest("external-sha3-verilog not present")

        # Runs an end-to-end smoke+correctness check against hashlib for a tiny message.
        # Uses the generated AVX-512 circuit and simulates the sequential core.
        cmd = [
            os.environ.get("PYTHON", ".venv/bin/python"),
            "scripts/verify_sha3_verilog_avx512.py",
            "--msg",
            "abc",
        ]
        import subprocess

        subprocess.run(cmd, check=True)
