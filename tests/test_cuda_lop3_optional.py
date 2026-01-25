import shutil
import subprocess
import unittest


class TestCudaLop3Optional(unittest.TestCase):
    def test_bp115_lop3_cuda_matches_cpu(self) -> None:
        # Optional: requires CUDA toolchain + NVIDIA driver + GPU.
        # User requested we skip if nvcc isn't present.
        if shutil.which("nvcc") is None:
            self.skipTest("nvcc not present")
        if shutil.which("ptxas") is None:
            self.skipTest("ptxas not present")
        if shutil.which("nvidia-smi") is None:
            self.skipTest("nvidia-smi not present (no GPU/driver)")
        smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        if smi.returncode != 0 or "GPU" not in smi.stdout:
            self.skipTest("no NVIDIA GPU detected")

        res = subprocess.run(
            [
                ".venv/bin/python",
                "scripts/bench_lop3_cuda.py",
                "--circuit",
                "bp115",
                "--sm",
                "sm_61",
                "--threads",
                "4096",
                "--block",
                "256",
                "--reps",
                "5",
                "--check",
            ],
            env={"PYTHONPATH": "."},
            capture_output=True,
            text=True,
            timeout=600,
        )
        if res.returncode != 0:
            msg = (res.stdout + "\n" + res.stderr).strip()
            self.fail(msg[:4000])

        # Script doesn't print PASS; it raises on mismatch.
        self.assertIn("circuit=bp115", res.stdout)

