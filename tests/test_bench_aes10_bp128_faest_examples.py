from __future__ import annotations

import unittest

from scripts import bench_aes10_bp128_faest_examples as faest_bench


class TestBenchAes10Bp128FaestExamples(unittest.TestCase):
    def test_parse_best(self) -> None:
        sample = """
threads=65536 block=64 reps=200
elapsed=300.000 ms
aes10_bp128_ptx: 1.000 ns/eval, 1.000B evals/sec, 16000.00 MiB/s
threads=65536 block=128 reps=200
elapsed=200.000 ms
aes10_bp128_ptx: 0.700 ns/eval, 1.500B evals/sec, 24000.00 MiB/s
"""
        eval_b, mib_s, block = faest_bench._parse_best(sample)
        self.assertAlmostEqual(eval_b, 1.5, places=6)
        self.assertAlmostEqual(mib_s, 24000.0, places=6)
        self.assertEqual(block, 128)

    def test_parse_best_raises_without_perf(self) -> None:
        with self.assertRaises(RuntimeError):
            faest_bench._parse_best("threads=1 block=64 reps=1\n")


if __name__ == "__main__":
    unittest.main()
