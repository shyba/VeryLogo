from __future__ import annotations

import unittest

from scripts import bench_aes10_bp128_byteio_cuda as bench_byteio


class TestBenchAes10Bp128ByteIoCodegen(unittest.TestCase):
    def test_converter_kernels_present(self) -> None:
        src = bench_byteio._converter_kernels_source()
        self.assertIn("bytes_to_planes4_kernel", src)
        self.assertIn("planes4_to_bytes_kernel", src)
        self.assertIn("__ballot_sync", src)
        self.assertIn("__shfl_sync", src)

    def test_host_source_has_three_stage_launch(self) -> None:
        host_src = bench_byteio._host_bench_cu_source()
        self.assertIn(
            'cuModuleGetFunction(&fn_in, mod, "bytes_to_planes4_kernel")', host_src
        )
        self.assertIn(
            'cuModuleGetFunction(&fn_aes, mod, "aes10_bp128_kernel")', host_src
        )
        self.assertIn(
            'cuModuleGetFunction(&fn_out, mod, "planes4_to_bytes_kernel")', host_src
        )
        self.assertIn("full_pipeline:", host_src)
        self.assertIn("converter_only:", host_src)
        self.assertIn("aes_only:", host_src)


if __name__ == "__main__":
    unittest.main()
