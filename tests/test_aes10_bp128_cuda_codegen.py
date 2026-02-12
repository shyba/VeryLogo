from __future__ import annotations

import unittest

from scripts import bench_aes10_bp128_cuda as aes10_cuda
from scripts.bp_circuit_sbox import build_bp_sbox


class TestAes10Bp128CudaCodegen(unittest.TestCase):
    def test_legacy_kernel_signature(self) -> None:
        mapped = build_bp_sbox()
        sbox_src = aes10_cuda._emit_sbox_inline_cuda(mapped)
        kernel_src = aes10_cuda._kernel_cu_source_legacy(sbox_src, input_mode="const")
        self.assertIn("uint32_t* out_ptr", kernel_src)
        self.assertIn("uint32_t n_threads", kernel_src)
        self.assertNotIn("const uint32_t* __restrict__ in_ptr", kernel_src)

    def test_replacement_kernel_signature_and_rk_symbol(self) -> None:
        mapped = build_bp_sbox()
        sbox_src = aes10_cuda._emit_sbox_inline_cuda(mapped)
        kernel_src = aes10_cuda._kernel_cu_source_replacement(sbox_src)
        self.assertIn("const uint32_t* __restrict__ in_ptr", kernel_src)
        self.assertIn("uint32_t* __restrict__ out_ptr", kernel_src)
        self.assertIn(
            "__device__ __constant__ uint32_t RK_BITS[11][16][8];", kernel_src
        )

    def test_replacement_host_uploads_round_keys(self) -> None:
        host_src = aes10_cuda._host_bench_cu_source_replacement()
        self.assertIn('cuModuleGetGlobal(&d_rk, &rk_nbytes, mod, "RK_BITS")', host_src)
        self.assertIn("cuMemcpyHtoD(d_rk, h_rk_bits, sizeof(h_rk_bits))", host_src)

    def test_streamed_kernels_remove_full_round_intermediates(self) -> None:
        mapped = build_bp_sbox()
        sbox_src = aes10_cuda._emit_sbox_inline_cuda(mapped)
        legacy_streamed = aes10_cuda._kernel_cu_source_legacy_streamed_inplace(
            sbox_src, input_mode="const"
        )
        replacement_streamed = (
            aes10_cuda._kernel_cu_source_replacement_streamed_inplace(sbox_src)
        )
        for kernel_src in (legacy_streamed, replacement_streamed):
            self.assertNotIn("uint32_t sb[16][8];", kernel_src)
            self.assertNotIn("uint32_t sr[16][8];", kernel_src)
            self.assertNotIn("uint32_t mc[16][8];", kernel_src)
            self.assertIn("mix_shifted_col_addkey(", kernel_src)
            self.assertIn("subbyte_addkey_store(", kernel_src)

    def test_coalesced_kernels_have_plane_major_indexing(self) -> None:
        mapped = build_bp_sbox()
        sbox_src = aes10_cuda._emit_sbox_inline_cuda(mapped)
        k_coal = aes10_cuda._kernel_cu_source_replacement_coalesced(sbox_src)
        self.assertIn("plane * threads + t", k_coal)
        self.assertIn("in_ptr[plane * threads + t]", k_coal)
        self.assertIn("out_ptr[plane * threads + t]", k_coal)

        k_coal4 = aes10_cuda._kernel_cu_source_replacement_coalesced4(sbox_src)
        self.assertIn("const uint4* in4 = (const uint4*)in_ptr;", k_coal4)
        self.assertIn("uint4* out4 = (uint4*)out_ptr;", k_coal4)
        self.assertIn("out4[idx0] = o0;", k_coal4)

    def test_replacement_host_layouts_generate(self) -> None:
        host_thread = aes10_cuda._host_bench_cu_source_replacement()
        host_plane = aes10_cuda._host_bench_cu_source_replacement("plane-major")
        host_plane4 = aes10_cuda._host_bench_cu_source_replacement("plane-major4")
        self.assertIn("size_t base = (size_t)tid * 128u;", host_thread)
        self.assertIn("plane * (size_t)threads + (size_t)tid", host_plane)
        self.assertIn("uint4* in4 = (uint4*)h_in;", host_plane4)


    def test_coalesced4_paramrk_kernel_signature(self) -> None:
        mapped = build_bp_sbox()
        sbox_src = aes10_cuda._emit_sbox_inline_cuda(mapped)
        kernel_src = aes10_cuda._kernel_cu_source_replacement_coalesced4_paramrk(
            sbox_src
        )
        self.assertIn("const uint32_t* __restrict__ rk_bits", kernel_src)
        self.assertNotIn("__device__ __constant__ uint32_t RK_BITS", kernel_src)
        self.assertIn("#define RK_AT", kernel_src)

    def test_paramrk_host_uses_explicit_key_buffer(self) -> None:
        host_src = aes10_cuda._host_bench_cu_source_replacement_paramrk(
            layout="plane-major4"
        )
        self.assertIn("cuMemAlloc(&d_rk, sizeof(h_rk_bits))", host_src)
        self.assertIn("cuMemcpyHtoD(d_rk, h_rk_bits, sizeof(h_rk_bits))", host_src)
        self.assertIn("void* params[] = { &d_in, &d_out, &d_rk, &threads };", host_src)
        self.assertNotIn('cuModuleGetGlobal(&d_rk, &rk_nbytes, mod, "RK_BITS")', host_src)


    def test_coalesced4_paramrk_shared_kernel_uses_shared_staging(self) -> None:
        mapped = build_bp_sbox()
        sbox_src = aes10_cuda._emit_sbox_inline_cuda(mapped)
        kernel_src = aes10_cuda._kernel_cu_source_replacement_coalesced4_paramrk_shared(
            sbox_src
        )
        self.assertIn("extern __shared__ uint4 rk4_shared[];", kernel_src)
        self.assertIn("__syncthreads();", kernel_src)
        self.assertIn("const uint32_t* rk_shared = (const uint32_t*)rk4_shared;", kernel_src)

    def test_paramrk_shared_host_sets_dynamic_shared_bytes(self) -> None:
        host_src = aes10_cuda._host_bench_cu_source_replacement_paramrk_shared(
            layout="plane-major4"
        )
        self.assertIn(
            "const unsigned int rk_shared_bytes = (unsigned int)sizeof(h_rk_bits);",
            host_src,
        )
        self.assertIn(
            "cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, rk_shared_bytes, 0, params, 0)",
            host_src,
        )


    def test_coalesced4_paramrk_soa_kernel_uses_thread_soa_index(self) -> None:
        mapped = build_bp_sbox()
        sbox_src = aes10_cuda._emit_sbox_inline_cuda(mapped)
        kernel_src = aes10_cuda._kernel_cu_source_replacement_coalesced4_paramrk_soa(
            sbox_src
        )
        self.assertIn("* threads) + t)]", kernel_src)

    def test_paramrk_soa_host_builds_soa_key_buffer(self) -> None:
        host_src = aes10_cuda._host_bench_cu_source_replacement_paramrk_soa(
            layout="plane-major4"
        )
        self.assertIn("const size_t rk_planes = (size_t)11u * 16u * 8u;", host_src)
        self.assertIn("uint32_t* h_rk_soa = (uint32_t*)malloc(rk_bytes);", host_src)
        self.assertIn("cuMemAlloc(&d_rk, rk_bytes)", host_src)
        self.assertIn("void* params[] = { &d_in, &d_out, &d_rk, &threads };", host_src)


if __name__ == "__main__":
    unittest.main()
