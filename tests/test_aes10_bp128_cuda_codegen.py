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
        self.assertNotIn(
            'cuModuleGetGlobal(&d_rk, &rk_nbytes, mod, "RK_BITS")', host_src
        )

    def test_coalesced4_paramrk_shared_kernel_uses_shared_staging(self) -> None:
        mapped = build_bp_sbox()
        sbox_src = aes10_cuda._emit_sbox_inline_cuda(mapped)
        kernel_src = aes10_cuda._kernel_cu_source_replacement_coalesced4_paramrk_shared(
            sbox_src
        )
        self.assertIn("extern __shared__ uint4 rk4_shared[];", kernel_src)
        self.assertIn("__syncthreads();", kernel_src)
        self.assertIn(
            "const uint32_t* rk_shared = (const uint32_t*)rk4_shared;", kernel_src
        )

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

    def test_coalesced4_paramrk_soa_packed_kernel_uses_byte_keys(self) -> None:
        mapped = build_bp_sbox()
        sbox_src = aes10_cuda._emit_sbox_inline_cuda(mapped)
        kernel_src = (
            aes10_cuda._kernel_cu_source_replacement_coalesced4_paramrk_soa_packed(
                sbox_src
            )
        )
        self.assertIn("const uint8_t* __restrict__ rk_bytes", kernel_src)
        self.assertIn("#define RK_BYTE_AT", kernel_src)
        self.assertIn("#define RK_AT", kernel_src)

    def test_paramrk_soa_packed_host_builds_packed_key_buffer(self) -> None:
        host_src = aes10_cuda._host_bench_cu_source_replacement_paramrk_soa_packed(
            layout="plane-major4"
        )
        self.assertIn("const size_t rk_planes = (size_t)11u * 16u;", host_src)
        self.assertIn(
            "uint8_t* h_rk_soa_packed = (uint8_t*)malloc(rk_bytes_len);", host_src
        )
        self.assertIn("cuMemAlloc(&d_rk, rk_bytes_len)", host_src)
        self.assertIn("cuMemcpyHtoD(d_rk, h_rk_soa_packed, rk_bytes_len)", host_src)

    def test_paramrk_soa_packed_host_uses_direct_rk_bytes(self) -> None:
        host_src = aes10_cuda._host_bench_cu_source_replacement_paramrk_soa_packed(
            layout="plane-major4"
        )
        self.assertIn("static const uint8_t rk_bytes[11][16]", host_src)
        self.assertIn("uint8_t kv = rk_bytes[r][b];", host_src)
        self.assertNotIn("build_rk_bits(h_rk_bits);", host_src)

    def test_pack_rk_soa_packed_helpers(self) -> None:
        rk_flat = b"".join(bytes.fromhex(x) for x in aes10_cuda.ROUND_KEYS_HEX)
        shared = aes10_cuda.pack_rk_soa_packed_shared_key(rk_flat, threads=4)
        self.assertEqual(len(shared), 11 * 16 * 4)
        for key_idx in range(11 * 16):
            row = shared[key_idx * 4 : (key_idx + 1) * 4]
            self.assertEqual(row, bytes([rk_flat[key_idx]]) * 4)

        thread0 = rk_flat
        thread1 = bytes((b ^ 0x5A) & 0xFF for b in rk_flat)
        packed = aes10_cuda.pack_rk_soa_packed_thread_keys(thread0 + thread1, threads=2)
        self.assertEqual(len(packed), 11 * 16 * 2)
        for key_idx in range(11 * 16):
            self.assertEqual(packed[key_idx * 2 + 0], thread0[key_idx])
            self.assertEqual(packed[key_idx * 2 + 1], thread1[key_idx])

    def test_coalesced4_masterkey_soa_kernel_signature(self) -> None:
        mapped = build_bp_sbox()
        sbox_src = aes10_cuda._emit_sbox_inline_cuda(mapped)
        kernel_src = aes10_cuda._kernel_cu_source_replacement_coalesced4_masterkey_soa(
            sbox_src
        )
        self.assertIn("const uint8_t* __restrict__ key_bytes", kernel_src)
        self.assertIn("__device__ __constant__ uint8_t AES_SBOX_KS[256]", kernel_src)
        self.assertIn("aes128_expand_round_key_u8(", kernel_src)
        self.assertIn("rk[b] = key_bytes[(size_t)b * threads + t];", kernel_src)

    def test_masterkey_soa_host_builds_key_buffer(self) -> None:
        host_src = aes10_cuda._host_bench_cu_source_replacement_masterkey_soa(
            layout="plane-major4"
        )
        self.assertIn("static const uint8_t key_bytes_ref[16]", host_src)
        self.assertIn("uint8_t* h_key_soa = (uint8_t*)malloc(key_bytes_len);", host_src)
        self.assertIn("cuMemAlloc(&d_key, key_bytes_len)", host_src)
        self.assertIn("cuMemcpyHtoD(d_key, h_key_soa, key_bytes_len)", host_src)
        self.assertIn("void* params[] = { &d_in, &d_out, &d_key, &threads };", host_src)

    def test_pack_masterkey_soa_helpers(self) -> None:
        key = bytes(range(16))
        shared = aes10_cuda.pack_masterkey_soa_shared_key(key, threads=4)
        self.assertEqual(len(shared), 16 * 4)
        for key_idx in range(16):
            row = shared[key_idx * 4 : (key_idx + 1) * 4]
            self.assertEqual(row, bytes([key[key_idx]]) * 4)

        thread0 = key
        thread1 = bytes((b ^ 0xA5) & 0xFF for b in key)
        packed = aes10_cuda.pack_masterkey_soa_thread_keys(thread0 + thread1, threads=2)
        self.assertEqual(len(packed), 16 * 2)
        for key_idx in range(16):
            self.assertEqual(packed[key_idx * 2 + 0], thread0[key_idx])
            self.assertEqual(packed[key_idx * 2 + 1], thread1[key_idx])


if __name__ == "__main__":
    unittest.main()
