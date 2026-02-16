from __future__ import annotations

import unittest

from stc.aes_bp128_variant_family import (
    Bp128DispatchRequest,
    Bp128Dispatcher,
    all_variant_metadata,
    build_variant_metadata,
    family_manifest_json,
    generate_cuda_source,
    make_key_material_soa,
)


class TestAesBp128VariantFamily(unittest.TestCase):
    def test_catalog_size_and_uniqueness(self) -> None:
        variants = all_variant_metadata()
        self.assertEqual(len(variants), 54)
        ids = {item.variant_id for item in variants}
        kernels = {item.kernel_name for item in variants}
        self.assertEqual(len(ids), 54)
        self.assertEqual(len(kernels), 54)

    def test_support_matrix_marks_fallbacks_and_bytes_supported(self) -> None:
        meta = build_variant_metadata(
            key_bits=256,
            ctr_group=1,
            key_source="masterkey_soa",
            io_layout="plane-major4",
        )
        self.assertTrue(meta.supported)
        self.assertIn("fallback", meta.notes)

        meta_const = build_variant_metadata(
            key_bits=256,
            ctr_group=1,
            key_source="const_key",
            io_layout="plane-major4",
        )
        self.assertTrue(meta_const.supported)
        self.assertIn("AES-192/256", meta_const.notes)

        meta_bytes = build_variant_metadata(
            key_bits=128,
            ctr_group=1,
            key_source="masterkey_soa",
            io_layout="bytes",
        )
        self.assertTrue(meta_bytes.supported)
        self.assertIn("converter kernels", meta_bytes.notes)

    def test_dispatch_prefers_supported_plane_major4(self) -> None:
        dispatcher = Bp128Dispatcher()
        picked = dispatcher.select(
            Bp128DispatchRequest(
                key_bits=128,
                key_source="masterkey_soa",
                io_layout="plane-major4",
                max_ctr_group=4,
            )
        )
        self.assertTrue(picked.supported)
        self.assertEqual(picked.key_source, "masterkey_soa")
        self.assertEqual(picked.io_layout, "plane-major4")

    def test_manifest_contains_support_flags(self) -> None:
        text = family_manifest_json(indent=0)
        self.assertIn('"supported": true', text)
        self.assertNotIn('"supported": false', text)
        self.assertIn('"family": "bp128-fast"', text)
        self.assertIn('"io_layout": "bytes"', text)
        self.assertIn("out_len_bytes", text)
        self.assertIn("tail_bytes", text)

    def test_make_key_material_sizes(self) -> None:
        meta_master = build_variant_metadata(
            key_bits=128,
            ctr_group=4,
            key_source="masterkey_soa",
            io_layout="plane-major4",
        )
        buf_master = make_key_material_soa(meta_master, logical_threads=8, seed=7)
        self.assertEqual(len(buf_master), 8 * 4 * 16)

        meta_master256 = build_variant_metadata(
            key_bits=256,
            ctr_group=2,
            key_source="masterkey_soa",
            io_layout="plane-major4",
        )
        buf_master256 = make_key_material_soa(meta_master256, logical_threads=8, seed=7)
        self.assertEqual(len(buf_master256), 8 * 2 * (15 * 16))

        meta_expanded = build_variant_metadata(
            key_bits=256,
            ctr_group=2,
            key_source="expanded_rk_soa",
            io_layout="plane-major4",
        )
        buf_expanded = make_key_material_soa(meta_expanded, logical_threads=8, seed=7)
        self.assertEqual(len(buf_expanded), 8 * 2 * (15 * 16))

        meta_const = build_variant_metadata(
            key_bits=128,
            ctr_group=1,
            key_source="const_key",
            io_layout="plane-major4",
        )
        buf_const = make_key_material_soa(meta_const, logical_threads=8, seed=7)
        self.assertEqual(len(buf_const), 0)

    def test_generate_source_patches_rounds_for_aes256_expanded(self) -> None:
        meta = build_variant_metadata(
            key_bits=256,
            ctr_group=1,
            key_source="expanded_rk_soa",
            io_layout="plane-major4",
        )
        try:
            src = generate_cuda_source(meta)
        except RuntimeError as exc:
            self.skipTest(str(exc))
        self.assertIn("for (int round = 1; round <= 13; round++) {", src)
        self.assertIn("uint8_t rkf = RK_BYTE_AT(14,b);", src)
        self.assertIn("o0.x = sr[b][0] ^ RK_MASK(rkf,0);", src)
        self.assertIn(meta.kernel_name, src)

    def test_generate_source_patches_rounds_for_aes256_masterkey_fallback(self) -> None:
        meta = build_variant_metadata(
            key_bits=256,
            ctr_group=1,
            key_source="masterkey_soa",
            io_layout="plane-major4",
        )
        try:
            src = generate_cuda_source(meta)
        except RuntimeError as exc:
            self.skipTest(str(exc))
        self.assertIn("for (int round = 1; round <= 13; round++) {", src)
        self.assertIn("const uint8_t* __restrict__ rk_bytes", src)
        self.assertIn(meta.kernel_name, src)

    def test_generate_source_patches_const_key_aes256(self) -> None:
        meta = build_variant_metadata(
            key_bits=256,
            ctr_group=1,
            key_source="const_key",
            io_layout="plane-major4",
        )
        try:
            src = generate_cuda_source(meta)
        except RuntimeError as exc:
            self.skipTest(str(exc))
        self.assertIn("__device__ __constant__ uint8_t RK_BYTES_CONST[15][16];", src)
        self.assertNotIn("const uint8_t* __restrict__ rk_bytes", src)
        self.assertIn("RK_BYTE_AT(r,b) RK_BYTES_CONST[(r)][(b)]", src)

    def test_generate_source_supports_xor_accumulate_post_op(self) -> None:
        meta = build_variant_metadata(
            key_bits=128,
            ctr_group=1,
            key_source="masterkey_soa",
            io_layout="plane-major4",
        )
        try:
            src = generate_cuda_source(meta, post_op="xor_accumulate")
        except RuntimeError as exc:
            self.skipTest(str(exc))
        self.assertIn("uint4 prev0 = out4[idx0];", src)
        self.assertIn("o0.x ^= prev0.x;", src)

    def test_generate_source_for_bytes_layout_includes_converters(self) -> None:
        meta = build_variant_metadata(
            key_bits=128,
            ctr_group=1,
            key_source="const_key",
            io_layout="bytes",
        )
        try:
            src = generate_cuda_source(meta)
        except RuntimeError as exc:
            self.skipTest(str(exc))
        self.assertIn(f'extern "C" __global__ void {meta.kernel_name}', src)
        self.assertIn("stc_bytes_to_planes4_kernel", src)
        self.assertIn("stc_planes4_to_bytes_kernel", src)
        self.assertIn("uint32_t out_len_bytes", src)
        self.assertIn("uint32_t tail_bytes", src)


if __name__ == "__main__":
    unittest.main()
