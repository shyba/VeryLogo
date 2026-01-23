import unittest

from stc.bitslice import (
    aes_sbox_bitslice,
    AES_SBOX_TABLE,
    transpose_to_bitplanes,
    transpose_from_bitplanes,
    find_batched_lut8,
    BatchedLut8,
)
from stc.tick_ir import (
    BitTranspose,
    SimdType,
    Var,
    BitVecType,
    Lut8,
    Slice,
    Concat,
    TickIR,
    Bitcast,
)
from stc.interp import eval_expr, infer_type
from stc.backend_x86_sse2 import emit_x86_sse2_c
from stc.bitslice import bitslice_aes_sbox_ir


class TestBitsliceTransform(unittest.TestCase):
    def test_bitslice_transform_correctness(self) -> None:
        from stc.bitslice import bitslice_aes_sbox_ir

        inputs = list(range(16))
        input_val = sum(b << (i * 8) for i, b in enumerate(inputs))

        expr = bitslice_aes_sbox_ir(Var("st"), num_bytes=16)
        types = {"st": SimdType(lane_width=8, lanes=16)}
        result = eval_expr(expr, types, {"st": input_val})

        for i in range(16):
            out_byte = (result >> (i * 8)) & 0xFF
            expected = AES_SBOX_TABLE[inputs[i]]
            self.assertEqual(
                out_byte,
                expected,
                f"byte {i}: sbox[{inputs[i]}] = {expected}, got {out_byte}",
            )

    def test_bitslice_transform_all_256(self) -> None:
        from stc.bitslice import bitslice_aes_sbox_ir

        for batch_start in range(0, 256, 16):
            inputs = list(range(batch_start, min(batch_start + 16, 256)))
            if len(inputs) < 16:
                inputs.extend([0] * (16 - len(inputs)))
            input_val = sum(b << (i * 8) for i, b in enumerate(inputs))

            expr = bitslice_aes_sbox_ir(Var("st"), num_bytes=16)
            types = {"st": SimdType(lane_width=8, lanes=16)}
            result = eval_expr(expr, types, {"st": input_val})

            for i in range(min(16, 256 - batch_start)):
                out_byte = (result >> (i * 8)) & 0xFF
                expected = AES_SBOX_TABLE[inputs[i]]
                self.assertEqual(out_byte, expected)


class TestBatchedLut8Detection(unittest.TestCase):
    def test_detects_16_contiguous_slices(self) -> None:
        table = list(AES_SBOX_TABLE)
        exprs = {
            f"sb{i}": Lut8(x=Slice(x=Var("st"), offset=i * 8, width=8), table=table)
            for i in range(16)
        }
        types = {"st": BitVecType(width=128)}
        batches = find_batched_lut8(exprs, types)
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual(batch.source_var, "st")
        self.assertEqual(batch.table, table)
        self.assertEqual(len(batch.outputs), 16)

    def test_detects_partial_batch(self) -> None:
        table = list(AES_SBOX_TABLE)
        exprs = {
            f"sb{i}": Lut8(x=Slice(x=Var("st"), offset=i * 8, width=8), table=table)
            for i in range(4)
        }
        types = {"st": BitVecType(width=128)}
        batches = find_batched_lut8(exprs, types)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0].outputs), 4)

    def test_different_tables_not_batched(self) -> None:
        table1 = list(AES_SBOX_TABLE)
        table2 = list(reversed(AES_SBOX_TABLE))
        exprs = {
            "sb0": Lut8(x=Slice(x=Var("st"), offset=0, width=8), table=table1),
            "sb1": Lut8(x=Slice(x=Var("st"), offset=8, width=8), table=table2),
        }
        types = {"st": BitVecType(width=128)}
        batches = find_batched_lut8(exprs, types)
        self.assertEqual(len(batches), 2)

    def test_different_sources_not_batched(self) -> None:
        table = list(AES_SBOX_TABLE)
        exprs = {
            "sb0": Lut8(x=Slice(x=Var("st1"), offset=0, width=8), table=table),
            "sb1": Lut8(x=Slice(x=Var("st2"), offset=0, width=8), table=table),
        }
        types = {"st1": BitVecType(width=128), "st2": BitVecType(width=128)}
        batches = find_batched_lut8(exprs, types)
        self.assertEqual(len(batches), 2)


class TestBitTransposeIR(unittest.TestCase):
    def test_bit_transpose_type_inference(self) -> None:
        expr = BitTranspose(x=Var("x"), lane_width=8, lanes=16)
        t = infer_type(expr, {"x": SimdType(lane_width=8, lanes=16)})
        self.assertEqual(t, SimdType(lane_width=16, lanes=8))

    def test_bit_transpose_eval_single_byte(self) -> None:
        expr = BitTranspose(x=Var("x"), lane_width=8, lanes=1)
        types = {"x": SimdType(lane_width=8, lanes=1)}
        result = eval_expr(expr, types, {"x": 0b10110001})
        self.assertEqual(result, 0b10110001)

    def test_bit_transpose_eval_two_bytes(self) -> None:
        expr = BitTranspose(x=Var("x"), lane_width=8, lanes=2)
        types = {"x": SimdType(lane_width=8, lanes=2)}
        input_val = 0b00000010_00000001
        result = eval_expr(expr, types, {"x": input_val})
        self.assertEqual((result >> 0) & 0b11, 0b01)
        self.assertEqual((result >> 2) & 0b11, 0b10)
        for i in range(2, 8):
            self.assertEqual((result >> (i * 2)) & 0b11, 0b00)

    def test_bit_transpose_roundtrip(self) -> None:
        fwd = BitTranspose(x=Var("x"), lane_width=8, lanes=16)
        bwd = BitTranspose(x=fwd, lane_width=16, lanes=8)
        types = {"x": SimdType(lane_width=8, lanes=16)}
        input_val = sum(i << (i * 8) for i in range(16))
        result = eval_expr(bwd, types, {"x": input_val})
        self.assertEqual(result, input_val)


class TestBitslice(unittest.TestCase):
    def test_transpose_roundtrip(self) -> None:
        bytes_in = list(range(16))
        planes = transpose_to_bitplanes(bytes_in, byte_width=8)
        self.assertEqual(len(planes), 8)
        bytes_out = transpose_from_bitplanes(planes, byte_width=8, count=16)
        self.assertEqual(bytes_out, bytes_in)

    def test_transpose_single_byte(self) -> None:
        planes = transpose_to_bitplanes([0b10110001], byte_width=8)
        self.assertEqual(planes[0], 1)
        self.assertEqual(planes[1], 0)
        self.assertEqual(planes[2], 0)
        self.assertEqual(planes[3], 0)
        self.assertEqual(planes[4], 1)
        self.assertEqual(planes[5], 1)
        self.assertEqual(planes[6], 0)
        self.assertEqual(planes[7], 1)

    def test_transpose_two_bytes(self) -> None:
        planes = transpose_to_bitplanes([0b00000001, 0b00000010], byte_width=8)
        self.assertEqual(planes[0], 0b01)
        self.assertEqual(planes[1], 0b10)
        for i in range(2, 8):
            self.assertEqual(planes[i], 0)

    def test_bitsliced_sbox_end_to_end(self) -> None:
        inputs = list(range(16))
        planes_in = transpose_to_bitplanes(inputs, byte_width=8)
        mask = (1 << 16) - 1
        planes_out = aes_sbox_bitslice(planes_in, mask)
        outputs = transpose_from_bitplanes(planes_out, byte_width=8, count=16)
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            self.assertEqual(
                out,
                AES_SBOX_TABLE[inp],
                f"mismatch at {i}: sbox[{inp}] = {AES_SBOX_TABLE[inp]}, got {out}",
            )

    def test_bitsliced_sbox_all_256(self) -> None:
        inputs = list(range(256))
        planes_in = transpose_to_bitplanes(inputs, byte_width=8)
        mask = (1 << 256) - 1
        planes_out = aes_sbox_bitslice(planes_in, mask)
        outputs = transpose_from_bitplanes(planes_out, byte_width=8, count=256)
        for i in range(256):
            self.assertEqual(outputs[i], AES_SBOX_TABLE[i])

    def test_aes_sbox_bitslice_matches_table(self) -> None:
        for i in range(256):
            bits_in = [(i >> b) & 1 for b in range(8)]
            bits_out = aes_sbox_bitslice(bits_in)
            result = sum(bits_out[b] << b for b in range(8))
            self.assertEqual(
                result,
                AES_SBOX_TABLE[i],
                f"mismatch at input {i}: got {result}, expected {AES_SBOX_TABLE[i]}",
            )

    def test_aes_sbox_bitslice_vectorized_64(self) -> None:
        inputs = list(range(64))
        bits_in = [
            sum(((inputs[j] >> b) & 1) << j for j in range(64)) for b in range(8)
        ]
        mask = (1 << 64) - 1
        bits_out = aes_sbox_bitslice(bits_in, mask)
        for j in range(64):
            result = sum(((bits_out[b] >> j) & 1) << b for b in range(8))
            self.assertEqual(
                result,
                AES_SBOX_TABLE[inputs[j]],
                f"mismatch at position {j}: got {result}, expected {AES_SBOX_TABLE[inputs[j]]}",
            )

    def test_aes_sbox_bitslice_all_256_vectorized(self) -> None:
        inputs = list(range(256))
        bits_in = [
            sum(((inputs[j] >> b) & 1) << j for j in range(256)) for b in range(8)
        ]
        mask = (1 << 256) - 1
        bits_out = aes_sbox_bitslice(bits_in, mask)
        for j in range(256):
            result = sum(((bits_out[b] >> j) & 1) << b for b in range(8))
            self.assertEqual(
                result,
                AES_SBOX_TABLE[j],
                f"mismatch at input {j}: got {result}, expected {AES_SBOX_TABLE[j]}",
            )


class TestBitTransposeBackend(unittest.TestCase):
    def test_sse2_bit_transpose_codegen(self) -> None:
        """Test that BitTranspose generates valid SSE2 code."""
        expr = BitTranspose(x=Var("x"), lane_width=8, lanes=16)
        ir = TickIR(
            name="test",
            inputs={"x": SimdType(lane_width=8, lanes=16)},
            state={},
            reset_state={},
            outputs={"y": SimdType(lane_width=16, lanes=8)},
            next_state={},
            output_exprs={"y": expr},
        )
        code = emit_x86_sse2_c(ir)
        self.assertIn("bit_transpose_8x16", code)
        self.assertIn("__m128i", code)

    def test_sse2_bit_transpose_roundtrip_codegen(self) -> None:
        """Test that forward + backward BitTranspose generates valid code."""
        fwd = BitTranspose(x=Var("x"), lane_width=8, lanes=16)
        bwd = BitTranspose(x=fwd, lane_width=16, lanes=8)
        ir = TickIR(
            name="test",
            inputs={"x": SimdType(lane_width=8, lanes=16)},
            state={},
            reset_state={},
            outputs={"y": SimdType(lane_width=8, lanes=16)},
            next_state={},
            output_exprs={"y": bwd},
        )
        code = emit_x86_sse2_c(ir)
        self.assertIn("bit_transpose_8x16", code)
        self.assertIn("bit_transpose_16x8", code)


class TestEndToEndBitslicing(unittest.TestCase):
    def test_lut8_to_bitsliced_equivalence(self) -> None:
        """Test that Lut8-based and bitsliced S-box produce equivalent results."""
        table = list(AES_SBOX_TABLE)

        lut_exprs = {
            f"sb{i}": Lut8(x=Slice(x=Var("st"), offset=i * 8, width=8), table=table)
            for i in range(16)
        }
        types = {"st": BitVecType(width=128)}

        batches = find_batched_lut8(lut_exprs, types)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0].outputs), 16)

        for batch_start in range(0, 256, 16):
            inputs = list(range(batch_start, min(batch_start + 16, 256)))
            if len(inputs) < 16:
                inputs.extend([0] * (16 - len(inputs)))
            input_val = sum(b << (i * 8) for i, b in enumerate(inputs))

            lut_results = []
            for i in range(16):
                lut_result = eval_expr(lut_exprs[f"sb{i}"], types, {"st": input_val})
                lut_results.append(lut_result)

            simd_input = Bitcast(x=Var("st"), to=SimdType(lane_width=8, lanes=16))
            bitsliced_expr = bitslice_aes_sbox_ir(simd_input, num_bytes=16)
            simd_types = {"st": SimdType(lane_width=8, lanes=16)}
            bitsliced_result = eval_expr(bitsliced_expr, simd_types, {"st": input_val})

            for i in range(min(16, 256 - batch_start)):
                out_byte = (bitsliced_result >> (i * 8)) & 0xFF
                self.assertEqual(
                    out_byte,
                    lut_results[i],
                    f"batch {batch_start}, byte {i}: lut={lut_results[i]}, bitsliced={out_byte}",
                )

    def test_full_transformation_pipeline(self) -> None:
        """Test the full transformation: detect batch -> transform -> verify."""
        table = list(AES_SBOX_TABLE)

        original_exprs = {
            f"sb{i}": Lut8(x=Slice(x=Var("st"), offset=i * 8, width=8), table=table)
            for i in range(16)
        }
        types = {"st": BitVecType(width=128)}

        batches = find_batched_lut8(original_exprs, types)
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual(batch.source_var, "st")
        self.assertEqual(batch.table, table)

        simd_input = Bitcast(x=Var("st"), to=SimdType(lane_width=8, lanes=16))
        transformed_expr = bitslice_aes_sbox_ir(simd_input, num_bytes=16)
        simd_types = {"st": SimdType(lane_width=8, lanes=16)}

        for test_val in [
            0x00112233445566778899AABBCCDDEEFF,
            0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
            0x00000000000000000000000000000000,
            0x000102030405060708090A0B0C0D0E0F,
        ]:
            original_results = []
            for i in range(16):
                res = eval_expr(original_exprs[f"sb{i}"], types, {"st": test_val})
                original_results.append(res)
            original_combined = sum(
                r << (i * 8) for i, r in enumerate(original_results)
            )

            transformed_result = eval_expr(
                transformed_expr, simd_types, {"st": test_val}
            )

            self.assertEqual(
                transformed_result,
                original_combined,
                f"mismatch for input {test_val:032x}",
            )
