import unittest

from stc.backend_x86_auto import AutoBackendError, emit_x86_auto_c
from stc.tick_ir import (
    SimdAdd,
    SimdAddMasked,
    SimdFDiv,
    SimdFFma,
    SimdMinU,
    SimdPackSS16To8,
    SimdPackSS32To16,
    SimdShuffle,
    SimdType,
    SimdUlt,
    TickIR,
    Var,
)


class TestBackendX86Auto(unittest.TestCase):
    def test_selects_sse2_for_128(self) -> None:
        t = SimdType(lane_width=32, lanes=4)
        ir = TickIR(
            name="t128",
            inputs={"x": t, "y": t},
            outputs={"o": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": SimdAdd(a=Var("x"), b=Var("y"))},
        )
        c = emit_x86_auto_c(ir, flags=set())
        self.assertIn("<emmintrin.h>", c)

    def test_selects_avx2_for_256(self) -> None:
        t = SimdType(lane_width=32, lanes=8)
        ir = TickIR(
            name="t256",
            inputs={"x": t, "y": t},
            outputs={"o": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": SimdAdd(a=Var("x"), b=Var("y"))},
        )
        c = emit_x86_auto_c(ir, flags={"avx2"})
        self.assertIn("_mm256_add_epi32", c)

    def test_selects_avx512vl_for_256_masked_ops(self) -> None:
        t = SimdType(lane_width=32, lanes=8)
        ir = TickIR(
            name="t256m",
            inputs={"x": t, "y": t},
            outputs={"o": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "o": SimdAddMasked(
                    mask=SimdUlt(a=Var("x"), b=Var("y")),
                    passthru=Var("x"),
                    a=Var("x"),
                    b=Var("y"),
                )
            },
        )
        with self.assertRaises(AutoBackendError):
            emit_x86_auto_c(ir, flags={"avx2"})
        c = emit_x86_auto_c(ir, flags={"avx2", "avx512f", "avx512vl"})
        self.assertIn("_mm256_mask_add_epi32", c)

    def test_selects_avx_for_256_float_ops(self) -> None:
        t = SimdType(lane_width=32, lanes=8)
        ir = TickIR(
            name="t256f",
            inputs={"x": t, "y": t},
            outputs={"o": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": SimdFDiv(a=Var("x"), b=Var("y"))},
        )
        with self.assertRaises(AutoBackendError):
            emit_x86_auto_c(ir, flags={"avx2"})
        c = emit_x86_auto_c(ir, flags={"avx"})
        self.assertIn("_mm256_div_ps", c)

    def test_requires_fma_for_256_fma(self) -> None:
        t = SimdType(lane_width=32, lanes=8)
        ir = TickIR(
            name="t256fma",
            inputs={"x": t, "y": t, "z": t},
            outputs={"o": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": SimdFFma(a=Var("x"), b=Var("y"), c=Var("z"))},
        )
        with self.assertRaises(AutoBackendError):
            emit_x86_auto_c(ir, flags={"avx"})
        c = emit_x86_auto_c(ir, flags={"avx", "fma"})
        self.assertIn("_mm256_fmadd_ps", c)

    def test_selects_avx512_for_512(self) -> None:
        t = SimdType(lane_width=32, lanes=16)
        ir = TickIR(
            name="t512",
            inputs={"x": t, "y": t},
            outputs={"o": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": SimdAdd(a=Var("x"), b=Var("y"))},
        )
        c = emit_x86_auto_c(ir, flags={"avx512f"})
        self.assertIn("_mm512_add_epi32", c)

    def test_requires_fma_for_512_fma(self) -> None:
        t = SimdType(lane_width=32, lanes=16)
        ir = TickIR(
            name="t512fma",
            inputs={"x": t, "y": t, "z": t},
            outputs={"o": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": SimdFFma(a=Var("x"), b=Var("y"), c=Var("z"))},
        )
        with self.assertRaises(AutoBackendError):
            emit_x86_auto_c(ir, flags={"avx512f"})
        c = emit_x86_auto_c(ir, flags={"avx512f", "fma"})
        self.assertIn("_mm512_fmadd_ps", c)

    def test_requires_avx2_flag_for_256(self) -> None:
        t = SimdType(lane_width=32, lanes=8)
        ir = TickIR(
            name="t256",
            inputs={"x": t, "y": t},
            outputs={"o": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": SimdAdd(a=Var("x"), b=Var("y"))},
        )
        with self.assertRaises(AutoBackendError):
            emit_x86_auto_c(ir, flags=set())

    def test_requires_avx512f_flag_for_512(self) -> None:
        t = SimdType(lane_width=32, lanes=16)
        ir = TickIR(
            name="t512",
            inputs={"x": t, "y": t},
            outputs={"o": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": SimdAdd(a=Var("x"), b=Var("y"))},
        )
        with self.assertRaises(AutoBackendError):
            emit_x86_auto_c(ir, flags={"avx2"})

    def test_requires_ssse3_for_shuffle(self) -> None:
        t = SimdType(lane_width=8, lanes=16)
        ir = TickIR(
            name="t128",
            inputs={"x": t},
            outputs={"o": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "o": SimdShuffle(x=Var("x"), indices=list(reversed(range(16))))
            },
        )
        with self.assertRaises(AutoBackendError):
            emit_x86_auto_c(ir, flags=set())
        c = emit_x86_auto_c(ir, flags={"ssse3"})
        self.assertIn("_mm_shuffle_epi8", c)

    def test_requires_avx512bw_for_epi16_ops_even_if_inputs_are_epi32(self) -> None:
        t32 = SimdType(lane_width=32, lanes=16)
        t16 = SimdType(lane_width=16, lanes=32)
        packed = SimdPackSS32To16(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="t512_pack_then_add_epi16",
            inputs={"x": t32, "y": t32},
            outputs={"o": t16},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": SimdAdd(a=packed, b=packed)},
        )
        with self.assertRaises(AutoBackendError):
            emit_x86_auto_c(ir, flags={"avx512f"})
        c = emit_x86_auto_c(ir, flags={"avx512f", "avx512bw"})
        self.assertIn("_mm512_add_epi16", c)

    def test_requires_avx512vbmi_for_epi8_shuffle(self) -> None:
        t16 = SimdType(lane_width=16, lanes=32)
        t8 = SimdType(lane_width=8, lanes=64)
        packed = SimdPackSS16To8(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="t512_pack_then_shuffle_epi8",
            inputs={"x": t16, "y": t16},
            outputs={"o": t8},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "o": SimdShuffle(x=packed, indices=list(reversed(range(64))))
            },
        )
        with self.assertRaises(AutoBackendError):
            emit_x86_auto_c(ir, flags={"avx512f", "avx512bw"})
        c = emit_x86_auto_c(ir, flags={"avx512f", "avx512vbmi"})
        self.assertIn("_mm512_permutexvar_epi8", c)

    def test_requires_sse41_for_minmax_or_extend_or_blend(self) -> None:
        t = SimdType(lane_width=8, lanes=16)
        ir = TickIR(
            name="t128",
            inputs={"x": t, "y": t},
            outputs={"o": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": SimdMinU(a=Var("x"), b=Var("y"))},
        )
        with self.assertRaises(AutoBackendError):
            emit_x86_auto_c(ir, flags=set())
        c = emit_x86_auto_c(ir, flags={"sse4_1"})
        self.assertIn("_mm_min_epu8", c)
