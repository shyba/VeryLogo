import unittest

from stc.backend_x86_sse2 import emit_x86_sse2_c
from stc.reduce import reduce_expr
from stc.replace import replace_vars
from stc.tick_ir import (
    SimdConst,
    SimdPackSS16To8,
    SimdPackSS32To16,
    SimdPackUS16To8,
    SimdType,
    SimdUnpackHi,
    SimdUnpackLo,
    TickIR,
    Var,
)
from stc.tick_ir_to_verilog import emit_verilog


class TestSimdPackUnpackPlumbing(unittest.TestCase):
    def test_reduce_expr_supports_pack_unpack(self) -> None:
        t = SimdType(lane_width=16, lanes=8)
        expr = SimdUnpackLo(
            a=SimdConst(lane_width=16, lanes=8, value=0),
            b=SimdConst(lane_width=16, lanes=8, value=1),
        )
        reduced = reduce_expr(expr, {})
        self.assertIsInstance(reduced, SimdUnpackLo)

    def test_replace_vars_supports_pack_unpack(self) -> None:
        t = SimdType(lane_width=16, lanes=8)
        expr = SimdPackSS16To8(a=Var("x"), b=Var("y"))
        out = replace_vars(
            expr,
            {
                "x": SimdConst(lane_width=16, lanes=8, value=0),
                "y": SimdConst(lane_width=16, lanes=8, value=1),
            },
        )
        self.assertIsInstance(out, SimdPackSS16To8)

    def test_emit_verilog_supports_pack_unpack(self) -> None:
        t16 = SimdType(lane_width=16, lanes=8)
        t8 = SimdType(lane_width=8, lanes=16)
        t32 = SimdType(lane_width=32, lanes=4)
        ir = TickIR(
            name="pack",
            inputs={"x16": t16, "y16": t16, "x32": t32, "y32": t32},
            outputs={"unlo": t16, "unhi": t16, "pss": t8, "pus": t8, "psd": t16},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "unlo": SimdUnpackLo(a=Var("x16"), b=Var("y16")),
                "unhi": SimdUnpackHi(a=Var("x16"), b=Var("y16")),
                "pss": SimdPackSS16To8(a=Var("x16"), b=Var("y16")),
                "pus": SimdPackUS16To8(a=Var("x16"), b=Var("y16")),
                "psd": SimdPackSS32To16(a=Var("x32"), b=Var("y32")),
            },
        )
        v = emit_verilog(ir)
        self.assertIn("module", v)

    def test_x86_sse2_codegen_supports_pack_unpack(self) -> None:
        t16 = SimdType(lane_width=16, lanes=8)
        t8 = SimdType(lane_width=8, lanes=16)
        t32 = SimdType(lane_width=32, lanes=4)
        ir = TickIR(
            name="pack",
            inputs={"x16": t16, "y16": t16, "x32": t32, "y32": t32},
            outputs={"unlo": t16, "unhi": t16, "pss": t8, "pus": t8, "psd": t16},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "unlo": SimdUnpackLo(a=Var("x16"), b=Var("y16")),
                "unhi": SimdUnpackHi(a=Var("x16"), b=Var("y16")),
                "pss": SimdPackSS16To8(a=Var("x16"), b=Var("y16")),
                "pus": SimdPackUS16To8(a=Var("x16"), b=Var("y16")),
                "psd": SimdPackSS32To16(a=Var("x32"), b=Var("y32")),
            },
        )
        c = emit_x86_sse2_c(ir)
        self.assertIn("_mm_unpacklo_epi16", c)
        self.assertIn("_mm_unpackhi_epi16", c)
        self.assertIn("_mm_packs_epi16", c)
        self.assertIn("_mm_packus_epi16", c)
        self.assertIn("_mm_packs_epi32", c)
