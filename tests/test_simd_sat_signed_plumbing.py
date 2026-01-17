import unittest

from stc.backend_x86_sse2 import emit_x86_sse2_c
from stc.reduce import reduce_expr
from stc.replace import replace_vars
from stc.tick_ir import (
    SimdAddSatS,
    SimdConst,
    SimdSubSatS,
    SimdType,
    TickIR,
    Var,
)
from stc.tick_ir_to_verilog import emit_verilog


class TestSimdSatSignedPlumbing(unittest.TestCase):
    def test_reduce_expr_supports_simd_signed_saturating_nodes(self) -> None:
        t = SimdType(lane_width=8, lanes=16)
        expr = SimdAddSatS(
            a=SimdConst(lane_width=8, lanes=16, value=0x00),
            b=SimdConst(lane_width=8, lanes=16, value=0xFF),
        )
        reduced = reduce_expr(expr, {})
        self.assertIsInstance(reduced, SimdAddSatS)

    def test_replace_vars_supports_simd_signed_saturating_nodes(self) -> None:
        t = SimdType(lane_width=16, lanes=8)
        expr = SimdSubSatS(a=Var("x"), b=Var("y"))
        out = replace_vars(
            expr,
            {
                "x": SimdConst(lane_width=16, lanes=8, value=0),
                "y": SimdConst(lane_width=16, lanes=8, value=1),
            },
        )
        self.assertIsInstance(out, SimdSubSatS)

    def test_emit_verilog_supports_simd_signed_saturating_nodes(self) -> None:
        t = SimdType(lane_width=8, lanes=16)
        ir = TickIR(
            name="sats",
            inputs={"x": t, "y": t},
            outputs={"o": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": SimdAddSatS(a=Var("x"), b=Var("y"))},
        )
        v = emit_verilog(ir)
        self.assertIn("module", v)

    def test_x86_sse2_codegen_supports_signed_saturating_add_sub(self) -> None:
        t8 = SimdType(lane_width=8, lanes=16)
        ir8 = TickIR(
            name="sats8",
            inputs={"x": t8, "y": t8},
            outputs={"adds": t8},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"adds": SimdAddSatS(a=Var("x"), b=Var("y"))},
        )
        c8 = emit_x86_sse2_c(ir8)
        self.assertIn("_mm_adds_epi8", c8)

        t16 = SimdType(lane_width=16, lanes=8)
        ir16 = TickIR(
            name="sats16",
            inputs={"x": t16, "y": t16},
            outputs={"subs": t16},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"subs": SimdSubSatS(a=Var("x"), b=Var("y"))},
        )
        c16 = emit_x86_sse2_c(ir16)
        self.assertIn("_mm_subs_epi16", c16)
