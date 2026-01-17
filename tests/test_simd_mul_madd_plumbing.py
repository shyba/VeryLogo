import unittest

from stc.backend_x86_sse2 import emit_x86_sse2_c
from stc.reduce import reduce_expr
from stc.replace import replace_vars
from stc.tick_ir import (
    SimdConst,
    SimdMaddS16,
    SimdMulHiS,
    SimdMulHiU,
    SimdMulLo,
    SimdType,
    TickIR,
    Var,
)
from stc.tick_ir_to_verilog import emit_verilog


class TestSimdMulMaddPlumbing(unittest.TestCase):
    def test_reduce_expr_supports_mul_and_madd(self) -> None:
        t = SimdType(lane_width=16, lanes=8)
        expr = SimdMulLo(
            a=SimdConst(lane_width=16, lanes=8, value=0),
            b=SimdConst(lane_width=16, lanes=8, value=1),
        )
        reduced = reduce_expr(expr, {})
        self.assertIsInstance(reduced, SimdMulLo)

    def test_replace_vars_supports_mul_and_madd(self) -> None:
        t = SimdType(lane_width=16, lanes=8)
        expr = SimdMulHiU(a=Var("x"), b=Var("y"))
        out = replace_vars(
            expr,
            {
                "x": SimdConst(lane_width=16, lanes=8, value=0),
                "y": SimdConst(lane_width=16, lanes=8, value=1),
            },
        )
        self.assertIsInstance(out, SimdMulHiU)

    def test_emit_verilog_supports_mul_and_madd(self) -> None:
        t = SimdType(lane_width=16, lanes=8)
        o32 = SimdType(lane_width=32, lanes=4)
        ir = TickIR(
            name="mul",
            inputs={"x": t, "y": t},
            outputs={"lo": t, "hiu": t, "his": t, "madd": o32},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "lo": SimdMulLo(a=Var("x"), b=Var("y")),
                "hiu": SimdMulHiU(a=Var("x"), b=Var("y")),
                "his": SimdMulHiS(a=Var("x"), b=Var("y")),
                "madd": SimdMaddS16(a=Var("x"), b=Var("y")),
            },
        )
        v = emit_verilog(ir)
        self.assertIn("module", v)

    def test_x86_sse2_codegen_supports_mul_and_madd(self) -> None:
        t = SimdType(lane_width=16, lanes=8)
        o32 = SimdType(lane_width=32, lanes=4)
        ir = TickIR(
            name="mul",
            inputs={"x": t, "y": t},
            outputs={"lo": t, "hiu": t, "his": t, "madd": o32},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "lo": SimdMulLo(a=Var("x"), b=Var("y")),
                "hiu": SimdMulHiU(a=Var("x"), b=Var("y")),
                "his": SimdMulHiS(a=Var("x"), b=Var("y")),
                "madd": SimdMaddS16(a=Var("x"), b=Var("y")),
            },
        )
        c = emit_x86_sse2_c(ir)
        self.assertIn("_mm_mullo_epi16", c)
        self.assertIn("_mm_mulhi_epu16", c)
        self.assertIn("_mm_mulhi_epi16", c)
        self.assertIn("_mm_madd_epi16", c)
