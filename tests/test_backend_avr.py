import unittest

from stc.backend_avr import emit_avr_c
from stc.tick_ir import BitVecType, BoolType, Not, TickIR, Var


class TestBackendAvr(unittest.TestCase):
    def test_codegen_basic(self) -> None:
        ir = TickIR(
            name="top",
            inputs={"i": BoolType()},
            outputs={"o": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Not(x=Var(name="i"))},
        )

        c = emit_avr_c(ir)
        self.assertIn("void stc_tick(void)", c)
        self.assertIn("(PINB >> 0)", c)
        self.assertIn("out_o", c)

    def test_codegen_bitvec_gpio_mapping(self) -> None:
        ir = TickIR(
            name="top",
            inputs={"a": BitVecType(width=2)},
            outputs={"y": BitVecType(width=2)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": Var(name="a")},
        )

        c = emit_avr_c(ir)
        self.assertIn("(PINB >> 0) & 0x3u", c)
        self.assertIn("<< 2", c)
