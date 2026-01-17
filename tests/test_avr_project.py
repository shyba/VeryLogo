import unittest

from stc.avr_project import emit_avr_project
from stc.io_map import default_io_map
from stc.tick_ir import BoolType, TickIR, Var
from stc.tick_ir_validate import validate_tick_ir


class TestAvrProject(unittest.TestCase):
    def test_emit_avr_project_has_makefile_and_main(self) -> None:
        ir = TickIR(
            name="t",
            inputs={"i": BoolType()},
            outputs={"o": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Var("i")},
        )
        validate_tick_ir(ir)
        io_map = default_io_map(ir)
        files = emit_avr_project(ir, io_map=io_map)
        self.assertIn("main.c", files)
        self.assertIn("Makefile", files)
        self.assertIn("attiny85", files["Makefile"])
