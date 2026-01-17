import shutil
import tempfile
import unittest
from pathlib import Path

from stc.extract import extract_tick_ir
from stc.tick_ir import Add, And, Eq, Mux, Not, Or, Sub, Xor
from stc.tick_ir_validate import validate_tick_ir
from stc.yosys_frontend import run_yosys
from stc.yosys_json import load_design


@unittest.skipUnless(shutil.which("yosys"), "requires yosys")
class TestVerilogMatrixOptional(unittest.TestCase):
    def test_basic_ops_fixture_extracts_expected_nodes(self) -> None:
        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog_matrix" / "basic_ops.v"

        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            normalized = out_dir / "normalized.json"
            run_yosys(src, normalized, top="top")
            design = load_design(normalized, top="top")
            ir = extract_tick_ir(design)
            validate_tick_ir(ir)

            self.assertIsInstance(ir.output_exprs["o_not"], Not)
            self.assertIsInstance(ir.output_exprs["o_and"], And)
            self.assertIsInstance(ir.output_exprs["o_or"], Or)
            self.assertIsInstance(ir.output_exprs["o_xor"], Xor)
            self.assertIsInstance(ir.output_exprs["o_add"], Add)
            self.assertIsInstance(ir.output_exprs["o_sub"], Sub)
            self.assertIsInstance(ir.output_exprs["o_mux"], Mux)
            self.assertIsInstance(ir.output_exprs["o_eq"], Eq)
