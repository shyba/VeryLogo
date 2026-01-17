import shutil
import tempfile
import unittest
from pathlib import Path

from stc.extract import extract_tick_ir
from stc.tick_ir import AShr, LShr, Shl, Sub, Uge, Ugt, Ule, Ult
from stc.tick_ir_validate import validate_tick_ir
from stc.yosys_frontend import run_yosys
from stc.yosys_json import load_design


@unittest.skipUnless(shutil.which("yosys"), "requires yosys")
class TestExtractOpsOptional(unittest.TestCase):
    def test_extract_sub_shifts_compares(self) -> None:
        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog" / "ops_sub_shift_cmp.v"

        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            normalized = out_dir / "normalized.json"
            run_yosys(src, normalized, top="top")
            design = load_design(normalized, top="top")
            ir = extract_tick_ir(design)
            validate_tick_ir(ir)

            self.assertIsInstance(ir.output_exprs["o_sub"], Sub)
            self.assertIsInstance(ir.output_exprs["o_shl"], Shl)
            self.assertIsInstance(ir.output_exprs["o_lshr"], LShr)
            self.assertIsInstance(ir.output_exprs["o_ashr"], AShr)
            self.assertIsInstance(ir.output_exprs["lt"], Ult)
            self.assertIsInstance(ir.output_exprs["le"], Ule)
            self.assertIsInstance(ir.output_exprs["gt"], Ugt)
            self.assertIsInstance(ir.output_exprs["ge"], Uge)
