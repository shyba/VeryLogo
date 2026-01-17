import shutil
import tempfile
import unittest
from pathlib import Path

from stc.autovec_pass import autovectorize_tick_ir
from stc.extract import extract_tick_ir
from stc.infer_simd import infer_simd_types
from stc.tick_ir import SimdType, SimdUlt, Var
from stc.tick_ir_validate import validate_tick_ir
from stc.yosys_frontend import run_yosys
from stc.yosys_json import load_design


@unittest.skipUnless(shutil.which("yosys"), "requires yosys")
class TestVerilogToSimdUltOptional(unittest.TestCase):
    def test_verilog_to_tick_ir_to_simd_ult(self) -> None:
        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog" / "simd_lane_ult_mask.v"

        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            normalized = out_dir / "normalized.json"
            run_yosys(src, normalized, top="top")
            design = load_design(normalized, top="top")

            ir = extract_tick_ir(design)
            validate_tick_ir(ir)

            simd = infer_simd_types(ir)
            validate_tick_ir(simd)
            out = autovectorize_tick_ir(simd, timeout_ms=200)
            validate_tick_ir(out)

            self.assertEqual(out.inputs["x"], SimdType(lane_width=4, lanes=2))
            self.assertEqual(out.inputs["y"], SimdType(lane_width=4, lanes=2))
            self.assertEqual(out.outputs["m"], SimdType(lane_width=1, lanes=2))
            self.assertEqual(out.output_exprs["m"], SimdUlt(a=Var("x"), b=Var("y")))
