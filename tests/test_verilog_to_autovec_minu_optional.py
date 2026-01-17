import shutil
import tempfile
import unittest
from pathlib import Path

from stc.autovec_pass import autovectorize_tick_ir
from stc.extract import extract_tick_ir
from stc.infer_simd import infer_simd_types
from stc.tick_ir import SimdMinU, SimdType, Var
from stc.tick_ir_validate import validate_tick_ir
from stc.yosys_frontend import run_yosys
from stc.yosys_json import load_design


@unittest.skipUnless(shutil.which("yosys"), "requires yosys")
class TestVerilogToAutovecMinUOptional(unittest.TestCase):
    def test_lane_minu_autovec(self) -> None:
        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog" / "astar_select_min_simd_slices.v"

        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            normalized = out_dir / "normalized.json"
            run_yosys(src, normalized, top="top")
            design = load_design(normalized, top="top")
            ir0 = extract_tick_ir(design)
            validate_tick_ir(ir0)

            ir1 = infer_simd_types(ir0)
            validate_tick_ir(ir1)

            ir2 = autovectorize_tick_ir(ir1, timeout_ms=200)
            validate_tick_ir(ir2)

            self.assertEqual(ir2.inputs["a"], SimdType(lane_width=16, lanes=8))
            self.assertEqual(ir2.inputs["b"], SimdType(lane_width=16, lanes=8))
            self.assertEqual(ir2.outputs["o"], SimdType(lane_width=16, lanes=8))
            self.assertEqual(ir2.output_exprs["o"], SimdMinU(a=Var("a"), b=Var("b")))
