import json
import shutil
import tempfile
import unittest
from pathlib import Path

from stc.cli import run_pipeline
from stc.tick_ir import SimdAdd, SimdType, TickIR


@unittest.skipUnless(shutil.which("yosys"), "requires yosys")
class TestCliAutovecOptional(unittest.TestCase):
    def test_cli_pipeline_infer_simd_autovec_no_backend(self) -> None:
        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog" / "simd_lane_add_slices.v"
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d) / "out"
            run_pipeline(
                src,
                out_dir,
                top="top",
                bound=2,
                infer_simd=True,
                autovec=True,
                no_backend=True,
            )
            self.assertTrue((out_dir / "normalized.ys").exists())
            reduced = TickIR.from_dict(
                json.loads(
                    (out_dir / "reduced_tick_ir.json").read_text(encoding="utf-8")
                )
            )
            self.assertEqual(reduced.outputs["o"], SimdType(lane_width=4, lanes=2))
            self.assertIsInstance(reduced.output_exprs["o"], SimdAdd)
