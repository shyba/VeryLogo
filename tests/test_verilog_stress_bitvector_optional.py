import random
import shutil
import tempfile
import unittest
from pathlib import Path

from stc.extract import extract_tick_ir
from stc.interp import eval_expr
from stc.reduce import optimize_tick_ir
from stc.tick_ir_validate import validate_tick_ir
from stc.yosys_frontend import run_yosys
from stc.yosys_json import load_design


def _compile_to_ir(src: Path, top: str, out_dir: Path):
    normalized = out_dir / "normalized.json"
    run_yosys(src, normalized, top=top)
    design = load_design(normalized, top=top)
    ir0 = extract_tick_ir(design)
    validate_tick_ir(ir0)
    ir1, _ = optimize_tick_ir(ir0, bound=1, autovec=False, superopt=False)
    validate_tick_ir(ir1)
    return ir1


def _eval_combinational(ir, inputs):
    ctx_types = {**ir.inputs, **ir.state}
    outputs = {}
    for name, expr in ir.output_exprs.items():
        outputs[name] = eval_expr(expr, ctx_types, inputs)
    return outputs


@unittest.skipUnless(
    shutil.which("yosys"),
    "requires yosys",
)
class TestVerilogStressBitvectorOptional(unittest.TestCase):
    def test_rotmix(self) -> None:
        from tests.fixtures_ref.bitvector_ref import rotmix

        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog_stress" / "bitvector__rotmix.v"

        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            ir = _compile_to_ir(src, "rotmix", out_dir)

            rng = random.Random(0)
            for _ in range(30):
                x = rng.getrandbits(64)
                expected = rotmix(x)

                outputs = _eval_combinational(ir, {"x": x})
                actual = outputs["y"]

                self.assertEqual(actual, expected)

    def test_slice_concat(self) -> None:
        from tests.fixtures_ref.bitvector_ref import slice_concat

        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog_stress" / "bitvector__slice_concat_patho.v"

        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            ir = _compile_to_ir(src, "slice_concat_patho", out_dir)

            rng = random.Random(0)
            for _ in range(30):
                x = rng.getrandbits(64)
                expected = slice_concat(x)

                outputs = _eval_combinational(ir, {"x": x})
                actual = outputs["y"]

                self.assertEqual(actual, expected)
