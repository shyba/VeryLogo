import shutil
import tempfile
import unittest
from pathlib import Path

from stc.extract import extract_tick_ir
from stc.tick_ir import Mux, Var
from stc.tick_ir import BitVecConst
from stc.tick_ir_validate import validate_tick_ir
from stc.yosys_frontend import run_yosys
from stc.yosys_json import load_design


@unittest.skipUnless(shutil.which("yosys"), "requires yosys")
class TestExtractDffeOptional(unittest.TestCase):
    def test_extract_dffe_as_mux_next_state(self) -> None:
        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog" / "dffe.v"

        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            normalized = out_dir / "normalized.json"
            run_yosys(src, normalized, top="top")
            design = load_design(normalized, top="top")
            ir = extract_tick_ir(design)
            validate_tick_ir(ir)

            self.assertEqual(len(ir.state), 1)
            reg = next(iter(ir.state.keys()))
            self.assertEqual(ir.reset_state[reg], BitVecConst(width=4, value=0))
            nxt = ir.next_state[reg]
            self.assertIsInstance(nxt, Mux)
            assert isinstance(nxt, Mux)
            self.assertEqual(nxt.cond, Var("en"))
            self.assertEqual(nxt.b, Var(reg))
