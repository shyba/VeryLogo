import shutil
import tempfile
import unittest
from pathlib import Path

from stc.extract import extract_tick_ir
from stc.subset import SubsetError
from stc.yosys_frontend import run_yosys
from stc.yosys_json import load_design


@unittest.skipUnless(shutil.which("yosys"), "requires yosys")
class TestSubsetHierarchyOptional(unittest.TestCase):
    def test_submodule_instantiation_is_rejected_explicitly(self) -> None:
        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog" / "hierarchy_submodule.v"

        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            normalized = out_dir / "normalized.json"
            run_yosys(src, normalized, top="top")
            design = load_design(normalized, top="top")
            with self.assertRaises(SubsetError) as ctx:
                extract_tick_ir(design)
            self.assertIn("submodule", str(ctx.exception))
