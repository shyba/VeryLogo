import os
import shutil
import tempfile
import unittest
from pathlib import Path

from stc.extract import extract_tick_ir
from stc.interp import reset_state, tick
from stc.yosys_frontend import run_yosys
from stc.yosys_json import load_design


def _can_run_yosys() -> bool:
    return shutil.which("yosys") is not None


def _can_run_aes() -> bool:
    return _can_run_yosys() and os.environ.get("STC_RUN_AES_TESTS") == "1"


@unittest.skipUnless(_can_run_aes(), "requires yosys and STC_RUN_AES_TESTS=1")
class TestVerilogAes128FixedKeyOptional(unittest.TestCase):
    def test_known_vector(self) -> None:
        v = Path("fixtures/verilog/aes128_fixedkey_comb.v")
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "aes.json"
            run_yosys(v, out, top="aes128_fixedkey_comb")
            design = load_design(out, top="aes128_fixedkey_comb")
        ir = extract_tick_ir(design)
        state = reset_state(ir)
        pt = int("00112233445566778899aabbccddeeff", 16)
        exp = int("69c4e0d86a7b0430d8cdb78070b4c55a", 16)
        _, outputs = tick(ir, state, {"pt": pt})
        self.assertEqual(int(outputs["ct"]) & ((1 << 128) - 1), exp)
