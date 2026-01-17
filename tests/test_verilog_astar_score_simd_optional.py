import platform
import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stc.autovec_pass import autovectorize_tick_ir
from stc.backend_x86_sse2 import emit_x86_sse2_c
from stc.extract import extract_tick_ir
from stc.infer_simd import infer_simd_types
from stc.tick_ir import SimdAdd, SimdType, TickIR, Var
from stc.tick_ir_validate import validate_tick_ir
from stc.yosys_frontend import run_yosys
from stc.yosys_json import load_design


def _load_ir_from_verilog(src: Path, *, top: str) -> TickIR:
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d)
        normalized = out_dir / "normalized.json"
        run_yosys(src, normalized, top=top)
        design = load_design(normalized, top=top)
        ir0 = extract_tick_ir(design)
        validate_tick_ir(ir0)
        ir1 = infer_simd_types(ir0)
        validate_tick_ir(ir1)
        return autovectorize_tick_ir(ir1, timeout_ms=200)


@unittest.skipUnless(
    shutil.which("yosys") and shutil.which("cc") and platform.machine() == "x86_64",
    "requires yosys, cc, x86_64",
)
class TestVerilogAstarScoreSimdOptional(unittest.TestCase):
    def test_astar_score_autovec_to_simd_add(self) -> None:
        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog" / "astar_score_simd_slices.v"

        ir = _load_ir_from_verilog(src, top="top")
        validate_tick_ir(ir)

        self.assertEqual(ir.inputs["g"], SimdType(lane_width=16, lanes=8))
        self.assertEqual(ir.inputs["h"], SimdType(lane_width=16, lanes=8))
        self.assertEqual(ir.outputs["f"], SimdType(lane_width=16, lanes=8))
        self.assertEqual(ir.output_exprs["f"], SimdAdd(a=Var("g"), b=Var("h")))

    def test_astar_score_sse2_matches_raw_c(self) -> None:
        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog" / "astar_score_simd_slices.v"

        ir = _load_ir_from_verilog(src, top="top")
        validate_tick_ir(ir)
        c = emit_x86_sse2_c(ir)

        runner = "\n".join(
            [
                "#include <stdint.h>",
                "#include <stdio.h>",
                "",
                "void stc_eval(const uint64_t* in, uint64_t* out);",
                "",
                "static void baseline(const uint64_t* in, uint64_t* out) {",
                "  uint64_t glo = in[0];",
                "  uint64_t ghi = in[1];",
                "  uint64_t hlo = in[2];",
                "  uint64_t hhi = in[3];",
                "  uint64_t out_lo = 0;",
                "  uint64_t out_hi = 0;",
                "  for (int lane = 0; lane < 8; lane++) {",
                "    uint16_t g;",
                "    uint16_t h;",
                "    if (lane < 4) {",
                "      g = (uint16_t)((glo >> (16 * lane)) & 0xffffu);",
                "      h = (uint16_t)((hlo >> (16 * lane)) & 0xffffu);",
                "    } else {",
                "      int l = lane - 4;",
                "      g = (uint16_t)((ghi >> (16 * l)) & 0xffffu);",
                "      h = (uint16_t)((hhi >> (16 * l)) & 0xffffu);",
                "    }",
                "    uint16_t f = (uint16_t)(g + h);",
                "    if (lane < 4) {",
                "      out_lo |= ((uint64_t)f) << (16 * lane);",
                "    } else {",
                "      int l = lane - 4;",
                "      out_hi |= ((uint64_t)f) << (16 * l);",
                "    }",
                "  }",
                "  out[0] = out_lo;",
                "  out[1] = out_hi;",
                "}",
                "",
                "int main(void) {",
                "  unsigned long long w0, w1, w2, w3;",
                '  while (scanf("%llx %llx %llx %llx", &w0, &w1, &w2, &w3) == 4) {',
                "    uint64_t in[4];",
                "    uint64_t out_stc[2];",
                "    uint64_t out_ref[2];",
                "    in[0] = (uint64_t)w0;",
                "    in[1] = (uint64_t)w1;",
                "    in[2] = (uint64_t)w2;",
                "    in[3] = (uint64_t)w3;",
                "    stc_eval(in, out_stc);",
                "    baseline(in, out_ref);",
                "    if (out_stc[0] != out_ref[0] || out_stc[1] != out_ref[1]) {",
                '      fprintf(stderr, "mismatch\\n");',
                "      return 1;",
                "    }",
                '    printf("%016llx %016llx\\n",',
                "           (unsigned long long)out_stc[0],",
                "           (unsigned long long)out_stc[1]);",
                "  }",
                "  return 0;",
                "}",
                "",
            ]
        )

        rng = random.Random(0)
        cases: list[tuple[int, int]] = []
        for _ in range(200):
            g = rng.getrandbits(128)
            h = rng.getrandbits(128)
            cases.append((g, h))

        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            impl_c = out_dir / "impl.c"
            impl_c.write_text(c, encoding="utf-8")
            main_c = out_dir / "main.c"
            main_c.write_text(runner, encoding="utf-8")
            exe = out_dir / "runner"
            subprocess.run(
                [
                    "cc",
                    "-std=c99",
                    "-O2",
                    "-msse2",
                    "-o",
                    str(exe),
                    str(main_c),
                    str(impl_c),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            stdin = ""
            for g, h in cases:
                stdin += (
                    f"{g & ((1<<64)-1):x} {g >> 64:x} {h & ((1<<64)-1):x} {h >> 64:x}\n"
                )
            subprocess.run(
                [str(exe)],
                input=stdin.encode("utf-8"),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
