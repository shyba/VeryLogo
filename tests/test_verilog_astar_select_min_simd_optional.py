import platform
import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stc.backend_x86_sse2 import emit_x86_sse2_c
from stc.extract import extract_tick_ir
from stc.infer_simd import infer_simd_types
from stc.reduce import optimize_tick_ir
from stc.tick_ir import SimdMinU, SimdType, TickIR, Var
from stc.tick_ir_validate import validate_tick_ir
from stc.yosys_frontend import run_yosys
from stc.yosys_json import load_design


def _has_cpu_flag(flag: str) -> bool:
    try:
        txt = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    for line in txt.splitlines():
        if line.startswith("flags") and flag in line.split():
            return True
    return False


def _load_ir_from_verilog(src: Path, *, top: str) -> TickIR:
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d)
        normalized = out_dir / "normalized.json"
        run_yosys(src, normalized, top=top)
        design = load_design(normalized, top=top)
        ir0 = extract_tick_ir(design)
        validate_tick_ir(ir0)
        return ir0


@unittest.skipUnless(
    shutil.which("yosys")
    and shutil.which("cc")
    and platform.machine() == "x86_64"
    and _has_cpu_flag("sse4_1"),
    "requires yosys, cc, x86_64, and sse4_1",
)
class TestVerilogAstarSelectMinSimdOptional(unittest.TestCase):
    def test_verilog_to_superopt_simd_minu(self) -> None:
        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog" / "astar_select_min_simd_slices.v"

        ir0 = _load_ir_from_verilog(src, top="top")
        ir1 = infer_simd_types(ir0)
        validate_tick_ir(ir1)
        self.assertEqual(ir1.inputs["a"], SimdType(lane_width=16, lanes=8))
        self.assertEqual(ir1.inputs["b"], SimdType(lane_width=16, lanes=8))
        self.assertEqual(ir1.outputs["o"], SimdType(lane_width=16, lanes=8))

        ir2, _ = optimize_tick_ir(
            ir1,
            bound=1,
            autovec=False,
            superopt=True,
            superopt_max_nodes=3,
            superopt_timeout_ms=200,
        )
        validate_tick_ir(ir2)
        self.assertEqual(ir2.output_exprs["o"], SimdMinU(a=Var("a"), b=Var("b")))

    def test_sse41_matches_scalar_c_baseline(self) -> None:
        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog" / "astar_select_min_simd_slices.v"

        ir0 = _load_ir_from_verilog(src, top="top")
        ir1 = infer_simd_types(ir0)
        validate_tick_ir(ir1)
        ir2, _ = optimize_tick_ir(
            ir1,
            bound=1,
            autovec=False,
            superopt=True,
            superopt_max_nodes=3,
            superopt_timeout_ms=200,
        )
        validate_tick_ir(ir2)
        c = emit_x86_sse2_c(ir2)

        runner = "\n".join(
            [
                "#include <stdint.h>",
                "#include <stdio.h>",
                "",
                "void stc_eval(const uint64_t* in, uint64_t* out);",
                "",
                "static void baseline(const uint64_t* in, uint64_t* out) {",
                "  uint64_t alo = in[0];",
                "  uint64_t ahi = in[1];",
                "  uint64_t blo = in[2];",
                "  uint64_t bhi = in[3];",
                "  uint64_t out_lo = 0;",
                "  uint64_t out_hi = 0;",
                "  for (int lane = 0; lane < 8; lane++) {",
                "    uint16_t a;",
                "    uint16_t b;",
                "    if (lane < 4) {",
                "      a = (uint16_t)((alo >> (16 * lane)) & 0xffffu);",
                "      b = (uint16_t)((blo >> (16 * lane)) & 0xffffu);",
                "    } else {",
                "      int l = lane - 4;",
                "      a = (uint16_t)((ahi >> (16 * l)) & 0xffffu);",
                "      b = (uint16_t)((bhi >> (16 * l)) & 0xffffu);",
                "    }",
                "    uint16_t o = (a < b) ? a : b;",
                "    if (lane < 4) {",
                "      out_lo |= ((uint64_t)o) << (16 * lane);",
                "    } else {",
                "      int l = lane - 4;",
                "      out_hi |= ((uint64_t)o) << (16 * l);",
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
                "  }",
                "  return 0;",
                "}",
                "",
            ]
        )

        rng = random.Random(0)
        cases: list[tuple[int, int]] = []
        for _ in range(200):
            a = rng.getrandbits(128)
            b = rng.getrandbits(128)
            cases.append((a, b))

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
                    "-msse4.1",
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
            for a, b in cases:
                stdin += (
                    f"{a & ((1<<64)-1):x} {a >> 64:x} {b & ((1<<64)-1):x} {b >> 64:x}\n"
                )
            subprocess.run(
                [str(exe)],
                input=stdin.encode("utf-8"),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
