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
from stc.tick_ir import SimdBlend, SimdEq, SimdType, TickIR, Var
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
class TestVerilogAstarBlendEqSimdOptional(unittest.TestCase):
    def test_verilog_to_superopt_simd_blend(self) -> None:
        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog" / "astar_blend_eq_simd_slices.v"

        ir0 = _load_ir_from_verilog(src, top="top")
        ir1 = infer_simd_types(ir0)
        validate_tick_ir(ir1)
        self.assertEqual(ir1.inputs["a"], SimdType(lane_width=16, lanes=8))
        self.assertEqual(ir1.inputs["b"], SimdType(lane_width=16, lanes=8))
        self.assertEqual(ir1.inputs["x"], SimdType(lane_width=16, lanes=8))
        self.assertEqual(ir1.inputs["y"], SimdType(lane_width=16, lanes=8))
        self.assertEqual(ir1.outputs["o"], SimdType(lane_width=16, lanes=8))

        ir2 = optimize_tick_ir(
            ir1,
            bound=1,
            autovec=False,
            superopt=True,
            superopt_max_nodes=6,
            superopt_timeout_ms=200,
        )
        validate_tick_ir(ir2)
        self.assertEqual(
            ir2.output_exprs["o"],
            SimdBlend(
                mask=SimdEq(a=Var("a"), b=Var("b")),
                a=Var("y"),
                b=Var("x"),
            ),
        )

    def test_sse41_matches_scalar_c_baseline(self) -> None:
        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog" / "astar_blend_eq_simd_slices.v"

        ir0 = _load_ir_from_verilog(src, top="top")
        ir1 = infer_simd_types(ir0)
        validate_tick_ir(ir1)
        ir2 = optimize_tick_ir(
            ir1,
            bound=1,
            autovec=False,
            superopt=True,
            superopt_max_nodes=6,
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
                "static uint16_t lane_u16(uint64_t lo, uint64_t hi, int lane) {",
                "  if (lane < 4) {",
                "    return (uint16_t)((lo >> (16 * lane)) & 0xffffu);",
                "  }",
                "  int l = lane - 4;",
                "  return (uint16_t)((hi >> (16 * l)) & 0xffffu);",
                "}",
                "",
                "static void baseline(const uint64_t* in, uint64_t* out) {",
                "  uint64_t alo = in[0];",
                "  uint64_t ahi = in[1];",
                "  uint64_t blo = in[2];",
                "  uint64_t bhi = in[3];",
                "  uint64_t xlo = in[4];",
                "  uint64_t xhi = in[5];",
                "  uint64_t ylo = in[6];",
                "  uint64_t yhi = in[7];",
                "  uint64_t out_lo = 0;",
                "  uint64_t out_hi = 0;",
                "  for (int lane = 0; lane < 8; lane++) {",
                "    uint16_t a = lane_u16(alo, ahi, lane);",
                "    uint16_t b = lane_u16(blo, bhi, lane);",
                "    uint16_t x = lane_u16(xlo, xhi, lane);",
                "    uint16_t y = lane_u16(ylo, yhi, lane);",
                "    uint16_t o = (a == b) ? x : y;",
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
                "  unsigned long long w[8];",
                '  while (scanf("%llx %llx %llx %llx %llx %llx %llx %llx",',
                "               &w[0], &w[1], &w[2], &w[3], &w[4], &w[5], &w[6], &w[7]) == 8) {",
                "    uint64_t in[8];",
                "    uint64_t out_stc[2];",
                "    uint64_t out_ref[2];",
                "    for (int i = 0; i < 8; i++) {",
                "      in[i] = (uint64_t)w[i];",
                "    }",
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
        cases: list[tuple[int, int, int, int]] = []
        for _ in range(200):
            a = rng.getrandbits(128)
            b = rng.getrandbits(128)
            x = rng.getrandbits(128)
            y = rng.getrandbits(128)
            cases.append((a, b, x, y))

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
            for a, b, x, y in cases:
                stdin += (
                    f"{a & ((1<<64)-1):x} {a >> 64:x} "
                    f"{b & ((1<<64)-1):x} {b >> 64:x} "
                    f"{x & ((1<<64)-1):x} {x >> 64:x} "
                    f"{y & ((1<<64)-1):x} {y >> 64:x}\n"
                )
            subprocess.run(
                [str(exe)],
                input=stdin.encode("utf-8"),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
