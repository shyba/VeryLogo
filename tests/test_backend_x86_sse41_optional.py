import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stc.backend_x86_sse2 import emit_x86_sse2_c
from stc.interp import eval_expr
from stc.tick_ir import (
    SimdBlend,
    SimdMaxS,
    SimdMaxU,
    SimdMinS,
    SimdMinU,
    SimdSExtLo,
    SimdType,
    SimdUlt,
    SimdZExtLo,
    TickIR,
    Var,
)
from stc.tick_ir_validate import validate_tick_ir


def _has_cpu_flag(flag: str) -> bool:
    try:
        txt = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    for line in txt.splitlines():
        if line.startswith("flags") and flag in line.split():
            return True
    return False


@unittest.skipUnless(
    shutil.which("cc") and platform.machine() == "x86_64" and _has_cpu_flag("sse4_1"),
    "requires cc, x86_64, and sse4_1",
)
class TestBackendX86Sse41Optional(unittest.TestCase):
    def test_min_max_blend_and_extend_matches_interpreter(self) -> None:
        v8 = SimdType(lane_width=8, lanes=16)
        v16 = SimdType(lane_width=16, lanes=8)
        m16 = SimdType(lane_width=1, lanes=16)
        ir = TickIR(
            name="sse41",
            inputs={"x": v8, "y": v8},
            outputs={
                "minu": v8,
                "maxu": v8,
                "mins": v8,
                "maxs": v8,
                "zext": v16,
                "sext": v16,
                "blend": v8,
                "mask": m16,
            },
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "minu": SimdMinU(a=Var("x"), b=Var("y")),
                "maxu": SimdMaxU(a=Var("x"), b=Var("y")),
                "mins": SimdMinS(a=Var("x"), b=Var("y")),
                "maxs": SimdMaxS(a=Var("x"), b=Var("y")),
                "zext": SimdZExtLo(to=v16, x=Var("x")),
                "sext": SimdSExtLo(to=v16, x=Var("x")),
                "blend": SimdBlend(
                    mask=SimdUlt(a=Var("x"), b=Var("y")),
                    a=Var("x"),
                    b=Var("y"),
                ),
                "mask": SimdUlt(a=Var("x"), b=Var("y")),
            },
        )
        validate_tick_ir(ir)
        c = emit_x86_sse2_c(ir)
        self.assertIn("_mm_blendv_epi8", c)
        self.assertIn("_mm_min_epu8", c)
        self.assertIn("_mm_max_epi8", c)
        self.assertIn("_mm_cvtepu8_epi16", c)

        cases = [
            (
                0x00_7F_80_FF_01_02_03_04_10_F0_20_30_40_C0_90_08,
                0x01_7E_81_00_FF_02_04_03_11_EF_21_2F_41_BF_91_07,
            ),
            (
                0x000102030405060708090A0B0C0D0E0F,
                0x0F0E0D0C0B0A09080706050403020100,
            ),
        ]

        out_order = sorted(ir.outputs.keys())
        out_words = len(out_order) * 2

        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            (out_dir / "impl.c").write_text(c, encoding="utf-8")
            (out_dir / "main.c").write_text(
                "\n".join(
                    [
                        "#include <stdint.h>",
                        "#include <stdio.h>",
                        "",
                        "void stc_eval(const uint64_t* in, uint64_t* out);",
                        "",
                        "int main(void) {",
                        "  unsigned long long xlo, xhi, ylo, yhi;",
                        '  while (scanf("%llx %llx %llx %llx", &xlo, &xhi, &ylo, &yhi) == 4) {',
                        "    uint64_t in[4];",
                        f"    uint64_t out[{out_words}];",
                        "    in[0] = (uint64_t)xlo;",
                        "    in[1] = (uint64_t)xhi;",
                        "    in[2] = (uint64_t)ylo;",
                        "    in[3] = (uint64_t)yhi;",
                        "    stc_eval(in, out);",
                        f"    for (int i = 0; i < {out_words}; i++) {{",
                        f'      printf("%016llx%s", (unsigned long long)out[i], (i == {out_words - 1}) ? "\\n" : " ");',
                        "    }",
                        "  }",
                        "  return 0;",
                        "}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            exe = out_dir / "runner"
            subprocess.run(
                [
                    "cc",
                    "-std=c99",
                    "-O2",
                    "-msse4.1",
                    "-o",
                    str(exe),
                    str(out_dir / "main.c"),
                    str(out_dir / "impl.c"),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            stdin = ""
            for x, y in cases:
                stdin += (
                    f"{x & ((1<<64)-1):x} {x >> 64:x} {y & ((1<<64)-1):x} {y >> 64:x}\n"
                )
            p = subprocess.run(
                [str(exe)],
                input=stdin.encode("utf-8"),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            lines = [
                ln.strip() for ln in p.stdout.decode("utf-8").splitlines() if ln.strip()
            ]
            self.assertEqual(len(lines), len(cases))

            types = {"x": v8, "y": v8}
            for (x, y), ln in zip(cases, lines, strict=True):
                ws = [int(w, 16) for w in ln.split()]
                self.assertEqual(len(ws), out_words)
                env = {"x": x, "y": y}
                for idx, name in enumerate(out_order):
                    got = ws[idx * 2] | (ws[idx * 2 + 1] << 64)
                    exp = int(eval_expr(ir.output_exprs[name], types, env))
                    self.assertEqual(got, exp)
