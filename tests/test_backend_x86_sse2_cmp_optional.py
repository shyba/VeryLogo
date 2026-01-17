import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stc.backend_x86_sse2 import emit_x86_sse2_c
from stc.interp import eval_expr
from stc.tick_ir import SimdEq, SimdMaskPack, SimdSlt, SimdType, SimdUlt, TickIR, Var
from stc.tick_ir_validate import validate_tick_ir


@unittest.skipUnless(
    shutil.which("cc") and platform.machine() == "x86_64",
    "requires cc and x86_64",
)
class TestBackendX86Sse2CmpOptional(unittest.TestCase):
    def _run_ir(self, ir: TickIR, cases: list[tuple[int, int]]) -> list[list[int]]:
        validate_tick_ir(ir)
        c = emit_x86_sse2_c(ir)
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
                    "-msse2",
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
            return [[int(w, 16) for w in ln.split()] for ln in lines]

    def test_cmp_eq_slt_ult_epi32_mask_pack(self) -> None:
        v32 = SimdType(lane_width=32, lanes=4)
        m4 = SimdType(lane_width=1, lanes=4)
        ir = TickIR(
            name="cmp32",
            inputs={"x": v32, "y": v32},
            outputs={"eq": m4, "slt": m4, "ult": m4},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "eq": SimdEq(a=Var("x"), b=Var("y")),
                "slt": SimdSlt(a=Var("x"), b=Var("y")),
                "ult": SimdUlt(a=Var("x"), b=Var("y")),
            },
        )
        validate_tick_ir(ir)
        c = emit_x86_sse2_c(ir)
        self.assertIn("_mm_cmpeq_epi32", c)
        self.assertIn("_mm_cmpgt_epi32", c)
        self.assertIn("_mm_movemask_epi8", c)

        x1 = 0x00000001_00000002_00000003_00000004
        y1 = 0x00000001_00000003_00000002_00000004
        x2 = 0xFFFFFFFF_00000000_80000000_7FFFFFFF
        y2 = 0x00000000_00000000_7FFFFFFF_80000000
        cases = [(x1, y1), (x2, y2)]

        out_lines = self._run_ir(ir, cases)
        types = {"x": v32, "y": v32}
        out_order = sorted(ir.outputs.keys())

        for (x, y), ws in zip(cases, out_lines, strict=True):
            self.assertEqual(len(ws), len(out_order) * 2)
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                got = ws[idx * 2]
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)

    def test_cmp_eq_ult_epi8_mask_pack(self) -> None:
        v8 = SimdType(lane_width=8, lanes=16)
        m16 = SimdType(lane_width=1, lanes=16)
        ir = TickIR(
            name="cmp8",
            inputs={"x": v8, "y": v8},
            outputs={"eq": m16, "ult": m16},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "eq": SimdEq(a=Var("x"), b=Var("y")),
                "ult": SimdUlt(a=Var("x"), b=Var("y")),
            },
        )
        validate_tick_ir(ir)
        c = emit_x86_sse2_c(ir)
        self.assertIn("_mm_cmpeq_epi8", c)
        self.assertIn("_mm_cmpgt_epi8", c)

        x = 0x000102030405060708090A0B0C0D0E0F
        y = 0x0001020304050607FFFFFFFFFFFFFFFF
        out_lines = self._run_ir(ir, [(x, y)])
        ws = out_lines[0]
        types = {"x": v8, "y": v8}
        env = {"x": x, "y": y}
        out_order = sorted(ir.outputs.keys())
        for idx, name in enumerate(out_order):
            got = ws[idx * 2]
            exp = int(eval_expr(ir.output_exprs[name], types, env))
            self.assertEqual(got, exp)

    def test_mask_pack_from_nonzero_epi32(self) -> None:
        v32 = SimdType(lane_width=32, lanes=4)
        m4 = SimdType(lane_width=1, lanes=4)
        ir = TickIR(
            name="maskpack32",
            inputs={"x": v32, "y": v32},
            outputs={"nzx": m4},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"nzx": SimdMaskPack(x=Var("x"))},
        )
        validate_tick_ir(ir)
        c = emit_x86_sse2_c(ir)
        self.assertIn("_mm_movemask_epi8", c)

        cases = [
            (0x00000000_00000001_00000000_00000002, 0),
            (0xFFFFFFFF_FFFFFFFF_00000000_00000000, 0),
            (0x00000000_00000000_00000000_00000000, 0),
        ]
        out_lines = self._run_ir(ir, cases)
        types = {"x": v32, "y": v32}
        for (x, y), ws in zip(cases, out_lines, strict=True):
            got = ws[0]
            exp = int(eval_expr(ir.output_exprs["nzx"], types, {"x": x, "y": y}))
            self.assertEqual(got, exp)
