import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stc.backend_x86_avx512vl import emit_x86_avx512vl_c
from stc.interp import eval_expr
from stc.tick_ir import (
    SimdAddMasked,
    SimdBlend,
    SimdEq,
    SimdSubMasked,
    SimdType,
    SimdUlt,
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
    shutil.which("cc")
    and platform.machine() == "x86_64"
    and _has_cpu_flag("avx512f")
    and _has_cpu_flag("avx512vl")
    and _has_cpu_flag("avx2"),
    "requires cc, x86_64, avx512f, avx512vl, and avx2",
)
class TestBackendX86Avx512VlOptional(unittest.TestCase):
    def test_emit_and_run_avx512vl_epi32_matches_interpreter(self) -> None:
        t = SimdType(lane_width=32, lanes=8)
        mask = SimdUlt(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="vl32x8",
            inputs={"x": t, "y": t},
            outputs={
                "addm": t,
                "subm": t,
                "blend": t,
                "eq": SimdType(lane_width=1, lanes=8),
                "ult": SimdType(lane_width=1, lanes=8),
            },
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "addm": SimdAddMasked(
                    mask=mask, passthru=Var("x"), a=Var("x"), b=Var("y")
                ),
                "subm": SimdSubMasked(
                    mask=mask, passthru=Var("x"), a=Var("x"), b=Var("y")
                ),
                "blend": SimdBlend(mask=mask, a=Var("x"), b=Var("y")),
                "eq": SimdEq(a=Var("x"), b=Var("y")),
                "ult": mask,
            },
        )
        validate_tick_ir(ir)
        c = emit_x86_avx512vl_c(ir)

        cases = [
            (
                int(
                    "0000000100000002000000030000000400000005000000060000000700000008",
                    16,
                ),
                int(
                    "0000000100000001000000010000000100000001000000010000000100000001",
                    16,
                ),
            ),
            (
                int(
                    "8000000080000000800000008000000080000000800000008000000080000000",
                    16,
                ),
                int(
                    "0000000100000001000000010000000100000001000000010000000100000001",
                    16,
                ),
            ),
        ]

        out_order = sorted(ir.outputs.keys())
        out_words = len(out_order) * 4

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
                        "  unsigned long long x0,x1,x2,x3,y0,y1,y2,y3;",
                        '  while (scanf("%llx %llx %llx %llx %llx %llx %llx %llx", &x0,&x1,&x2,&x3,&y0,&y1,&y2,&y3) == 8) {',
                        "    uint64_t in[8];",
                        f"    uint64_t out[{out_words}];",
                        "    in[0]=(uint64_t)x0; in[1]=(uint64_t)x1; in[2]=(uint64_t)x2; in[3]=(uint64_t)x3;",
                        "    in[4]=(uint64_t)y0; in[5]=(uint64_t)y1; in[6]=(uint64_t)y2; in[7]=(uint64_t)y3;",
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
                    "-mavx2",
                    "-mavx512f",
                    "-mavx512vl",
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
                xs = [(x >> (64 * i)) & ((1 << 64) - 1) for i in range(4)]
                ys = [(y >> (64 * i)) & ((1 << 64) - 1) for i in range(4)]
                stdin += " ".join(f"{w:x}" for w in xs + ys) + "\n"

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

            types = {"x": t, "y": t}
            for (x, y), ln in zip(cases, lines, strict=True):
                ws = [int(w, 16) for w in ln.split()]
                self.assertEqual(len(ws), out_words)
                env = {"x": x, "y": y}
                for idx, name in enumerate(out_order):
                    if name in {"eq", "ult"}:
                        got = ws[idx * 4 + 0]
                        exp = int(eval_expr(ir.output_exprs[name], types, env))
                        self.assertEqual(got, exp)
                        for i in range(1, 4):
                            self.assertEqual(ws[idx * 4 + i], 0)
                    else:
                        got = 0
                        for widx in range(4):
                            got |= ws[idx * 4 + widx] << (64 * widx)
                        exp = int(eval_expr(ir.output_exprs[name], types, env))
                        self.assertEqual(got, exp)
