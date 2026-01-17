import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stc.backend_x86_sse2 import emit_x86_sse2_c
from stc.interp import eval_expr
from stc.tick_ir import (
    BitVecConst,
    SimdAShr,
    SimdAdd,
    SimdAnd,
    SimdLShr,
    SimdNot,
    SimdOr,
    SimdShl,
    SimdSub,
    SimdType,
    SimdXor,
    TickIR,
    Var,
)
from stc.tick_ir_validate import validate_tick_ir


@unittest.skipUnless(
    shutil.which("cc") and platform.machine() == "x86_64",
    "requires cc and x86_64",
)
class TestBackendX86Sse2Optional(unittest.TestCase):
    def test_emit_and_run_sse2_intrinsics_matches_interpreter(self) -> None:
        t = SimdType(lane_width=32, lanes=4)
        sh1 = BitVecConst(width=32, value=1)

        ir = TickIR(
            name="t",
            inputs={"x": t, "y": t},
            outputs={
                "add": t,
                "sub": t,
                "and_": t,
                "andnot": t,
                "or_": t,
                "xor_": t,
                "notx": t,
                "shl1": t,
                "lshr1": t,
                "ashr1": t,
            },
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "add": SimdAdd(a=Var("x"), b=Var("y")),
                "sub": SimdSub(a=Var("x"), b=Var("y")),
                "and_": SimdAnd(a=Var("x"), b=Var("y")),
                "andnot": SimdAnd(a=SimdNot(x=Var("x")), b=Var("y")),
                "or_": SimdOr(a=Var("x"), b=Var("y")),
                "xor_": SimdXor(a=Var("x"), b=Var("y")),
                "notx": SimdNot(x=Var("x")),
                "shl1": SimdShl(a=Var("x"), sh=sh1),
                "lshr1": SimdLShr(a=Var("x"), sh=sh1),
                "ashr1": SimdAShr(a=Var("x"), sh=sh1),
            },
        )
        validate_tick_ir(ir)
        c = emit_x86_sse2_c(ir)
        self.assertIn("_mm_andnot_si128", c)

        cases = [
            (0x00000001000000020000000300000004, 0x00000001000000010000000100000001),
            (0x80000000800000008000000080000000, 0x00000001000000010000000100000001),
            (0xFFFFFFFF00000000FFFFFFFF00000000, 0x00000000FFFFFFFF00000000FFFFFFFF),
        ]

        with tempfile.TemporaryDirectory() as d:
            out_words = len(sorted(ir.outputs.keys())) * 2
            out_dir = Path(d)
            impl_c = out_dir / "impl.c"
            impl_c.write_text(c, encoding="utf-8")
            main_c = out_dir / "main.c"
            main_c.write_text(
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
                    str(main_c),
                    str(impl_c),
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

            types = {"x": t, "y": t}
            out_order = sorted(ir.outputs.keys())

            for (x, y), ln in zip(cases, lines, strict=True):
                words = [int(w, 16) for w in ln.split()]
                self.assertEqual(len(words), out_words)
                env = {"x": x, "y": y}
                for idx, name in enumerate(out_order):
                    got = words[idx * 2] | (words[idx * 2 + 1] << 64)
                    exp = int(eval_expr(ir.output_exprs[name], types, env))
                    self.assertEqual(got, exp)
