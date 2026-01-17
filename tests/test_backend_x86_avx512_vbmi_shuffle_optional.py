import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stc.backend_x86_avx512 import emit_x86_avx512_c
from stc.interp import eval_expr
from stc.tick_ir import SimdShuffle, SimdType, TickIR, Var
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
    and _has_cpu_flag("avx512vbmi"),
    "requires cc, x86_64, avx512f, and avx512vbmi",
)
class TestBackendX86Avx512VbmiShuffleOptional(unittest.TestCase):
    def test_shuffle_epi8_full_width_matches_interpreter(self) -> None:
        t = SimdType(lane_width=8, lanes=64)
        rev = list(range(64))[::-1]
        rot1 = [(i + 1) % 64 for i in range(64)]
        ir = TickIR(
            name="vbmi_shuffle",
            inputs={"x": t},
            outputs={"rev": t, "rot1": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "rev": SimdShuffle(x=Var("x"), indices=rev),
                "rot1": SimdShuffle(x=Var("x"), indices=rot1),
            },
        )
        validate_tick_ir(ir)
        c = emit_x86_avx512_c(ir)
        self.assertIn("_mm512_permutexvar_epi8", c)

        x = int("00112233445566778899aabbccddeeff" * 4, 16)
        out_order = sorted(ir.outputs.keys())
        out_words = len(out_order) * 8

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
                        "  unsigned long long x0,x1,x2,x3,x4,x5,x6,x7;",
                        '  while (scanf("%llx %llx %llx %llx %llx %llx %llx %llx", &x0,&x1,&x2,&x3,&x4,&x5,&x6,&x7) == 8) {',
                        "    uint64_t in[8];",
                        f"    uint64_t out[{out_words}];",
                        "    in[0]=(uint64_t)x0; in[1]=(uint64_t)x1; in[2]=(uint64_t)x2; in[3]=(uint64_t)x3;",
                        "    in[4]=(uint64_t)x4; in[5]=(uint64_t)x5; in[6]=(uint64_t)x6; in[7]=(uint64_t)x7;",
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
                    "-mavx512f",
                    "-mavx512vbmi",
                    "-o",
                    str(exe),
                    str(out_dir / "main.c"),
                    str(out_dir / "impl.c"),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            xs = [(x >> (64 * i)) & ((1 << 64) - 1) for i in range(8)]
            stdin = " ".join(f"{w:x}" for w in xs) + "\n"
            p = subprocess.run(
                [str(exe)],
                input=stdin.encode("utf-8"),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            ws = [int(w, 16) for w in p.stdout.decode("utf-8").strip().split()]
            self.assertEqual(len(ws), out_words)

            types = {"x": t}
            env = {"x": x}
            for idx, name in enumerate(out_order):
                got = 0
                for widx in range(8):
                    got |= ws[idx * 8 + widx] << (64 * widx)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)
