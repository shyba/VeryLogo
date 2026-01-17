import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stc.backend_x86_sse2 import emit_x86_sse2_c
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
    shutil.which("cc") and platform.machine() == "x86_64" and _has_cpu_flag("ssse3"),
    "requires cc, x86_64, and ssse3",
)
class TestBackendX86Ssse3ShuffleOptional(unittest.TestCase):
    def test_shuffle_epi8_matches_interpreter(self) -> None:
        t = SimdType(lane_width=8, lanes=16)
        ir = TickIR(
            name="shuf",
            inputs={"x": t},
            outputs={"y": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "y": SimdShuffle(x=Var("x"), indices=list(reversed(range(16))))
            },
        )
        validate_tick_ir(ir)
        c = emit_x86_sse2_c(ir)
        self.assertIn("_mm_shuffle_epi8", c)

        x = 0x000102030405060708090A0B0C0D0E0F
        expected = int(eval_expr(ir.output_exprs["y"], {"x": t}, {"x": x}))

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
                        "  unsigned long long xlo, xhi;",
                        '  while (scanf("%llx %llx", &xlo, &xhi) == 2) {',
                        "    uint64_t in[2];",
                        "    uint64_t out[2];",
                        "    in[0] = (uint64_t)xlo;",
                        "    in[1] = (uint64_t)xhi;",
                        "    stc_eval(in, out);",
                        '    printf("%016llx %016llx\\n", (unsigned long long)out[0], (unsigned long long)out[1]);',
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
                    "-mssse3",
                    "-o",
                    str(exe),
                    str(out_dir / "main.c"),
                    str(out_dir / "impl.c"),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            stdin = f"{x & ((1<<64)-1):x} {x >> 64:x}\n"
            p = subprocess.run(
                [str(exe)],
                input=stdin.encode("utf-8"),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            parts = p.stdout.decode("utf-8").strip().split()
            self.assertEqual(len(parts), 2)
            got = int(parts[0], 16) | (int(parts[1], 16) << 64)
            self.assertEqual(got, expected)
