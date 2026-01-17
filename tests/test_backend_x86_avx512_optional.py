import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stc.backend_x86_avx512 import emit_x86_avx512_c
from stc.interp import eval_expr
from stc.tick_ir import (
    BitVecConst,
    SimdAdd,
    SimdAddMasked,
    SimdAnd,
    SimdEq,
    SimdLShr,
    SimdMaxS,
    SimdMaxU,
    SimdMaskPack,
    SimdMinS,
    SimdMinU,
    SimdNot,
    SimdShl,
    SimdShuffle,
    SimdSub,
    SimdSubMasked,
    SimdType,
    SimdUlt,
    SimdXor,
    SimdBlend,
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
    shutil.which("cc") and platform.machine() == "x86_64" and _has_cpu_flag("avx512f"),
    "requires cc, x86_64, and avx512f",
)
class TestBackendX86Avx512Optional(unittest.TestCase):
    def test_emit_and_run_avx512_epi32_and_cmp_matches_interpreter(self) -> None:
        t = SimdType(lane_width=32, lanes=16)
        sh = BitVecConst(width=32, value=1)
        rev = list(range(16))[::-1]
        ir = TickIR(
            name="t32x16",
            inputs={"x": t, "y": t},
            outputs={
                "add": t,
                "addm": t,
                "sub": t,
                "subm": t,
                "and_": t,
                "xor_": t,
                "notx": t,
                "shl1": t,
                "lshr1": t,
                "minu": t,
                "maxu": t,
                "mins": t,
                "maxs": t,
                "blend": t,
                "rev": t,
                "eq": SimdType(lane_width=1, lanes=16),
                "ult": SimdType(lane_width=1, lanes=16),
                "nzx": SimdType(lane_width=1, lanes=16),
            },
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "add": SimdAdd(a=Var("x"), b=Var("y")),
                "addm": SimdAddMasked(
                    mask=SimdUlt(a=Var("x"), b=Var("y")),
                    passthru=Var("x"),
                    a=Var("x"),
                    b=Var("y"),
                ),
                "sub": SimdSub(a=Var("x"), b=Var("y")),
                "subm": SimdSubMasked(
                    mask=SimdUlt(a=Var("x"), b=Var("y")),
                    passthru=Var("x"),
                    a=Var("x"),
                    b=Var("y"),
                ),
                "and_": SimdAnd(a=Var("x"), b=Var("y")),
                "xor_": SimdXor(a=Var("x"), b=Var("y")),
                "notx": SimdNot(x=Var("x")),
                "shl1": SimdShl(a=Var("x"), sh=sh),
                "lshr1": SimdLShr(a=Var("x"), sh=sh),
                "minu": SimdMinU(a=Var("x"), b=Var("y")),
                "maxu": SimdMaxU(a=Var("x"), b=Var("y")),
                "mins": SimdMinS(a=Var("x"), b=Var("y")),
                "maxs": SimdMaxS(a=Var("x"), b=Var("y")),
                "blend": SimdBlend(
                    a=Var("x"), b=Var("y"), mask=SimdUlt(a=Var("x"), b=Var("y"))
                ),
                "rev": SimdShuffle(x=Var("x"), indices=rev),
                "eq": SimdEq(a=Var("x"), b=Var("y")),
                "ult": SimdUlt(a=Var("x"), b=Var("y")),
                "nzx": SimdMaskPack(x=Var("x")),
            },
        )
        validate_tick_ir(ir)
        c = emit_x86_avx512_c(ir)

        cases = [
            (
                int(
                    "0000000100000002000000030000000400000005000000060000000700000008"
                    "000000090000000a0000000b0000000c0000000d0000000e0000000f00000010",
                    16,
                ),
                int(
                    "0000000100000001000000010000000100000001000000010000000100000001"
                    "0000000100000001000000010000000100000001000000010000000100000001",
                    16,
                ),
            ),
            (
                int(
                    "8000000080000000800000008000000080000000800000008000000080000000"
                    "8000000080000000800000008000000080000000800000008000000080000000",
                    16,
                ),
                int(
                    "0000000100000001000000010000000100000001000000010000000100000001"
                    "0000000100000001000000010000000100000001000000010000000100000001",
                    16,
                ),
            ),
        ]

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
                        "  unsigned long long x0,x1,x2,x3,x4,x5,x6,x7,y0,y1,y2,y3,y4,y5,y6,y7;",
                        '  while (scanf("%llx %llx %llx %llx %llx %llx %llx %llx %llx %llx %llx %llx %llx %llx %llx %llx", &x0,&x1,&x2,&x3,&x4,&x5,&x6,&x7,&y0,&y1,&y2,&y3,&y4,&y5,&y6,&y7) == 16) {',
                        "    uint64_t in[16];",
                        f"    uint64_t out[{out_words}];",
                        "    in[0]=(uint64_t)x0; in[1]=(uint64_t)x1; in[2]=(uint64_t)x2; in[3]=(uint64_t)x3;",
                        "    in[4]=(uint64_t)x4; in[5]=(uint64_t)x5; in[6]=(uint64_t)x6; in[7]=(uint64_t)x7;",
                        "    in[8]=(uint64_t)y0; in[9]=(uint64_t)y1; in[10]=(uint64_t)y2; in[11]=(uint64_t)y3;",
                        "    in[12]=(uint64_t)y4; in[13]=(uint64_t)y5; in[14]=(uint64_t)y6; in[15]=(uint64_t)y7;",
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
                xs = [(x >> (64 * i)) & ((1 << 64) - 1) for i in range(8)]
                ys = [(y >> (64 * i)) & ((1 << 64) - 1) for i in range(8)]
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
                    got = 0
                    for widx in range(8):
                        got |= ws[idx * 8 + widx] << (64 * widx)
                    exp = int(eval_expr(ir.output_exprs[name], types, env))
                    self.assertEqual(got, exp)
