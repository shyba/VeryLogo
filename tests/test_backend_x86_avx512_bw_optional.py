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
    SimdAShr,
    SimdAdd,
    SimdAddMasked,
    SimdAddSatS,
    SimdAddSatU,
    SimdAnd,
    SimdBlend,
    SimdEq,
    SimdLShr,
    SimdMaddS16,
    SimdMaxS,
    SimdMaxU,
    SimdMaskPack,
    SimdMinS,
    SimdMinU,
    SimdMulHiS,
    SimdMulHiU,
    SimdMulLo,
    SimdNot,
    SimdOr,
    SimdSExtLo,
    SimdShl,
    SimdSub,
    SimdSubMasked,
    SimdSubSatS,
    SimdSubSatU,
    SimdType,
    SimdUlt,
    SimdXor,
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
    shutil.which("cc") and platform.machine() == "x86_64" and _has_cpu_flag("avx512bw"),
    "requires cc, x86_64, and avx512bw",
)
class TestBackendX86Avx512BwOptional(unittest.TestCase):
    def _compile_and_run(
        self, ir: TickIR, cases: list[tuple[int, int]]
    ) -> list[list[int]]:
        validate_tick_ir(ir)
        c = emit_x86_avx512_c(ir)
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
                    "-mavx512bw",
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
            out = [[int(w, 16) for w in ln.split()] for ln in lines]
            for ws in out:
                self.assertEqual(len(ws), out_words)
            return out

    def test_emit_and_run_avx512_epi8_matches_interpreter(self) -> None:
        t = SimdType(lane_width=8, lanes=64)
        m = SimdType(lane_width=1, lanes=64)
        ir = TickIR(
            name="epi8",
            inputs={"x": t, "y": t},
            outputs={
                "add": t,
                "addm": t,
                "sub": t,
                "subm": t,
                "addsatu": t,
                "subsatu": t,
                "addsats": t,
                "subsats": t,
                "and_": t,
                "or_": t,
                "xor_": t,
                "notx": t,
                "minu": t,
                "maxu": t,
                "mins": t,
                "maxs": t,
                "blend": t,
                "eq": m,
                "ult": m,
                "nzx": m,
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
                "addsatu": SimdAddSatU(a=Var("x"), b=Var("y")),
                "subsatu": SimdSubSatU(a=Var("x"), b=Var("y")),
                "addsats": SimdAddSatS(a=Var("x"), b=Var("y")),
                "subsats": SimdSubSatS(a=Var("x"), b=Var("y")),
                "and_": SimdAnd(a=Var("x"), b=Var("y")),
                "or_": SimdOr(a=Var("x"), b=Var("y")),
                "xor_": SimdXor(a=Var("x"), b=Var("y")),
                "notx": SimdNot(x=Var("x")),
                "minu": SimdMinU(a=Var("x"), b=Var("y")),
                "maxu": SimdMaxU(a=Var("x"), b=Var("y")),
                "mins": SimdMinS(a=Var("x"), b=Var("y")),
                "maxs": SimdMaxS(a=Var("x"), b=Var("y")),
                "blend": SimdBlend(
                    a=Var("x"), b=Var("y"), mask=SimdUlt(a=Var("x"), b=Var("y"))
                ),
                "eq": SimdEq(a=Var("x"), b=Var("y")),
                "ult": SimdUlt(a=Var("x"), b=Var("y")),
                "nzx": SimdMaskPack(x=Var("x")),
            },
        )
        validate_tick_ir(ir)

        cases = [
            (
                int("00112233445566778899aabbccddeeff" * 4, 16),
                int("01010101010101010101010101010101" * 4, 16),
            ),
            (
                int("ffffffffffffffffffffffffffffffff" * 4, 16),
                int("00000000000000000000000000000000" * 4, 16),
            ),
            (
                int("7f" * 64, 16),
                int("01" * 64, 16),
            ),
            (
                int("80" * 64, 16),
                int("01" * 64, 16),
            ),
        ]
        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x": t, "y": t}
        for (x, y), ws in zip(cases, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                if name in {"eq", "ult", "nzx"}:
                    got = 0
                    got |= ws[idx * 8 + 0]
                    exp = int(eval_expr(ir.output_exprs[name], types, env))
                    self.assertEqual(got, exp)
                    for i in range(1, 8):
                        self.assertEqual(ws[idx * 8 + i], 0)
                else:
                    got = 0
                    for widx in range(8):
                        got |= ws[idx * 8 + widx] << (64 * widx)
                    exp = int(eval_expr(ir.output_exprs[name], types, env))
                    self.assertEqual(got, exp)

    def test_emit_and_run_avx512_epi16_shifts_matches_interpreter(self) -> None:
        t = SimdType(lane_width=16, lanes=32)
        m = SimdType(lane_width=1, lanes=32)
        sh = BitVecConst(width=32, value=1)
        ir = TickIR(
            name="epi16",
            inputs={"x": t, "y": t},
            outputs={
                "add": t,
                "addm": t,
                "sub": t,
                "subm": t,
                "shl1": t,
                "lshr1": t,
                "ashr1": t,
                "addsatu": t,
                "subsatu": t,
                "addsats": t,
                "subsats": t,
                "mullo": t,
                "mulhis": t,
                "mulhiu": t,
                "madd": SimdType(lane_width=32, lanes=16),
                "minu": t,
                "maxu": t,
                "mins": t,
                "maxs": t,
                "blend": t,
                "eq": m,
                "ult": m,
                "nzx": m,
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
                "shl1": SimdShl(a=Var("x"), sh=sh),
                "lshr1": SimdLShr(a=Var("x"), sh=sh),
                "ashr1": SimdAShr(a=Var("x"), sh=sh),
                "addsatu": SimdAddSatU(a=Var("x"), b=Var("y")),
                "subsatu": SimdSubSatU(a=Var("x"), b=Var("y")),
                "addsats": SimdAddSatS(a=Var("x"), b=Var("y")),
                "subsats": SimdSubSatS(a=Var("x"), b=Var("y")),
                "mullo": SimdMulLo(a=Var("x"), b=Var("y")),
                "mulhis": SimdMulHiS(a=Var("x"), b=Var("y")),
                "mulhiu": SimdMulHiU(a=Var("x"), b=Var("y")),
                "madd": SimdMaddS16(a=Var("x"), b=Var("y")),
                "minu": SimdMinU(a=Var("x"), b=Var("y")),
                "maxu": SimdMaxU(a=Var("x"), b=Var("y")),
                "mins": SimdMinS(a=Var("x"), b=Var("y")),
                "maxs": SimdMaxS(a=Var("x"), b=Var("y")),
                "blend": SimdBlend(
                    a=Var("x"), b=Var("y"), mask=SimdUlt(a=Var("x"), b=Var("y"))
                ),
                "eq": SimdEq(a=Var("x"), b=Var("y")),
                "ult": SimdUlt(a=Var("x"), b=Var("y")),
                "nzx": SimdMaskPack(x=Var("x")),
            },
        )
        validate_tick_ir(ir)

        cases = [
            (
                int(
                    "00010002000300040005000600070008"
                    "ffff80007fff00010002000300040005"
                    "00110012001300140015001600170018"
                    "0019001a001b001c001d001e001f0020",
                    16,
                ),
                int(
                    "00010002000300040005000600070008"
                    "00010002000300040005000600070008"
                    "00110012001300140015001600170018"
                    "00110012001300140015001600170018",
                    16,
                ),
            ),
            (
                int("7fff" * 32, 16),
                int("0001" * 32, 16),
            ),
            (
                int("8000" * 32, 16),
                int("0001" * 32, 16),
            ),
            (
                int("ffff" * 32, 16),
                int("0002" * 32, 16),
            ),
        ]
        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x": t, "y": t}
        for (x, y), ws in zip(cases, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                if name in {"eq", "ult", "nzx"}:
                    got = ws[idx * 8 + 0]
                    exp = int(eval_expr(ir.output_exprs[name], types, env))
                    self.assertEqual(got, exp)
                    for i in range(1, 8):
                        self.assertEqual(ws[idx * 8 + i], 0)
                else:
                    got = 0
                    for widx in range(8):
                        got |= ws[idx * 8 + widx] << (64 * widx)
                    exp = int(eval_expr(ir.output_exprs[name], types, env))
                    self.assertEqual(got, exp)

    def test_emit_and_run_avx512_zext_sext_matches_interpreter(self) -> None:
        t8 = SimdType(lane_width=8, lanes=64)
        t16 = SimdType(lane_width=16, lanes=32)
        ir = TickIR(
            name="zext_sext",
            inputs={"x": t8, "y": t8},
            outputs={"zext": t16, "sext": t16},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "zext": SimdZExtLo(x=Var("x"), to=t16),
                "sext": SimdSExtLo(x=Var("x"), to=t16),
            },
        )
        validate_tick_ir(ir)

        cases = [
            (
                int("0001027f8081feff" * 8, 16),
                0,
            ),
        ]
        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x": t8, "y": t8}
        for (x, y), ws in zip(cases, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                got = 0
                for widx in range(8):
                    got |= ws[idx * 8 + widx] << (64 * widx)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)
