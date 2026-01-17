import platform
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from stc.backend_x86_avx512_float import emit_x86_avx512_float_c
from stc.interp import eval_expr
from stc.tick_ir import (
    SimdBlend,
    SimdFCmpEq,
    SimdFCmpLe,
    SimdFCmpLt,
    SimdFCmpNe,
    SimdFAbs,
    SimdFAdd,
    SimdFDiv,
    SimdFNeg,
    SimdFSqrt,
    SimdFSub,
    SimdFMul,
    SimdType,
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


def _f32_bits(x: float) -> int:
    return int.from_bytes(struct.pack("<f", float(x)), "little")


def _f64_bits(x: float) -> int:
    return int.from_bytes(struct.pack("<d", float(x)), "little")


@unittest.skipUnless(
    shutil.which("cc") and platform.machine() == "x86_64" and _has_cpu_flag("avx512f"),
    "requires cc, x86_64, and avx512f",
)
class TestBackendX86Avx512FloatOptional(unittest.TestCase):
    def _compile_and_run(
        self, ir: TickIR, cases: list[tuple[int, int]]
    ) -> list[list[int]]:
        validate_tick_ir(ir)
        c = emit_x86_avx512_float_c(ir)
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
            out = [[int(w, 16) for w in ln.split()] for ln in lines]
            for ws in out:
                self.assertEqual(len(ws), out_words)
            return out

    def test_emit_and_run_avx512_ps_matches_interpreter(self) -> None:
        t = SimdType(lane_width=32, lanes=16)
        m = SimdType(lane_width=1, lanes=16)
        ir = TickIR(
            name="ps512",
            inputs={"x": t, "y": t},
            outputs={
                "add": t,
                "sub": t,
                "mul": t,
                "div": t,
                "sqrtx": t,
                "negx": t,
                "absx": t,
                "blend": t,
                "eq": m,
                "lt": m,
                "le": m,
                "ne": m,
            },
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "add": SimdFAdd(a=Var("x"), b=Var("y")),
                "sub": SimdFSub(a=Var("x"), b=Var("y")),
                "mul": SimdFMul(a=Var("x"), b=Var("y")),
                "div": SimdFDiv(a=Var("x"), b=Var("y")),
                "sqrtx": SimdFSqrt(x=SimdFAbs(x=Var("x"))),
                "negx": SimdFNeg(x=Var("x")),
                "absx": SimdFAbs(x=Var("x")),
                "blend": SimdBlend(
                    mask=SimdFCmpLt(a=Var("x"), b=Var("y")), a=Var("x"), b=Var("y")
                ),
                "eq": SimdFCmpEq(a=Var("x"), b=Var("y")),
                "lt": SimdFCmpLt(a=Var("x"), b=Var("y")),
                "le": SimdFCmpLe(a=Var("x"), b=Var("y")),
                "ne": SimdFCmpNe(a=Var("x"), b=Var("y")),
            },
        )
        validate_tick_ir(ir)

        xs = [float(i + 1) for i in range(16)]
        ys = [float(16 - i) for i in range(16)]
        x_bits = 0
        y_bits = 0
        for i, (a, b) in enumerate(zip(xs, ys, strict=True)):
            x_bits |= _f32_bits(a) << (32 * i)
            y_bits |= _f32_bits(b) << (32 * i)

        x_nan = x_bits & ~0xFFFFFFFF
        x_nan |= 0x7FA12345
        cases = [(x_bits, y_bits), (x_nan, y_bits)]
        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x": t, "y": t}
        for (x, y), ws in zip(cases, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                if name in {"eq", "lt", "le", "ne"}:
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

    def test_emit_and_run_avx512_pd_matches_interpreter(self) -> None:
        t = SimdType(lane_width=64, lanes=8)
        m = SimdType(lane_width=1, lanes=8)
        ir = TickIR(
            name="pd512",
            inputs={"x": t, "y": t},
            outputs={
                "add": t,
                "sub": t,
                "mul": t,
                "div": t,
                "sqrtx": t,
                "negx": t,
                "absx": t,
                "blend": t,
                "eq": m,
                "lt": m,
                "le": m,
                "ne": m,
            },
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "add": SimdFAdd(a=Var("x"), b=Var("y")),
                "sub": SimdFSub(a=Var("x"), b=Var("y")),
                "mul": SimdFMul(a=Var("x"), b=Var("y")),
                "div": SimdFDiv(a=Var("x"), b=Var("y")),
                "sqrtx": SimdFSqrt(x=SimdFAbs(x=Var("x"))),
                "negx": SimdFNeg(x=Var("x")),
                "absx": SimdFAbs(x=Var("x")),
                "blend": SimdBlend(
                    mask=SimdFCmpLt(a=Var("x"), b=Var("y")), a=Var("x"), b=Var("y")
                ),
                "eq": SimdFCmpEq(a=Var("x"), b=Var("y")),
                "lt": SimdFCmpLt(a=Var("x"), b=Var("y")),
                "le": SimdFCmpLe(a=Var("x"), b=Var("y")),
                "ne": SimdFCmpNe(a=Var("x"), b=Var("y")),
            },
        )
        validate_tick_ir(ir)

        xs = [3.25, -1.5, 0.25, 10.0, 2.0, 5.5, -0.5, 1.0]
        ys = [0.5, -2.0, 1.25, -2.0, 2.25, 0.5, 0.5, 1.0]
        x_bits = 0
        y_bits = 0
        for i, (a, b) in enumerate(zip(xs, ys, strict=True)):
            x_bits |= _f64_bits(a) << (64 * i)
            y_bits |= _f64_bits(b) << (64 * i)

        x_nan = x_bits & ~0xFFFFFFFFFFFFFFFF
        x_nan |= 0x7FF0123456789ABC
        cases = [(x_bits, y_bits), (x_nan, y_bits)]
        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x": t, "y": t}
        for (x, y), ws in zip(cases, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                if name in {"eq", "lt", "le", "ne"}:
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
