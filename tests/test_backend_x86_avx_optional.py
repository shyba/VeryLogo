import platform
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from stc.backend_x86_avx import emit_x86_avx_c
from stc.interp import eval_expr
from stc.tick_ir import (
    SimdBlend,
    SimdFCmpEq,
    SimdFCmpLe,
    SimdFCmpLt,
    SimdFCmpNe,
    SimdFAdd,
    SimdFAbs,
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
    shutil.which("cc") and platform.machine() == "x86_64" and _has_cpu_flag("avx"),
    "requires cc, x86_64, and avx",
)
class TestBackendX86AvxOptional(unittest.TestCase):
    def _compile_and_run(
        self, ir: TickIR, cases: list[tuple[int, int]]
    ) -> list[list[int]]:
        validate_tick_ir(ir)
        c = emit_x86_avx_c(ir)
        out_order = sorted(ir.outputs.keys())

        with tempfile.TemporaryDirectory() as d:
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
                        "  unsigned long long x0,x1,x2,x3,y0,y1,y2,y3;",
                        '  while (scanf("%llx %llx %llx %llx %llx %llx %llx %llx", &x0,&x1,&x2,&x3,&y0,&y1,&y2,&y3) == 8) {',
                        "    uint64_t in[8];",
                        f"    uint64_t out[{len(out_order) * 4}];",
                        "    in[0] = (uint64_t)x0;",
                        "    in[1] = (uint64_t)x1;",
                        "    in[2] = (uint64_t)x2;",
                        "    in[3] = (uint64_t)x3;",
                        "    in[4] = (uint64_t)y0;",
                        "    in[5] = (uint64_t)y1;",
                        "    in[6] = (uint64_t)y2;",
                        "    in[7] = (uint64_t)y3;",
                        "    stc_eval(in, out);",
                        f"    for (int i = 0; i < {len(out_order) * 4}; i++) {{",
                        '      printf("%016llx%s", (unsigned long long)out[i], (i == '
                        f"{len(out_order) * 4 - 1}"
                        ') ? "\\n" : " ");',
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
                    "-mavx",
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
                xs = [(x >> (64 * i)) & ((1 << 64) - 1) for i in range(4)]
                ys = [(y >> (64 * i)) & ((1 << 64) - 1) for i in range(4)]
                stdin += f"{xs[0]:x} {xs[1]:x} {xs[2]:x} {xs[3]:x} {ys[0]:x} {ys[1]:x} {ys[2]:x} {ys[3]:x}\n"
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
            out_words = [[int(w, 16) for w in ln.split()] for ln in lines]
            for ws in out_words:
                self.assertEqual(len(ws), len(out_order) * 4)
            return out_words

    def test_emit_and_run_avx_ps_matches_interpreter(self) -> None:
        t = SimdType(lane_width=32, lanes=8)
        m = SimdType(lane_width=1, lanes=8)
        ir = TickIR(
            name="ps",
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

        xs = [1.0, 2.0, -3.0, 4.5, 0.25, -0.5, 10.0, -1.25]
        ys = [0.5, -2.0, 3.0, -1.5, 1.25, 0.5, -2.0, 2.25]
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
                got = 0
                if name in {"eq", "lt", "le", "ne"}:
                    got = ws[idx * 4 + 0]
                else:
                    for w in range(4):
                        got |= ws[idx * 4 + w] << (64 * w)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)

    def test_emit_and_run_avx_pd_matches_interpreter(self) -> None:
        t = SimdType(lane_width=64, lanes=4)
        m = SimdType(lane_width=1, lanes=4)
        ir = TickIR(
            name="pd",
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

        xs = [3.25, -1.5, 0.25, 10.0]
        ys = [0.5, -2.0, 1.25, -2.0]
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
                got = 0
                if name in {"eq", "lt", "le", "ne"}:
                    got = ws[idx * 4 + 0]
                else:
                    for w in range(4):
                        got |= ws[idx * 4 + w] << (64 * w)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)
