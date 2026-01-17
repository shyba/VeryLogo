import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stc.backend_x86_avx2 import emit_x86_avx2_c
from stc.interp import eval_expr
from stc.tick_ir import (
    BitVecConst,
    SimdAShr,
    SimdAdd,
    SimdAnd,
    SimdBlend,
    SimdEq,
    SimdLShr,
    SimdMaskPack,
    SimdMaddS16,
    SimdMulLo,
    SimdNot,
    SimdOr,
    SimdPackSS16To8,
    SimdPackSS32To16,
    SimdPackUS16To8,
    SimdShl,
    SimdShuffle,
    SimdSub,
    SimdType,
    SimdUnpackHi,
    SimdUnpackLo,
    SimdUlt,
    SimdXor,
    SimdZExtLo,
    SimdSExtLo,
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
    shutil.which("cc") and platform.machine() == "x86_64" and _has_cpu_flag("avx2"),
    "requires cc, x86_64, and avx2",
)
class TestBackendX86Avx2Optional(unittest.TestCase):
    def _compile_and_run(
        self, ir: TickIR, cases: list[tuple[int, int]]
    ) -> list[list[int]]:
        validate_tick_ir(ir)
        c = emit_x86_avx2_c(ir)
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
                    "-mavx2",
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

    def test_emit_and_run_avx2_epi32_and_cmp_matches_interpreter(self) -> None:
        t = SimdType(lane_width=32, lanes=8)
        sh = BitVecConst(width=32, value=1)
        ir = TickIR(
            name="t32",
            inputs={"x": t, "y": t},
            outputs={
                "add": t,
                "sub": t,
                "and_": t,
                "or_": t,
                "xor_": t,
                "notx": t,
                "shl1": t,
                "lshr1": t,
                "ashr1": t,
                "eq": SimdType(lane_width=1, lanes=8),
                "ult": SimdType(lane_width=1, lanes=8),
                "nzx": SimdType(lane_width=1, lanes=8),
            },
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "add": SimdAdd(a=Var("x"), b=Var("y")),
                "sub": SimdSub(a=Var("x"), b=Var("y")),
                "and_": SimdAnd(a=Var("x"), b=Var("y")),
                "or_": SimdOr(a=Var("x"), b=Var("y")),
                "xor_": SimdXor(a=Var("x"), b=Var("y")),
                "notx": SimdNot(x=Var("x")),
                "shl1": SimdShl(a=Var("x"), sh=sh),
                "lshr1": SimdLShr(a=Var("x"), sh=sh),
                "ashr1": SimdAShr(a=Var("x"), sh=sh),
                "eq": SimdEq(a=Var("x"), b=Var("y")),
                "ult": SimdUlt(a=Var("x"), b=Var("y")),
                "nzx": SimdMaskPack(x=Var("x")),
            },
        )
        validate_tick_ir(ir)

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

        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x": t, "y": t}
        for (x, y), ws in zip(cases, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                got = 0
                for w in range(4):
                    got |= ws[idx * 4 + w] << (64 * w)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)

    def test_emit_and_run_avx2_epi16_mul_madd_matches_interpreter(self) -> None:
        t16 = SimdType(lane_width=16, lanes=16)
        t32 = SimdType(lane_width=32, lanes=8)
        ir = TickIR(
            name="t16",
            inputs={"x": t16, "y": t16},
            outputs={"mullo": t16, "madd": t32},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "mullo": SimdMulLo(a=Var("x"), b=Var("y")),
                "madd": SimdMaddS16(a=Var("x"), b=Var("y")),
            },
        )
        validate_tick_ir(ir)
        cases = [
            (
                int(
                    "00010002000300040005000600070008FFFF8000123400020000000100FF00FF",
                    16,
                ),
                int(
                    "FFFF000200020002000100020002100000020002000200020002000200020002",
                    16,
                ),
            ),
            (
                int(
                    "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
                    16,
                ),
                int(
                    "0002000200020002000200020002000200020002000200020002000200020002",
                    16,
                ),
            ),
        ]
        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x": t16, "y": t16}
        for (x, y), ws in zip(cases, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                got = 0
                for w in range(4):
                    got |= ws[idx * 4 + w] << (64 * w)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)

    def test_emit_and_run_avx2_unpack_u8_matches_interpreter(self) -> None:
        t8 = SimdType(lane_width=8, lanes=32)
        ir = TickIR(
            name="unpack8",
            inputs={"x": t8, "y": t8},
            outputs={"lo": t8, "hi": t8},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "lo": SimdUnpackLo(a=Var("x"), b=Var("y")),
                "hi": SimdUnpackHi(a=Var("x"), b=Var("y")),
            },
        )
        validate_tick_ir(ir)
        cases = [
            (
                int(
                    "1f1e1d1c1b1a191817161514131211100f0e0d0c0b0a09080706050403020100",
                    16,
                ),
                int(
                    "3f3e3d3c3b3a393837363534333231302f2e2d2c2b2a29282726252423222120",
                    16,
                ),
            ),
            (
                int(
                    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                    16,
                ),
                int(
                    "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
                    16,
                ),
            ),
        ]
        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x": t8, "y": t8}
        for (x, y), ws in zip(cases, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                got = 0
                for w in range(4):
                    got |= ws[idx * 4 + w] << (64 * w)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)

    def test_emit_and_run_avx2_unpack_u16_matches_interpreter(self) -> None:
        t16 = SimdType(lane_width=16, lanes=16)
        ir = TickIR(
            name="unpack16",
            inputs={"x": t16, "y": t16},
            outputs={"lo": t16, "hi": t16},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "lo": SimdUnpackLo(a=Var("x"), b=Var("y")),
                "hi": SimdUnpackHi(a=Var("x"), b=Var("y")),
            },
        )
        validate_tick_ir(ir)
        cases = [
            (
                int(
                    "000100020003000400050006000700080009000a000b000c000d000e000f0010",
                    16,
                ),
                int(
                    "001100120013001400150016001700180019001a001b001c001d001e001f0020",
                    16,
                ),
            ),
            (
                int(
                    "ffff80007fff00010002000300040005fff0fff1000000010002000300040005",
                    16,
                ),
                int(
                    "00010002000300040005000600070008ffff80007fff00010002000300040005",
                    16,
                ),
            ),
        ]
        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x": t16, "y": t16}
        for (x, y), ws in zip(cases, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                got = 0
                for w in range(4):
                    got |= ws[idx * 4 + w] << (64 * w)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)

    def test_emit_and_run_avx2_pack_u16_to_u8_matches_interpreter(self) -> None:
        t16 = SimdType(lane_width=16, lanes=16)
        t8 = SimdType(lane_width=8, lanes=32)
        ir = TickIR(
            name="pack16to8",
            inputs={"x": t16, "y": t16},
            outputs={"ss": t8, "us": t8},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "ss": SimdPackSS16To8(a=Var("x"), b=Var("y")),
                "us": SimdPackUS16To8(a=Var("x"), b=Var("y")),
            },
        )
        validate_tick_ir(ir)
        cases = [
            (
                int(
                    "7fff8000007f0080ff80ff7f0100feff0000000100ff00c80064007fffff8000",
                    16,
                ),
                int(
                    "00010002000300040005000600070008fff0fff1000000010002000300040005",
                    16,
                ),
            ),
            (
                int(
                    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                    16,
                ),
                int(
                    "0000000100020003000400050006000700080009000a000b000c000d000e000f",
                    16,
                ),
            ),
        ]
        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x": t16, "y": t16}
        for (x, y), ws in zip(cases, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                got = 0
                for w in range(4):
                    got |= ws[idx * 4 + w] << (64 * w)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)

    def test_emit_and_run_avx2_pack_s32_to_s16_matches_interpreter(self) -> None:
        t32 = SimdType(lane_width=32, lanes=8)
        t16 = SimdType(lane_width=16, lanes=16)
        ir = TickIR(
            name="pack32to16",
            inputs={"x": t32, "y": t32},
            outputs={"ss": t16},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"ss": SimdPackSS32To16(a=Var("x"), b=Var("y"))},
        )
        validate_tick_ir(ir)
        cases = [
            (
                int(
                    "7fffffff8000000000000001ffffffff00010000fffe00007fff00008000ffff",
                    16,
                ),
                int(
                    "000000000000000000000000000000007fffffff80000000ffffffff00000001",
                    16,
                ),
            ),
            (
                int(
                    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                    16,
                ),
                int(
                    "0000000100020003000400050006000700080009000a000b000c000d000e000f",
                    16,
                ),
            ),
        ]
        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x": t32, "y": t32}
        for (x, y), ws in zip(cases, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                got = 0
                for w in range(4):
                    got |= ws[idx * 4 + w] << (64 * w)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)

    def test_emit_and_run_avx2_shuffle_u8_matches_interpreter(self) -> None:
        t8 = SimdType(lane_width=8, lanes=32)
        ir = TickIR(
            name="shuf8",
            inputs={"x": t8, "y": t8},
            outputs={"rev": t8, "mix": t8},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "rev": SimdShuffle(x=Var("x"), indices=list(reversed(range(32)))),
                "mix": SimdShuffle(
                    x=Var("x"),
                    indices=[
                        0,
                        16,
                        1,
                        17,
                        2,
                        18,
                        3,
                        19,
                        4,
                        20,
                        5,
                        21,
                        6,
                        22,
                        7,
                        23,
                        8,
                        24,
                        9,
                        25,
                        10,
                        26,
                        11,
                        27,
                        12,
                        28,
                        13,
                        29,
                        14,
                        30,
                        15,
                        31,
                    ],
                ),
            },
        )
        validate_tick_ir(ir)
        cases = [
            (
                int(
                    "1f1e1d1c1b1a191817161514131211100f0e0d0c0b0a09080706050403020100",
                    16,
                ),
                0,
            ),
            (
                int(
                    "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
                    16,
                ),
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
                for w in range(4):
                    got |= ws[idx * 4 + w] << (64 * w)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)

    def test_emit_and_run_avx2_shuffle_epi32_matches_interpreter(self) -> None:
        t32 = SimdType(lane_width=32, lanes=8)
        ir = TickIR(
            name="shuf32",
            inputs={"x": t32, "y": t32},
            outputs={"rev": t32},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "rev": SimdShuffle(x=Var("x"), indices=list(reversed(range(8))))
            },
        )
        validate_tick_ir(ir)
        cases = [
            (
                int(
                    "0000000100000002000000030000000400000005000000060000000700000008",
                    16,
                ),
                0,
            ),
            (
                int(
                    "ffffffff800000007fffffff0000000100010000fffe00007fff00008000ffff",
                    16,
                ),
                0,
            ),
        ]
        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x": t32, "y": t32}
        for (x, y), ws in zip(cases, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                got = 0
                for w in range(4):
                    got |= ws[idx * 4 + w] << (64 * w)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)

    def test_emit_and_run_avx2_extend_lo_matches_interpreter(self) -> None:
        t8 = SimdType(lane_width=8, lanes=32)
        t16 = SimdType(lane_width=16, lanes=16)
        t16s = SimdType(lane_width=16, lanes=16)
        t32 = SimdType(lane_width=32, lanes=8)
        ir = TickIR(
            name="extlo",
            inputs={"x8": t8, "y16": t16},
            outputs={"z8": t16, "s8": t16s, "z16": t32, "s16": t32},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "z8": SimdZExtLo(to=t16, x=Var("x8")),
                "s8": SimdSExtLo(to=t16s, x=Var("x8")),
                "z16": SimdZExtLo(to=t32, x=Var("y16")),
                "s16": SimdSExtLo(to=t32, x=Var("y16")),
            },
        )
        validate_tick_ir(ir)
        cases = [
            (
                int(
                    "ff80ff7f0100feff0000000100ff00c80064007f8001ffff7ffe000102030405",
                    16,
                ),
                int(
                    "ffff80007fff00010002000300040005fff0fff1000000010002000300040005",
                    16,
                ),
            ),
            (
                int(
                    "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
                    16,
                ),
                int(
                    "000100020003000400050006000700080009000a000b000c000d000e000f0010",
                    16,
                ),
            ),
        ]
        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x8": t8, "y16": t16}
        for (x8, y16), ws in zip(cases, out_words, strict=True):
            env = {"x8": x8, "y16": y16}
            for idx, name in enumerate(out_order):
                got = 0
                for w in range(4):
                    got |= ws[idx * 4 + w] << (64 * w)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)

    def test_emit_and_run_avx2_blend_matches_interpreter(self) -> None:
        t = SimdType(lane_width=16, lanes=16)
        ir = TickIR(
            name="blend",
            inputs={"x": t, "y": t},
            outputs={"out": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "out": SimdBlend(
                    mask=SimdUlt(a=Var("x"), b=Var("y")), a=Var("x"), b=Var("y")
                )
            },
        )
        validate_tick_ir(ir)
        cases = [
            (
                int(
                    "000100020003000400050006000700080009000a000b000c000d000e000f0010",
                    16,
                ),
                int(
                    "001100020013000400150006001700080019000a001b000c001d000e001f0010",
                    16,
                ),
            ),
            (
                int(
                    "ffff80007fff00010002000300040005fff0fff1000000010002000300040005",
                    16,
                ),
                int(
                    "00010002000300040005000600070008ffff80007fff00010002000300040005",
                    16,
                ),
            ),
        ]
        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x": t, "y": t}
        for (x, y), ws in zip(cases, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                got = 0
                for w in range(4):
                    got |= ws[idx * 4 + w] << (64 * w)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)

    def test_emit_and_run_avx2_mask_pack_u8_u16_matches_interpreter(self) -> None:
        t8 = SimdType(lane_width=8, lanes=32)
        t16 = SimdType(lane_width=16, lanes=16)
        m8 = SimdType(lane_width=1, lanes=32)
        m16 = SimdType(lane_width=1, lanes=16)

        ir8 = TickIR(
            name="mask8",
            inputs={"x": t8, "y": t8},
            outputs={"nzx": m8},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"nzx": SimdMaskPack(x=Var("x"))},
        )
        validate_tick_ir(ir8)
        cases8 = [
            (
                int(
                    "0001000000010000000100000001000000010000000100000001000000010000",
                    16,
                ),
                0,
            ),
            (
                int(
                    "00000000000000000000000000000000ffffffffffffffffffffffffffffffff",
                    16,
                ),
                0,
            ),
        ]
        out_words = self._compile_and_run(ir8, cases8)
        out_order = sorted(ir8.outputs.keys())
        types = {"x": t8, "y": t8}
        for (x, y), ws in zip(cases8, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                got = ws[idx * 4 + 0]
                exp = int(eval_expr(ir8.output_exprs[name], types, env))
                self.assertEqual(got, exp)
                self.assertEqual(ws[idx * 4 + 1], 0)
                self.assertEqual(ws[idx * 4 + 2], 0)
                self.assertEqual(ws[idx * 4 + 3], 0)

        ir16 = TickIR(
            name="mask16",
            inputs={"x": t16, "y": t16},
            outputs={"nzx": m16},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"nzx": SimdMaskPack(x=Var("x"))},
        )
        validate_tick_ir(ir16)
        cases16 = [
            (
                int(
                    "0001000000010000000100000001000000010000000100000001000000010000",
                    16,
                ),
                0,
            ),
            (
                int(
                    "0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff",
                    16,
                ),
                0,
            ),
        ]
        out_words = self._compile_and_run(ir16, cases16)
        out_order = sorted(ir16.outputs.keys())
        types = {"x": t16, "y": t16}
        for (x, y), ws in zip(cases16, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                got = ws[idx * 4 + 0]
                exp = int(eval_expr(ir16.output_exprs[name], types, env))
                self.assertEqual(got, exp)
                self.assertEqual(ws[idx * 4 + 1], 0)
                self.assertEqual(ws[idx * 4 + 2], 0)
                self.assertEqual(ws[idx * 4 + 3], 0)

    def test_emit_and_run_avx2_mask_only_program_matches_interpreter(self) -> None:
        t = SimdType(lane_width=32, lanes=8)
        m = SimdType(lane_width=1, lanes=8)
        ir = TickIR(
            name="maskonly",
            inputs={"x": t, "y": t},
            outputs={"eq": m, "ult": m, "nzx": m},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "eq": SimdEq(a=Var("x"), b=Var("y")),
                "ult": SimdUlt(a=Var("x"), b=Var("y")),
                "nzx": SimdMaskPack(x=Var("x")),
            },
        )
        validate_tick_ir(ir)
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
                    "00000000000000000000000000000000ffffffffffffffffffffffffffffffff",
                    16,
                ),
                int(
                    "00000000000000000000000000000000ffffffffffffffffffffffffffffffff",
                    16,
                ),
            ),
        ]
        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x": t, "y": t}
        for (x, y), ws in zip(cases, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                got = ws[idx * 4 + 0]
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)
                self.assertEqual(ws[idx * 4 + 1], 0)
                self.assertEqual(ws[idx * 4 + 2], 0)
                self.assertEqual(ws[idx * 4 + 3], 0)
