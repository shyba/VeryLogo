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
    SimdAddSatS,
    SimdAddSatU,
    SimdAnd,
    SimdLShr,
    SimdMaddS16,
    SimdMulHiS,
    SimdMulHiU,
    SimdMulLo,
    SimdNot,
    SimdOr,
    SimdPackSS16To8,
    SimdPackSS32To16,
    SimdPackUS16To8,
    SimdShl,
    SimdSub,
    SimdSubSatS,
    SimdSubSatU,
    SimdType,
    SimdUnpackHi,
    SimdUnpackLo,
    SimdXor,
    TickIR,
    Var,
)
from stc.tick_ir_validate import validate_tick_ir


@unittest.skipUnless(
    shutil.which("cc") and platform.machine() == "x86_64",
    "requires cc and x86_64",
)
class TestBackendX86Sse2LaneWidthsOptional(unittest.TestCase):
    def _compile_and_run(
        self, ir: TickIR, cases: list[tuple[int, int]]
    ) -> list[list[int]]:
        validate_tick_ir(ir)
        c = emit_x86_sse2_c(ir)
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
                        "  unsigned long long xlo, xhi, ylo, yhi;",
                        '  while (scanf("%llx %llx %llx %llx", &xlo, &xhi, &ylo, &yhi) == 4) {',
                        f"    uint64_t out[{len(out_order) * 2}];",
                        "    uint64_t in[4];",
                        "    in[0] = (uint64_t)xlo;",
                        "    in[1] = (uint64_t)xhi;",
                        "    in[2] = (uint64_t)ylo;",
                        "    in[3] = (uint64_t)yhi;",
                        "    stc_eval(in, out);",
                        f"    for (int i = 0; i < {len(out_order) * 2}; i++) {{",
                        '      printf("%016llx%s", (unsigned long long)out[i], (i == '
                        f"{len(out_order) * 2 - 1}"
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
            out_words = [[int(w, 16) for w in ln.split()] for ln in lines]
            for ws in out_words:
                self.assertEqual(len(ws), len(out_order) * 2)
            return out_words

    def test_lane_width_8_add_sub_bitwise(self) -> None:
        t = SimdType(lane_width=8, lanes=16)
        ir = TickIR(
            name="t8",
            inputs={"x": t, "y": t},
            outputs={"add": t, "sub": t, "xor_": t, "and_": t, "or_": t, "notx": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "add": SimdAdd(a=Var("x"), b=Var("y")),
                "sub": SimdSub(a=Var("x"), b=Var("y")),
                "xor_": SimdXor(a=Var("x"), b=Var("y")),
                "and_": SimdAnd(a=Var("x"), b=Var("y")),
                "or_": SimdOr(a=Var("x"), b=Var("y")),
                "notx": SimdNot(x=Var("x")),
            },
        )
        cases = [
            (0x000102030405060708090A0B0C0D0E0F, 0x0F0E0D0C0B0A09080706050403020100),
            (0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF, 0x0102030405060708090A0B0C0D0E0F10),
        ]
        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x": t, "y": t}
        for (x, y), ws in zip(cases, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                got = ws[idx * 2] | (ws[idx * 2 + 1] << 64)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)

    def test_lane_width_8_unsigned_saturating_add_sub(self) -> None:
        t = SimdType(lane_width=8, lanes=16)
        ir = TickIR(
            name="sat8",
            inputs={"x": t, "y": t},
            outputs={"adds": t, "subs": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "adds": SimdAddSatU(a=Var("x"), b=Var("y")),
                "subs": SimdSubSatU(a=Var("x"), b=Var("y")),
            },
        )
        cases = [
            (
                0x00_FF_01_FE_10_F0_80_7F_00_01_02_03_04_05_06_07,
                0x00_01_FF_02_F0_20_80_01_FF_FE_FD_FC_FB_FA_F9_F8,
            ),
            (
                0x00000000000000000000000000000000,
                0x0102030405060708090A0B0C0D0E0F10,
            ),
        ]
        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x": t, "y": t}
        for (x, y), ws in zip(cases, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                got = ws[idx * 2] | (ws[idx * 2 + 1] << 64)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)

    def test_lane_width_8_signed_saturating_add_sub(self) -> None:
        t = SimdType(lane_width=8, lanes=16)
        ir = TickIR(
            name="sats8",
            inputs={"x": t, "y": t},
            outputs={"adds": t, "subs": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "adds": SimdAddSatS(a=Var("x"), b=Var("y")),
                "subs": SimdSubSatS(a=Var("x"), b=Var("y")),
            },
        )
        cases = [
            (
                0x00_00_00_00_00_00_00_00_F0_10_C0_40_FE_01_80_7F,
                0x00_00_00_00_00_00_00_00_90_70_C0_40_80_7F_FF_01,
            ),
            (
                0x7F7F7F7F7F7F7F7F8080808080808080,
                0x0101010101010101FFFFFFFFFFFFFFFF,
            ),
        ]
        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x": t, "y": t}
        for (x, y), ws in zip(cases, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                got = ws[idx * 2] | (ws[idx * 2 + 1] << 64)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)

    def test_lane_width_16_shifts(self) -> None:
        t = SimdType(lane_width=16, lanes=8)
        sh = BitVecConst(width=16, value=3)
        ir = TickIR(
            name="t16",
            inputs={"x": t, "y": t},
            outputs={"shl": t, "lshr": t, "ashr": t, "add": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "shl": SimdShl(a=Var("x"), sh=sh),
                "lshr": SimdLShr(a=Var("x"), sh=sh),
                "ashr": SimdAShr(a=Var("x"), sh=sh),
                "add": SimdAdd(a=Var("x"), b=Var("y")),
            },
        )
        cases = [
            (0x80018002800380048005800680078008, 0x00010002000300040005000600070008),
            (0xFFFF0001FFFF0001FFFF0001FFFF0001, 0x00000001000000010000000100000001),
        ]
        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x": t, "y": t}
        for (x, y), ws in zip(cases, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                got = ws[idx * 2] | (ws[idx * 2 + 1] << 64)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)

    def test_lane_width_16_mul_and_madd(self) -> None:
        t = SimdType(lane_width=16, lanes=8)
        o32 = SimdType(lane_width=32, lanes=4)
        ir = TickIR(
            name="mul16",
            inputs={"x": t, "y": t},
            outputs={"lo": t, "hiu": t, "his": t, "madd": o32},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "lo": SimdMulLo(a=Var("x"), b=Var("y")),
                "hiu": SimdMulHiU(a=Var("x"), b=Var("y")),
                "his": SimdMulHiS(a=Var("x"), b=Var("y")),
                "madd": SimdMaddS16(a=Var("x"), b=Var("y")),
            },
        )
        cases = [
            (
                0x0001_0002_0003_0004_7FFF_8000_FFFF_1234,
                0xFFFF_0002_0002_0002_0001_0002_0002_1000,
            ),
            (
                0xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF,
                0x0002_0002_0002_0002_0002_0002_0002_0002,
            ),
        ]
        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x": t, "y": t}
        for (x, y), ws in zip(cases, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                got = ws[idx * 2] | (ws[idx * 2 + 1] << 64)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)

    def test_unpack_and_pack_16_to_8(self) -> None:
        t16 = SimdType(lane_width=16, lanes=8)
        t8 = SimdType(lane_width=8, lanes=16)
        ir = TickIR(
            name="pack16",
            inputs={"x": t16, "y": t16},
            outputs={"unlo": t16, "unhi": t16, "pss": t8, "pus": t8},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "unlo": SimdUnpackLo(a=Var("x"), b=Var("y")),
                "unhi": SimdUnpackHi(a=Var("x"), b=Var("y")),
                "pss": SimdPackSS16To8(a=Var("x"), b=Var("y")),
                "pus": SimdPackUS16To8(a=Var("x"), b=Var("y")),
            },
        )
        cases = [
            (
                0x0000_0001_0002_0003_0004_0005_0006_0007,
                0x0064_0065_0066_0067_0068_0069_006A_006B,
            ),
            (
                0x7FFF_8000_00FF_FF00_0000_007F_0080_FF80,
                0xFFFF_0001_0100_FF00_00C8_FF38_012C_FED4,
            ),
        ]
        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x": t16, "y": t16}
        for (x, y), ws in zip(cases, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                got = ws[idx * 2] | (ws[idx * 2 + 1] << 64)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)

    def test_pack_32_to_16(self) -> None:
        t32 = SimdType(lane_width=32, lanes=4)
        t16 = SimdType(lane_width=16, lanes=8)
        ir = TickIR(
            name="pack32",
            inputs={"x": t32, "y": t32},
            outputs={"psd": t16},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"psd": SimdPackSS32To16(a=Var("x"), b=Var("y"))},
        )
        cases = [
            (
                0x00000000_00007FFF_00008000_FFFF7FFF,
                0x7FFFFFFF_80000000_00000001_FFFFFFFF,
            ),
            (
                0x00010000_FFFE0000_7FFFFFFF_80000000,
                0x00000000_00000000_00000000_00000000,
            ),
        ]
        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x": t32, "y": t32}
        for (x, y), ws in zip(cases, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                got = ws[idx * 2] | (ws[idx * 2 + 1] << 64)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)

    def test_lane_width_64_add_sub_shifts(self) -> None:
        t = SimdType(lane_width=64, lanes=2)
        sh = BitVecConst(width=64, value=1)
        ir = TickIR(
            name="t64",
            inputs={"x": t, "y": t},
            outputs={"add": t, "sub": t, "shl": t, "lshr": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "add": SimdAdd(a=Var("x"), b=Var("y")),
                "sub": SimdSub(a=Var("x"), b=Var("y")),
                "shl": SimdShl(a=Var("x"), sh=sh),
                "lshr": SimdLShr(a=Var("x"), sh=sh),
            },
        )
        cases = [
            (0x0123456789ABCDEF_FEDCBA9876543210, 0x0000000000000001_0000000000000001),
            (0xFFFFFFFFFFFFFFFF_0000000000000000, 0x0000000000000001_FFFFFFFFFFFFFFFF),
        ]
        out_words = self._compile_and_run(ir, cases)
        out_order = sorted(ir.outputs.keys())
        types = {"x": t, "y": t}
        for (x, y), ws in zip(cases, out_words, strict=True):
            env = {"x": x, "y": y}
            for idx, name in enumerate(out_order):
                got = ws[idx * 2] | (ws[idx * 2 + 1] << 64)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)
