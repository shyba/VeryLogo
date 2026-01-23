import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stc.backend_x86_sse2 import emit_x86_sse2_c
from stc.bitslice import AES_SBOX_TABLE
from stc.interp import eval_expr
from stc.tick_ir import (
    BitTranspose,
    SimdType,
    TickIR,
    Var,
)
from stc.tick_ir_validate import validate_tick_ir


@unittest.skipUnless(
    shutil.which("cc") and platform.machine() == "x86_64",
    "requires cc and x86_64",
)
class TestBitTransposeBackendOptional(unittest.TestCase):
    def _compile_and_run_single_input(
        self, ir: TickIR, cases: list[int]
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
                        "  unsigned long long xlo, xhi;",
                        '  while (scanf("%llx %llx", &xlo, &xhi) == 2) {',
                        f"    uint64_t out[{len(out_order) * 2}];",
                        "    uint64_t in[2];",
                        "    in[0] = (uint64_t)xlo;",
                        "    in[1] = (uint64_t)xhi;",
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
                    "-msse4.1",
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
            for x in cases:
                stdin += f"{x & ((1<<64)-1):x} {x >> 64:x}\n"
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

    def test_bit_transpose_8x16_roundtrip(self) -> None:
        """Test that forward+backward BitTranspose is identity."""
        fwd = BitTranspose(x=Var("x"), lane_width=8, lanes=16)
        bwd = BitTranspose(x=fwd, lane_width=16, lanes=8)
        ir = TickIR(
            name="bt_roundtrip",
            inputs={"x": SimdType(lane_width=8, lanes=16)},
            state={},
            reset_state={},
            outputs={"y": SimdType(lane_width=8, lanes=16)},
            next_state={},
            output_exprs={"y": bwd},
        )
        cases = [
            0x000102030405060708090A0B0C0D0E0F,
            0xFFEEDDCCBBAA99887766554433221100,
            0x00000000000000000000000000000000,
            0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
        ]
        out_words = self._compile_and_run_single_input(ir, cases)
        for case, ws in zip(cases, out_words, strict=True):
            got = ws[0] | (ws[1] << 64)
            self.assertEqual(got, case, f"roundtrip mismatch for input {case:032x}")

    def test_bit_transpose_8x16_correctness(self) -> None:
        """Test BitTranspose produces correct transposition."""
        fwd = BitTranspose(x=Var("x"), lane_width=8, lanes=16)
        ir = TickIR(
            name="bt_fwd",
            inputs={"x": SimdType(lane_width=8, lanes=16)},
            state={},
            reset_state={},
            outputs={"y": SimdType(lane_width=16, lanes=8)},
            next_state={},
            output_exprs={"y": fwd},
        )
        cases = [
            0x000102030405060708090A0B0C0D0E0F,
            0xFFEEDDCCBBAA99887766554433221100,
        ]
        types = {"x": SimdType(lane_width=8, lanes=16)}
        out_words = self._compile_and_run_single_input(ir, cases)
        for case, ws in zip(cases, out_words, strict=True):
            got = ws[0] | (ws[1] << 64)
            expected = eval_expr(fwd, types, {"x": case})
            self.assertEqual(got, expected, f"transpose mismatch for input {case:032x}")
