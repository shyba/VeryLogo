import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stc.backend_x86_avx512 import emit_x86_avx512_c
from stc.interp import eval_expr
from stc.tick_ir import (
    SimdPackSS16To8,
    SimdPackSS32To16,
    SimdPackUS16To8,
    SimdType,
    SimdUnpackHi,
    SimdUnpackLo,
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


def _pack_lanes(values: list[int], lane_width: int) -> int:
    mask = (1 << lane_width) - 1
    out = 0
    for i, v in enumerate(values):
        out |= (int(v) & mask) << (lane_width * i)
    return out


@unittest.skipUnless(
    shutil.which("cc") and platform.machine() == "x86_64" and _has_cpu_flag("avx512bw"),
    "requires cc, x86_64, and avx512bw",
)
class TestBackendX86Avx512PackUnpackOptional(unittest.TestCase):
    def test_emit_and_run_pack_unpack_matches_interpreter(self) -> None:
        t16 = SimdType(lane_width=16, lanes=32)
        t8 = SimdType(lane_width=8, lanes=64)
        t32 = SimdType(lane_width=32, lanes=16)

        ir = TickIR(
            name="pack_unpack",
            inputs={"a16": t16, "b16": t16, "a32": t32, "b32": t32},
            outputs={
                "unpacklo16": t16,
                "unpackhi16": t16,
                "packss16to8": t8,
                "packus16to8": t8,
                "packss32to16": t16,
            },
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "unpacklo16": SimdUnpackLo(a=Var("a16"), b=Var("b16")),
                "unpackhi16": SimdUnpackHi(a=Var("a16"), b=Var("b16")),
                "packss16to8": SimdPackSS16To8(a=Var("a16"), b=Var("b16")),
                "packus16to8": SimdPackUS16To8(a=Var("a16"), b=Var("b16")),
                "packss32to16": SimdPackSS32To16(a=Var("a32"), b=Var("b32")),
            },
        )
        validate_tick_ir(ir)
        c = emit_x86_avx512_c(ir)

        a16 = _pack_lanes(
            [
                -200,
                -129,
                -128,
                -1,
                0,
                1,
                127,
                128,
                200,
                300,
                -32768,
                32767,
                42,
                -42,
                255,
                -255,
            ]
            * 2,
            16,
        )
        b16 = _pack_lanes([i * 3 for i in range(32)], 16)
        a32 = _pack_lanes(
            [
                -40000,
                -32769,
                -32768,
                -1,
                0,
                1,
                32767,
                32768,
                40000,
                123456,
                -123456,
                0x7FFFFFFF,
                -0x80000000,
                999,
                -999,
                42,
            ],
            32,
        )
        b32 = _pack_lanes([i - 1000 for i in range(16)], 32)
        cases = [(a16, b16, a32, b32)]

        input_order = sorted(ir.inputs.keys())
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
                        "  unsigned long long w;",
                        "  uint64_t in[32];",
                        f"  uint64_t out[{out_words}];",
                        "  for (int i = 0; i < 32; i++) {",
                        '    if (scanf("%llx", &w) != 1) return 2;',
                        "    in[i] = (uint64_t)w;",
                        "  }",
                        "  stc_eval(in, out);",
                        f"  for (int i = 0; i < {out_words}; i++) {{",
                        f'    printf("%016llx%s", (unsigned long long)out[i], (i == {out_words - 1}) ? "\\n" : " ");',
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

            types = dict(ir.inputs)
            for a16, b16, a32, b32 in cases:
                env = {"a16": a16, "b16": b16, "a32": a32, "b32": b32}
                words: list[int] = []
                for name in input_order:
                    x = env[name]
                    words.extend([(x >> (64 * i)) & ((1 << 64) - 1) for i in range(8)])
                stdin = " ".join(f"{w:x}" for w in words) + "\n"
                p = subprocess.run(
                    [str(exe)],
                    input=stdin.encode("utf-8"),
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                ws = [int(w, 16) for w in p.stdout.decode("utf-8").strip().split()]
                self.assertEqual(len(ws), out_words)
                for idx, name in enumerate(out_order):
                    got = 0
                    for widx in range(8):
                        got |= ws[idx * 8 + widx] << (64 * widx)
                    exp = int(eval_expr(ir.output_exprs[name], types, env))
                    self.assertEqual(got, exp)
