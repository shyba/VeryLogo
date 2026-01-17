import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stc.backend_x86_avx512_float import emit_x86_avx512_float_c
from stc.interp import eval_expr
from stc.tick_ir import SimdFFma, SimdType, TickIR, Var
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
    import struct

    return int.from_bytes(struct.pack("<f", float(x)), "little")


def _f64_bits(x: float) -> int:
    import struct

    return int.from_bytes(struct.pack("<d", float(x)), "little")


@unittest.skipUnless(
    shutil.which("cc")
    and platform.machine() == "x86_64"
    and _has_cpu_flag("avx512f")
    and _has_cpu_flag("fma"),
    "requires cc, x86_64, avx512f, and fma",
)
class TestBackendX86Avx512FloatFmaOptional(unittest.TestCase):
    def test_emit_and_run_avx512_float_fma_matches_interpreter(self) -> None:
        t32 = SimdType(lane_width=32, lanes=16)
        t64 = SimdType(lane_width=64, lanes=8)
        ir = TickIR(
            name="fma512",
            inputs={
                "x32": t32,
                "y32": t32,
                "z32": t32,
                "x64": t64,
                "y64": t64,
                "z64": t64,
            },
            outputs={"fma32": t32, "fma64": t64},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "fma32": SimdFFma(a=Var("x32"), b=Var("y32"), c=Var("z32")),
                "fma64": SimdFFma(a=Var("x64"), b=Var("y64"), c=Var("z64")),
            },
        )
        validate_tick_ir(ir)
        c = emit_x86_avx512_float_c(ir)
        self.assertIn("_mm512_fmadd_ps", c)
        self.assertIn("_mm512_fmadd_pd", c)

        def pack(bits: list[int], lane_width: int) -> int:
            out = 0
            for i, v in enumerate(bits):
                out |= int(v) << (lane_width * i)
            return out

        base32 = [_f32_bits(v) for v in [1.0, -2.0, 3.5, -4.25, 0.25, 10.0, -0.5, 2.0]]
        x32 = pack(base32 * 2, 32)
        y32 = pack(
            [_f32_bits(v) for v in [0.5, 1.25, -1.0, 2.0, -4.0, 0.125, 3.0, -0.75]] * 2,
            32,
        )
        z32 = pack(
            [_f32_bits(v) for v in [2.0, -1.0, 0.0, 1.0, 0.5, -2.0, 4.0, 0.25]] * 2, 32
        )

        x64 = pack(
            [_f64_bits(v) for v in [1.0, -2.0, 0.5, 10.0, 3.0, -1.0, 2.0, -0.25]], 64
        )
        y64 = pack(
            [_f64_bits(v) for v in [3.0, 0.25, -4.0, -0.125, 0.5, 2.0, -1.5, 4.0]], 64
        )
        z64 = pack(
            [_f64_bits(v) for v in [0.0, 1.0, -1.0, 2.0, 0.25, -0.5, 3.0, -4.0]], 64
        )

        env = {"x32": x32, "y32": y32, "z32": z32, "x64": x64, "y64": y64, "z64": z64}
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
                        "  uint64_t in[48];",
                        f"  uint64_t out[{out_words}];",
                        "  for (int i = 0; i < 48; i++) {",
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
                    "-mavx512f",
                    "-mfma",
                    "-o",
                    str(exe),
                    str(out_dir / "main.c"),
                    str(out_dir / "impl.c"),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

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

            types = dict(ir.inputs)
            for idx, name in enumerate(out_order):
                got = 0
                for widx in range(8):
                    got |= ws[idx * 8 + widx] << (64 * widx)
                exp = int(eval_expr(ir.output_exprs[name], types, env))
                self.assertEqual(got, exp)
