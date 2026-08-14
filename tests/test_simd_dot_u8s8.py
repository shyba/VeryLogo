import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import z3

from stc.backend_x86_auto import AutoBackendError, emit_x86_auto_c
from stc.backend_x86_avx512 import emit_x86_avx512_c
from stc.interp import eval_expr, infer_type
from stc.reduce import reduce_expr
from stc.replace import replace_vars
from stc.tick_ir import (
    SimdDotU8S8AccI32,
    SimdType,
    TickIR,
    Var,
)
from stc.tick_ir_bin2 import read_tick_ir_bin, write_tick_ir_bin
from stc.tick_ir_classify_arith import classify_arithmetic
from stc.tick_ir_to_verilog import emit_verilog
from stc.tick_ir_validate import validate_tick_ir
from stc.z3_encode import encode_expr


def _has_cpu_flag(flag: str) -> bool:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    aliases = {flag, flag.replace("avx512", "avx512_")}
    return any(
        line.startswith("flags") and aliases.intersection(line.split())
        for line in text.splitlines()
    )


class TestSimdDotU8S8(unittest.TestCase):
    def _ir(self) -> TickIR:
        byte_t = SimdType(lane_width=8, lanes=64)
        acc_t = SimdType(lane_width=32, lanes=16)
        return TickIR(
            name="dot_u8s8",
            inputs={"a": byte_t, "b": byte_t, "acc": acc_t},
            outputs={"out": acc_t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "out": SimdDotU8S8AccI32(a=Var("a"), b=Var("b"), acc=Var("acc"))
            },
        )

    def test_interpreter_and_validation(self) -> None:
        ir = self._ir()
        validate_tick_ir(ir)
        expr = ir.output_exprs["out"]
        self.assertEqual(infer_type(expr, ir.inputs), SimdType(32, 16))

        a_bytes = [1, 2, 3, 4] + [0] * 60
        b_bytes = [1, 254, 3, 252] + [0] * 60
        a = sum(v << (8 * i) for i, v in enumerate(a_bytes))
        b = sum(v << (8 * i) for i, v in enumerate(b_bytes))
        self.assertEqual(
            eval_expr(expr, ir.inputs, {"a": a, "b": b, "acc": 7}), 0xFFFFFFFD
        )
        z3_vars: dict[str, z3.ExprRef] = {}
        encoded = encode_expr(expr, ir.inputs, z3_vars)
        solver = z3.Solver()
        solver.add(z3_vars["a"] == a, z3_vars["b"] == b, z3_vars["acc"] == 7)
        self.assertEqual(solver.check(), z3.sat)
        self.assertEqual(
            solver.model().eval(encoded).as_long(),
            int(eval_expr(expr, ir.inputs, {"a": a, "b": b, "acc": 7})),
        )

    def test_all_ir_boundaries_roundtrip(self) -> None:
        ir = self._ir()
        expr = ir.output_exprs["out"]
        self.assertEqual(expr, SimdDotU8S8AccI32(Var("a"), Var("b"), Var("acc")))
        self.assertEqual(
            replace_vars(expr, {"a": Var("b")}),
            SimdDotU8S8AccI32(Var("b"), Var("b"), Var("acc")),
        )
        self.assertEqual(reduce_expr(expr, ir.inputs), expr)
        self.assertEqual(expr.to_dict()["kind"], "simd_dot_u8s8_acc_i32")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dot.tickir"
            write_tick_ir_bin(ir, str(path))
            self.assertEqual(read_tick_ir_bin(str(path)).to_dict(), ir.to_dict())

        encoded = encode_expr(expr, ir.inputs)
        self.assertIsInstance(encoded, z3.BitVecRef)
        self.assertEqual(encoded.size(), 512)

        classified, _ = classify_arithmetic(ir)
        self.assertEqual(classified.to_dict(), ir.to_dict())
        verilog = emit_verilog(ir)
        self.assertIn("$unsigned", verilog)
        self.assertIn("$signed", verilog)

    def test_auto_backend_requires_vnni(self) -> None:
        ir = self._ir()
        with self.assertRaises(AutoBackendError):
            emit_x86_auto_c(ir, flags={"avx512f"})
        generated = emit_x86_auto_c(ir, flags={"avx512f", "avx512vnni"})
        self.assertIn("_mm512_dpbusd_epi32(acc, a, b)", generated)


@unittest.skipUnless(
    shutil.which("cc")
    and platform.machine() == "x86_64"
    and _has_cpu_flag("avx512f")
    and _has_cpu_flag("avx512vnni"),
    "requires cc, x86_64, avx512f, and avx512vnni",
)
class TestSimdDotU8S8Avx512(unittest.TestCase):
    def test_compiled_instruction_matches_interpreter(self) -> None:
        byte_t = SimdType(lane_width=8, lanes=64)
        acc_t = SimdType(lane_width=32, lanes=16)
        ir = TickIR(
            name="dot_u8s8",
            inputs={"a": byte_t, "b": byte_t, "acc": acc_t},
            outputs={"out": acc_t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"out": SimdDotU8S8AccI32(Var("a"), Var("b"), Var("acc"))},
        )
        validate_tick_ir(ir)
        generated = emit_x86_avx512_c(ir)
        a_values = [1, 2, 3, 4] + [0] * 60
        b_values = [1, 254, 3, 252] + [0] * 60
        a = sum(v << (8 * i) for i, v in enumerate(a_values))
        b = sum(v << (8 * i) for i, v in enumerate(b_values))
        acc = 7
        expected = int(
            eval_expr(ir.output_exprs["out"], ir.inputs, {"a": a, "b": b, "acc": acc})
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "impl.c").write_text(generated, encoding="utf-8")
            (root / "main.c").write_text(
                "#include <stdint.h>\n"
                "#include <stdio.h>\n"
                "void stc_eval(const uint64_t*, uint64_t*);\n"
                "int main(void) {\n"
                "  uint64_t in[24] = {0}; uint64_t out[8] = {0};\n"
                f"  in[0] = 0x{a & ((1 << 64) - 1):016x}ULL;\n"
                f"  in[8] = 0x{acc:016x}ULL;\n"
                f"  in[16] = 0x{b & ((1 << 64) - 1):016x}ULL;\n"
                '  stc_eval(in, out); printf("%016llx\\n", (unsigned long long)out[0]);\n'
                "  return 0;\n"
                "}\n",
                encoding="utf-8",
            )
            executable = root / "runner"
            subprocess.run(
                [
                    "cc",
                    "-std=c99",
                    "-O2",
                    "-mavx512f",
                    "-mavx512vnni",
                    "-o",
                    str(executable),
                    str(root / "main.c"),
                    str(root / "impl.c"),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            result = subprocess.run(
                [str(executable)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        self.assertEqual(int(result.stdout.strip(), 16), expected)
