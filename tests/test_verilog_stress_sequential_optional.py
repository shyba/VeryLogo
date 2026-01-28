import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stc.extract import extract_tick_ir
from stc.lowering_coordinator import coordinate_lowering
from stc.backend_sched import generate_scheduled_code
from stc.reduce import optimize_tick_ir
from stc.testing.native_x86_runner import have_avx512
from stc.tick_ir_validate import validate_tick_ir
from stc.yosys_frontend import run_yosys
from stc.yosys_json import load_design


def _compile_to_c_with_stepping(src: Path, top: str, out_dir: Path) -> Path:
    normalized = out_dir / "normalized.json"
    run_yosys(src, normalized, top=top)
    design = load_design(normalized, top=top)
    ir0 = extract_tick_ir(design)
    validate_tick_ir(ir0)
    ir1, _ = optimize_tick_ir(ir0, bound=8, autovec=False, superopt=False)
    validate_tick_ir(ir1)

    circuit, layout, choice = coordinate_lowering(ir1, prefer_packed=True)
    if choice.path != "packed":
        raise unittest.SkipTest("packed lowering required for sequential stress tests")

    input_io_words = sum(
        (int(v["width_bits"]) + 63) // 64 for v in layout.inputs.values()
    )
    output_io_words = sum(
        (int(v["width_bits"]) + 63) // 64 for v in layout.outputs.values()
    )

    c_code = generate_scheduled_code(
        circuit,
        target="avx512_u64",
        scheduler="list",
        function_name="stc_eval",
        io_split=(input_io_words, output_io_words),
    )
    c_file = out_dir / "impl.c"
    c_file.write_text(c_code, encoding="utf-8")
    return c_file


def _compile_and_link(c_file: Path, runner_c: Path, out_dir: Path) -> Path:
    exe = out_dir / "runner"
    subprocess.run(
        [
            "cc",
            "-std=c99",
            "-O2",
            "-mavx512f",
            "-mavx512dq",
            "-mavx512vl",
            "-mavx512bw",
            "-o",
            str(exe),
            str(runner_c),
            str(c_file),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return exe


@unittest.skipUnless(
    shutil.which("yosys") and shutil.which("cc") and have_avx512(),
    "requires yosys, cc, and avx512f",
)
class TestVerilogStressSequentialOptional(unittest.TestCase):
    def test_acc_fsm_limit_small(self) -> None:
        from tests.fixtures_ref.sequential_ref import acc_fsm

        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog_stress" / "sequential__acc_fsm.v"

        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            c_file = _compile_to_c_with_stepping(src, "acc_fsm", out_dir)

            runner = "\n".join(
                [
                    "#include <stdint.h>",
                    "#include <stdio.h>",
                    "#include <immintrin.h>",
                    "",
                    "void stc_eval_steps_shared(const __m512i* in_io, __m512i* out_io, const __m512i* state_in, __m512i* state_out, int steps);",
                    "",
                    "int main(void) {",
                    "  unsigned int limit, x, ticks;",
                    '  while (scanf("%u %u %u", &limit, &x, &ticks) == 3) {',
                    "    __m512i in_io[3];",
                    "    __m512i out_io[1];",
                    "    __m512i st0[2];",
                    "    __m512i st1[2];",
                    "    in_io[0] = _mm512_set1_epi64((long long)(uint64_t)limit);",
                    "    in_io[1] = _mm512_setzero_si512();",
                    "    in_io[2] = _mm512_set1_epi64((long long)(uint64_t)x);",
                    "    st0[0] = _mm512_setzero_si512();",
                    "    st0[1] = _mm512_setzero_si512();",
                    "    stc_eval_steps_shared(in_io, out_io, st0, st1, (int)ticks);",
                    "    uint64_t acc = (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(st1[0]));",
                    '    printf("%u\\n", (unsigned int)(acc & 0xFFFFFFFFu));',
                    "  }",
                    "  return 0;",
                    "}",
                    "",
                ]
            )

            runner_c = out_dir / "runner.c"
            runner_c.write_text(runner, encoding="utf-8")
            exe = _compile_and_link(c_file, runner_c, out_dir)

            test_cases = [
                (10, 0x12345678, 15),
                (5, 0xABCDEF01, 10),
                (20, 0x11111111, 25),
                (0, 0xFFFFFFFF, 5),
                (100, 0x87654321, 50),
            ]

            stdin_lines = [f"{limit} {x} {ticks}" for limit, x, ticks in test_cases]

            result = subprocess.run(
                [str(exe)],
                input="\n".join(stdin_lines).encode("utf-8"),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            output_lines = result.stdout.decode("utf-8").strip().split("\n")
            self.assertEqual(len(output_lines), len(test_cases))

            for i, (limit, x, ticks) in enumerate(test_cases):
                states = acc_fsm(limit, x, ticks)
                expected = states[-1]["acc"]
                actual = int(output_lines[i])
                self.assertEqual(actual, expected)

    def test_acc_fsm_limit_large(self) -> None:
        from tests.fixtures_ref.sequential_ref import acc_fsm

        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog_stress" / "sequential__acc_fsm.v"

        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            c_file = _compile_to_c_with_stepping(src, "acc_fsm", out_dir)

            runner = "\n".join(
                [
                    "#include <stdint.h>",
                    "#include <stdio.h>",
                    "#include <immintrin.h>",
                    "",
                    "void stc_eval_steps_shared(const __m512i* in_io, __m512i* out_io, const __m512i* state_in, __m512i* state_out, int steps);",
                    "",
                    "int main(void) {",
                    "  unsigned int limit, x, ticks;",
                    '  while (scanf("%u %u %u", &limit, &x, &ticks) == 3) {',
                    "    __m512i in_io[3];",
                    "    __m512i out_io[1];",
                    "    __m512i st0[2];",
                    "    __m512i st1[2];",
                    "    in_io[0] = _mm512_set1_epi64((long long)(uint64_t)limit);",
                    "    in_io[1] = _mm512_setzero_si512();",
                    "    in_io[2] = _mm512_set1_epi64((long long)(uint64_t)x);",
                    "    st0[0] = _mm512_setzero_si512();",
                    "    st0[1] = _mm512_setzero_si512();",
                    "    stc_eval_steps_shared(in_io, out_io, st0, st1, (int)ticks);",
                    "    uint64_t acc = (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(st1[0]));",
                    '    printf("%u\\n", (unsigned int)(acc & 0xFFFFFFFFu));',
                    "  }",
                    "  return 0;",
                    "}",
                    "",
                ]
            )

            runner_c = out_dir / "runner.c"
            runner_c.write_text(runner, encoding="utf-8")
            exe = _compile_and_link(c_file, runner_c, out_dir)

            test_cases = [
                (1000, 0x12345678, 1500),
                (500, 0xABCDEF01, 600),
            ]

            stdin_lines = [f"{limit} {x} {ticks}" for limit, x, ticks in test_cases]

            result = subprocess.run(
                [str(exe)],
                input="\n".join(stdin_lines).encode("utf-8"),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            output_lines = result.stdout.decode("utf-8").strip().split("\n")
            self.assertEqual(len(output_lines), len(test_cases))

            for i, (limit, x, ticks) in enumerate(test_cases):
                states = acc_fsm(limit, x, ticks)
                expected = states[-1]["acc"]
                actual = int(output_lines[i])
                self.assertEqual(actual, expected)
