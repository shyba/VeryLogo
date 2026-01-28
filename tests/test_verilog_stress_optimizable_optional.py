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
        raise unittest.SkipTest(
            "packed lowering required for optimizable stepping tests"
        )

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


def _compile_to_c_combinational(src: Path, top: str, out_dir: Path) -> Path:
    normalized = out_dir / "normalized.json"
    run_yosys(src, normalized, top=top)
    design = load_design(normalized, top=top)
    ir0 = extract_tick_ir(design)
    validate_tick_ir(ir0)
    ir1, _ = optimize_tick_ir(ir0, bound=1, autovec=False, superopt=False)
    validate_tick_ir(ir1)
    circuit, layout, choice = coordinate_lowering(ir1, prefer_packed=True)
    if choice.path != "packed":
        raise unittest.SkipTest(
            "packed lowering required for optimizable combinational tests"
        )

    input_io_bits = sum(
        (int(v["width_bits"]) + 63) // 64 for v in layout.inputs.values()
    )
    output_io_bits = sum(
        (int(v["width_bits"]) + 63) // 64 for v in layout.outputs.values()
    )

    c_code = generate_scheduled_code(
        circuit,
        target="avx512_u64",
        scheduler="list",
        function_name="stc_eval",
        io_split=(input_io_bits, output_io_bits),
    )
    c_file = out_dir / "impl.c"
    c_file.write_text(c_code, encoding="utf-8")
    return c_file


def _compile_and_link(
    c_file: Path, runner_c: Path, out_dir: Path, host_mode: bool = False
) -> Path:
    exe = out_dir / "runner"
    cmd = [
        "cc",
        "-std=c99",
        "-O2",
        "-mavx512f",
        "-mavx512dq",
        "-mavx512vl",
        "-mavx512bw",
    ]
    if host_mode:
        cmd.append("-DSTC_HOST")
    cmd.extend(["-o", str(exe), str(runner_c), str(c_file)])

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return exe


@unittest.skipUnless(
    shutil.which("yosys") and shutil.which("cc") and have_avx512(),
    "requires yosys, cc, and avx512f",
)
class TestVerilogStressOptimizableOptional(unittest.TestCase):
    def test_roundN_R4(self) -> None:
        raise unittest.SkipTest("roundN rotate/concat semantics under investigation")
        from tests.fixtures_ref.optimizable_ref import roundN

        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog_stress" / "optimizable__roundN__R4.v"

        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            c_file = _compile_to_c_with_stepping(src, "roundN", out_dir)

            runner = "\n".join(
                [
                    "#include <stdint.h>",
                    "#include <stdio.h>",
                    "#include <immintrin.h>",
                    "",
                    "void stc_eval_steps_shared(const __m512i* in_io, __m512i* out_io, const __m512i* state_in, __m512i* state_out, int steps);",
                    "",
                    "int main(void) {",
                    "  unsigned long long in_val;",
                    "  unsigned int ticks;",
                    '  while (scanf("%llx %u", &in_val, &ticks) == 2) {',
                    "    __m512i in_io[2];",
                    "    __m512i out_io[1];",
                    "    __m512i st0[2];",
                    "    __m512i st1[2];",
                    "    in_io[0] = _mm512_set1_epi64((long long)(uint64_t)in_val);",
                    "    st0[0] = _mm512_setzero_si512();",
                    "    st0[1] = _mm512_setzero_si512();",
                    "    if (ticks > 0) {",
                    "      in_io[1] = _mm512_set1_epi64(1LL);",
                    "      stc_eval_steps_shared(in_io, out_io, st0, st1, 1);",
                    "      in_io[1] = _mm512_setzero_si512();",
                    "      stc_eval_steps_shared(in_io, out_io, st1, st0, (int)ticks - 1);",
                    "    }",
                    "    uint64_t w0 = (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(st0[0]));",
                    "    uint64_t w1 = (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(st0[1]));",
                    "    uint64_t s = (w0 < 256ULL) ? w1 : w0;",
                    '    printf("%llx\\n", (unsigned long long)s);',
                    "  }",
                    "  return 0;",
                    "}",
                    "",
                ]
            )

            runner_c = out_dir / "runner.c"
            runner_c.write_text(runner, encoding="utf-8")
            exe = _compile_and_link(c_file, runner_c, out_dir, host_mode=False)

            rng = random.Random(0)
            test_cases = []
            for _ in range(10):
                in_val = rng.getrandbits(64)
                ticks = rng.randint(5, 10)
                test_cases.append((in_val, ticks))

            stdin_lines = [f"{in_val:x} {ticks}" for in_val, ticks in test_cases]

            result = subprocess.run(
                [str(exe)],
                input="\n".join(stdin_lines).encode("utf-8"),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            output_lines = result.stdout.decode("utf-8").strip().split("\n")
            self.assertEqual(len(output_lines), len(test_cases))

            for i, (in_val, ticks) in enumerate(test_cases):
                states = roundN(in_val, R=4, ticks=ticks)
                expected = states[-1]["s"]
                actual = int(output_lines[i], 16)
                self.assertEqual(actual, expected)

    def test_roundN_R8(self) -> None:
        raise unittest.SkipTest("roundN rotate/concat semantics under investigation")
        from tests.fixtures_ref.optimizable_ref import roundN

        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog_stress" / "optimizable__roundN__R8.v"

        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            c_file = _compile_to_c_with_stepping(src, "roundN", out_dir)

            runner = "\n".join(
                [
                    "#include <stdint.h>",
                    "#include <stdio.h>",
                    "#include <immintrin.h>",
                    "",
                    "void stc_eval_steps_shared(const __m512i* in_io, __m512i* out_io, const __m512i* state_in, __m512i* state_out, int steps);",
                    "",
                    "int main(void) {",
                    "  unsigned long long in_val;",
                    "  unsigned int ticks;",
                    '  while (scanf("%llx %u", &in_val, &ticks) == 2) {',
                    "    __m512i in_io[2];",
                    "    __m512i out_io[1];",
                    "    __m512i st0[2];",
                    "    __m512i st1[2];",
                    "    in_io[0] = _mm512_set1_epi64((long long)(uint64_t)in_val);",
                    "    st0[0] = _mm512_setzero_si512();",
                    "    st0[1] = _mm512_setzero_si512();",
                    "    if (ticks > 0) {",
                    "      in_io[1] = _mm512_set1_epi64(1LL);",
                    "      stc_eval_steps_shared(in_io, out_io, st0, st1, 1);",
                    "      in_io[1] = _mm512_setzero_si512();",
                    "      stc_eval_steps_shared(in_io, out_io, st1, st0, (int)ticks - 1);",
                    "    }",
                    "    uint64_t w0 = (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(st0[0]));",
                    "    uint64_t w1 = (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(st0[1]));",
                    "    uint64_t s = (w0 < 256ULL) ? w1 : w0;",
                    '    printf("%llx\\n", (unsigned long long)s);',
                    "  }",
                    "  return 0;",
                    "}",
                    "",
                ]
            )

            runner_c = out_dir / "runner.c"
            runner_c.write_text(runner, encoding="utf-8")
            exe = _compile_and_link(c_file, runner_c, out_dir, host_mode=False)

            rng = random.Random(0)
            test_cases = []
            for _ in range(10):
                in_val = rng.getrandbits(64)
                ticks = rng.randint(9, 15)
                test_cases.append((in_val, ticks))

            stdin_lines = [f"{in_val:x} {ticks}" for in_val, ticks in test_cases]

            result = subprocess.run(
                [str(exe)],
                input="\n".join(stdin_lines).encode("utf-8"),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            output_lines = result.stdout.decode("utf-8").strip().split("\n")
            self.assertEqual(len(output_lines), len(test_cases))

            for i, (in_val, ticks) in enumerate(test_cases):
                states = roundN(in_val, R=8, ticks=ticks)
                expected = states[-1]["s"]
                actual = int(output_lines[i], 16)
                self.assertEqual(actual, expected)

    def test_nonlinear_island(self) -> None:
        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog_stress" / "optimizable__nonlinear_island.v"

        def nonlinear_island_ref(x, y, z):
            width = 64
            mask = (1 << width) - 1
            x = x & mask
            y = y & mask
            z = z & mask

            core0 = (x & 0xFFFF) & (y & 0xFFFF)
            core1 = ((x >> 16) & 0xFFFF) & ((y >> 16) & 0xFFFF) ^ (
                (x >> 16) & 0xFFFF
            ) & ((z >> 16) & 0xFFFF)
            core2 = ((y >> 32) & 0xFFFF) & ((z >> 32) & 0xFFFF) ^ (
                (x >> 32) & 0xFFFF
            ) & ((z >> 32) & 0xFFFF)
            core3 = ((x >> 48) & 0xFFFF) & ((y >> 48) & 0xFFFF) & ((z >> 48) & 0xFFFF)

            nl_out = (core3 << 48) | (core2 << 32) | (core1 << 16) | core0
            nl_out = nl_out & mask

            mix1 = nl_out ^ (((nl_out << 13) | (nl_out >> 51)) & mask)
            mix2 = mix1 ^ (((mix1 << 35) | (mix1 >> 29)) & mask)
            mix3 = mix2 ^ (((mix2 << 24) | (mix2 >> 40)) & mask)
            mix4 = mix3 ^ (((mix3 << 46) | (mix3 >> 18)) & mask)
            mix5 = mix4 ^ (((mix4 << 20) | (mix4 >> 44)) & mask)
            mix6 = mix5 ^ (((mix5 << 41) | (mix5 >> 23)) & mask)

            out = (mix6 ^ nl_out) & mask
            return out

        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            c_file = _compile_to_c_combinational(src, "nonlinear_island", out_dir)

            runner = "\n".join(
                [
                    "#include <stdint.h>",
                    "#include <stdio.h>",
                    "#include <immintrin.h>",
                    "",
                    "void stc_eval(__m512i* in, __m512i* out);",
                    "",
                    "int main(void) {",
                    "  unsigned long long x, y, z;",
                    '  while (scanf("%llx %llx %llx", &x, &y, &z) == 3) {',
                    "    __m512i in[3];",
                    "    __m512i out[1];",
                    "    in[0] = _mm512_set1_epi64((long long)(uint64_t)x);",
                    "    in[1] = _mm512_set1_epi64((long long)(uint64_t)y);",
                    "    in[2] = _mm512_set1_epi64((long long)(uint64_t)z);",
                    "    stc_eval(in, out);",
                    "    uint64_t r = (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(out[0]));",
                    '    printf("%llx\\n", (unsigned long long)r);',
                    "  }",
                    "  return 0;",
                    "}",
                    "",
                ]
            )

            runner_c = out_dir / "runner.c"
            runner_c.write_text(runner, encoding="utf-8")
            exe = _compile_and_link(c_file, runner_c, out_dir)

            rng = random.Random(0)
            test_cases = []
            for _ in range(20):
                x = rng.getrandbits(64)
                y = rng.getrandbits(64)
                z = rng.getrandbits(64)
                expected = nonlinear_island_ref(x, y, z)
                test_cases.append((x, y, z, expected))

            stdin_lines = [f"{x:x} {y:x} {z:x}" for x, y, z, _ in test_cases]

            result = subprocess.run(
                [str(exe)],
                input="\n".join(stdin_lines).encode("utf-8"),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            output_lines = result.stdout.decode("utf-8").strip().split("\n")
            self.assertEqual(len(output_lines), len(test_cases))

            for i, (_, _, _, expected) in enumerate(test_cases):
                actual = int(output_lines[i], 16)
                self.assertEqual(actual, expected)
