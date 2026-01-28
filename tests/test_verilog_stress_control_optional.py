import json
import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any

from stc.backend_sched import generate_scheduled_code
from stc.extract import extract_tick_ir
from stc.lowering_coordinator import coordinate_lowering
from stc.packed_circuit import PackedCircuitState, eval_packed_circuit_words
from stc.reduce import optimize_tick_ir
from stc.testing.native_x86_runner import have_avx512
from stc.tick_ir_validate import validate_tick_ir
from stc.yosys_frontend import run_yosys
from stc.yosys_json import load_design


def _dump_debug_artifacts(
    out_dir: Path,
    circuit: PackedCircuitState,
    layout: Any,
    c_code: str,
    inputs: list[int],
    expected: list[int],
    packed_eval: list[int],
    native_output: list[int] | None = None,
) -> None:
    debug_dir = out_dir / "debug_artifacts"
    debug_dir.mkdir(exist_ok=True)

    with open(debug_dir / "packed_circuit_state.json", "w") as f:
        json.dump(circuit.to_dict(), f, indent=2)

    with open(debug_dir / "packed_word_layout.json", "w") as f:
        json.dump(layout, f, indent=2)

    with open(debug_dir / "impl.c", "w") as f:
        f.write(c_code)

    with open(debug_dir / "inputs.json", "w") as f:
        json.dump([hex(x) for x in inputs], f, indent=2)

    with open(debug_dir / "expected.json", "w") as f:
        json.dump([hex(x) for x in expected], f, indent=2)

    with open(debug_dir / "packed_eval.json", "w") as f:
        json.dump([hex(x) for x in packed_eval], f, indent=2)

    if native_output is not None:
        with open(debug_dir / "native_output.json", "w") as f:
            json.dump([hex(x) for x in native_output], f, indent=2)

    print(f"Debug artifacts saved to: {debug_dir}")


def _compile_to_c(
    src: Path, top: str, out_dir: Path
) -> tuple[Path, PackedCircuitState, Any, str]:
    normalized = out_dir / "normalized.json"
    run_yosys(src, normalized, top=top)
    design = load_design(normalized, top=top)
    ir0 = extract_tick_ir(design)
    validate_tick_ir(ir0)
    ir1, _ = optimize_tick_ir(ir0, bound=1, autovec=False, superopt=False)
    validate_tick_ir(ir1)

    circuit, layout, choice = coordinate_lowering(ir1, prefer_packed=True)
    if choice.path != "packed":
        raise unittest.SkipTest("packed lowering required for this stress test")

    if choice.path == "packed":
        input_io_bits = sum(
            (int(v["width_bits"]) + 63) // 64 for v in layout.inputs.values()
        )
        output_io_bits = sum(
            (int(v["width_bits"]) + 63) // 64 for v in layout.outputs.values()
        )
    else:
        input_io_bits = sum(int(v["width"]) for v in layout.inputs.values())
        output_io_bits = sum(int(v["width"]) for v in layout.outputs.values())

    c_code = generate_scheduled_code(
        circuit,
        target="avx512_u64",
        scheduler="list",
        function_name="stc_eval",
        io_split=(input_io_bits, output_io_bits),
    )
    c_file = out_dir / "impl.c"
    c_file.write_text(c_code, encoding="utf-8")
    return c_file, circuit, layout, c_code


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
class TestVerilogStressControlOptional(unittest.TestCase):
    def test_pmux16(self) -> None:
        from tests.fixtures_ref.control_ref import pmux16

        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog_stress" / "control__pmux16__W64.v"

        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            c_file, circuit, layout, c_code = _compile_to_c(src, "pmux16", out_dir)

            input_mapping = []
            for name in sorted(layout.inputs.keys()):
                info = layout.inputs[name]
                input_mapping.append((name, info["lsw"]))

            runner_lines = [
                "#include <stdint.h>",
                "#include <stdio.h>",
                "#include <immintrin.h>",
                "",
                "void stc_eval(__m512i* in, __m512i* out);",
                "",
                "int main(void) {",
                "  unsigned int sel;",
                "  unsigned long long a[16];",
                '  while (scanf("%u", &sel) == 1) {',
                "    for (int i = 0; i < 16; i++) {",
                '      if (scanf("%llx", &a[i]) != 1) return 1;',
                "    }",
                "    __m512i in[17];",
                "    __m512i out[1];",
            ]

            for name, word_idx in input_mapping:
                if name == "sel":
                    runner_lines.append(
                        f"    in[{word_idx}] = _mm512_set1_epi64((long long)sel);"
                    )
                else:
                    input_idx = int(name[1:])
                    runner_lines.append(
                        f"    in[{word_idx}] = _mm512_set1_epi64((long long)a[{input_idx}]);"
                    )

            runner_lines.extend(
                [
                    "    stc_eval(in, out);",
                    '    printf("%llx\\n", (unsigned long long)_mm_cvtsi128_si64(_mm512_castsi512_si128(out[0])));',
                    "  }",
                    "  return 0;",
                    "}",
                    "",
                ]
            )

            runner = "\n".join(runner_lines)

            runner_c = out_dir / "runner.c"
            runner_c.write_text(runner, encoding="utf-8")
            exe = _compile_and_link(c_file, runner_c, out_dir)

            rng = random.Random(0)
            test_cases = []

            input_names = sorted([k for k in layout.inputs.keys() if k != "sel"])
            for _ in range(50):
                sel = rng.randint(0, 15)
                inputs = [rng.getrandbits(64) for _ in range(16)]
                expected = pmux16(sel, *inputs)
                test_cases.append((sel, inputs, expected))

            stdin_lines = []
            for sel, inputs, _ in test_cases:
                stdin_lines.append(f"{sel} " + " ".join(f"{x:x}" for x in inputs))

            result = subprocess.run(
                [str(exe)],
                input="\n".join(stdin_lines).encode("utf-8"),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            output_lines = result.stdout.decode("utf-8").strip().split("\n")
            self.assertEqual(len(output_lines), len(test_cases))

            for i, (sel, inputs, expected) in enumerate(test_cases):
                actual = int(output_lines[i], 16)
                if actual != expected:
                    ordered_inputs = [0] * circuit.input_words
                    for name, word_idx in input_mapping:
                        if name == "sel":
                            ordered_inputs[word_idx] = sel
                        else:
                            input_idx = int(name[1:])
                            ordered_inputs[word_idx] = inputs[input_idx]
                    packed_eval_result = eval_packed_circuit_words(
                        circuit, ordered_inputs
                    )
                    packed_eval = packed_eval_result[0] if packed_eval_result else 0

                    _dump_debug_artifacts(
                        out_dir,
                        circuit,
                        layout.to_dict() if hasattr(layout, "to_dict") else layout,
                        c_code,
                        ordered_inputs,
                        [expected],
                        [packed_eval],
                        [actual],
                    )

                    if packed_eval != expected:
                        self.fail(
                            f"PACKED LOWERING BUG: Test case {i}\n"
                            f"  Inputs: sel={sel}, inputs={[hex(x) for x in inputs[:3]]}\n"
                            f"  Reference: {expected:#x}\n"
                            f"  PackedCircuit eval: {packed_eval:#x}\n"
                            f"  Native output: {actual:#x}\n"
                            f"  Debug artifacts saved to: {out_dir / 'debug_artifacts'}"
                        )
                    else:
                        self.fail(
                            f"EMITTER/SCHEDULER/REGALLOC BUG: Test case {i}\n"
                            f"  Inputs: sel={sel}, inputs={[hex(x) for x in inputs[:3]]}\n"
                            f"  Reference: {expected:#x}\n"
                            f"  PackedCircuit eval: {packed_eval:#x}\n"
                            f"  Native output: {actual:#x}\n"
                            f"  Debug artifacts saved to: {out_dir / 'debug_artifacts'}"
                        )
                self.assertEqual(actual, expected)

    def test_pmux32(self) -> None:
        from tests.fixtures_ref.control_ref import pmux32

        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog_stress" / "control__pmux32__W64.v"

        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            c_file, circuit, layout, c_code = _compile_to_c(src, "pmux32", out_dir)

            input_mapping = []
            for name in sorted(layout.inputs.keys()):
                info = layout.inputs[name]
                input_mapping.append((name, info["lsw"]))

            runner_lines = [
                "#include <stdint.h>",
                "#include <stdio.h>",
                "#include <immintrin.h>",
                "",
                "void stc_eval(__m512i* in, __m512i* out);",
                "",
                "int main(void) {",
                "  unsigned int sel;",
                "  unsigned long long a[32];",
                '  while (scanf("%u", &sel) == 1) {',
                "    for (int i = 0; i < 32; i++) {",
                '      if (scanf("%llx", &a[i]) != 1) return 1;',
                "    }",
                "    __m512i in[33];",
                "    __m512i out[1];",
            ]

            for name, word_idx in input_mapping:
                if name == "sel":
                    runner_lines.append(
                        f"    in[{word_idx}] = _mm512_set1_epi64((long long)sel);"
                    )
                else:
                    input_idx = int(name[1:])
                    runner_lines.append(
                        f"    in[{word_idx}] = _mm512_set1_epi64((long long)a[{input_idx}]);"
                    )

            runner_lines.extend(
                [
                    "    stc_eval(in, out);",
                    '    printf("%llx\\n", (unsigned long long)_mm_cvtsi128_si64(_mm512_castsi512_si128(out[0])));',
                    "  }",
                    "  return 0;",
                    "}",
                    "",
                ]
            )

            runner = "\n".join(runner_lines)

            runner_c = out_dir / "runner.c"
            runner_c.write_text(runner, encoding="utf-8")
            exe = _compile_and_link(c_file, runner_c, out_dir)

            rng = random.Random(0)
            test_cases = []
            for _ in range(50):
                sel = rng.randint(0, 31)
                inputs = [rng.getrandbits(64) for _ in range(32)]
                expected = pmux32(sel, *inputs)
                test_cases.append((sel, inputs, expected))

            stdin_lines = []
            for sel, inputs, _ in test_cases:
                stdin_lines.append(f"{sel} " + " ".join(f"{x:x}" for x in inputs))

            result = subprocess.run(
                [str(exe)],
                input="\n".join(stdin_lines).encode("utf-8"),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            output_lines = result.stdout.decode("utf-8").strip().split("\n")
            self.assertEqual(len(output_lines), len(test_cases))

            for i, (sel, inputs, expected) in enumerate(test_cases):
                actual = int(output_lines[i], 16)
                if actual != expected:
                    ordered_inputs = [0] * circuit.input_words
                    for name, word_idx in input_mapping:
                        if name == "sel":
                            ordered_inputs[word_idx] = sel
                        else:
                            input_idx = int(name[1:])
                            ordered_inputs[word_idx] = inputs[input_idx]
                    packed_eval_result = eval_packed_circuit_words(
                        circuit, ordered_inputs
                    )
                    packed_eval = packed_eval_result[0] if packed_eval_result else 0

                    _dump_debug_artifacts(
                        out_dir,
                        circuit,
                        layout.to_dict() if hasattr(layout, "to_dict") else layout,
                        c_code,
                        ordered_inputs,
                        [expected],
                        [packed_eval],
                        [actual],
                    )

                    if packed_eval != expected:
                        self.fail(
                            f"PACKED LOWERING BUG: Test case {i}\n"
                            f"  Inputs: sel={sel}, inputs={[hex(x) for x in inputs[:3]]}\n"
                            f"  Reference: {expected:#x}\n"
                            f"  PackedCircuit eval: {packed_eval:#x}\n"
                            f"  Native output: {actual:#x}\n"
                            f"  Debug artifacts saved to: {out_dir / 'debug_artifacts'}"
                        )
                    else:
                        self.fail(
                            f"EMITTER/SCHEDULER/REGALLOC BUG: Test case {i}\n"
                            f"  Inputs: sel={sel}, inputs={[hex(x) for x in inputs[:3]]}\n"
                            f"  Reference: {expected:#x}\n"
                            f"  PackedCircuit eval: {packed_eval:#x}\n"
                            f"  Native output: {actual:#x}\n"
                            f"  Debug artifacts saved to: {out_dir / 'debug_artifacts'}"
                        )
                self.assertEqual(actual, expected)

    def test_mux_reconverge(self) -> None:
        from tests.fixtures_ref.control_ref import mux_reconverge

        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog_stress" / "control__mux_reconverge__W64.v"

        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            c_file, circuit, layout, c_code = _compile_to_c(
                src, "mux_reconverge", out_dir
            )

            runner = "\n".join(
                [
                    "#include <stdint.h>",
                    "#include <stdio.h>",
                    "#include <immintrin.h>",
                    "",
                    "void stc_eval(__m512i* in, __m512i* out);",
                    "",
                    "int main(void) {",
                    "  unsigned long long a, b, c, d;",
                    "  unsigned int sel0, sel1;",
                    '  while (scanf("%llx %llx %llx %llx %u %u", &a, &b, &c, &d, &sel0, &sel1) == 6) {',
                    "    __m512i in[6];",
                    "    __m512i out[1];",
                    "    in[0] = _mm512_set1_epi64((long long)a);",
                    "    in[1] = _mm512_set1_epi64((long long)b);",
                    "    in[2] = _mm512_set1_epi64((long long)c);",
                    "    in[3] = _mm512_set1_epi64((long long)d);",
                    "    in[4] = _mm512_set1_epi64((long long)sel0);",
                    "    in[5] = _mm512_set1_epi64((long long)sel1);",
                    "    stc_eval(in, out);",
                    '    printf("%llx\\n", (unsigned long long)_mm_cvtsi128_si64(_mm512_castsi512_si128(out[0])));',
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
            for _ in range(30):
                a = rng.getrandbits(64)
                b = rng.getrandbits(64)
                c = rng.getrandbits(64)
                d = rng.getrandbits(64)
                sel0 = rng.randint(0, 3)
                sel1 = rng.randint(0, 3)
                expected = mux_reconverge(sel0, sel1, a, b, c, d)
                test_cases.append((a, b, c, d, sel0, sel1, expected))

            stdin_lines = [
                f"{a:x} {b:x} {c:x} {d:x} {sel0} {sel1}"
                for a, b, c, d, sel0, sel1, _ in test_cases
            ]

            result = subprocess.run(
                [str(exe)],
                input="\n".join(stdin_lines).encode("utf-8"),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            output_lines = result.stdout.decode("utf-8").strip().split("\n")
            self.assertEqual(len(output_lines), len(test_cases))

            for i, (a, b, c, d, sel0, sel1, expected) in enumerate(test_cases):
                actual = int(output_lines[i], 16)
                if actual != expected:
                    inputs = [a, b, c, d, sel0, sel1]
                    packed_eval_result = eval_packed_circuit_words(circuit, inputs)
                    packed_eval = packed_eval_result[0] if packed_eval_result else 0

                    _dump_debug_artifacts(
                        out_dir,
                        circuit,
                        layout.to_dict() if hasattr(layout, "to_dict") else layout,
                        c_code,
                        inputs,
                        [expected],
                        [packed_eval],
                        [actual],
                    )

                    if packed_eval != expected:
                        self.fail(
                            f"PACKED LOWERING BUG: Test case {i}\n"
                            f"  Inputs: a={a:#x}, b={b:#x}, c={c:#x}, d={d:#x}, sel0={sel0}, sel1={sel1}\n"
                            f"  Reference: {expected:#x}\n"
                            f"  PackedCircuit eval: {packed_eval:#x}\n"
                            f"  Native output: {actual:#x}\n"
                            f"  Debug artifacts saved to: {out_dir / 'debug_artifacts'}"
                        )
                    else:
                        self.fail(
                            f"EMITTER/SCHEDULER/REGALLOC BUG: Test case {i}\n"
                            f"  Inputs: a={a:#x}, b={b:#x}, c={c:#x}, d={d:#x}, sel0={sel0}, sel1={sel1}\n"
                            f"  Reference: {expected:#x}\n"
                            f"  PackedCircuit eval: {packed_eval:#x}\n"
                            f"  Native output: {actual:#x}\n"
                            f"  Debug artifacts saved to: {out_dir / 'debug_artifacts'}"
                        )
                self.assertEqual(actual, expected)
