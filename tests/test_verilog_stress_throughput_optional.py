import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stc.backend_sched import generate_scheduled_code
from stc.extract import extract_tick_ir
from stc.lowering_coordinator import coordinate_lowering
from stc.testing.native_x86_runner import have_avx512
from stc.tick_ir_validate import validate_tick_ir
from stc.yosys_frontend import run_yosys
from stc.yosys_json import load_design


def _compile_to_c(src: Path, top: str, out_dir: Path) -> Path:
    normalized = out_dir / "normalized.json"
    run_yosys(src, normalized, top=top)
    design = load_design(normalized, top=top)
    ir0 = extract_tick_ir(design)
    validate_tick_ir(ir0)
    # Throughput stress fixtures are intentionally large. Keep this harness
    # close to extraction to avoid spending time in generic optimization
    # passes that can dominate test runtime.
    ir1 = ir0

    circuit, layout, choice = coordinate_lowering(ir1, prefer_packed=True)
    if choice.path != "packed":
        raise unittest.SkipTest("packed lowering required for throughput stress tests")

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
class TestVerilogStressThroughputOptional(unittest.TestCase):
    def test_vec_add_N256(self) -> None:
        from tests.fixtures_ref.throughput_ref import vec_add

        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog_stress" / "throughput__vec_add__N256.v"
        N = 256

        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            c_file = _compile_to_c(src, "vec_add", out_dir)

            runner = "\n".join(
                [
                    "#include <stdint.h>",
                    "#include <stdio.h>",
                    "#include <string.h>",
                    "#include <immintrin.h>",
                    "",
                    "void stc_eval(__m512i* in, __m512i* out);",
                    "",
                    f"#define N {N}",
                    "#define N_WORDS ((N * 32 + 63) / 64)",
                    "",
                    "int main(void) {",
                    "  uint64_t a[N_WORDS];",
                    "  uint64_t b[N_WORDS];",
                    "  uint64_t c[N_WORDS];",
                    "  __m512i in[N_WORDS * 2];",
                    "  __m512i out[N_WORDS];",
                    "  char line[1000000];",
                    "  while (fgets(line, sizeof(line), stdin)) {",
                    "    char* p = line;",
                    "    for (int i = 0; i < N_WORDS; i++) {",
                    '      if (sscanf(p, "%llx", (unsigned long long*)&a[i]) != 1) return 1;',
                    "      while (*p == ' ') p++;",
                    "      while (*p != ' ' && *p != '\\n' && *p != 0) p++;",
                    "    }",
                    "    for (int i = 0; i < N_WORDS; i++) {",
                    '      if (sscanf(p, "%llx", (unsigned long long*)&b[i]) != 1) return 1;',
                    "      while (*p == ' ') p++;",
                    "      while (*p != ' ' && *p != '\\n' && *p != 0) p++;",
                    "    }",
                    "    for (int i = 0; i < N_WORDS; i++) in[i] = _mm512_set1_epi64((long long)a[i]);",
                    "    for (int i = 0; i < N_WORDS; i++) in[N_WORDS + i] = _mm512_set1_epi64((long long)b[i]);",
                    "    stc_eval(in, out);",
                    "    for (int i = 0; i < N_WORDS; i++) c[i] = (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(out[i]));",
                    "    for (int i = 0; i < N_WORDS; i++) {",
                    "      printf(\"%llx%c\", (unsigned long long)c[i], i == N_WORDS - 1 ? '\\n' : ' ');",
                    "    }",
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
            cases = []
            for _ in range(10):
                a = rng.getrandbits(32 * N)
                b = rng.getrandbits(32 * N)
                expected = vec_add(a, b, N)
                cases.append((a, b, expected))

            stdin_lines = []
            for a, b, _ in cases:
                words_a = []
                words_b = []
                for i in range((N * 32 + 63) // 64):
                    words_a.append((a >> (64 * i)) & ((1 << 64) - 1))
                    words_b.append((b >> (64 * i)) & ((1 << 64) - 1))
                stdin_lines.append(
                    " ".join(f"{w:x}" for w in words_a)
                    + " "
                    + " ".join(f"{w:x}" for w in words_b)
                )

            result = subprocess.run(
                [str(exe)],
                input="\n".join(stdin_lines).encode("utf-8"),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            output_lines = result.stdout.decode("utf-8").strip().split("\n")
            self.assertEqual(len(output_lines), len(cases))

            for i, (_, _, expected) in enumerate(cases):
                words = [int(w, 16) for w in output_lines[i].split()]
                actual = 0
                for j, w in enumerate(words):
                    actual |= w << (64 * j)
                mask = (1 << (N * 32)) - 1
                self.assertEqual(actual & mask, expected & mask)

    def test_xor_diamonds(self) -> None:
        def diamond_node(a, b, c, d):
            mask = 0xFFFFFFFFFFFFFFFF
            return ((a ^ b) ^ (c ^ d)) ^ ((a ^ c) ^ (b ^ d)) & mask

        def xor_diamonds_ref(seed, depth):
            mask = 0xFFFFFFFFFFFFFFFF
            level = seed & mask
            for _ in range(depth):
                x = level
                rot1 = ((x << 1) | (x >> 63)) & mask
                rot2 = ((x << 2) | (x >> 62)) & mask
                rot3 = ((x << 3) | (x >> 61)) & mask
                rot4 = ((x << 4) | (x >> 60)) & mask

                d1 = diamond_node(x, rot1, rot2, rot3)
                d2 = diamond_node(rot1, rot2, rot3, rot4)
                d3 = diamond_node(x, rot2, rot4, d1)
                d4 = diamond_node(rot1, rot3, d1, d2)

                reconverge1 = (d1 ^ d2 ^ d3) & mask
                reconverge2 = (d2 ^ d3 ^ d4) & mask
                reconverge3 = (d1 ^ d3 ^ d4) & mask
                reconverge4 = (d1 ^ d2 ^ d4) & mask

                level = (reconverge1 ^ reconverge2 ^ reconverge3 ^ reconverge4) & mask

            return level

        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog_stress" / "throughput__xor_diamonds__D6.v"

        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            c_file = _compile_to_c(src, "xor_diamonds", out_dir)

            runner = "\n".join(
                [
                    "#include <stdint.h>",
                    "#include <stdio.h>",
                    "#include <immintrin.h>",
                    "",
                    "void stc_eval(__m512i* in, __m512i* out);",
                    "",
                    "int main(void) {",
                    "  unsigned long long seed;",
                    '  while (scanf("%llx", &seed) == 1) {',
                    "    __m512i in[1];",
                    "    __m512i out[1];",
                    "    in[0] = _mm512_set1_epi64((long long)seed);",
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
            cases = []
            for _ in range(20):
                seed = rng.getrandbits(64)
                expected = xor_diamonds_ref(seed, 6)
                cases.append((seed, expected))

            stdin = "\n".join(f"{seed:x}" for seed, _ in cases)
            result = subprocess.run(
                [str(exe)],
                input=stdin.encode("utf-8"),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            output_lines = result.stdout.decode("utf-8").strip().split("\n")
            self.assertEqual(len(output_lines), len(cases))

            for i, (_, expected) in enumerate(cases):
                actual = int(output_lines[i], 16)
                self.assertEqual(actual, expected)
