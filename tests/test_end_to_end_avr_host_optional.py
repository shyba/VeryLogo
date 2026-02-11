import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stc.cli import run_pipeline
from stc.interp import reset_state, tick
from stc.tick_ir import TickIR
from stc.io_map_bin import read_io_map_bin
from stc.tick_ir_bin2 import read_tick_ir_bin


@unittest.skipUnless(
    shutil.which("yosys") and shutil.which("cc"), "requires yosys and cc"
)
class TestEndToEndAvrHostOptional(unittest.TestCase):
    def test_verilog_to_avr_c_host_trace_matches_tick_ir(self) -> None:
        root = Path(__file__).resolve().parent.parent
        src = root / "fixtures" / "verilog" / "combinational_not.v"

        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d) / "out"
            run_pipeline(src, out_dir, top="top", bound=2)

            iom = read_io_map_bin(out_dir / "io_map.bin")
            in_lsb = int(iom.inputs["i"]["lsb"])
            out_lsb = int(iom.outputs["o"]["lsb"])

            reduced = read_tick_ir_bin(out_dir / "reduced_tick_ir.bin")
            st = reset_state(reduced)

            seq = [0, 1, 1, 0, 1, 0, 0, 1]
            expected = []
            for v in seq:
                st, outs = tick(reduced, st, {"i": v})
                expected.append(1 if outs["o"] else 0)

            main_c = out_dir / "host_main.c"
            main_c.write_text(
                "\n".join(
                    [
                        "#include <stdint.h>",
                        "#include <stdio.h>",
                        "#include <stdlib.h>",
                        "",
                        "uint8_t PINB = 0;",
                        "uint8_t PORTB = 0;",
                        "",
                        "void stc_reset(void);",
                        "void stc_tick(void);",
                        "",
                        "int main(void) {",
                        "  stc_reset();",
                        "  char buf[64];",
                        "  while (fgets(buf, sizeof buf, stdin)) {",
                        "    unsigned long v = strtoul(buf, 0, 0);",
                        "    PINB = (uint8_t)v;",
                        "    stc_tick();",
                        '    printf("%u\\n", (unsigned)PORTB);',
                        "  }",
                        "  return 0;",
                        "}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            exe = out_dir / "host_runner"
            subprocess.run(
                [
                    "cc",
                    "-std=c99",
                    "-O2",
                    "-DSTC_HOST",
                    "-o",
                    str(exe),
                    str(main_c),
                    str(out_dir / "avr.c"),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            stdin = "".join(f"{(v & 1) << in_lsb}\n" for v in seq).encode("utf-8")
            p = subprocess.run(
                [str(exe)],
                input=stdin,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            portbs = [
                int(x) for x in p.stdout.decode("utf-8").splitlines() if x.strip()
            ]
            got = [((v >> out_lsb) & 1) for v in portbs]
            self.assertEqual(got, expected)
