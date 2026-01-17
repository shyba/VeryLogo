import shutil
import unittest

from stc.verilator_golden import simulate_ticks


@unittest.skipUnless(shutil.which("verilator"), "verilator not installed")
class TestVerilatorGolden(unittest.TestCase):
    def test_combinational_not(self) -> None:
        verilog = "\n".join(
            [
                "module top(input wire clk, input wire rst, input wire i, output wire o);",
                "  assign o = ~i;",
                "endmodule",
            ]
        )

        outs = simulate_ticks(
            verilog=verilog,
            top="top",
            inputs=[{"i": 0}, {"i": 1}, {"i": 0}],
            outputs=["o"],
            rst_ticks=0,
        )

        self.assertEqual([o["o"] & 1 for o in outs], [1, 0, 1])
