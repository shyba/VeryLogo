import shutil
import unittest

from stc.bounded_equiv import check_equiv_smtbmc
from stc.cases import (
    case_combinational_not,
    case_fsm_toggle,
    case_temporal_pipeline_counter,
)
from stc.tick_ir import Add, BitVecConst, BitVecType, TickIR, Var


def _has_z3_module() -> bool:
    try:
        import z3  # noqa: F401
    except Exception:
        return False
    return True


def _has_venv_z3() -> bool:
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    return (root / ".venv" / "bin" / "z3").exists()


@unittest.skipUnless(
    shutil.which("yosys")
    and shutil.which("yosys-smtbmc")
    and _has_z3_module()
    and _has_venv_z3(),
    "requires yosys, yosys-smtbmc, and z3-solver",
)
class TestBoundedEquivOptional(unittest.TestCase):
    def test_equiv_comb_not(self) -> None:
        case = case_combinational_not()
        original = "\n".join(
            [
                "module top(input logic clk, input logic rst, input logic i, output logic o);",
                "  assign o = ~i;",
                "endmodule",
            ]
        )

        check_equiv_smtbmc(
            original_verilog=original,
            original_top="top",
            ir=case.ir,
            bound=3,
        )

    def test_equiv_fsm_toggle(self) -> None:
        case = case_fsm_toggle()
        original = "\n".join(
            [
                "module top(input logic clk, input logic rst, input logic i, output logic o);",
                "  logic s;",
                "  always_ff @(posedge clk) begin",
                "    if (rst) s <= 1'b0; else s <= s ^ i;",
                "  end",
                "  assign o = s;",
                "endmodule",
            ]
        )

        check_equiv_smtbmc(
            original_verilog=original,
            original_top="top",
            ir=case.ir,
            bound=6,
        )

    def test_equiv_counter_pipe(self) -> None:
        case = case_temporal_pipeline_counter()
        original = "\n".join(
            [
                "module top(input logic clk, input logic rst, output logic [2:0] o);",
                "  logic [2:0] c;",
                "  always_ff @(posedge clk) begin",
                "    if (rst) c <= 3'd0; else c <= c + 3'd1;",
                "  end",
                "  assign o = c;",
                "endmodule",
            ]
        )

        check_equiv_smtbmc(
            original_verilog=original,
            original_top="top",
            ir=case.ir,
            bound=6,
        )

    def test_equiv_comb_add_bitvec_input(self) -> None:
        ir = TickIR(
            name="comb_add",
            inputs={"a": BitVecType(width=4)},
            outputs={"o": BitVecType(width=4)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Add(a=Var(name="a"), b=BitVecConst(width=4, value=1))},
        )
        original = "\n".join(
            [
                "module top(input logic clk, input logic rst, input logic [3:0] a, output logic [3:0] o);",
                "  assign o = a + 4'd1;",
                "endmodule",
            ]
        )

        check_equiv_smtbmc(
            original_verilog=original,
            original_top="top",
            ir=ir,
            bound=3,
        )
