import unittest

from stc.reduce import optimize_tick_ir
from stc.tick_ir import BitVecType, TickIR, Var, Xor


class TestOptimizeSuperopt(unittest.TestCase):
    def test_optimize_can_superopt_xor_cancellation(self) -> None:
        bv8 = BitVecType(width=8)
        types = {"x": bv8, "y": bv8}
        spec = Xor(a=Var("x"), b=Xor(a=Var("x"), b=Var("y")))

        ir = TickIR(
            name="t",
            inputs=types,
            outputs={"o": bv8},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": spec},
        )

        out = optimize_tick_ir(
            ir,
            bound=1,
            superopt=True,
            superopt_max_nodes=1,
            superopt_timeout_ms=200,
        )
        self.assertEqual(out.output_exprs["o"], Var("y"))
