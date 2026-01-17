import unittest

from stc.metrics import compute_metrics
from stc.reduce import reduce_tick_ir
from stc.tick_ir import And, BoolConst, BoolType, TickIR, Var


class TestMetricsAndReduce(unittest.TestCase):
    def test_reduce_decreases_ops(self) -> None:
        ir = TickIR(
            name="top",
            inputs={"i": BoolType()},
            outputs={"o": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": And(a=Var(name="i"), b=BoolConst(value=True))},
        )

        before = compute_metrics(ir)
        reduced = reduce_tick_ir(ir)
        after = compute_metrics(reduced)

        self.assertLess(after.ops_total, before.ops_total)
        self.assertEqual(reduced.output_exprs["o"], Var(name="i"))
