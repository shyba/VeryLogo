import unittest

from stc.metrics import compute_metrics
from stc.tick_ir import SimdAdd, SimdConst, SimdType, TickIR, Var


class TestMetricsSimd(unittest.TestCase):
    def test_compute_metrics_accepts_simd_state(self) -> None:
        ir = TickIR(
            name="t",
            inputs={
                "x": SimdType(lane_width=4, lanes=2),
                "y": SimdType(lane_width=4, lanes=2),
            },
            outputs={"o": SimdType(lane_width=4, lanes=2)},
            state={"s": SimdType(lane_width=8, lanes=1)},
            reset_state={"s": SimdConst(lane_width=8, lanes=1, value=0)},
            next_state={"s": Var("s")},
            output_exprs={"o": SimdAdd(a=Var("x"), b=Var("y"))},
        )
        m = compute_metrics(ir)
        self.assertEqual(m.state_bits, 8)
        self.assertGreaterEqual(m.ops_total, 1)
