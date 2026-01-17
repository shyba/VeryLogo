import unittest

from stc.superopt import superopt_expr
from stc.tick_ir import SimdType, Var, Xor, SimdXor


class TestSuperoptCostModel(unittest.TestCase):
    def test_superopt_prefers_explicit_simd_xor(self) -> None:
        t = SimdType(lane_width=4, lanes=2)
        types = {"x": t, "y": t}
        spec = Xor(a=Var("x"), b=Var("y"))

        out = superopt_expr(spec, types, max_nodes=3, timeout_ms=200)
        self.assertEqual(out, SimdXor(a=Var("x"), b=Var("y")))
