import unittest

from stc.superopt import SuperoptError, superopt_expr
from stc.tick_ir import BitVecType, Var, Xor


class TestSuperoptCaps(unittest.TestCase):
    def test_superopt_honors_candidate_caps(self) -> None:
        bv8 = BitVecType(width=8)
        types = {"x": bv8, "y": bv8}
        spec = Xor(a=Var("x"), b=Var("y"))

        with self.assertRaises(SuperoptError):
            superopt_expr(
                spec,
                types,
                max_nodes=5,
                timeout_ms=200,
                max_total_candidates=1,
            )
