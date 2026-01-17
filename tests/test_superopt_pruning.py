import unittest

from stc.superopt import SuperoptError, SuperoptStats, superopt_expr
from stc.tick_ir import Add, BitVecConst, BitVecType, Shl, Var


class TestSuperoptPruning(unittest.TestCase):
    def test_cegis_pruning_reduces_solver_checks(self) -> None:
        bv8 = BitVecType(width=8)
        types = {"x": bv8, "y": bv8}
        spec = Shl(a=Add(a=Var("x"), b=Var("y")), b=BitVecConst(width=8, value=3))

        stats_no = SuperoptStats()
        with self.assertRaises(SuperoptError):
            superopt_expr(
                spec,
                types,
                max_nodes=4,
                timeout_ms=200,
                use_cegis=False,
                stats=stats_no,
                max_total_candidates=2000,
            )

        stats_yes = SuperoptStats()
        with self.assertRaises(SuperoptError):
            superopt_expr(
                spec,
                types,
                max_nodes=4,
                timeout_ms=200,
                use_cegis=True,
                stats=stats_yes,
                max_total_candidates=2000,
            )

        self.assertGreater(stats_no.solver_checks, stats_yes.solver_checks)
        self.assertGreater(stats_yes.counterexamples, 0)
