import unittest

from stc.passes.ternary_enumerate import TernaryEnumeratePass
from stc.ternary_db import build_bgc_table, build_q0_table, compute_base_vectors
from stc.tick_ir import TickIR, Var, BoolType
from stc.passmgr import PassContext
from stc.tech import GenericTechnology


class TestTernaryEnumerate(unittest.TestCase):
    def test_pass_exists(self):
        """TernaryEnumeratePass can be instantiated."""
        pass_instance = TernaryEnumeratePass(db_path=None)
        self.assertEqual(pass_instance.name, "ternary-enumerate")

    def test_should_not_run_without_ternary_support(self):
        """Pass should not run if backend lacks ternary support."""
        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
        )

        pass_instance = TernaryEnumeratePass(db_path=None)
        self.assertFalse(pass_instance.should_run(ctx))

    def test_no_change_without_db(self):
        """Pass returns unchanged IR when no DB provided."""
        ir = TickIR(
            name="test",
            inputs={"a": BoolType()},
            outputs={"o": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Var(name="a")},
        )

        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
        )

        pass_instance = TernaryEnumeratePass(db_path=None)
        new_ir, metrics = pass_instance.run(ir, ctx)

        self.assertFalse(metrics.changed)
        self.assertEqual(new_ir, ir)
