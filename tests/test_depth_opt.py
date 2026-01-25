import unittest

from stc.passes.balance import (
    BalanceAssociativePass,
    collect_assoc_leaves,
    build_balanced_tree,
)
from stc.passes.depth_resynth import DepthResynthesisPass
from stc.passmgr import PassContext
from stc.tech import GenericTechnology
from stc.tick_ir import TickIR, Xor, And, Or, Var, BoolType
from stc.metrics import compute_depth


class TestBalanceAssociativePass(unittest.TestCase):
    def test_collect_xor_leaves(self):
        """Collect leaves from XOR chain."""
        a = Var("a")
        b = Var("b")
        c = Var("c")
        d = Var("d")

        expr = Xor(Xor(Xor(a, b), c), d)
        leaves = collect_assoc_leaves(expr, Xor)

        self.assertEqual(len(leaves), 4)
        self.assertIn(a, leaves)
        self.assertIn(b, leaves)
        self.assertIn(c, leaves)
        self.assertIn(d, leaves)

    def test_build_balanced_tree(self):
        """Build balanced tree from leaves."""
        a = Var("a")
        b = Var("b")
        c = Var("c")
        d = Var("d")

        leaves = [a, b, c, d]
        balanced = build_balanced_tree(Xor, leaves)

        self.assertIsInstance(balanced, Xor)
        self.assertIsInstance(balanced.a, Xor)
        self.assertIsInstance(balanced.b, Xor)

    def test_balance_pass_chain(self):
        """Balance pass reduces depth of XOR chain."""
        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
        )

        a = Var("a")
        b = Var("b")
        c = Var("c")
        d = Var("d")

        unbalanced = Xor(Xor(Xor(a, b), c), d)
        ir = TickIR(
            name="test",
            inputs={"a": BoolType(), "b": BoolType(), "c": BoolType(), "d": BoolType()},
            outputs={"o": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": unbalanced},
        )

        pass_instance = BalanceAssociativePass()
        new_ir, metrics = pass_instance.run(ir, ctx)

        self.assertTrue(metrics.changed)
        self.assertEqual(metrics.expressions_modified, 1)

        original_depth = compute_depth(unbalanced, ctx.depth_model)
        new_depth = compute_depth(new_ir.output_exprs["o"], ctx.depth_model)

        self.assertLess(new_depth, original_depth)

    def test_balance_pass_no_change(self):
        """Balance pass doesn't change already balanced expressions."""
        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
        )

        a = Var("a")
        b = Var("b")

        expr = Xor(a, b)
        ir = TickIR(
            name="test",
            inputs={"a": BoolType(), "b": BoolType()},
            outputs={"o": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )

        pass_instance = BalanceAssociativePass()
        new_ir, metrics = pass_instance.run(ir, ctx)

        self.assertFalse(metrics.changed)


class TestDepthResynthesisPass(unittest.TestCase):
    def test_pass_exists(self):
        """DepthResynthesisPass can be instantiated."""
        pass_instance = DepthResynthesisPass()
        self.assertEqual(pass_instance.name, "depth-resynth")

    def test_should_not_run_without_depth_budget(self):
        """Pass should not run if no depth budget set."""
        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
            depth_budget=None,
        )

        pass_instance = DepthResynthesisPass()
        self.assertFalse(pass_instance.should_run(ctx))

    def test_should_run_with_depth_budget(self):
        """Pass should run if depth budget set."""
        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
            depth_budget=10,
        )

        pass_instance = DepthResynthesisPass()
        self.assertTrue(pass_instance.should_run(ctx))

    def test_no_change_placeholder(self):
        """Pass returns unchanged IR (placeholder implementation)."""
        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
            depth_budget=10,
        )

        a = Var("a")
        b = Var("b")
        expr = And(a, b)
        ir = TickIR(
            name="test",
            inputs={"a": BoolType(), "b": BoolType()},
            outputs={"o": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )

        pass_instance = DepthResynthesisPass()
        new_ir, metrics = pass_instance.run(ir, ctx)

        self.assertFalse(metrics.changed)
        self.assertEqual(new_ir, ir)
