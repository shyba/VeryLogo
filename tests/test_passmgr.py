import unittest

from stc.passmgr import Pass, PassContext, PassManager, PassMetrics, PassSchedule
from stc.tech import GenericTechnology
from stc.tick_ir import TickIR


class CountingPass(Pass):
    """Test pass that counts invocations."""

    def __init__(self):
        self.count = 0

    @property
    def name(self) -> str:
        return "counting"

    def run(self, ir, ctx):
        self.count += 1
        return ir, PassMetrics(changed=self.count < 3)


class NoOpPass(Pass):
    """Test pass that never changes anything."""

    @property
    def name(self) -> str:
        return "noop"

    def run(self, ir, ctx):
        return ir, PassMetrics(changed=False)


class TestPassManager(unittest.TestCase):
    def test_fixpoint(self):
        """Pass manager reaches fixpoint when no changes."""
        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
        )

        p = CountingPass()
        schedule = PassSchedule(max_iterations=10).add(p)
        mgr = PassManager(schedule)

        ir = TickIR(
            name="test",
            inputs={},
            outputs={},
            state={},
            reset_state={},
            next_state={},
            output_exprs={},
        )
        result, metrics = mgr.run(ir, ctx)

        self.assertEqual(p.count, 3)

    def test_max_iterations(self):
        """Pass manager respects max_iterations."""
        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
        )

        p = CountingPass()
        schedule = PassSchedule(max_iterations=2).add(p)
        mgr = PassManager(schedule)

        ir = TickIR(
            name="test",
            inputs={},
            outputs={},
            state={},
            reset_state={},
            next_state={},
            output_exprs={},
        )
        result, metrics = mgr.run(ir, ctx)

        self.assertEqual(p.count, 2)

    def test_no_change_stops(self):
        """Pass manager stops when no passes make changes."""
        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
        )

        p = NoOpPass()
        schedule = PassSchedule(max_iterations=10).add(p)
        mgr = PassManager(schedule)

        ir = TickIR(
            name="test",
            inputs={},
            outputs={},
            state={},
            reset_state={},
            next_state={},
            output_exprs={},
        )
        result, metrics = mgr.run(ir, ctx)

        self.assertEqual(len(metrics), 1)

    def test_interrupt(self):
        """Pass manager can be interrupted."""
        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
        )

        p = CountingPass()
        schedule = PassSchedule(max_iterations=100).add(p)
        mgr = PassManager(schedule)

        mgr.interrupt()

        ir = TickIR(
            name="test",
            inputs={},
            outputs={},
            state={},
            reset_state={},
            next_state={},
            output_exprs={},
        )
        result, metrics = mgr.run(ir, ctx)

        self.assertEqual(p.count, 0)


class TestPassSchedule(unittest.TestCase):
    def test_add_pass(self):
        """Can add passes to schedule."""
        schedule = PassSchedule()
        self.assertEqual(len(schedule.passes), 0)

        schedule.add(NoOpPass())
        self.assertEqual(len(schedule.passes), 1)

        schedule.add(CountingPass())
        self.assertEqual(len(schedule.passes), 2)

    def test_chained_add(self):
        """add() returns self for chaining."""
        schedule = PassSchedule().add(NoOpPass()).add(CountingPass())
        self.assertEqual(len(schedule.passes), 2)


class TestPassContext(unittest.TestCase):
    def test_update_best(self):
        """PassContext tracks best-so-far IR."""
        from stc.tick_ir import BoolConst, BoolType

        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
        )

        ir1 = TickIR(
            name="test",
            inputs={},
            outputs={"o": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": BoolConst(value=True)},
        )

        updated = ctx.update_best(ir1, 1.0)
        self.assertTrue(updated)
        self.assertEqual(ctx.best_cost, 1.0)
        self.assertEqual(ctx.best_ir, ir1)

        updated = ctx.update_best(ir1, 2.0)
        self.assertFalse(updated)
        self.assertEqual(ctx.best_cost, 1.0)

        updated = ctx.update_best(ir1, 0.5)
        self.assertTrue(updated)
        self.assertEqual(ctx.best_cost, 0.5)
