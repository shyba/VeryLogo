import unittest
import tempfile
from pathlib import Path

from stc.anytime import AnytimeRunner, AnytimeConfig, AnytimeState, Checkpoint
from stc.passmgr import PassManager, PassSchedule
from stc.tech import GenericTechnology
from stc.tick_ir import TickIR, Xor, Var, BoolType


class TestAnytimeConfig(unittest.TestCase):
    def test_default_config(self):
        """Default config has reasonable values."""
        config = AnytimeConfig()
        self.assertEqual(config.timeout_seconds, 300.0)
        self.assertEqual(config.no_improvement_seconds, 60.0)
        self.assertEqual(config.max_iterations, 10000)
        self.assertIsNone(config.checkpoint_dir)
        self.assertIsNone(config.seed)


class TestAnytimeState(unittest.TestCase):
    def test_initial_state(self):
        """Initial state has expected values."""
        state = AnytimeState()
        self.assertIsNone(state.best_ir)
        self.assertEqual(state.best_cost, float("inf"))
        self.assertEqual(state.iterations, 0)
        self.assertFalse(state.interrupted)


class TestCheckpoint(unittest.TestCase):
    def test_checkpoint_roundtrip(self):
        """Checkpoint can be serialized and deserialized."""
        ir = TickIR(
            name="test",
            inputs={"a": BoolType(), "b": BoolType()},
            outputs={"o": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Xor(Var("a"), Var("b"))},
        )

        checkpoint = Checkpoint(
            ir_json=ir.to_dict(),
            cost=42.0,
            depth=3,
            iterations=100,
            elapsed_seconds=5.5,
            seed=12345,
        )

        json_str = checkpoint.to_json()
        loaded = Checkpoint.from_json(json_str)

        self.assertEqual(loaded.cost, 42.0)
        self.assertEqual(loaded.depth, 3)
        self.assertEqual(loaded.iterations, 100)
        self.assertEqual(loaded.elapsed_seconds, 5.5)
        self.assertEqual(loaded.seed, 12345)


class TestAnytimeRunner(unittest.TestCase):
    def test_runner_creation(self):
        """AnytimeRunner can be created."""
        config = AnytimeConfig(timeout_seconds=1.0, max_iterations=1)
        tech = GenericTechnology()
        schedule = PassSchedule(max_iterations=1)
        mgr = PassManager(schedule)

        runner = AnytimeRunner(config, mgr, tech)

        self.assertEqual(runner.config, config)
        self.assertEqual(runner.technology, tech)

    def test_stop_on_max_iterations(self):
        """Runner stops after max iterations."""
        config = AnytimeConfig(
            timeout_seconds=999.0, no_improvement_seconds=999.0, max_iterations=5
        )
        tech = GenericTechnology()
        schedule = PassSchedule(max_iterations=1, stop_on_no_change=False)
        mgr = PassManager(schedule)

        runner = AnytimeRunner(config, mgr, tech)

        a = Var("a")
        b = Var("b")
        ir = TickIR(
            name="test",
            inputs={"a": BoolType(), "b": BoolType()},
            outputs={"o": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Xor(a, b)},
        )

        result = runner.run(ir)

        self.assertIsNotNone(result)
        self.assertEqual(runner.state.iterations, 5)

    def test_checkpoint_write(self):
        """Runner writes checkpoints to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"

            config = AnytimeConfig(
                timeout_seconds=0.1,
                checkpoint_interval_seconds=0.01,
                checkpoint_dir=checkpoint_dir,
                max_iterations=2,
            )
            tech = GenericTechnology()
            schedule = PassSchedule(max_iterations=1)
            mgr = PassManager(schedule)

            runner = AnytimeRunner(config, mgr, tech)

            a = Var("a")
            b = Var("b")
            ir = TickIR(
                name="test",
                inputs={"a": BoolType(), "b": BoolType()},
                outputs={"o": BoolType()},
                state={},
                reset_state={},
                next_state={},
                output_exprs={"o": Xor(a, b)},
            )

            result = runner.run(ir)

            final_checkpoint = checkpoint_dir / "final_checkpoint.json"
            self.assertTrue(final_checkpoint.exists())

            loaded = AnytimeRunner.load_checkpoint(final_checkpoint)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.iterations, runner.state.iterations)

    def test_no_checkpoint_without_dir(self):
        """Runner doesn't write checkpoints if no directory set."""
        config = AnytimeConfig(
            timeout_seconds=0.1, checkpoint_dir=None, max_iterations=2
        )
        tech = GenericTechnology()
        schedule = PassSchedule(max_iterations=1)
        mgr = PassManager(schedule)

        runner = AnytimeRunner(config, mgr, tech)

        a = Var("a")
        b = Var("b")
        ir = TickIR(
            name="test",
            inputs={"a": BoolType(), "b": BoolType()},
            outputs={"o": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Xor(a, b)},
        )

        result = runner.run(ir)
        self.assertIsNotNone(result)
