from __future__ import annotations

import json
import signal
import time
import random
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from stc.tick_ir import TickIR

if TYPE_CHECKING:
    from stc.passmgr import PassManager, PassContext
    from stc.tech import Technology

logger = logging.getLogger(__name__)


@dataclass
class AnytimeConfig:
    """Configuration for anytime optimization."""

    timeout_seconds: float = 300.0
    no_improvement_seconds: float = 60.0
    max_iterations: int = 10000
    checkpoint_interval_seconds: float = 30.0
    checkpoint_dir: Path | None = None
    seed: int | None = None


@dataclass
class AnytimeState:
    """State tracked during anytime optimization."""

    best_ir: TickIR | None = None
    best_cost: float = float("inf")
    best_depth: int = float("inf")
    iterations: int = 0
    start_time: float = field(default_factory=time.time)
    last_improvement_time: float = field(default_factory=time.time)
    last_checkpoint_time: float = field(default_factory=time.time)
    interrupted: bool = False


@dataclass
class Checkpoint:
    """Serializable checkpoint."""

    ir_json: dict
    cost: float
    depth: int
    iterations: int
    elapsed_seconds: float
    seed: int | None

    def to_json(self) -> str:
        return json.dumps(
            {
                "ir": self.ir_json,
                "cost": self.cost,
                "depth": self.depth,
                "iterations": self.iterations,
                "elapsed_seconds": self.elapsed_seconds,
                "seed": self.seed,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, s: str) -> Checkpoint:
        d = json.loads(s)
        return cls(
            ir_json=d["ir"],
            cost=d["cost"],
            depth=d["depth"],
            iterations=d["iterations"],
            elapsed_seconds=d["elapsed_seconds"],
            seed=d.get("seed"),
        )


class AnytimeRunner:
    """Anytime optimization runner with checkpointing."""

    def __init__(
        self,
        config: AnytimeConfig,
        pass_manager: PassManager,
        technology: Technology,
    ):
        self.config = config
        self.pass_manager = pass_manager
        self.technology = technology
        self.state = AnytimeState()
        self._original_sigint = None

        if config.seed is not None:
            random.seed(config.seed)

    def run(self, ir: TickIR) -> TickIR:
        """Run optimization until stopping criteria met."""
        self._setup_signal_handler()

        try:
            self.state.best_ir = ir
            self.state.best_cost = self._compute_cost(ir)
            self.state.start_time = time.time()
            self.state.last_improvement_time = time.time()

            from stc.passmgr import PassContext

            ctx = PassContext(
                technology=self.technology,
                cost_model=self.technology.cost_model(),
                depth_model=self.technology.depth_model(),
            )

            current_ir = ir

            while not self._should_stop():
                self.state.iterations += 1
                ctx.iteration = self.state.iterations

                new_ir, metrics = self.pass_manager.run(current_ir, ctx)

                new_cost = self._compute_cost(new_ir)
                if new_cost < self.state.best_cost:
                    self.state.best_ir = new_ir
                    self.state.best_cost = new_cost
                    self.state.last_improvement_time = time.time()
                    logger.info(
                        f"Improvement: cost={new_cost:.2f} at iteration {self.state.iterations}"
                    )

                current_ir = new_ir

                self._maybe_checkpoint()

            return self.state.best_ir

        finally:
            self._restore_signal_handler()
            self._final_checkpoint()

    def _should_stop(self) -> bool:
        """Check all stopping criteria."""
        if self.state.interrupted:
            logger.info("Stopping: interrupted")
            return True

        elapsed = time.time() - self.state.start_time
        if elapsed > self.config.timeout_seconds:
            logger.info(f"Stopping: timeout ({elapsed:.1f}s)")
            return True

        no_improvement = time.time() - self.state.last_improvement_time
        if no_improvement > self.config.no_improvement_seconds:
            logger.info(f"Stopping: no improvement for {no_improvement:.1f}s")
            return True

        if self.state.iterations >= self.config.max_iterations:
            logger.info(f"Stopping: max iterations ({self.state.iterations})")
            return True

        return False

    def _compute_cost(self, ir: TickIR) -> float:
        """Compute cost using technology's cost model."""
        from stc.cost import expr_cost

        total = 0.0
        for expr in ir.output_exprs.values():
            total += expr_cost(expr, ir.inputs)
        for expr in ir.next_state.values():
            total += expr_cost(expr, ir.inputs)
        return total

    def _setup_signal_handler(self) -> None:
        """Install SIGINT handler for graceful shutdown."""

        def handler(signum, frame):
            logger.info("SIGINT received, finishing current iteration...")
            self.state.interrupted = True

        self._original_sigint = signal.signal(signal.SIGINT, handler)

    def _restore_signal_handler(self) -> None:
        """Restore original SIGINT handler."""
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)

    def _maybe_checkpoint(self) -> None:
        """Write checkpoint if interval elapsed."""
        if self.config.checkpoint_dir is None:
            return

        now = time.time()
        if (
            now - self.state.last_checkpoint_time
            < self.config.checkpoint_interval_seconds
        ):
            return

        self._write_checkpoint()
        self.state.last_checkpoint_time = now

    def _final_checkpoint(self) -> None:
        """Write final checkpoint with best result."""
        if self.config.checkpoint_dir is None:
            return
        self._write_checkpoint(final=True)

    def _write_checkpoint(self, final: bool = False) -> None:
        """Write checkpoint to disk."""
        if self.state.best_ir is None:
            return

        depth_val = (
            int(self.state.best_depth) if self.state.best_depth != float("inf") else 0
        )

        checkpoint = Checkpoint(
            ir_json=self.state.best_ir.to_dict(),
            cost=self.state.best_cost,
            depth=depth_val,
            iterations=self.state.iterations,
            elapsed_seconds=time.time() - self.state.start_time,
            seed=self.config.seed,
        )

        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if final:
            path = self.config.checkpoint_dir / "final_checkpoint.json"
        else:
            path = (
                self.config.checkpoint_dir / f"checkpoint_{self.state.iterations}.json"
            )

        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(checkpoint.to_json())
        tmp_path.rename(path)

        logger.debug(f"Checkpoint written: {path}")

    @classmethod
    def load_checkpoint(cls, path: Path) -> Checkpoint:
        """Load checkpoint from disk."""
        return Checkpoint.from_json(path.read_text())
