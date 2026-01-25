from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
import logging

from stc.tick_ir import TickIR
from stc.tech import Technology, CostModel, DepthModel

logger = logging.getLogger(__name__)


@dataclass
class PassMetrics:
    """Metrics from a pass execution."""

    changed: bool = False
    expressions_modified: int = 0
    cost_delta: float = 0.0
    depth_delta: int = 0
    time_seconds: float = 0.0


@dataclass
class PassContext:
    """Context passed to all passes."""

    technology: Technology
    cost_model: CostModel
    depth_model: DepthModel
    depth_budget: int | None = None
    cost_budget: float | None = None
    iteration: int = 0
    best_cost: float = float("inf")
    best_ir: TickIR | None = None
    verbosity: int = 0

    def update_best(self, ir: TickIR, cost: float) -> bool:
        """Update best-so-far if improved. Returns True if updated."""
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_ir = ir
            return True
        return False


class Pass(ABC):
    """Abstract optimization pass."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def run(self, ir: TickIR, ctx: PassContext) -> tuple[TickIR, PassMetrics]:
        """Run pass on IR. Returns (new_ir, metrics)."""
        ...

    def should_run(self, ctx: PassContext) -> bool:
        """Check if pass should run given context."""
        return True


@dataclass
class PassSchedule:
    """A sequence of passes to run."""

    passes: list[Pass] = field(default_factory=list)
    max_iterations: int = 100
    timeout_seconds: float | None = None
    stop_on_no_change: bool = True

    def add(self, p: Pass) -> PassSchedule:
        self.passes.append(p)
        return self


class PassManager:
    """Manages pass execution with fixpoint iteration."""

    def __init__(self, schedule: PassSchedule):
        self.schedule = schedule
        self._interrupted = False

    def run(self, ir: TickIR, ctx: PassContext) -> tuple[TickIR, list[PassMetrics]]:
        """Run all passes to fixpoint or budget exhaustion."""
        all_metrics: list[PassMetrics] = []
        current_ir = ir
        start_time = time.time()

        for iteration in range(self.schedule.max_iterations):
            if self._interrupted:
                logger.info("Pass manager interrupted, returning best-so-far")
                break

            ctx.iteration = iteration
            iteration_changed = False

            for p in self.schedule.passes:
                if self._interrupted:
                    break

                if self.schedule.timeout_seconds:
                    elapsed = time.time() - start_time
                    if elapsed > self.schedule.timeout_seconds:
                        logger.info(f"Timeout after {elapsed:.1f}s")
                        break

                if not p.should_run(ctx):
                    continue

                pass_start = time.time()
                new_ir, metrics = p.run(current_ir, ctx)
                metrics.time_seconds = time.time() - pass_start
                all_metrics.append(metrics)

                if metrics.changed:
                    iteration_changed = True
                    current_ir = new_ir

                    cost = ctx.cost_model.expr_cost(current_ir)
                    ctx.update_best(current_ir, cost)

                if ctx.verbosity > 0:
                    logger.info(
                        f"  {p.name}: changed={metrics.changed}, "
                        f"time={metrics.time_seconds:.3f}s"
                    )

            if self.schedule.stop_on_no_change and not iteration_changed:
                logger.info(f"Fixpoint reached after {iteration + 1} iterations")
                break

        return ctx.best_ir or current_ir, all_metrics

    def interrupt(self) -> None:
        """Signal interrupt (called from signal handler)."""
        self._interrupted = True
