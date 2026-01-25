from __future__ import annotations

from stc.passmgr import Pass, PassContext, PassMetrics
from stc.tick_ir import TickIR, Expr
from stc.metrics import compute_depth


class DepthResynthesisPass(Pass):
    """Use Z3 to find lower-depth implementations of small windows."""

    def __init__(self, max_window_inputs: int = 5, target_depth: int | None = None):
        self.max_window_inputs = max_window_inputs
        self.target_depth = target_depth

    @property
    def name(self) -> str:
        return "depth-resynth"

    def should_run(self, ctx: PassContext) -> bool:
        return ctx.depth_budget is not None

    def run(self, ir: TickIR, ctx: PassContext) -> tuple[TickIR, PassMetrics]:
        changed = False
        expressions_modified = 0

        return ir, PassMetrics(
            changed=changed,
            expressions_modified=expressions_modified,
        )
