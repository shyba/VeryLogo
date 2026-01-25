from __future__ import annotations
from pathlib import Path

from stc.passmgr import Pass, PassContext, PassMetrics
from stc.tick_ir import TickIR, Expr, TernaryLut
from stc.ternary_db import TernaryDB, load_db, enumerate_min_ti


class TernaryEnumeratePass(Pass):
    """Replace small windows with optimal TI sequences from precomputed DB."""

    def __init__(self, db_path: Path | None = None, max_window_size: int = 4):
        self.db_path = db_path
        self.max_window_size = max_window_size
        self.db: TernaryDB | None = None

    @property
    def name(self) -> str:
        return "ternary-enumerate"

    def should_run(self, ctx: PassContext) -> bool:
        prims = {p.name for p in ctx.technology.primitives()}
        return "lop3" in prims or "vpternlog" in prims

    def run(self, ir: TickIR, ctx: PassContext) -> tuple[TickIR, PassMetrics]:
        if self.db is None:
            if self.db_path is None:
                return ir, PassMetrics(changed=False)
            self.db = load_db(self.db_path)

        changed = False
        expressions_modified = 0

        return ir, PassMetrics(
            changed=changed,
            expressions_modified=expressions_modified,
        )
