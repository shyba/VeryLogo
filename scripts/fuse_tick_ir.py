from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stc.fuse_ticks import FuseBudget, fuse_ticks
from stc.reduce import reduce_tick_ir
from stc.tick_ir import TickIR
from stc.tick_ir_validate import validate_tick_ir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="fuse_tick_ir")
    p.add_argument(
        "tick_ir_json", type=Path, help="Input TickIR JSON (TickIR.to_dict())"
    )
    p.add_argument("--out", type=Path, required=True, help="Output TickIR JSON")
    p.add_argument("--n", type=int, required=True, help="Number of ticks to fuse (>=1)")
    p.add_argument(
        "--mode",
        choices=["final", "all", "state-only"],
        default="final",
        help="Fusion output mode (default: final).",
    )
    p.add_argument(
        "--input-policy",
        choices=["shared", "replicate"],
        default="shared",
        help="Input policy for fusion (default: shared).",
    )
    p.add_argument("--budget-max-nodes", type=int, default=1 << 60)
    p.add_argument("--budget-max-depth", type=int, default=1 << 60)
    return p.parse_args([] if argv is None else argv)


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)
    data = json.loads(ns.tick_ir_json.read_text(encoding="utf-8"))
    ir = TickIR.from_dict(data)
    validate_tick_ir(ir)

    fused = fuse_ticks(
        reduce_tick_ir(ir),
        ns.n,
        mode=ns.mode,
        input_policy=ns.input_policy,
        budget=FuseBudget(
            max_nodes=ns.budget_max_nodes,
            max_depth=ns.budget_max_depth,
            stop_on_budget_hit=True,
        ),
    )
    validate_tick_ir(fused)
    ns.out.write_text(
        json.dumps(fused.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
