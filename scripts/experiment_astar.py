from __future__ import annotations

import argparse
import json
from pathlib import Path

from stc.autovec_pass import autovectorize_tick_ir
from stc.extract import extract_tick_ir
from stc.infer_simd import infer_simd_types
from stc.reduce import optimize_tick_ir
from stc.tick_ir_validate import validate_tick_ir
from stc.yosys_frontend import run_yosys
from stc.yosys_json import load_design


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--verilog", type=Path, required=True)
    p.add_argument("--top", type=str, default="top")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--autovec", action="store_true", default=False)
    p.add_argument("--autovec-timeout-ms", type=int, default=200)
    p.add_argument("--superopt", action="store_true", default=False)
    p.add_argument("--superopt-max-nodes", type=int, default=3)
    p.add_argument("--superopt-timeout-ms", type=int, default=200)
    args = p.parse_args()

    tmp = Path(".") / "tmp_out" / "experiment_astar"
    tmp.mkdir(parents=True, exist_ok=True)
    normalized = tmp / "normalized.json"
    run_yosys(
        args.verilog, normalized, top=args.top, output_script=tmp / "normalized.ys"
    )
    design = load_design(normalized, top=args.top)
    ir0 = extract_tick_ir(design)
    validate_tick_ir(ir0)

    ir1 = infer_simd_types(ir0)
    validate_tick_ir(ir1)

    ir2 = ir1
    if args.autovec:
        ir2 = autovectorize_tick_ir(ir2, timeout_ms=args.autovec_timeout_ms)
        validate_tick_ir(ir2)
    if args.superopt:
        ir2 = optimize_tick_ir(
            ir2,
            bound=1,
            autovec=False,
            superopt=True,
            superopt_max_nodes=args.superopt_max_nodes,
            superopt_timeout_ms=args.superopt_timeout_ms,
        )
        validate_tick_ir(ir2)

    payload = {
        "inputs": {k: v.to_dict() for k, v in ir2.inputs.items()},
        "outputs": {k: v.to_dict() for k, v in ir2.outputs.items()},
        "output_exprs": {k: repr(v) for k, v in ir2.output_exprs.items()},
    }
    out = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(out, end="")
    else:
        args.out.write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
