#!/usr/bin/env python3
"""
Emit PTX for the BP (128 gate) AES S-box circuit, opportunistically mapping
3-input cones into ternary gates so the PTX emitter can use `lop3.b32`.

This is intended as a quick "does it generate lop3?" utility, not a full GPU
benchmark harness.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bp_circuit_sbox import build_bp_sbox
from scripts.bench_lop3_cuda import _compute_cone_imm8
from scripts.benchmark_ternary_sbox import find_3input_cones
from stc.backend_sched import generate_scheduled_code
from stc.circuit_synth import CircuitState


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/bp128_lop3.ptx")
    ap.add_argument(
        "--assemble", action="store_true", help="Try to assemble with ptxas"
    )
    ap.add_argument("--sm", default="sm_61", help="SM to pass to ptxas (if --assemble)")
    args = ap.parse_args()

    bp = build_bp_sbox()

    cones = find_3input_cones(bp)
    new_gates = list(bp.gates)
    mapped = 0
    for cone in cones:
        root = cone["root"]
        leaves = cone["leaves"]
        imm8 = _compute_cone_imm8(bp, root, leaves)
        # Replace the root gate with a ternary LUT over the three leaves.
        # This keeps node numbering stable (same gate index/root node).
        new_gates[root] = ("ternary", leaves[0], leaves[1], leaves[2], imm8)
        mapped += 1

    mapped_circuit = CircuitState(
        input_bits=bp.input_bits,
        output_bits=bp.output_bits,
        gates=new_gates,
        outputs=list(bp.outputs),
        gate_count=len(new_gates),
    ).eliminate_dead_code()

    # Emit scheduled PTX function using PTXEmitter (lop3 for ternary gates).
    ptx = generate_scheduled_code(
        mapped_circuit, target="ptx", scheduler="list", function_name="sbox_bp128_lop3"
    )

    lop3_count = len(re.findall(r"\blop3\.b32\b", ptx))
    ternary_count = sum(
        1 for g in mapped_circuit.gates if len(g) == 5 and g[0] == "ternary"
    )

    header = "\n".join(
        [
            ".version 7.0",
            f".target {args.sm}",
            ".address_size 64",
            "",
        ]
    )
    ptx_full = header + ptx

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(ptx_full)

    print(f"BP gates: {bp.gate_count}")
    print(
        f"Cones mapped to ternary: {mapped} (ternary gates after DCE: {ternary_count})"
    )
    print(f"PTX written: {args.out}")
    print(f"lop3.b32 count: {lop3_count}")

    if args.assemble:
        ptxas = shutil.which("ptxas")
        if ptxas is None:
            print("ptxas not found on PATH")
            return 2
        cubin_out = os.path.splitext(args.out)[0] + ".cubin"
        cmd = [ptxas, f"-arch={args.sm}", args.out, "-o", cubin_out]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stderr.strip()[:2000])
            return res.returncode
        print(f"Assembled cubin: {cubin_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
