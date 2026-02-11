#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from stc.packed_circuit import eval_packed_circuit_words
from stc.packed_region_emit import EmitRegionsConfig, emit_avx512_u64_regions
from stc.packed_regions import RegionCaps
from stc.testing.native_x86_runner import (
    compile_shared,
    have_avx512,
    run_avx512_steps_shared,
)
from stc.tick_ir_bin2 import read_tick_ir_bin
from stc.tick_ir_to_packed_circuit_state import lower_tick_ir_to_packed_circuit_state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tick-ir", type=Path, default=Path("out/reduced_tick_ir.bin"))
    ap.add_argument("--out", type=Path, default=Path("out/packed_bench"))
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--region-max-gates", type=int, default=50000)
    ap.add_argument("--region-max-boundary", type=int, default=8192)
    ap.add_argument("--check", action="store_true", default=False)
    ns = ap.parse_args()

    if not have_avx512():
        raise SystemExit("requires avx512f")

    ir = read_tick_ir_bin(str(ns.tick_ir))
    circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)

    code = emit_avx512_u64_regions(
        circuit,
        layout,
        function_name="circuit",
        config=EmitRegionsConfig(
            caps=RegionCaps(
                max_gates=ns.region_max_gates, max_boundary=ns.region_max_boundary
            )
        ),
    )

    ns.out.mkdir(parents=True, exist_ok=True)
    c_path = ns.out / "circuit_avx512_u64_regions.c"
    c_path.write_text(code, encoding="utf-8")
    build = compile_shared(c_path, cflags=["-mavx512f"])

    input_io_words = sum(
        (int(v["width_bits"]) + 63) // 64 for v in layout.inputs.values()
    )
    state_words = sum((int(v["width_bits"]) + 63) // 64 for v in layout.state.values())
    output_io_words = sum(
        (int(v["width_bits"]) + 63) // 64 for v in layout.outputs.values()
    )

    in_io = [0] * input_io_words
    st = [0] * state_words

    if ns.check:
        out_ref, st_ref = [], []
        outs = eval_packed_circuit_words(circuit, in_io + st)
        out_ref = outs[:output_io_words]
        st_ref = outs[output_io_words:]
        out_io, st_out = run_avx512_steps_shared(
            build.so_path,
            in_io,
            st,
            input_io_bits=input_io_words,
            state_bits=state_words,
            output_io_bits=output_io_words,
            steps=1,
        )
        if out_io != out_ref or st_out != st_ref:
            raise SystemExit("check failed for 1 step")

    t0 = time.perf_counter()
    out_io = [0] * output_io_words
    for _ in range(ns.iters):
        out_io, st = run_avx512_steps_shared(
            build.so_path,
            in_io,
            st,
            input_io_bits=input_io_words,
            state_bits=state_words,
            output_io_bits=output_io_words,
            steps=ns.steps,
        )
    t1 = time.perf_counter()

    total_steps = ns.iters * ns.steps
    secs = t1 - t0
    print(
        f"steps: {total_steps}  seconds: {secs:.6f}  steps/sec: {total_steps / secs:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
