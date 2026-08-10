#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

from stc.backend_sched import generate_scheduled_code
from stc.extract import extract_tick_ir
from stc.packed_bitslice import bitslice_circuit_to_packed
from stc.slice_circuit import slice_circuit_by_block_size
from stc.tick_ir_to_circuit_state import (
    lower_tick_ir_to_circuit_state,
    rust_lower_tick_ir_to_circuit_state,
    PackedLayout,
)
from stc.yosys_json import load_design


STAGES = [
    ("aes_sub_bytes", "subbytes"),
    ("aes_shift_rows", "shiftrows"),
    ("aes_mix_columns", "mixcolumns"),
]


def _run_yosys_flatten(verilog: Path, top: str, out_json: Path) -> None:
    script = (
        f"read_verilog -sv {verilog}; hierarchy -top {top}; "
        "proc; flatten; opt; opt_clean; write_json " + str(out_json)
    )
    subprocess.run(["yosys", "-q", "-p", script], check=True)


def _rename_circuit_symbols(code: str, name: str) -> str:
    code = code.replace("circuit_steps_shared", f"{name}_steps_shared")
    code = code.replace("circuit__core", f"{name}__core")
    code = code.replace("void circuit(", f"void {name}(")
    return code


def _compile_stage(
    verilog: Path,
    top: str,
    name: str,
    out_dir: Path,
    scheduler: str,
    block_size: int | None = None,
) -> list[tuple[str, int, int]]:
    json_path = out_dir / f"{name}.json"
    t0 = time.perf_counter()
    _run_yosys_flatten(verilog, top, json_path)
    print(f"[{name}] yosys: {time.perf_counter()-t0:.2f}s", flush=True)
    design = load_design(json_path, top=top)
    t0 = time.perf_counter()
    tick_ir = extract_tick_ir(design)
    print(f"[{name}] extract: {time.perf_counter()-t0:.2f}s", flush=True)
    t0 = time.perf_counter()
    rust_result = rust_lower_tick_ir_to_circuit_state(tick_ir)
    if rust_result is None:
        circuit, layout = lower_tick_ir_to_circuit_state(tick_ir)
        lower_path = "python"
    else:
        circuit, layout = rust_result
        lower_path = "rust"
    print(
        f"[{name}] lower({lower_path}): {time.perf_counter()-t0:.2f}s gates={len(circuit.gates)}",
        flush=True,
    )
    if block_size is not None:
        slices = slice_circuit_by_block_size(circuit, block_size)
    else:
        slices = None

    slice_infos: list[tuple[str, int, int]] = []

    if slices:
        for sl in slices:
            if sl.output_offset is None:
                raise SystemExit(
                    f"{name}: slice outputs are non-contiguous; cannot map directly"
                )
            slice_layout = PackedLayout(
                inputs={"in": {"lsb": 0, "width": sl.input_bits}},
                state={},
                outputs={"out": {"lsb": 0, "width": sl.circuit.output_bits}},
                next_state={},
                input_bits=sl.input_bits,
                output_bits=sl.circuit.output_bits,
            )
            packed, packed_layout = bitslice_circuit_to_packed(sl.circuit, slice_layout)
            input_io_words, output_io_words = packed_layout.io_words()
            t0 = time.perf_counter()
            code = generate_scheduled_code(
                packed,
                target="avx512_u64",
                scheduler=scheduler,
                io_split=(input_io_words, output_io_words),
            )
            print(
                f"[{name}] slice{sl.block_index}: emit {time.perf_counter()-t0:.2f}s "
                f"code={len(code)} bytes",
                flush=True,
            )
            func_name = f"{name}_s{sl.block_index}"
            code = _rename_circuit_symbols(code, func_name)
            c_path = out_dir / f"{func_name}.c"
            c_path.write_text(code, encoding="utf-8")
            slice_infos.append((func_name, sl.input_offset, sl.output_offset))
        return slice_infos

    t0 = time.perf_counter()
    packed, packed_layout = bitslice_circuit_to_packed(circuit, layout)
    print(f"[{name}] bitslice pack: {time.perf_counter()-t0:.2f}s", flush=True)
    input_io_words, output_io_words = packed_layout.io_words()
    t0 = time.perf_counter()
    code = generate_scheduled_code(
        packed,
        target="avx512_u64",
        scheduler=scheduler,
        io_split=(input_io_words, output_io_words),
    )
    print(
        f"[{name}] emit: {time.perf_counter()-t0:.2f}s code={len(code)} bytes",
        flush=True,
    )
    code = _rename_circuit_symbols(code, name)
    c_path = out_dir / f"{name}.c"
    c_path.write_text(code, encoding="utf-8")
    slice_infos.append((name, 0, 0))
    return slice_infos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--verilog",
        type=Path,
        default=Path("fixtures/verilog/aes128_fixedkey_seq_modular.v"),
    )
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument(
        "--scheduler",
        choices=["list", "serial", "pipelined"],
        default="list",
    )
    ap.add_argument("--slice-subbytes", type=int, default=0)
    ap.add_argument("--slice-shiftrows", type=int, default=0)
    ap.add_argument("--slice-mixcolumns", type=int, default=0)
    ap.add_argument(
        "--emit-naive",
        action="store_true",
        default=False,
        help="Use naive emitter (skip scheduling/regalloc)",
    )
    args = ap.parse_args()

    if not args.verilog.exists():
        raise SystemExit(f"Missing verilog: {args.verilog}")

    with tempfile.TemporaryDirectory(prefix="aes_round_piped_") as td:
        td = Path(td)
        if args.emit_naive:
            import os

            os.environ["STC_EMIT_NAIVE"] = "1"

        stage_paths: list[Path] = []
        stage_calls: dict[str, list[tuple[str, int, int]]] = {}
        for top, name in STAGES:
            if name == "subbytes":
                block = args.slice_subbytes or None
            elif name == "shiftrows":
                block = args.slice_shiftrows or None
            else:
                block = args.slice_mixcolumns or None
            info = _compile_stage(
                args.verilog, top, name, td, args.scheduler, block_size=block
            )
            stage_calls[name] = info
            for func_name, _in_off, _out_off in info:
                stage_paths.append(td / f"{func_name}.c")

        bench_c = td / "bench.c"
        bench_bin = td / "bench"

        def _emit_calls(stage_name: str, in_name: str, out_name: str) -> str:
            lines = []
            for func_name, in_off, out_off in stage_calls[stage_name]:
                lines.append(
                    f"        {func_name}({in_name} + {in_off}, {out_name} + {out_off});"
                )
            return "\n".join(lines)

        bench_c.write_text(
            f"""
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>

{chr(10).join([f"void {fn}(__m512i* in, __m512i* out);" for fn,_,_ in stage_calls['subbytes']])}
{chr(10).join([f"void {fn}(__m512i* in, __m512i* out);" for fn,_,_ in stage_calls['shiftrows']])}
{chr(10).join([f"void {fn}(__m512i* in, __m512i* out);" for fn,_,_ in stage_calls['mixcolumns']])}

int main(int argc, char** argv) {{
    int iterations = 20000;
    if (argc > 1) iterations = atoi(argv[1]);

    struct timespec start, end;
    __m512i U[128];
    __m512i T0[128];
    __m512i T1[128];
    __m512i O[128];

    for (int i = 0; i < 128; i++) U[i] = _mm512_set1_epi64(0);

    volatile uint64_t sink = 0;
    for (int i = 0; i < 1000; i++) {{
{_emit_calls('subbytes', 'U', 'T0')}
{_emit_calls('shiftrows', 'T0', 'T1')}
{_emit_calls('mixcolumns', 'T1', 'O')}
        sink += _mm512_cvtsi512_si32(O[0]);
    }}

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {{
{_emit_calls('subbytes', 'U', 'T0')}
{_emit_calls('shiftrows', 'T0', 'T1')}
{_emit_calls('mixcolumns', 'T1', 'O')}
        sink += _mm512_cvtsi512_si32(O[0]);
    }}
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double evals = (double)iterations * 512.0;
    double ns_per_eval = elapsed * 1e9 / evals;
    double evals_per_sec = evals / elapsed;

    printf("AES round piped (subbytes->shiftrows->mixcolumns) packed-bitslice AVX-512:\\n");
    printf("  Iterations: %d x 512 lanes\\n", iterations);
    printf("  Time per eval: %.3f ns\\n", ns_per_eval);
    printf("  Throughput: %.3fB evals/sec\\n", evals_per_sec / 1e9);
    printf("  (Sink: %lu)\\n", sink);
    return 0;
}}
""",
            encoding="utf-8",
        )

        cmd = [
            "gcc",
            "-O3",
            "-march=native",
            "-mavx512f",
            "-mavx512vl",
            "-mavx512dq",
            "-mavx512bw",
            "-o",
            str(bench_bin),
            str(bench_c),
            *[str(p) for p in stage_paths],
        ]
        subprocess.run(cmd, check=True)
        result = subprocess.run(
            [str(bench_bin), str(args.iters)],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
