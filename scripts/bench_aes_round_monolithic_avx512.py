#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import time
from pathlib import Path

from stc.backend_sched import generate_scheduled_code
from stc.extract import extract_tick_ir
from stc.packed_bitslice import bitslice_circuit_to_packed
from stc.tick_ir_to_circuit_state import (
    lower_tick_ir_to_circuit_state,
    rust_lower_tick_ir_to_circuit_state,
)
from stc.testing.native_x86_runner import compile_shared, run_avx512_circuit
from stc.yosys_json import load_design
from stc.bitslice import AES_SBOX_TABLE


def _run_yosys_flatten(verilog: Path, extra_verilog: Path, top: str, out_json: Path) -> None:
    script = (
        f"read_verilog -sv {verilog} {extra_verilog}; hierarchy -top {top}; "
        "proc; flatten; opt; opt_clean; write_json "
        + str(out_json)
    )
    subprocess.run(["yosys", "-q", "-p", script], check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--verilog",
        type=Path,
        default=Path("fixtures/verilog/aes128_fixedkey_seq_modular.v"),
    )
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--rounds", type=int, default=32)
    ap.add_argument(
        "--scheduler",
        choices=["list", "serial", "pipelined"],
        default="serial",
    )
    ap.add_argument(
        "--emit-naive",
        action="store_true",
        default=False,
        help="Use naive emitter (skip scheduling/regalloc)",
    )
    ap.add_argument("--check", action="store_true", default=False)
    ap.add_argument(
        "--inplace",
        action="store_true",
        default=True,
        help="Use in-place round updates (out pointer aliases st region)",
    )
    args = ap.parse_args()

    if not args.verilog.exists():
        raise SystemExit(f"Missing verilog: {args.verilog}")

    with tempfile.TemporaryDirectory(prefix="aes_round_mono_") as td:
        td = Path(td)
        round_v = td / "aes_round.v"
        round_v.write_text(
            """
module aes_round(
  input  logic [127:0] st,
  input  logic [127:0] rk,
  output logic [127:0] out
);
  wire [127:0] sb;
  wire [127:0] sr;
  wire [127:0] mc;
  aes_sub_bytes u_sb(.s(st), .o(sb));
  aes_shift_rows u_sr(.s(sb), .o(sr));
  aes_mix_columns u_mc(.s(sr), .o(mc));
  assign out = mc ^ rk;
endmodule
""",
            encoding="utf-8",
        )
        json_path = td / "aes_round.json"

        t0 = time.perf_counter()
        _run_yosys_flatten(args.verilog, round_v, "aes_round", json_path)
        print(f"[aes_round] yosys: {time.perf_counter()-t0:.2f}s", flush=True)

        design = load_design(json_path, top="aes_round")
        t0 = time.perf_counter()
        tick_ir = extract_tick_ir(design)
        print(f"[aes_round] extract: {time.perf_counter()-t0:.2f}s", flush=True)

        t0 = time.perf_counter()
        rust_result = rust_lower_tick_ir_to_circuit_state(tick_ir)
        if rust_result is None:
            circuit, layout = lower_tick_ir_to_circuit_state(tick_ir)
            lower_path = "python"
        else:
            circuit, layout = rust_result
            lower_path = "rust"
        print(
            f"[aes_round] lower({lower_path}): {time.perf_counter()-t0:.2f}s "
            f"gates={len(circuit.gates)}",
            flush=True,
        )

        packed, packed_layout = bitslice_circuit_to_packed(circuit, layout)
        input_io_words, output_io_words = packed_layout.io_words()

        st_info = packed_layout.inputs.get("st")
        rk_info = packed_layout.inputs.get("rk")
        out_info = packed_layout.outputs.get("out")
        if st_info is None or rk_info is None or out_info is None:
            raise SystemExit("Expected inputs st/rk and output out in layout")

        st_off = int(st_info["lsw"])
        st_width = int(st_info["width_bits"])
        rk_off = int(rk_info["lsw"])
        rk_width = int(rk_info["width_bits"])
        out_off = int(out_info["lsw"])
        out_width = int(out_info["width_bits"])

        if st_width != 128 or rk_width != 128 or out_width != 128:
            raise SystemExit("aes_round expected 128-bit st/rk/out")

        if args.emit_naive:
            os.environ["STC_EMIT_NAIVE"] = "1"

        t0 = time.perf_counter()
        code = generate_scheduled_code(
            packed,
            target="avx512_u64",
            scheduler=args.scheduler,
            io_split=(input_io_words, output_io_words),
        )
        print(f"[aes_round] emit: {time.perf_counter()-t0:.2f}s", flush=True)

        out_expr = f"U + {st_off}" if args.inplace else f"S + {out_off}"
        memcpy_stmt = (
            "/* in-place, no memcpy */"
            if args.inplace
            else f"memcpy(&U[{st_off}], &S[{out_off}], sizeof(__m512i) * {st_width});"
        )
        sink_expr = f"U[{st_off}]" if args.inplace else "S[0]"

        bench_code = fr'''
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

void circuit(__m512i* in, __m512i* out);

int main(int argc, char** argv) {{
    int iterations = {args.iters};
    int rounds = {args.rounds};
    if (argc > 1) iterations = atoi(argv[1]);

    struct timespec start, end;
    __m512i *U = (__m512i*)aligned_alloc(64, sizeof(__m512i) * {input_io_words});
    __m512i *S = (__m512i*)aligned_alloc(64, sizeof(__m512i) * {output_io_words});
    for (int i = 0; i < {input_io_words}; i++) U[i] = _mm512_set1_epi64(0);

    // Initialize a nonzero key pattern so rk is not all-zeros.
    for (int i = 0; i < {rk_width}; i++) {{
        U[{rk_off} + i] = _mm512_set1_epi64((long long)(0xA5A5A5A5A5A5A5A5ULL ^ i));
    }}

    volatile uint64_t sink = 0;
    for (int i = 0; i < 1000; i++) {{
        for (int r = 0; r < rounds; r++) {{
            circuit(U, {out_expr});
            {memcpy_stmt}
        }}
        sink += _mm512_cvtsi512_si32({sink_expr});
    }}

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {{
        for (int r = 0; r < rounds; r++) {{
            circuit(U, {out_expr});
            {memcpy_stmt}
        }}
        sink += _mm512_cvtsi512_si32({sink_expr});
    }}
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double evals = (double)iterations * 512.0 * rounds;
    double ns_per_eval = elapsed * 1e9 / evals;
    double evals_per_sec = evals / elapsed;

    printf("AES round (monolithic, %d rounds) packed-bitslice AVX-512:\\n", rounds);
    printf("  Iterations: %d x 512 lanes\\n", iterations);
    printf("  Time per eval: %.3f ns\\n", ns_per_eval);
    printf("  Throughput: %.3fB evals/sec\\n", evals_per_sec / 1e9);
    printf("  (Sink: %lu)\\n", sink);
    return 0;
}}
'''

        cdir = td / "c"
        cdir.mkdir()
        circuit_c = cdir / "circuit.c"
        bench_c = cdir / "bench.c"
        bench_bin = cdir / "bench"
        circuit_c.write_text(code, encoding="utf-8")
        bench_c.write_text(bench_code, encoding="utf-8")

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
            str(circuit_c),
        ]
        subprocess.run(cmd, check=True)

        if args.check:
            build = compile_shared(
                circuit_c,
                cflags=["-mavx512f", "-mavx512vl", "-mavx512dq", "-mavx512bw"],
            )

            def sub_bytes(state):
                return [AES_SBOX_TABLE[b] for b in state]

            def shift_rows(state):
                b = state
                return [
                    b[0], b[5], b[10], b[15],
                    b[4], b[9], b[14], b[3],
                    b[8], b[13], b[2], b[7],
                    b[12], b[1], b[6], b[11],
                ]

            def xtime(x):
                return ((x << 1) & 0xFF) ^ (0x1B if (x & 0x80) else 0)

            def mix_col(col):
                a0, a1, a2, a3 = col
                t = a0 ^ a1 ^ a2 ^ a3
                u = a0
                b0 = a0 ^ t ^ xtime(a0 ^ a1)
                b1 = a1 ^ t ^ xtime(a1 ^ a2)
                b2 = a2 ^ t ^ xtime(a2 ^ a3)
                b3 = a3 ^ t ^ xtime(a3 ^ u)
                return [b0 & 0xFF, b1 & 0xFF, b2 & 0xFF, b3 & 0xFF]

            def mix_columns(state):
                out = []
                for c in range(4):
                    col = state[c * 4 : (c + 1) * 4]
                    out.extend(mix_col(col))
                return out

            def aes_round_ref(st_bytes, rk_bytes):
                sb = sub_bytes(st_bytes)
                sr = shift_rows(sb)
                mc = mix_columns(sr)
                return [a ^ b for a, b in zip(mc, rk_bytes)]

            # Deterministic test vector.
            st_bytes = [(i * 13 + 7) & 0xFF for i in range(16)]
            rk_bytes = [(0xA5 ^ (i * 17)) & 0xFF for i in range(16)]
            ref = aes_round_ref(st_bytes, rk_bytes)

            # Build bitsliced inputs (all lanes identical).
            in_words = [0] * input_io_words
            for byte_idx, byte in enumerate(rk_bytes):
                base = (15 - byte_idx) * 8
                for bit in range(8):
                    if (byte >> bit) & 1:
                        in_words[rk_off + base + bit] = 0xFFFFFFFFFFFFFFFF
            for byte_idx, byte in enumerate(st_bytes):
                base = (15 - byte_idx) * 8
                for bit in range(8):
                    if (byte >> bit) & 1:
                        in_words[st_off + base + bit] = 0xFFFFFFFFFFFFFFFF

            out_words = run_avx512_circuit(
                build.so_path,
                in_words,
                input_bits=input_io_words,
                output_bits=output_io_words,
            )
            out_bits = []
            for i in range(out_width):
                w = out_words[out_off + i]
                out_bits.append(1 if (w & 1) else 0)
            out_bytes = []
            for byte_idx in range(16):
                base = (15 - byte_idx) * 8
                v = 0
                for bit in range(8):
                    v |= (out_bits[base + bit] & 1) << bit
                out_bytes.append(v)
            if out_bytes != ref:
                raise SystemExit(
                    f"check failed: got {out_bytes} expected {ref}"
                )
            print("Correctness check: OK")
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
