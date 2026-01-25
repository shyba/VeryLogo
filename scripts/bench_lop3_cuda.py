#!/usr/bin/env python3
"""
Benchmark and/or test correctness of lop3-based PTX for S-box circuits.

This is a higher-level wrapper around:
- Circuit construction (e.g. BP-115/128)
- Optional mapping of some 3-input cones into ternary gates
- PTX emission (lop3 for ternary gates)
- PTX assembly via ptxas
- Execution and timing via the CUDA Driver API (ctypes)
"""

from __future__ import annotations

import argparse
import ctypes
import shutil
from dataclasses import dataclass
from pathlib import Path

from scripts.benchmark_ternary_sbox import find_3input_cones
from scripts.bp_circuit_sbox import build_bp_sbox
from stc.bitslice import AES_SBOX_TABLE
from stc.circuit_synth import CircuitState
from stc.cuda_driver import Cuda
from stc.ptx_lop3 import assemble_ptx_to_cubin, emit_inline_lop3_kernel, emit_lop3_kernel


def build_bp128_sbox() -> CircuitState:
    # Existing builder named "BP tower" in some scripts.
    return build_bp_sbox()


def build_bp115_sbox() -> CircuitState:
    # "Official hardcoded smallest circuit" = the Boyar-Peralta optimal circuit
    # transcribed from external-bitsliced/bs.c.
    return build_bp_sbox()


def _compute_cone_imm8(circuit: CircuitState, root_gate_idx: int, leaves: list[int]) -> int:
    """Compute imm8 for a 3-input cone using the repo's ternary convention.

    imm8 bit i corresponds to i = (a<<2)|(b<<1)|c where (a,b,c) are the values of
    leaves[0], leaves[1], leaves[2] respectively.
    """
    gates = list(circuit.gates)
    input_bits = circuit.input_bits
    full_root = input_bits + root_gate_idx

    def eval_at(assignments: dict[int, int]) -> int:
        node_vals: dict[int, int] = {i: assignments.get(i, 0) for i in range(input_bits)}

        for g_idx, gate in enumerate(gates):
            full_idx = input_bits + g_idx
            # Leaves may include internal nodes; treat them as independent inputs by
            # honoring any explicit assignment at that node instead of recomputing it.
            if full_idx in assignments:
                node_vals[full_idx] = assignments[full_idx]
                if full_idx == full_root:
                    return node_vals[full_idx]
                continue
            if len(gate) == 5:
                _, a, b, c, imm8 = gate
                va = node_vals.get(a, assignments.get(a, 0))
                vb = node_vals.get(b, assignments.get(b, 0))
                vc = node_vals.get(c, assignments.get(c, 0))
                tidx = (va << 2) | (vb << 1) | vc
                node_vals[full_idx] = (imm8 >> tidx) & 1
            else:
                op, left, right = gate
                l_val = node_vals.get(left, assignments.get(left, 0))
                r_val = node_vals.get(right, assignments.get(right, 0)) if right >= 0 else 0
                if op == "xor":
                    node_vals[full_idx] = l_val ^ r_val
                elif op == "and":
                    node_vals[full_idx] = l_val & r_val
                elif op == "or":
                    node_vals[full_idx] = l_val | r_val
                elif op == "not":
                    node_vals[full_idx] = 1 - l_val
                elif op == "const":
                    node_vals[full_idx] = left & 1
                else:
                    node_vals[full_idx] = 0

            if full_idx == full_root:
                return node_vals[full_idx]
        return node_vals.get(full_root, 0)

    imm8 = 0
    for i in range(8):
        a_val = (i >> 2) & 1
        b_val = (i >> 1) & 1
        c_val = i & 1
        assignments = {
            leaves[0]: a_val,
            leaves[1]: b_val,
            leaves[2]: c_val,
        }
        if eval_at(assignments):
            imm8 |= 1 << i
    return imm8


def map_some_cones_to_ternary(circuit: CircuitState) -> CircuitState:
    cones = find_3input_cones(circuit)
    if not cones:
        return circuit

    new_gates = list(circuit.gates)
    for cone in cones:
        root = cone["root"]
        leaves = cone["leaves"]
        imm8 = _compute_cone_imm8(circuit, root, leaves)
        new_gates[root] = ("ternary", leaves[0], leaves[1], leaves[2], imm8)

    return CircuitState(
        input_bits=circuit.input_bits,
        output_bits=circuit.output_bits,
        gates=new_gates,
        outputs=list(circuit.outputs),
        gate_count=len(new_gates),
    ).eliminate_dead_code()


def cpu_eval_bitplanes(circuit: CircuitState, in_words: list[int]) -> list[int]:
    assert circuit.input_bits == 8 and circuit.output_bits == 8
    assert len(in_words) == 8

    values: dict[int, int] = {i: (in_words[i] & 0xFFFFFFFF) for i in range(8)}

    for g_idx, gate in enumerate(circuit.gates):
        node = 8 + g_idx
        if len(gate) == 5:
            _, a, b, c, imm8 = gate
            a_w = values[a]
            b_w = values[b]
            c_w = values[c]
            na, nb, nc = (~a_w) & 0xFFFFFFFF, (~b_w) & 0xFFFFFFFF, (~c_w) & 0xFFFFFFFF
            r = 0
            for i in range(8):
                if (imm8 >> i) & 1:
                    aa = a_w if ((i >> 2) & 1) else na
                    bb = b_w if ((i >> 1) & 1) else nb
                    cc = c_w if (i & 1) else nc
                    r |= aa & bb & cc
            values[node] = r & 0xFFFFFFFF
            continue

        op, a, b = gate
        if op == "xor":
            values[node] = (values[a] ^ values[b]) & 0xFFFFFFFF
        elif op == "and":
            values[node] = (values[a] & values[b]) & 0xFFFFFFFF
        elif op == "or":
            values[node] = (values[a] | values[b]) & 0xFFFFFFFF
        elif op == "not":
            values[node] = (~values[a]) & 0xFFFFFFFF
        elif op == "const":
            values[node] = 0xFFFFFFFF if a else 0
        else:
            values[node] = 0

    out_words: list[int] = []
    for node, inv in circuit.outputs:
        w = values[node]
        out_words.append(((~w) if inv else w) & 0xFFFFFFFF)
    return out_words


def _pack_truth_table_inputs_to_bitplanes() -> list[int]:
    # Pack inputs 0..255 into 8 instances × 32 lanes (thread instances),
    # represented as 8 uint32 bitplanes per instance.
    #
    # Layout matches kernel IO: instance t uses words at base=t*8.
    words = [0] * (8 * 8)
    for x in range(256):
        t = x // 32
        lane = x % 32
        for bit in range(8):
            if (x >> bit) & 1:
                words[t * 8 + bit] |= 1 << lane
    return [w & 0xFFFFFFFF for w in words]


def _unpack_bitplanes_to_truth_table(out_words: list[int]) -> list[int]:
    # Inverse of _pack_truth_table_inputs_to_bitplanes for outputs.
    assert len(out_words) == 8 * 8
    out_bytes = [0] * 256
    for x in range(256):
        t = x // 32
        lane = x % 32
        b = 0
        for bit in range(8):
            if (out_words[t * 8 + bit] >> lane) & 1:
                b |= 1 << bit
        out_bytes[x] = b
    return out_bytes


@dataclass(frozen=True)
class BenchResult:
    ns_per_eval: float
    evals_per_sec: float
    lop3_count: int


def run_cuda_bench(
    cubin: bytes,
    kernel_name: str,
    *,
    threads: int,
    block: int,
    reps: int,
    do_check: bool,
    do_check_truth_table: bool,
    ref_circuit: CircuitState | None,
) -> BenchResult:
    cuda = Cuda()
    cuda.init()
    if cuda.device_count() < 1:
        raise RuntimeError("No CUDA devices found")

    dev = cuda.device(0)
    ctx = cuda.ctx_create(dev)
    try:
        mod = cuda.module_load_data(cubin)
        fn = cuda.module_get_function(mod, kernel_name)

        words = threads * 8
        nbytes = words * 4
        d_in = cuda.mem_alloc(nbytes)
        d_out = cuda.mem_alloc(nbytes)
        try:
            host_in = (ctypes.c_uint32 * words)()
            host_out = (ctypes.c_uint32 * words)()

            if do_check_truth_table:
                if threads != 8:
                    raise RuntimeError("truth-table check requires threads=8 (8*32=256 lanes)")
                packed = _pack_truth_table_inputs_to_bitplanes()
                for i, w in enumerate(packed):
                    host_in[i] = ctypes.c_uint32(w)
            else:
                # Deterministic pseudo-random bitplanes.
                x = 0x12345678
                for i in range(words):
                    x = (1103515245 * x + 12345) & 0xFFFFFFFF
                    host_in[i] = ctypes.c_uint32(x ^ (i * 0x9E3779B9))

            cuda.memcpy_htod(d_in, host_in, nbytes)

            # Optional correctness check: compare word-for-word with CPU bitplane eval.
            if do_check:
                if ref_circuit is None:
                    raise RuntimeError("ref_circuit required for --check")
                # One launch
                grid = ((threads + block - 1) // block, 1, 1)
                arg_in = ctypes.c_uint64(d_in)
                arg_out = ctypes.c_uint64(d_out)
                arg_n = ctypes.c_uint32(threads)
                cuda.launch(
                    fn,
                    grid=grid,
                    block=(block, 1, 1),
                    args=[
                        ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_n), ctypes.c_void_p),
                    ],
                )
                cuda.memcpy_dtoh(host_out, d_out, nbytes)

                errors = 0
                for t in range(threads):
                    base = t * 8
                    in_words = [int(host_in[base + j]) for j in range(8)]
                    exp = cpu_eval_bitplanes(ref_circuit, in_words)
                    for j in range(8):
                        got = int(host_out[base + j])
                        if got != exp[j]:
                            if errors < 8:
                                raise AssertionError(
                                    f"mismatch t={t} plane={j}: got=0x{got:08x} exp=0x{exp[j]:08x}"
                                )
                            errors += 1
                if errors:
                    raise AssertionError(f"{errors} mismatches")

            if do_check_truth_table:
                # One launch, then compare all 256 outputs against AES_SBOX_TABLE.
                grid = ((threads + block - 1) // block, 1, 1)
                arg_in = ctypes.c_uint64(d_in)
                arg_out = ctypes.c_uint64(d_out)
                arg_n = ctypes.c_uint32(threads)
                cuda.launch(
                    fn,
                    grid=grid,
                    block=(block, 1, 1),
                    args=[
                        ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_n), ctypes.c_void_p),
                    ],
                )
                cuda.memcpy_dtoh(host_out, d_out, nbytes)
                out_words = [int(host_out[i]) for i in range(words)]
                out_bytes = _unpack_bitplanes_to_truth_table(out_words)
                for i in range(256):
                    if out_bytes[i] != AES_SBOX_TABLE[i]:
                        raise AssertionError(
                            f"S-box[{i}] got=0x{out_bytes[i]:02x} exp=0x{AES_SBOX_TABLE[i]:02x}"
                        )

            # Timed loop via events
            grid = ((threads + block - 1) // block, 1, 1)
            start = cuda.event_create()
            end = cuda.event_create()
            try:
                cuda.event_record(start)
                for _ in range(reps):
                    arg_in = ctypes.c_uint64(d_in)
                    arg_out = ctypes.c_uint64(d_out)
                    arg_n = ctypes.c_uint32(threads)
                    cuda.launch_async(
                        fn,
                        grid=grid,
                        block=(block, 1, 1),
                        args=[
                            ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
                            ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
                            ctypes.cast(ctypes.byref(arg_n), ctypes.c_void_p),
                        ],
                    )
                cuda.event_record(end)
                cuda.event_synchronize(end)
                ms = cuda.event_elapsed_ms(start, end)
            finally:
                cuda.event_destroy(start)
                cuda.event_destroy(end)

            seconds = ms / 1000.0
            total_evals = float(threads) * 32.0 * float(reps)
            return BenchResult(
                ns_per_eval=(seconds / total_evals) * 1e9,
                evals_per_sec=total_evals / seconds,
                lop3_count=-1,
            )
        finally:
            cuda.mem_free(d_in)
            cuda.mem_free(d_out)
    finally:
        cuda.ctx_destroy(ctx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", choices=["bp115", "bp128"], default="bp115")
    ap.add_argument(
        "--sm",
        default="sm_61",
        help="PTX target (kept at sm_61 for portability to sm_61+ by default).",
    )
    ap.add_argument("--threads", type=int, default=1_048_576)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--check", action="store_true")
    ap.add_argument(
        "--check-truth-table",
        action="store_true",
        help="Verify all 256 S-box outputs (requires --mode mem and --threads 8).",
    )
    ap.add_argument("--no-map", action="store_true", help="Skip cone→ternary mapping")
    ap.add_argument(
        "--mode",
        choices=["mem", "inline"],
        default="mem",
        help="mem=load/store bitplanes; inline=register-only loop + checksum store",
    )
    ap.add_argument("--iters", type=int, default=1, help="Inner iterations per thread (inline mode)")
    ap.add_argument(
        "--load",
        choices=["ptxas", "jit"],
        default="ptxas",
        help="ptxas=assemble PTX to cubin; jit=load PTX directly (sm_61+ portability).",
    )
    ap.add_argument(
        "--save-ptx",
        default="",
        help="Optional path to write the generated PTX for reuse in other projects.",
    )
    args = ap.parse_args()

    if args.load == "ptxas" and shutil.which("ptxas") is None:
        raise SystemExit("ptxas not found on PATH (use --load jit to avoid ptxas)")

    if args.check_truth_table:
        if args.mode != "mem":
            raise SystemExit("--check-truth-table requires --mode mem")
        if args.threads != 8:
            raise SystemExit("--check-truth-table requires --threads 8 (8*32=256 lanes)")
        if args.reps != 1:
            raise SystemExit("--check-truth-table requires --reps 1")

    if args.circuit == "bp115":
        circuit = build_bp115_sbox()
    else:
        circuit = build_bp128_sbox()

    mapped = circuit if args.no_map else map_some_cones_to_ternary(circuit)

    if args.mode == "inline":
        kernel = emit_inline_lop3_kernel(mapped, sm=args.sm)
    else:
        kernel = emit_lop3_kernel(mapped, sm=args.sm, func_name="sbox_lop3")

    if args.save_ptx:
        Path(args.save_ptx).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_ptx).write_text(kernel.ptx, encoding="utf-8")

    cubin: bytes | None
    if args.load == "ptxas":
        cubin = assemble_ptx_to_cubin(kernel.ptx, sm=args.sm)
    else:
        cubin = None

    if args.mode == "inline":
        # Inline kernel signature differs; implement a small timing loop here.
        cuda = Cuda()
        cuda.init()
        dev = cuda.device(0)
        ctx = cuda.ctx_create(dev)
        try:
            mod = cuda.module_load_data(cubin) if cubin is not None else cuda.module_load_ptx(kernel.ptx)
            fn = cuda.module_get_function(mod, kernel.kernel_name)

            nbytes = args.threads * 4
            d_out = cuda.mem_alloc(nbytes)
            try:
                grid = ((args.threads + args.block - 1) // args.block, 1, 1)
                start = cuda.event_create()
                end = cuda.event_create()
                try:
                    cuda.synchronize()
                    cuda.event_record(start)
                    for _ in range(args.reps):
                        arg_out = ctypes.c_uint64(d_out)
                        arg_n = ctypes.c_uint32(args.threads)
                        arg_i = ctypes.c_uint32(args.iters)
                        cuda.launch_async(
                            fn,
                            grid=grid,
                            block=(args.block, 1, 1),
                            args=[
                                ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
                                ctypes.cast(ctypes.byref(arg_n), ctypes.c_void_p),
                                ctypes.cast(ctypes.byref(arg_i), ctypes.c_void_p),
                            ],
                        )
                    cuda.event_record(end)
                    cuda.event_synchronize(end)
                    ms = cuda.event_elapsed_ms(start, end)
                finally:
                    cuda.event_destroy(start)
                    cuda.event_destroy(end)

                seconds = ms / 1000.0
                total_evals = float(args.threads) * 32.0 * float(args.reps) * float(args.iters)
                result = BenchResult(
                    ns_per_eval=(seconds / total_evals) * 1e9,
                    evals_per_sec=total_evals / seconds,
                    lop3_count=kernel.lop3_count,
                )
                print(f"elapsed={ms:.3f} ms")
            finally:
                cuda.mem_free(d_out)
        finally:
            cuda.ctx_destroy(ctx)
    else:
        if cubin is None:
            # JIT mode: load PTX and run; skip the ptxas path entirely.
            cuda = Cuda()
            cuda.init()
            dev = cuda.device(0)
            ctx = cuda.ctx_create(dev)
            try:
                mod = cuda.module_load_ptx(kernel.ptx)
                fn = cuda.module_get_function(mod, kernel.kernel_name)

                words = args.threads * 8
                nbytes = words * 4
                d_in = cuda.mem_alloc(nbytes)
                d_out = cuda.mem_alloc(nbytes)
                try:
                    host_in = (ctypes.c_uint32 * words)()
                    host_out = (ctypes.c_uint32 * words)()

                    if args.check_truth_table:
                        packed = _pack_truth_table_inputs_to_bitplanes()
                        for i, w in enumerate(packed):
                            host_in[i] = ctypes.c_uint32(w)
                    else:
                        x = 0x12345678
                        for i in range(words):
                            x = (1103515245 * x + 12345) & 0xFFFFFFFF
                            host_in[i] = ctypes.c_uint32(x ^ (i * 0x9E3779B9))

                    cuda.memcpy_htod(d_in, host_in, nbytes)

                    if args.check_truth_table:
                        grid = ((args.threads + args.block - 1) // args.block, 1, 1)
                        arg_in = ctypes.c_uint64(d_in)
                        arg_out = ctypes.c_uint64(d_out)
                        arg_n = ctypes.c_uint32(args.threads)
                        cuda.launch(
                            fn,
                            grid=grid,
                            block=(args.block, 1, 1),
                            args=[
                                ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
                                ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
                                ctypes.cast(ctypes.byref(arg_n), ctypes.c_void_p),
                            ],
                        )
                        cuda.memcpy_dtoh(host_out, d_out, nbytes)
                        out_words = [int(host_out[i]) for i in range(words)]
                        out_bytes = _unpack_bitplanes_to_truth_table(out_words)
                        for i in range(256):
                            if out_bytes[i] != AES_SBOX_TABLE[i]:
                                raise AssertionError(
                                    f"S-box[{i}] got=0x{out_bytes[i]:02x} exp=0x{AES_SBOX_TABLE[i]:02x}"
                                )

                    grid = ((args.threads + args.block - 1) // args.block, 1, 1)
                    start = cuda.event_create()
                    end = cuda.event_create()
                    try:
                        cuda.synchronize()
                        cuda.event_record(start)
                        for _ in range(args.reps):
                            arg_in = ctypes.c_uint64(d_in)
                            arg_out = ctypes.c_uint64(d_out)
                            arg_n = ctypes.c_uint32(args.threads)
                            cuda.launch_async(
                                fn,
                                grid=grid,
                                block=(args.block, 1, 1),
                                args=[
                                    ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
                                    ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
                                    ctypes.cast(ctypes.byref(arg_n), ctypes.c_void_p),
                                ],
                            )
                        cuda.event_record(end)
                        cuda.event_synchronize(end)
                        ms = cuda.event_elapsed_ms(start, end)
                    finally:
                        cuda.event_destroy(start)
                        cuda.event_destroy(end)

                    seconds = ms / 1000.0
                    total_evals = float(args.threads) * 32.0 * float(args.reps)
                    result = BenchResult(
                        ns_per_eval=(seconds / total_evals) * 1e9,
                        evals_per_sec=total_evals / seconds,
                        lop3_count=kernel.lop3_count,
                    )
                finally:
                    cuda.mem_free(d_in)
                    cuda.mem_free(d_out)
            finally:
                cuda.ctx_destroy(ctx)
        else:
            result = run_cuda_bench(
                cubin,
                kernel.kernel_name,
                threads=args.threads,
                block=args.block,
                reps=args.reps,
                do_check=args.check,
                do_check_truth_table=args.check_truth_table,
                ref_circuit=mapped if args.check else None,
            )

    total = args.threads * 32 * args.reps * (args.iters if args.mode == "inline" else 1)
    print(f"circuit={args.circuit} gates={mapped.gate_count} lop3={kernel.lop3_count}")
    print(f"threads={args.threads} block={args.block} reps={args.reps} total_evals={total}")
    print(f"lop3: {result.ns_per_eval:.3f} ns/eval, {result.evals_per_sec/1e9:.3f}B evals/sec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
