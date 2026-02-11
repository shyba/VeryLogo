from __future__ import annotations

import argparse
import ctypes
import os
import pprint
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from stc.layout_bin import read_packed_layout_bin, read_packed_word_layout_bin
from stc.metrics_bin import read_metrics_bin


def _run(
    cmd: list[str], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, env=env)


def _have_avx2() -> bool:
    try:
        flags = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "avx2" in flags


def _have_avx512() -> bool:
    try:
        flags = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "avx512f" in flags


def _build_shared(c_path: Path, so_path: Path, *, cflags: list[str]) -> None:
    cc = os.environ.get("CC", "cc")
    _run([cc, "-O3", "-shared", "-fPIC", *cflags, str(c_path), "-o", str(so_path)])


Vec = ctypes.c_uint64 * 8


def _aligned_vec_array(n: int) -> tuple[ctypes.Array, ctypes.POINTER(Vec)]:
    raw = ctypes.create_string_buffer(n * ctypes.sizeof(Vec) + 63)
    base = ctypes.addressof(raw)
    aligned = (base + 63) & ~63
    ptr = ctypes.cast(aligned, ctypes.POINTER(Vec))
    return raw, ptr


def _benchmark_combinational(
    lib: ctypes.CDLL,
    *,
    input_io_bits: int,
    output_io_bits: int,
    steps: int,
    iters: int,
) -> dict:
    lib.circuit.argtypes = [ctypes.POINTER(Vec), ctypes.POINTER(Vec)]
    lib.circuit.restype = None

    in_raw, in_ptr = _aligned_vec_array(input_io_bits)
    out_raw, out_ptr = _aligned_vec_array(output_io_bits)

    in_arr = ctypes.cast(in_ptr, ctypes.POINTER(Vec * input_io_bits)).contents
    ZEROS = Vec(*([0] * 8))
    for i in range(input_io_bits):
        in_arr[i] = ZEROS

    lib.circuit(in_ptr, out_ptr)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for _ in range(steps):
            lib.circuit(in_ptr, out_ptr)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    min_time = min(times)
    avg_time = sum(times) / len(times)
    time_per_step_s = min_time / steps
    time_per_step_us = time_per_step_s * 1e6
    throughput = 1.0 / time_per_step_s if time_per_step_s > 0 else 0

    return {
        "time_per_step_us": time_per_step_us,
        "throughput_steps_per_s": throughput,
        "total_time_s": avg_time,
        "min_time_s": min_time,
        "avg_time_s": avg_time,
    }


def _benchmark_steps_shared(
    lib: ctypes.CDLL,
    *,
    input_io_bits: int,
    state_bits: int,
    output_io_bits: int,
    steps: int,
    iters: int,
) -> dict:
    if not hasattr(lib, "circuit_steps_shared"):
        if state_bits == 0:
            return _benchmark_combinational(
                lib,
                input_io_bits=input_io_bits,
                output_io_bits=output_io_bits,
                steps=steps,
                iters=iters,
            )
        raise RuntimeError("missing symbol: circuit_steps_shared")

    lib.circuit_steps_shared.argtypes = [
        ctypes.POINTER(Vec),
        ctypes.POINTER(Vec),
        ctypes.POINTER(Vec),
        ctypes.POINTER(Vec),
        ctypes.c_int,
    ]
    lib.circuit_steps_shared.restype = None

    in_raw, in_ptr = _aligned_vec_array(input_io_bits)
    out_raw, out_ptr = _aligned_vec_array(output_io_bits)
    st0_raw, st0_ptr = _aligned_vec_array(state_bits)
    st1_raw, st1_ptr = _aligned_vec_array(state_bits)

    in_arr = ctypes.cast(in_ptr, ctypes.POINTER(Vec * input_io_bits)).contents
    st0_arr = ctypes.cast(st0_ptr, ctypes.POINTER(Vec * state_bits)).contents
    st1_arr = ctypes.cast(st1_ptr, ctypes.POINTER(Vec * state_bits)).contents

    ZEROS = Vec(*([0] * 8))
    for i in range(input_io_bits):
        in_arr[i] = ZEROS
    for i in range(state_bits):
        st0_arr[i] = ZEROS
        st1_arr[i] = ZEROS

    lib.circuit_steps_shared(in_ptr, out_ptr, st0_ptr, st1_ptr, 1)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        lib.circuit_steps_shared(in_ptr, out_ptr, st0_ptr, st1_ptr, steps)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    min_time = min(times)
    avg_time = sum(times) / len(times)
    time_per_step_s = min_time / steps
    time_per_step_us = time_per_step_s * 1e6
    throughput = 1.0 / time_per_step_s if time_per_step_s > 0 else 0

    return {
        "time_per_step_us": time_per_step_us,
        "throughput_steps_per_s": throughput,
        "total_time_s": avg_time,
        "min_time_s": min_time,
        "avg_time_s": avg_time,
    }


def _have_cuda() -> bool:
    try:
        import stc.cuda_driver

        cuda = stc.cuda_driver.Cuda()
        cuda.init()
        return cuda.device_count() > 0
    except Exception:
        return False


def _get_backend_cflags(backend: str) -> tuple[list[str], str]:
    if backend == "x86-avx512":
        if not _have_avx512():
            raise SystemExit("CPU lacks AVX-512 support")
        return (
            ["-mavx512f", "-mavx512vl", "-mavx512dq", "-mavx512bw"],
            "avx512",
        )
    elif backend == "x86-avx2":
        if not _have_avx2():
            raise SystemExit("CPU lacks AVX2 support")
        return (["-mavx2"], "avx2")
    elif backend == "ptx":
        if not _have_cuda():
            raise SystemExit("CUDA not available")
        return ([], "ptx")
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _find_rust_bin(candidates: list[str]) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise SystemExit(f"Missing required binary: {candidates[0]}")


def _parse_circuit_offsets(c_path: Path) -> tuple[int, int]:
    content = c_path.read_text(encoding="utf-8")
    import re

    match = re.search(
        r"void circuit\(__m512i\* in, __m512i\* out\) \{\s*circuit__core\(in, in \+ (\d+), out, out \+ (\d+)\);",
        content,
    )
    if not match:
        raise ValueError("Could not parse circuit offsets from generated code")
    input_io_words = int(match.group(1))
    output_io_words = int(match.group(2))
    return input_io_words, output_io_words


def _get_metrics_from_bin(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {}
    data = read_metrics_bin(metrics_path)
    return {
        "gates": data.get("gates", 0),
        "depth": data.get("depth", 0),
        "nodes": data.get("nodes", 0),
    }


def _benchmark_ptx(
    ptx_code: str,
    *,
    input_io_words: int,
    state_words: int,
    output_io_words: int,
    threads: int,
    block: int,
    reps: int,
) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from stc.cuda_driver import Cuda

    cuda = Cuda()
    cuda.init()
    dev = cuda.device(0)
    ctx = cuda.ctx_create(dev)

    try:
        mod = cuda.module_load_ptx(ptx_code)
        fn = cuda.module_get_function(mod, "stc_eval")

        n = threads
        h_in = (ctypes.c_uint32 * max(1, n * input_io_words))()
        h_out = (ctypes.c_uint32 * max(1, n * output_io_words))()
        h_st0 = (ctypes.c_uint32 * max(1, n * state_words))()
        h_st1 = (ctypes.c_uint32 * max(1, n * state_words))()

        for i in range(len(h_in)):
            h_in[i] = ctypes.c_uint32(0)
        for i in range(len(h_st0)):
            h_st0[i] = ctypes.c_uint32(0)
            h_st1[i] = ctypes.c_uint32(0)

        d_in = cuda.mem_alloc(ctypes.sizeof(h_in))
        d_out = cuda.mem_alloc(ctypes.sizeof(h_out))
        d_st0 = cuda.mem_alloc(ctypes.sizeof(h_st0))
        d_st1 = cuda.mem_alloc(ctypes.sizeof(h_st1))

        try:
            cuda.memcpy_htod(d_in, h_in, ctypes.sizeof(h_in))
            cuda.memcpy_htod(d_out, h_out, ctypes.sizeof(h_out))
            cuda.memcpy_htod(d_st0, h_st0, ctypes.sizeof(h_st0))
            cuda.memcpy_htod(d_st1, h_st1, ctypes.sizeof(h_st1))

            def launch() -> None:
                arg_in = ctypes.c_uint64(d_in)
                arg_st_in = ctypes.c_uint64(d_st0)
                arg_out = ctypes.c_uint64(d_out)
                arg_st_out = ctypes.c_uint64(d_st1)
                arg_n = ctypes.c_uint32(n)
                kernel_args = [
                    ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_st_in), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_st_out), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_n), ctypes.c_void_p),
                ]
                grid = ((n + block - 1) // block, 1, 1)
                cuda.launch_async(fn, grid, (block, 1, 1), kernel_args)

            launch()
            cuda.synchronize()

            times = []
            for _ in range(reps):
                t0 = time.perf_counter()
                launch()
                cuda.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)

            min_time = min(times)
            avg_time = sum(times) / len(times)
            throughput = threads / min_time if min_time > 0 else 0

            return {
                "time_per_iter_us": (min_time * 1e6),
                "throughput_evals_per_s": throughput,
                "total_time_s": avg_time,
                "min_time_s": min_time,
                "avg_time_s": avg_time,
            }
        finally:
            cuda.mem_free(d_in)
            cuda.mem_free(d_out)
            cuda.mem_free(d_st0)
            cuda.mem_free(d_st1)
    finally:
        cuda.ctx_destroy(ctx)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generic benchmark harness for VeryLogo pipeline"
    )
    ap.add_argument("--case", type=str, help="Case name (e.g., vec_add)")
    ap.add_argument(
        "--input", type=Path, help="Input Verilog or JSON (alternative to --case)"
    )
    ap.add_argument("--top", type=str, default=None, help="Top module name")
    ap.add_argument(
        "--backend",
        type=str,
        default="x86-avx512",
        choices=["x86-avx512", "x86-avx2", "ptx"],
        help="Target backend (default: x86-avx512)",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of steps per iteration (CPU only)",
    )
    ap.add_argument("--reps", type=int, default=100, help="Number of repetitions")
    ap.add_argument(
        "--threads", type=int, default=1048576, help="Number of threads (PTX only)"
    )
    ap.add_argument("--block", type=int, default=256, help="Block size (PTX only)")
    ap.add_argument(
        "--bound", type=int, default=8, help="Bound for equivalence checking"
    )
    ap.add_argument(
        "--abc-lut3-aggressive",
        action="store_true",
        default=False,
        help="Use aggressive ABC LUT3 script during Yosys normalization",
    )
    ap.add_argument(
        "--scheduler",
        choices=["list", "pipelined", "serial"],
        default=None,
        help="Scheduling algorithm for CPU backends (default: list)",
    )
    ap.add_argument(
        "--max-live-pressure",
        type=int,
        default=None,
        help="Cap register pressure for list scheduler (CPU backends)",
    )
    ap.add_argument(
        "--no-compile", action="store_true", help="Skip compilation, only benchmark"
    )
    ap.add_argument(
        "--rust-only",
        action="store_true",
        default=False,
        help="Use Rust lower + sched emit (skip Python coordinate_lowering)",
    )
    ap.add_argument(
        "--packed-bitslice",
        action="store_true",
        default=False,
        help="Use packed bitslice lowering (bit-level -> packed u64)",
    )
    return ap.parse_args(argv)


def main() -> int:
    args = parse_args(sys.argv[1:])

    if args.case and args.input:
        raise SystemExit("Cannot specify both --case and --input")
    if not args.case and not args.input:
        raise SystemExit("Must specify either --case or --input")

    if args.case:
        input_path = Path(f"fixtures/verilog_stress/{args.case}.v")
        if not input_path.exists():
            raise SystemExit(f"Fixture not found: {input_path}")
        top_module = args.top if args.top else args.case
        design_name = args.case
        out_dir = Path(f"out/{args.case}")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_report = out_dir / "bench.txt"
        cleanup = False
    else:
        input_path = args.input
        if not input_path.exists():
            raise SystemExit(f"Input file not found: {input_path}")
        top_module = args.top
        design_name = input_path.stem
        work_dir = Path(tempfile.mkdtemp(prefix="stc_bench_"))
        out_dir = work_dir / "out"
        out_dir.mkdir(exist_ok=True)
        output_report = None
        cleanup = True

    python = os.environ.get("PYTHON", ".venv/bin/python")
    cflags, target = _get_backend_cflags(args.backend)

    try:

        if not args.no_compile:
            compile_start = time.perf_counter()

            stc_cmd = [
                python,
                "-m",
                "stc",
                str(input_path),
                "--out",
                str(out_dir),
                "--backend",
                args.backend,
                "--bound",
                str(args.bound),
            ]
            if top_module:
                stc_cmd += ["--top", top_module]
            if args.abc_lut3_aggressive:
                stc_cmd += ["--abc-lut3-aggressive"]
            if args.packed_bitslice:
                stc_cmd += ["--packed-bitslice"]
            if args.scheduler:
                stc_cmd += ["--scheduler", args.scheduler]
            if args.max_live_pressure is not None:
                stc_cmd += ["--max-live-pressure", str(args.max_live_pressure)]
            if args.rust_only:
                stc_cmd += ["--no-backend"]

            _run(stc_cmd, env={**os.environ, "PYTHONPATH": "."})

            if args.rust_only and args.backend in {"x86-avx2", "x86-avx512"}:
                rust_lower = _find_rust_bin(
                    [
                        "rust/tick_lower_rs/target/release/tick_lower_rs",
                        "rust/tick_lower_rs/target/debug/tick_lower_rs",
                    ]
                )
                sched_emit = _find_rust_bin(
                    [
                        "rust/sched_emit_rs/target/release/sched_emit_rs",
                        "rust/sched_emit_rs/target/debug/sched_emit_rs",
                    ]
                )
                tick_ir_path = out_dir / "reduced_tick_ir.bin"
                circuit_bin = out_dir / "circuit_state.bin"
                _run(
                    [
                        rust_lower,
                        "--input",
                        str(tick_ir_path),
                        "--output",
                        str(circuit_bin),
                        "--format",
                        "bin",
                        "--output-format",
                        "bin",
                    ]
                )
                out_c = out_dir / f"circuit_{target}.c"
                sched_cmd = [
                    sched_emit,
                    "--input",
                    str(circuit_bin),
                    "--output",
                    str(out_c),
                    "--target",
                    target,
                    "--scheduler",
                    args.scheduler or "list",
                    "--format",
                    "bin",
                ]
                if args.max_live_pressure is not None:
                    sched_cmd += ["--max-live-pressure", str(args.max_live_pressure)]
                _run(sched_cmd)

            compile_end = time.perf_counter()
            compile_time = compile_end - compile_start
        else:
            compile_time = 0.0

        if args.backend == "ptx":
            ptx_path = out_dir / "circuit.ptx"
            if not ptx_path.exists():
                raise SystemExit(f"Generated PTX file not found: {ptx_path}")

            ptx_code = ptx_path.read_text(encoding="utf-8")

            def _words_for_width(w: int) -> int:
                return (w + 31) // 32

            io_layout_path = out_dir / "io_layout.bin"
            if io_layout_path.exists():
                layout = read_packed_layout_bin(io_layout_path).to_dict()
                input_io_bits = sum(int(v["width"]) for v in layout["inputs"].values())
                state_bits = sum(int(v["width"]) for v in layout["state"].values())
                output_io_bits = sum(
                    int(v["width"]) for v in layout["outputs"].values()
                )
                input_io_words = (
                    _words_for_width(input_io_bits) if input_io_bits > 0 else 0
                )
                state_words = _words_for_width(state_bits) if state_bits > 0 else 0
                output_io_words = (
                    _words_for_width(output_io_bits) if output_io_bits > 0 else 0
                )
            else:
                raise SystemExit(f"Layout file not found: {io_layout_path}")

            print(f"Design: {design_name}")
            print(f"Backend: {args.backend}")
            print(f"Input words: {input_io_words}")
            print(f"State words: {state_words}")
            print(f"Output words: {output_io_words}")
            print(f"Threads: {args.threads}")
            print(f"Block size: {args.block}")
            print(f"Repetitions: {args.reps}")
            if not args.no_compile:
                print(f"Compilation time: {compile_time:.2f}s")
            print()

            bench_results = _benchmark_ptx(
                ptx_code,
                input_io_words=input_io_words,
                state_words=state_words,
                output_io_words=output_io_words,
                threads=args.threads,
                block=args.block,
                reps=args.reps,
            )

            print("Benchmark Results:")
            print(f"  Time per iteration: {bench_results['time_per_iter_us']:.3f} us")
            print(
                f"  Throughput: {bench_results['throughput_evals_per_s']:.0f} evals/s"
            )
            print(f"  Min time: {bench_results['min_time_s']:.6f}s")
            print(f"  Avg time: {bench_results['avg_time_s']:.6f}s")

            reduced_metrics = _get_metrics_from_bin(out_dir / "reduced_metrics.bin")
            code_size = ptx_path.stat().st_size if ptx_path.exists() else 0

            output_data = {
                "design": design_name,
                "backend": args.backend,
                "threads": args.threads,
                "block": args.block,
                "reps": args.reps,
                "results": bench_results,
                "compilation": {
                    "time_s": compile_time,
                    "gates": reduced_metrics.get("gates", 0),
                    "depth": reduced_metrics.get("depth", 0),
                    "nodes": reduced_metrics.get("nodes", 0),
                    "code_size_bytes": code_size,
                },
                "layout": {
                    "input_words": input_io_words,
                    "state_words": state_words,
                    "output_words": output_io_words,
                },
            }
        else:
            c_path_packed = out_dir / f"circuit_{target}_u64.c"
            c_path_bitsliced = out_dir / f"circuit_{target}.c"

            if c_path_packed.exists():
                c_path = c_path_packed
                layout_path = out_dir / "packed_word_layout.bin"
                layout = read_packed_word_layout_bin(layout_path)
                input_io_words, output_io_words = _parse_circuit_offsets(c_path)
                total_input_words = int(layout.input_words)
                total_output_words = int(layout.output_words)
                input_io_bits = input_io_words
                state_bits = total_input_words - input_io_words
                output_io_bits = output_io_words
            elif c_path_bitsliced.exists():
                c_path = c_path_bitsliced
                layout_path = out_dir / "io_layout.bin"
                if args.rust_only and not layout_path.exists():
                    layout_path = out_dir / "circuit_state_layout.bin"
                layout = read_packed_layout_bin(layout_path).to_dict()
                input_io_bits = sum(int(v["width"]) for v in layout["inputs"].values())
                state_bits = sum(int(v["width"]) for v in layout["state"].values())
                output_io_bits = sum(
                    int(v["width"]) for v in layout["outputs"].values()
                )
            else:
                raise SystemExit(
                    f"Generated C file not found: {c_path_bitsliced} or {c_path_packed}"
                )

            so_path = c_path.with_suffix(".so")
            _build_shared(c_path, so_path, cflags=cflags)
            lib = ctypes.CDLL(str(so_path))

            print(f"Design: {design_name}")
            print(f"Backend: {args.backend}")
            print(f"Input bits: {input_io_bits}")
            print(f"State bits: {state_bits}")
            print(f"Output bits: {output_io_bits}")
            print(f"Steps: {args.steps}")
            print(f"Repetitions: {args.reps}")
            if not args.no_compile:
                print(f"Compilation time: {compile_time:.2f}s")
            print()

            bench_results = _benchmark_steps_shared(
                lib,
                input_io_bits=input_io_bits,
                state_bits=state_bits,
                output_io_bits=output_io_bits,
                steps=args.steps,
                iters=args.reps,
            )

            print("Benchmark Results:")
            print(f"  Time per step: {bench_results['time_per_step_us']:.3f} us")
            print(
                f"  Throughput: {bench_results['throughput_steps_per_s']:.0f} steps/s"
            )
            print(f"  Min time: {bench_results['min_time_s']:.6f}s")
            print(f"  Avg time: {bench_results['avg_time_s']:.6f}s")

            reduced_metrics = _get_metrics_from_bin(out_dir / "reduced_metrics.bin")
            code_size = c_path.stat().st_size if c_path.exists() else 0

            output_data = {
                "design": design_name,
                "backend": args.backend,
                "steps": args.steps,
                "reps": args.reps,
                "results": bench_results,
                "compilation": {
                    "time_s": compile_time,
                    "gates": reduced_metrics.get("gates", 0),
                    "depth": reduced_metrics.get("depth", 0),
                    "nodes": reduced_metrics.get("nodes", 0),
                    "code_size_bytes": code_size,
                },
                "layout": {
                    "input_bits": input_io_bits,
                    "state_bits": state_bits,
                    "output_bits": output_io_bits,
                },
            }

        if output_report:
            output_report.parent.mkdir(parents=True, exist_ok=True)
            output_report.write_text(
                pprint.pformat(output_data, sort_dicts=True) + "\n",
                encoding="utf-8",
            )
            print(f"\nResults written to: {output_report}")

    finally:
        if cleanup:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
