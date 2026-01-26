from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


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
    else:
        raise ValueError(f"Unsupported backend: {backend}")


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


def _get_metrics_from_json(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {}
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "gates": data.get("gates", 0),
        "depth": data.get("depth", 0),
        "nodes": data.get("nodes", 0),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generic benchmark harness for VeryLogo pipeline"
    )
    ap.add_argument("--input", type=Path, required=True, help="Input Verilog or JSON")
    ap.add_argument("--top", type=str, default=None, help="Top module name")
    ap.add_argument(
        "--backend",
        type=str,
        default="x86-avx512",
        choices=["x86-avx512", "x86-avx2"],
        help="Target backend (default: x86-avx512)",
    )
    ap.add_argument(
        "--steps", type=int, default=1000, help="Number of steps per iteration"
    )
    ap.add_argument("--iters", type=int, default=100, help="Number of iterations")
    ap.add_argument(
        "--output", type=Path, default=None, help="Output JSON file (optional)"
    )
    ap.add_argument(
        "--bound", type=int, default=8, help="Bound for equivalence checking"
    )
    ap.add_argument(
        "--no-compile", action="store_true", help="Skip compilation, only benchmark"
    )
    ap.add_argument(
        "--work-dir", type=Path, default=None, help="Working directory (default: tmp)"
    )
    return ap.parse_args(argv)


def main() -> int:
    import sys

    args = parse_args(sys.argv[1:])

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    python = os.environ.get("PYTHON", ".venv/bin/python")
    cflags, target = _get_backend_cflags(args.backend)

    if args.work_dir is not None:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="stc_bench_"))
        cleanup = True

    try:
        out_dir = work_dir / "out"
        out_dir.mkdir(exist_ok=True)

        design_name = args.input.stem

        if not args.no_compile:
            compile_start = time.perf_counter()

            _run(
                [
                    python,
                    "-m",
                    "stc",
                    str(args.input),
                    "--out",
                    str(out_dir),
                    "--backend",
                    args.backend,
                    "--bound",
                    str(args.bound),
                ]
                + (["--top", args.top] if args.top else []),
                env={**os.environ, "PYTHONPATH": "."},
            )

            compile_end = time.perf_counter()
            compile_time = compile_end - compile_start
        else:
            compile_time = 0.0

        c_path_packed = out_dir / f"circuit_{target}_u64.c"
        c_path_bitsliced = out_dir / f"circuit_{target}.c"

        if c_path_packed.exists():
            c_path = c_path_packed
            layout_path = out_dir / "packed_word_layout.json"
            layout = json.loads(layout_path.read_text(encoding="utf-8"))
            input_io_words, output_io_words = _parse_circuit_offsets(c_path)
            total_input_words = int(layout["input_words"])
            total_output_words = int(layout["output_words"])
            input_io_bits = input_io_words
            state_bits = total_input_words - input_io_words
            output_io_bits = output_io_words
        elif c_path_bitsliced.exists():
            c_path = c_path_bitsliced
            layout_path = out_dir / "io_layout.json"
            layout = json.loads(layout_path.read_text(encoding="utf-8"))
            input_io_bits = sum(int(v["width"]) for v in layout["inputs"].values())
            state_bits = sum(int(v["width"]) for v in layout["state"].values())
            output_io_bits = sum(int(v["width"]) for v in layout["outputs"].values())
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
        print(f"Iterations: {args.iters}")
        if not args.no_compile:
            print(f"Compilation time: {compile_time:.2f}s")
        print()

        bench_results = _benchmark_steps_shared(
            lib,
            input_io_bits=input_io_bits,
            state_bits=state_bits,
            output_io_bits=output_io_bits,
            steps=args.steps,
            iters=args.iters,
        )

        print("Benchmark Results:")
        print(f"  Time per step: {bench_results['time_per_step_us']:.3f} us")
        print(f"  Throughput: {bench_results['throughput_steps_per_s']:.0f} steps/s")
        print(f"  Min iteration time: {bench_results['min_time_s']:.6f}s")
        print(f"  Avg iteration time: {bench_results['avg_time_s']:.6f}s")

        reduced_metrics = _get_metrics_from_json(out_dir / "reduced_metrics.json")
        code_size = c_path.stat().st_size if c_path.exists() else 0

        output_data = {
            "design": design_name,
            "backend": args.backend,
            "steps": args.steps,
            "iters": args.iters,
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

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(output_data, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"\nResults written to: {args.output}")

    finally:
        if cleanup:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
