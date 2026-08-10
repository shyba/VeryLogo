#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from statistics import median
from typing import Any

from stc.cli import run_pipeline


def _mask(width: int) -> int:
    if width < 1:
        raise ValueError(f"invalid bit width: {width}")
    return (1 << width) - 1


def _rand_bool_array(n: int, rng: random.Random) -> str:
    return (
        "["
        + ", ".join("true" if rng.getrandbits(1) else "false" for _ in range(n))
        + "]"
    )


def _rand_u64_array(n: int, width: int, rng: random.Random) -> str:
    m = _mask(width)
    return "[" + ", ".join(f"{rng.getrandbits(width) & m}u64" for _ in range(n)) + "]"


def _rand_chunk_array(n: int, chunks: int, rng: random.Random) -> str:
    rows: list[str] = []
    for _ in range(n):
        words = ", ".join(f"{rng.getrandbits(64)}u64" for _ in range(chunks))
        rows.append(f"[{words}]")
    return "[" + ", ".join(rows) + "]"


def _rand_soa_arg(desc: dict[str, Any], n: int, rng: random.Random) -> str:
    kind = str(desc["kind"])
    width = int(desc["width"])
    if kind == "bool":
        return _rand_bool_array(n, rng)
    if width <= 64:
        return _rand_u64_array(n, width, rng)
    chunks = int(desc["chunks"])
    return _rand_chunk_array(n, chunks, rng)


def _rand_seq_soa_arg(
    desc: dict[str, Any], steps: int, n: int, rng: random.Random
) -> str:
    return "[" + ", ".join(_rand_soa_arg(desc, n, rng) for _ in range(steps)) + "]"


def _choose_entry(manifest: dict[str, Any], requested: str) -> str:
    entries = dict(manifest.get("entries", {}))
    if requested != "auto":
        if requested not in entries:
            raise ValueError(
                f"entry {requested!r} not present in manifest entries={sorted(entries.keys())}"
            )
        return requested
    if "eval_batch" in entries:
        return "eval_batch"
    if "step_batch" in entries:
        return "step_batch"
    if "run_steps_batch" in entries:
        return "run_steps_batch"
    raise ValueError("manifest contains no runnable entry")


def _build_stdin(
    *,
    entry: str,
    manifest: dict[str, Any],
    n: int,
    steps: int,
    rng: random.Random,
) -> str:
    state = list(manifest.get("state", []))
    inputs = list(manifest.get("inputs", []))
    lines: list[str] = []
    if entry == "run_steps_batch":
        lines.append(f"{steps}i64")
        lines.append(f"{n}i64")
        lines.extend(_rand_soa_arg(s, n, rng) for s in state)
        lines.extend(_rand_seq_soa_arg(inp, steps, n, rng) for inp in inputs)
    else:
        lines.append(f"{n}i64")
        lines.extend(_rand_soa_arg(s, n, rng) for s in state)
        lines.extend(_rand_soa_arg(inp, n, rng) for inp in inputs)
    return "\n".join(lines) + "\n"


def _sum_output_bits(manifest: dict[str, Any]) -> int:
    return sum(int(out["width"]) for out in manifest.get("outputs", []))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark throughput of generated Futhark backend entries."
    )
    ap.add_argument("input", type=Path, help="Verilog file/dir or normalized.json")
    ap.add_argument("--top", type=str, default=None)
    ap.add_argument(
        "--out", type=Path, default=None, help="Pipeline output dir (default: temp)"
    )
    ap.add_argument("--bound", type=int, default=8)
    ap.add_argument(
        "--futhark-mode",
        choices=["auto", "combinational_fast", "step_legacy"],
        default="auto",
    )
    ap.add_argument(
        "--entry",
        choices=["auto", "eval_batch", "step_batch", "run_steps_batch"],
        default="auto",
    )
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip-compile", action="store_true", default=False)
    ap.add_argument(
        "--json",
        type=Path,
        default=Path("out/futhark_bench.json"),
        help="Write benchmark report JSON here.",
    )
    ns = ap.parse_args()

    if shutil.which("futhark") is None:
        raise SystemExit("futhark not found on PATH")
    if ns.batch < 1:
        raise SystemExit("--batch must be >= 1")
    if ns.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if ns.runs < 1:
        raise SystemExit("--runs must be >= 1")

    tmp_dir: tempfile.TemporaryDirectory[str] | None = None
    if ns.out is None:
        tmp_dir = tempfile.TemporaryDirectory(prefix="stc_futhark_bench_")
        out_dir = Path(tmp_dir.name)
    else:
        out_dir = ns.out
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_pipeline(
            ns.input,
            out_dir,
            top=ns.top,
            bound=ns.bound,
            backend="futhark",
            futhark_mode=ns.futhark_mode,
        )
        fut_path = out_dir / "circuit_futhark.fut"
        manifest_path = out_dir / "futhark_io_manifest.json"
        if not fut_path.exists() or not manifest_path.exists():
            raise SystemExit("missing generated Futhark artifacts")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = _choose_entry(manifest, ns.entry)
        exe_path = out_dir / "circuit_futhark"

        compile_seconds = 0.0
        if not ns.skip_compile or not exe_path.exists():
            t0 = time.perf_counter()
            subprocess.run(["futhark", "c", str(fut_path)], check=True)
            compile_seconds = time.perf_counter() - t0

        rng = random.Random(ns.seed)
        stdin_data = _build_stdin(
            entry=entry,
            manifest=manifest,
            n=int(ns.batch),
            steps=int(ns.steps),
            rng=rng,
        )

        runtime_path = out_dir / "futhark_runtime_us.txt"
        subprocess.run(
            [
                str(exe_path),
                "-e",
                entry,
                "-n",
                "-r",
                str(ns.runs),
                "-t",
                str(runtime_path),
            ],
            input=stdin_data,
            text=True,
            check=True,
        )
        samples_us = [
            float(line.strip())
            for line in runtime_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not samples_us:
            raise SystemExit("no runtime samples were written by futhark executable")

        avg_us = sum(samples_us) / len(samples_us)
        med_us = median(samples_us)
        evals_per_call = int(ns.batch) * (
            int(ns.steps) if entry == "run_steps_batch" else 1
        )
        eval_s = evals_per_call / (avg_us * 1e-6)
        ns_per_eval = (avg_us * 1000.0) / evals_per_call
        out_bits = _sum_output_bits(manifest)
        bytes_per_eval = out_bits / 8.0
        mib_s = (eval_s * bytes_per_eval) / (1024.0 * 1024.0)

        report = {
            "input": str(ns.input),
            "top": ns.top,
            "entry": entry,
            "mode": manifest.get("mode"),
            "batch": int(ns.batch),
            "steps": int(ns.steps),
            "runs": int(ns.runs),
            "compile_seconds": compile_seconds,
            "samples": len(samples_us),
            "avg_us_per_call": avg_us,
            "median_us_per_call": med_us,
            "evals_per_call": evals_per_call,
            "eval_per_s": eval_s,
            "ns_per_eval": ns_per_eval,
            "output_bits_per_eval": out_bits,
            "bytes_per_eval": bytes_per_eval,
            "mib_per_s": mib_s,
            "out_dir": str(out_dir),
        }

        ns.json.parent.mkdir(parents=True, exist_ok=True)
        ns.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        print(f"entry: {entry}")
        print(f"avg: {avg_us:.3f} us/call  median: {med_us:.3f} us/call")
        print(f"throughput: {eval_s:.2f} eval/s  {mib_s:.2f} MiB/s")
        print(f"ns/eval: {ns_per_eval:.3f}")
        print(f"report: {ns.json}")
        return 0
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
