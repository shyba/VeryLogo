from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModeResult:
    mode: str
    best_eval_b: float
    best_mib_s: float
    best_block: int


def _parse_best(stdout: str) -> tuple[float, float, int]:
    best_eval = -1.0
    best_mib = -1.0
    best_block = -1
    current_block = -1
    for line in stdout.splitlines():
        m_block = re.search(r"threads=\d+\s+block=(\d+)\s+reps=\d+", line)
        if m_block:
            current_block = int(m_block.group(1))
            continue
        m_perf = re.search(
            r"([0-9]+(?:\.[0-9]+)?)B evals/sec,\s+([0-9]+(?:\.[0-9]+)?) MiB/s", line
        )
        if m_perf and current_block >= 0:
            eval_b = float(m_perf.group(1))
            mib = float(m_perf.group(2))
            if eval_b > best_eval:
                best_eval = eval_b
                best_mib = mib
                best_block = current_block
    if best_eval < 0:
        raise RuntimeError("failed to parse benchmark output")
    return best_eval, best_mib, best_block


def _run_mode(
    script: Path,
    mode: str,
    threads: int,
    reps: int,
    sm: str,
    do_check: bool,
) -> ModeResult:
    cmd = [
        sys.executable,
        str(script),
        "--kernel-mode",
        mode,
        "--threads",
        str(threads),
        "--reps",
        str(reps),
        "--sm",
        sm,
    ]
    if do_check:
        cmd.append("--check")
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if run.returncode != 0:
        raise RuntimeError(
            f"{mode} failed (rc={run.returncode})\nSTDOUT:\n{run.stdout}\nSTDERR:\n{run.stderr}"
        )
    eval_b, mib_s, block = _parse_best(run.stdout)
    return ModeResult(mode=mode, best_eval_b=eval_b, best_mib_s=mib_s, best_block=block)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=65_536)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--sm", default="sm_61")
    ap.add_argument(
        "--check",
        action="store_true",
        help="Run correctness check on the first mode only to reduce runtime",
    )
    args = ap.parse_args()

    smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if smi.returncode != 0:
        print("nvidia-smi -L failed; GPU/NVML not ready for valid benchmark runs")
        if smi.stderr.strip():
            print(smi.stderr.strip())
        return 2

    script = Path(__file__).with_name("bench_aes10_bp128_cuda.py")
    if not script.exists():
        raise SystemExit(f"missing {script}")

    modes = [
        "legacy_tuned",
        "replacement_tuned",
        "replacement_coalesced_tuned",
        "replacement_coalesced4_tuned",
    ]

    results: list[ModeResult] = []
    for idx, mode in enumerate(modes):
        try:
            result = _run_mode(
                script=script,
                mode=mode,
                threads=args.threads,
                reps=args.reps,
                sm=args.sm,
                do_check=(args.check and idx == 0),
            )
        except RuntimeError as exc:
            print(str(exc))
            return 2
        results.append(result)

    print(f"threads={args.threads} reps={args.reps} sm={args.sm}")
    print("| mode | best block | B eval/s | MiB/s |")
    print("|---|---:|---:|---:|")
    for row in results:
        print(
            f"| `{row.mode}` | {row.best_block} | {row.best_eval_b:.3f} | {row.best_mib_s:.2f} |"
        )

    baseline = next((row for row in results if row.mode == "replacement_tuned"), None)
    target = next(
        (row for row in results if row.mode == "replacement_coalesced4_tuned"), None
    )
    if baseline and target and baseline.best_eval_b > 0:
        speedup = target.best_eval_b / baseline.best_eval_b
        print(f"\ncoalesced4 vs replacement speedup: {speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
