#!/usr/bin/env python3
"""Benchmark pressure-aware scheduling on Keccak."""
import subprocess
import tempfile
from pathlib import Path

from stc.schedule_stats_bin import read_schedule_stats_bin


def run_with_pressure(pressure_limit: int | None, out_dir: Path, flat_json: Path):
    """Compile Keccak with given pressure limit."""
    cmd = [
        ".venv/bin/python",
        "-m",
        "stc",
        str(flat_json),
        "--top",
        "keccak",
        "--out",
        str(out_dir),
        "--backend",
        "x86-avx512",
        "--force-bitsliced",
        "--bound",
        "8",
    ]
    if pressure_limit is not None:
        cmd.extend(["--max-live-pressure", str(pressure_limit)])

    subprocess.run(cmd, check=True, env={"PYTHONPATH": "."})

    stats_path = out_dir / "schedule_stats.bin"
    if stats_path.exists():
        return read_schedule_stats_bin(stats_path)
    return None


def main():
    rtl = Path("external-sha3-verilog/low_throughput_core/rtl")
    if not rtl.exists():
        raise SystemExit("Missing external-sha3-verilog checkout")

    with tempfile.TemporaryDirectory(prefix="pressure_bench_") as td:
        td = Path(td)
        flat_json = td / "keccak_flat.json"

        subprocess.run(
            [
                "yosys",
                "-q",
                "-p",
                f"read_verilog {rtl}/*.v; "
                "hierarchy -top keccak; proc; flatten; opt; opt_clean; "
                f"write_json {flat_json}",
            ],
            check=True,
        )

        print("Pressure Limit | Max Live | Spills | Cycles")
        print("-" * 50)

        for limit in [None, 256, 128, 64, 32]:
            out_dir = td / f"out_{limit}"
            out_dir.mkdir()

            stats = run_with_pressure(limit, out_dir, flat_json)
            if stats:
                limit_str = str(limit) if limit else "unlimited"
                print(
                    f"{limit_str:14} | {stats.get('max_live', 'N/A'):8} | "
                    f"{stats.get('num_spills', 'N/A'):6} | {stats.get('total_cycles', 'N/A')}"
                )


if __name__ == "__main__":
    main()
