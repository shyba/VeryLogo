#!/usr/bin/env python3
"""Benchmark ternary synthesis gate reduction on Keccak."""
import json
import subprocess
import tempfile
from pathlib import Path


def main():
    rtl = Path("external-sha3-verilog/low_throughput_core/rtl")
    if not rtl.exists():
        raise SystemExit("Missing external-sha3-verilog checkout")

    with tempfile.TemporaryDirectory(prefix="ternary_bench_") as td:
        td = Path(td)
        flat_json = td / "keccak_flat.json"
        out_dir = td / "out"
        out_dir.mkdir()

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

        subprocess.run(
            [
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
            ],
            check=True,
            env={"PYTHONPATH": "."},
        )

        stats_path = out_dir / "ternary_synth_stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            print(f"Gates before ternary: {stats['gates_before']}")
            print(f"Gates after ternary:  {stats['gates_after']}")
            print(f"Ternary gates created: {stats['ternary_gates_created']}")
            print(f"Reduction: {stats['reduction_percent']}%")
            print(f"Patterns: {stats['patterns_found']}")
        else:
            print("ERROR: ternary_synth_stats.json not found")

        c_path = out_dir / "circuit_avx512.c"
        if c_path.exists():
            c_code = c_path.read_text()
            ternary_count = c_code.count("ternarylogic")
            print(f"VPTERNLOG calls in generated C: {ternary_count}")


if __name__ == "__main__":
    main()
