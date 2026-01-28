#!/usr/bin/env python3
"""Compare fused vs unfused Keccak performance."""
import subprocess
import tempfile
import time
from pathlib import Path


def compile_keccak(out_dir: Path, flat_json: Path, fuse_ticks: int):
    """Compile Keccak with given fusion level."""
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
    if fuse_ticks > 1:
        cmd.extend(["--fuse-ticks", str(fuse_ticks), "--fuse-mode", "final"])

    start = time.perf_counter()
    subprocess.run(
        cmd,
        check=True,
        env={"PYTHONPATH": "."},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    compile_time = time.perf_counter() - start
    return compile_time


def main():
    rtl = Path("external-sha3-verilog/low_throughput_core/rtl")
    if not rtl.exists():
        raise SystemExit("Missing external-sha3-verilog checkout")

    with tempfile.TemporaryDirectory(prefix="fusion_bench_") as td:
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

        print("Fusion Level | Compile Time | Gate Count")
        print("-" * 45)

        for fuse in [1, 2, 4, 8]:
            out_dir = td / f"out_fuse{fuse}"
            out_dir.mkdir()

            compile_time = compile_keccak(out_dir, flat_json, fuse)

            # Count gates in generated circuit
            import json

            circuit_path = out_dir / "circuit_state.json"
            if circuit_path.exists():
                circuit = json.loads(circuit_path.read_text())
                gate_count = len(circuit.get("gates", []))
            else:
                gate_count = "N/A"

            print(f"{fuse:12} | {compile_time:11.1f}s | {gate_count}")


if __name__ == "__main__":
    main()
