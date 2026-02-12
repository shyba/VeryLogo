#!/usr/bin/env python3
"""
Analyze gate explosion in Keccak compilation.
Compiles Keccak with both bit-level and packed lowering with instrumentation.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from stc.circuit_state_bin import read_circuit_state_bin
from stc.packed_circuit_bin import read_packed_circuit_bin
from stc.tick_ir_bin2 import read_tick_ir_bin

def run(cmd: list[str], *, env: dict[str, str] | None = None, capture: bool = False):
    """Run command, optionally capturing output."""
    try:
        if capture:
            result = subprocess.run(
                cmd, check=False, env=env, capture_output=True, text=True
            )
            if result.returncode != 0:
                print(
                    f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
                )
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, cmd)
            return result
        else:
            result = subprocess.run(
                cmd, check=False, env=env, capture_output=True, text=True
            )
            if result.returncode != 0:
                print(
                    f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
                )
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, cmd)
    except subprocess.CalledProcessError:
        raise


def yosys_flatten_keccak(out_json: Path) -> None:
    """Flatten Keccak circuit with yosys."""
    rtl = Path("external-sha3-verilog/low_throughput_core/rtl")
    if not rtl.exists():
        raise FileNotFoundError("missing external-sha3-verilog checkout")
    script = (
        f"read_verilog {rtl}/*.v; "
        "hierarchy -top keccak; proc; flatten; opt; opt_clean; "
        f"write_json {out_json}"
    )
    run(["yosys", "-q", "-p", script])


def compile_keccak(
    flat_json: Path,
    out_dir: Path,
    *,
    force_bitsliced: bool = False,
    force_packed: bool = False,
) -> None:
    """Compile Keccak with STC."""
    python = os.environ.get("PYTHON", ".venv/bin/python")
    cmd = [
        python,
        "-m",
        "stc",
        str(flat_json),
        "--top",
        "keccak",
        "--out",
        str(out_dir),
        "--backend",
        "x86-avx512",
        "--bound",
        "8",
    ]
    if force_bitsliced:
        cmd.append("--force-bitsliced")
    if force_packed:
        cmd.append("--force-packed")

    env = {**os.environ, "PYTHONPATH": ".", "STC_DEBUG_GATE_EXPLOSION": "1"}
    run(cmd, env=env)


def load_tick_ir_dict(path: Path) -> dict:
    """Load TickIR binary and return dict."""
    return read_tick_ir_bin(str(path)).to_dict()


def analyze_tick_ir(tick_ir: dict) -> dict:
    """Analyze TickIR structure and expression distribution."""

    def count_expr_types(node, counts: dict[str, int]) -> None:
        """Recursively count expression types."""
        if not isinstance(node, dict):
            return

        node_type = node.get("type")
        if node_type:
            counts[node_type] = counts.get(node_type, 0) + 1

        for k, v in node.items():
            if k == "type":
                continue
            if isinstance(v, dict):
                count_expr_types(v, counts)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        count_expr_types(item, counts)

    def max_depth(node) -> int:
        """Calculate maximum expression depth."""
        if not isinstance(node, dict):
            return 0

        max_child = 0
        for k, v in node.items():
            if k == "type":
                continue
            if isinstance(v, dict):
                max_child = max(max_child, max_depth(v))
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        max_child = max(max_child, max_depth(item))

        return 1 + max_child

    expr_counts: dict[str, int] = {}
    max_expr_depth = 0

    for output_expr in tick_ir.get("outputs", {}).values():
        count_expr_types(output_expr, expr_counts)
        max_expr_depth = max(max_expr_depth, max_depth(output_expr))

    for next_state in tick_ir.get("next_state", {}).values():
        count_expr_types(next_state, expr_counts)
        max_expr_depth = max(max_expr_depth, max_depth(next_state))

    return {
        "expr_type_counts": expr_counts,
        "max_expr_depth": max_expr_depth,
        "num_state_bits": len(tick_ir.get("state", {})),
        "num_input_bits": sum(w for w in tick_ir.get("input_widths", {}).values()),
        "num_output_bits": sum(w for w in tick_ir.get("output_widths", {}).values()),
    }


def analyze_circuit_state(circuit_state) -> dict:
    """Analyze gate distribution in bit-level circuit state."""
    gate_counts: dict[str, int] = {}

    for gate in circuit_state.gates:
        if isinstance(gate, list) and len(gate) > 0:
            op = gate[0]
        elif isinstance(gate, tuple) and len(gate) > 0:
            op = gate[0]
        else:
            op = "unknown"
        gate_counts[op] = gate_counts.get(op, 0) + 1

    return {
        "total_gates": circuit_state.gate_count,
        "gate_type_counts": gate_counts,
        "num_state_bits": None,
    }


def analyze_packed_circuit_state(packed_state) -> dict:
    """Analyze gate distribution in packed circuit state."""
    gate_counts: dict[str, int] = {}

    for gate in packed_state.gates:
        if isinstance(gate, tuple) and len(gate) > 0:
            op = gate[0]
        else:
            op = "unknown"
        gate_counts[op] = gate_counts.get(op, 0) + 1

    return {
        "total_gates": packed_state.gate_count,
        "gate_type_counts": gate_counts,
        "num_state_words": None,
    }


def main() -> int:
    print("=" * 80)
    print("KECCAK GATE EXPLOSION ANALYSIS")
    print("=" * 80)

    td = Path(tempfile.mkdtemp(prefix="keccak_analysis_"))
    print(f"\nWorking directory: {td}\n")

    print("\n[1/5] Flattening Keccak with Yosys...")
    flat_json = td / "keccak_flat.json"
    yosys_flatten_keccak(flat_json)
    print(f"  Written: {flat_json}")

    print("\n[2/5] Extracting and optimizing TickIR...")
    ir_dir = td / "ir"
    ir_dir.mkdir(exist_ok=True)
    python = os.environ.get("PYTHON", ".venv/bin/python")
    print(f"  Running: {python} -m stc ... (this may take 30-60 seconds)")
    run(
        [
            python,
            "-m",
            "stc",
            str(flat_json),
            "--top",
            "keccak",
            "--out",
            str(ir_dir),
            "--no-backend",
            "--bound",
            "8",
        ],
        env={**os.environ, "PYTHONPATH": "."},
    )

    reduced_ir = ir_dir / "reduced_tick_ir.bin"
    if not reduced_ir.exists():
        raise FileNotFoundError(f"Expected output not created: {reduced_ir}")
    print(f"  Written: {reduced_ir} ({reduced_ir.stat().st_size} bytes)")

    print("\n[3/5] Analyzing TickIR structure...")
    tick_ir = load_tick_ir_dict(reduced_ir)
    ir_analysis = analyze_tick_ir(tick_ir)
    print(f"  State bits: {ir_analysis['num_state_bits']}")
    print(f"  Input bits: {ir_analysis['num_input_bits']}")
    print(f"  Output bits: {ir_analysis['num_output_bits']}")
    print(f"  Max expr depth: {ir_analysis['max_expr_depth']}")
    print(f"  Expression type distribution:")
    for expr_type, count in sorted(
        ir_analysis["expr_type_counts"].items(), key=lambda x: -x[1]
    ):
        print(f"    {expr_type}: {count}")

    print("\n[4/5] Compiling with bit-level lowering (--force-bitsliced)...")
    bitsliced_dir = td / "bitsliced"
    bitsliced_dir.mkdir(exist_ok=True)
    compile_keccak(flat_json, bitsliced_dir, force_bitsliced=True)

    circuit_state_path = bitsliced_dir / "circuit_state.bin"
    if circuit_state_path.exists():
        circuit_state = read_circuit_state_bin(circuit_state_path)
        bit_analysis = analyze_circuit_state(circuit_state)
        print(f"  Total gates: {bit_analysis['total_gates']}")
        print(f"  Gate type distribution:")
        for gate_type, count in sorted(
            bit_analysis["gate_type_counts"].items(), key=lambda x: -x[1]
        ):
            print(f"    {gate_type}: {count}")
    else:
        print(f"  WARNING: {circuit_state_path} not found")

    print("\n[5/5] Compiling with packed lowering (--force-packed)...")
    packed_dir = td / "packed"
    packed_dir.mkdir(exist_ok=True)
    compile_keccak(flat_json, packed_dir, force_packed=True)

    packed_state_path = packed_dir / "packed_circuit_state.bin"
    if packed_state_path.exists():
        packed_state = read_packed_circuit_bin(packed_state_path)
        packed_analysis = analyze_packed_circuit_state(packed_state)
        print(f"  Total gates: {packed_analysis['total_gates']}")
        print(f"  Gate type distribution:")
        for gate_type, count in sorted(
            packed_analysis["gate_type_counts"].items(), key=lambda x: -x[1]
        ):
            print(f"    {gate_type}: {count}")
    else:
        print(f"  WARNING: {packed_state_path} not found")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if circuit_state_path.exists() and packed_state_path.exists():
        ratio = packed_analysis["total_gates"] / bit_analysis["total_gates"]
        print(f"\nBit-level gates: {bit_analysis['total_gates']}")
        print(f"Packed gates: {packed_analysis['total_gates']}")
        print(f"Ratio (packed/bit): {ratio:.2f}x")

        print("\nGate explosion breakdown:")
        for gate_type in sorted(
            set(bit_analysis["gate_type_counts"].keys())
            | set(packed_analysis["gate_type_counts"].keys())
        ):
            bit_count = bit_analysis["gate_type_counts"].get(gate_type, 0)
            packed_count = packed_analysis["gate_type_counts"].get(gate_type, 0)
            if packed_count > bit_count:
                increase = packed_count - bit_count
                print(f"  {gate_type}: +{increase} ({bit_count} -> {packed_count})")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {td}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
