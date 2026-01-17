from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from stc.tick_ir import BitVecType, BoolType, SimdType, TickIR, Type
from stc.tick_ir_to_verilog import emit_verilog
from stc.tooling import require_tool


@dataclass(frozen=True)
class BoundedEquivError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def check_equiv_smtbmc(
    *,
    original_verilog: str,
    original_top: str,
    ir: TickIR,
    bound: int,
    timeout_s: int = 120,
) -> None:
    yosys = require_tool("yosys")
    smtbmc = require_tool("yosys-smtbmc")

    if bound < 1:
        raise BoundedEquivError("bound must be >= 1")

    try:
        import z3  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise BoundedEquivError(
            "z3-solver is not installed in the active Python environment"
        ) from e

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        orig_v = root / "orig.v"
        impl_v = root / "impl.v"
        miter_v = root / "miter.v"
        script_ys = root / "miter.ys"

        orig_v.write_text(original_verilog, encoding="utf-8")
        impl_v.write_text(emit_verilog(ir, module_name="tickir_impl"), encoding="utf-8")
        miter_v.write_text(_emit_miter(original_top, ir), encoding="utf-8")

        script_ys.write_text(
            "\n".join(
                [
                    f"read_verilog -sv {orig_v}",
                    f"read_verilog -sv {impl_v}",
                    f"read_verilog -sv {miter_v}",
                    "hierarchy -check -top miter",
                    "proc",
                    "opt",
                    "dffunmap",
                    "opt -noff",
                    "opt_clean",
                    f"write_smt2 -wires {root/'miter.smt2'}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        try:
            subprocess.run(
                [yosys, "-q", "-s", str(script_ys)], check=True, timeout=timeout_s
            )
        except subprocess.TimeoutExpired as e:
            raise BoundedEquivError("yosys miter build timed out") from e
        except subprocess.CalledProcessError as e:
            raise BoundedEquivError("yosys miter build failed") from e

        smt2 = root / "miter.smt2"
        cmd = [
            smtbmc,
            "-s",
            "z3",
            "--noincr",
            "--noinfo",
            "-t",
            str(bound),
            str(smt2),
        ]

        try:
            repo_root = Path(__file__).resolve().parent.parent
            venv_bin = repo_root / ".venv" / "bin"
            env = dict(os.environ)
            env["PATH"] = f"{venv_bin}:{env.get('PATH','')}"
            subprocess.run(
                cmd,
                check=True,
                timeout=timeout_s,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            raise BoundedEquivError("smtbmc timed out") from e
        except subprocess.CalledProcessError as e:
            raise BoundedEquivError("bounded equivalence check failed") from e


def _width(t: Type) -> int:
    if isinstance(t, BoolType):
        return 1
    if isinstance(t, BitVecType):
        return t.width
    assert isinstance(t, SimdType)
    return t.total_width


def _decl(name: str, t: Type) -> str:
    w = _width(t)
    if w == 1:
        return f"logic {name}"
    return f"logic [{w - 1}:0] {name}"


def _emit_miter(original_top: str, ir: TickIR) -> str:
    inputs = dict(ir.inputs)
    outputs = dict(ir.outputs)

    ports: list[str] = ["input logic clk"]
    ports.extend([f"input {_decl(name, t)}" for name, t in inputs.items()])
    ports.append("output logic stc_ok")

    lines: list[str] = []
    lines.append(f"module miter({', '.join(ports)});")
    lines.append("  logic rst;")
    lines.append("  initial rst = 1'b1;")
    lines.append("  always_ff @(posedge clk) rst <= 1'b0;")

    for name, t in outputs.items():
        lines.append(f"  {_decl(f'u0_{name}', t)};")
        lines.append(f"  {_decl(f'u1_{name}', t)};")

    conns0: list[str] = [".clk(clk)", ".rst(rst)"]
    conns1: list[str] = [".clk(clk)", ".rst(rst)"]
    conns0.extend([f".{name}({name})" for name in inputs])
    conns1.extend([f".{name}({name})" for name in inputs])
    conns0.extend([f".{name}(u0_{name})" for name in outputs])
    conns1.extend([f".{name}(u1_{name})" for name in outputs])

    lines.append(f"  {original_top} u0({', '.join(conns0)});")
    lines.append(f"  tickir_impl u1({', '.join(conns1)});")

    lines.append("  always_comb begin")
    lines.append("    stc_ok = 1'b1;")
    for name in outputs:
        lines.append(f"    stc_ok = stc_ok & (u0_{name} == u1_{name});")
    lines.append("  end")

    lines.append("  always_ff @(posedge clk) begin")
    lines.append("    if (!rst) assert(stc_ok);")
    lines.append("  end")
    lines.append("endmodule")
    lines.append("")
    return "\n".join(lines)
