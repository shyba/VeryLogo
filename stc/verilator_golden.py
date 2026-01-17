from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from stc.tooling import require_tool


@dataclass(frozen=True)
class VerilatorError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def simulate_ticks(
    *,
    verilog: str,
    top: str,
    inputs: list[dict[str, int]],
    outputs: list[str],
    clk: str = "clk",
    rst: str = "rst",
    rst_ticks: int = 1,
    timeout_s: int = 120,
) -> list[dict[str, int]]:
    if rst_ticks < 0:
        raise ValueError("rst_ticks must be >= 0")

    verilator = require_tool("verilator")
    ordered_inputs = sorted(set().union(*(d.keys() for d in inputs)))
    ordered_outputs = list(outputs)

    normalized_inputs: list[list[int]] = []
    for tick in inputs:
        row = [int(tick.get(name, 0)) for name in ordered_inputs]
        normalized_inputs.append(row)

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        v_path = root / f"{top}.v"
        v_path.write_text(verilog, encoding="utf-8")

        cpp_path = root / "sim_main.cpp"
        cpp_path.write_text(
            _render_cpp(
                top=top,
                clk=clk,
                rst=rst,
                ordered_inputs=ordered_inputs,
                ordered_outputs=ordered_outputs,
                inputs=normalized_inputs,
                rst_ticks=rst_ticks,
            ),
            encoding="utf-8",
        )

        obj_dir = root / "obj"
        exe_path = obj_dir / f"V{top}"

        cmd = [
            verilator,
            "--sv",
            "-cc",
            "--exe",
            "--build",
            "--Mdir",
            str(obj_dir),
            "--top-module",
            top,
            str(v_path),
            str(cpp_path),
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                timeout=timeout_s,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.TimeoutExpired as e:
            raise VerilatorError("verilator build timed out") from e
        except subprocess.CalledProcessError as e:
            msg = "verilator build failed"
            if e.stderr:
                msg = msg + "\n" + e.stderr.strip()
            raise VerilatorError(msg) from e

        try:
            proc = subprocess.run(
                [str(exe_path)],
                check=True,
                timeout=timeout_s,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.TimeoutExpired as e:
            raise VerilatorError("verilator simulation timed out") from e
        except subprocess.CalledProcessError as e:
            msg = "verilator simulation failed"
            if e.stderr:
                msg = msg + "\n" + e.stderr.strip()
            raise VerilatorError(msg) from e

        out_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        result = [json.loads(ln) for ln in out_lines]
        return [{k: int(v) for k, v in row.items()} for row in result]


def _render_cpp(
    *,
    top: str,
    clk: str,
    rst: str,
    ordered_inputs: list[str],
    ordered_outputs: list[str],
    inputs: list[list[int]],
    rst_ticks: int,
) -> str:
    h = f"V{top}.h"
    assigns = "\n".join(
        f"        top->{name} = stims[t].{name};" for name in ordered_inputs
    )
    fields = "\n".join(f"    uint32_t {name};" for name in ordered_inputs)
    inits = ",\n".join(
        "        {" + ", ".join(f"{val}u" for val in row) + "}" for row in inputs
    )

    out_print = []
    out_print.append('        std::printf("{");')
    for i, name in enumerate(ordered_outputs):
        if i > 0:
            out_print.append('        std::printf(",");')
        out_print.append(
            f'        std::printf("\\"{name}\\":%u", (unsigned)top->{name});'
        )
    out_print.append('        std::printf("}\\n");')
    out_print_code = "\n".join(out_print)

    return f"""
#include <cstdint>
#include <cstdio>

#include "verilated.h"
#include "{h}"

static vluint64_t main_time = 0;
double sc_time_stamp() {{ return static_cast<double>(main_time); }}

struct Stim {{
{fields}
}};

static const Stim stims[] = {{
{inits}
}};

int main(int argc, char** argv) {{
    Verilated::commandArgs(argc, argv);
    V{top}* top = new V{top}();

    top->{clk} = 0;
    top->{rst} = 1;
    top->eval();

    top->{clk} = 1;
    top->eval();
    top->{clk} = 0;
    top->eval();

    for (size_t t = 0; t < (sizeof(stims) / sizeof(stims[0])); t++) {{
        top->{rst} = (t < (size_t){rst_ticks}) ? 1 : 0;
{assigns}
        top->eval();
{out_print_code}
        top->{clk} = 1;
        top->eval();
        top->{clk} = 0;
        top->eval();
        main_time += 1;
    }}

    delete top;
    return 0;
}}
"""
