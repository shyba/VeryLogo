from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from stc.tooling import require_tool


@dataclass(frozen=True)
class YosysError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def run_yosys(
    input_v: Path,
    output_json: Path,
    *,
    top: str | None = None,
    output_script: Path | None = None,
    use_synth: bool = False,
    use_abc_lut3: bool = False,
    abc_lut3_depth: int | None = None,
    abc_script: Path | None = None,
) -> None:
    yosys = require_tool("yosys")
    top_arg = f"; hierarchy -check -top {top}" if top else "; hierarchy -check"
    input_arg = f"{input_v}/*.v" if input_v.is_dir() else f"{input_v}"

    if abc_script is not None:
        script = (
            f"read_verilog -sv {input_arg}"
            f"{top_arg}"
            "; proc"
            "; flatten"
            "; opt"
            "; techmap"
            f"; abc -script {abc_script}"
            "; opt"
            f"; write_json {output_json}"
        )
    elif use_abc_lut3:
        if abc_lut3_depth is not None and abc_lut3_depth <= 0:
            raise ValueError("abc_lut3_depth must be positive")
        # LUT3 mapping via synth -lut 3 (ABC depth-driven + area recovery).
        synth_top = f" -top {top}" if top else ""
        script = (
            f"read_verilog -sv {input_arg}"
            f"; synth{synth_top} -flatten -lut 3"
            f"; write_json {output_json}"
        )
    elif use_synth:
        script = (
            f"read_verilog -sv {input_arg}"
            f"{top_arg}"
            "; proc"
            "; opt"
            "; techmap"
            "; abc"
            "; opt"
            f"; write_json {output_json}"
        )
    else:
        script = (
            f"read_verilog -sv {input_arg}"
            f"{top_arg}"
            "; proc"
            "; opt"
            "; opt_clean"
            f"; write_json {output_json}"
        )

    if output_script is not None:
        output_script.write_text(script + "\n", encoding="utf-8")

    try:
        subprocess.run([yosys, "-q", "-p", script], check=True)
    except subprocess.CalledProcessError as e:
        raise YosysError(message="yosys failed") from e
