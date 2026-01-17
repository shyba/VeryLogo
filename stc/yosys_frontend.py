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
) -> None:
    yosys = require_tool("yosys")
    top_arg = f"; hierarchy -check -top {top}" if top else "; hierarchy -check"
    script = (
        f"read_verilog -sv {input_v}"
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
