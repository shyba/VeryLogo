from __future__ import annotations

import os
import shutil
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolMissing(Exception):
    name: str

    def __str__(self) -> str:
        return f"required tool not found: {self.name}"


def require_tool(name: str) -> str:
    override = os.environ.get(f"STC_{name.upper()}", "").strip()
    if override:
        return override
    path = shutil.which(name)
    if path is None:
        raise ToolMissing(name=name)
    return path
