from __future__ import annotations

from pathlib import Path


def cpu_flags() -> set[str]:
    p = Path("/proc/cpuinfo")
    if not p.exists():
        return set()
    txt = p.read_text(encoding="utf-8", errors="replace")
    for line in txt.splitlines():
        if not line.startswith("flags"):
            continue
        parts = line.split()
        return {p for p in parts if p not in {"flags", ":"}}
    return set()


def has_flag(flag: str, flags: set[str] | None = None) -> bool:
    if flags is None:
        flags = cpu_flags()
    return flag in flags
