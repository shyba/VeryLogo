from __future__ import annotations

import argparse
import platform
from pathlib import Path


def _cpuinfo() -> dict[str, str]:
    p = Path("/proc/cpuinfo")
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k and k not in out and v:
            out[k] = v
    return out


def _parse_flags(info: dict[str, str]) -> set[str]:
    raw = info.get("flags") or info.get("Features") or ""
    return {x.strip() for x in raw.split() if x.strip()}


def _render_markdown(info: dict[str, str]) -> str:
    arch = platform.machine()
    model = info.get("model name") or info.get("Processor") or "unknown"
    flags = _parse_flags(info)

    sse_flags = [
        ("sse", "SSE"),
        ("sse2", "SSE2"),
        ("sse3|pni", "SSE3"),
        ("ssse3", "SSSE3"),
        ("sse4_1", "SSE4.1"),
        ("sse4_2", "SSE4.2"),
        ("sse4a", "SSE4a"),
    ]

    avx_flags = [
        ("avx", "AVX"),
        ("avx2", "AVX2"),
        ("fma", "FMA"),
        ("avx_vnni", "AVX-VNNI"),
    ]

    avx512 = sorted([f for f in flags if f.startswith("avx512")])

    lines: list[str] = []
    lines.append("# CPU SIMD Features")
    lines.append("")
    lines.append("This file is generated for the current machine.")
    lines.append(
        "Regenerate with `python3 scripts/cpu_simd_features.py --out CPU_SIMD_FEATURES.md`."
    )
    lines.append("")
    lines.append("## CPU")
    lines.append(f"- Architecture: `{arch}`")
    lines.append(f"- Model: `{model}`")
    lines.append("")
    lines.append("## SIMD ISA Support (from `/proc/cpuinfo` flags)")
    lines.append("")
    lines.append("### SSE Family")
    for flag, label in sse_flags:
        parts = flag.split("|")
        present = any(p in flags for p in parts)
        flag_note = "|".join(f"`{p}`" for p in parts)
        lines.append(f"- {label}: {'yes' if present else 'no'} ({flag_note})")
    lines.append("")
    lines.append("### AVX Family")
    for flag, label in avx_flags:
        lines.append(f"- {label}: {'yes' if flag in flags else 'no'} (`{flag}`)")
    lines.append("")
    lines.append("### AVX-512 Family")
    if avx512:
        lines.append("- Present flags:")
        for f in avx512:
            lines.append(f"  - `{f}`")
    else:
        lines.append("- Present flags: none")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    info = _cpuinfo()
    md = _render_markdown(info)

    if args.out is not None:
        args.out.write_text(md, encoding="utf-8")
    else:
        print(md, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
