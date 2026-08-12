"""Shared helpers for property-based tests of the AVX-512 GEMM/SIMD kernels.

Compiles the hand-scheduled asm kernels (stc.gemm_asm) plus a small C
checker driver once per process, then lets hypothesis drive the binaries
with (shape, seed) arguments. The C side generates its own data from the
seed (xorshift64), so the Python side only needs the shape and seed, and
hypothesis can shrink the counterexample to a seed.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from inference.gemm_asm import emit_gemm_kernel

_CACHE: dict[str, str] = {}


def cpu_has_avx512_vnni() -> bool:
    """The kernels need AVX512-VNNI (and BF16 for the bf16 kernel)."""
    try:
        with open("/proc/cpuinfo") as f:
            flags = f.read()
        return "avx512_vnni" in flags and "avx512_bf16" in flags
    except OSError:  # pragma: no cover
        return True  # non-Linux: let the build/run decide


def build_binary(
    key: str, c_source: str, asm_kernels: dict[str, tuple[str, int, int]] | None = None
) -> str:
    """Compile c_source plus the requested asm kernels once per process.

    asm_kernels: {symbol_name: (family, mr, nr)} emitted via stc.gemm_asm.
    """
    if key in _CACHE:
        return _CACHE[key]
    d = Path(tempfile.mkdtemp(prefix="prop_"))
    objs: list[str] = []
    for name, (family, mr, nr) in (asm_kernels or {}).items():
        asm = d / (name + ".S")
        asm.write_text(emit_gemm_kernel(family, mr, nr, name=name))
        obj = d / (name + ".o")
        subprocess.run(
            ["gcc", "-c", str(asm), "-o", str(obj)],
            check=True,
            capture_output=True,
            text=True,
        )
        objs.append(str(obj))
    cfile = d / "main.c"
    cfile.write_text(c_source)
    exe = d / "t"
    subprocess.run(
        ["gcc", "-O2", "-march=native", "-o", str(exe), str(cfile), *objs, "-lm"],
        check=True,
        capture_output=True,
        text=True,
    )
    _CACHE[key] = str(exe)
    return _CACHE[key]


def run_check(exe: str, args: list, timeout: int = 30) -> str:
    r = subprocess.run(
        [exe, *[str(a) for a in args]],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return r.stdout.strip()


def assert_ok(exe: str, args: list) -> None:
    out = run_check(exe, args)
    assert out == "OK", f"kernel check failed for args={args}: {out!r}"


# -- data-generation strategies for the supported domain -------------------

# The asm kernel's K-loop advances a_adv*unroll bytes per iteration and exits
# when its row pointer equals A+K*elem_bytes, so K must be a multiple of
# kpc*unroll (8 for vnni8, 4 for bf16) or the loop never terminates.
# M must be a multiple of 8 (MR), N a multiple of 32 (NR).

import hypothesis.strategies as st  # noqa: E402

MUL8 = st.integers(min_value=1, max_value=6).map(lambda t: 8 * t)  # 8..48
MUL32 = st.integers(min_value=1, max_value=4).map(lambda t: 32 * t)  # 32..128
K_VNNI = st.integers(min_value=1, max_value=8).map(lambda t: 8 * t)  # 8..64
K_BF16 = st.integers(min_value=1, max_value=8).map(lambda t: 4 * t)  # 4..32
SEED = st.integers(min_value=0, max_value=2**32 - 1)

# fixed boundary shapes (smallest supported, and edges of the domain)
EDGE_VNNI = st.sampled_from(
    [(8, 32, 8), (8, 32, 16), (8, 64, 8), (16, 32, 8), (8, 96, 32)]
)
EDGE_BF16 = st.sampled_from(
    [(8, 32, 4), (8, 32, 8), (8, 64, 4), (16, 32, 4), (8, 96, 12)]
)
