from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def have_avx2() -> bool:
    try:
        return "avx2" in Path("/proc/cpuinfo").read_text(errors="ignore")
    except OSError:
        return False


def have_avx512() -> bool:
    try:
        return "avx512f" in Path("/proc/cpuinfo").read_text(errors="ignore")
    except OSError:
        return False


@dataclass(frozen=True)
class NativeBuild:
    so_path: Path
    tmp_dir: Path


def compile_shared(c_path: Path, *, cflags: list[str]) -> NativeBuild:
    cc = os.environ.get("CC", "cc")
    if shutil.which(cc) is None:
        raise RuntimeError(f"missing compiler: {cc}")

    td = Path(tempfile.mkdtemp(prefix="stc_native_"))
    so = td / "circuit.so"
    cmd = [cc, "-O3", "-shared", "-fPIC", *cflags, str(c_path), "-o", str(so)]
    subprocess.run(cmd, check=True)
    return NativeBuild(so_path=so, tmp_dir=td)


def _aligned_vec_array(ctypes, Vec, n: int):
    raw = ctypes.create_string_buffer(n * ctypes.sizeof(Vec) + 63)
    base = ctypes.addressof(raw)
    aligned = (base + 63) & ~63
    ptr = ctypes.cast(aligned, ctypes.POINTER(Vec))
    return raw, ptr


def run_avx512_circuit(
    so_path: Path, in_words: Iterable[int], *, input_bits: int, output_bits: int
) -> list[int]:
    import ctypes

    Vec = ctypes.c_uint64 * 8
    lib = ctypes.CDLL(str(so_path))
    lib.circuit.argtypes = [ctypes.POINTER(Vec), ctypes.POINTER(Vec)]
    lib.circuit.restype = None

    in_words = list(in_words)
    if len(in_words) != input_bits:
        raise ValueError("input length mismatch")

    in_raw, in_ptr = _aligned_vec_array(ctypes, Vec, input_bits)
    out_raw, out_ptr = _aligned_vec_array(ctypes, Vec, output_bits)

    for i, w in enumerate(in_words):
        # Expect w to already be 0 or ~0 for bitsliced booleans.
        in_ptr[i] = Vec(
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
        )

    lib.circuit(in_ptr, out_ptr)
    out = []
    for i in range(output_bits):
        out.append(int(out_ptr[i][0]) & 0xFFFFFFFFFFFFFFFF)
    return out


def run_avx512_steps_shared(
    so_path: Path,
    in_io_words: Iterable[int],
    state_in_words: Iterable[int],
    *,
    input_io_bits: int,
    state_bits: int,
    output_io_bits: int,
    steps: int,
) -> tuple[list[int], list[int]]:
    import ctypes

    Vec = ctypes.c_uint64 * 8
    lib = ctypes.CDLL(str(so_path))
    if not hasattr(lib, "circuit_steps_shared"):
        raise RuntimeError("missing symbol: circuit_steps_shared")
    lib.circuit_steps_shared.argtypes = [
        ctypes.POINTER(Vec),
        ctypes.POINTER(Vec),
        ctypes.POINTER(Vec),
        ctypes.POINTER(Vec),
        ctypes.c_int,
    ]
    lib.circuit_steps_shared.restype = None

    in_io_words = list(in_io_words)
    state_in_words = list(state_in_words)
    if len(in_io_words) != input_io_bits:
        raise ValueError("input_io length mismatch")
    if len(state_in_words) != state_bits:
        raise ValueError("state length mismatch")

    in_raw, in_ptr = _aligned_vec_array(ctypes, Vec, input_io_bits)
    out_raw, out_ptr = _aligned_vec_array(ctypes, Vec, output_io_bits)
    st0_raw, st0_ptr = _aligned_vec_array(ctypes, Vec, state_bits)
    st1_raw, st1_ptr = _aligned_vec_array(ctypes, Vec, state_bits)

    for i, w in enumerate(in_io_words):
        in_ptr[i] = Vec(
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
        )
    for i, w in enumerate(state_in_words):
        st0_ptr[i] = Vec(
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
        )

    lib.circuit_steps_shared(in_ptr, out_ptr, st0_ptr, st1_ptr, int(steps))

    out_io = []
    for i in range(output_io_bits):
        out_io.append(int(out_ptr[i][0]) & 0xFFFFFFFFFFFFFFFF)
    st_out = []
    for i in range(state_bits):
        st_out.append(int(st1_ptr[i][0]) & 0xFFFFFFFFFFFFFFFF)
    return out_io, st_out


def run_avx2_circuit(
    so_path: Path, in_words: Iterable[int], *, input_bits: int, output_bits: int
) -> list[int]:
    import ctypes

    Vec = ctypes.c_uint64 * 4
    lib = ctypes.CDLL(str(so_path))
    lib.circuit.argtypes = [ctypes.POINTER(Vec), ctypes.POINTER(Vec)]
    lib.circuit.restype = None

    in_words = list(in_words)
    if len(in_words) != input_bits:
        raise ValueError("input length mismatch")

    in_raw, in_ptr = _aligned_vec_array(ctypes, Vec, input_bits)
    out_raw, out_ptr = _aligned_vec_array(ctypes, Vec, output_bits)

    for i, w in enumerate(in_words):
        in_ptr[i] = Vec(
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
            w & 0xFFFFFFFFFFFFFFFF,
        )

    lib.circuit(in_ptr, out_ptr)
    out = []
    for i in range(output_bits):
        out.append(int(out_ptr[i][0]) & 0xFFFFFFFFFFFFFFFF)
    return out
