from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import tempfile
import time
from pathlib import Path

from stc.layout_bin import read_packed_layout_bin

def _have_avx512() -> bool:
    try:
        flags = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "avx512f" in flags


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def _yosys_flatten_keccak(out_json: Path) -> None:
    rtl = Path("external-sha3-verilog/low_throughput_core/rtl")
    if not rtl.exists():
        raise FileNotFoundError("missing external-sha3-verilog checkout")
    script = (
        f"read_verilog {rtl}/*.v; "
        "hierarchy -top keccak; proc; flatten; opt; opt_clean; "
        f"write_json {out_json}"
    )
    _run(["yosys", "-q", "-p", script])


def _build_shared(c_path: Path, so_path: Path) -> None:
    cc = os.environ.get("CC", "cc")
    _run(
        [
            cc,
            "-O1",
            "-shared",
            "-fPIC",
            "-mavx512f",
            "-mavx512vl",
            "-mavx512dq",
            "-mavx512bw",
            str(c_path),
            "-o",
            str(so_path),
        ]
    )


Vec = ctypes.c_uint64 * 8  # 64 bytes, matches __m512i


def _aligned_vec_array(n: int) -> tuple[ctypes.Array, ctypes.POINTER(Vec)]:
    raw = ctypes.create_string_buffer(n * ctypes.sizeof(Vec) + 63)
    base = ctypes.addressof(raw)
    aligned = (base + 63) & ~63
    ptr = ctypes.cast(aligned, ctypes.POINTER(Vec))
    return raw, ptr


ONES = Vec(*([0xFFFFFFFFFFFFFFFF] * 8))
ZEROS = Vec(*([0] * 8))


def _set_bit(arr: ctypes.Array, idx: int, bit: int) -> None:
    arr[idx] = ONES if (bit & 1) else ZEROS


def _set_field_bits_bitsliced(
    arr: ctypes.Array, *, lsb: int, width: int, value: int
) -> None:
    for i in range(width):
        _set_bit(arr, lsb + i, (value >> i) & 1)


def _pack_word_be(bs: bytes) -> int:
    v = 0
    for i, b in enumerate(bs):
        v |= int(b) << (24 - 8 * i)
    return v


def _out_bits_to_bytes_le(bits_lsb_first: list[int]) -> bytes:
    assert len(bits_lsb_first) % 8 == 0
    out = bytearray(len(bits_lsb_first) // 8)
    for i in range(len(out)):
        v = 0
        for b in range(8):
            v |= (bits_lsb_first[i * 8 + b] & 1) << b
        out[i] = v
    return bytes(out)


def _out_bits_to_digest_bytes(bits_512_lsb_first: list[int]) -> bytes:
    raw = _out_bits_to_bytes_le(bits_512_lsb_first)
    if len(raw) != 64:
        raise ValueError("expected 512 bits")
    return raw[::-1]


def _openssl_keccak_512(msg: bytes) -> bytes:
    p = subprocess.run(
        ["openssl", "dgst", "-keccak-512"],
        input=msg,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    out = p.stdout.decode("utf-8", errors="ignore").strip()
    return bytes.fromhex(out.split()[-1])


def simulate_keccak_512_once_steps(
    lib: ctypes.CDLL,
    layout: dict,
    message: bytes,
    *,
    idle_chunk: int,
) -> bytes:
    inputs = layout["inputs"]
    state = layout["state"]
    outputs = layout["outputs"]

    input_bits = sum(int(v["width"]) for v in inputs.values())
    state_bits = sum(int(v["width"]) for v in state.values())

    out_word = outputs["out"]
    out_ready = outputs["out_ready"]
    buffer_full = outputs.get("buffer_full")

    in_raw, in_ptr = _aligned_vec_array(input_bits)
    out_bits = sum(int(v["width"]) for v in outputs.values())
    out_raw, out_ptr = _aligned_vec_array(out_bits)
    st0_raw, st0_ptr = _aligned_vec_array(state_bits)
    st1_raw, st1_ptr = _aligned_vec_array(state_bits)

    in_arr = ctypes.cast(in_ptr, ctypes.POINTER(Vec * input_bits)).contents
    out_arr = ctypes.cast(out_ptr, ctypes.POINTER(Vec * out_bits)).contents

    # Initialize state to zero.
    st0_arr = ctypes.cast(st0_ptr, ctypes.POINTER(Vec * state_bits)).contents
    st1_arr = ctypes.cast(st1_ptr, ctypes.POINTER(Vec * state_bits)).contents
    for i in range(state_bits):
        st0_arr[i] = ZEROS
        st1_arr[i] = ZEROS

    lib.circuit_steps_shared.argtypes = [
        ctypes.POINTER(Vec),
        ctypes.POINTER(Vec),
        ctypes.POINTER(Vec),
        ctypes.POINTER(Vec),
        ctypes.c_int,
    ]
    lib.circuit_steps_shared.restype = None

    st_ptrs: list[ctypes.POINTER(Vec)] = [st0_ptr, st1_ptr]

    def step_once(
        *, in_word: int, in_ready_v: int, is_last_v: int, byte_num_v: int, reset_v: int
    ) -> None:
        _set_field_bits_bitsliced(
            in_arr, lsb=int(inputs["in"]["lsb"]), width=32, value=in_word
        )
        _set_field_bits_bitsliced(
            in_arr,
            lsb=int(inputs["in_ready"]["lsb"]),
            width=int(inputs["in_ready"]["width"]),
            value=in_ready_v,
        )
        _set_field_bits_bitsliced(
            in_arr,
            lsb=int(inputs["is_last"]["lsb"]),
            width=int(inputs["is_last"]["width"]),
            value=is_last_v,
        )
        _set_field_bits_bitsliced(
            in_arr,
            lsb=int(inputs["byte_num"]["lsb"]),
            width=int(inputs["byte_num"]["width"]),
            value=byte_num_v,
        )
        _set_field_bits_bitsliced(
            in_arr,
            lsb=int(inputs["reset"]["lsb"]),
            width=int(inputs["reset"]["width"]),
            value=reset_v,
        )
        lib.circuit_steps_shared(in_ptr, out_ptr, st_ptrs[0], st_ptrs[1], 1)
        st_ptrs[0], st_ptrs[1] = st_ptrs[1], st_ptrs[0]

    # Reset pulse.
    step_once(in_word=0, in_ready_v=0, is_last_v=0, byte_num_v=0, reset_v=1)
    step_once(in_word=0, in_ready_v=0, is_last_v=0, byte_num_v=0, reset_v=0)

    # Feed message.
    pos = 0
    while pos + 4 <= len(message):
        step_once(
            in_word=_pack_word_be(message[pos : pos + 4]),
            in_ready_v=1,
            is_last_v=0,
            byte_num_v=0,
            reset_v=0,
        )
        pos += 4

    rem = len(message) - pos
    if rem:
        step_once(
            in_word=_pack_word_be(message[pos:]),
            in_ready_v=1,
            is_last_v=1,
            byte_num_v=rem,
            reset_v=0,
        )
    else:
        step_once(in_word=0, in_ready_v=1, is_last_v=1, byte_num_v=0, reset_v=0)

    # Idle: run in chunks via emit-time stepper (no per-tick memcpy).
    _set_field_bits_bitsliced(in_arr, lsb=int(inputs["in"]["lsb"]), width=32, value=0)
    _set_field_bits_bitsliced(
        in_arr,
        lsb=int(inputs["in_ready"]["lsb"]),
        width=int(inputs["in_ready"]["width"]),
        value=0,
    )
    _set_field_bits_bitsliced(
        in_arr,
        lsb=int(inputs["is_last"]["lsb"]),
        width=int(inputs["is_last"]["width"]),
        value=0,
    )
    _set_field_bits_bitsliced(
        in_arr,
        lsb=int(inputs["byte_num"]["lsb"]),
        width=int(inputs["byte_num"]["width"]),
        value=0,
    )
    _set_field_bits_bitsliced(
        in_arr,
        lsb=int(inputs["reset"]["lsb"]),
        width=int(inputs["reset"]["width"]),
        value=0,
    )

    for _ in range(0, 5000, idle_chunk):
        lib.circuit_steps_shared(in_ptr, out_ptr, st_ptrs[0], st_ptrs[1], idle_chunk)
        st_ptrs[0], st_ptrs[1] = st_ptrs[1], st_ptrs[0]
        # Check out_ready.
        ready_bit = 1 if (int(out_arr[int(out_ready["lsb"])][0]) & 1) else 0
        if buffer_full is not None:
            _ = int(out_arr[int(buffer_full["lsb"])][0]) & 1
        if ready_bit:
            out_lsb = int(out_word["lsb"])
            out_w = int(out_word["width"])
            bits = [
                1 if (int(out_arr[out_lsb + i][0]) & 1) else 0 for i in range(out_w)
            ]
            return _out_bits_to_digest_bytes(bits)

    raise RuntimeError("digest not produced")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", type=str, default="abc")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--idle-chunk", type=int, default=64)
    return ap.parse_args(argv)


def main() -> int:
    import sys

    args = parse_args(sys.argv[1:])
    if not _have_avx512():
        raise SystemExit("CPU lacks AVX-512")
    if (
        subprocess.run(["openssl", "version"], stdout=subprocess.DEVNULL).returncode
        != 0
    ):
        raise SystemExit("missing openssl")

    msg = args.msg.encode("utf-8")
    exp = _openssl_keccak_512(msg)
    python = os.environ.get("PYTHON", ".venv/bin/python")

    with tempfile.TemporaryDirectory(prefix="stc_keccak_steps_") as td:
        td = Path(td)
        flat = td / "keccak_flat.json"
        out_dir = td / "out"
        out_dir.mkdir()
        _yosys_flatten_keccak(flat)

        _run(
            [
                python,
                "-m",
                "stc",
                str(flat),
                "--top",
                "keccak",
                "--out",
                str(out_dir),
                "--backend",
                "x86-avx512",
                "--force-bitsliced",
                "--bound",
                "8",
                "--max-live-pressure",
                "32",
            ],
            env={**os.environ, "PYTHONPATH": "."},
        )

        c_path = out_dir / "circuit_avx512.c"
        so_path = out_dir / "circuit_avx512.so"
        layout_path = out_dir / "io_layout.bin"
        _build_shared(c_path, so_path)

        layout = read_packed_layout_bin(layout_path).to_dict()
        lib = ctypes.CDLL(str(so_path))

        got = simulate_keccak_512_once_steps(
            lib, layout, msg, idle_chunk=args.idle_chunk
        )
        if got != exp:
            raise SystemExit("MISMATCH vs openssl")

        simulate_keccak_512_once_steps(lib, layout, msg, idle_chunk=args.idle_chunk)
        t0 = time.perf_counter()
        for _ in range(args.reps):
            simulate_keccak_512_once_steps(lib, layout, msg, idle_chunk=args.idle_chunk)
        dt = time.perf_counter() - t0
        dps = args.reps / dt
        print(
            f"VeryLogo AVX512 (emit-time steps, idle_chunk={args.idle_chunk}): {dps:.1f} digests/s"
        )

        print("\nOpenSSL speed keccak-512 (3s):")
        subprocess.run(
            ["openssl", "speed", "-seconds", "3", "-evp", "keccak-512"], check=False
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
