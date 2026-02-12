from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from stc.interp import eval_expr
from stc.circuit_state_bin import read_circuit_state_bin
from stc.layout_bin import read_packed_layout_bin
from stc.tick_ir_bin2 import read_tick_ir_bin
from stc.tick_ir_validate import validate_tick_ir


def _have_avx512() -> bool:
    try:
        return "avx512f" in Path("/proc/cpuinfo").read_text(errors="ignore")
    except OSError:
        return False


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def _yosys_flatten_sha3(out_json: Path) -> None:
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
    if shutil.which(cc) is None:
        raise RuntimeError(f"missing compiler: {cc}")
    _run(
        [
            cc,
            "-O3",
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


def _pack_word_be(bs: bytes) -> int:
    v = 0
    for i, b in enumerate(bs):
        v |= int(b) << (24 - 8 * i)
    return v


def _bits_to_int_le(bits: list[int]) -> int:
    v = 0
    for i, b in enumerate(bits):
        v |= (b & 1) << i
    return v


def _int_to_bits_le(value: int, width: int) -> list[int]:
    return [((value >> i) & 1) for i in range(width)]


def _get_field_bits(packed: list[int], *, lsb: int, width: int) -> list[int]:
    return packed[lsb : lsb + width]


def _set_field_bits(packed: list[int], *, lsb: int, width: int, value: int) -> None:
    for i in range(width):
        packed[lsb + i] = (value >> i) & 1


@dataclass(frozen=True)
class Layout:
    input_bits: int
    output_bits: int
    inputs: dict[str, dict[str, int]]
    outputs: dict[str, dict[str, int]]
    state: dict[str, dict[str, int]]
    next_state: dict[str, dict[str, int]]

    @staticmethod
    def load(path: Path) -> "Layout":
        d = read_packed_layout_bin(path).to_dict()
        return Layout(
            input_bits=int(d["input_bits"]),
            output_bits=int(d["output_bits"]),
            inputs=d["inputs"],
            outputs=d["outputs"],
            state=d["state"],
            next_state=d["next_state"],
        )


def _load_circuit_so(so_path: Path):
    import ctypes

    Vec = ctypes.c_uint64 * 8
    lib = ctypes.CDLL(str(so_path))
    lib.circuit.argtypes = [ctypes.POINTER(Vec), ctypes.POINTER(Vec)]
    lib.circuit.restype = None

    def aligned_vec_array(n: int):
        raw = ctypes.create_string_buffer(n * ctypes.sizeof(Vec) + 63)
        base = ctypes.addressof(raw)
        aligned = (base + 63) & ~63
        ptr = ctypes.cast(aligned, ctypes.POINTER(Vec))
        return raw, ptr

    return lib, Vec, aligned_vec_array


def eval_circuit_state_once(circuit_state, in_bits: list[int]) -> list[int]:
    input_bits = int(circuit_state.input_bits)
    gates = circuit_state.gates
    outputs = circuit_state.outputs
    if len(in_bits) != input_bits:
        raise ValueError("input bit length mismatch")
    nodes = list(int(b) & 1 for b in in_bits)
    for gate in gates:
        if len(gate) == 5:
            op, a, b, c, imm8 = gate
            assert op == "ternary"
            ia = nodes[a] & 1
            ib = nodes[b] & 1
            ic = nodes[c] & 1
            idx = (ic << 2) | (ib << 1) | ia
            nodes.append((int(imm8) >> idx) & 1)
            continue

        op = gate[0]
        if op == "const":
            nodes.append(int(gate[1]) & 1)
        elif op == "not":
            nodes.append(nodes[int(gate[1])] ^ 1)
        elif op == "xor":
            nodes.append(nodes[int(gate[1])] ^ nodes[int(gate[2])])
        elif op == "and":
            nodes.append(nodes[int(gate[1])] & nodes[int(gate[2])])
        elif op == "or":
            nodes.append(nodes[int(gate[1])] | nodes[int(gate[2])])
        else:
            raise RuntimeError(f"unknown gate op {op}")
    out = []
    for idx, inv in outputs:
        v = nodes[int(idx)] & 1
        if inv:
            v ^= 1
        out.append(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", type=str, default="abc")
    ap.add_argument("--cycles", type=int, default=300)
    ap.add_argument("--bound", type=int, default=8)
    args = ap.parse_args()

    if not _have_avx512():
        raise SystemExit("CPU lacks AVX-512; cannot run AVX-512 circuit")
    if shutil.which("yosys") is None:
        raise SystemExit("missing yosys")

    msg = args.msg.encode("utf-8")

    keep = os.environ.get("STC_KEEP", "") not in {"", "0"}
    if keep:
        td_path = Path(tempfile.mkdtemp(prefix="stc_sha3_diff_keep_"))
        print(f"keeping temp dir: {td_path}")
        td_ctx = None
    else:
        td_ctx = tempfile.TemporaryDirectory(prefix="stc_sha3_diff_")
        td_path = Path(td_ctx.name)

    try:
        td = td_path
        flat = td / "keccak_flat.json"
        out_dir = td / "out"
        out_dir.mkdir()

        _yosys_flatten_sha3(flat)
        _run(
            [
                os.environ.get("PYTHON", ".venv/bin/python"),
                "-m",
                "stc",
                str(flat),
                "--top",
                "keccak",
                "--out",
                str(out_dir),
                "--backend",
                "x86-avx512",
                "--bound",
                str(int(args.bound)),
            ],
            env={**os.environ, "PYTHONPATH": "."},
        )

        layout = Layout.load(out_dir / "io_layout.bin")
        ir = read_tick_ir_bin(str(out_dir / "reduced_tick_ir.bin"))
        validate_tick_ir(ir)
        circuit_state = read_circuit_state_bin(out_dir / "circuit_state.bin")

        so_path = out_dir / "circuit_avx512.so"
        _build_shared(out_dir / "circuit_avx512.c", so_path)
        lib, Vec, aligned_vec_array = _load_circuit_so(so_path)

        in_raw, in_ptr = aligned_vec_array(layout.input_bits)
        out_raw, out_ptr = aligned_vec_array(layout.output_bits)

        # Treat each packed bit as a full __m512i, with lane0 in bit0.
        # Bitsliced convention: false = all-zeros, true = all-ones.
        V0 = Vec(0, 0, 0, 0, 0, 0, 0, 0)
        V1 = Vec(
            0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF,
        )

        def set_in_bit(i: int, b: int) -> None:
            in_ptr[i] = V1 if (b & 1) else V0

        def get_out_bit(i: int) -> int:
            return (
                1
                if (int(out_ptr[i][0]) & 0xFFFFFFFFFFFFFFFF) == 0xFFFFFFFFFFFFFFFF
                else 0
            )

        packed_state_bits = [0] * layout.input_bits

        # Initialize state bits to reset_state (ignoring any clock gating; we treat reset as input).
        types = {**ir.inputs, **ir.state}
        interp_state: dict[str, int | bool] = {}
        for name, expr in ir.reset_state.items():
            interp_state[name] = eval_expr(expr, types, {})

        for name, meta in layout.state.items():
            w = int(meta["width"])
            lsb = int(meta["lsb"])
            v = interp_state.get(name, 0)
            if isinstance(v, bool):
                bits = [1 if v else 0]
            else:
                bits = _int_to_bits_le(int(v), w)
            for i in range(w):
                packed_state_bits[lsb + i] = bits[i]

        # Helper: call AVX512 circuit once for current packed bits.
        def call_avx(packed_in_bits: list[int]) -> list[int]:
            for i, bit in enumerate(packed_in_bits):
                set_in_bit(i, bit)
            if os.environ.get("STC_DEBUG_RESET", "") not in {"", "0"}:
                ri = int(layout.inputs["reset"]["lsb"])
                print(
                    "reset in_ptr[lsb][0] =",
                    hex(int(in_ptr[ri][0]) & 0xFFFFFFFFFFFFFFFF),
                )
            lib.circuit(in_ptr, out_ptr)
            if os.environ.get("STC_DEBUG_OUTIDX", "") not in {"", "0"}:
                oi = int(os.environ["STC_DEBUG_OUTIDX"])
                print(
                    "out_ptr[idx][0] =", hex(int(out_ptr[oi][0]) & 0xFFFFFFFFFFFFFFFF)
                )
            return [get_out_bit(i) for i in range(layout.output_bits)]

        def call_python_circuit(packed_in_bits: list[int]) -> list[int]:
            return eval_circuit_state_once(circuit_state, packed_in_bits)

        # Helper: update packed_state_bits from the circuit next_state slice.
        def apply_circuit_next_state(out_bits: list[int]) -> None:
            for name, meta in layout.state.items():
                w = int(meta["width"])
                st_lsb = int(meta["lsb"])
                nx_lsb = int(layout.next_state[name]["lsb"])
                for i in range(w):
                    packed_state_bits[st_lsb + i] = out_bits[nx_lsb + i]

        def tick_inputs(
            in_word: int, in_ready: int, is_last: int, byte_num: int, reset: int
        ) -> None:
            _set_field_bits(
                packed_state_bits,
                lsb=int(layout.inputs["in"]["lsb"]),
                width=int(layout.inputs["in"]["width"]),
                value=in_word,
            )
            _set_field_bits(
                packed_state_bits,
                lsb=int(layout.inputs["in_ready"]["lsb"]),
                width=int(layout.inputs["in_ready"]["width"]),
                value=in_ready,
            )
            _set_field_bits(
                packed_state_bits,
                lsb=int(layout.inputs["is_last"]["lsb"]),
                width=int(layout.inputs["is_last"]["width"]),
                value=is_last,
            )
            _set_field_bits(
                packed_state_bits,
                lsb=int(layout.inputs["byte_num"]["lsb"]),
                width=int(layout.inputs["byte_num"]["width"]),
                value=byte_num,
            )
            _set_field_bits(
                packed_state_bits,
                lsb=int(layout.inputs["reset"]["lsb"]),
                width=int(layout.inputs["reset"]["width"]),
                value=reset,
            )

        def step_interp(
            inp: dict[str, int | bool]
        ) -> tuple[dict[str, int | bool], dict[str, int | bool]]:
            env = {**inp, **interp_state}
            outs = {k: eval_expr(ir.output_exprs[k], types, env) for k in ir.outputs}
            if bool(inp.get("reset", False)):
                nxt = {k: eval_expr(ir.reset_state[k], types, env) for k in ir.state}
            else:
                nxt = {k: eval_expr(ir.next_state[k], types, env) for k in ir.state}
            interp_state.update(nxt)
            return outs, nxt

        def unpack_circuit_outputs(
            out_bits: list[int],
        ) -> tuple[dict[str, int | bool], dict[str, int | bool]]:
            outs: dict[str, int | bool] = {}
            for name, meta in layout.outputs.items():
                w = int(meta["width"])
                lsb = int(meta["lsb"])
                bits = out_bits[lsb : lsb + w]
                if w == 1:
                    outs[name] = bool(bits[0] & 1)
                else:
                    outs[name] = _bits_to_int_le(bits)

            nxt: dict[str, int | bool] = {}
            for name, meta in layout.next_state.items():
                w = int(meta["width"])
                lsb = int(meta["lsb"])
                bits = out_bits[lsb : lsb + w]
                if w == 1:
                    nxt[name] = bool(bits[0] & 1)
                else:
                    nxt[name] = _bits_to_int_le(bits)
            return outs, nxt

        # Cycle 0: reset=1
        tick_inputs(0, 0, 0, 0, 1)
        out_bits = call_avx(packed_state_bits)
        py_bits = call_python_circuit(packed_state_bits)
        if out_bits != py_bits:
            for i, (a, b) in enumerate(zip(out_bits, py_bits)):
                if a != b:
                    raise RuntimeError(
                        f"AVX512 != python circuit at cycle 0 (out[{i}] avx={a} py={b})"
                    )
            raise RuntimeError(
                "AVX512 circuit output differs from python CircuitState eval at cycle 0"
            )
        c_outs, c_nxt = unpack_circuit_outputs(out_bits)
        apply_circuit_next_state(out_bits)

        i_outs, i_nxt = step_interp(
            {"in": 0, "in_ready": False, "is_last": False, "byte_num": 0, "reset": True}
        )

        def check_cycle(
            cycle: int, out_bits: list[int], c_outs, i_outs, c_nxt, i_nxt
        ) -> None:
            for k in sorted(i_outs.keys()):
                if i_outs[k] != c_outs.get(k):
                    raise RuntimeError(
                        f"cycle {cycle}: output {k} mismatch: interp={i_outs[k]} circuit={c_outs.get(k)}"
                    )
            for k in sorted(i_nxt.keys()):
                if i_nxt[k] != c_nxt.get(k):
                    meta = layout.next_state.get(k)
                    if meta is not None and int(meta["width"]) > 1:
                        w = int(meta["width"])
                        lsb = int(meta["lsb"])
                        got_bits = out_bits[lsb : lsb + w]
                        exp_bits = (
                            [1 if bool(i_nxt[k]) else 0]
                            if isinstance(i_nxt[k], bool)
                            else _int_to_bits_le(int(i_nxt[k]), w)
                        )
                        for bi, (eb, gb) in enumerate(zip(exp_bits, got_bits)):
                            if (eb & 1) != (gb & 1):
                                raise RuntimeError(
                                    f"cycle {cycle}: next_state {k} bit {bi} mismatch: interp={eb} circuit={gb}"
                                )
                    raise RuntimeError(
                        f"cycle {cycle}: next_state {k} mismatch: interp={i_nxt[k]} circuit={c_nxt.get(k)}"
                    )

        check_cycle(0, out_bits, c_outs, i_outs, c_nxt, i_nxt)

        # Feed message.
        pos = 0
        while pos + 4 <= len(msg):
            word = _pack_word_be(msg[pos : pos + 4])
            tick_inputs(word, 1, 0, 0, 0)
            out_bits = call_avx(packed_state_bits)
            py_bits = call_python_circuit(packed_state_bits)
            if out_bits != py_bits:
                for i, (a, b) in enumerate(zip(out_bits, py_bits)):
                    if a != b:
                        raise RuntimeError(
                            f"AVX512 != python circuit at cycle {1 + pos // 4} (out[{i}] avx={a} py={b})"
                        )
                raise RuntimeError(
                    f"AVX512 circuit output differs from python CircuitState eval at cycle {1 + pos // 4}"
                )
            c_outs, c_nxt = unpack_circuit_outputs(out_bits)
            apply_circuit_next_state(out_bits)

            i_outs, i_nxt = step_interp(
                {
                    "in": word,
                    "in_ready": True,
                    "is_last": False,
                    "byte_num": 0,
                    "reset": False,
                }
            )
            check_cycle(1 + pos // 4, out_bits, c_outs, i_outs, c_nxt, i_nxt)
            pos += 4

        rem = len(msg) - pos
        if rem:
            word = _pack_word_be(msg[pos:])
            tick_inputs(word, 1, 1, rem, 0)
            out_bits = call_avx(packed_state_bits)
            py_bits = call_python_circuit(packed_state_bits)
            if out_bits != py_bits:
                for i, (a, b) in enumerate(zip(out_bits, py_bits)):
                    if a != b:
                        raise RuntimeError(
                            f"AVX512 != python circuit at cycle {1 + pos // 4} (out[{i}] avx={a} py={b})"
                        )
                raise RuntimeError(
                    f"AVX512 circuit output differs from python CircuitState eval at cycle {1 + pos // 4}"
                )
            c_outs, c_nxt = unpack_circuit_outputs(out_bits)
            apply_circuit_next_state(out_bits)
            i_outs, i_nxt = step_interp(
                {
                    "in": word,
                    "in_ready": True,
                    "is_last": True,
                    "byte_num": rem,
                    "reset": False,
                }
            )
            check_cycle(1 + pos // 4, out_bits, c_outs, i_outs, c_nxt, i_nxt)
        else:
            tick_inputs(0, 1, 1, 0, 0)
            out_bits = call_avx(packed_state_bits)
            py_bits = call_python_circuit(packed_state_bits)
            if out_bits != py_bits:
                for i, (a, b) in enumerate(zip(out_bits, py_bits)):
                    if a != b:
                        raise RuntimeError(
                            f"AVX512 != python circuit at cycle {1 + pos // 4} (out[{i}] avx={a} py={b})"
                        )
                raise RuntimeError(
                    f"AVX512 circuit output differs from python CircuitState eval at cycle {1 + pos // 4}"
                )
            c_outs, c_nxt = unpack_circuit_outputs(out_bits)
            apply_circuit_next_state(out_bits)
            i_outs, i_nxt = step_interp(
                {
                    "in": 0,
                    "in_ready": True,
                    "is_last": True,
                    "byte_num": 0,
                    "reset": False,
                }
            )
            check_cycle(1 + pos // 4, out_bits, c_outs, i_outs, c_nxt, i_nxt)

        # Run remaining cycles with no input.
        base_cycle = 2 + (len(msg) // 4)
        for cycle in range(base_cycle, base_cycle + int(args.cycles)):
            tick_inputs(0, 0, 0, 0, 0)
            out_bits = call_avx(packed_state_bits)
            py_bits = call_python_circuit(packed_state_bits)
            if out_bits != py_bits:
                for i, (a, b) in enumerate(zip(out_bits, py_bits)):
                    if a != b:
                        raise RuntimeError(
                            f"AVX512 != python circuit at cycle {cycle} (out[{i}] avx={a} py={b})"
                        )
                raise RuntimeError(
                    f"AVX512 circuit output differs from python CircuitState eval at cycle {cycle}"
                )
            c_outs, c_nxt = unpack_circuit_outputs(out_bits)
            apply_circuit_next_state(out_bits)
            i_outs, i_nxt = step_interp(
                {
                    "in": 0,
                    "in_ready": False,
                    "is_last": False,
                    "byte_num": 0,
                    "reset": False,
                }
            )
            try:
                check_cycle(cycle, out_bits, c_outs, i_outs, c_nxt, i_nxt)
            except RuntimeError as e:
                print(str(e))
                return 1
            if bool(c_outs.get("out_ready", False)):
                print(f"MATCH through out_ready at cycle {cycle}")
                return 0

        print("MATCH for tested cycles (no out_ready yet)")
        return 0
    finally:
        if td_ctx is not None:
            td_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
