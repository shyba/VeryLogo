from __future__ import annotations

from dataclasses import dataclass

from stc.circuit_synth import CircuitState
from stc.interp import infer_type
from stc.tick_ir import (
    AShr,
    Add,
    And,
    BitVecConst,
    BitVecType,
    BoolConst,
    BoolType,
    Concat,
    Div,
    Eq,
    Expr,
    LShr,
    Mul,
    Ult,
    Mux,
    Not,
    Or,
    Shl,
    Slice,
    Sub,
    TernaryLut,
    TickIR,
    Type,
    Var,
    Xor,
)
from stc.tick_ir_validate import validate_tick_ir
import os
import subprocess
import tempfile
import time
from pathlib import Path


@dataclass(frozen=True)
class LoweringError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class PackedLayout:
    """Packed bit layout used by CircuitState lowering.

    All inputs and state are concatenated (LSB-first per value) into a single
    bitvector exposed as CircuitState inputs.
    Outputs are similarly concatenated as: (primary outputs, next_state).
    """

    inputs: dict[str, dict[str, int]]
    state: dict[str, dict[str, int]]
    outputs: dict[str, dict[str, int]]
    next_state: dict[str, dict[str, int]]
    input_bits: int
    output_bits: int

    def to_dict(self) -> dict:
        return {
            "inputs": self.inputs,
            "state": self.state,
            "outputs": self.outputs,
            "next_state": self.next_state,
            "input_bits": self.input_bits,
            "output_bits": self.output_bits,
        }


def _width(t: Type) -> int:
    if isinstance(t, BoolType):
        return 1
    if isinstance(t, BitVecType):
        return int(t.width)
    raise LoweringError(f"unsupported type: {type(t)}")


def _ternary_imm8(fn: callable) -> int:
    # Intel ternarylogic uses bit index = (c<<2)|(b<<1)|a.
    imm = 0
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                out = 1 if fn(a, b, c) else 0
                idx = (c << 2) | (b << 1) | a
                imm |= out << idx
    return imm & 0xFF


_IMM_MUX_A_THEN_B_ELSE_C = _ternary_imm8(lambda a, b, c: (b if a else c) == 1)


def lower_tick_ir_to_circuit_state(ir: TickIR) -> tuple[CircuitState, PackedLayout]:
    """Lower TickIR to CircuitState via per-bit (boolean) lowering.

    This produces a combinational CircuitState that maps packed inputs+state bits
    to packed outputs+next_state bits. It is intended for scheduled AVX2/AVX-512/PTX
    emitters (bitsliced simulation).
    """
    validate_tick_ir(ir)

    input_order = sorted(ir.inputs.keys())
    state_order = sorted(ir.state.keys())
    output_order = sorted(ir.outputs.keys())

    in_offsets: dict[str, int] = {}
    st_offsets: dict[str, int] = {}
    out_offsets: dict[str, int] = {}
    nx_offsets: dict[str, int] = {}

    off = 0
    for name in input_order:
        in_offsets[name] = off
        off += _width(ir.inputs[name])
    for name in state_order:
        st_offsets[name] = off
        off += _width(ir.state[name])
    input_bits = off

    out_off = 0
    for name in output_order:
        out_offsets[name] = out_off
        out_off += _width(ir.outputs[name])
    for name in state_order:
        nx_offsets[name] = out_off
        out_off += _width(ir.state[name])
    output_bits = out_off

    layout = PackedLayout(
        inputs={
            k: {"lsb": in_offsets[k], "width": _width(ir.inputs[k])}
            for k in input_order
        },
        state={
            k: {"lsb": st_offsets[k], "width": _width(ir.state[k])} for k in state_order
        },
        outputs={
            k: {"lsb": out_offsets[k], "width": _width(ir.outputs[k])}
            for k in output_order
        },
        next_state={
            k: {"lsb": nx_offsets[k], "width": _width(ir.state[k])} for k in state_order
        },
        input_bits=input_bits,
        output_bits=output_bits,
    )

    # We build nodes for boolean expressions. Inputs live at indices [0..input_bits).
    gates: list[tuple] = []
    memo_bits: dict[str, list[int]] = {}
    const_cache: dict[tuple[int, int], int] = {}

    ctx_types: dict[str, Type] = {**ir.inputs, **ir.state}

    def packed_var_bit(name: str, bit: int) -> int:
        if name in in_offsets:
            return in_offsets[name] + bit
        if name in st_offsets:
            return st_offsets[name] + bit
        raise LoweringError(f"unknown packed var: {name}")

    def new_gate(op: str, *args) -> int:
        if op == "const":
            key = (args[0], args[1])
            cached_idx = const_cache.get(key)
            if cached_idx is not None:
                return cached_idx
            idx = input_bits + len(gates)
            gates.append((op, *args))
            const_cache[key] = idx
            return idx
        idx = input_bits + len(gates)
        gates.append((op, *args))
        return idx

    def lower_bool(e: Expr) -> int:
        bits = lower_bits(e)
        if len(bits) != 1:
            raise LoweringError("expected bool expression")
        return bits[0]

    def lower_bits(e: Expr) -> list[int]:
        key = id(e)
        cached = memo_bits.get(key)
        if cached is not None:
            return cached
        t = infer_type(e, ctx_types)
        w = _width(t)

        if isinstance(e, Var):
            bits = [packed_var_bit(e.name, i) for i in range(w)]
            memo_bits[key] = bits
            return bits

        if isinstance(e, BoolConst):
            bits = [new_gate("const", 1 if e.value else 0, 1)]
            memo_bits[key] = bits
            return bits

        if isinstance(e, BitVecConst):
            bits = [new_gate("const", (e.value >> i) & 1, 1) for i in range(w)]
            memo_bits[key] = bits
            return bits

        if isinstance(e, Slice):
            xb = lower_bits(e.x)
            bits = xb[e.offset : e.offset + e.width]
            if len(bits) != e.width:
                raise LoweringError("slice out of range")
            memo_bits[key] = bits
            return bits

        if isinstance(e, Concat):
            # TickIR Concat is MSB-first. Our lowering represents values as
            # LSB-first bit lists, so we append parts from LSB to MSB while
            # keeping each part's internal LSB-first ordering intact.
            bits: list[int] = []
            for p in reversed(e.parts):
                bits.extend(lower_bits(p))
            if len(bits) != w:
                raise LoweringError("concat width mismatch")
            memo_bits[key] = bits
            return bits

        if isinstance(e, Not):
            xb = lower_bits(e.x)
            bits = [new_gate("not", b, 0) for b in xb]
            memo_bits[key] = bits
            return bits

        if isinstance(e, (Xor, And, Or)):
            ab = lower_bits(e.a)
            bb = lower_bits(e.b)
            if len(ab) != len(bb):
                raise LoweringError("binary op width mismatch")
            op = (
                "xor" if isinstance(e, Xor) else ("and" if isinstance(e, And) else "or")
            )
            bits = [new_gate(op, a, b) for a, b in zip(ab, bb)]
            memo_bits[key] = bits
            return bits

        if isinstance(e, TernaryLut):
            ab = lower_bits(e.a)
            bb = lower_bits(e.b)
            cb = lower_bits(e.c)
            if len(ab) != len(bb) or len(ab) != len(cb):
                raise LoweringError("ternary width mismatch")
            bits = [new_gate("ternary", a, b, c, e.imm8) for a, b, c in zip(ab, bb, cb)]
            memo_bits[key] = bits
            return bits

        if isinstance(e, Eq):
            a_bits = lower_bits(e.a)
            b_bits = lower_bits(e.b)
            if len(a_bits) != len(b_bits):
                raise LoweringError("eq width mismatch")
            if len(a_bits) == 1:
                # a == b  <=>  !(a ^ b)
                x = new_gate("xor", a_bits[0], b_bits[0])
                bits = [new_gate("not", x, 0)]
                memo_bits[key] = bits
                return bits

            # AND over bitwise XNORs.
            acc = None
            for abit, bbit in zip(a_bits, b_bits):
                x = new_gate("xor", abit, bbit)
                xnor = new_gate("not", x, 0)
                acc = xnor if acc is None else new_gate("and", acc, xnor)
            assert acc is not None
            bits = [acc]
            memo_bits[key] = bits
            return bits

        if isinstance(e, Ult):
            a_bits = lower_bits(e.a)
            b_bits = lower_bits(e.b)
            if len(a_bits) != len(b_bits):
                raise LoweringError("ult width mismatch")
            n = len(a_bits)
            if n == 0:
                raise LoweringError("ult on empty bitvector")

            # Unsigned compare: MSB-first.
            eq = new_gate("const", 1, 1)  # all higher bits equal so far
            lt = new_gate("const", 0, 1)
            for i in reversed(range(n)):
                abit = a_bits[i]
                bbit = b_bits[i]
                not_a = new_gate("not", abit, 0)
                a_lt_b = new_gate("and", not_a, bbit)
                lt_here = new_gate("and", eq, a_lt_b)
                lt = new_gate("or", lt, lt_here)

                axb = new_gate("xor", abit, bbit)
                xnor = new_gate("not", axb, 0)
                eq = new_gate("and", eq, xnor)

            bits = [lt]
            memo_bits[key] = bits
            return bits

        if isinstance(e, Mux):
            c = lower_bool(e.cond)
            tb = lower_bits(e.a)
            fb = lower_bits(e.b)
            if len(tb) != len(fb):
                raise LoweringError("mux width mismatch")
            # Implement mux as (c & t) | (~c & f). This avoids relying on ternary
            # imm8 conventions during lowering.
            c_not = new_gate("not", c, 0)
            bits = []
            for tbit, fbit in zip(tb, fb):
                ct = new_gate("and", c, tbit)
                cnf = new_gate("and", c_not, fbit)
                bits.append(new_gate("or", ct, cnf))
            memo_bits[key] = bits
            return bits

        if isinstance(e, (Shl, LShr, AShr)):
            if not isinstance(e.b, BitVecConst):
                raise LoweringError("shifts require constant shift amount for lowering")
            sh = int(e.b.value)
            xb = lower_bits(e.a)
            if sh < 0:
                raise LoweringError("shift amount must be >= 0")
            if isinstance(e, Shl):
                bits = ([new_gate("const", 0, 1)] * sh) + xb[: max(0, w - sh)]
            elif isinstance(e, LShr):
                bits = xb[sh:] + ([new_gate("const", 0, 1)] * sh)
            else:
                sign = xb[w - 1] if w > 0 else new_gate("const", 0, 1)
                bits = xb[sh:] + ([sign] * sh)
            bits = bits[:w]
            memo_bits[key] = bits
            return bits

        if isinstance(e, Add):
            ab = lower_bits(e.a)
            bb = lower_bits(e.b)
            if len(ab) != len(bb):
                raise LoweringError("add width mismatch")
            if w == 0:
                memo_bits[key] = []
                return []
            bits = []
            c = new_gate("const", 0, 1)
            for i in range(w):
                s_partial = new_gate("xor", ab[i], bb[i])
                s = new_gate("xor", s_partial, c)
                bits.append(s)
                ab_and_bb = new_gate("and", ab[i], bb[i])
                ab_xor_bb = s_partial
                c_and_xor = new_gate("and", c, ab_xor_bb)
                c = new_gate("or", ab_and_bb, c_and_xor)
            memo_bits[key] = bits
            return bits

        if isinstance(e, Sub):
            ab = lower_bits(e.a)
            bb = lower_bits(e.b)
            if len(ab) != len(bb):
                raise LoweringError("sub width mismatch")
            if w == 0:
                memo_bits[key] = []
                return []
            bits = []
            c = new_gate("const", 1, 1)
            for i in range(w):
                bb_not = new_gate("not", bb[i], 0)
                s_partial = new_gate("xor", ab[i], bb_not)
                s = new_gate("xor", s_partial, c)
                bits.append(s)
                ab_and_bbnot = new_gate("and", ab[i], bb_not)
                ab_xor_bbnot = s_partial
                c_and_xor = new_gate("and", c, ab_xor_bbnot)
                c = new_gate("or", ab_and_bbnot, c_and_xor)
            memo_bits[key] = bits
            return bits

        if isinstance(e, Mul):
            ab = lower_bits(e.a)
            bb = lower_bits(e.b)
            if len(ab) != len(bb):
                raise LoweringError("mul width mismatch")
            if w == 0:
                memo_bits[key] = []
                return []
            if w > 16:
                raise LoweringError(
                    f"Mul with width {w} exceeds bit-level lowering cap of 16 bits. "
                    f"Suggestion: Use power-of-2 constants (will be reduced to shifts) "
                    f"or synthesize multiplication as a Verilog module."
                )

            # Schoolbook multiplication: compute partial products and sum them
            # For each bit position i, compute ab[i] * bb (shifted left by i positions)
            # and accumulate into result
            result_bits = [new_gate("const", 0, 1) for _ in range(w)]

            for i in range(w):
                # Compute partial product: ab[i] * bb, which gives w bits starting at position i
                # We accumulate this into result_bits[i:i+w]
                carry = new_gate("const", 0, 1)
                for j in range(w - i):
                    # Partial product bit: ab[i] AND bb[j]
                    pp = new_gate("and", ab[i], bb[j])
                    # Three-input addition: result_bits[i+j] + pp + carry
                    sum1 = new_gate("xor", result_bits[i + j], pp)
                    sum_final = new_gate("xor", sum1, carry)
                    # Compute carry for next position
                    and1 = new_gate("and", result_bits[i + j], pp)
                    and2 = new_gate("and", result_bits[i + j], carry)
                    and3 = new_gate("and", pp, carry)
                    or1 = new_gate("or", and1, and2)
                    carry = new_gate("or", or1, and3)
                    result_bits[i + j] = sum_final

            memo_bits[key] = result_bits
            return result_bits

        if isinstance(e, Div):
            ab = lower_bits(e.a)
            bb = lower_bits(e.b)
            if len(ab) != len(bb):
                raise LoweringError("div width mismatch")
            if w == 0:
                memo_bits[key] = []
                return []
            if w > 16:
                raise LoweringError(
                    f"Div with width {w} exceeds bit-level lowering cap of 16 bits. "
                    f"Suggestion: Use power-of-2 constants (will be reduced to right shifts) "
                    f"or synthesize division as a Verilog module."
                )

            # Restoring division algorithm (unsigned)
            # We process bits from MSB to LSB of dividend (ab[w-1] down to ab[0])
            # and generate quotient bits from MSB to LSB
            quotient_bits = [new_gate("const", 0, 1) for _ in range(w)]
            remainder_bits = [new_gate("const", 0, 1) for _ in range(w)]

            for idx in range(w):
                # Process dividend bit from MSB to LSB
                dividend_bit_idx = w - 1 - idx
                # Store quotient bit from MSB to LSB
                quotient_bit_idx = w - 1 - idx

                # Shift remainder left by 1 and bring down ab[dividend_bit_idx]
                new_remainder_bits = [ab[dividend_bit_idx]] + remainder_bits[: w - 1]

                # Subtract divisor from remainder (unsigned subtract: remainder - bb)
                # Using two's complement: remainder - bb = remainder + (~bb + 1)
                carry = new_gate("const", 1, 1)
                diff_bits = []
                for j in range(w):
                    bb_not = new_gate("not", bb[j], 0)
                    s_partial = new_gate("xor", new_remainder_bits[j], bb_not)
                    s = new_gate("xor", s_partial, carry)
                    diff_bits.append(s)
                    r_and_bbnot = new_gate("and", new_remainder_bits[j], bb_not)
                    r_xor_bbnot = s_partial
                    c_and_xor = new_gate("and", carry, r_xor_bbnot)
                    carry = new_gate("or", r_and_bbnot, c_and_xor)

                # Check if remainder >= divisor by examining final carry
                # In two's complement subtraction, carry out = 1 means no borrow (remainder >= divisor)
                # If carry is 1, then remainder >= divisor, so quotient bit is 1
                # If carry is 0, then remainder < divisor, so quotient bit is 0
                quotient_bit = carry
                quotient_bits[quotient_bit_idx] = quotient_bit

                # If quotient_bit is 1, update remainder to diff_bits
                updated_remainder = []
                for j in range(w):
                    selected = new_gate("and", quotient_bit, diff_bits[j])
                    not_qbit = new_gate("not", quotient_bit, 0)
                    unselected = new_gate("and", not_qbit, new_remainder_bits[j])
                    updated_remainder.append(new_gate("or", selected, unselected))
                remainder_bits = updated_remainder

            memo_bits[key] = quotient_bits
            return quotient_bits

        raise LoweringError(
            f"unsupported expression for CircuitState lowering: {type(e)}"
        )

    packed_outputs: list[Expr] = []
    for name in output_order:
        packed_outputs.append(ir.output_exprs[name])
    for name in state_order:
        # Model synchronous reset if the design has a conventional `reset` input.
        # Yosys may represent resets as $sdff cells where reset semantics are not
        # encoded in `next_state`, but rather in `reset_state`.
        if "reset" in ir.inputs and isinstance(ir.inputs["reset"], BoolType):
            packed_outputs.append(
                Mux(cond=Var("reset"), a=ir.reset_state[name], b=ir.next_state[name])
            )
        else:
            packed_outputs.append(ir.next_state[name])

    out_bits: list[int] = []
    for expr in packed_outputs:
        out_bits.extend(lower_bits(expr))

    if len(out_bits) != output_bits:
        raise LoweringError("packed output width mismatch")

    outputs = [(idx, False) for idx in out_bits]
    circuit = CircuitState(
        input_bits=input_bits,
        output_bits=output_bits,
        gates=gates,
        outputs=outputs,
        gate_count=len(gates),
    )
    return circuit, layout


def _find_rust_lower_bin() -> Path | None:
    env = os.environ.get("STC_RUST_LOWER_BIN")
    if env:
        p = Path(env)
        return p if p.exists() else None
    for candidate in [
        Path("rust/tick_lower_rs/target/release/tick_lower_rs"),
        Path("rust/tick_lower_rs/target/debug/tick_lower_rs"),
    ]:
        if candidate.exists():
            return candidate
    return None


def rust_lower_tick_ir_to_circuit_state(
    ir: TickIR,
) -> tuple[CircuitState, PackedLayout] | None:
    if os.environ.get("STC_RUST_LOWER", "1").lower() in {"0", "false", "no"}:
        return None
    bin_path = _find_rust_lower_bin()
    if bin_path is None:
        return None
    fmt = os.environ.get("STC_RUST_LOWER_FORMAT", "bin").lower()
    if fmt != "bin":
        return None
    try:
        from stc.tick_ir_bin2 import write_tick_ir_bin
    except Exception:
        return None
    timing_enabled = os.environ.get("STC_TIMING", "0").lower() not in {
        "0",
        "false",
        "no",
    }

    def _log(label: str, start: float) -> None:
        if timing_enabled:
            elapsed = time.perf_counter() - start
            print(f"[timing] rust_lower:{label}: {elapsed:.3f}s", flush=True)

    output_fmt = os.environ.get("STC_RUST_LOWER_OUTPUT", "bin").lower()
    if output_fmt != "bin":
        return None
    with tempfile.TemporaryDirectory(prefix="stc_rust_lower_") as td:
        td_path = Path(td)
        in_path = td_path / "in.bin"
        out_path = td_path / "circuit_state.bin"
        if timing_enabled:
            print("[timing] rust_lower:start", flush=True)
        t0 = time.perf_counter()
        write_tick_ir_bin(ir, str(in_path))
        _log("write_input_bin", t0)
        try:
            t0 = time.perf_counter()
            subprocess.run(
                [
                    str(bin_path),
                    "--input",
                    str(in_path),
                    "--output",
                    str(out_path),
                    "--format",
                    fmt,
                    "--output-format",
                    output_fmt,
                ],
                check=True,
                capture_output=not timing_enabled,
                text=True,
            )
            _log("subprocess", t0)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        try:
            t0 = time.perf_counter()
            from stc.circuit_state_bin import read_circuit_state_bin
            from stc.layout_bin import read_packed_layout_bin

            circuit = read_circuit_state_bin(out_path)
            _log("read_output_bin", t0)
            t0 = time.perf_counter()
            layout_path = out_path.with_name(out_path.stem + "_layout.bin")
            layout = read_packed_layout_bin(layout_path)
            _log("read_layout_bin", t0)
            return circuit, layout
        except Exception:
            return None
