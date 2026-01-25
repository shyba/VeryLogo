from __future__ import annotations

from dataclasses import dataclass

from stc.circuit_synth import CircuitState
from stc.interp import infer_type
from stc.tick_ir import (
    AShr,
    And,
    BitVecConst,
    BitVecType,
    BoolConst,
    BoolType,
    Concat,
    Eq,
    Expr,
    LShr,
    Mux,
    Not,
    Or,
    Shl,
    Slice,
    TickIR,
    Type,
    Var,
    Xor,
)
from stc.tick_ir_validate import validate_tick_ir


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
        inputs={k: {"lsb": in_offsets[k], "width": _width(ir.inputs[k])} for k in input_order},
        state={k: {"lsb": st_offsets[k], "width": _width(ir.state[k])} for k in state_order},
        outputs={k: {"lsb": out_offsets[k], "width": _width(ir.outputs[k])} for k in output_order},
        next_state={k: {"lsb": nx_offsets[k], "width": _width(ir.state[k])} for k in state_order},
        input_bits=input_bits,
        output_bits=output_bits,
    )

    # We build nodes for boolean expressions. Inputs live at indices [0..input_bits).
    gates: list[tuple] = []
    memo_bits: dict[str, list[int]] = {}

    ctx_types: dict[str, Type] = {**ir.inputs, **ir.state}

    def packed_var_bit(name: str, bit: int) -> int:
        if name in in_offsets:
            return in_offsets[name] + bit
        if name in st_offsets:
            return st_offsets[name] + bit
        raise LoweringError(f"unknown packed var: {name}")

    def new_gate(op: str, *args) -> int:
        idx = input_bits + len(gates)
        gates.append((op, *args))
        return idx

    def lower_bool(e: Expr) -> int:
        bits = lower_bits(e)
        if len(bits) != 1:
            raise LoweringError("expected bool expression")
        return bits[0]

    def lower_bits(e: Expr) -> list[int]:
        key = repr(e)
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
            op = "xor" if isinstance(e, Xor) else ("and" if isinstance(e, And) else "or")
            bits = [new_gate(op, a, b) for a, b in zip(ab, bb)]
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

        raise LoweringError(f"unsupported expression for CircuitState lowering: {type(e)}")

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
