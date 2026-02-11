from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import dataclass

from stc.packed_circuit import PackedCircuitState, PackedGate
from stc.hashcons import hashcons_tick_ir, _key_expr
from stc.interp import infer_type
from stc.tick_ir import (
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
    Ult,
    LShr,
    Mul,
    Mux,
    Not,
    Or,
    Rotl,
    Rotr,
    Shl,
    Slice,
    Sub,
    TickIR,
    TernaryLut,
    Type,
    Var,
    Xor,
)
from stc.tick_ir_validate import validate_tick_ir


@dataclass
class PackedLoweringError(Exception):
    message: str
    expression_type: str | None = None
    variable_name: str | None = None
    subexpression_context: str | None = None

    def __str__(self) -> str:
        parts = [self.message]
        if self.expression_type:
            parts.append(f"Expression type: {self.expression_type}")
        if self.variable_name:
            parts.append(f"Variable: {self.variable_name}")
        if self.subexpression_context:
            parts.append(f"Context: {self.subexpression_context}")
        parts.append(
            "\nSuggestion: Use bit-level lowering (fallback path) for designs with unsupported expressions."
        )
        return "\n".join(parts)


@dataclass(frozen=True)
class PackedWordLayout:
    """Packed 64-bit word layout for PackedCircuitState lowering.

    All inputs and state are concatenated (LSW-first per value) into a single
    word-vector exposed as PackedCircuitState inputs.
    Outputs are similarly concatenated as: (primary outputs, next_state).
    """

    inputs: dict[str, dict[str, int]]
    state: dict[str, dict[str, int]]
    outputs: dict[str, dict[str, int]]
    next_state: dict[str, dict[str, int]]
    input_words: int
    output_words: int
    mode: str = "packed"

    def to_dict(self) -> dict:
        return {
            "inputs": self.inputs,
            "state": self.state,
            "outputs": self.outputs,
            "next_state": self.next_state,
            "input_words": self.input_words,
            "output_words": self.output_words,
            "mode": self.mode,
        }

    def __post_init__(self) -> None:
        if self.mode not in {"packed", "bitslice"}:
            raise ValueError(f"invalid PackedWordLayout mode: {self.mode}")

    def io_words(self) -> tuple[int, int]:
        """Return (input_io_words, output_io_words) for emission."""
        if self.mode == "bitslice":
            input_io_words = sum(int(v["width_bits"]) for v in self.inputs.values())
            output_io_words = sum(int(v["width_bits"]) for v in self.outputs.values())
            return input_io_words, output_io_words
        input_io_words = sum(
            (int(v["width_bits"]) + 63) // 64 for v in self.inputs.values()
        )
        output_io_words = sum(
            (int(v["width_bits"]) + 63) // 64 for v in self.outputs.values()
        )
        return input_io_words, output_io_words


WORD_BITS = 64
WORD_MASK = (1 << WORD_BITS) - 1


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass
class GateStats:
    total_gates: int = 0
    expr_gate_counts: dict[str, int] = None
    large_expressions: list[tuple[str, int]] = None

    def __post_init__(self):
        if self.expr_gate_counts is None:
            self.expr_gate_counts = defaultdict(int)
        if self.large_expressions is None:
            self.large_expressions = []


@dataclass
class MemoStats:
    hits: int = 0
    misses: int = 0
    hits_by_type: dict[str, int] = None
    misses_by_type: dict[str, int] = None

    def __post_init__(self):
        if self.hits_by_type is None:
            self.hits_by_type = defaultdict(int)
        if self.misses_by_type is None:
            self.misses_by_type = defaultdict(int)

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def hit_rate_by_type(self, expr_type: str) -> float:
        total = self.hits_by_type[expr_type] + self.misses_by_type[expr_type]
        return self.hits_by_type[expr_type] / total if total > 0 else 0.0


def lower_tick_ir_to_packed_circuit_state(
    ir: TickIR,
) -> tuple[PackedCircuitState, PackedWordLayout]:
    """
    Lower TickIR to a word-level PackedCircuitState (64-bit words).

    This lowering is intentionally conservative. It supports common patterns
    in stateful designs that operate on bitvectors (bitwise boolean ops, slices,
    concatenation, and constant shifts). If unsupported constructs are found,
    raise PackedLoweringError so callers can fall back to bit-level lowering.
    """
    validate_tick_ir(ir)
    ir = hashcons_tick_ir(ir)

    input_order = sorted(ir.inputs.keys())
    state_order = sorted(ir.state.keys())
    output_order = sorted(ir.outputs.keys())

    def width_bits(t: Type) -> int:
        if isinstance(t, BoolType):
            return 1
        if isinstance(t, BitVecType):
            return int(t.width)
        raise PackedLoweringError(
            message=f"unsupported type for packed lowering: {type(t).__name__}",
            expression_type=type(t).__name__,
        )

    def width_words(t: Type) -> int:
        return _ceil_div(width_bits(t), WORD_BITS)

    # Compute packed word offsets.
    in_offsets: dict[str, int] = {}
    st_offsets: dict[str, int] = {}
    out_offsets: dict[str, int] = {}
    nx_offsets: dict[str, int] = {}

    off = 0
    for name in input_order:
        in_offsets[name] = off
        off += width_words(ir.inputs[name])
    for name in state_order:
        st_offsets[name] = off
        off += width_words(ir.state[name])
    input_words = off

    out_off = 0
    for name in output_order:
        out_offsets[name] = out_off
        out_off += width_words(ir.outputs[name])
    for name in state_order:
        nx_offsets[name] = out_off
        out_off += width_words(ir.state[name])
    output_words = out_off

    layout = PackedWordLayout(
        inputs={
            k: {"lsw": in_offsets[k], "width_bits": width_bits(ir.inputs[k])}
            for k in input_order
        },
        state={
            k: {"lsw": st_offsets[k], "width_bits": width_bits(ir.state[k])}
            for k in state_order
        },
        outputs={
            k: {"lsw": out_offsets[k], "width_bits": width_bits(ir.outputs[k])}
            for k in output_order
        },
        next_state={
            k: {"lsw": nx_offsets[k], "width_bits": width_bits(ir.state[k])}
            for k in state_order
        },
        input_words=input_words,
        output_words=output_words,
    )

    # Build word variables for inputs+state in the packed order.
    var_words: dict[str, list[int]] = {}
    var_widths: dict[str, int] = {}

    def add_var_words(name: str, lsw: int, width_b: int) -> None:
        nwords = _ceil_div(width_b, WORD_BITS)
        var_words[name] = [lsw + i for i in range(nwords)]
        var_widths[name] = width_b

    for name in input_order:
        add_var_words(name, in_offsets[name], width_bits(ir.inputs[name]))
    for name in state_order:
        add_var_words(name, st_offsets[name], width_bits(ir.state[name]))

    gates: list[PackedGate] = []
    memo: dict[tuple, tuple[list[int], int]] = {}
    gate_stats = GateStats()
    memo_stats = MemoStats()

    def emit_const_u64(val: int) -> int:
        gates.append(("const", int(val) & WORD_MASK, 0))
        return input_words + (len(gates) - 1)

    def emit_unary(op: str, a: int) -> int:
        if op == "not":
            gates.append(("not", a, 0))
        else:
            raise PackedLoweringError(
                message=f"unsupported unary op: {op}",
                subexpression_context=f"unary operation: {op}",
            )
        return input_words + (len(gates) - 1)

    def emit_bin(op: str, a: int, b: int) -> int:
        gates.append((op, a, b))
        return input_words + (len(gates) - 1)

    def emit_shift(op: str, a: int, imm: int) -> int:
        if imm < 0 or imm >= WORD_BITS:
            raise PackedLoweringError(
                message=f"shift immediate {imm} out of range for word op (0-{WORD_BITS-1})",
                subexpression_context=f"shift operation with immediate {imm}",
            )
        gates.append((op, a, int(imm), 0))
        return input_words + (len(gates) - 1)

    def emit_ternary(a: int, b: int, c: int, imm8: int) -> int:
        gates.append(("ternary", a, b, c, int(imm8) & 0xFF))
        return input_words + (len(gates) - 1)

    def get_var_words(e: Var) -> tuple[list[int], int]:
        name = e.name
        if name not in var_words:
            raise PackedLoweringError(
                message=f"unknown var: {name}",
                variable_name=name,
                expression_type="Var",
            )
        return (var_words[name], var_widths[name])

    def lower_expr_to_words(e: Expr) -> tuple[list[int], int]:
        """
        Returns (word_nodes_le, width_bits), where word_nodes_le[0] is bits [0..63].
        """
        ekey = _key_expr(e)
        expr_type = type(e).__name__
        cached = memo.get(ekey)
        if cached is not None:
            memo_stats.hits += 1
            memo_stats.hits_by_type[expr_type] += 1
            ws, w = cached
            return (list(ws), int(w))

        memo_stats.misses += 1
        memo_stats.misses_by_type[expr_type] += 1
        gates_before = len(gates)

        def _track_and_return(result: tuple[list[int], int]) -> tuple[list[int], int]:
            gates_after = len(gates)
            gate_delta = gates_after - gates_before
            gate_stats.total_gates += gate_delta
            gate_stats.expr_gate_counts[expr_type] += gate_delta
            if gate_delta > 1000:
                print(
                    f"WARNING: {expr_type} generated {gate_delta} gates",
                    file=sys.stderr,
                )
                gate_stats.large_expressions.append((expr_type, gate_delta))
            return result

        def _truncate(words_le: list[int], width_b: int) -> tuple[list[int], int]:
            if width_b <= 0:
                return (words_le, width_b)
            top = width_b % WORD_BITS
            if top == 0:
                return (words_le, width_b)
            mask = (1 << top) - 1
            words_le = list(words_le)
            words_le[-1] = emit_bin("and", words_le[-1], emit_const_u64(mask))
            return (words_le, width_b)

        if isinstance(e, Var):
            ws, w = get_var_words(e)
            out = _truncate(list(ws), w)
            memo[ekey] = (list(out[0]), int(out[1]))
            return _track_and_return(out)
        if isinstance(e, BoolConst):
            node = emit_const_u64(1 if e.value else 0)
            out = _truncate([node], 1)
            memo[ekey] = (list(out[0]), int(out[1]))
            return _track_and_return(out)
        if isinstance(e, BitVecConst):
            nwords = _ceil_div(e.width, WORD_BITS)
            out: list[int] = []
            for i in range(nwords):
                chunk = (e.value >> (i * WORD_BITS)) & WORD_MASK
                out.append(emit_const_u64(chunk))
            res = _truncate(out, e.width)
            memo[ekey] = (list(res[0]), int(res[1]))
            return _track_and_return(res)
        if isinstance(e, Not):
            ws, w = lower_expr_to_words(e.x)
            out = [emit_unary("not", wi) for wi in ws]
            res = _truncate(out, w)
            memo[ekey] = (list(res[0]), int(res[1]))
            return _track_and_return(res)
        if isinstance(e, TernaryLut):
            wa, aw = lower_expr_to_words(e.a)
            wb, bw = lower_expr_to_words(e.b)
            wc, cw = lower_expr_to_words(e.c)
            if aw != bw or aw != cw:
                raise PackedLoweringError(
                    message=f"width mismatch in ternary lut: {aw}, {bw}, {cw}",
                    expression_type="TernaryLut",
                    subexpression_context="ternary lut",
                )
            out = [emit_ternary(wa[i], wb[i], wc[i], e.imm8) for i in range(len(wa))]
            res = _truncate(out, aw)
            memo[ekey] = (list(res[0]), int(res[1]))
            return _track_and_return(res)
        if isinstance(e, (And, Or, Xor)):
            wa, aw = lower_expr_to_words(e.a)  # type: ignore[attr-defined]
            wb, bw = lower_expr_to_words(e.b)  # type: ignore[attr-defined]
            if aw != bw:
                raise PackedLoweringError(
                    message=f"width mismatch in bitwise op: {aw} vs {bw}",
                    expression_type=type(e).__name__,
                    subexpression_context=f"bitwise {type(e).__name__} with widths {aw} and {bw}",
                )
            op = "and" if isinstance(e, And) else "or" if isinstance(e, Or) else "xor"
            out = [emit_bin(op, wa[i], wb[i]) for i in range(len(wa))]
            res = _truncate(out, aw)
            memo[ekey] = (list(res[0]), int(res[1]))
            return _track_and_return(res)
        if isinstance(e, Ult):
            wa, aw = lower_expr_to_words(e.a)
            wb, bw = lower_expr_to_words(e.b)
            if aw != bw:
                raise PackedLoweringError("width mismatch in ult")
            if aw > WORD_BITS:
                raise PackedLoweringError(
                    "ult >64 bits not supported in packed lowering yet"
                )
            if aw < WORD_BITS:
                m = (1 << aw) - 1
                wa0 = emit_bin("and", wa[0], emit_const_u64(m))
                wb0 = emit_bin("and", wb[0], emit_const_u64(m))
            else:
                wa0 = wa[0]
                wb0 = wb[0]
            out = [emit_bin("ult", wa0, wb0)]
            res = _truncate(out, 1)
            memo[ekey] = (list(res[0]), int(res[1]))
            return _track_and_return(res)
        if isinstance(e, Slice):
            src_words, src_w = lower_expr_to_words(e.x)
            if e.offset < 0 or e.width < 1 or e.offset + e.width > src_w:
                raise PackedLoweringError(
                    message=f"invalid slice: offset={e.offset}, width={e.width}, source_width={src_w}",
                    expression_type="Slice",
                    subexpression_context=f"slice[{e.offset}:{e.offset+e.width}] of {src_w}-bit value",
                )
            # Fast path: 64-bit aligned slice.
            if e.offset % WORD_BITS == 0 and e.width % WORD_BITS == 0:
                start = e.offset // WORD_BITS
                nwords = e.width // WORD_BITS
                res = (src_words[start : start + nwords], e.width)
                memo[ekey] = (list(res[0]), int(res[1]))
                return _track_and_return(res)
            # Fast path: single-word slice (fits in one 64-bit word).
            word_i = e.offset // WORD_BITS
            bit_end = e.offset + e.width
            word_i_end = (bit_end - 1) // WORD_BITS if bit_end > 0 else word_i
            if e.width <= WORD_BITS and word_i == word_i_end:
                # Slice fits entirely within one source word.
                bit_off = e.offset % WORD_BITS
                src_word = (
                    src_words[word_i] if word_i < len(src_words) else emit_const_u64(0)
                )
                if bit_off == 0 and e.width == WORD_BITS:
                    # Full word extraction
                    res = ([src_word], e.width)
                    memo[ekey] = (list(res[0]), int(res[1]))
                    return _track_and_return(res)
                # Shift right to align bits to LSB, then mask.
                shifted = (
                    emit_shift("lshr", src_word, bit_off) if bit_off > 0 else src_word
                )
                if e.width < WORD_BITS:
                    mask = (1 << e.width) - 1
                    masked = emit_bin("and", shifted, emit_const_u64(mask))
                    res = ([masked], e.width)
                else:
                    res = ([shifted], e.width)
                memo[ekey] = (list(res[0]), int(res[1]))
                return _track_and_return(res)
            # Multi-word unaligned slice: word-by-word gather.
            bit_off = e.offset
            word_i = bit_off // WORD_BITS
            inner = bit_off % WORD_BITS
            nwords = _ceil_div(e.width, WORD_BITS)
            if inner == 0:
                # Aligned to word boundary, just need to mask final word if partial.
                out: list[int] = src_words[word_i : word_i + nwords]
                if e.width % WORD_BITS:
                    top_bits = e.width % WORD_BITS
                    mask = (1 << top_bits) - 1
                    out[-1] = emit_bin("and", out[-1], emit_const_u64(mask))
                res = (out, e.width)
                memo[ekey] = (list(res[0]), int(res[1]))
                return _track_and_return(res)
            # Unaligned: gather bits from multiple source words.
            zeros_node = emit_const_u64(0)

            def get_word(i: int) -> int:
                if 0 <= i < len(src_words):
                    return src_words[i]
                return zeros_node

            out: list[int] = []
            for i in range(nwords):
                src_i = word_i + i
                lo = get_word(src_i)
                lo_shifted = emit_shift("lshr", lo, inner)
                hi = get_word(src_i + 1)
                hi_shifted = emit_shift("shl", hi, WORD_BITS - inner)
                merged = emit_bin("or", lo_shifted, hi_shifted)
                out.append(merged)
            # Mask top word if width not multiple of 64.
            if e.width % WORD_BITS:
                top_bits = e.width % WORD_BITS
                mask = (1 << top_bits) - 1
                out[-1] = emit_bin("and", out[-1], emit_const_u64(mask))
            res = (out, e.width)
            memo[ekey] = (list(res[0]), int(res[1]))
            return _track_and_return(res)
        if isinstance(e, Concat):
            parts: list[tuple[list[int], int]] = [
                lower_expr_to_words(p) for p in e.parts
            ]
            total_w = sum(w for _ws, w in parts)
            # Fast path: word-aligned parts.
            if all((w % WORD_BITS) == 0 for _ws, w in parts):
                out_words: list[int] = []
                for ws, _w in reversed(parts):
                    out_words.extend(ws)
                res = (out_words, total_w)
                memo[ekey] = (list(res[0]), int(res[1]))
                return _track_and_return(res)
            # General case: pack mixed-width parts across word boundaries.
            # Concat semantics: parts[0] is MSB, parts[-1] is LSB.
            # We build output words LSB-first, so process parts in reverse.
            # Optimization: process words at a time instead of bits.
            out_words: list[int] = []
            current_word = emit_const_u64(0)
            bits_in_current = 0
            for ws, part_width in reversed(parts):
                # Process this part word-by-word instead of bit-by-bit.
                part_word_idx = 0
                part_bit_offset = 0
                while part_bit_offset < part_width:
                    bits_remaining_in_part = part_width - part_bit_offset
                    space_in_current = WORD_BITS - bits_in_current
                    # Determine how many bits to extract from current source word.
                    bit_offset_in_word = part_bit_offset % WORD_BITS
                    bits_avail_in_src_word = WORD_BITS - bit_offset_in_word
                    bits_to_copy = min(
                        bits_remaining_in_part, space_in_current, bits_avail_in_src_word
                    )
                    # Extract bits_to_copy from source word.
                    src_word = (
                        ws[part_word_idx]
                        if part_word_idx < len(ws)
                        else emit_const_u64(0)
                    )
                    # Shift right to align the bits we want to the LSB.
                    if bit_offset_in_word > 0:
                        src_word = emit_shift("lshr", src_word, bit_offset_in_word)
                    # Mask to extract exactly bits_to_copy.
                    if bits_to_copy < WORD_BITS:
                        mask = (1 << bits_to_copy) - 1
                        src_word = emit_bin("and", src_word, emit_const_u64(mask))
                    # Shift left to position in current output word.
                    if bits_in_current > 0:
                        src_word = emit_shift("shl", src_word, bits_in_current)
                    current_word = emit_bin("or", current_word, src_word)
                    bits_in_current += bits_to_copy
                    part_bit_offset += bits_to_copy
                    # Move to next source word if we consumed all bits from current word.
                    if (part_bit_offset % WORD_BITS) == 0:
                        part_word_idx += 1
                    # If current output word is full, emit it and start a new one.
                    if bits_in_current == WORD_BITS:
                        out_words.append(current_word)
                        current_word = emit_const_u64(0)
                        bits_in_current = 0
            # Emit final partial word if any bits remain.
            if bits_in_current > 0:
                out_words.append(current_word)
            res = (out_words, total_w)
            memo[ekey] = (list(res[0]), int(res[1]))
            return _track_and_return(res)
        if isinstance(e, (Shl, LShr)):
            wa, aw = lower_expr_to_words(e.a)
            shift_expr = e.b
            if not isinstance(shift_expr, BitVecConst):
                raise PackedLoweringError(
                    message="non-constant shift amount not supported",
                    expression_type=type(e).__name__,
                    subexpression_context=f"{type(e).__name__} with dynamic shift amount",
                )
            sh = int(shift_expr.value)
            if sh < 0:
                raise PackedLoweringError(
                    message=f"negative shift not supported: {sh}",
                    expression_type=type(e).__name__,
                    subexpression_context=f"{type(e).__name__} by {sh}",
                )
            if sh == 0:
                res = (wa, aw)
                memo[ekey] = (list(res[0]), int(res[1]))
                return _track_and_return(res)
            if sh >= aw:
                # shift out completely
                zeros = [emit_const_u64(0) for _ in range(_ceil_div(aw, WORD_BITS))]
                res = (zeros, aw)
                memo[ekey] = (list(res[0]), int(res[1]))
                return _track_and_return(res)
            word_shift = sh // WORD_BITS
            inner = sh % WORD_BITS
            nwords = _ceil_div(aw, WORD_BITS)
            zeros_node = emit_const_u64(0)

            def get_word(i: int) -> int:
                if 0 <= i < len(wa):
                    return wa[i]
                return zeros_node

            out: list[int] = []
            if isinstance(e, Shl):
                for i in range(nwords):
                    src_i = i - word_shift
                    w0 = get_word(src_i)
                    if inner == 0:
                        out.append(w0)
                    else:
                        lo = emit_shift("shl", w0, inner)
                        hi = get_word(src_i - 1)
                        carry = emit_shift("lshr", hi, WORD_BITS - inner)
                        out.append(emit_bin("or", lo, carry))
            else:
                for i in range(nwords):
                    src_i = i + word_shift
                    w0 = get_word(src_i)
                    if inner == 0:
                        out.append(w0)
                    else:
                        lo = emit_shift("lshr", w0, inner)
                        hi = get_word(src_i + 1)
                        carry = emit_shift("shl", hi, WORD_BITS - inner)
                        out.append(emit_bin("or", lo, carry))
            # Mask top word if width not multiple of 64
            if aw % WORD_BITS:
                top_bits = aw % WORD_BITS
                mask = (1 << top_bits) - 1
                out[-1] = emit_bin("and", out[-1], emit_const_u64(mask))
            return _track_and_return((out, aw))
        if isinstance(e, (Rotl, Rotr)):
            wa, aw = lower_expr_to_words(e.x)
            sh_expr = e.sh
            if not isinstance(sh_expr, BitVecConst):
                raise PackedLoweringError(
                    message="non-constant rotate amount not supported",
                    expression_type=type(e).__name__,
                    subexpression_context=f"{type(e).__name__} with dynamic rotate amount",
                )
            sh = int(sh_expr.value) % aw
            if sh == 0:
                res = (wa, aw)
                memo[ekey] = (list(res[0]), int(res[1]))
                return _track_and_return(res)
            # Implement rotate using: rotl(x, k) = (x << k) | (x >> (n-k))
            # For rotr, swap the shift amounts.
            if isinstance(e, Rotl):
                left_sh = sh
                right_sh = aw - sh
            else:
                left_sh = aw - sh
                right_sh = sh
            # Compute left shift: x << left_sh
            left_word_shift = left_sh // WORD_BITS
            left_inner = left_sh % WORD_BITS
            # Compute right shift: x >> right_sh
            right_word_shift = right_sh // WORD_BITS
            right_inner = right_sh % WORD_BITS
            nwords = _ceil_div(aw, WORD_BITS)
            zeros_node = emit_const_u64(0)

            def get_word(i: int) -> int:
                if 0 <= i < len(wa):
                    return wa[i]
                return zeros_node

            # Build left-shifted words
            left_words: list[int] = []
            for i in range(nwords):
                src_i = i - left_word_shift
                w0 = get_word(src_i)
                if left_inner == 0:
                    left_words.append(w0)
                else:
                    lo = emit_shift("shl", w0, left_inner)
                    hi = get_word(src_i - 1)
                    carry = emit_shift("lshr", hi, WORD_BITS - left_inner)
                    left_words.append(emit_bin("or", lo, carry))
            # Build right-shifted words
            right_words: list[int] = []
            for i in range(nwords):
                src_i = i + right_word_shift
                w0 = get_word(src_i)
                if right_inner == 0:
                    right_words.append(w0)
                else:
                    lo = emit_shift("lshr", w0, right_inner)
                    hi = get_word(src_i + 1)
                    carry = emit_shift("shl", hi, WORD_BITS - right_inner)
                    right_words.append(emit_bin("or", lo, carry))
            # Combine with OR
            out: list[int] = []
            for i in range(nwords):
                out.append(emit_bin("or", left_words[i], right_words[i]))
            # Mask top word if width not multiple of 64
            if aw % WORD_BITS:
                top_bits = aw % WORD_BITS
                mask = (1 << top_bits) - 1
                out[-1] = emit_bin("and", out[-1], emit_const_u64(mask))
            return _track_and_return((out, aw))
        if isinstance(e, Eq):
            wa, aw = lower_expr_to_words(e.a)
            wb, bw = lower_expr_to_words(e.b)
            if aw != bw:
                raise PackedLoweringError(
                    message=f"width mismatch in equality comparison: {aw} vs {bw}",
                    expression_type="Eq",
                    subexpression_context=f"equality comparison of {aw}-bit and {bw}-bit values",
                )
            # XOR all word pairs to find differences
            diff_words: list[int] = []
            for i in range(len(wa)):
                diff_words.append(emit_bin("xor", wa[i], wb[i]))
            # OR-reduce all diff words to single word
            combined = diff_words[0]
            for i in range(1, len(diff_words)):
                combined = emit_bin("or", combined, diff_words[i])
            # Check if combined is zero (all bits equal)
            # Strategy: if combined == 0, result is 1; otherwise 0
            # Use: result = (combined == 0) ? 1 : 0
            # Implement as: result = 1 - (combined != 0)
            # To check (combined != 0), we can use: (combined | -combined) >> 63
            # But simpler: use subtraction to create a mask
            # If combined == 0, then (0 - (combined | -combined)) >> 63 == 0
            # Actually simplest: negate combined, check if it became 0
            # Better approach: use the fact that if combined == 0, all bits are 0
            # We want: combined == 0 ? 0xFFFF...FFFF : 0x0
            # Then mask with 1 to get the boolean result
            # Actually, we need to implement: is_zero = (combined == 0) as a boolean
            # Technique: Create a bitmask from combined
            # If combined != 0, at least one bit is set
            # Use: sign = (combined | -combined) to propagate any set bit to sign
            # Then negate to get 0 if any bit was set, or stay 0 if all zero
            # Actually for equality check returning bool (1 bit):
            # We can use: ~(combined | -combined) >> 63 but need arithmetic
            # Simpler for packed circuit: use (0 - combined) to negate
            # If combined == 0: result = 1
            # If combined != 0: result = 0
            # Method: Create boolean from (combined == 0)
            # Use double negation trick:
            # Step 1: or_neg = combined | (0 - combined)  [sign bit set if != 0]
            # Step 2: shift right 63 to get sign bit [gives 1 if != 0, 0 if == 0]
            # Step 3: xor with 1 to invert [gives 0 if != 0, 1 if == 0]
            zero = emit_const_u64(0)
            neg_combined = emit_bin("sub", zero, combined)
            or_neg = emit_bin("or", combined, neg_combined)
            sign_bit = emit_shift("lshr", or_neg, 63)
            one = emit_const_u64(1)
            result = emit_bin("xor", sign_bit, one)
            return _track_and_return(([result], 1))
        if isinstance(e, Mux):
            cond_words, cond_w = lower_expr_to_words(e.cond)
            if cond_w != 1:
                raise PackedLoweringError(
                    message=f"mux condition must be bool (1 bit), got {cond_w} bits",
                    expression_type="Mux",
                    subexpression_context=f"mux with {cond_w}-bit condition",
                )
            cond = cond_words[0]
            # Expand cond (1-bit) to full mask: 0 → 0x0, 1 → 0xFFFF...FFFF
            # Use: mask = 0 - cond
            zero = emit_const_u64(0)
            mask = emit_bin("sub", zero, cond)
            # Lower both branches
            a_words, a_w = lower_expr_to_words(e.a)
            b_words, b_w = lower_expr_to_words(e.b)
            if a_w != b_w:
                raise PackedLoweringError(
                    message=f"mux branch width mismatch: {a_w} vs {b_w}",
                    expression_type="Mux",
                    subexpression_context=f"mux with {a_w}-bit and {b_w}-bit branches",
                )
            # For each word: result[i] = (mask & a[i]) | (~mask & b[i])
            not_mask = emit_unary("not", mask)
            result_words: list[int] = []
            for i in range(len(a_words)):
                masked_a = emit_bin("and", mask, a_words[i])
                masked_b = emit_bin("and", not_mask, b_words[i])
                merged = emit_bin("or", masked_a, masked_b)
                result_words.append(merged)
            return _track_and_return((result_words, a_w))
        if isinstance(e, (Add, Sub)):
            wa, aw = lower_expr_to_words(e.a)
            wb, bw = lower_expr_to_words(e.b)
            if aw != bw:
                raise PackedLoweringError(
                    message=f"width mismatch in {type(e).__name__}: {aw} vs {bw}",
                    expression_type=type(e).__name__,
                    subexpression_context=f"{type(e).__name__} with widths {aw} and {bw}",
                )
            # Multiword add/sub with carry/borrow propagation.
            result_words: list[int] = []
            carry = emit_const_u64(0)

            if isinstance(e, Add):
                for i in range(len(wa)):
                    # Compute a[i] + b[i] + carry
                    temp = emit_bin("add", wa[i], wb[i])
                    result = emit_bin("add", temp, carry)
                    result_words.append(result)
                    # Detect carry-out using: carry = ((a & b) | ((a ^ b) & ~sum)) >> 63
                    # But we need to account for the carry-in too.
                    # carry_from_ab = ((a & b) | ((a ^ b) & ~temp)) >> 63
                    # carry_from_temp_cin = ((temp & cin) | ((temp ^ cin) & ~result)) >> 63
                    # carry_out = carry_from_ab | carry_from_temp_cin
                    if i < len(wa) - 1:
                        # Carry from a[i] + b[i]
                        a_and_b = emit_bin("and", wa[i], wb[i])
                        a_xor_b = emit_bin("xor", wa[i], wb[i])
                        not_temp = emit_unary("not", temp)
                        term1 = emit_bin("and", a_xor_b, not_temp)
                        carry_ab = emit_bin("or", a_and_b, term1)
                        carry_ab_bit = emit_shift("lshr", carry_ab, 63)
                        # Carry from temp + carry_in
                        temp_and_cin = emit_bin("and", temp, carry)
                        temp_xor_cin = emit_bin("xor", temp, carry)
                        not_result = emit_unary("not", result)
                        term2 = emit_bin("and", temp_xor_cin, not_result)
                        carry_cin = emit_bin("or", temp_and_cin, term2)
                        carry_cin_bit = emit_shift("lshr", carry_cin, 63)
                        # Combine carries
                        carry = emit_bin("or", carry_ab_bit, carry_cin_bit)
            else:
                for i in range(len(wa)):
                    # Compute a[i] - b[i] - borrow
                    temp = emit_bin("sub", wa[i], wb[i])
                    result = emit_bin("sub", temp, carry)
                    result_words.append(result)
                    # Detect borrow-out using: borrow = ((~a & b) | ((~a ^ b) & sum)) >> 63
                    # borrow_from_ab = ((~a & b) | ((~a ^ b) & temp)) >> 63
                    # borrow_from_temp_cin = ((~temp & borrow) | ((~temp ^ borrow) & result)) >> 63
                    # borrow_out = borrow_from_ab | borrow_from_temp_cin
                    if i < len(wa) - 1:
                        # Borrow from a[i] - b[i]
                        not_a = emit_unary("not", wa[i])
                        not_a_and_b = emit_bin("and", not_a, wb[i])
                        not_a_xor_b = emit_bin("xor", not_a, wb[i])
                        term1 = emit_bin("and", not_a_xor_b, temp)
                        borrow_ab = emit_bin("or", not_a_and_b, term1)
                        borrow_ab_bit = emit_shift("lshr", borrow_ab, 63)
                        # Borrow from temp - borrow_in
                        not_temp = emit_unary("not", temp)
                        not_temp_and_cin = emit_bin("and", not_temp, carry)
                        not_temp_xor_cin = emit_bin("xor", not_temp, carry)
                        term2 = emit_bin("and", not_temp_xor_cin, result)
                        borrow_cin = emit_bin("or", not_temp_and_cin, term2)
                        borrow_cin_bit = emit_shift("lshr", borrow_cin, 63)
                        # Combine borrows
                        carry = emit_bin("or", borrow_ab_bit, borrow_cin_bit)

            # Mask the top word if width not multiple of 64
            if aw % WORD_BITS:
                top_bits = aw % WORD_BITS
                mask = (1 << top_bits) - 1
                result_words[-1] = emit_bin(
                    "and", result_words[-1], emit_const_u64(mask)
                )
            return _track_and_return((result_words, aw))

        if isinstance(e, Mul):
            # Try to infer width for better error message
            width = None
            if isinstance(e.a, Var) and e.a.name in var_widths:
                width = var_widths[e.a.name]
            elif isinstance(e.a, BitVecConst):
                width = e.a.width
            width_str = str(width) if width else "unknown"
            raise PackedLoweringError(
                message=f"Mul with width {width_str} not supported in packed lowering. Suggestion: Use power-of-2 constants (will be reduced to shifts by classification pass) or pre-map multiplication in Verilog synthesis.",
                expression_type="Mul",
                subexpression_context=f"Mul operation with {width_str}-bit operands",
            )

        if isinstance(e, Div):
            # Try to infer width for better error message
            width = None
            if isinstance(e.a, Var) and e.a.name in var_widths:
                width = var_widths[e.a.name]
            elif isinstance(e.a, BitVecConst):
                width = e.a.width
            width_str = str(width) if width else "unknown"
            raise PackedLoweringError(
                message=f"Div with width {width_str} not supported in packed lowering. Suggestion: Use power-of-2 constants (will be reduced to LShr by classification pass) or pre-map division in Verilog synthesis.",
                expression_type="Div",
                subexpression_context=f"Div operation with {width_str}-bit operands",
            )

        raise PackedLoweringError(
            message=f"unsupported expression for packed lowering: {type(e).__name__}",
            expression_type=type(e).__name__,
            subexpression_context=f"expression of type {type(e).__name__}",
        )

    # Lower outputs and next_state into output word nodes, in packed order.
    output_nodes: list[tuple[int, bool]] = []

    # Primary outputs
    for name in output_order:
        expr = ir.output_exprs[name]
        ws, w = lower_expr_to_words(expr)
        want = width_words(ir.outputs[name])
        if len(ws) != want:
            raise PackedLoweringError(
                message=f"output word width mismatch for '{name}': expected {want} words, got {len(ws)}",
                variable_name=name,
                subexpression_context=f"output expression for {name}",
            )
        output_nodes.extend((wi, False) for wi in ws)

    # Next state (in state order)
    for name in state_order:
        expr = ir.next_state[name]
        ws, w = lower_expr_to_words(expr)
        want = width_words(ir.state[name])
        if len(ws) != want:
            raise PackedLoweringError(
                message=f"next_state word width mismatch for '{name}': expected {want} words, got {len(ws)}",
                variable_name=name,
                subexpression_context=f"next_state expression for {name}",
            )
        output_nodes.extend((wi, False) for wi in ws)

    circuit = PackedCircuitState(
        word_bits=WORD_BITS,
        input_words=input_words,
        output_words=output_words,
        gates=tuple(gates),
        outputs=tuple(output_nodes),
    )

    debug_enabled = os.environ.get("STC_DEBUG_GATE_EXPLOSION", "") in {
        "1",
        "true",
        "TRUE",
    }

    if debug_enabled:
        print(f"\n=== Packed Lowering Gate Statistics ===", file=sys.stderr)
        print(f"Total gates generated: {len(gates)}", file=sys.stderr)
        print(f"Total gates tracked: {gate_stats.total_gates}", file=sys.stderr)

        if gate_stats.expr_gate_counts:
            sorted_expr_types = sorted(
                gate_stats.expr_gate_counts.items(), key=lambda x: x[1], reverse=True
            )
            print(f"\nTop 5 expression types by gate count:", file=sys.stderr)
            for expr_type, count in sorted_expr_types[:5]:
                print(f"  {expr_type}: {count} gates", file=sys.stderr)

        if gate_stats.large_expressions:
            print(f"\nExpressions that generated > 1000 gates:", file=sys.stderr)
            for expr_type, count in gate_stats.large_expressions:
                print(f"  {expr_type}: {count} gates", file=sys.stderr)

        print(f"=====================================\n", file=sys.stderr)

        print(f"\n=== Memoization Statistics ===", file=sys.stderr)
        print(
            f"Total expressions lowered: {memo_stats.hits + memo_stats.misses}",
            file=sys.stderr,
        )
        print(f"Memo hits: {memo_stats.hits}", file=sys.stderr)
        print(f"Memo misses: {memo_stats.misses}", file=sys.stderr)
        print(f"Hit rate: {memo_stats.hit_rate() * 100:.1f}%", file=sys.stderr)

        if memo_stats.hits > 0:
            dedup_ratio = (memo_stats.hits + memo_stats.misses) / memo_stats.misses
            print(f"Deduplication ratio: {dedup_ratio:.1f}x", file=sys.stderr)

        all_types = set(memo_stats.hits_by_type.keys()) | set(
            memo_stats.misses_by_type.keys()
        )
        if all_types:
            type_stats = []
            for expr_type in all_types:
                hits = memo_stats.hits_by_type[expr_type]
                misses = memo_stats.misses_by_type[expr_type]
                total = hits + misses
                hit_rate = hits / total if total > 0 else 0.0
                type_stats.append((expr_type, hits, misses, total, hit_rate))

            type_stats.sort(key=lambda x: x[3], reverse=True)
            print(f"\nMemo hit rates by expression type (top 10):", file=sys.stderr)
            for expr_type, hits, misses, total, hit_rate in type_stats[:10]:
                print(
                    f"  {expr_type}: {hits}/{total} hits ({hit_rate * 100:.1f}%)",
                    file=sys.stderr,
                )

        print(f"==============================\n", file=sys.stderr)

    if len(gates) > 100_000:
        print(
            f"WARNING: Packed lowering generated {len(gates)} gates",
            file=sys.stderr,
        )
        print(
            "  This design may be better suited for bit-level lowering.",
            file=sys.stderr,
        )
        print(
            "  Consider using bit-level lowering (without --force-packed) for more efficient compilation.",
            file=sys.stderr,
        )

    return (circuit, layout)
