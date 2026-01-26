from __future__ import annotations

from dataclasses import dataclass

from stc.packed_circuit import PackedCircuitState, PackedGate
from stc.interp import infer_type
from stc.tick_ir import (
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
    Rotl,
    Rotr,
    Shl,
    Slice,
    TickIR,
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

    def to_dict(self) -> dict:
        return {
            "inputs": self.inputs,
            "state": self.state,
            "outputs": self.outputs,
            "next_state": self.next_state,
            "input_words": self.input_words,
            "output_words": self.output_words,
        }


WORD_BITS = 64
WORD_MASK = (1 << WORD_BITS) - 1


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


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
        if isinstance(e, Var):
            ws, w = get_var_words(e)
            return (list(ws), w)
        if isinstance(e, BoolConst):
            node = emit_const_u64(1 if e.value else 0)
            return ([node], 1)
        if isinstance(e, BitVecConst):
            nwords = _ceil_div(e.width, WORD_BITS)
            out: list[int] = []
            for i in range(nwords):
                chunk = (e.value >> (i * WORD_BITS)) & WORD_MASK
                out.append(emit_const_u64(chunk))
            return (out, e.width)
        if isinstance(e, Not):
            ws, w = lower_expr_to_words(e.x)
            out = [emit_unary("not", wi) for wi in ws]
            return (out, w)
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
            return (out, aw)
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
                return (src_words[start : start + nwords], e.width)
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
                return (out, e.width)
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
            return (out, e.width)
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
                return (out_words, total_w)
            # General case: pack mixed-width parts across word boundaries.
            # Concat semantics: parts[0] is MSB, parts[-1] is LSB.
            # We build output words LSB-first, so process parts in reverse.
            out_words: list[int] = []
            current_word = emit_const_u64(0)
            bits_in_current = 0
            for ws, part_width in reversed(parts):
                # Extract bits from this part and pack into output words.
                part_bit_offset = 0
                while part_bit_offset < part_width:
                    bits_remaining_in_part = part_width - part_bit_offset
                    space_in_current = WORD_BITS - bits_in_current
                    bits_to_copy = min(bits_remaining_in_part, space_in_current)
                    # Extract bits_to_copy from part starting at part_bit_offset.
                    part_word_idx = part_bit_offset // WORD_BITS
                    bit_offset_in_word = part_bit_offset % WORD_BITS
                    src_word = ws[part_word_idx]
                    # Shift right to align the bits we want to the LSB.
                    if bit_offset_in_word > 0:
                        src_word = emit_shift("lshr", src_word, bit_offset_in_word)
                    # If we need bits spanning two words, merge them.
                    if (
                        bit_offset_in_word + bits_to_copy > WORD_BITS
                        and part_word_idx + 1 < len(ws)
                    ):
                        next_word = ws[part_word_idx + 1]
                        carry_bits = bit_offset_in_word + bits_to_copy - WORD_BITS
                        carry = emit_shift(
                            "shl", next_word, WORD_BITS - bit_offset_in_word
                        )
                        src_word = emit_bin("or", src_word, carry)
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
                    # If current word is full, emit it and start a new one.
                    if bits_in_current == WORD_BITS:
                        out_words.append(current_word)
                        current_word = emit_const_u64(0)
                        bits_in_current = 0
            # Emit final partial word if any bits remain.
            if bits_in_current > 0:
                out_words.append(current_word)
            return (out_words, total_w)
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
                return (wa, aw)
            if sh >= aw:
                # shift out completely
                zeros = [emit_const_u64(0) for _ in range(_ceil_div(aw, WORD_BITS))]
                return (zeros, aw)
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
            return (out, aw)
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
                return (wa, aw)
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
            return (out, aw)
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
            return ([result], 1)
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
            return (result_words, a_w)

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
    return (circuit, layout)
