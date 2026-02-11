from __future__ import annotations

from dataclasses import dataclass

from stc.subset import check_subset
from stc.tick_ir import (
    AShr,
    Add,
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
    TernaryLut,
    Lut8,
    Sub,
    TickIR,
    Type,
    Uge,
    Ugt,
    Ule,
    Ult,
    Var,
    Xor,
)
from stc.yosys_json import YosysCell, YosysDesign


@dataclass(frozen=True)
class ExtractionError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def _parse_param_int(value: str) -> int:
    v = value.strip()
    if v and all(c in "01" for c in v):
        return int(v, 2)
    return int(v, 10)


def _parse_lut_bits(raw: str, *, size: int) -> list[int]:
    s = raw.strip()
    if not s:
        raise ExtractionError("lut parameter empty")
    # Support Verilog-like width/base literals (e.g., 8'b1010, 8'hA5)
    if "'" in s:
        parts = s.split("'")
        if len(parts) == 2:
            _, rhs = parts
            if rhs:
                base = rhs[0].lower()
                digits = rhs[1:]
                if base == "b":
                    s = digits
                elif base == "h":
                    v = int(digits, 16)
                    s = bin(v)[2:]
                elif base == "d":
                    v = int(digits, 10)
                    s = bin(v)[2:]
    if any(c in "abcdefABCDEF" for c in s):
        v = int(s, 16)
        s = bin(v)[2:]
    if any(c not in "01" for c in s):
        raise ExtractionError("lut parameter must be binary/hex")
    bits = [1 if c == "1" else 0 for c in s[::-1]]  # LSB-first
    if len(bits) < size:
        bits += [0] * (size - len(bits))
    if len(bits) != size:
        raise ExtractionError("lut parameter size mismatch")
    return bits


def _lut1_expr(x: Expr, bits: list[int]) -> Expr:
    if bits[0] == 0 and bits[1] == 0:
        return BoolConst(value=False)
    if bits[0] == 1 and bits[1] == 1:
        return BoolConst(value=True)
    if bits[0] == 0 and bits[1] == 1:
        return x
    if bits[0] == 1 and bits[1] == 0:
        return Not(x=x)
    raise ExtractionError("invalid LUT1 bits")


def _lut2_expr(x: Expr, y: Expr, bits: list[int]) -> Expr:
    # bits index = x + 2*y (LSB-first)
    t = bits[0] | (bits[1] << 1) | (bits[2] << 2) | (bits[3] << 3)
    if t == 0b0000:
        return BoolConst(value=False)
    if t == 0b1111:
        return BoolConst(value=True)
    if t == 0b1010:
        return x
    if t == 0b1100:
        return y
    if t == 0b0101:
        return Not(x=x)
    if t == 0b0011:
        return Not(x=y)
    if t == 0b1000:
        return And(a=x, b=y)
    if t == 0b1110:
        return Or(a=x, b=y)
    if t == 0b0110:
        return Xor(a=x, b=y)
    if t == 0b1001:
        return Not(x=Xor(a=x, b=y))
    # Shannon expansion via mux on y, then x
    def _const(v: int) -> Expr:
        return BoolConst(value=bool(v))

    f0 = Mux(cond=x, a=_const(bits[1]), b=_const(bits[0]))
    f1 = Mux(cond=x, a=_const(bits[3]), b=_const(bits[2]))
    return Mux(cond=y, a=f1, b=f0)


def _port_type(bits: list[int]) -> Type:
    if len(bits) == 1:
        return BoolType()
    return BitVecType(width=len(bits))


def _bus_from_bits(
    bits_lsb_first: list[int], bit_expr: callable, width_of_expr: callable
) -> Expr:
    if len(bits_lsb_first) == 1:
        return bit_expr(bits_lsb_first[0])

    bit_exprs_lsb = [bit_expr(b) for b in bits_lsb_first]

    segments: list[Expr] = []
    i = 0
    while i < len(bit_exprs_lsb):
        e = bit_exprs_lsb[i]
        if isinstance(e, Slice) and e.width == 1:
            base = e.x
            start = e.offset
            j = i + 1
            while j < len(bit_exprs_lsb):
                ej = bit_exprs_lsb[j]
                if (
                    isinstance(ej, Slice)
                    and ej.width == 1
                    and ej.x == base
                    and ej.offset == start + (j - i)
                ):
                    j += 1
                    continue
                break
            run_len = j - i
            if run_len == 1:
                seg = e
            else:
                seg = Slice(x=base, offset=start, width=run_len)
            try:
                if start == 0 and run_len == width_of_expr(base):
                    seg = base
            except Exception:
                pass
            segments.append(seg)
            i = j
            continue
        segments.append(e)
        i += 1

    if len(segments) == 1:
        return segments[0]
    return Concat(parts=list(reversed(segments)))


def extract_tick_ir(design: YosysDesign) -> TickIR:
    check_subset(design)
    module = design.modules[design.top]

    def decode_mem_init_bits(init: str, *, size: int, width: int) -> list[int]:
        s = init.strip()
        if not s or any(c not in {"0", "1"} for c in s):
            raise ExtractionError("mem init must be a bitstring")
        if len(s) != size * width:
            raise ExtractionError("mem init length mismatch")
        bits_lsb_first = [1 if c == "1" else 0 for c in s[::-1]]
        table: list[int] = []
        for i in range(size):
            v = 0
            base = i * width
            for k in range(width):
                v |= bits_lsb_first[base + k] << k
            table.append(v)
        return table

    mem_tables: dict[str, list[int]] = {}
    for cell in module.cells.values():
        if cell.type not in {"$meminit_v2", "$meminit"}:
            continue
        memid = cell.parameters.get("MEMID")
        if memid is None:
            raise ExtractionError("meminit missing MEMID")
        width = _parse_param_int(cell.parameters.get("WIDTH", "0"))
        words = _parse_param_int(cell.parameters.get("WORDS", "0"))
        if width != 8 or words != 256:
            raise ExtractionError("meminit supports only WIDTH=8, WORDS=256")
        bits = cell.connections.get("DATA")
        if bits is None:
            raise ExtractionError("meminit missing DATA")
        if len(bits) != width * words:
            raise ExtractionError("meminit DATA width mismatch")
        table: list[int] = []
        for i in range(words):
            v = 0
            base = i * width
            for k in range(width):
                v |= int(bits[base + k]) << k
            table.append(v)
        if memid in mem_tables:
            raise ExtractionError("multiple meminit for same MEMID")
        mem_tables[memid] = table

    inputs: dict[str, Type] = {}
    outputs: dict[str, Type] = {}
    input_bits: dict[int, Expr] = {}
    state: dict[str, Type] = {}
    state_bits: dict[int, Expr] = {}
    reset_state: dict[str, Expr] = {}
    next_state: dict[str, Expr] = {}

    for port in module.ports.values():
        t = _port_type(port.bits)
        if port.direction == "input":
            if port.name == "clk":
                continue
            inputs[port.name] = t
            if isinstance(t, BoolType):
                input_bits[port.bits[0]] = Var(name=port.name)
            else:
                assert isinstance(t, BitVecType)
                for i, bit in enumerate(port.bits):
                    input_bits[bit] = Slice(x=Var(name=port.name), offset=i, width=1)
        else:
            outputs[port.name] = t

    seq_cells = {
        "$dff",
        "$dffe",
        "$sdff",
        "$sdffe",
        "$_SDFF_PP0_",
        "$_SDFFE_PP0P_",
        "$_SDFFE_PP0N_",
    }
    for cell in module.cells.values():
        if cell.type not in seq_cells:
            continue
        q_bits = cell.connections["Q"]
        d_bits = cell.connections["D"]
        if len(q_bits) != len(d_bits):
            raise ExtractionError("dff D and Q widths must match")
        t = _port_type(q_bits)
        state[cell.name] = t
        if isinstance(t, BoolType):
            if cell.type in {
                "$sdff",
                "$sdffe",
                "$_SDFF_PP0_",
                "$_SDFFE_PP0P_",
                "$_SDFFE_PP0N_",
            }:
                v = _parse_param_int(cell.parameters.get("SRST_VALUE", "0"))
                reset_state[cell.name] = BoolConst(value=bool(v & 1))
            else:
                reset_state[cell.name] = BoolConst(value=False)
            state_bits[q_bits[0]] = Var(name=cell.name)
        else:
            assert isinstance(t, BitVecType)
            if cell.type in {
                "$sdff",
                "$sdffe",
                "$_SDFF_PP0_",
                "$_SDFFE_PP0P_",
                "$_SDFFE_PP0N_",
            }:
                v = _parse_param_int(cell.parameters.get("SRST_VALUE", "0"))
                reset_state[cell.name] = BitVecConst(width=t.width, value=v)
            else:
                reset_state[cell.name] = BitVecConst(width=t.width, value=0)
            for i, bit in enumerate(q_bits):
                state_bits[bit] = Slice(x=Var(name=cell.name), offset=i, width=1)

    bit_drivers: dict[int, tuple[str, str, int]] = {}
    for cell_name, cell in module.cells.items():
        if cell.type in seq_cells:
            continue
        for port_name, direction in cell.port_directions.items():
            if direction != "output":
                continue
            bits = cell.connections.get(port_name, [])
            for i, bit in enumerate(bits):
                if bit in {0, 1}:
                    continue
                if bit in bit_drivers:
                    raise ExtractionError(f"multiple drivers for net {bit}")
                bit_drivers[bit] = (cell_name, port_name, i)

    cell_out_cache: dict[tuple[str, str], Expr] = {}
    bit_cache: dict[int, Expr] = {}

    var_types: dict[str, Type] = {**inputs, **state}

    def width_of_expr(expr: Expr) -> int:
        if isinstance(expr, BoolConst):
            return 1
        if isinstance(expr, BitVecConst):
            return expr.width
        if isinstance(expr, Var):
            t = var_types.get(expr.name)
            if isinstance(t, BoolType):
                return 1
            assert isinstance(t, BitVecType)
            return t.width
        if isinstance(expr, Not):
            return width_of_expr(expr.x)
        if isinstance(expr, (And, Or, Xor, Add, Sub, Shl, LShr, AShr)):
            return width_of_expr(expr.a)
        if isinstance(expr, (Eq, Ult, Ule, Ugt, Uge)):
            return 1
        if isinstance(expr, Mux):
            return width_of_expr(expr.a)
        if isinstance(expr, Concat):
            return sum(width_of_expr(p) for p in expr.parts)
        if isinstance(expr, Slice):
            return 1 if expr.width == 1 else expr.width
        if isinstance(expr, Lut8):
            return 8
        raise ExtractionError("cannot infer width")

    def expr_for_bit(bit: int) -> Expr:
        if bit in bit_cache:
            return bit_cache[bit]
        if bit == 0:
            bit_cache[bit] = BoolConst(value=False)
            return bit_cache[bit]
        if bit == 1:
            bit_cache[bit] = BoolConst(value=True)
            return bit_cache[bit]
        if bit in input_bits:
            bit_cache[bit] = input_bits[bit]
            return bit_cache[bit]
        if bit in state_bits:
            bit_cache[bit] = state_bits[bit]
            return bit_cache[bit]
        src = bit_drivers.get(bit)
        if src is None:
            raise ExtractionError(f"no driver for net {bit}")
        cell_name, port_name, offset = src
        cell = module.cells[cell_name]
        bus = expr_for_cell_output(cell, port_name)
        out_bits = cell.connections.get(port_name, [])
        if len(out_bits) == 1:
            bit_cache[bit] = bus
        else:
            bit_cache[bit] = Slice(x=bus, offset=offset, width=1)
        return bit_cache[bit]

    def expr_for_cell_output(cell: YosysCell, out_port: str) -> Expr:
        key = (cell.name, out_port)
        if key in cell_out_cache:
            return cell_out_cache[key]

        def bus(port_name: str) -> Expr:
            bits = cell.connections.get(port_name)
            if bits is None:
                raise ExtractionError(f"missing port {port_name} on {cell.name}")
            return _bus_from_bits(bits, expr_for_bit, width_of_expr)

        y_bits = cell.connections.get(out_port, [])
        y_width = len(y_bits)

        def _repeat_bool(bit: Expr, n: int) -> Expr:
            if n == 1:
                return bit
            return Concat(parts=tuple(bit for _ in range(n)))

        def ext_to_width(x: Expr, from_w: int, to_w: int, *, signed: bool) -> Expr:
            if from_w == to_w:
                return x
            if from_w < 1 or to_w < 1 or from_w > to_w:
                raise ExtractionError("invalid zext widths")
            pad = to_w - from_w
            if pad == 0:
                return x
            # Concat parts are MSB-first; pad on the left.
            if signed:
                sign = Slice(x=x, offset=from_w - 1, width=1) if from_w > 1 else x
                pad_expr = _repeat_bool(sign, pad)
                return Concat(parts=(pad_expr, x))
            if pad == 1:
                return Concat(parts=(BoolConst(value=False), x))
            return Concat(parts=(BitVecConst(width=pad, value=0), x))

        def match_widths_to(target_w: int, a: Expr, b: Expr) -> tuple[Expr, Expr]:
            wa = width_of_expr(a)
            wb = width_of_expr(b)
            if wa != target_w:
                a = ext_to_width(
                    a,
                    wa,
                    target_w,
                    signed=bool(_parse_param_int(cell.parameters.get("A_SIGNED", "0"))),
                )
            if wb != target_w:
                b = ext_to_width(
                    b,
                    wb,
                    target_w,
                    signed=bool(_parse_param_int(cell.parameters.get("B_SIGNED", "0"))),
                )
            return (a, b)

        t = cell.type
        if t == "$not":
            expr = Not(x=bus("A"))
        elif t == "$logic_not":
            a = bus("A")
            w = width_of_expr(a)
            if w == 1:
                expr = Not(x=a)
            else:
                expr = Eq(a=a, b=BitVecConst(width=w, value=0))
        elif t == "$and":
            a, b = match_widths_to(y_width, bus("A"), bus("B"))
            expr = And(a=a, b=b)
        elif t == "$or":
            a, b = match_widths_to(y_width, bus("A"), bus("B"))
            expr = Or(a=a, b=b)
        elif t == "$xor":
            a, b = match_widths_to(y_width, bus("A"), bus("B"))
            expr = Xor(a=a, b=b)
        elif t == "$add":
            a, b = match_widths_to(y_width, bus("A"), bus("B"))
            expr = Add(a=a, b=b)
        elif t == "$sub":
            a, b = match_widths_to(y_width, bus("A"), bus("B"))
            expr = Sub(a=a, b=b)
        elif t == "$shl":
            a, b = match_widths_to(y_width, bus("A"), bus("B"))
            expr = Shl(a=a, b=b)
        elif t == "$shr":
            a, b = match_widths_to(y_width, bus("A"), bus("B"))
            expr = LShr(a=a, b=b)
        elif t == "$sshr":
            a, b = match_widths_to(y_width, bus("A"), bus("B"))
            expr = AShr(a=a, b=b)
        elif t == "$eq":
            a, b = match_widths_to(
                max(width_of_expr(bus("A")), width_of_expr(bus("B"))),
                bus("A"),
                bus("B"),
            )
            expr = Eq(a=a, b=b)
        elif t == "$ne":
            a, b = match_widths_to(
                max(width_of_expr(bus("A")), width_of_expr(bus("B"))),
                bus("A"),
                bus("B"),
            )
            expr = Not(x=Eq(a=a, b=b))
        elif t == "$lt":
            a, b = match_widths_to(
                max(width_of_expr(bus("A")), width_of_expr(bus("B"))),
                bus("A"),
                bus("B"),
            )
            expr = Ult(a=a, b=b)
        elif t == "$le":
            a, b = match_widths_to(
                max(width_of_expr(bus("A")), width_of_expr(bus("B"))),
                bus("A"),
                bus("B"),
            )
            expr = Ule(a=a, b=b)
        elif t == "$gt":
            a, b = match_widths_to(
                max(width_of_expr(bus("A")), width_of_expr(bus("B"))),
                bus("A"),
                bus("B"),
            )
            expr = Ugt(a=a, b=b)
        elif t == "$ge":
            a, b = match_widths_to(
                max(width_of_expr(bus("A")), width_of_expr(bus("B"))),
                bus("A"),
                bus("B"),
            )
            expr = Uge(a=a, b=b)
        elif t == "$mux":
            s_bits = cell.connections.get("S")
            if s_bits is None or len(s_bits) != 1:
                raise ExtractionError("mux select must be 1 bit")
            sel = expr_for_bit(s_bits[0])
            a = bus("A")
            b = bus("B")
            expr = Mux(cond=sel, a=b, b=a)
        elif t == "$pmux":
            # Priority mux:
            # - A: default value (width W)
            # - B: concatenation of K alternatives, each width W (LSB chunk is index 0)
            # - S: K 1-bit selects
            #
            # Semantics: cascade muxes so higher-index selects override lower ones.
            a = bus("A")
            b_bits = cell.connections.get("B")
            s_bits = cell.connections.get("S")
            if b_bits is None or s_bits is None:
                raise ExtractionError("pmux requires B and S ports")
            if len(s_bits) < 1:
                raise ExtractionError("pmux requires at least 1 select bit")
            w = width_of_expr(a)
            if len(b_bits) != w * len(s_bits):
                raise ExtractionError("pmux B width must be A_WIDTH * S_WIDTH")

            out = a
            # B is laid out as consecutive W-bit chunks, one per select bit.
            for i in range(len(s_bits)):
                sel = expr_for_bit(s_bits[i])
                chunk_bits = b_bits[i * w : (i + 1) * w]
                alt = _bus_from_bits(chunk_bits, expr_for_bit, width_of_expr)
                out = Mux(cond=sel, a=alt, b=out)
            expr = out
        elif t == "$reduce_or":
            a = bus("A")
            w = width_of_expr(a)
            if w == 1:
                expr = a
            else:
                expr = Not(x=Eq(a=a, b=BitVecConst(width=w, value=0)))
        elif t == "$reduce_bool":
            a = bus("A")
            w = width_of_expr(a)
            if w == 1:
                expr = a
            else:
                expr = Not(x=Eq(a=a, b=BitVecConst(width=w, value=0)))
        elif t == "$_NOT_":
            a_bit = cell.connections.get("A")
            if a_bit is None or len(a_bit) != 1:
                raise ExtractionError("$_NOT_ requires single-bit A")
            expr = Not(x=expr_for_bit(a_bit[0]))
        elif t == "$_AND_":
            a_bit = cell.connections.get("A")
            b_bit = cell.connections.get("B")
            if a_bit is None or len(a_bit) != 1:
                raise ExtractionError("$_AND_ requires single-bit A")
            if b_bit is None or len(b_bit) != 1:
                raise ExtractionError("$_AND_ requires single-bit B")
            expr = And(a=expr_for_bit(a_bit[0]), b=expr_for_bit(b_bit[0]))
        elif t == "$_OR_":
            a_bit = cell.connections.get("A")
            b_bit = cell.connections.get("B")
            if a_bit is None or len(a_bit) != 1:
                raise ExtractionError("$_OR_ requires single-bit A")
            if b_bit is None or len(b_bit) != 1:
                raise ExtractionError("$_OR_ requires single-bit B")
            expr = Or(a=expr_for_bit(a_bit[0]), b=expr_for_bit(b_bit[0]))
        elif t == "$_XOR_":
            a_bit = cell.connections.get("A")
            b_bit = cell.connections.get("B")
            if a_bit is None or len(a_bit) != 1:
                raise ExtractionError("$_XOR_ requires single-bit A")
            if b_bit is None or len(b_bit) != 1:
                raise ExtractionError("$_XOR_ requires single-bit B")
            expr = Xor(a=expr_for_bit(a_bit[0]), b=expr_for_bit(b_bit[0]))
        elif t == "$_XNOR_":
            a_bit = cell.connections.get("A")
            b_bit = cell.connections.get("B")
            if a_bit is None or len(a_bit) != 1:
                raise ExtractionError("$_XNOR_ requires single-bit A")
            if b_bit is None or len(b_bit) != 1:
                raise ExtractionError("$_XNOR_ requires single-bit B")
            expr = Not(x=Xor(a=expr_for_bit(a_bit[0]), b=expr_for_bit(b_bit[0])))
        elif t == "$_NAND_":
            a_bit = cell.connections.get("A")
            b_bit = cell.connections.get("B")
            if a_bit is None or len(a_bit) != 1:
                raise ExtractionError("$_NAND_ requires single-bit A")
            if b_bit is None or len(b_bit) != 1:
                raise ExtractionError("$_NAND_ requires single-bit B")
            expr = Not(x=And(a=expr_for_bit(a_bit[0]), b=expr_for_bit(b_bit[0])))
        elif t == "$_NOR_":
            a_bit = cell.connections.get("A")
            b_bit = cell.connections.get("B")
            if a_bit is None or len(a_bit) != 1:
                raise ExtractionError("$_NOR_ requires single-bit A")
            if b_bit is None or len(b_bit) != 1:
                raise ExtractionError("$_NOR_ requires single-bit B")
            expr = Not(x=Or(a=expr_for_bit(a_bit[0]), b=expr_for_bit(b_bit[0])))
        elif t == "$_MUX_":
            s_bit = cell.connections.get("S")
            a_bit = cell.connections.get("A")
            b_bit = cell.connections.get("B")
            if s_bit is None or len(s_bit) != 1:
                raise ExtractionError("$_MUX_ requires single-bit S")
            if a_bit is None or len(a_bit) != 1:
                raise ExtractionError("$_MUX_ requires single-bit A")
            if b_bit is None or len(b_bit) != 1:
                raise ExtractionError("$_MUX_ requires single-bit B")
            expr = Mux(
                cond=expr_for_bit(s_bit[0]),
                a=expr_for_bit(b_bit[0]),
                b=expr_for_bit(a_bit[0]),
            )
        elif t == "$_ANDNOT_":
            a_bit = cell.connections.get("A")
            b_bit = cell.connections.get("B")
            if a_bit is None or len(a_bit) != 1:
                raise ExtractionError("$_ANDNOT_ requires single-bit A")
            if b_bit is None or len(b_bit) != 1:
                raise ExtractionError("$_ANDNOT_ requires single-bit B")
            expr = And(a=expr_for_bit(a_bit[0]), b=Not(x=expr_for_bit(b_bit[0])))
        elif t == "$_ORNOT_":
            a_bit = cell.connections.get("A")
            b_bit = cell.connections.get("B")
            if a_bit is None or len(a_bit) != 1:
                raise ExtractionError("$_ORNOT_ requires single-bit A")
            if b_bit is None or len(b_bit) != 1:
                raise ExtractionError("$_ORNOT_ requires single-bit B")
            expr = Or(a=expr_for_bit(a_bit[0]), b=Not(x=expr_for_bit(b_bit[0])))
        elif t == "$mem_v2":
            if out_port != "RD_DATA":
                raise ExtractionError("mem_v2 supports only RD_DATA output")

            width = _parse_param_int(cell.parameters.get("WIDTH", "0"))
            abits = _parse_param_int(cell.parameters.get("ABITS", "0"))
            size = _parse_param_int(cell.parameters.get("SIZE", "0"))
            rd_ports = _parse_param_int(cell.parameters.get("RD_PORTS", "0"))
            init = cell.parameters.get("INIT")
            if init is None:
                raise ExtractionError("mem_v2 missing INIT parameter")
            if width != 8 or abits != 8 or size != 256 or rd_ports < 1:
                raise ExtractionError("mem_v2 supports only 256x8 ROM with ABITS=8")
            table = decode_mem_init_bits(init, size=size, width=width)
            if len(table) != 256:
                raise ExtractionError("mem_v2 table decode failed")

            rd_addr_bits = cell.connections.get("RD_ADDR")
            rd_en_bits = cell.connections.get("RD_EN")
            if rd_addr_bits is None or rd_en_bits is None:
                raise ExtractionError("mem_v2 missing RD_ADDR/RD_EN")
            if len(rd_addr_bits) != abits * rd_ports:
                raise ExtractionError("mem_v2 RD_ADDR width mismatch")
            if len(rd_en_bits) != rd_ports:
                raise ExtractionError("mem_v2 RD_EN width mismatch")

            port_values: list[Expr] = []
            for p in range(rd_ports):
                addr_bits = rd_addr_bits[p * abits : (p + 1) * abits]
                addr = _bus_from_bits(addr_bits, expr_for_bit, width_of_expr)
                value: Expr = Lut8(x=addr, table=table)
                en = expr_for_bit(rd_en_bits[p])
                if not isinstance(en, BoolConst) or not en.value:
                    value = Mux(cond=en, a=value, b=BitVecConst(width=8, value=0))
                port_values.append(value)

            expr = (
                port_values[0]
                if len(port_values) == 1
                else Concat(parts=list(reversed(port_values)))
            )
        elif t == "$memrd":
            if out_port != "DATA":
                raise ExtractionError("memrd supports only DATA output")
            memid = cell.parameters.get("MEMID")
            if memid is None:
                raise ExtractionError("memrd missing MEMID")
            table = mem_tables.get(memid)
            if table is None:
                raise ExtractionError("memrd missing matching meminit_v2")
            addr = bus("ADDR")
            expr = Lut8(x=addr, table=table)
        elif t == "$lut":
            if out_port not in {"Y", "OUT"}:
                raise ExtractionError("lut supports only Y/OUT output")
            a_bits = cell.connections.get("A")
            if a_bits is None:
                raise ExtractionError("lut missing A port")
            if y_width != 1:
                raise ExtractionError("lut output must be 1 bit")
            lut_param = cell.parameters.get("LUT")
            if lut_param is None:
                raise ExtractionError("lut missing LUT parameter")
            n_inputs = len(a_bits)
            lut_width = _parse_param_int(cell.parameters.get("WIDTH", str(n_inputs)))
            if lut_width != n_inputs:
                raise ExtractionError("lut WIDTH does not match A width")
            if n_inputs < 1 or n_inputs > 3:
                raise ExtractionError("lut supports only 1-3 inputs")
            bits = _parse_lut_bits(lut_param, size=1 << n_inputs)
            inputs = [expr_for_bit(b) for b in a_bits]
            if n_inputs == 1:
                expr = _lut1_expr(inputs[0], bits)
            elif n_inputs == 2:
                expr = _lut2_expr(inputs[0], inputs[1], bits)
            else:
                imm8 = 0
                for i in range(8):
                    if bits[i]:
                        imm8 |= 1 << i
                # Map A[2],A[1],A[0] -> (a,b,c) so imm8 index matches A LSB order.
                expr = TernaryLut(a=inputs[2], b=inputs[1], c=inputs[0], imm8=imm8)
        else:
            raise ExtractionError(f"unsupported cell type: {t}")

        cell_out_cache[key] = expr
        return expr

    output_exprs: dict[str, Expr] = {}
    for port in module.ports.values():
        if port.direction != "output":
            continue
        output_exprs[port.name] = _bus_from_bits(port.bits, expr_for_bit, width_of_expr)

    for cell in module.cells.values():
        if cell.type not in seq_cells:
            continue
        d_bits = cell.connections["D"]
        d_bus = _bus_from_bits(d_bits, expr_for_bit, width_of_expr)
        if cell.type in {"$dffe", "$sdffe", "$_SDFFE_PP0P_", "$_SDFFE_PP0N_"}:
            en_bits = cell.connections.get("EN") or cell.connections.get("E")
            if en_bits is None or len(en_bits) != 1:
                raise ExtractionError("dffe enable must be 1 bit")
            en = expr_for_bit(en_bits[0])
            if cell.type == "$_SDFFE_PP0N_":
                en = Not(en)
            next_expr = Mux(cond=en, a=d_bus, b=Var(name=cell.name))
        else:
            next_expr = d_bus

        if cell.type in {"$sdff", "$sdffe"}:
            rst_bits = cell.connections.get("SRST")
            if rst_bits is None or len(rst_bits) != 1:
                raise ExtractionError("sdff reset must be 1 bit")
            rst = expr_for_bit(rst_bits[0])
            next_state[cell.name] = Mux(
                cond=rst, a=reset_state[cell.name], b=next_expr
            )
        elif cell.type in {"$_SDFF_PP0_", "$_SDFFE_PP0P_", "$_SDFFE_PP0N_"}:
            rst_bits = cell.connections.get("R")
            if rst_bits is None or len(rst_bits) != 1:
                raise ExtractionError("sdff reset must be 1 bit")
            rst = expr_for_bit(rst_bits[0])
            next_state[cell.name] = Mux(
                cond=rst, a=reset_state[cell.name], b=next_expr
            )
        else:
            next_state[cell.name] = next_expr

    return TickIR(
        name=design.top,
        inputs=inputs,
        outputs=outputs,
        state=state,
        reset_state=reset_state,
        next_state=next_state,
        output_exprs=output_exprs,
    )
