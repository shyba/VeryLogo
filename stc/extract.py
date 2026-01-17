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
            if port.name in {"clk", "rst"}:
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

    for cell in module.cells.values():
        if cell.type not in {"$dff", "$dffe", "$sdff", "$sdffe"}:
            continue
        q_bits = cell.connections["Q"]
        d_bits = cell.connections["D"]
        if len(q_bits) != len(d_bits):
            raise ExtractionError("dff D and Q widths must match")
        t = _port_type(q_bits)
        state[cell.name] = t
        if isinstance(t, BoolType):
            if cell.type in {"$sdff", "$sdffe"}:
                v = _parse_param_int(cell.parameters.get("SRST_VALUE", "0"))
                reset_state[cell.name] = BoolConst(value=bool(v & 1))
            else:
                reset_state[cell.name] = BoolConst(value=False)
            state_bits[q_bits[0]] = Var(name=cell.name)
        else:
            assert isinstance(t, BitVecType)
            if cell.type in {"$sdff", "$sdffe"}:
                v = _parse_param_int(cell.parameters.get("SRST_VALUE", "0"))
                reset_state[cell.name] = BitVecConst(width=t.width, value=v)
            else:
                reset_state[cell.name] = BitVecConst(width=t.width, value=0)
            for i, bit in enumerate(q_bits):
                state_bits[bit] = Slice(x=Var(name=cell.name), offset=i, width=1)

    bit_drivers: dict[int, tuple[str, str, int]] = {}
    for cell_name, cell in module.cells.items():
        if cell.type in {"$dff", "$dffe", "$sdff", "$sdffe"}:
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
            expr = And(a=bus("A"), b=bus("B"))
        elif t == "$or":
            expr = Or(a=bus("A"), b=bus("B"))
        elif t == "$xor":
            expr = Xor(a=bus("A"), b=bus("B"))
        elif t == "$add":
            expr = Add(a=bus("A"), b=bus("B"))
        elif t == "$sub":
            expr = Sub(a=bus("A"), b=bus("B"))
        elif t == "$shl":
            expr = Shl(a=bus("A"), b=bus("B"))
        elif t == "$shr":
            expr = LShr(a=bus("A"), b=bus("B"))
        elif t == "$sshr":
            expr = AShr(a=bus("A"), b=bus("B"))
        elif t == "$eq":
            expr = Eq(a=bus("A"), b=bus("B"))
        elif t == "$ne":
            expr = Not(x=Eq(a=bus("A"), b=bus("B")))
        elif t == "$lt":
            expr = Ult(a=bus("A"), b=bus("B"))
        elif t == "$le":
            expr = Ule(a=bus("A"), b=bus("B"))
        elif t == "$gt":
            expr = Ugt(a=bus("A"), b=bus("B"))
        elif t == "$ge":
            expr = Uge(a=bus("A"), b=bus("B"))
        elif t == "$mux":
            s_bits = cell.connections.get("S")
            if s_bits is None or len(s_bits) != 1:
                raise ExtractionError("mux select must be 1 bit")
            sel = expr_for_bit(s_bits[0])
            a = bus("A")
            b = bus("B")
            expr = Mux(cond=sel, a=b, b=a)
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
        if cell.type not in {"$dff", "$dffe", "$sdff", "$sdffe"}:
            continue
        d_bits = cell.connections["D"]
        d_bus = _bus_from_bits(d_bits, expr_for_bit, width_of_expr)
        if cell.type in {"$dffe", "$sdffe"}:
            en_bits = cell.connections.get("EN")
            if en_bits is None or len(en_bits) != 1:
                raise ExtractionError("dffe enable must be 1 bit")
            en = expr_for_bit(en_bits[0])
            next_state[cell.name] = Mux(cond=en, a=d_bus, b=Var(name=cell.name))
        else:
            next_state[cell.name] = d_bus

    return TickIR(
        name=design.top,
        inputs=inputs,
        outputs=outputs,
        state=state,
        reset_state=reset_state,
        next_state=next_state,
        output_exprs=output_exprs,
    )
