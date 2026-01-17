from __future__ import annotations

from dataclasses import dataclass

from stc.interp import infer_type
from stc.io_map import IoMap, default_io_map, validate_io_map
from stc.tick_ir import (
    AShr,
    Add,
    And,
    Bitcast,
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
    SimdConst,
    SimdType,
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
from stc.tick_ir_validate import TickIRValidationError, validate_tick_ir


@dataclass(frozen=True)
class CodegenError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def _c_ident(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    if not s:
        return "_"
    if s[0].isdigit():
        return "_" + s
    return s


def _c_type(t: Type) -> str:
    if isinstance(t, BoolType):
        return "uint8_t"
    if isinstance(t, SimdType):
        raise CodegenError("simd types are not supported in the AVR backend")
    assert isinstance(t, BitVecType)
    if t.width <= 8:
        return "uint8_t"
    if t.width <= 16:
        return "uint16_t"
    if t.width <= 32:
        return "uint32_t"
    raise CodegenError("bitvec width > 32 is not supported in the MVP")


def _mask(width: int) -> str:
    if width == 32:
        return "0xFFFFFFFFu"
    return hex((1 << width) - 1) + "u"


def _emit_expr(expr: Expr, types: dict[str, Type]) -> str:
    if isinstance(expr, BoolConst):
        return "1u" if expr.value else "0u"
    if isinstance(expr, BitVecConst):
        return f"({expr.value}u)"
    if isinstance(expr, SimdConst):
        raise CodegenError("simd expressions are not supported in the AVR backend")
    if isinstance(expr, Var):
        return _c_ident(expr.name)
    if isinstance(expr, Bitcast):
        return _emit_expr(expr.x, types)

    if isinstance(expr, Not):
        t = infer_type(expr.x, types)
        x = _emit_expr(expr.x, types)
        if isinstance(t, BoolType):
            return f"(({x}) ^ 1u)"
        assert isinstance(t, BitVecType)
        return f"((~({x})) & {_mask(t.width)})"

    if isinstance(
        expr, (And, Or, Xor, Add, Sub, Shl, LShr, AShr, Eq, Ult, Ule, Ugt, Uge)
    ):
        a = _emit_expr(expr.a, types)
        b = _emit_expr(expr.b, types)
        t = infer_type(expr.a, types)

        if isinstance(expr, (Eq, Ult, Ule, Ugt, Uge)):
            if isinstance(t, BoolType):
                if not isinstance(expr, Eq):
                    raise CodegenError("unsigned compare requires bitvec operands")
                return f"((({a}) ^ ({b})) ^ 1u)"
            assert isinstance(t, BitVecType)
            m = _mask(t.width)
            if isinstance(expr, Eq):
                op = "=="
            elif isinstance(expr, Ult):
                op = "<"
            elif isinstance(expr, Ule):
                op = "<="
            elif isinstance(expr, Ugt):
                op = ">"
            else:
                op = ">="
            return f"((((({a}) & {m}) {op} ((({b}) & {m}))) ? 1u : 0u))"

        if isinstance(t, BoolType):
            if isinstance(expr, And):
                return f"(({a}) & ({b}))"
            if isinstance(expr, Or):
                return f"(({a}) | ({b}))"
            if isinstance(expr, Xor):
                return f"(({a}) ^ ({b}))"
            raise CodegenError("add requires bitvec operands")

        assert isinstance(t, BitVecType)
        m = _mask(t.width)
        if isinstance(expr, And):
            return f"((({a}) & ({b})) & {m})"
        if isinstance(expr, Or):
            return f"((({a}) | ({b})) & {m})"
        if isinstance(expr, Xor):
            return f"((({a}) ^ ({b})) & {m})"
        if isinstance(expr, Add):
            return f"((({a}) + ({b})) & {m})"
        if isinstance(expr, Sub):
            return f"((({a}) - ({b})) & {m})"
        if isinstance(expr, Shl):
            return f"((({a}) << ({b})) & {m})"
        if isinstance(expr, LShr):
            return f"((({a}) >> ({b})) & {m})"
        if isinstance(expr, AShr):
            raise CodegenError("ashr is not supported in the AVR backend")
        raise CodegenError("unsupported binary op")

    if isinstance(expr, Mux):
        cond = _emit_expr(expr.cond, types)
        a = _emit_expr(expr.a, types)
        b = _emit_expr(expr.b, types)
        t = infer_type(expr.a, types)
        if isinstance(t, BoolType):
            return f"mux_u1({cond}, {a}, {b})"
        assert isinstance(t, BitVecType)
        return f"(mux_u32({cond}, {a}, {b}) & {_mask(t.width)})"

    if isinstance(expr, Concat):
        parts = []
        for p in expr.parts:
            t = infer_type(p, types)
            w = 1 if isinstance(t, BoolType) else int(t.width)
            parts.append((w, _emit_expr(p, types)))

        acc = "0u"
        for w, v in parts:
            if w == 1:
                acc = f"(({acc} << 1) | (({v}) & 1u))"
            else:
                acc = f"(({acc} << {w}) | (({v}) & {_mask(w)}))"
        return acc

    if isinstance(expr, Slice):
        x = _emit_expr(expr.x, types)
        if expr.width == 1:
            return f"((({x}) >> {expr.offset}) & 1u)"
        return f"((({x}) >> {expr.offset}) & {_mask(expr.width)})"

    raise CodegenError("unsupported expression kind")


def emit_avr_c(ir: TickIR, *, io_map: IoMap | None = None) -> str:
    validate_tick_ir(ir)
    types: dict[str, Type] = {**ir.inputs, **ir.state}

    input_order = sorted(ir.inputs.keys())
    output_order = sorted(ir.outputs.keys())
    state_order = sorted(ir.state.keys())

    def width_of(name: str, t: Type) -> int:
        if isinstance(t, BoolType):
            return 1
        if isinstance(t, BitVecType):
            return t.width
        raise CodegenError(f"unsupported type for {name}")

    lines: list[str] = []
    lines.append("#include <stdint.h>")
    lines.append("#ifdef STC_HOST")
    lines.append("extern uint8_t PINB;")
    lines.append("extern uint8_t PORTB;")
    lines.append("#else")
    lines.append("#include <avr/io.h>")
    lines.append("#endif")
    lines.append("")
    lines.append("static inline uint8_t mux_u1(uint8_t c, uint8_t t, uint8_t f) {")
    lines.append("    uint8_t m = (uint8_t)(0u - (c & 1u));")
    lines.append("    return (uint8_t)((t & m) | (f & (uint8_t)~m));")
    lines.append("}")
    lines.append("")
    lines.append("static inline uint32_t mux_u32(uint32_t c, uint32_t t, uint32_t f) {")
    lines.append("    uint32_t m = 0u - (c & 1u);")
    lines.append("    return (t & m) | (f & ~m);")
    lines.append("}")
    lines.append("")

    for name in state_order:
        t = ir.state[name]
        lines.append(f"static {_c_type(t)} st_{_c_ident(name)};")

    if state_order:
        lines.append("")

    lines.append("void stc_reset(void) {")
    if not state_order:
        lines.append("}")
    else:
        for name in state_order:
            t = ir.state[name]
            expr = ir.reset_state[name]
            lines.append(f"    st_{_c_ident(name)} = ({_emit_expr(expr, types)});")
            if isinstance(t, BitVecType):
                lines.append(f"    st_{_c_ident(name)} &= {_mask(t.width)};")
        lines.append("}")

    lines.append("")
    lines.append("void stc_tick(void) {")

    if io_map is None:
        io_map = default_io_map(ir)
    try:
        validate_io_map(ir, io_map)
    except ValueError as e:
        raise CodegenError(str(e)) from e

    for name in input_order:
        t = ir.inputs[name]
        w = width_of(name, t)
        pos = int(io_map.inputs[name]["lsb"])
        cname = _c_ident(name)
        if w == 1:
            lines.append(f"    {_c_type(t)} {cname} = (uint8_t)((PINB >> {pos}) & 1u);")
        else:
            lines.append(
                f"    {_c_type(t)} {cname} = ({_c_type(t)})((PINB >> {pos}) & {_mask(w)});"
            )

    for name in state_order:
        t = ir.state[name]
        cname = _c_ident(name)
        lines.append(f"    {_c_type(t)} {cname} = st_{cname};")

    if input_order or state_order:
        lines.append("")

    for name in output_order:
        expr = ir.output_exprs[name]
        t = ir.outputs[name]
        cname = _c_ident(name)
        lines.append(f"    {_c_type(t)} out_{cname} = ({_emit_expr(expr, types)});")
        if isinstance(t, BitVecType):
            lines.append(f"    out_{cname} &= {_mask(t.width)};")

    if output_order:
        lines.append("")

    for name in state_order:
        expr = ir.next_state[name]
        t = ir.state[name]
        cname = _c_ident(name)
        lines.append(f"    {_c_type(t)} nx_{cname} = ({_emit_expr(expr, types)});")
        if isinstance(t, BitVecType):
            lines.append(f"    nx_{cname} &= {_mask(t.width)};")

    if state_order:
        lines.append("")

    for name in state_order:
        cname = _c_ident(name)
        lines.append(f"    st_{cname} = nx_{cname};")

    if output_order:
        lines.append("")
        lines.append("    uint8_t portb = PORTB;")
        for name in output_order:
            t = ir.outputs[name]
            w = width_of(name, t)
            pos = int(io_map.outputs[name]["lsb"])
            cname = _c_ident(name)
            mask = _mask(w)
            lines.append(
                f"    portb = (uint8_t)((portb & (uint8_t)~({mask} << {pos})) | ((out_{cname} & {mask}) << {pos}));"
            )
        lines.append("    PORTB = portb;")

    lines.append("}")
    lines.append("")
    return "\n".join(lines)
