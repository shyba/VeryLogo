from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from stc.interp import _INFER_TYPE_CACHE, infer_type
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
    Lut8,
    Mux,
    Not,
    Or,
    Shl,
    Slice,
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
from stc.tick_ir_validate import iter_vars, validate_tick_ir

FUTHARK_MODE_AUTO = "auto"
FUTHARK_MODE_COMBINATIONAL_FAST = "combinational_fast"
FUTHARK_MODE_STEP_LEGACY = "step_legacy"
FUTHARK_MODES = {
    FUTHARK_MODE_AUTO,
    FUTHARK_MODE_COMBINATIONAL_FAST,
    FUTHARK_MODE_STEP_LEGACY,
}


def resolve_futhark_mode(
    ir: TickIR, requested_mode: str = FUTHARK_MODE_AUTO
) -> tuple[str, str | None]:
    if requested_mode not in FUTHARK_MODES:
        raise CodegenError(
            f"unsupported futhark mode: {requested_mode} "
            f"(expected one of {sorted(FUTHARK_MODES)})"
        )

    if requested_mode == FUTHARK_MODE_STEP_LEGACY:
        return FUTHARK_MODE_STEP_LEGACY, None
    if requested_mode == FUTHARK_MODE_COMBINATIONAL_FAST:
        return FUTHARK_MODE_COMBINATIONAL_FAST, None

    # Auto policy: use combinational_fast for acyclic datapaths and step_legacy
    # for explicit sequential feedback through state.
    state_names = set(ir.state.keys())
    has_state_feedback = False
    for expr in ir.next_state.values():
        if any(name in state_names for name in iter_vars(expr)):
            has_state_feedback = True
            break
    if has_state_feedback:
        return (
            FUTHARK_MODE_STEP_LEGACY,
            "auto_fallback_sequential_feedback_in_next_state",
        )
    return FUTHARK_MODE_COMBINATIONAL_FAST, None


def compute_futhark_source_metrics(source: str) -> dict[str, int]:
    def _count(pattern: str) -> int:
        return len(re.findall(pattern, source))

    return {
        "lines": source.count("\n") + (0 if not source else 1),
        "bytes_utf8": len(source.encode("utf-8")),
        "map": _count(r"\bmap\b"),
        "map2": _count(r"\bmap2\b"),
        "map3": _count(r"\bmap3\b"),
        "reduce": _count(r"\breduce\b"),
        "scan": _count(r"\bscan\b"),
        "bits_to_u64_dyn": _count(r"\bbits_to_u64_dyn\b"),
        "pack_bits_helpers": _count(r"\bpack_bits_w\d+\b"),
        "unpack_bits_helpers": _count(r"\bunpack_bits_w\d+\b"),
        "concat_ops": source.count("++"),
    }


def _flatten_associative_expr(expr: Expr, op_cls: type[Expr]) -> list[Expr]:
    parts: list[Expr] = []
    stack: list[Expr] = [expr]
    while stack:
        cur = stack.pop()
        if isinstance(cur, op_cls):
            stack.append(cur.b)  # type: ignore[attr-defined]
            stack.append(cur.a)  # type: ignore[attr-defined]
        else:
            parts.append(cur)
    return parts


@dataclass(frozen=True)
class CodegenError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class _ValueRef:
    name: str
    kind: str  # "bool", "bits_u64", "bits_arr", "bits_chunks"
    width: int | None = None


def _f_ident(name: str) -> str:
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


def _u64(value: int) -> str:
    return f"0x{int(value) & 0xFFFFFFFFFFFFFFFF:x}u64"


def _mask(width: int) -> int:
    if width < 1:
        raise CodegenError(f"unsupported bitvec width for futhark backend: {width}")
    return (1 << width) - 1


def _mask_u64(width: int) -> int:
    if width < 1 or width > 64:
        raise CodegenError(f"unsupported bitvec width for futhark backend: {width}")
    return _mask(width) & 0xFFFFFFFFFFFFFFFF


def _chunks_for_width(width: int) -> int:
    if width < 1:
        raise CodegenError(f"unsupported bitvec width for futhark backend: {width}")
    return (width + 63) // 64


def _high_chunk_mask(width: int) -> int:
    rem = width % 64
    if rem == 0:
        return 0xFFFFFFFFFFFFFFFF
    return (1 << rem) - 1


def _bitvec_chunks(width: int, value: int) -> list[int]:
    value &= _mask(width)
    chunks = _chunks_for_width(width)
    return [((value >> (64 * i)) & 0xFFFFFFFFFFFFFFFF) for i in range(chunks)]


def _bitvec_chunks_literal(width: int, value: int) -> str:
    chunks = _bitvec_chunks(width, value)
    return "[" + ", ".join(_u64(v) for v in chunks) + "]"


def _array_type(t: Type, size_name: str) -> str:
    if isinstance(t, BoolType):
        return f"[{size_name}]bool"
    if isinstance(t, BitVecType):
        if t.width <= 64:
            return f"[{size_name}]u64"
        return f"[{size_name}][{_chunks_for_width(t.width)}]u64"
    raise CodegenError(f"futhark backend supports only bool/bitvec types (got {t})")


def _array2_type(t: Type, outer_size: str, inner_size: str) -> str:
    if isinstance(t, BoolType):
        return f"[{outer_size}][{inner_size}]bool"
    if isinstance(t, BitVecType):
        if t.width <= 64:
            return f"[{outer_size}][{inner_size}]u64"
        return f"[{outer_size}][{inner_size}][{_chunks_for_width(t.width)}]u64"
    raise CodegenError(f"futhark backend supports only bool/bitvec types (got {t})")


def _tuple_type(types: list[str]) -> str:
    if not types:
        return "()"
    if len(types) == 1:
        return types[0]
    return "(" + ", ".join(types) + ")"


def _tuple_expr(items: list[str]) -> str:
    if not items:
        return "()"
    if len(items) == 1:
        return items[0]
    return "(" + ", ".join(items) + ")"


def _tuple_pattern(items: list[str]) -> str:
    if not items:
        return "()"
    if len(items) == 1:
        return items[0]
    return "(" + ", ".join(items) + ")"


def _const_reset_expr(expr: Expr, t: Type) -> str:
    if isinstance(t, BoolType):
        if not isinstance(expr, BoolConst):
            raise CodegenError(
                "futhark init_state currently supports only constant bool reset values"
            )
        return "true" if expr.value else "false"
    if isinstance(t, BitVecType):
        if not isinstance(expr, BitVecConst):
            raise CodegenError(
                "futhark init_state currently supports only constant bitvec reset values"
            )
        if t.width <= 64:
            return _u64(expr.value & _mask_u64(t.width))
        return _bitvec_chunks_literal(t.width, expr.value)
    raise CodegenError(f"unsupported state type in init_state: {t}")


class _StepExprEmitter:
    def __init__(
        self,
        *,
        n_name: str,
        ctx_types: dict[str, Type],
        var_refs: dict[str, _ValueRef],
    ) -> None:
        self.n_name = n_name
        self.ctx_types = ctx_types
        self.var_refs = var_refs
        self.lines: list[str] = []
        self._memo: dict[Expr, _ValueRef] = {}
        self._tmp_index = 0
        self._needed_small_shift_helpers: set[int] = set()
        self._needed_u64_to_bits: set[int] = set()
        self._needed_bits_to_u64: set[int] = set()
        self._needed_pack_unpack: set[int] = set()
        self._needed_add_bits: set[int] = set()
        self._needed_sub_bits: set[int] = set()
        self._needed_ult_bits: set[int] = set()
        self._needed_big_shift_helpers: set[tuple[str, int]] = set()
        self._lut8_helpers: dict[tuple[int, ...], str] = {}

    @property
    def needed_small_shift_helpers(self) -> set[int]:
        return set(self._needed_small_shift_helpers)

    @property
    def needed_u64_to_bits(self) -> set[int]:
        return set(self._needed_u64_to_bits)

    @property
    def needed_bits_to_u64(self) -> set[int]:
        return set(self._needed_bits_to_u64)

    @property
    def needed_pack_unpack(self) -> set[int]:
        return set(self._needed_pack_unpack)

    @property
    def needed_add_bits(self) -> set[int]:
        return set(self._needed_add_bits)

    @property
    def needed_sub_bits(self) -> set[int]:
        return set(self._needed_sub_bits)

    @property
    def needed_ult_bits(self) -> set[int]:
        return set(self._needed_ult_bits)

    @property
    def needed_big_shift_helpers(self) -> set[tuple[str, int]]:
        return set(self._needed_big_shift_helpers)

    @property
    def lut8_helpers(self) -> dict[str, tuple[int, ...]]:
        out: dict[str, tuple[int, ...]] = {}
        for table, name in self._lut8_helpers.items():
            out[name] = table
        return out

    def _next_tmp(self) -> str:
        name = f"e{self._tmp_index}"
        self._tmp_index += 1
        return name

    def _emit_line(self, text: str) -> None:
        self.lines.append(f"  let {text}")

    def _require_bool(self, value: _ValueRef) -> None:
        if value.kind != "bool":
            raise CodegenError("expected bool expression")

    def _require_bits_u64(self, value: _ValueRef) -> None:
        if value.kind != "bits_u64" or value.width is None:
            raise CodegenError("expected <=64-bit bitvec expression")
        if value.width < 1 or value.width > 64:
            raise CodegenError("invalid <=64-bit bitvec width")

    def _require_bits_arr(self, value: _ValueRef) -> None:
        if value.kind != "bits_arr" or value.width is None:
            raise CodegenError("expected bit-array bitvec expression")
        if value.width < 1:
            raise CodegenError("invalid bit-array bitvec width")

    def _require_bits_chunks(self, value: _ValueRef) -> None:
        if value.kind != "bits_chunks" or value.width is None:
            raise CodegenError("expected >64-bit chunked bitvec expression")
        if value.width <= 64:
            raise CodegenError("invalid >64-bit chunked bitvec width")

    def _chunks_high_mask(self, width: int) -> str:
        return _u64(_high_chunk_mask(width))

    def _emit_map_unary(
        self, value: _ValueRef, *, out_kind: str, out_width: int | None, body: str
    ) -> _ValueRef:
        out_name = self._next_tmp()
        self._emit_line(f"{out_name} = map (\\x -> {body}) {value.name}")
        return _ValueRef(name=out_name, kind=out_kind, width=out_width)

    def _emit_map_binary(
        self,
        left: _ValueRef,
        right: _ValueRef,
        *,
        out_kind: str,
        out_width: int | None,
        body: str,
    ) -> _ValueRef:
        out_name = self._next_tmp()
        self._emit_line(f"{out_name} = map2 (\\x y -> {body}) {left.name} {right.name}")
        return _ValueRef(name=out_name, kind=out_kind, width=out_width)

    def _emit_map_ternary(
        self,
        a: _ValueRef,
        b: _ValueRef,
        c: _ValueRef,
        *,
        out_kind: str,
        out_width: int | None,
        body: str,
    ) -> _ValueRef:
        out_name = self._next_tmp()
        self._emit_line(
            f"{out_name} = map3 (\\a b c -> {body}) {a.name} {b.name} {c.name}"
        )
        return _ValueRef(name=out_name, kind=out_kind, width=out_width)

    def _emit_to_bits_arr(self, value: _ValueRef, width: int) -> _ValueRef:
        if width < 1:
            raise CodegenError(f"invalid bit width conversion target: {width}")
        if value.kind == "bits_arr":
            self._require_bits_arr(value)
            if value.width != width:
                raise CodegenError(
                    f"bitvec width mismatch in conversion ({value.width} != {width})"
                )
            return value
        if value.kind == "bool":
            if width != 1:
                raise CodegenError("cannot widen bool to multi-bit bitvec")
            return self._emit_map_unary(
                value, out_kind="bits_arr", out_width=1, body="[x]"
            )
        if value.kind == "bits_u64":
            self._require_bits_u64(value)
            if value.width != width:
                raise CodegenError(
                    f"bitvec width mismatch in conversion ({value.width} != {width})"
                )
            self._needed_u64_to_bits.add(width)
            return self._emit_map_unary(
                value,
                out_kind="bits_arr",
                out_width=width,
                body=f"u64_to_bits_w{width} x",
            )
        self._require_bits_chunks(value)
        if value.width != width:
            raise CodegenError(
                f"bitvec width mismatch in conversion ({value.width} != {width})"
            )
        self._needed_pack_unpack.add(width)
        return self._emit_map_unary(
            value,
            out_kind="bits_arr",
            out_width=width,
            body=f"unpack_bits_w{width} x",
        )

    def _emit_to_bits_u64(self, value: _ValueRef, width: int) -> _ValueRef:
        if width < 1 or width > 64:
            raise CodegenError(
                f"cannot convert to u64 bitvec representation for width {width}"
            )
        if value.kind == "bits_u64":
            self._require_bits_u64(value)
            if value.width != width:
                raise CodegenError(
                    f"bitvec width mismatch in conversion ({value.width} != {width})"
                )
            return value
        if value.kind == "bool":
            if width != 1:
                raise CodegenError("cannot widen bool to multi-bit bitvec")
            return self._emit_map_unary(
                value,
                out_kind="bits_u64",
                out_width=1,
                body="(if x then 1u64 else 0u64)",
            )
        if value.kind == "bits_arr":
            self._require_bits_arr(value)
            if value.width != width:
                raise CodegenError(
                    f"bitvec width mismatch in conversion ({value.width} != {width})"
                )
            self._needed_bits_to_u64.add(width)
            return self._emit_map_unary(
                value,
                out_kind="bits_u64",
                out_width=width,
                body="bits_to_u64_dyn x",
            )
        self._require_bits_chunks(value)
        if value.width != width:
            raise CodegenError(
                f"bitvec width mismatch in conversion ({value.width} != {width})"
            )
        body = f"slice_u64_chunks x 0u64 {_u64(width)}"
        return self._emit_map_unary(
            value,
            out_kind="bits_u64",
            out_width=width,
            body=body,
        )

    def _emit_to_bits_chunks(self, value: _ValueRef, width: int) -> _ValueRef:
        if width <= 64:
            raise CodegenError(
                f"cannot convert to chunked bitvec representation for width {width}"
            )
        if value.kind == "bits_chunks":
            self._require_bits_chunks(value)
            if value.width != width:
                raise CodegenError(
                    f"bitvec width mismatch in conversion ({value.width} != {width})"
                )
            return value
        if value.kind == "bits_arr":
            self._require_bits_arr(value)
            if value.width != width:
                raise CodegenError(
                    f"bitvec width mismatch in conversion ({value.width} != {width})"
                )
            self._needed_pack_unpack.add(width)
            return self._emit_map_unary(
                value,
                out_kind="bits_chunks",
                out_width=width,
                body=f"pack_bits_w{width} x",
            )
        raise CodegenError(
            "conversion to chunked bitvec requires >64-bit source expression"
        )

    def _emit_lut8_helper(self, table: tuple[int, ...]) -> str:
        existing = self._lut8_helpers.get(table)
        if existing is not None:
            return existing
        name = f"lut8_t{len(self._lut8_helpers)}"
        self._lut8_helpers[table] = name
        return name

    def emit_expr(self, expr: Expr) -> _ValueRef:
        cached = self._memo.get(expr)
        if cached is not None:
            return cached

        if isinstance(expr, Var):
            ref = self.var_refs.get(expr.name)
            if ref is None:
                raise CodegenError(f"unknown variable in futhark backend: {expr.name}")
            self._memo[expr] = ref
            return ref

        if isinstance(expr, BoolConst):
            out_name = self._next_tmp()
            lit = "true" if expr.value else "false"
            self._emit_line(f"{out_name} = replicate {self.n_name} {lit}")
            ref = _ValueRef(name=out_name, kind="bool")
            self._memo[expr] = ref
            return ref

        if isinstance(expr, BitVecConst):
            if expr.width <= 64:
                out_name = self._next_tmp()
                mask = _mask_u64(expr.width)
                value = expr.value & mask
                self._emit_line(f"{out_name} = replicate {self.n_name} {_u64(value)}")
                ref = _ValueRef(name=out_name, kind="bits_u64", width=expr.width)
                self._memo[expr] = ref
                return ref
            chunks_name = self._next_tmp()
            lit_chunks = _bitvec_chunks_literal(expr.width, expr.value)
            self._emit_line(f"{chunks_name} = replicate {self.n_name} {lit_chunks}")
            ref = _ValueRef(name=chunks_name, kind="bits_chunks", width=expr.width)
            self._memo[expr] = ref
            return ref

        expr_type = infer_type(expr, self.ctx_types)
        if isinstance(expr_type, BoolType):
            out_kind = "bool"
            out_width: int | None = None
        elif isinstance(expr_type, BitVecType):
            if expr_type.width <= 64:
                out_kind = "bits_u64"
            else:
                out_kind = "bits_chunks"
            out_width = expr_type.width
        else:
            raise CodegenError(
                f"unsupported expression type in futhark backend: {expr}"
            )

        if isinstance(expr, Bitcast):
            if not isinstance(expr_type, BitVecType):
                raise CodegenError(
                    "futhark backend bitcast currently supports bitvec only"
                )
            inner = self.emit_expr(expr.x)
            expected_kind = "bits_u64" if expr_type.width <= 64 else "bits_chunks"
            if inner.kind != expected_kind or inner.width != expr_type.width:
                raise CodegenError(
                    "futhark backend bitcast does not support width-changing casts"
                )
            self._memo[expr] = inner
            return inner

        if isinstance(expr, Not):
            x = self.emit_expr(expr.x)
            if out_kind == "bool":
                self._require_bool(x)
                ref = self._emit_map_unary(
                    x, out_kind="bool", out_width=None, body="!x"
                )
            elif out_kind == "bits_u64":
                self._require_bits_u64(x)
                assert out_width is not None
                mask = _mask_u64(out_width)
                ref = self._emit_map_unary(
                    x,
                    out_kind="bits_u64",
                    out_width=out_width,
                    body=f"(x ^ {_u64(mask)})",
                )
            else:
                self._require_bits_chunks(x)
                assert out_width is not None
                high_mask = self._chunks_high_mask(out_width)
                ref = self._emit_map_unary(
                    x,
                    out_kind="bits_chunks",
                    out_width=out_width,
                    body=f"not_chunks x {high_mask}",
                )
            self._memo[expr] = ref
            return ref

        if isinstance(
            expr, (And, Or, Xor, Add, Sub, Shl, LShr, AShr, Eq, Ult, Ule, Ugt, Uge)
        ):
            if isinstance(expr, (And, Or, Xor)):
                assoc_parts = _flatten_associative_expr(expr, type(expr))
                if len(assoc_parts) > 2:
                    assoc_refs = [self.emit_expr(part) for part in assoc_parts]
                    if out_kind == "bool" and all(r.kind == "bool" for r in assoc_refs):
                        idx_name = self._next_tmp()
                        self._emit_line(f"{idx_name} = iota {self.n_name}")
                        if isinstance(expr, And):
                            reducer = "reduce (\\p q -> p && q) true"
                        elif isinstance(expr, Or):
                            reducer = "reduce (\\p q -> p || q) false"
                        else:
                            reducer = "reduce (\\p q -> p != q) false"
                        terms = ", ".join(f"{ref.name}[i]" for ref in assoc_refs)
                        out_name = self._next_tmp()
                        self._emit_line(
                            f"{out_name} = map (\\i -> {reducer} [{terms}]) {idx_name}"
                        )
                        ref = _ValueRef(name=out_name, kind="bool")
                        self._memo[expr] = ref
                        return ref

                    if (
                        out_kind == "bits_u64"
                        and out_width is not None
                        and all(
                            r.kind == "bits_u64" and r.width == out_width
                            for r in assoc_refs
                        )
                    ):
                        idx_name = self._next_tmp()
                        self._emit_line(f"{idx_name} = iota {self.n_name}")
                        mask = _mask_u64(out_width)
                        if isinstance(expr, And):
                            reducer = f"reduce (\\p q -> p & q) {_u64(mask)}"
                        elif isinstance(expr, Or):
                            reducer = "reduce (\\p q -> p | q) 0u64"
                        else:
                            reducer = "reduce (\\p q -> p ^ q) 0u64"
                        terms = ", ".join(f"{ref.name}[i]" for ref in assoc_refs)
                        body = f"({reducer} [{terms}] & {_u64(mask)})"
                        out_name = self._next_tmp()
                        self._emit_line(f"{out_name} = map (\\i -> {body}) {idx_name}")
                        ref = _ValueRef(name=out_name, kind="bits_u64", width=out_width)
                        self._memo[expr] = ref
                        return ref

            a = self.emit_expr(expr.a)
            b = self.emit_expr(expr.b)

            if isinstance(expr, (Eq, Ult, Ule, Ugt, Uge)):
                if a.kind == "bool" and b.kind == "bool":
                    if not isinstance(expr, Eq):
                        raise CodegenError(
                            "futhark bool expression supports only eq comparison"
                        )
                    ref = self._emit_map_binary(
                        a, b, out_kind="bool", out_width=None, body="x == y"
                    )
                    self._memo[expr] = ref
                    return ref

                if a.kind == "bits_u64" and b.kind == "bits_u64":
                    self._require_bits_u64(a)
                    self._require_bits_u64(b)
                    if isinstance(expr, Eq):
                        body = "x == y"
                    elif isinstance(expr, Ult):
                        body = "x < y"
                    elif isinstance(expr, Ule):
                        body = "x <= y"
                    elif isinstance(expr, Ugt):
                        body = "x > y"
                    else:
                        body = "x >= y"
                    ref = self._emit_map_binary(
                        a, b, out_kind="bool", out_width=None, body=body
                    )
                    self._memo[expr] = ref
                    return ref

                if (
                    a.width is not None
                    and b.width is not None
                    and a.width > 64
                    and a.width == b.width
                ):
                    if a.kind != "bits_chunks":
                        a = self._emit_to_bits_chunks(a, a.width)
                    if b.kind != "bits_chunks":
                        b = self._emit_to_bits_chunks(b, b.width)

                if a.kind == "bits_chunks" and b.kind == "bits_chunks":
                    self._require_bits_chunks(a)
                    self._require_bits_chunks(b)
                    assert a.width is not None
                    width = a.width
                    if b.width != width:
                        raise CodegenError("comparison requires matching bitvec widths")
                    high_mask = self._chunks_high_mask(width)
                    if isinstance(expr, Eq):
                        body = f"eq_chunks x y {high_mask}"
                    elif isinstance(expr, Ult):
                        body = f"ult_chunks x y {_u64(width)}"
                    elif isinstance(expr, Ule):
                        body = f"(eq_chunks x y {high_mask} || ult_chunks x y {_u64(width)})"
                    elif isinstance(expr, Ugt):
                        body = f"ult_chunks y x {_u64(width)}"
                    else:
                        body = f"(eq_chunks x y {high_mask} || ult_chunks y x {_u64(width)})"
                    ref = self._emit_map_binary(
                        a, b, out_kind="bool", out_width=None, body=body
                    )
                    self._memo[expr] = ref
                    return ref

                if a.kind == "bits_arr" and b.kind == "bits_arr":
                    self._require_bits_arr(a)
                    self._require_bits_arr(b)
                    assert a.width is not None
                    width = a.width
                    if b.width != width:
                        raise CodegenError("comparison requires matching bitvec widths")
                    if isinstance(expr, Eq):
                        body = "x == y"
                    elif isinstance(expr, Ult):
                        self._needed_ult_bits.add(width)
                        body = f"ult_bits_w{width} x y"
                    elif isinstance(expr, Ule):
                        self._needed_ult_bits.add(width)
                        body = f"((x == y) || ult_bits_w{width} x y)"
                    elif isinstance(expr, Ugt):
                        self._needed_ult_bits.add(width)
                        body = f"ult_bits_w{width} y x"
                    else:
                        self._needed_ult_bits.add(width)
                        body = f"((x == y) || ult_bits_w{width} y x)"
                    ref = self._emit_map_binary(
                        a, b, out_kind="bool", out_width=None, body=body
                    )
                    self._memo[expr] = ref
                    return ref

                raise CodegenError("comparison operands must both be bool or bitvec")

            if out_kind == "bool":
                self._require_bool(a)
                self._require_bool(b)
                if isinstance(expr, And):
                    body = "x && y"
                elif isinstance(expr, Or):
                    body = "x || y"
                elif isinstance(expr, Xor):
                    body = "x != y"
                else:
                    raise CodegenError(
                        "futhark bool expression supports only and/or/xor"
                    )
                ref = self._emit_map_binary(
                    a, b, out_kind="bool", out_width=None, body=body
                )
                self._memo[expr] = ref
                return ref

            assert out_width is not None
            if out_kind == "bits_u64":
                self._require_bits_u64(a)
                self._require_bits_u64(b)
                mask = _mask_u64(out_width)
                if isinstance(expr, And):
                    body = f"((x & y) & {_u64(mask)})"
                    ref = self._emit_map_binary(
                        a, b, out_kind="bits_u64", out_width=out_width, body=body
                    )
                elif isinstance(expr, Or):
                    body = f"((x | y) & {_u64(mask)})"
                    ref = self._emit_map_binary(
                        a, b, out_kind="bits_u64", out_width=out_width, body=body
                    )
                elif isinstance(expr, Xor):
                    body = f"((x ^ y) & {_u64(mask)})"
                    ref = self._emit_map_binary(
                        a, b, out_kind="bits_u64", out_width=out_width, body=body
                    )
                elif isinstance(expr, Add):
                    body = f"((x + y) & {_u64(mask)})"
                    ref = self._emit_map_binary(
                        a, b, out_kind="bits_u64", out_width=out_width, body=body
                    )
                elif isinstance(expr, Sub):
                    body = f"((x - y) & {_u64(mask)})"
                    ref = self._emit_map_binary(
                        a, b, out_kind="bits_u64", out_width=out_width, body=body
                    )
                elif isinstance(expr, Shl):
                    self._needed_small_shift_helpers.add(out_width)
                    body = f"shl_w{out_width} x y"
                    ref = self._emit_map_binary(
                        a, b, out_kind="bits_u64", out_width=out_width, body=body
                    )
                elif isinstance(expr, LShr):
                    self._needed_small_shift_helpers.add(out_width)
                    body = f"lshr_w{out_width} x y"
                    ref = self._emit_map_binary(
                        a, b, out_kind="bits_u64", out_width=out_width, body=body
                    )
                elif isinstance(expr, AShr):
                    self._needed_small_shift_helpers.add(out_width)
                    body = f"ashr_w{out_width} x y"
                    ref = self._emit_map_binary(
                        a, b, out_kind="bits_u64", out_width=out_width, body=body
                    )
                else:
                    raise CodegenError(
                        "futhark <=64-bit bitvec expression supports only bitwise/arithmetic/shift"
                    )
                self._memo[expr] = ref
                return ref

            if out_kind == "bits_chunks":
                if a.kind != "bits_chunks":
                    a = self._emit_to_bits_chunks(a, out_width)
                if b.kind != "bits_chunks":
                    b = self._emit_to_bits_chunks(b, out_width)
                self._require_bits_chunks(a)
                self._require_bits_chunks(b)
                high_mask = self._chunks_high_mask(out_width)
                if isinstance(expr, And):
                    body = f"and_chunks x y {high_mask}"
                    ref = self._emit_map_binary(
                        a, b, out_kind="bits_chunks", out_width=out_width, body=body
                    )
                elif isinstance(expr, Or):
                    body = f"or_chunks x y {high_mask}"
                    ref = self._emit_map_binary(
                        a, b, out_kind="bits_chunks", out_width=out_width, body=body
                    )
                elif isinstance(expr, Xor):
                    body = f"xor_chunks x y {high_mask}"
                    ref = self._emit_map_binary(
                        a, b, out_kind="bits_chunks", out_width=out_width, body=body
                    )
                elif isinstance(expr, Add):
                    body = f"add_chunks x y {high_mask}"
                    ref = self._emit_map_binary(
                        a, b, out_kind="bits_chunks", out_width=out_width, body=body
                    )
                elif isinstance(expr, Sub):
                    body = f"sub_chunks x y {high_mask}"
                    ref = self._emit_map_binary(
                        a, b, out_kind="bits_chunks", out_width=out_width, body=body
                    )
                elif isinstance(expr, Shl):
                    body = f"shl_chunks x y {_u64(out_width)} {high_mask}"
                    ref = self._emit_map_binary(
                        a, b, out_kind="bits_chunks", out_width=out_width, body=body
                    )
                elif isinstance(expr, LShr):
                    body = f"lshr_chunks x y {_u64(out_width)} {high_mask}"
                    ref = self._emit_map_binary(
                        a, b, out_kind="bits_chunks", out_width=out_width, body=body
                    )
                elif isinstance(expr, AShr):
                    body = f"ashr_chunks x y {_u64(out_width)} {high_mask}"
                    ref = self._emit_map_binary(
                        a, b, out_kind="bits_chunks", out_width=out_width, body=body
                    )
                else:
                    raise CodegenError(
                        "futhark >64-bit chunked bitvec supports only bitwise/arithmetic/shift"
                    )
                self._memo[expr] = ref
                return ref

            self._require_bits_arr(a)
            self._require_bits_arr(b)
            if isinstance(expr, And):
                body = "map2 (\\p q -> p && q) x y"
                ref = self._emit_map_binary(
                    a, b, out_kind="bits_arr", out_width=out_width, body=body
                )
            elif isinstance(expr, Or):
                body = "map2 (\\p q -> p || q) x y"
                ref = self._emit_map_binary(
                    a, b, out_kind="bits_arr", out_width=out_width, body=body
                )
            elif isinstance(expr, Xor):
                body = "map2 (\\p q -> p != q) x y"
                ref = self._emit_map_binary(
                    a, b, out_kind="bits_arr", out_width=out_width, body=body
                )
            elif isinstance(expr, Add):
                self._needed_add_bits.add(out_width)
                body = f"add_bits_w{out_width} x y"
                ref = self._emit_map_binary(
                    a, b, out_kind="bits_arr", out_width=out_width, body=body
                )
            elif isinstance(expr, Sub):
                self._needed_sub_bits.add(out_width)
                body = f"sub_bits_w{out_width} x y"
                ref = self._emit_map_binary(
                    a, b, out_kind="bits_arr", out_width=out_width, body=body
                )
            elif isinstance(expr, Shl):
                self._needed_big_shift_helpers.add(("shl", out_width))
                self._needed_bits_to_u64.add(min(out_width, 64))
                body = f"shl_bits_w{out_width} x y"
                ref = self._emit_map_binary(
                    a, b, out_kind="bits_arr", out_width=out_width, body=body
                )
            elif isinstance(expr, LShr):
                self._needed_big_shift_helpers.add(("lshr", out_width))
                self._needed_bits_to_u64.add(min(out_width, 64))
                body = f"lshr_bits_w{out_width} x y"
                ref = self._emit_map_binary(
                    a, b, out_kind="bits_arr", out_width=out_width, body=body
                )
            elif isinstance(expr, AShr):
                self._needed_big_shift_helpers.add(("ashr", out_width))
                self._needed_bits_to_u64.add(min(out_width, 64))
                body = f"ashr_bits_w{out_width} x y"
                ref = self._emit_map_binary(
                    a, b, out_kind="bits_arr", out_width=out_width, body=body
                )
            else:
                raise CodegenError(
                    "futhark >64-bit bitvec expression supports only bitwise/arithmetic/shift"
                )
            self._memo[expr] = ref
            return ref

        if isinstance(expr, Mux):
            cond = self.emit_expr(expr.cond)
            self._require_bool(cond)
            a = self.emit_expr(expr.a)
            b = self.emit_expr(expr.b)
            if out_kind == "bool":
                self._require_bool(a)
                self._require_bool(b)
                body = "if a then b else c"
                ref = self._emit_map_ternary(
                    cond, a, b, out_kind="bool", out_width=None, body=body
                )
            elif out_kind == "bits_u64":
                self._require_bits_u64(a)
                self._require_bits_u64(b)
                assert out_width is not None
                mask = _mask_u64(out_width)
                body = f"((if a then b else c) & {_u64(mask)})"
                ref = self._emit_map_ternary(
                    cond, a, b, out_kind="bits_u64", out_width=out_width, body=body
                )
            elif out_kind == "bits_chunks":
                self._require_bits_chunks(a)
                self._require_bits_chunks(b)
                assert out_width is not None
                high_mask = self._chunks_high_mask(out_width)
                body = f"mask_chunks (if a then b else c) {high_mask}"
                ref = self._emit_map_ternary(
                    cond, a, b, out_kind="bits_chunks", out_width=out_width, body=body
                )
            else:
                self._require_bits_arr(a)
                self._require_bits_arr(b)
                assert out_width is not None
                body = "if a then b else c"
                ref = self._emit_map_ternary(
                    cond, a, b, out_kind="bits_arr", out_width=out_width, body=body
                )
            self._memo[expr] = ref
            return ref

        if isinstance(expr, Slice):
            x = self.emit_expr(expr.x)
            if x.kind == "bits_u64":
                self._require_bits_u64(x)
                mask = _mask_u64(expr.width)
                body = f"((x >> {_u64(expr.offset)}) & {_u64(mask)})"
                sliced = self._emit_map_unary(
                    x, out_kind="bits_u64", out_width=expr.width, body=body
                )
                if expr.width == 1:
                    ref = self._emit_map_unary(
                        sliced, out_kind="bool", out_width=None, body="x == 1u64"
                    )
                else:
                    ref = sliced
                self._memo[expr] = ref
                return ref
            if x.kind == "bits_chunks":
                self._require_bits_chunks(x)
                if expr.width == 1:
                    bit_ref = self._emit_map_unary(
                        x,
                        out_kind="bits_u64",
                        out_width=1,
                        body=f"(slice_u64_chunks x {_u64(expr.offset)} 0x1u64)",
                    )
                    ref = self._emit_map_unary(
                        bit_ref, out_kind="bool", out_width=None, body="x == 1u64"
                    )
                    self._memo[expr] = ref
                    return ref
                if expr.width <= 64:
                    ref = self._emit_map_unary(
                        x,
                        out_kind="bits_u64",
                        out_width=expr.width,
                        body=(
                            f"slice_u64_chunks x {_u64(expr.offset)} {_u64(expr.width)}"
                        ),
                    )
                    self._memo[expr] = ref
                    return ref
                high_mask = _u64(_high_chunk_mask(expr.width))
                out_chunks = _chunks_for_width(expr.width)
                ref = self._emit_map_unary(
                    x,
                    out_kind="bits_chunks",
                    out_width=expr.width,
                    body=(
                        f"(slice_chunks x {_u64(expr.offset)} {high_mask} : [{out_chunks}]u64)"
                    ),
                )
                self._memo[expr] = ref
                return ref
            if x.kind != "bits_arr":
                raise CodegenError("slice source must be bitvec")
            self._require_bits_arr(x)
            sliced_arr = self._emit_map_unary(
                x,
                out_kind="bits_arr",
                out_width=expr.width,
                body=f"x[{expr.offset}:{expr.offset + expr.width}]",
            )
            if expr.width == 1:
                ref = self._emit_map_unary(
                    sliced_arr, out_kind="bool", out_width=None, body="x[0]"
                )
            elif expr.width <= 64:
                self._needed_bits_to_u64.add(expr.width)
                ref = self._emit_map_unary(
                    sliced_arr,
                    out_kind="bits_u64",
                    out_width=expr.width,
                    body="bits_to_u64_dyn x",
                )
            else:
                self._needed_pack_unpack.add(expr.width)
                packed = self._emit_map_unary(
                    sliced_arr,
                    out_kind="bits_u64",
                    out_width=expr.width,
                    body=f"pack_bits_w{expr.width} x",
                )
                ref = self._emit_map_unary(
                    packed,
                    out_kind="bits_arr",
                    out_width=expr.width,
                    body=f"unpack_bits_w{expr.width} x",
                )
            self._memo[expr] = ref
            return ref

        if isinstance(expr, Concat):
            if not expr.parts:
                raise CodegenError("concat requires at least one part")

            part_refs = [self.emit_expr(p) for p in expr.parts]
            part_widths: list[int] = []
            for p in expr.parts:
                t = infer_type(p, self.ctx_types)
                if isinstance(t, BoolType):
                    part_widths.append(1)
                elif isinstance(t, BitVecType):
                    part_widths.append(t.width)
                else:
                    raise CodegenError(
                        "futhark concat currently supports bool/bitvec parts only"
                    )
            total_width = sum(part_widths)
            if total_width < 1:
                raise CodegenError("concat total width must be >= 1")

            if total_width <= 64:
                part_words: list[_ValueRef] = []
                for ref, width in zip(part_refs, part_widths):
                    if ref.kind == "bool":
                        if width != 1:
                            raise CodegenError(
                                "bool concat part width must be 1 in futhark backend"
                            )
                        part_words.append(
                            self._emit_map_unary(
                                ref,
                                out_kind="bits_u64",
                                out_width=1,
                                body="(if x then 1u64 else 0u64)",
                            )
                        )
                        continue
                    if ref.kind == "bits_u64":
                        self._require_bits_u64(ref)
                        if ref.width != width:
                            raise CodegenError(
                                "concat part width mismatch in futhark backend"
                            )
                        part_words.append(ref)
                        continue
                    if ref.kind == "bits_arr":
                        self._require_bits_arr(ref)
                        if ref.width != width:
                            raise CodegenError(
                                "concat part width mismatch in futhark backend"
                            )
                        self._needed_bits_to_u64.add(width)
                        part_words.append(
                            self._emit_map_unary(
                                ref,
                                out_kind="bits_u64",
                                out_width=width,
                                body="bits_to_u64_dyn x",
                            )
                        )
                        continue
                    if ref.kind == "bits_chunks":
                        self._require_bits_chunks(ref)
                        if ref.width != width:
                            raise CodegenError(
                                "concat part width mismatch in futhark backend"
                            )
                        part_words.append(
                            self._emit_map_unary(
                                ref,
                                out_kind="bits_u64",
                                out_width=width,
                                body=f"slice_u64_chunks x 0u64 {_u64(width)}",
                            )
                        )
                        continue
                    raise CodegenError(
                        f"unsupported concat part kind in futhark backend: {ref.kind}"
                    )

                acc = part_words[0]
                acc_width = part_widths[0]
                for idx in range(1, len(part_words)):
                    nxt = part_words[idx]
                    nxt_width = part_widths[idx]
                    total_mask = _mask_u64(acc_width + nxt_width)
                    if nxt_width == 64:
                        nxt_term = "y"
                    else:
                        nxt_term = f"(y & {_u64(_mask_u64(nxt_width))})"
                    body = (
                        f"((((x << {_u64(nxt_width)}) | {nxt_term}) & "
                        f"{_u64(total_mask)}))"
                    )
                    acc = self._emit_map_binary(
                        acc,
                        nxt,
                        out_kind="bits_u64",
                        out_width=acc_width + nxt_width,
                        body=body,
                    )
                    acc_width += nxt_width

                if total_width == 1:
                    ref = self._emit_map_unary(
                        acc, out_kind="bool", out_width=None, body="x == 1u64"
                    )
                else:
                    ref = acc
                self._memo[expr] = ref
                return ref

            can_chunk_concat = all(
                width >= 64 and (width % 64) == 0 for width in part_widths
            )
            chunk_parts: list[_ValueRef] = []
            if can_chunk_concat:
                for ref, width in zip(part_refs, part_widths):
                    if ref.kind == "bits_chunks":
                        self._require_bits_chunks(ref)
                        if ref.width != width:
                            can_chunk_concat = False
                            break
                        chunk_parts.append(ref)
                    elif ref.kind == "bits_u64" and width == 64:
                        self._require_bits_u64(ref)
                        chunk_parts.append(
                            self._emit_map_unary(
                                ref,
                                out_kind="bits_chunks",
                                out_width=64,
                                body="[x]",
                            )
                        )
                    else:
                        can_chunk_concat = False
                        break

            if can_chunk_concat:
                idx_name = self._next_tmp()
                self._emit_line(f"{idx_name} = iota {self.n_name}")
                pieces: list[str] = []
                for ref, width in reversed(list(zip(chunk_parts, part_widths))):
                    chunks = _chunks_for_width(width)
                    for ci in range(chunks):
                        pieces.append(f"{ref.name}[i][{ci}]")
                out_name = self._next_tmp()
                self._emit_line(
                    f"{out_name} = map (\\i -> [{', '.join(pieces)}]) {idx_name}"
                )
                ref = _ValueRef(name=out_name, kind="bits_chunks", width=total_width)
                self._memo[expr] = ref
                return ref

            part_bits = [
                self._emit_to_bits_arr(ref, width)
                for ref, width in zip(part_refs, part_widths)
            ]
            acc = part_bits[-1]
            acc_width = part_widths[-1]
            for idx in range(len(part_bits) - 2, -1, -1):
                part = part_bits[idx]
                part_width = part_widths[idx]
                acc = self._emit_map_binary(
                    acc,
                    part,
                    out_kind="bits_arr",
                    out_width=acc_width + part_width,
                    body="x ++ y",
                )
                acc_width += part_width

            if total_width == 1:
                ref = self._emit_map_unary(
                    acc, out_kind="bool", out_width=None, body="x[0]"
                )
            elif total_width <= 64:
                self._needed_bits_to_u64.add(total_width)
                ref = self._emit_map_unary(
                    acc,
                    out_kind="bits_u64",
                    out_width=total_width,
                    body="bits_to_u64_dyn x",
                )
            else:
                self._needed_pack_unpack.add(total_width)
                ref = self._emit_map_unary(
                    acc,
                    out_kind="bits_chunks",
                    out_width=total_width,
                    body=f"pack_bits_w{total_width} x",
                )
            self._memo[expr] = ref
            return ref

        if isinstance(expr, Lut8):
            x = self.emit_expr(expr.x)
            x_u64 = self._emit_to_bits_u64(x, 8)
            table = tuple(int(v) & 0xFF for v in expr.table)
            helper = self._emit_lut8_helper(table)
            ref = self._emit_map_unary(
                x_u64,
                out_kind="bits_u64",
                out_width=8,
                body=f"{helper} x",
            )
            self._memo[expr] = ref
            return ref

        raise CodegenError(
            f"unsupported expression kind in futhark backend: {type(expr).__name__}"
        )


def _emit_small_width_shift_helpers(widths: set[int]) -> list[str]:
    lines: list[str] = []
    for w in sorted(widths):
        if w < 1 or w > 64:
            raise CodegenError(f"unsupported helper width {w}")
        mask = _mask_u64(w)
        lines.append(f"def shl_w{w} (x:u64) (sh:u64) : u64 =")
        lines.append(f"  if sh >= {_u64(w)} then 0u64 else ((x << sh) & {_u64(mask)})")
        lines.append("")
        lines.append(f"def lshr_w{w} (x:u64) (sh:u64) : u64 =")
        lines.append(f"  if sh >= {_u64(w)} then 0u64 else ((x >> sh) & {_u64(mask)})")
        lines.append("")
        lines.append(f"def ashr_w{w} (x:u64) (sh:u64) : u64 =")
        lines.append(f"  let xw = x & {_u64(mask)}")
        lines.append(f"  let sign = ((xw >> {_u64(w - 1)}) & 1u64) == 1u64")
        lines.append(f"  let s = if sh >= {_u64(w)} then {_u64(w)} else sh")
        lines.append(f"  let logical = if s == {_u64(w)} then 0u64 else xw >> s")
        lines.append("  let fill =")
        lines.append("    if sign")
        lines.append(
            f"    then if s == 0u64 then 0u64 else if s == {_u64(w)} then {_u64(mask)} else {_u64(mask)} ^ ((1u64 << ({_u64(w)} - s)) - 1u64)"
        )
        lines.append("    else 0u64")
        lines.append(f"  in (logical | fill) & {_u64(mask)}")
        lines.append("")
    return lines


def _emit_bitpack_helpers(
    *,
    u64_to_bits_widths: set[int],
    bits_to_u64_widths: set[int],
    pack_unpack_widths: set[int],
) -> list[str]:
    lines: list[str] = []
    for w in sorted(u64_to_bits_widths):
        if w < 1 or w > 64:
            raise CodegenError(f"unsupported u64->bits helper width {w}")
        lines.append(f"def u64_to_bits_w{w} (x:u64) : [{w}]bool =")
        lines.append(f"  loop out = replicate {w} false for i < {w}i64 do")
        lines.append("    let bit = ((x >> u64.i64 i) & 1u64) == 1u64")
        lines.append("    in out with [i] = bit")
        lines.append("")
    if bits_to_u64_widths:
        for w in sorted(bits_to_u64_widths):
            if w < 1 or w > 64:
                raise CodegenError(f"unsupported bits->u64 helper width {w}")
        lines.append("def bits_to_u64_dyn [k] (bits:[k]bool) : u64 =")
        lines.append("  loop acc = 0u64 for i < k do")
        lines.append("    let bit = if bits[i] then 1u64 else 0u64")
        lines.append("    in if i >= 64i64 then acc else acc | (bit << u64.i64 i)")
        lines.append("")
    for w in sorted(pack_unpack_widths):
        if w <= 64:
            continue
        chunks = _chunks_for_width(w)
        high_mask = _high_chunk_mask(w)
        lines.append(f"def unpack_bits_w{w} (words:[{chunks}]u64) : [{w}]bool =")
        lines.append(f"  loop out = replicate {w} false for i < {w}i64 do")
        lines.append("    let word = words[i / 64i64]")
        lines.append("    let sh = u64.i64 (i % 64i64)")
        lines.append("    let bit = ((word >> sh) & 1u64) == 1u64")
        lines.append("    in out with [i] = bit")
        lines.append("")
        lines.append(f"def pack_bits_w{w} [k] (bits:[k]bool) : [{chunks}]u64 =")
        lines.append(f"  let words = loop words = replicate {chunks} 0u64 for i < k do")
        lines.append(f"    if i >= {w}i64 then words else")
        lines.append("      let ci = i / 64i64")
        lines.append("      let sh = u64.i64 (i % 64i64)")
        lines.append("      let bit = if bits[i] then 1u64 else 0u64")
        lines.append("      let word = words[ci] | (bit << sh)")
        lines.append("      in words with [ci] = word")
        lines.append(
            f"  in words with [{chunks - 1}] = words[{chunks - 1}] & {_u64(high_mask)}"
        )
        lines.append("")
    return lines


def _emit_chunk_helpers() -> list[str]:
    lines: list[str] = []
    lines.append("def mask_chunks [k] (x:[k]u64) (high_mask:u64) : [k]u64 =")
    lines.append(
        "  map (\\i -> if i == k - 1i64 then x[i] & high_mask else x[i]) (iota k)"
    )
    lines.append("")
    lines.append("def and_chunks [k] (a:[k]u64) (b:[k]u64) (high_mask:u64) : [k]u64 =")
    lines.append("  mask_chunks (map2 (\\x y -> x & y) a b) high_mask")
    lines.append("")
    lines.append("def or_chunks [k] (a:[k]u64) (b:[k]u64) (high_mask:u64) : [k]u64 =")
    lines.append("  mask_chunks (map2 (\\x y -> x | y) a b) high_mask")
    lines.append("")
    lines.append("def xor_chunks [k] (a:[k]u64) (b:[k]u64) (high_mask:u64) : [k]u64 =")
    lines.append("  mask_chunks (map2 (\\x y -> x ^ y) a b) high_mask")
    lines.append("")
    lines.append("def not_chunks [k] (a:[k]u64) (high_mask:u64) : [k]u64 =")
    lines.append("  mask_chunks (map (\\x -> x ^ 0xffffffffffffffffu64) a) high_mask")
    lines.append("")
    lines.append("def add_chunks [k] (a:[k]u64) (b:[k]u64) (high_mask:u64) : [k]u64 =")
    lines.append("  let (out, _carry) =")
    lines.append("    loop (out, carry) = (replicate k 0u64, 0u64) for i < k do")
    lines.append("      let ai = a[i]")
    lines.append("      let bi = b[i]")
    lines.append("      let s0 = ai + bi")
    lines.append("      let c0 = if s0 < ai then 1u64 else 0u64")
    lines.append("      let s1 = s0 + carry")
    lines.append("      let c1 = if carry == 1u64 && s1 == 0u64 then 1u64 else 0u64")
    lines.append("      let carry' = if (c0 | c1) != 0u64 then 1u64 else 0u64")
    lines.append("      in (out with [i] = s1, carry')")
    lines.append("  in mask_chunks out high_mask")
    lines.append("")
    lines.append("def sub_chunks [k] (a:[k]u64) (b:[k]u64) (high_mask:u64) : [k]u64 =")
    lines.append("  let (out, _borrow) =")
    lines.append("    loop (out, borrow) = (replicate k 0u64, 0u64) for i < k do")
    lines.append("      let ai = a[i]")
    lines.append("      let bi = b[i]")
    lines.append("      let d0 = ai - bi")
    lines.append("      let b0 = if ai < bi then 1u64 else 0u64")
    lines.append("      let d1 = d0 - borrow")
    lines.append("      let b1 = if borrow == 1u64 && d0 == 0u64 then 1u64 else 0u64")
    lines.append("      let borrow' = if (b0 | b1) != 0u64 then 1u64 else 0u64")
    lines.append("      in (out with [i] = d1, borrow')")
    lines.append("  in mask_chunks out high_mask")
    lines.append("")
    lines.append("def eq_chunks [k] (a:[k]u64) (b:[k]u64) (_high_mask:u64) : bool =")
    lines.append("  reduce (\\p q -> p && q) true (map2 (\\x y -> x == y) a b)")
    lines.append("")
    lines.append("def ult_chunks [k] (a:[k]u64) (b:[k]u64) (_width:u64) : bool =")
    lines.append("  let (lt, gt) =")
    lines.append("    loop (lt, gt) = (false, false) for i < k do")
    lines.append("      let j = k - 1i64 - i")
    lines.append("      let ai = a[j]")
    lines.append("      let bi = b[j]")
    lines.append("      in if lt || gt then (lt, gt) else (ai < bi, ai > bi)")
    lines.append("  in lt")
    lines.append("")
    lines.append("def shift_amt_chunks [k] (sh:[k]u64) (width:u64) : u64 =")
    lines.append("  let hi_any =")
    lines.append("    loop any = false for i < k do")
    lines.append("      if i == 0i64 then any else any || sh[i] != 0u64")
    lines.append("  let low = sh[0]")
    lines.append("  in if hi_any || low >= width then width else low")
    lines.append("")
    lines.append("def slice_u64_chunks [k] (x:[k]u64) (off:u64) (width:u64) : u64 =")
    lines.append("  let ci = i64.u64 (off / 64u64)")
    lines.append("  let bi = off % 64u64")
    lines.append("  let lo = if ci < k then x[ci] >> bi else 0u64")
    lines.append("  let hi =")
    lines.append("    if bi == 0u64 then 0u64")
    lines.append("    else if ci + 1i64 < k then x[ci + 1i64] << (64u64 - bi)")
    lines.append("    else 0u64")
    lines.append("  let raw = lo | hi")
    lines.append(
        "  let mask = if width >= 64u64 then 0xffffffffffffffffu64 else (1u64 << width) - 1u64"
    )
    lines.append("  in raw & mask")
    lines.append("")
    lines.append(
        "def slice_chunks [ks] [ko] (x:[ks]u64) (off:u64) (high_mask:u64) : [ko]u64 ="
    )
    lines.append("  let out = map (\\oi ->")
    lines.append("    let start = off + (u64.i64 oi) * 64u64")
    lines.append("    let ci = i64.u64 (start / 64u64)")
    lines.append("    let bi = start % 64u64")
    lines.append("    let lo = if ci < ks then x[ci] >> bi else 0u64")
    lines.append("    let hi =")
    lines.append("      if bi == 0u64 then 0u64")
    lines.append("      else if ci + 1i64 < ks then x[ci + 1i64] << (64u64 - bi)")
    lines.append("      else 0u64")
    lines.append("    in lo | hi")
    lines.append("  ) (iota ko)")
    lines.append("  in mask_chunks out high_mask")
    lines.append("")
    lines.append(
        "def shl_chunks [k] (x:[k]u64) (sh:[k]u64) (width:u64) (high_mask:u64) : [k]u64 ="
    )
    lines.append("  let s = shift_amt_chunks sh width")
    lines.append("  let ws = i64.u64 (s / 64u64)")
    lines.append("  let bs = s % 64u64")
    lines.append("  let out = map (\\i ->")
    lines.append("    if i < ws then 0u64 else")
    lines.append("      let src = i - ws")
    lines.append("      let lo = x[src] << bs")
    lines.append(
        "      let hi = if bs == 0u64 || src == 0i64 then 0u64 else x[src - 1i64] >> (64u64 - bs)"
    )
    lines.append("      in lo | hi")
    lines.append("  ) (iota k)")
    lines.append("  in mask_chunks out high_mask")
    lines.append("")
    lines.append(
        "def lshr_chunks [k] (x:[k]u64) (sh:[k]u64) (width:u64) (high_mask:u64) : [k]u64 ="
    )
    lines.append("  let s = shift_amt_chunks sh width")
    lines.append("  let ws = i64.u64 (s / 64u64)")
    lines.append("  let bs = s % 64u64")
    lines.append("  let out = map (\\i ->")
    lines.append("    if i + ws >= k then 0u64 else")
    lines.append("      let src = i + ws")
    lines.append("      let lo = x[src] >> bs")
    lines.append(
        "      let hi = if bs == 0u64 || src + 1i64 >= k then 0u64 else x[src + 1i64] << (64u64 - bs)"
    )
    lines.append("      in lo | hi")
    lines.append("  ) (iota k)")
    lines.append("  in mask_chunks out high_mask")
    lines.append("")
    lines.append(
        "def ashr_chunks [k] (x:[k]u64) (sh:[k]u64) (width:u64) (high_mask:u64) : [k]u64 ="
    )
    lines.append("  let s = shift_amt_chunks sh width")
    lines.append("  let ws = i64.u64 (s / 64u64)")
    lines.append("  let bs = s % 64u64")
    lines.append("  let sign_bit = (width - 1u64) % 64u64")
    lines.append("  let sign = ((x[k - 1i64] >> sign_bit) & 1u64) == 1u64")
    lines.append("  let fill = if sign then 0xffffffffffffffffu64 else 0u64")
    lines.append("  let out = map (\\i ->")
    lines.append("    if i + ws >= k then fill else")
    lines.append("      let src = i + ws")
    lines.append("      let lo = x[src] >> bs")
    lines.append("      let hi =")
    lines.append("        if bs == 0u64 then 0u64")
    lines.append("        else if src + 1i64 < k then x[src + 1i64] << (64u64 - bs)")
    lines.append("        else fill << (64u64 - bs)")
    lines.append("      in lo | hi")
    lines.append("  ) (iota k)")
    lines.append("  in mask_chunks out high_mask")
    lines.append("")
    return lines


def _emit_big_bits_helpers(
    *,
    add_widths: set[int],
    sub_widths: set[int],
    ult_widths: set[int],
) -> list[str]:
    lines: list[str] = []
    if any(w > 64 for w in add_widths):
        lines.append(
            "def gp_compose_add (lhs:(bool,bool)) (rhs:(bool,bool)) : (bool,bool) ="
        )
        lines.append("  let g0 = lhs.0")
        lines.append("  let p0 = lhs.1")
        lines.append("  let g1 = rhs.0")
        lines.append("  let p1 = rhs.1")
        lines.append("  in (g1 || (p1 && g0), p1 && p0)")
        lines.append("")
    if any(w > 64 for w in sub_widths):
        lines.append(
            "def gp_compose_sub (lhs:(bool,bool)) (rhs:(bool,bool)) : (bool,bool) ="
        )
        lines.append("  let g0 = lhs.0")
        lines.append("  let p0 = lhs.1")
        lines.append("  let g1 = rhs.0")
        lines.append("  let p1 = rhs.1")
        lines.append("  in (g1 || (p1 && g0), p1 && p0)")
        lines.append("")
    if any(w > 64 for w in ult_widths):
        lines.append(
            "def cmp_compose (hi:(bool,bool,bool)) (lo:(bool,bool,bool)) : (bool,bool,bool) ="
        )
        lines.append("  let lt_hi = hi.0")
        lines.append("  let gt_hi = hi.1")
        lines.append("  let eq_hi = hi.2")
        lines.append("  let lt_lo = lo.0")
        lines.append("  let gt_lo = lo.1")
        lines.append("  let eq_lo = lo.2")
        lines.append(
            "  in (lt_hi || (eq_hi && lt_lo), gt_hi || (eq_hi && gt_lo), eq_hi && eq_lo)"
        )
        lines.append("")
    for w in sorted(add_widths):
        if w <= 64:
            continue
        lines.append(f"def add_bits_w{w} (a:[{w}]bool) (b:[{w}]bool) : [{w}]bool =")
        lines.append("  let gp = map2 (\\ai bi -> (ai && bi, ai != bi)) a b")
        lines.append("  let pref = scan gp_compose_add (false, true) gp")
        lines.append(
            f"  let carry_in = map (\\i -> if i == 0i64 then false else (pref[i - 1i64]).0) (iota {w})"
        )
        lines.append("  let pbits = map (\\x -> x.1) gp")
        lines.append("  in map2 (\\p c -> p != c) pbits carry_in")
        lines.append("")
    for w in sorted(sub_widths):
        if w <= 64:
            continue
        lines.append(f"def sub_bits_w{w} (a:[{w}]bool) (b:[{w}]bool) : [{w}]bool =")
        lines.append("  let gp = map2 (\\ai bi -> ((!ai) && bi, ai == bi)) a b")
        lines.append("  let pref = scan gp_compose_sub (false, true) gp")
        lines.append(
            f"  let borrow_in = map (\\i -> if i == 0i64 then false else (pref[i - 1i64]).0) (iota {w})"
        )
        lines.append("  in map3 (\\ai bi br -> (ai != bi) != br) a b borrow_in")
        lines.append("")
    for w in sorted(ult_widths):
        if w <= 64:
            continue
        lines.append(f"def ult_bits_w{w} (a:[{w}]bool) (b:[{w}]bool) : bool =")
        lines.append("  let rev_a = reverse a")
        lines.append("  let rev_b = reverse b")
        lines.append(
            "  let cmp = map2 (\\ai bi -> ((!ai) && bi, ai && (!bi), ai == bi)) rev_a rev_b"
        )
        lines.append("  let res = reduce cmp_compose (false, false, true) cmp")
        lines.append("  in res.0")
        lines.append("")
    return lines


def _emit_big_shift_helpers(helpers: set[tuple[str, int]]) -> list[str]:
    lines: list[str] = []
    widths = sorted({width for _, width in helpers if width > 64})
    for w in widths:
        low_w = min(w, 64)
        lines.append(f"def shift_amt_w{w} (sh:[{w}]bool) : u64 =")
        lines.append(f"  let low = bits_to_u64_dyn sh[0:{low_w}]")
        if w > low_w:
            lines.append(
                f"  let hi_any = reduce (\\p q -> p || q) false sh[{low_w}:{w}]"
            )
            lines.append(f"  in if hi_any || low >= {_u64(w)} then {_u64(w)} else low")
        else:
            lines.append(f"  in if low >= {_u64(w)} then {_u64(w)} else low")
        lines.append("")

    for op, w in sorted(helpers):
        if w <= 64:
            continue
        w_lit = _u64(w)
        lines.append(f"def {op}_bits_w{w} (x:[{w}]bool) (sh:[{w}]bool) : [{w}]bool =")
        lines.append(f"  let s = shift_amt_w{w} sh")
        if op == "shl":
            lines.append(f"  in map (\\i ->")
            lines.append("    let ui = u64.i64 i")
            lines.append("    in if ui < s then false else x[i64.u64 (ui - s)]")
            lines.append(f"  ) (iota {w})")
        elif op == "lshr":
            lines.append(f"  in map (\\i ->")
            lines.append("    let ui = u64.i64 i")
            lines.append("    let src = ui + s")
            lines.append(f"    in if src >= {w_lit} then false else x[i64.u64 src]")
            lines.append(f"  ) (iota {w})")
        elif op == "ashr":
            lines.append(f"  let sign = x[{w - 1}]")
            lines.append(f"  in map (\\i ->")
            lines.append("    let ui = u64.i64 i")
            lines.append("    let src = ui + s")
            lines.append(f"    in if src >= {w_lit} then sign else x[i64.u64 src]")
            lines.append(f"  ) (iota {w})")
        else:
            raise CodegenError(f"unsupported >64-bit shift helper op: {op}")
        lines.append("")
    return lines


def _emit_lut8_helpers(helpers: dict[str, tuple[int, ...]]) -> list[str]:
    lines: list[str] = []
    for name in sorted(helpers.keys()):
        table = helpers[name]
        if len(table) != 256:
            raise CodegenError("lut8 helper table must have 256 entries")
        table_lit = "[" + ", ".join(_u64(v & 0xFF) for v in table) + "]"
        lines.append(f"def {name} (x:u64) : u64 =")
        lines.append("  let idx = i64.u64 (x & 0xffu64)")
        lines.append(f"  let tab : [256]u64 = {table_lit}")
        lines.append("  in tab[idx]")
        lines.append("")
    return lines


def emit_futhark(
    ir: TickIR,
    *,
    module_name: str = "circuit",
    mode: str = FUTHARK_MODE_AUTO,
) -> str:
    validate_tick_ir(ir)
    selected_mode, _fallback_reason = resolve_futhark_mode(ir, mode)
    if not module_name:
        raise CodegenError("module_name must be non-empty")
    mod_name = _f_ident(module_name)

    state_names = sorted(ir.state.keys())
    input_names = sorted(ir.inputs.keys())
    output_names = sorted(ir.outputs.keys())

    state_param: dict[str, str] = {name: f"st_{_f_ident(name)}" for name in state_names}
    input_param: dict[str, str] = {name: f"in_{_f_ident(name)}" for name in input_names}
    input_seq_param: dict[str, str] = {
        name: f"in_{_f_ident(name)}_seq" for name in input_names
    }

    ctx_types: dict[str, Type] = {**ir.inputs, **ir.state}
    _INFER_TYPE_CACHE.pop(id(ctx_types), None)
    var_refs: dict[str, _ValueRef] = {}
    step_lines: list[str] = []
    post_eval_state_lines: list[str] = []
    post_eval_output_lines: list[str] = []

    for name in state_names:
        t = ir.state[name]
        param_name = state_param[name]
        if isinstance(t, BoolType):
            var_refs[name] = _ValueRef(name=param_name, kind="bool")
        elif isinstance(t, BitVecType):
            if t.width <= 64:
                if t.width == 64:
                    canonical_name = param_name
                    step_lines.append(f"  let {canonical_name} = {param_name}")
                else:
                    canonical_name = f"{param_name}_m"
                    step_lines.append(
                        f"  let {canonical_name} = map (\\x -> x & {_u64(_mask_u64(t.width))}) {param_name}"
                    )
                var_refs[name] = _ValueRef(
                    name=canonical_name, kind="bits_u64", width=t.width
                )
            else:
                chunks = _chunks_for_width(t.width)
                high_mask = _high_chunk_mask(t.width)
                if high_mask != 0xFFFFFFFFFFFFFFFF:
                    canonical_words = f"{param_name}_m"
                    step_lines.append(
                        f"  let {canonical_words} = map (\\w -> w with [{chunks - 1}] = w[{chunks - 1}] & {_u64(high_mask)}) {param_name}"
                    )
                else:
                    canonical_words = param_name
                    step_lines.append(f"  let {canonical_words} = {param_name}")
                var_refs[name] = _ValueRef(
                    name=canonical_words, kind="bits_chunks", width=t.width
                )
        else:
            raise CodegenError(
                f"futhark backend supports only bool/bitvec types (state {name})"
            )

    for name in input_names:
        t = ir.inputs[name]
        param_name = input_param[name]
        if isinstance(t, BoolType):
            var_refs[name] = _ValueRef(name=param_name, kind="bool")
        elif isinstance(t, BitVecType):
            if t.width <= 64:
                if t.width == 64:
                    canonical_name = param_name
                    step_lines.append(f"  let {canonical_name} = {param_name}")
                else:
                    canonical_name = f"{param_name}_m"
                    step_lines.append(
                        f"  let {canonical_name} = map (\\x -> x & {_u64(_mask_u64(t.width))}) {param_name}"
                    )
                var_refs[name] = _ValueRef(
                    name=canonical_name, kind="bits_u64", width=t.width
                )
            else:
                chunks = _chunks_for_width(t.width)
                high_mask = _high_chunk_mask(t.width)
                if high_mask != 0xFFFFFFFFFFFFFFFF:
                    canonical_words = f"{param_name}_m"
                    step_lines.append(
                        f"  let {canonical_words} = map (\\w -> w with [{chunks - 1}] = w[{chunks - 1}] & {_u64(high_mask)}) {param_name}"
                    )
                else:
                    canonical_words = param_name
                    step_lines.append(f"  let {canonical_words} = {param_name}")
                var_refs[name] = _ValueRef(
                    name=canonical_words, kind="bits_chunks", width=t.width
                )
        else:
            raise CodegenError(
                f"futhark backend supports only bool/bitvec types (input {name})"
            )

    emitter = _StepExprEmitter(n_name="n", ctx_types=ctx_types, var_refs=var_refs)
    needed_pack_unpack_returns: set[int] = set()

    next_state_vars: list[str] = []
    next_state_types: list[str] = []
    for name in state_names:
        expr = ir.next_state.get(name)
        if expr is None:
            expr = Var(name=name)
        out_ref = emitter.emit_expr(expr)
        t = ir.state[name]
        if isinstance(t, BoolType):
            if out_ref.kind != "bool":
                raise CodegenError(f"next_state type mismatch for {name}")
            next_state_vars.append(out_ref.name)
        elif isinstance(t, BitVecType):
            if t.width <= 64:
                if out_ref.kind != "bits_u64" or out_ref.width != t.width:
                    raise CodegenError(f"next_state type mismatch for {name}")
                next_state_vars.append(out_ref.name)
            else:
                if out_ref.kind == "bits_chunks" and out_ref.width == t.width:
                    next_state_vars.append(out_ref.name)
                elif out_ref.kind == "bits_arr" and out_ref.width == t.width:
                    packed_name = f"ret_st_{_f_ident(name)}"
                    post_eval_state_lines.append(
                        f"  let {packed_name} = map pack_bits_w{t.width} {out_ref.name}"
                    )
                    next_state_vars.append(packed_name)
                    needed_pack_unpack_returns.add(t.width)
                else:
                    raise CodegenError(f"next_state type mismatch for {name}")
        next_state_types.append(_array_type(t, "n"))

    state_expr_line_count = len(emitter.lines)

    output_vars: list[str] = []
    output_types: list[str] = []
    for name in output_names:
        expr = ir.output_exprs[name]
        out_ref = emitter.emit_expr(expr)
        t = ir.outputs[name]
        if isinstance(t, BoolType):
            if out_ref.kind != "bool":
                raise CodegenError(f"output type mismatch for {name}")
            output_vars.append(out_ref.name)
        elif isinstance(t, BitVecType):
            if t.width <= 64:
                if out_ref.kind != "bits_u64" or out_ref.width != t.width:
                    raise CodegenError(f"output type mismatch for {name}")
                output_vars.append(out_ref.name)
            else:
                if out_ref.kind == "bits_chunks" and out_ref.width == t.width:
                    output_vars.append(out_ref.name)
                elif out_ref.kind == "bits_arr" and out_ref.width == t.width:
                    packed_name = f"ret_out_{_f_ident(name)}"
                    post_eval_output_lines.append(
                        f"  let {packed_name} = map pack_bits_w{t.width} {out_ref.name}"
                    )
                    output_vars.append(packed_name)
                    needed_pack_unpack_returns.add(t.width)
                else:
                    raise CodegenError(f"output type mismatch for {name}")
        output_types.append(_array_type(t, "n"))

    used_small_shift_helpers = emitter.needed_small_shift_helpers
    used_u64_to_bits = emitter.needed_u64_to_bits
    used_bits_to_u64 = emitter.needed_bits_to_u64
    used_pack_unpack = emitter.needed_pack_unpack | needed_pack_unpack_returns
    used_add_bits = emitter.needed_add_bits
    used_sub_bits = emitter.needed_sub_bits
    used_ult_bits = emitter.needed_ult_bits
    used_big_shift_helpers = emitter.needed_big_shift_helpers
    lut8_helpers = emitter.lut8_helpers

    lines: list[str] = []
    lines.append(f"-- Generated by VeryLogo (module: {mod_name})")
    lines.append(
        f"-- Futhark backend: bool/bitvec SoA batched kernel (mode: {selected_mode})"
    )
    lines.append("")
    lines.extend(_emit_small_width_shift_helpers(used_small_shift_helpers))
    lines.extend(_emit_chunk_helpers())
    lines.extend(
        _emit_bitpack_helpers(
            u64_to_bits_widths=used_u64_to_bits,
            bits_to_u64_widths=used_bits_to_u64,
            pack_unpack_widths=used_pack_unpack,
        )
    )
    lines.extend(
        _emit_big_bits_helpers(
            add_widths=used_add_bits,
            sub_widths=used_sub_bits,
            ult_widths=used_ult_bits,
        )
    )
    lines.extend(_emit_big_shift_helpers(used_big_shift_helpers))
    lines.extend(_emit_lut8_helpers(lut8_helpers))

    init_types = [_array_type(ir.state[name], "n") for name in state_names]
    lines.append(f"entry init_state (n:i64) : {_tuple_type(init_types)} =")
    init_exprs: list[str] = []
    for name in state_names:
        t = ir.state[name]
        reset_expr = ir.reset_state.get(name)
        if reset_expr is None:
            if isinstance(t, BoolType):
                reset_expr = BoolConst(False)
            elif isinstance(t, BitVecType):
                reset_expr = BitVecConst(width=t.width, value=0)
            else:
                raise CodegenError(f"unsupported state type for reset: {t}")
        lit = _const_reset_expr(reset_expr, t)
        init_exprs.append(f"replicate n {lit}")
    lines.append(f"  {_tuple_expr(init_exprs)}")
    lines.append("")

    step_params: list[tuple[str, str]] = [("n", "i64")]
    step_params.extend(
        (state_param[name], _array_type(ir.state[name], "n")) for name in state_names
    )
    step_params.extend(
        (input_param[name], _array_type(ir.inputs[name], "n")) for name in input_names
    )
    step_ret_types = next_state_types + output_types
    step_sig = " ".join(f"({p}:{t})" for p, t in step_params)
    lines.append(f"def step_state_impl {step_sig} : {_tuple_type(next_state_types)} =")
    state_body_lines = (
        step_lines + emitter.lines[:state_expr_line_count] + post_eval_state_lines
    )
    if state_body_lines:
        lines.extend(state_body_lines)
        lines.append(f"  in {_tuple_expr(next_state_vars)}")
    else:
        lines.append(f"  {_tuple_expr(next_state_vars)}")
    lines.append("")

    lines.append(f"def step_impl {step_sig} : {_tuple_type(step_ret_types)} =")
    step_body_lines = (
        step_lines + emitter.lines + post_eval_state_lines + post_eval_output_lines
    )
    if step_body_lines:
        lines.extend(step_body_lines)
        lines.append(f"  in {_tuple_expr(next_state_vars + output_vars)}")
    else:
        lines.append(f"  {_tuple_expr(next_state_vars + output_vars)}")
    lines.append("")

    call_args = (
        ["n"]
        + [state_param[nm] for nm in state_names]
        + [input_param[nm] for nm in input_names]
    )

    if selected_mode == FUTHARK_MODE_COMBINATIONAL_FAST:
        lines.append(f"entry eval_batch {step_sig} : {_tuple_type(step_ret_types)} =")
        lines.append(f"  step_impl {' '.join(call_args)}")
        lines.append("")

        # Compatibility alias for scripts/tests that still call step_batch.
        lines.append(f"entry step_batch {step_sig} : {_tuple_type(step_ret_types)} =")
        lines.append(f"  eval_batch {' '.join(call_args)}")
        lines.append("")

        if output_names:
            xor_acc_params: list[tuple[str, str]] = []
            for idx, out_name in enumerate(output_names):
                xor_acc_params.append((f"acc_{_f_ident(out_name)}", output_types[idx]))
            xor_params = step_params + xor_acc_params
            xor_sig = " ".join(f"({p}:{t})" for p, t in xor_params)
            lines.append(
                f"entry eval_batch_xor {xor_sig} : {_tuple_type(step_ret_types)} ="
            )
            tmp_state_names = [f"tmp_st_{_f_ident(name)}" for name in state_names]
            tmp_out_names = [f"tmp_out_{_f_ident(name)}" for name in output_names]
            lines.append(
                f"  let {_tuple_pattern(tmp_state_names + tmp_out_names)} = step_impl {' '.join(call_args)}"
            )
            xor_out_names: list[str] = []
            for idx, out_name in enumerate(output_names):
                tmp_out = tmp_out_names[idx]
                acc_name = xor_acc_params[idx][0]
                xor_name = f"xor_out_{_f_ident(out_name)}"
                out_t = ir.outputs[out_name]
                if isinstance(out_t, BoolType):
                    lines.append(
                        f"  let {xor_name} = map2 (\\x y -> x != y) {tmp_out} {acc_name}"
                    )
                elif isinstance(out_t, BitVecType):
                    if out_t.width <= 64:
                        lines.append(
                            f"  let {xor_name} = map2 (\\x y -> x ^ y) {tmp_out} {acc_name}"
                        )
                    else:
                        lines.append(
                            f"  let {xor_name} = map2 (\\x y -> map2 (\\p q -> p ^ q) x y) {tmp_out} {acc_name}"
                        )
                else:
                    raise CodegenError(
                        f"unsupported output type in eval_batch_xor: {out_t}"
                    )
                xor_out_names.append(xor_name)
            lines.append(f"  in {_tuple_expr(tmp_state_names + xor_out_names)}")
            lines.append("")
    else:
        lines.append(f"entry step_batch {step_sig} : {_tuple_type(step_ret_types)} =")
        lines.append(f"  step_impl {' '.join(call_args)}")
        lines.append("")

        run_params: list[tuple[str, str]] = [("steps", "i64"), ("n", "i64")]
        run_params.extend(
            (state_param[name], _array_type(ir.state[name], "n"))
            for name in state_names
        )
        run_params.extend(
            (input_seq_param[name], _array2_type(ir.inputs[name], "steps", "n"))
            for name in input_names
        )
        run_sig = " ".join(f"({p}:{t})" for p, t in run_params)
        lines.append(
            f"entry run_steps_batch {run_sig} : {_tuple_type(step_ret_types)} ="
        )

        zero_output_values: list[str] = []
        for name in output_names:
            t = ir.outputs[name]
            if isinstance(t, BoolType):
                zero = "false"
            elif isinstance(t, BitVecType):
                if t.width <= 64:
                    zero = "0u64"
                else:
                    zero = _bitvec_chunks_literal(t.width, 0)
            else:
                raise CodegenError(f"unsupported output type in run_steps: {t}")
            zero_output_values.append(f"replicate n {zero}")

        lines.append(
            f"  if steps == 0i64 then {_tuple_expr([state_param[name] for name in state_names] + zero_output_values)}"
        )
        lines.append("  else")
        lines.append("    let prefix_steps = steps - 1i64")
        loop_state_names = [f"loop_{_f_ident(name)}" for name in state_names]
        lines.append(
            f"    let {_tuple_pattern(loop_state_names)} = loop {_tuple_pattern(loop_state_names)} = {_tuple_expr([state_param[name] for name in state_names])} for i < prefix_steps do"
        )
        loop_call_args = list(loop_state_names)
        for name in input_names:
            seq_name = input_seq_param[name]
            tmp_name = f"{input_param[name]}_loop"
            lines.append(f"      let {tmp_name} = {seq_name}[i]")
            loop_call_args.append(tmp_name)
        state_step_call = "step_state_impl " + " ".join(["n"] + loop_call_args)
        loop_next_state_names = [f"loop_next_{_f_ident(name)}" for name in state_names]
        lines.append(
            f"      let {_tuple_pattern(loop_next_state_names)} = {state_step_call}"
        )
        lines.append(f"      in {_tuple_expr(loop_next_state_names)}")
        lines.append("    let i_last = steps - 1i64")

        final_call_args = list(loop_state_names)
        for name in input_names:
            seq_name = input_seq_param[name]
            tmp_name = f"{input_param[name]}_last"
            lines.append(f"    let {tmp_name} = {seq_name}[i_last]")
            final_call_args.append(tmp_name)
        final_step_call = "step_impl " + " ".join(["n"] + final_call_args)
        final_names = [f"final_{_f_ident(name)}" for name in state_names] + [
            f"final_out_{_f_ident(name)}" for name in output_names
        ]
        lines.append(f"    let {_tuple_pattern(final_names)} = {final_step_call}")
        lines.append("")
        lines.append(f"    in {_tuple_expr(final_names)}")
        lines.append("")

    return "\n".join(lines)


def emit_futhark_manifest(
    ir: TickIR,
    *,
    module_name: str | None = None,
    mode: str = FUTHARK_MODE_AUTO,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    validate_tick_ir(ir)
    selected_mode, auto_fallback_reason = resolve_futhark_mode(ir, mode)
    if fallback_reason is None:
        fallback_reason = auto_fallback_reason
    mod_name = _f_ident(module_name or ir.name or "circuit")
    state_names = sorted(ir.state.keys())
    input_names = sorted(ir.inputs.keys())
    output_names = sorted(ir.outputs.keys())

    def _type_desc(t: Type) -> dict[str, Any]:
        if isinstance(t, BoolType):
            return {"kind": "bool", "width": 1, "repr": "bool"}
        if isinstance(t, BitVecType):
            out = {"kind": "bitvec", "width": t.width}
            if t.width > 64:
                out["chunks"] = _chunks_for_width(t.width)
                out["repr"] = "chunks_u64"
            else:
                out["repr"] = "u64"
            return out
        raise CodegenError(f"unsupported type for futhark manifest: {t}")

    entries = {
        "init_state": "init_state",
        "step_batch": "step_batch",
    }
    if selected_mode == FUTHARK_MODE_COMBINATIONAL_FAST:
        entries["eval_batch"] = "eval_batch"
        if output_names:
            entries["eval_batch_xor"] = "eval_batch_xor"
    else:
        entries["run_steps_batch"] = "run_steps_batch"

    return {
        "backend": "futhark",
        "abi_version": 2,
        "module_name": mod_name,
        "mode": selected_mode,
        "layout": "soa-batch",
        "target_support": ["c", "cuda", "opencl"],
        "entries": entries,
        "fallback_reason": fallback_reason,
        "state": [
            {
                "name": name,
                "param": f"st_{_f_ident(name)}",
                **_type_desc(ir.state[name]),
            }
            for name in state_names
        ],
        "inputs": [
            {
                "name": name,
                "param": f"in_{_f_ident(name)}",
                "seq_param": f"in_{_f_ident(name)}_seq",
                **_type_desc(ir.inputs[name]),
            }
            for name in input_names
        ],
        "outputs": [
            {
                "name": name,
                "param": f"out_{_f_ident(name)}",
                "acc_param": f"acc_{_f_ident(name)}",
                **_type_desc(ir.outputs[name]),
            }
            for name in output_names
        ],
    }
