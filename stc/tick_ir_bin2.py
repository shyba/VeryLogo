from __future__ import annotations

from dataclasses import fields
from typing import Any

from stc.tick_ir import (
    AShr,
    Add,
    And,
    Bitcast,
    BitTranspose,
    BitVecConst,
    BitVecType,
    BoolConst,
    BoolType,
    Concat,
    Delay,
    Div,
    Eq,
    Expr,
    FAbs,
    FAdd,
    FDiv,
    FEq,
    FLe,
    FLt,
    FMul,
    FNe,
    FNeg,
    FSqrt,
    FSub,
    FloatConst,
    FloatType,
    LShr,
    Lut8,
    Mul,
    Mux,
    Not,
    Or,
    Rotl,
    Rotr,
    Shl,
    SimdAShr,
    SimdAdd,
    SimdAddMasked,
    SimdAddSatS,
    SimdAddSatU,
    SimdAnd,
    SimdBlend,
    SimdConst,
    SimdEq,
    SimdExtractLane,
    SimdFAbs,
    SimdFAdd,
    SimdFCmpEq,
    SimdFCmpLe,
    SimdFCmpLt,
    SimdFCmpNe,
    SimdFDiv,
    SimdFFma,
    SimdFNeg,
    SimdFSqrt,
    SimdFSub,
    SimdFMul,
    SimdInsertLane,
    SimdLShr,
    SimdMaddS16,
    SimdMaskExpand,
    SimdMaskPack,
    SimdMaxS,
    SimdMaxU,
    SimdMinS,
    SimdMinU,
    SimdMulHiS,
    SimdMulHiU,
    SimdMulLo,
    SimdNot,
    SimdOr,
    SimdPackSS16To8,
    SimdPackSS32To16,
    SimdPackUS16To8,
    SimdSExtLo,
    SimdSge,
    SimdSgt,
    SimdShl,
    SimdShuffle,
    SimdSle,
    SimdSlt,
    SimdSplat,
    SimdSub,
    SimdSubMasked,
    SimdSubSatS,
    SimdSubSatU,
    SimdType,
    SimdUge,
    SimdUgt,
    SimdUle,
    SimdUlt,
    SimdUnpackHi,
    SimdUnpackLo,
    SimdXor,
    SimdZExtLo,
    Slice,
    Sub,
    TernaryLut,
    TickIR,
    Type,
    Uge,
    Ugt,
    Ule,
    Ult,
    Var,
    Xor,
    EXPR_CLASSES,
)
import os
import time


class BinWriter:
    def __init__(self) -> None:
        self.buf = bytearray()

    def write_u8(self, v: int) -> None:
        self.buf.append(v & 0xFF)

    def write_u32(self, v: int) -> None:
        self.buf.extend(int(v).to_bytes(4, "little", signed=False))

    def write_u64(self, v: int) -> None:
        self.buf.extend(int(v).to_bytes(8, "little", signed=False))

    def write_bytes(self, b: bytes) -> None:
        self.write_u32(len(b))
        self.buf.extend(b)

    def write_big_uint(self, v: int) -> None:
        if v < 0:
            raise ValueError("negative values not supported")
        if v == 0:
            self.write_u32(0)
            return
        b = int(v).to_bytes((int(v).bit_length() + 7) // 8, "little", signed=False)
        self.write_u32(len(b))
        self.buf.extend(b)


class BinReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.off = 0

    def read_u8(self) -> int:
        v = self.data[self.off]
        self.off += 1
        return v

    def read_u32(self) -> int:
        v = int.from_bytes(self.data[self.off : self.off + 4], "little", signed=False)
        self.off += 4
        return v

    def read_u64(self) -> int:
        v = int.from_bytes(self.data[self.off : self.off + 8], "little", signed=False)
        self.off += 8
        return v

    def read_bytes(self) -> bytes:
        n = self.read_u32()
        b = self.data[self.off : self.off + n]
        self.off += n
        return b

    def read_big_uint(self) -> int:
        n = self.read_u32()
        if n == 0:
            return 0
        b = self.data[self.off : self.off + n]
        self.off += n
        return int.from_bytes(b, "little", signed=False)


_OPCODES: dict[type[Expr], int] = {cls: i for i, cls in enumerate(EXPR_CLASSES)}


def _type_tag(t: Type) -> int:
    if isinstance(t, BoolType):
        return 0
    if isinstance(t, BitVecType):
        return 1
    if isinstance(t, FloatType):
        return 2
    if isinstance(t, SimdType):
        return 3
    raise TypeError("unknown type")


def _write_type(w: BinWriter, t: Type) -> None:
    tag = _type_tag(t)
    w.write_u8(tag)
    if tag == 1:
        w.write_u32(t.width)
    elif tag == 2:
        w.write_u32(t.width)
    elif tag == 3:
        w.write_u32(t.lane_width)
        w.write_u32(t.lanes)


def _read_type(r: BinReader) -> Type:
    tag = r.read_u8()
    if tag == 0:
        return BoolType()
    if tag == 1:
        return BitVecType(width=r.read_u32())
    if tag == 2:
        return FloatType(width=r.read_u32())
    if tag == 3:
        return SimdType(lane_width=r.read_u32(), lanes=r.read_u32())
    raise ValueError("bad type tag")


def _assign_nodes_and_strings(
    ir: TickIR,
) -> tuple[list[Expr], dict[int, int], list[str]]:
    nodes: list[Expr] = []
    id_map: dict[int, int] = {}
    strings: set[str] = set()
    for m in (ir.inputs, ir.outputs, ir.state):
        strings.update(m.keys())
    strings.update(ir.reset_state.keys())
    strings.update(ir.next_state.keys())
    strings.update(ir.output_exprs.keys())

    def visit(e: Expr) -> int:
        eid = id(e)
        if eid in id_map:
            return id_map[eid]
        if isinstance(e, Var):
            strings.add(e.name)
        for f in fields(type(e)):
            v = getattr(e, f.name)
            if isinstance(v, EXPR_CLASSES):
                visit(v)
            elif isinstance(v, (list, tuple)):
                for it in v:
                    if isinstance(it, EXPR_CLASSES):
                        visit(it)
        for v in vars(e).values():
            if isinstance(v, EXPR_CLASSES):
                visit(v)
            elif isinstance(v, (list, tuple)):
                for it in v:
                    if isinstance(it, EXPR_CLASSES):
                        visit(it)
        node_id = len(nodes)
        nodes.append(e)
        id_map[eid] = node_id
        return node_id

    for m in (ir.reset_state, ir.next_state, ir.output_exprs):
        for e in m.values():
            visit(e)
    return nodes, id_map, sorted(strings)


def write_tick_ir_bin(ir: TickIR, path: str) -> None:
    timing_enabled = os.environ.get("STC_TIMING", "0").lower() not in {
        "0",
        "false",
        "no",
    }

    def _log(label: str, start: float) -> None:
        if timing_enabled:
            elapsed = time.perf_counter() - start
            print(f"[timing] tick_ir_bin:{label}: {elapsed:.3f}s", flush=True)

    t0 = time.perf_counter()
    w = BinWriter()
    w.write_u8(1)  # version
    _log("header", t0)
    t0 = time.perf_counter()
    nodes, id_map, strings = _assign_nodes_and_strings(ir)
    _log("assign_nodes_strings", t0)
    string_index = {s: i for i, s in enumerate(strings)}
    t0 = time.perf_counter()
    w.write_u32(len(strings))
    for s in strings:
        w.write_bytes(s.encode("utf-8"))
    _log("write_strings", t0)
    t0 = time.perf_counter()
    w.write_u32(len(nodes))
    for node in nodes:
        opcode = _OPCODES[type(node)]
        w.write_u8(opcode)
        _write_node_payload(w, node, id_map, string_index)
    _log("write_nodes", t0)
    t0 = time.perf_counter()
    _write_type_map(w, ir.inputs, string_index)
    _write_type_map(w, ir.outputs, string_index)
    _write_type_map(w, ir.state, string_index)
    _write_expr_map(w, ir.reset_state, id_map, string_index)
    _write_expr_map(w, ir.next_state, id_map, string_index)
    _write_expr_map(w, ir.output_exprs, id_map, string_index)
    _log("write_maps", t0)
    with open(path, "wb") as f:
        f.write(w.buf)


def read_tick_ir_bin(path: str) -> TickIR:
    timing_enabled = os.environ.get("STC_TIMING", "0").lower() not in {
        "0",
        "false",
        "no",
    }

    def _log(label: str, start: float) -> None:
        if timing_enabled:
            elapsed = time.perf_counter() - start
            print(f"[timing] tick_ir_bin_read:{label}: {elapsed:.3f}s", flush=True)

    t0 = time.perf_counter()
    data = open(path, "rb").read()
    _log("read_file", t0)
    r = BinReader(data)
    t0 = time.perf_counter()
    version = r.read_u8()
    if version != 1:
        raise ValueError("bad tick ir bin version")
    _log("version", t0)
    t0 = time.perf_counter()
    strings = [r.read_bytes().decode("utf-8") for _ in range(r.read_u32())]
    _log("strings", t0)
    t0 = time.perf_counter()
    nodes = _read_nodes(r, strings)
    _log("nodes", t0)
    t0 = time.perf_counter()
    inputs = _read_type_map(r, strings)
    outputs = _read_type_map(r, strings)
    state = _read_type_map(r, strings)
    reset_state = _read_expr_map(r, nodes, strings)
    next_state = _read_expr_map(r, nodes, strings)
    output_exprs = _read_expr_map(r, nodes, strings)
    _log("maps", t0)
    return TickIR(
        name="tickir",
        inputs=inputs,
        outputs=outputs,
        state=state,
        reset_state=reset_state,
        next_state=next_state,
        output_exprs=output_exprs,
    )


def _write_type_map(w: BinWriter, m: dict[str, Type], sidx: dict[str, int]) -> None:
    w.write_u32(len(m))
    for k, t in m.items():
        w.write_u32(sidx[k])
        _write_type(w, t)


def _read_type_map(r: BinReader, strings: list[str]) -> dict[str, Type]:
    out: dict[str, Type] = {}
    for _ in range(r.read_u32()):
        k = strings[r.read_u32()]
        out[k] = _read_type(r)
    return out


def _write_expr_map(
    w: BinWriter, m: dict[str, Expr], id_map: dict[int, int], sidx: dict[str, int]
) -> None:
    w.write_u32(len(m))
    for k, e in m.items():
        w.write_u32(sidx[k])
        w.write_u32(id_map[id(e)])


def _read_expr_map(
    r: BinReader, nodes: list[Expr], strings: list[str]
) -> dict[str, Expr]:
    out: dict[str, Expr] = {}
    for _ in range(r.read_u32()):
        k = strings[r.read_u32()]
        e = nodes[r.read_u32()]
        out[k] = e
    return out


def _write_node_payload(
    w: BinWriter,
    node: Expr,
    id_map: dict[int, int],
    sidx: dict[str, int],
) -> None:
    def ref(e: Expr) -> int:
        try:
            return id_map[id(e)]
        except KeyError as exc:
            raise KeyError(f"missing node id for {type(e).__name__}") from exc

    if isinstance(node, Var):
        w.write_u32(sidx[node.name])
        return
    if isinstance(node, BoolConst):
        w.write_u8(1 if node.value else 0)
        return
    if isinstance(node, BitVecConst):
        w.write_u32(node.width)
        w.write_big_uint(node.value)
        return
    if isinstance(node, FloatConst):
        w.write_u32(node.width)
        w.write_u64(node.bits)
        return
    if isinstance(node, SimdConst):
        w.write_u32(node.lane_width)
        w.write_u32(node.lanes)
        w.write_big_uint(node.value)
        return
    if isinstance(node, Not):
        w.write_u32(ref(node.x))
        return
    if isinstance(
        node,
        (
            And,
            Or,
            Xor,
            Add,
            Sub,
            Mul,
            Div,
            Eq,
            Ult,
            Ule,
            Ugt,
            Uge,
            SimdAdd,
            SimdSub,
            SimdAddSatU,
            SimdSubSatU,
            SimdAddSatS,
            SimdSubSatS,
            SimdMulLo,
            SimdMulHiU,
            SimdMulHiS,
            SimdMaddS16,
            SimdUnpackLo,
            SimdUnpackHi,
            SimdPackSS16To8,
            SimdPackUS16To8,
            SimdPackSS32To16,
            SimdMinU,
            SimdMaxU,
            SimdMinS,
            SimdMaxS,
            SimdEq,
            SimdAnd,
            SimdOr,
            SimdXor,
            SimdUlt,
            SimdUle,
            SimdUgt,
            SimdUge,
            SimdSlt,
            SimdSle,
            SimdSgt,
            SimdSge,
            FAdd,
            FSub,
            FMul,
            FDiv,
            FEq,
            FLt,
            FLe,
            FNe,
            SimdFAdd,
            SimdFSub,
            SimdFMul,
            SimdFDiv,
            SimdFCmpEq,
            SimdFCmpLt,
            SimdFCmpLe,
            SimdFCmpNe,
        ),
    ):
        w.write_u32(ref(node.a))
        w.write_u32(ref(node.b))
        return
    if isinstance(node, (Shl, LShr, AShr)):
        w.write_u32(ref(node.a))
        w.write_u32(ref(node.b))
        return
    if isinstance(node, (SimdShl, SimdLShr, SimdAShr)):
        w.write_u32(ref(node.a))
        w.write_u32(ref(node.sh))
        return
    if isinstance(node, Mux):
        w.write_u32(ref(node.cond))
        w.write_u32(ref(node.a))
        w.write_u32(ref(node.b))
        return
    if isinstance(node, Concat):
        w.write_u32(len(node.parts))
        for p in node.parts:
            w.write_u32(ref(p))
        return
    if isinstance(node, Slice):
        w.write_u32(ref(node.x))
        w.write_u32(node.offset)
        w.write_u32(node.width)
        return
    if isinstance(node, Lut8):
        w.write_u32(ref(node.x))
        w.write_u32(len(node.table))
        for v in node.table:
            w.write_u8(int(v) & 0xFF)
        return
    if isinstance(node, TernaryLut):
        w.write_u32(ref(node.a))
        w.write_u32(ref(node.b))
        w.write_u32(ref(node.c))
        w.write_u8(node.imm8 & 0xFF)
        return
    if isinstance(node, Bitcast):
        _write_type(w, node.to)
        w.write_u32(ref(node.x))
        return
    if isinstance(node, BitTranspose):
        w.write_u32(ref(node.x))
        w.write_u32(node.lane_width)
        w.write_u32(node.lanes)
        return
    if isinstance(node, (Rotl, Rotr)):
        w.write_u32(ref(node.x))
        w.write_u32(ref(node.sh))
        return
    if isinstance(node, Delay):
        w.write_u32(ref(node.x))
        w.write_u32(node.ticks)
        return
    if isinstance(node, (FNeg, FAbs, FSqrt, SimdFNeg, SimdFAbs, SimdFSqrt, SimdNot)):
        w.write_u32(ref(node.x))
        return
    if isinstance(node, SimdAddMasked) or isinstance(node, SimdSubMasked):
        w.write_u32(ref(node.mask))
        w.write_u32(ref(node.a))
        w.write_u32(ref(node.b))
        return
    if isinstance(node, SimdBlend):
        w.write_u32(ref(node.mask))
        w.write_u32(ref(node.a))
        w.write_u32(ref(node.b))
        return
    if isinstance(node, SimdMaskExpand):
        _write_type(w, node.to)
        w.write_u32(ref(node.x))
        return
    if isinstance(node, SimdMaskPack):
        w.write_u32(ref(node.x))
        return
    if isinstance(node, SimdZExtLo) or isinstance(node, SimdSExtLo):
        _write_type(w, node.to)
        w.write_u32(ref(node.x))
        return
    if isinstance(node, SimdSplat):
        _write_type(w, node.to)
        w.write_u32(ref(node.x))
        return
    if isinstance(node, SimdExtractLane):
        w.write_u32(ref(node.x))
        w.write_u32(node.lane)
        return
    if isinstance(node, SimdInsertLane):
        w.write_u32(ref(node.x))
        w.write_u32(node.lane)
        w.write_u32(ref(node.value))
        return
    if isinstance(node, SimdShuffle):
        w.write_u32(ref(node.x))
        w.write_u32(len(node.indices))
        for i in node.indices:
            w.write_u32(i)
        return
    if isinstance(node, SimdFFma):
        w.write_u32(ref(node.a))
        w.write_u32(ref(node.b))
        w.write_u32(ref(node.c))
        return
    raise TypeError(f"unsupported expr {type(node)}")


def _read_nodes(r: BinReader, strings: list[str]) -> list[Expr]:
    nodes: list[Expr] = []
    node_count = r.read_u32()
    opcode_classes = list(EXPR_CLASSES)
    for _ in range(node_count):
        opcode = r.read_u8()
        cls = opcode_classes[opcode]
        if cls is Var:
            nodes.append(Var(name=strings[r.read_u32()]))
        elif cls is BoolConst:
            nodes.append(BoolConst(value=bool(r.read_u8())))
        elif cls is BitVecConst:
            nodes.append(BitVecConst(width=r.read_u32(), value=r.read_big_uint()))
        elif cls is FloatConst:
            nodes.append(FloatConst(width=r.read_u32(), bits=r.read_u64()))
        elif cls is SimdConst:
            nodes.append(
                SimdConst(
                    lane_width=r.read_u32(),
                    lanes=r.read_u32(),
                    value=r.read_big_uint(),
                )
            )
        elif cls in {Not, SimdNot, FNeg, FAbs, FSqrt, SimdFNeg, SimdFAbs, SimdFSqrt}:
            x = nodes[r.read_u32()]
            nodes.append(cls(x=x))
        elif cls in {
            And,
            Or,
            Xor,
            Add,
            Sub,
            Mul,
            Div,
            Eq,
            Ult,
            Ule,
            Ugt,
            Uge,
            SimdAdd,
            SimdSub,
            SimdAddSatU,
            SimdSubSatU,
            SimdAddSatS,
            SimdSubSatS,
            SimdMulLo,
            SimdMulHiU,
            SimdMulHiS,
            SimdMaddS16,
            SimdUnpackLo,
            SimdUnpackHi,
            SimdPackSS16To8,
            SimdPackUS16To8,
            SimdPackSS32To16,
            SimdMinU,
            SimdMaxU,
            SimdMinS,
            SimdMaxS,
            SimdEq,
            SimdAnd,
            SimdOr,
            SimdXor,
            SimdUlt,
            SimdUle,
            SimdUgt,
            SimdUge,
            SimdSlt,
            SimdSle,
            SimdSgt,
            SimdSge,
            FAdd,
            FSub,
            FMul,
            FDiv,
            FEq,
            FLt,
            FLe,
            FNe,
            SimdFAdd,
            SimdFSub,
            SimdFMul,
            SimdFDiv,
            SimdFCmpEq,
            SimdFCmpLt,
            SimdFCmpLe,
            SimdFCmpNe,
        }:
            a = nodes[r.read_u32()]
            b = nodes[r.read_u32()]
            nodes.append(cls(a=a, b=b))
        elif cls in {Shl, LShr, AShr}:
            a = nodes[r.read_u32()]
            b = nodes[r.read_u32()]
            nodes.append(cls(a=a, b=b))
        elif cls in {SimdShl, SimdLShr, SimdAShr}:
            a = nodes[r.read_u32()]
            sh = nodes[r.read_u32()]
            nodes.append(cls(a=a, sh=sh))
        elif cls is Mux:
            cond = nodes[r.read_u32()]
            a = nodes[r.read_u32()]
            b = nodes[r.read_u32()]
            nodes.append(Mux(cond=cond, a=a, b=b))
        elif cls is Concat:
            n = r.read_u32()
            parts = [nodes[r.read_u32()] for _ in range(n)]
            nodes.append(Concat(parts=parts))
        elif cls is Slice:
            x = nodes[r.read_u32()]
            offset = r.read_u32()
            width = r.read_u32()
            nodes.append(Slice(x=x, offset=offset, width=width))
        elif cls is Lut8:
            x = nodes[r.read_u32()]
            n = r.read_u32()
            table = [r.read_u8() for _ in range(n)]
            nodes.append(Lut8(x=x, table=table))
        elif cls is TernaryLut:
            a = nodes[r.read_u32()]
            b = nodes[r.read_u32()]
            c = nodes[r.read_u32()]
            imm8 = r.read_u8()
            nodes.append(TernaryLut(a=a, b=b, c=c, imm8=imm8))
        elif cls is Bitcast:
            to = _read_type(r)
            x = nodes[r.read_u32()]
            nodes.append(Bitcast(to=to, x=x))
        elif cls is BitTranspose:
            x = nodes[r.read_u32()]
            lane_width = r.read_u32()
            lanes = r.read_u32()
            nodes.append(BitTranspose(x=x, lane_width=lane_width, lanes=lanes))
        elif cls in {Rotl, Rotr}:
            x = nodes[r.read_u32()]
            sh = nodes[r.read_u32()]
            nodes.append(cls(x=x, sh=sh))
        elif cls is Delay:
            x = nodes[r.read_u32()]
            ticks = r.read_u32()
            nodes.append(Delay(x=x, ticks=ticks))
        elif cls in {SimdAddMasked, SimdSubMasked}:
            mask = nodes[r.read_u32()]
            a = nodes[r.read_u32()]
            b = nodes[r.read_u32()]
            nodes.append(cls(mask=mask, a=a, b=b))
        elif cls is SimdBlend:
            mask = nodes[r.read_u32()]
            a = nodes[r.read_u32()]
            b = nodes[r.read_u32()]
            nodes.append(SimdBlend(mask=mask, a=a, b=b))
        elif cls is SimdMaskExpand:
            to = _read_type(r)
            x = nodes[r.read_u32()]
            nodes.append(SimdMaskExpand(to=to, x=x))
        elif cls is SimdMaskPack:
            x = nodes[r.read_u32()]
            nodes.append(SimdMaskPack(x=x))
        elif cls in {SimdZExtLo, SimdSExtLo, SimdSplat}:
            to = _read_type(r)
            x = nodes[r.read_u32()]
            nodes.append(cls(to=to, x=x))
        elif cls is SimdExtractLane:
            x = nodes[r.read_u32()]
            lane = r.read_u32()
            nodes.append(SimdExtractLane(x=x, lane=lane))
        elif cls is SimdInsertLane:
            x = nodes[r.read_u32()]
            lane = r.read_u32()
            value = nodes[r.read_u32()]
            nodes.append(SimdInsertLane(x=x, lane=lane, value=value))
        elif cls is SimdShuffle:
            x = nodes[r.read_u32()]
            n = r.read_u32()
            indices = [r.read_u32() for _ in range(n)]
            nodes.append(SimdShuffle(x=x, indices=indices))
        elif cls is SimdFFma:
            a = nodes[r.read_u32()]
            b = nodes[r.read_u32()]
            c = nodes[r.read_u32()]
            nodes.append(SimdFFma(a=a, b=b, c=c))
        else:
            raise ValueError(f"unsupported expr class {cls}")
    return nodes


def _kind_for_class(cls: type[Expr]) -> str:
    name = cls.__name__
    mapping = {
        "Var": "var",
        "BoolConst": "bool_const",
        "BitVecConst": "bitvec_const",
        "FloatConst": "float_const",
        "SimdConst": "simd_const",
        "Not": "not",
        "And": "and",
        "Or": "or",
        "Xor": "xor",
        "Add": "add",
        "Sub": "sub",
        "Mul": "mul",
        "Div": "div",
        "Shl": "shl",
        "LShr": "lshr",
        "AShr": "ashr",
        "SimdAdd": "simd_add",
        "SimdSub": "simd_sub",
        "SimdAddMasked": "simd_add_masked",
        "SimdSubMasked": "simd_sub_masked",
        "SimdAddSatU": "simd_add_sat_u",
        "SimdSubSatU": "simd_sub_sat_u",
        "SimdAddSatS": "simd_add_sat_s",
        "SimdSubSatS": "simd_sub_sat_s",
        "SimdMulLo": "simd_mul_lo",
        "SimdMulHiU": "simd_mul_hi_u",
        "SimdMulHiS": "simd_mul_hi_s",
        "SimdMaddS16": "simd_madd_s16",
        "SimdUnpackLo": "simd_unpack_lo",
        "SimdUnpackHi": "simd_unpack_hi",
        "SimdPackSS16To8": "simd_pack_ss16_to_8",
        "SimdPackUS16To8": "simd_pack_us16_to_8",
        "SimdPackSS32To16": "simd_pack_ss32_to_16",
        "SimdMaskExpand": "simd_mask_expand",
        "SimdMaskPack": "simd_mask_pack",
        "SimdMinU": "simd_min_u",
        "SimdMaxU": "simd_max_u",
        "SimdMinS": "simd_min_s",
        "SimdMaxS": "simd_max_s",
        "SimdBlend": "simd_blend",
        "SimdZExtLo": "simd_zext_lo",
        "SimdSExtLo": "simd_sext_lo",
        "SimdEq": "simd_eq",
        "SimdNot": "simd_not",
        "SimdAnd": "simd_and",
        "SimdOr": "simd_or",
        "SimdXor": "simd_xor",
        "SimdShl": "simd_shl",
        "SimdLShr": "simd_lshr",
        "SimdAShr": "simd_ashr",
        "SimdUlt": "simd_ult",
        "SimdUle": "simd_ule",
        "SimdUgt": "simd_ugt",
        "SimdUge": "simd_uge",
        "SimdSlt": "simd_slt",
        "SimdSle": "simd_sle",
        "SimdSgt": "simd_sgt",
        "SimdSge": "simd_sge",
        "SimdSplat": "simd_splat",
        "SimdExtractLane": "simd_extract_lane",
        "SimdInsertLane": "simd_insert_lane",
        "SimdShuffle": "simd_shuffle",
        "Eq": "eq",
        "Ult": "ult",
        "Ule": "ule",
        "Ugt": "ugt",
        "Uge": "uge",
        "Mux": "mux",
        "Concat": "concat",
        "Slice": "slice",
        "Lut8": "lut8",
        "TernaryLut": "ternary_lut",
        "Bitcast": "bitcast",
        "BitTranspose": "bit_transpose",
        "Rotl": "rotl",
        "Rotr": "rotr",
        "Delay": "delay",
        "FNeg": "fneg",
        "FAbs": "fabs",
        "FAdd": "fadd",
        "FSub": "fsub",
        "FMul": "fmul",
        "FDiv": "fdiv",
        "FSqrt": "fsqrt",
        "FEq": "feq",
        "FLt": "flt",
        "FLe": "fle",
        "FNe": "fne",
        "SimdFAdd": "simd_fadd",
        "SimdFSub": "simd_fsub",
        "SimdFMul": "simd_fmul",
        "SimdFDiv": "simd_fdiv",
        "SimdFFma": "simd_ffma",
        "SimdFSqrt": "simd_fsqrt",
        "SimdFNeg": "simd_fneg",
        "SimdFAbs": "simd_fabs",
        "SimdFCmpEq": "simd_fcmp_eq",
        "SimdFCmpLt": "simd_fcmp_lt",
        "SimdFCmpLe": "simd_fcmp_le",
        "SimdFCmpNe": "simd_fcmp_ne",
    }
    if name not in mapping:
        raise KeyError(f"missing kind mapping for {name}")
    return mapping[name]
