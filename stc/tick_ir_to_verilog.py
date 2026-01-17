from __future__ import annotations

from dataclasses import dataclass

from stc.interp import infer_type
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
    SimdAdd,
    SimdAddSatS,
    SimdAddSatU,
    SimdAShr,
    SimdConst,
    SimdEq,
    SimdExtractLane,
    SimdInsertLane,
    SimdLShr,
    SimdMaddS16,
    SimdMaskExpand,
    SimdMaskPack,
    SimdBlend,
    SimdMaxS,
    SimdMaxU,
    SimdMinS,
    SimdMinU,
    SimdSExtLo,
    SimdMulHiS,
    SimdMulHiU,
    SimdMulLo,
    SimdNot,
    SimdAnd,
    SimdOr,
    SimdPackSS16To8,
    SimdPackSS32To16,
    SimdPackUS16To8,
    SimdShl,
    SimdShuffle,
    SimdSplat,
    SimdSub,
    SimdSubSatS,
    SimdSubSatU,
    SimdType,
    SimdUnpackHi,
    SimdUnpackLo,
    SimdZExtLo,
    SimdSge,
    SimdSgt,
    SimdSle,
    SimdSlt,
    SimdUge,
    SimdUgt,
    SimdUle,
    SimdUlt,
    SimdXor,
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
from stc.tick_ir_validate import validate_tick_ir


@dataclass(frozen=True)
class VerilogEmitError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def _width(t: Type) -> int:
    if isinstance(t, BoolType):
        return 1
    if isinstance(t, BitVecType):
        return t.width
    assert isinstance(t, SimdType)
    return t.total_width


def _decl(name: str, t: Type) -> str:
    w = _width(t)
    if w == 1:
        return f"logic {name}"
    return f"logic [{w - 1}:0] {name}"


def _emit_expr(expr: Expr, types: dict[str, Type]) -> str:
    if isinstance(expr, BoolConst):
        return "1'b1" if expr.value else "1'b0"
    if isinstance(expr, BitVecConst):
        return f"{expr.width}'d{expr.value}"
    if isinstance(expr, SimdConst):
        total = expr.lane_width * expr.lanes
        return f"{total}'d{expr.value}"
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Bitcast):
        return _emit_expr(expr.x, types)
    if isinstance(expr, Not):
        return f"(~({_emit_expr(expr.x, types)}))"
    if isinstance(expr, And):
        return f"(({_emit_expr(expr.a, types)}) & ({_emit_expr(expr.b, types)}))"
    if isinstance(expr, Or):
        return f"(({_emit_expr(expr.a, types)}) | ({_emit_expr(expr.b, types)}))"
    if isinstance(expr, Xor):
        return f"(({_emit_expr(expr.a, types)}) ^ ({_emit_expr(expr.b, types)}))"
    if isinstance(expr, Add):
        return f"(({_emit_expr(expr.a, types)}) + ({_emit_expr(expr.b, types)}))"
    if isinstance(expr, Sub):
        return f"(({_emit_expr(expr.a, types)}) - ({_emit_expr(expr.b, types)}))"
    if isinstance(expr, Shl):
        return f"(({_emit_expr(expr.a, types)}) << ({_emit_expr(expr.b, types)}))"
    if isinstance(expr, LShr):
        return f"(({_emit_expr(expr.a, types)}) >> ({_emit_expr(expr.b, types)}))"
    if isinstance(expr, AShr):
        return (
            f"($signed({_emit_expr(expr.a, types)}) >>> ({_emit_expr(expr.b, types)}))"
        )
    if isinstance(expr, SimdSplat):
        if not isinstance(expr.to, SimdType):
            raise VerilogEmitError("simd_splat target must be simd")
        x = _emit_expr(expr.x, types)
        return f"{{{expr.to.lanes}{{{x}}}}}"
    if isinstance(expr, SimdAdd):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise VerilogEmitError("simd_add requires simd operands")
        a = _emit_expr(expr.a, types)
        b = _emit_expr(expr.b, types)
        lane_exprs: list[str] = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            if t.lane_width == 1:
                sa = f"({a})[{off}]"
                sb = f"({b})[{off}]"
            else:
                hi = off + t.lane_width - 1
                sa = f"({a})[{hi}:{off}]"
                sb = f"({b})[{hi}:{off}]"
            lane_exprs.append(f"(({sa}) + ({sb}))")
        return "{" + ", ".join(lane_exprs) + "}"
    if isinstance(expr, SimdSub):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise VerilogEmitError("simd_sub requires simd operands")
        a = _emit_expr(expr.a, types)
        b = _emit_expr(expr.b, types)
        lane_exprs = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            if t.lane_width == 1:
                sa = f"({a})[{off}]"
                sb = f"({b})[{off}]"
            else:
                hi = off + t.lane_width - 1
                sa = f"({a})[{hi}:{off}]"
                sb = f"({b})[{hi}:{off}]"
            lane_exprs.append(f"(({sa}) - ({sb}))")
        return "{" + ", ".join(lane_exprs) + "}"
    if isinstance(expr, SimdAddSatU):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise VerilogEmitError("simd_add_satu requires simd operands")
        a = _emit_expr(expr.a, types)
        b = _emit_expr(expr.b, types)
        lane_exprs: list[str] = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            if t.lane_width == 1:
                sa = f"({a})[{off}]"
                sb = f"({b})[{off}]"
            else:
                hi = off + t.lane_width - 1
                sa = f"({a})[{hi}:{off}]"
                sb = f"({b})[{hi}:{off}]"
            s = f"({{1'b0, ({sa})}} + {{1'b0, ({sb})}})"
            carry = f"({s})[{t.lane_width}]"
            low = f"({s})[{t.lane_width - 1}:0]" if t.lane_width > 1 else f"({s})[0]"
            lane_exprs.append(f"(({carry}) ? {{{t.lane_width}{{1'b1}}}} : ({low}))")
        return "{" + ", ".join(lane_exprs) + "}"
    if isinstance(expr, SimdSubSatU):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise VerilogEmitError("simd_sub_satu requires simd operands")
        a = _emit_expr(expr.a, types)
        b = _emit_expr(expr.b, types)
        lane_exprs = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            if t.lane_width == 1:
                sa = f"({a})[{off}]"
                sb = f"({b})[{off}]"
            else:
                hi = off + t.lane_width - 1
                sa = f"({a})[{hi}:{off}]"
                sb = f"({b})[{hi}:{off}]"
            lane_exprs.append(
                f"((({sa}) < ({sb})) ? {{{t.lane_width}{{1'b0}}}} : (({sa}) - ({sb})))"
            )
        return "{" + ", ".join(lane_exprs) + "}"
    if isinstance(expr, (SimdAddSatS, SimdSubSatS)):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise VerilogEmitError("simd saturating signed op requires simd operands")
        a = _emit_expr(expr.a, types)
        b = _emit_expr(expr.b, types)
        w = t.lane_width
        maxv = (1 << (w - 1)) - 1
        minv = 1 << (w - 1)
        maxc = f"{w}'d{maxv}"
        minc = f"{w}'d{minv}"
        lane_exprs: list[str] = []
        for i in reversed(range(t.lanes)):
            off = i * w
            if w == 1:
                sa = f"({a})[{off}]"
                sb = f"({b})[{off}]"
            else:
                hi = off + w - 1
                sa = f"({a})[{hi}:{off}]"
                sb = f"({b})[{hi}:{off}]"
            r = (
                f"(({sa}) + ({sb}))"
                if isinstance(expr, SimdAddSatS)
                else f"(({sa}) - ({sb}))"
            )
            sign_idx = w - 1
            sa_s = f"({sa})[{sign_idx}]"
            sb_s = f"({sb})[{sign_idx}]"
            r_s = f"({r})[{sign_idx}]"
            if isinstance(expr, SimdAddSatS):
                ov = f"((~(({sa_s}) ^ ({sb_s}))) & (({r_s}) ^ ({sa_s})))"
            else:
                ov = f"((({sa_s}) ^ ({sb_s})) & (({r_s}) ^ ({sa_s})))"
            lane_exprs.append(f"(({ov}) ? (({sa_s}) ? ({minc}) : ({maxc})) : ({r}))")
        return "{" + ", ".join(lane_exprs) + "}"
    if isinstance(expr, (SimdMulLo, SimdMulHiU, SimdMulHiS)):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise VerilogEmitError("simd mul requires simd operands")
        a = _emit_expr(expr.a, types)
        b = _emit_expr(expr.b, types)
        w = t.lane_width
        lane_exprs: list[str] = []
        for i in reversed(range(t.lanes)):
            off = i * w
            if w == 1:
                sa = f"({a})[{off}]"
                sb = f"({b})[{off}]"
                lane_exprs.append(f"(({sa}) & ({sb}))")
                continue
            hi = off + w - 1
            sa = f"({a})[{hi}:{off}]"
            sb = f"({b})[{hi}:{off}]"
            prod = f"(({sa}) * ({sb}))"
            if isinstance(expr, SimdMulLo):
                lane_exprs.append(f"({prod})[{w - 1}:0]")
            elif isinstance(expr, SimdMulHiU):
                lane_exprs.append(f"({prod})[{(2 * w) - 1}:{w}]")
            else:
                sprod = f"($signed({sa}) * $signed({sb}))"
                lane_exprs.append(f"({sprod})[{(2 * w) - 1}:{w}]")
        return "{" + ", ".join(lane_exprs) + "}"
    if isinstance(expr, SimdMaddS16):
        out_t = infer_type(expr, types)
        if not isinstance(out_t, SimdType):
            raise VerilogEmitError("simd_madd_s16 requires simd operands")
        a_t = infer_type(expr.a, types)
        if not isinstance(a_t, SimdType):
            raise VerilogEmitError("simd_madd_s16 requires simd operands")
        if a_t.lane_width != 16 or a_t.lanes % 2 != 0:
            raise VerilogEmitError("simd_madd_s16 requires simd[16, even]")
        a = _emit_expr(expr.a, types)
        b = _emit_expr(expr.b, types)
        lane_exprs: list[str] = []
        for i in reversed(range(out_t.lanes)):
            a0 = f"({a})[{(16 * (2 * i)) + 15}:{16 * (2 * i)}]"
            a1 = f"({a})[{(16 * (2 * i + 1)) + 15}:{16 * (2 * i + 1)}]"
            b0 = f"({b})[{(16 * (2 * i)) + 15}:{16 * (2 * i)}]"
            b1 = f"({b})[{(16 * (2 * i + 1)) + 15}:{16 * (2 * i + 1)}]"
            s = f"(($signed({a0}) * $signed({b0})) + ($signed({a1}) * $signed({b1})))"
            lane_exprs.append(f"({s})[31:0]")
        return "{" + ", ".join(lane_exprs) + "}"
    if isinstance(expr, (SimdUnpackLo, SimdUnpackHi)):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise VerilogEmitError("simd unpack requires simd operands")
        if t.lanes % 2 != 0:
            raise VerilogEmitError("simd unpack requires even lanes")
        a = _emit_expr(expr.a, types)
        b = _emit_expr(expr.b, types)
        half = t.lanes // 2
        w = t.lane_width
        lane_exprs: list[str] = []
        for out_lane in reversed(range(t.lanes)):
            src_lane = (out_lane // 2) + (0 if isinstance(expr, SimdUnpackLo) else half)
            src_vec = a if (out_lane % 2 == 0) else b
            off = src_lane * w
            if w == 1:
                lane_exprs.append(f"({src_vec})[{off}]")
            else:
                lane_exprs.append(f"({src_vec})[{off + w - 1}:{off}]")
        return "{" + ", ".join(lane_exprs) + "}"
    if isinstance(expr, (SimdPackSS16To8, SimdPackUS16To8)):
        a_t = infer_type(expr.a, types)
        if not isinstance(a_t, SimdType):
            raise VerilogEmitError("simd pack requires simd operands")
        if a_t.lane_width != 16:
            raise VerilogEmitError("simd_pack_*16_to_8 requires lane_width=16")
        a = _emit_expr(expr.a, types)
        b = _emit_expr(expr.b, types)
        lane_exprs: list[str] = []
        for out_lane in reversed(range(a_t.lanes * 2)):
            src_vec = a if out_lane < a_t.lanes else b
            src_lane = out_lane if out_lane < a_t.lanes else (out_lane - a_t.lanes)
            off = 16 * src_lane
            src = f"({src_vec})[{off + 15}:{off}]"
            if isinstance(expr, SimdPackUS16To8):
                lane_exprs.append(
                    f"(($signed({src}) < 16'sd0) ? 8'd0 : (($signed({src}) > 16'sd255) ? 8'd255 : ({src})[7:0]))"
                )
            else:
                lane_exprs.append(
                    f"(($signed({src}) > 16'sd127) ? 8'd127 : (($signed({src}) < -16'sd128) ? 8'd128 : ({src})[7:0]))"
                )
        return "{" + ", ".join(lane_exprs) + "}"
    if isinstance(expr, SimdPackSS32To16):
        a_t = infer_type(expr.a, types)
        if not isinstance(a_t, SimdType):
            raise VerilogEmitError("simd pack requires simd operands")
        if a_t.lane_width != 32:
            raise VerilogEmitError("simd_pack_ss32_to_16 requires lane_width=32")
        a = _emit_expr(expr.a, types)
        b = _emit_expr(expr.b, types)
        lane_exprs: list[str] = []
        for out_lane in reversed(range(a_t.lanes * 2)):
            src_vec = a if out_lane < a_t.lanes else b
            src_lane = out_lane if out_lane < a_t.lanes else (out_lane - a_t.lanes)
            off = 32 * src_lane
            src = f"({src_vec})[{off + 31}:{off}]"
            lane_exprs.append(
                f"(($signed({src}) > 32'sd32767) ? 16'd32767 : (($signed({src}) < -32'sd32768) ? 16'd32768 : ({src})[15:0]))"
            )
        return "{" + ", ".join(lane_exprs) + "}"
    if isinstance(expr, SimdMaskExpand):
        if not isinstance(expr.to, SimdType):
            raise VerilogEmitError("simd_mask_expand target must be simd")
        src_t = infer_type(expr.x, types)
        if not (isinstance(src_t, SimdType) and src_t.lane_width == 1):
            raise VerilogEmitError("simd_mask_expand source must be simd mask")
        if src_t.lanes != expr.to.lanes:
            raise VerilogEmitError("simd_mask_expand lane mismatch")
        m = _emit_expr(expr.x, types)
        w = expr.to.lane_width
        lane_exprs: list[str] = []
        for i in reversed(range(expr.to.lanes)):
            lane_exprs.append(f"(({m})[{i}] ? {{{w}{{1'b1}}}} : {{{w}{{1'b0}}}})")
        return "{" + ", ".join(lane_exprs) + "}"
    if isinstance(expr, SimdMaskPack):
        src_t = infer_type(expr.x, types)
        if not isinstance(src_t, SimdType):
            raise VerilogEmitError("simd_mask_pack source must be simd")
        x = _emit_expr(expr.x, types)
        w = src_t.lane_width
        lane_exprs: list[str] = []
        for i in reversed(range(src_t.lanes)):
            off = i * w
            if w == 1:
                lane = f"({x})[{off}]"
            else:
                lane = f"({x})[{off + w - 1}:{off}]"
            lane_exprs.append(f"(({lane}) != {{{w}{{1'b0}}}})")
        return "{" + ", ".join(lane_exprs) + "}"
    if isinstance(expr, (SimdMinU, SimdMaxU, SimdMinS, SimdMaxS)):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise VerilogEmitError("simd min/max requires simd operands")
        a = _emit_expr(expr.a, types)
        b = _emit_expr(expr.b, types)
        w = t.lane_width
        lane_exprs: list[str] = []
        for i in reversed(range(t.lanes)):
            off = i * w
            if w == 1:
                sa = f"({a})[{off}]"
                sb = f"({b})[{off}]"
            else:
                sa = f"({a})[{off + w - 1}:{off}]"
                sb = f"({b})[{off + w - 1}:{off}]"
            if isinstance(expr, SimdMinU):
                lane_exprs.append(f"(({sa}) < ({sb}) ? ({sa}) : ({sb}))")
            elif isinstance(expr, SimdMaxU):
                lane_exprs.append(f"(({sa}) > ({sb}) ? ({sa}) : ({sb}))")
            elif isinstance(expr, SimdMinS):
                lane_exprs.append(f"($signed({sa}) < $signed({sb}) ? ({sa}) : ({sb}))")
            else:
                lane_exprs.append(f"($signed({sa}) > $signed({sb}) ? ({sa}) : ({sb}))")
        return "{" + ", ".join(lane_exprs) + "}"
    if isinstance(expr, SimdBlend):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise VerilogEmitError("simd_blend requires simd operands")
        mask_t = infer_type(expr.mask, types)
        if not isinstance(mask_t, SimdType) or mask_t.lane_width != 1:
            raise VerilogEmitError("simd_blend mask must be simd mask")
        if mask_t.lanes != t.lanes:
            raise VerilogEmitError("simd_blend mask lane mismatch")
        m = _emit_expr(expr.mask, types)
        a = _emit_expr(expr.a, types)
        b = _emit_expr(expr.b, types)
        w = t.lane_width
        lane_exprs: list[str] = []
        for i in reversed(range(t.lanes)):
            off = i * w
            if w == 1:
                sa = f"({a})[{off}]"
                sb = f"({b})[{off}]"
            else:
                sa = f"({a})[{off + w - 1}:{off}]"
                sb = f"({b})[{off + w - 1}:{off}]"
            lane_exprs.append(f"(({m})[{i}] ? ({sb}) : ({sa}))")
        return "{" + ", ".join(lane_exprs) + "}"
    if isinstance(expr, (SimdZExtLo, SimdSExtLo)):
        if not isinstance(expr.to, SimdType):
            raise VerilogEmitError("simd extend target must be simd")
        src_t = infer_type(expr.x, types)
        if not isinstance(src_t, SimdType):
            raise VerilogEmitError("simd extend source must be simd")
        dst_t = expr.to
        if dst_t.total_width != src_t.total_width:
            raise VerilogEmitError("simd extend total_width mismatch")
        if dst_t.lanes != src_t.lanes // 2 or dst_t.lane_width != src_t.lane_width * 2:
            raise VerilogEmitError("simd extend shape mismatch")
        x = _emit_expr(expr.x, types)
        in_w = src_t.lane_width
        out_w = dst_t.lane_width
        lane_exprs: list[str] = []
        for i in reversed(range(dst_t.lanes)):
            off = i * in_w
            src = f"({x})[{off + in_w - 1}:{off}]"
            if isinstance(expr, SimdZExtLo):
                lane_exprs.append(f"{{{out_w - in_w}{{1'b0}}, {src}}}")
            else:
                lane_exprs.append(f"{{{out_w - in_w}{{({src})[{in_w - 1}]}}, {src}}}")
        return "{" + ", ".join(lane_exprs) + "}"
    if isinstance(expr, SimdEq):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise VerilogEmitError("simd_eq requires simd operands")
        a = _emit_expr(expr.a, types)
        b = _emit_expr(expr.b, types)
        lane_exprs: list[str] = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            if t.lane_width == 1:
                sa = f"({a})[{off}]"
                sb = f"({b})[{off}]"
            else:
                hi = off + t.lane_width - 1
                sa = f"({a})[{hi}:{off}]"
                sb = f"({b})[{hi}:{off}]"
            lane_exprs.append(f"(({sa}) == ({sb}))")
        return "{" + ", ".join(lane_exprs) + "}"
    if isinstance(expr, SimdNot):
        t = infer_type(expr.x, types)
        if not isinstance(t, SimdType):
            raise VerilogEmitError("simd_not requires simd operand")
        return f"(~({_emit_expr(expr.x, types)}))"
    if isinstance(expr, SimdAnd):
        return f"(({_emit_expr(expr.a, types)}) & ({_emit_expr(expr.b, types)}))"
    if isinstance(expr, SimdOr):
        return f"(({_emit_expr(expr.a, types)}) | ({_emit_expr(expr.b, types)}))"
    if isinstance(expr, SimdXor):
        return f"(({_emit_expr(expr.a, types)}) ^ ({_emit_expr(expr.b, types)}))"
    if isinstance(expr, (SimdShl, SimdLShr, SimdAShr)):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise VerilogEmitError("simd shift requires simd lhs")
        a = _emit_expr(expr.a, types)
        sh = _emit_expr(expr.sh, types)
        lane_exprs = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            if t.lane_width == 1:
                sa = f"({a})[{off}]"
            else:
                hi = off + t.lane_width - 1
                sa = f"({a})[{hi}:{off}]"
            if isinstance(expr, SimdShl):
                lane_exprs.append(f"(({sa}) << ({sh}))")
            elif isinstance(expr, SimdLShr):
                lane_exprs.append(f"(({sa}) >> ({sh}))")
            else:
                lane_exprs.append(f"($signed({sa}) >>> ({sh}))")
        return "{" + ", ".join(lane_exprs) + "}"
    if isinstance(expr, (SimdUlt, SimdUle, SimdUgt, SimdUge)):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise VerilogEmitError("simd unsigned compare requires simd operands")
        a = _emit_expr(expr.a, types)
        b = _emit_expr(expr.b, types)
        lane_exprs = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            if t.lane_width == 1:
                sa = f"({a})[{off}]"
                sb = f"({b})[{off}]"
            else:
                hi = off + t.lane_width - 1
                sa = f"({a})[{hi}:{off}]"
                sb = f"({b})[{hi}:{off}]"
            if isinstance(expr, SimdUlt):
                lane_exprs.append(f"(({sa}) < ({sb}))")
            elif isinstance(expr, SimdUle):
                lane_exprs.append(f"(({sa}) <= ({sb}))")
            elif isinstance(expr, SimdUgt):
                lane_exprs.append(f"(({sa}) > ({sb}))")
            else:
                lane_exprs.append(f"(({sa}) >= ({sb}))")
        return "{" + ", ".join(lane_exprs) + "}"
    if isinstance(expr, (SimdSlt, SimdSle, SimdSgt, SimdSge)):
        t = infer_type(expr.a, types)
        if not isinstance(t, SimdType):
            raise VerilogEmitError("simd signed compare requires simd operands")
        a = _emit_expr(expr.a, types)
        b = _emit_expr(expr.b, types)
        lane_exprs = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            if t.lane_width == 1:
                sa = f"({a})[{off}]"
                sb = f"({b})[{off}]"
            else:
                hi = off + t.lane_width - 1
                sa = f"({a})[{hi}:{off}]"
                sb = f"({b})[{hi}:{off}]"
            if isinstance(expr, SimdSlt):
                lane_exprs.append(f"($signed({sa}) < $signed({sb}))")
            elif isinstance(expr, SimdSle):
                lane_exprs.append(f"($signed({sa}) <= $signed({sb}))")
            elif isinstance(expr, SimdSgt):
                lane_exprs.append(f"($signed({sa}) > $signed({sb}))")
            else:
                lane_exprs.append(f"($signed({sa}) >= $signed({sb}))")
        return "{" + ", ".join(lane_exprs) + "}"
    if isinstance(expr, SimdExtractLane):
        t = infer_type(expr.x, types)
        if not isinstance(t, SimdType):
            raise VerilogEmitError("simd_extract_lane requires simd source")
        if expr.lane >= t.lanes:
            raise VerilogEmitError("lane out of bounds")
        x = _emit_expr(expr.x, types)
        off = expr.lane * t.lane_width
        if t.lane_width == 1:
            return f"({x})[{off}]"
        return f"({x})[{off + t.lane_width - 1}:{off}]"
    if isinstance(expr, SimdInsertLane):
        t = infer_type(expr.x, types)
        if not isinstance(t, SimdType):
            raise VerilogEmitError("simd_insert_lane requires simd base")
        if expr.lane >= t.lanes:
            raise VerilogEmitError("lane out of bounds")
        x = _emit_expr(expr.x, types)
        v = _emit_expr(expr.value, types)
        parts: list[str] = []
        for i in reversed(range(t.lanes)):
            off = i * t.lane_width
            if i == expr.lane:
                parts.append(v)
            else:
                if t.lane_width == 1:
                    parts.append(f"({x})[{off}]")
                else:
                    parts.append(f"({x})[{off + t.lane_width - 1}:{off}]")
        return "{" + ", ".join(parts) + "}"
    if isinstance(expr, SimdShuffle):
        t = infer_type(expr.x, types)
        if not isinstance(t, SimdType):
            raise VerilogEmitError("simd_shuffle requires simd source")
        if not expr.indices:
            raise VerilogEmitError("indices must be non-empty")
        x = _emit_expr(expr.x, types)
        parts: list[str] = []
        for out_lane in reversed(range(len(expr.indices))):
            src_lane = int(expr.indices[out_lane])
            if src_lane < 0 or src_lane >= t.lanes:
                raise VerilogEmitError("shuffle index out of bounds")
            off = src_lane * t.lane_width
            if t.lane_width == 1:
                parts.append(f"({x})[{off}]")
            else:
                parts.append(f"({x})[{off + t.lane_width - 1}:{off}]")
        return "{" + ", ".join(parts) + "}"
    if isinstance(expr, Eq):
        return f"(({_emit_expr(expr.a, types)}) == ({_emit_expr(expr.b, types)}))"
    if isinstance(expr, Ult):
        return f"(({_emit_expr(expr.a, types)}) < ({_emit_expr(expr.b, types)}))"
    if isinstance(expr, Ule):
        return f"(({_emit_expr(expr.a, types)}) <= ({_emit_expr(expr.b, types)}))"
    if isinstance(expr, Ugt):
        return f"(({_emit_expr(expr.a, types)}) > ({_emit_expr(expr.b, types)}))"
    if isinstance(expr, Uge):
        return f"(({_emit_expr(expr.a, types)}) >= ({_emit_expr(expr.b, types)}))"
    if isinstance(expr, Mux):
        return f"(({_emit_expr(expr.cond, types)}) ? ({_emit_expr(expr.a, types)}) : ({_emit_expr(expr.b, types)}))"
    if isinstance(expr, Concat):
        if not expr.parts:
            raise VerilogEmitError("empty concat is not supported")
        inner = ", ".join(_emit_expr(p, types) for p in expr.parts)
        return f"{{{inner}}}"
    if isinstance(expr, Slice):
        if expr.width == 1:
            return f"({_emit_expr(expr.x, types)})[{expr.offset}]"
        return f"({_emit_expr(expr.x, types)})[{expr.offset + expr.width - 1}:{expr.offset}]"
    raise VerilogEmitError("unsupported expression kind")


def emit_verilog(ir: TickIR, *, module_name: str = "tickir_impl") -> str:
    validate_tick_ir(ir)

    inputs = dict(ir.inputs)
    outputs = dict(ir.outputs)
    types: dict[str, Type] = {**inputs, **ir.state}

    ports: list[str] = ["input logic clk", "input logic rst"]
    ports.extend([f"input {_decl(name, t)}" for name, t in inputs.items()])
    ports.extend([f"output {_decl(name, t)}" for name, t in outputs.items()])

    lines: list[str] = []
    lines.append(f"module {module_name}({', '.join(ports)});")

    for name, t in ir.state.items():
        lines.append(f"  {_decl(name, t)};")

    for name, t in ir.outputs.items():
        lines.append(f"  assign {name} = {_emit_expr(ir.output_exprs[name], types)};")

    if ir.state:
        lines.append("  always_ff @(posedge clk) begin")
        lines.append("    if (rst) begin")
        for name, expr in ir.reset_state.items():
            lines.append(f"      {name} <= {_emit_expr(expr, types)};")
        lines.append("    end else begin")
        for name, expr in ir.next_state.items():
            lines.append(f"      {name} <= {_emit_expr(expr, types)};")
        lines.append("    end")
        lines.append("  end")

    lines.append("endmodule")
    lines.append("")
    return "\n".join(lines)
