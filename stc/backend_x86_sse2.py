from __future__ import annotations

from dataclasses import dataclass

from stc.interp import infer_type
from stc.tick_ir import (
    BitVecConst,
    BitVecType,
    Bitcast,
    Expr,
    SimdAdd,
    SimdAddSatS,
    SimdAddSatU,
    SimdAnd,
    SimdAShr,
    SimdBlend,
    SimdConst,
    SimdEq,
    SimdLShr,
    SimdMaskPack,
    SimdMaxS,
    SimdMaxU,
    SimdMaddS16,
    SimdMinS,
    SimdMinU,
    SimdSExtLo,
    SimdMulHiS,
    SimdMulHiU,
    SimdMulLo,
    SimdNot,
    SimdOr,
    SimdPackSS16To8,
    SimdPackSS32To16,
    SimdPackUS16To8,
    SimdShl,
    SimdShuffle,
    SimdSlt,
    SimdSub,
    SimdSubSatS,
    SimdSubSatU,
    SimdType,
    SimdUnpackHi,
    SimdUnpackLo,
    SimdUlt,
    SimdXor,
    SimdZExtLo,
    TickIR,
    Type,
    Var,
)
from stc.tick_ir_validate import validate_tick_ir


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


def _require_simd128(t: Type) -> SimdType:
    if not isinstance(t, SimdType):
        raise CodegenError("x86 sse2 backend requires simd types")
    if t.total_width != 128:
        raise CodegenError("x86 sse2 backend requires total_width=128")
    return t


def _emit_simd_const(x: SimdConst) -> str:
    lo = x.value & ((1 << 64) - 1)
    hi = (x.value >> 64) & ((1 << 64) - 1)
    return f"load128_imm({lo}ull, {hi}ull)"


def _emit_sh_amount(expr: Expr, types: dict[str, Type]) -> int:
    if not isinstance(expr, BitVecConst):
        raise CodegenError("sse2 simd shifts require constant shift amount")
    if expr.value < 0:
        raise CodegenError("shift amount must be >= 0")
    return int(expr.value)


def emit_x86_sse2_c(ir: TickIR) -> str:
    validate_tick_ir(ir)
    if ir.state:
        raise CodegenError("x86 sse2 backend does not support state yet")

    types: dict[str, Type] = dict(ir.inputs)

    for t in ir.inputs.values():
        _require_simd128(t)

    input_order = sorted(ir.inputs.keys())
    output_order = sorted(ir.outputs.keys())

    lines: list[str] = []
    lines.append("#include <stdint.h>")
    lines.append("#include <emmintrin.h>")
    lines.append("#include <tmmintrin.h>")
    lines.append("#include <smmintrin.h>")
    lines.append("")
    lines.append("typedef union { __m128i v; uint64_t u64[2]; } u128;")
    lines.append("")
    lines.append("static inline __m128i load128(const uint64_t* p) {")
    lines.append("  u128 x;")
    lines.append("  x.u64[0] = p[0];")
    lines.append("  x.u64[1] = p[1];")
    lines.append("  return x.v;")
    lines.append("}")
    lines.append("")
    lines.append("static inline __m128i load128_imm(uint64_t lo, uint64_t hi) {")
    lines.append("  u128 x;")
    lines.append("  x.u64[0] = lo;")
    lines.append("  x.u64[1] = hi;")
    lines.append("  return x.v;")
    lines.append("}")
    lines.append("")
    lines.append("static inline void store128(uint64_t* p, __m128i v) {")
    lines.append("  u128 x;")
    lines.append("  x.v = v;")
    lines.append("  p[0] = x.u64[0];")
    lines.append("  p[1] = x.u64[1];")
    lines.append("}")
    lines.append("")

    tmp_id = 0
    memo: dict[str, str] = {}

    def emit_expr(expr: Expr) -> str:
        nonlocal tmp_id
        key = repr(expr)
        if key in memo:
            return memo[key]

        if isinstance(expr, Var):
            name = _c_ident(expr.name)
            memo[key] = name
            return name

        if isinstance(expr, SimdConst):
            name = f"t{tmp_id}"
            tmp_id += 1
            lines.append(f"  __m128i {name} = {_emit_simd_const(expr)};")
            memo[key] = name
            return name

        if isinstance(expr, Bitcast):
            name = emit_expr(expr.x)
            memo[key] = name
            return name

        if isinstance(expr, SimdNot):
            x = emit_expr(expr.x)
            name = f"t{tmp_id}"
            tmp_id += 1
            lines.append(f"  __m128i {name} = _mm_xor_si128({x}, _mm_set1_epi32(-1));")
            memo[key] = name
            return name

        if isinstance(expr, (SimdAnd, SimdOr, SimdXor)):
            name = f"t{tmp_id}"
            tmp_id += 1
            if isinstance(expr, SimdAnd):
                if isinstance(expr.a, SimdNot):
                    a0 = emit_expr(expr.a.x)
                    b0 = emit_expr(expr.b)
                    lines.append(f"  __m128i {name} = _mm_andnot_si128({a0}, {b0});")
                    memo[key] = name
                    return name
                if isinstance(expr.b, SimdNot):
                    b0 = emit_expr(expr.b.x)
                    a0 = emit_expr(expr.a)
                    lines.append(f"  __m128i {name} = _mm_andnot_si128({b0}, {a0});")
                    memo[key] = name
                    return name
                a = emit_expr(expr.a)
                b = emit_expr(expr.b)
                op = "_mm_and_si128"
            elif isinstance(expr, SimdOr):
                a = emit_expr(expr.a)
                b = emit_expr(expr.b)
                op = "_mm_or_si128"
            else:
                a = emit_expr(expr.a)
                b = emit_expr(expr.b)
                op = "_mm_xor_si128"
            lines.append(f"  __m128i {name} = {op}({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdAdd, SimdSub)):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd128(infer_type(expr.a, types))
            name = f"t{tmp_id}"
            tmp_id += 1
            if t.lane_width == 8:
                op = "_mm_add_epi8" if isinstance(expr, SimdAdd) else "_mm_sub_epi8"
            elif t.lane_width == 16:
                op = "_mm_add_epi16" if isinstance(expr, SimdAdd) else "_mm_sub_epi16"
            elif t.lane_width == 32:
                op = "_mm_add_epi32" if isinstance(expr, SimdAdd) else "_mm_sub_epi32"
            elif t.lane_width == 64:
                op = "_mm_add_epi64" if isinstance(expr, SimdAdd) else "_mm_sub_epi64"
            else:
                raise CodegenError("unsupported lane_width for add/sub")
            lines.append(f"  __m128i {name} = {op}({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdAddSatU, SimdSubSatU)):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd128(infer_type(expr.a, types))
            name = f"t{tmp_id}"
            tmp_id += 1
            if t.lane_width == 8:
                op = (
                    "_mm_adds_epu8"
                    if isinstance(expr, SimdAddSatU)
                    else "_mm_subs_epu8"
                )
            elif t.lane_width == 16:
                op = (
                    "_mm_adds_epu16"
                    if isinstance(expr, SimdAddSatU)
                    else "_mm_subs_epu16"
                )
            else:
                raise CodegenError(
                    "sse2 unsigned saturating ops require lane_width in {8,16}"
                )
            lines.append(f"  __m128i {name} = {op}({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdAddSatS, SimdSubSatS)):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd128(infer_type(expr.a, types))
            name = f"t{tmp_id}"
            tmp_id += 1
            if t.lane_width == 8:
                op = (
                    "_mm_adds_epi8"
                    if isinstance(expr, SimdAddSatS)
                    else "_mm_subs_epi8"
                )
            elif t.lane_width == 16:
                op = (
                    "_mm_adds_epi16"
                    if isinstance(expr, SimdAddSatS)
                    else "_mm_subs_epi16"
                )
            else:
                raise CodegenError(
                    "sse2 signed saturating ops require lane_width in {8,16}"
                )
            lines.append(f"  __m128i {name} = {op}({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdMulLo, SimdMulHiU, SimdMulHiS)):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd128(infer_type(expr.a, types))
            name = f"t{tmp_id}"
            tmp_id += 1
            if t.lane_width != 16:
                raise CodegenError("sse2 mul ops require lane_width=16")
            if isinstance(expr, SimdMulLo):
                op = "_mm_mullo_epi16"
            elif isinstance(expr, SimdMulHiU):
                op = "_mm_mulhi_epu16"
            else:
                op = "_mm_mulhi_epi16"
            lines.append(f"  __m128i {name} = {op}({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, SimdMaddS16):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd128(infer_type(expr.a, types))
            name = f"t{tmp_id}"
            tmp_id += 1
            if t.lane_width != 16:
                raise CodegenError("sse2 madd requires lane_width=16")
            lines.append(f"  __m128i {name} = _mm_madd_epi16({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdUnpackLo, SimdUnpackHi)):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd128(infer_type(expr.a, types))
            name = f"t{tmp_id}"
            tmp_id += 1
            if t.lane_width == 8:
                op = (
                    "_mm_unpacklo_epi8"
                    if isinstance(expr, SimdUnpackLo)
                    else "_mm_unpackhi_epi8"
                )
            elif t.lane_width == 16:
                op = (
                    "_mm_unpacklo_epi16"
                    if isinstance(expr, SimdUnpackLo)
                    else "_mm_unpackhi_epi16"
                )
            elif t.lane_width == 32:
                op = (
                    "_mm_unpacklo_epi32"
                    if isinstance(expr, SimdUnpackLo)
                    else "_mm_unpackhi_epi32"
                )
            elif t.lane_width == 64:
                op = (
                    "_mm_unpacklo_epi64"
                    if isinstance(expr, SimdUnpackLo)
                    else "_mm_unpackhi_epi64"
                )
            else:
                raise CodegenError("sse2 unpack requires lane_width in {8,16,32,64}")
            lines.append(f"  __m128i {name} = {op}({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdPackSS16To8, SimdPackUS16To8)):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd128(infer_type(expr.a, types))
            name = f"t{tmp_id}"
            tmp_id += 1
            if t.lane_width != 16:
                raise CodegenError("sse2 pack 16->8 requires lane_width=16")
            op = (
                "_mm_packs_epi16"
                if isinstance(expr, SimdPackSS16To8)
                else "_mm_packus_epi16"
            )
            lines.append(f"  __m128i {name} = {op}({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, SimdPackSS32To16):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd128(infer_type(expr.a, types))
            name = f"t{tmp_id}"
            tmp_id += 1
            if t.lane_width != 32:
                raise CodegenError("sse2 pack 32->16 requires lane_width=32")
            lines.append(f"  __m128i {name} = _mm_packs_epi32({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdMinU, SimdMaxU, SimdMinS, SimdMaxS)):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd128(infer_type(expr.a, types))
            name = f"t{tmp_id}"
            tmp_id += 1
            if isinstance(expr, (SimdMinU, SimdMaxU)):
                if t.lane_width == 8:
                    op = (
                        "_mm_min_epu8" if isinstance(expr, SimdMinU) else "_mm_max_epu8"
                    )
                elif t.lane_width == 16:
                    op = (
                        "_mm_min_epu16"
                        if isinstance(expr, SimdMinU)
                        else "_mm_max_epu16"
                    )
                elif t.lane_width == 32:
                    op = (
                        "_mm_min_epu32"
                        if isinstance(expr, SimdMinU)
                        else "_mm_max_epu32"
                    )
                else:
                    raise CodegenError(
                        "sse4.1 min/max unsigned supports lane_width in {8,16,32}"
                    )
            else:
                if t.lane_width == 8:
                    op = (
                        "_mm_min_epi8" if isinstance(expr, SimdMinS) else "_mm_max_epi8"
                    )
                elif t.lane_width == 16:
                    op = (
                        "_mm_min_epi16"
                        if isinstance(expr, SimdMinS)
                        else "_mm_max_epi16"
                    )
                elif t.lane_width == 32:
                    op = (
                        "_mm_min_epi32"
                        if isinstance(expr, SimdMinS)
                        else "_mm_max_epi32"
                    )
                else:
                    raise CodegenError(
                        "sse4.1 min/max signed supports lane_width in {8,16,32}"
                    )
            lines.append(f"  __m128i {name} = {op}({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdZExtLo, SimdSExtLo)):
            x = emit_expr(expr.x)
            src_t = _require_simd128(infer_type(expr.x, types))
            if not isinstance(expr.to, SimdType):
                raise CodegenError("simd extend target must be simd")
            dst_t = expr.to
            if dst_t.total_width != src_t.total_width:
                raise CodegenError("simd extend total_width mismatch")
            if (
                dst_t.lanes != src_t.lanes // 2
                or dst_t.lane_width != src_t.lane_width * 2
            ):
                raise CodegenError("simd extend shape mismatch")
            name = f"t{tmp_id}"
            tmp_id += 1
            if src_t.lane_width == 8 and dst_t.lane_width == 16:
                op = (
                    "_mm_cvtepu8_epi16"
                    if isinstance(expr, SimdZExtLo)
                    else "_mm_cvtepi8_epi16"
                )
            elif src_t.lane_width == 16 and dst_t.lane_width == 32:
                op = (
                    "_mm_cvtepu16_epi32"
                    if isinstance(expr, SimdZExtLo)
                    else "_mm_cvtepi16_epi32"
                )
            elif src_t.lane_width == 32 and dst_t.lane_width == 64:
                op = (
                    "_mm_cvtepu32_epi64"
                    if isinstance(expr, SimdZExtLo)
                    else "_mm_cvtepi32_epi64"
                )
            else:
                raise CodegenError("sse4.1 extend supports 8->16, 16->32, 32->64")
            lines.append(f"  __m128i {name} = {op}({x});")
            memo[key] = name
            return name

        if isinstance(expr, SimdShuffle):
            x = emit_expr(expr.x)
            t = _require_simd128(infer_type(expr.x, types))
            if t.lane_width != 8 or t.lanes != 16 or len(expr.indices) != 16:
                raise CodegenError("ssse3 shuffle requires simd[8,16] with 16 indices")
            if any(i < 0 or i >= 16 for i in expr.indices):
                raise CodegenError("shuffle index out of bounds")
            ctl = f"t{tmp_id}"
            tmp_id += 1
            name = f"t{tmp_id}"
            tmp_id += 1
            args = ", ".join(f"(char){int(i)}" for i in expr.indices)
            lines.append(f"  __m128i {ctl} = _mm_setr_epi8({args});")
            lines.append(f"  __m128i {name} = _mm_shuffle_epi8({x}, {ctl});")
            memo[key] = name
            return name

        if isinstance(expr, SimdBlend):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd128(infer_type(expr.a, types))
            mask_bits = emit_mask(expr.mask)
            lane_width = t.lane_width
            if lane_width not in {8, 16, 32, 64}:
                raise CodegenError("simd_blend supports lane_width in {8,16,32,64}")

            lane_mask = {
                8: 0xFF,
                16: 0xFFFF,
                32: 0xFFFFFFFF,
                64: 0xFFFFFFFFFFFFFFFF,
            }[lane_width]
            mlo = f"t{tmp_id}_mlo"
            mhi = f"t{tmp_id}_mhi"
            mv = f"t{tmp_id}_mv"
            tmp_id += 1
            lines.append(f"  uint64_t {mlo} = 0ull;")
            lines.append(f"  uint64_t {mhi} = 0ull;")
            for i in range(t.lanes):
                fill = f"t{tmp_id}_f{i}"
                tmp_id += 1
                lines.append(
                    f"  uint64_t {fill} = 0ull - (({mask_bits} >> {i}) & 1ull);"
                )
                off = i * lane_width
                if off < 64:
                    lines.append(f"  {mlo} |= ({fill} & {hex(lane_mask)}ull) << {off};")
                else:
                    lines.append(
                        f"  {mhi} |= ({fill} & {hex(lane_mask)}ull) << {off - 64};"
                    )
            maskv = f"t{tmp_id}"
            tmp_id += 1
            lines.append(f"  __m128i {maskv} = load128_imm({mlo}, {mhi});")
            name = f"t{tmp_id}"
            tmp_id += 1
            lines.append(f"  __m128i {name} = _mm_blendv_epi8({a}, {b}, {maskv});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdShl, SimdLShr, SimdAShr)):
            a = emit_expr(expr.a)
            t = _require_simd128(infer_type(expr.a, types))
            sh = _emit_sh_amount(expr.sh, types)
            name = f"t{tmp_id}"
            tmp_id += 1

            if t.lane_width == 16:
                if isinstance(expr, SimdShl):
                    op = "_mm_slli_epi16"
                elif isinstance(expr, SimdLShr):
                    op = "_mm_srli_epi16"
                else:
                    op = "_mm_srai_epi16"
            elif t.lane_width == 32:
                if isinstance(expr, SimdShl):
                    op = "_mm_slli_epi32"
                elif isinstance(expr, SimdLShr):
                    op = "_mm_srli_epi32"
                else:
                    op = "_mm_srai_epi32"
            elif t.lane_width == 64:
                if isinstance(expr, SimdAShr):
                    raise CodegenError("sse2 does not support ashr for 64-bit lanes")
                op = "_mm_slli_epi64" if isinstance(expr, SimdShl) else "_mm_srli_epi64"
            else:
                raise CodegenError("unsupported lane_width for shifts")

            lines.append(f"  __m128i {name} = {op}({a}, {sh});")
            memo[key] = name
            return name

        raise CodegenError("unsupported expression for x86 sse2 backend")

    def emit_mask(expr: Expr) -> str:
        nonlocal tmp_id
        key = f"mask:{repr(expr)}"
        if key in memo:
            return memo[key]

        if not isinstance(expr, (SimdEq, SimdSlt, SimdUlt, SimdMaskPack)):
            raise CodegenError("unsupported mask expression for x86 sse2 backend")

        if isinstance(expr, SimdMaskPack):
            src_t = _require_simd128(infer_type(expr.x, types))
            a = emit_expr(expr.x)
            b = None
            lane_width = src_t.lane_width
        else:
            at = _require_simd128(infer_type(expr.a, types))
            bt = _require_simd128(infer_type(expr.b, types))
            if at.lane_width != bt.lane_width or at.lanes != bt.lanes:
                raise CodegenError("mask compare requires matching simd operands")

            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            lane_width = at.lane_width
        if lane_width not in {8, 16, 32}:
            raise CodegenError("sse2 compare supports lane_width in {8,16,32}")

        cmp_name = f"t{tmp_id}"
        tmp_id += 1
        if isinstance(expr, SimdMaskPack):
            zero = f"t{tmp_id}"
            tmp_id += 1
            eq0 = f"t{tmp_id}"
            tmp_id += 1
            lines.append(f"  __m128i {zero} = _mm_setzero_si128();")
            if lane_width == 8:
                op = "_mm_cmpeq_epi8"
            elif lane_width == 16:
                op = "_mm_cmpeq_epi16"
            else:
                op = "_mm_cmpeq_epi32"
            lines.append(f"  __m128i {eq0} = {op}({a}, {zero});")
            lines.append(
                f"  __m128i {cmp_name} = _mm_xor_si128({eq0}, _mm_set1_epi32(-1));"
            )
        elif isinstance(expr, SimdEq):
            if lane_width == 8:
                op = "_mm_cmpeq_epi8"
            elif lane_width == 16:
                op = "_mm_cmpeq_epi16"
            else:
                op = "_mm_cmpeq_epi32"
            lines.append(f"  __m128i {cmp_name} = {op}({a}, {b});")
        elif isinstance(expr, SimdSlt):
            if lane_width == 8:
                op = "_mm_cmpgt_epi8"
            elif lane_width == 16:
                op = "_mm_cmpgt_epi16"
            else:
                op = "_mm_cmpgt_epi32"
            lines.append(f"  __m128i {cmp_name} = {op}({b}, {a});")
        else:
            if lane_width == 8:
                bias = "_mm_set1_epi8((char)0x80)"
                op = "_mm_cmpgt_epi8"
            elif lane_width == 16:
                bias = "_mm_set1_epi16((short)0x8000)"
                op = "_mm_cmpgt_epi16"
            else:
                bias = "_mm_set1_epi32((int)0x80000000u)"
                op = "_mm_cmpgt_epi32"
            ax = f"t{tmp_id}"
            tmp_id += 1
            bx = f"t{tmp_id}"
            tmp_id += 1
            lines.append(f"  __m128i {ax} = _mm_xor_si128({a}, {bias});")
            lines.append(f"  __m128i {bx} = _mm_xor_si128({b}, {bias});")
            lines.append(f"  __m128i {cmp_name} = {op}({bx}, {ax});")

        mask16 = f"t{tmp_id}"
        tmp_id += 1
        out = f"t{tmp_id}"
        tmp_id += 1
        group = lane_width // 8
        lanes = 16 // group
        lines.append(f"  int {mask16} = _mm_movemask_epi8({cmp_name});")
        lines.append(f"  uint64_t {out} = 0ull;")
        for i in range(lanes):
            lines.append(
                f"  {out} |= (uint64_t)(({mask16} >> {i * group}) & 1) << {i};"
            )

        memo[key] = out
        return out

    lines.append("void stc_eval(const uint64_t* in, uint64_t* out) {")
    for idx, name in enumerate(input_order):
        lines.append(f"  __m128i {_c_ident(name)} = load128(in + {idx * 2});")

    for out_idx, name in enumerate(output_order):
        expr = ir.output_exprs[name]
        out_t = infer_type(expr, types)
        if isinstance(out_t, BitVecType):
            if out_t.width > 128:
                raise CodegenError("x86 sse2 backend supports widths up to 128")
            raise CodegenError("bitvec outputs are not supported yet")
        if isinstance(out_t, SimdType) and out_t.total_width == 128:
            v = emit_expr(expr)
            lines.append(f"  store128(out + {out_idx * 2}, {v});")
        elif isinstance(out_t, SimdType) and out_t.lane_width == 1:
            lanes = out_t.lanes
            if lanes > 64:
                raise CodegenError("mask outputs wider than 64 bits are not supported")
            m = emit_mask(expr)
            lines.append(f"  out[{out_idx * 2}] = {m};")
            lines.append(f"  out[{out_idx * 2 + 1}] = 0ull;")
        else:
            raise CodegenError("unsupported output type for x86 sse2 backend")

    lines.append("}")
    lines.append("")
    return "\n".join(lines)
