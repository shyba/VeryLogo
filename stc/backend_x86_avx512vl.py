from __future__ import annotations

from dataclasses import dataclass

from stc.interp import infer_type
from stc.tick_ir import (
    BitVecType,
    Bitcast,
    Expr,
    SimdAdd,
    SimdAddMasked,
    SimdAnd,
    SimdBlend,
    SimdConst,
    SimdEq,
    SimdMaskPack,
    SimdNot,
    SimdOr,
    SimdSub,
    SimdSubMasked,
    SimdType,
    SimdUlt,
    TickIR,
    Type,
    Var,
    SimdXor,
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


def _require_simd(t: Type, *, total_width: int) -> SimdType:
    if not isinstance(t, SimdType):
        raise CodegenError("x86 avx512vl backend requires simd types")
    if t.total_width != total_width:
        raise CodegenError(f"x86 avx512vl backend requires total_width={total_width}")
    return t


def _emit_simd_const(x: SimdConst, *, total_width: int) -> str:
    if total_width == 256:
        words = [(x.value >> (64 * i)) & ((1 << 64) - 1) for i in range(4)]
        return "load256_imm(" + ", ".join(f"{w}ull" for w in words) + ")"
    if total_width == 128:
        lo = x.value & ((1 << 64) - 1)
        hi = (x.value >> 64) & ((1 << 64) - 1)
        return f"load128_imm({lo}ull, {hi}ull)"
    raise CodegenError("unsupported total_width for simd const")


def emit_x86_avx512vl_c(ir: TickIR) -> str:
    validate_tick_ir(ir)
    if ir.state:
        raise CodegenError("x86 avx512vl backend does not support state yet")

    if not ir.inputs:
        raise CodegenError("x86 avx512vl backend requires inputs")

    widths = {t.total_width for t in ir.inputs.values() if isinstance(t, SimdType)}
    if len(widths) != 1:
        raise CodegenError("x86 avx512vl backend requires uniform simd total_width")
    (w,) = tuple(widths)
    if w not in {128, 256}:
        raise CodegenError("x86 avx512vl backend supports total_width in {128,256}")

    types: dict[str, Type] = dict(ir.inputs)
    for t in ir.inputs.values():
        _require_simd(t, total_width=w)

    input_order = sorted(ir.inputs.keys())
    output_order = sorted(ir.outputs.keys())

    vec_t = "__m256i" if w == 256 else "__m128i"
    words_per_vec = 4 if w == 256 else 2

    lines: list[str] = []
    lines.append("#include <stdint.h>")
    lines.append("#include <immintrin.h>")
    lines.append("")

    if w == 256:
        lines.append("typedef union { __m256i v; uint64_t u64[4]; } u256;")
        lines.append("")
        lines.append("static inline __m256i load256(const uint64_t* p) {")
        lines.append("  u256 x;")
        for i in range(4):
            lines.append(f"  x.u64[{i}] = p[{i}];")
        lines.append("  return x.v;")
        lines.append("}")
        lines.append("")
        lines.append(
            "static inline __m256i load256_imm(uint64_t w0, uint64_t w1, uint64_t w2, uint64_t w3) {"
        )
        lines.append("  u256 x;")
        for i in range(4):
            lines.append(f"  x.u64[{i}] = w{i};")
        lines.append("  return x.v;")
        lines.append("}")
        lines.append("")
        lines.append("static inline void store256(uint64_t* p, __m256i v) {")
        lines.append("  u256 x;")
        lines.append("  x.v = v;")
        for i in range(4):
            lines.append(f"  p[{i}] = x.u64[{i}];")
        lines.append("}")
    else:
        lines.append("typedef union { __m128i v; uint64_t u64[2]; } u128;")
        lines.append("")
        lines.append("static inline __m128i load128(const uint64_t* p) {")
        lines.append("  u128 x;")
        for i in range(2):
            lines.append(f"  x.u64[{i}] = p[{i}];")
        lines.append("  return x.v;")
        lines.append("}")
        lines.append("")
        lines.append("static inline __m128i load128_imm(uint64_t w0, uint64_t w1) {")
        lines.append("  u128 x;")
        for i in range(2):
            lines.append(f"  x.u64[{i}] = w{i};")
        lines.append("  return x.v;")
        lines.append("}")
        lines.append("")
        lines.append("static inline void store128(uint64_t* p, __m128i v) {")
        lines.append("  u128 x;")
        lines.append("  x.v = v;")
        for i in range(2):
            lines.append(f"  p[{i}] = x.u64[{i}];")
        lines.append("}")

    lines.append("")

    tmp_id = 0
    memo: dict[str, str] = {}

    def emit_mask(expr: Expr) -> str:
        nonlocal tmp_id
        key = f"mask:{repr(expr)}"
        if key in memo:
            return memo[key]

        if not isinstance(expr, (SimdEq, SimdUlt, SimdMaskPack)):
            raise CodegenError("unsupported mask expression for x86 avx512vl backend")

        if isinstance(expr, SimdMaskPack):
            src_t = infer_type(expr.x, types)
            src_t = _require_simd(src_t, total_width=w)
            a = emit_expr(expr.x)
            z = "_mm256_setzero_si256()" if w == 256 else "_mm_setzero_si128()"
            name = f"t{tmp_id}"
            tmp_id += 1
            if w == 256:
                if src_t.lane_width == 8:
                    lines.append(
                        f"  __mmask32 {name} = _mm256_cmpneq_epi8_mask({a}, {z});"
                    )
                elif src_t.lane_width == 16:
                    lines.append(
                        f"  __mmask16 {name} = _mm256_cmpneq_epi16_mask({a}, {z});"
                    )
                elif src_t.lane_width == 32:
                    lines.append(
                        f"  __mmask8 {name} = _mm256_cmpneq_epi32_mask({a}, {z});"
                    )
                elif src_t.lane_width == 64:
                    lines.append(
                        f"  __mmask8 {name} = _mm256_cmpneq_epi64_mask({a}, {z});"
                    )
                else:
                    raise CodegenError(
                        "simd_mask_pack supports lane_width in {8,16,32,64}"
                    )
            else:
                if src_t.lane_width == 8:
                    lines.append(
                        f"  __mmask16 {name} = _mm_cmpneq_epi8_mask({a}, {z});"
                    )
                elif src_t.lane_width == 16:
                    lines.append(
                        f"  __mmask8 {name} = _mm_cmpneq_epi16_mask({a}, {z});"
                    )
                elif src_t.lane_width == 32:
                    lines.append(
                        f"  __mmask8 {name} = _mm_cmpneq_epi32_mask({a}, {z});"
                    )
                elif src_t.lane_width == 64:
                    lines.append(
                        f"  __mmask8 {name} = _mm_cmpneq_epi64_mask({a}, {z});"
                    )
                else:
                    raise CodegenError(
                        "simd_mask_pack supports lane_width in {8,16,32,64}"
                    )

            memo[key] = name
            return name

        at = infer_type(expr.a, types)
        bt = infer_type(expr.b, types)
        at = _require_simd(at, total_width=w)
        bt = _require_simd(bt, total_width=w)
        if at.lane_width != bt.lane_width or at.lanes != bt.lanes:
            raise CodegenError("mask compare requires matching simd operands")

        a = emit_expr(expr.a)
        b = emit_expr(expr.b)
        name = f"t{tmp_id}"
        tmp_id += 1

        if w == 256:
            if at.lane_width == 8:
                op = (
                    "_mm256_cmpeq_epi8_mask"
                    if isinstance(expr, SimdEq)
                    else "_mm256_cmplt_epu8_mask"
                )
                lines.append(f"  __mmask32 {name} = {op}({a}, {b});")
            elif at.lane_width == 16:
                op = (
                    "_mm256_cmpeq_epi16_mask"
                    if isinstance(expr, SimdEq)
                    else "_mm256_cmplt_epu16_mask"
                )
                lines.append(f"  __mmask16 {name} = {op}({a}, {b});")
            elif at.lane_width == 32:
                op = (
                    "_mm256_cmpeq_epi32_mask"
                    if isinstance(expr, SimdEq)
                    else "_mm256_cmplt_epu32_mask"
                )
                lines.append(f"  __mmask8 {name} = {op}({a}, {b});")
            elif at.lane_width == 64:
                op = (
                    "_mm256_cmpeq_epi64_mask"
                    if isinstance(expr, SimdEq)
                    else "_mm256_cmplt_epu64_mask"
                )
                lines.append(f"  __mmask8 {name} = {op}({a}, {b});")
            else:
                raise CodegenError("mask compare supports lane_width in {8,16,32,64}")
        else:
            if at.lane_width == 8:
                op = (
                    "_mm_cmpeq_epi8_mask"
                    if isinstance(expr, SimdEq)
                    else "_mm_cmplt_epu8_mask"
                )
                lines.append(f"  __mmask16 {name} = {op}({a}, {b});")
            elif at.lane_width == 16:
                op = (
                    "_mm_cmpeq_epi16_mask"
                    if isinstance(expr, SimdEq)
                    else "_mm_cmplt_epu16_mask"
                )
                lines.append(f"  __mmask8 {name} = {op}({a}, {b});")
            elif at.lane_width == 32:
                op = (
                    "_mm_cmpeq_epi32_mask"
                    if isinstance(expr, SimdEq)
                    else "_mm_cmplt_epu32_mask"
                )
                lines.append(f"  __mmask8 {name} = {op}({a}, {b});")
            elif at.lane_width == 64:
                op = (
                    "_mm_cmpeq_epi64_mask"
                    if isinstance(expr, SimdEq)
                    else "_mm_cmplt_epu64_mask"
                )
                lines.append(f"  __mmask8 {name} = {op}({a}, {b});")
            else:
                raise CodegenError("mask compare supports lane_width in {8,16,32,64}")

        memo[key] = name
        return name

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
            lines.append(f"  {vec_t} {name} = {_emit_simd_const(expr, total_width=w)};")
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
            all1 = "_mm256_set1_epi32(-1)" if w == 256 else "_mm_set1_epi32(-1)"
            xor = "_mm256_xor_si256" if w == 256 else "_mm_xor_si128"
            lines.append(f"  {vec_t} {name} = {xor}({x}, {all1});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdAnd, SimdOr, SimdXor)):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            name = f"t{tmp_id}"
            tmp_id += 1
            if w == 256:
                if isinstance(expr, SimdAnd):
                    op = "_mm256_and_si256"
                elif isinstance(expr, SimdOr):
                    op = "_mm256_or_si256"
                else:
                    op = "_mm256_xor_si256"
            else:
                if isinstance(expr, SimdAnd):
                    op = "_mm_and_si128"
                elif isinstance(expr, SimdOr):
                    op = "_mm_or_si128"
                else:
                    op = "_mm_xor_si128"
            lines.append(f"  {vec_t} {name} = {op}({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdAdd, SimdSub)):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd(infer_type(expr.a, types), total_width=w)
            name = f"t{tmp_id}"
            tmp_id += 1
            if w == 256:
                if t.lane_width == 8:
                    op = (
                        "_mm256_add_epi8"
                        if isinstance(expr, SimdAdd)
                        else "_mm256_sub_epi8"
                    )
                elif t.lane_width == 16:
                    op = (
                        "_mm256_add_epi16"
                        if isinstance(expr, SimdAdd)
                        else "_mm256_sub_epi16"
                    )
                elif t.lane_width == 32:
                    op = (
                        "_mm256_add_epi32"
                        if isinstance(expr, SimdAdd)
                        else "_mm256_sub_epi32"
                    )
                elif t.lane_width == 64:
                    op = (
                        "_mm256_add_epi64"
                        if isinstance(expr, SimdAdd)
                        else "_mm256_sub_epi64"
                    )
                else:
                    raise CodegenError("add/sub supports lane_width in {8,16,32,64}")
            else:
                if t.lane_width == 8:
                    op = "_mm_add_epi8" if isinstance(expr, SimdAdd) else "_mm_sub_epi8"
                elif t.lane_width == 16:
                    op = (
                        "_mm_add_epi16"
                        if isinstance(expr, SimdAdd)
                        else "_mm_sub_epi16"
                    )
                elif t.lane_width == 32:
                    op = (
                        "_mm_add_epi32"
                        if isinstance(expr, SimdAdd)
                        else "_mm_sub_epi32"
                    )
                elif t.lane_width == 64:
                    op = (
                        "_mm_add_epi64"
                        if isinstance(expr, SimdAdd)
                        else "_mm_sub_epi64"
                    )
                else:
                    raise CodegenError("add/sub supports lane_width in {8,16,32,64}")
            lines.append(f"  {vec_t} {name} = {op}({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdAddMasked, SimdSubMasked)):
            p = emit_expr(expr.passthru)
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd(infer_type(expr.a, types), total_width=w)
            m = emit_mask(expr.mask)
            name = f"t{tmp_id}"
            tmp_id += 1
            if w == 256:
                if isinstance(expr, SimdAddMasked):
                    if t.lane_width == 8:
                        op = "_mm256_mask_add_epi8"
                    elif t.lane_width == 16:
                        op = "_mm256_mask_add_epi16"
                    elif t.lane_width == 32:
                        op = "_mm256_mask_add_epi32"
                    elif t.lane_width == 64:
                        op = "_mm256_mask_add_epi64"
                    else:
                        raise CodegenError(
                            "simd_add_masked supports lane_width in {8,16,32,64}"
                        )
                else:
                    if t.lane_width == 8:
                        op = "_mm256_mask_sub_epi8"
                    elif t.lane_width == 16:
                        op = "_mm256_mask_sub_epi16"
                    elif t.lane_width == 32:
                        op = "_mm256_mask_sub_epi32"
                    elif t.lane_width == 64:
                        op = "_mm256_mask_sub_epi64"
                    else:
                        raise CodegenError(
                            "simd_sub_masked supports lane_width in {8,16,32,64}"
                        )
            else:
                if isinstance(expr, SimdAddMasked):
                    if t.lane_width == 8:
                        op = "_mm_mask_add_epi8"
                    elif t.lane_width == 16:
                        op = "_mm_mask_add_epi16"
                    elif t.lane_width == 32:
                        op = "_mm_mask_add_epi32"
                    elif t.lane_width == 64:
                        op = "_mm_mask_add_epi64"
                    else:
                        raise CodegenError(
                            "simd_add_masked supports lane_width in {8,16,32,64}"
                        )
                else:
                    if t.lane_width == 8:
                        op = "_mm_mask_sub_epi8"
                    elif t.lane_width == 16:
                        op = "_mm_mask_sub_epi16"
                    elif t.lane_width == 32:
                        op = "_mm_mask_sub_epi32"
                    elif t.lane_width == 64:
                        op = "_mm_mask_sub_epi64"
                    else:
                        raise CodegenError(
                            "simd_sub_masked supports lane_width in {8,16,32,64}"
                        )
            lines.append(f"  {vec_t} {name} = {op}({p}, {m}, {a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, SimdBlend):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd(infer_type(expr.a, types), total_width=w)
            m = emit_mask(expr.mask)
            name = f"t{tmp_id}"
            tmp_id += 1
            if w == 256:
                if t.lane_width == 8:
                    op = "_mm256_mask_blend_epi8"
                elif t.lane_width == 16:
                    op = "_mm256_mask_blend_epi16"
                elif t.lane_width == 32:
                    op = "_mm256_mask_blend_epi32"
                elif t.lane_width == 64:
                    op = "_mm256_mask_blend_epi64"
                else:
                    raise CodegenError("simd_blend supports lane_width in {8,16,32,64}")
            else:
                if t.lane_width == 8:
                    op = "_mm_mask_blend_epi8"
                elif t.lane_width == 16:
                    op = "_mm_mask_blend_epi16"
                elif t.lane_width == 32:
                    op = "_mm_mask_blend_epi32"
                elif t.lane_width == 64:
                    op = "_mm_mask_blend_epi64"
                else:
                    raise CodegenError("simd_blend supports lane_width in {8,16,32,64}")
            lines.append(f"  {vec_t} {name} = {op}({m}, {a}, {b});")
            memo[key] = name
            return name

        raise CodegenError("unsupported expression for x86 avx512vl backend")

    lines.append("void stc_eval(const uint64_t* in, uint64_t* out) {")
    for idx, name in enumerate(input_order):
        load = "load256" if w == 256 else "load128"
        lines.append(
            f"  {vec_t} {_c_ident(name)} = {load}(in + {idx * words_per_vec});"
        )

    for out_idx, name in enumerate(output_order):
        expr = ir.output_exprs[name]
        out_t = infer_type(expr, types)
        if isinstance(out_t, BitVecType):
            raise CodegenError("bitvec outputs are not supported yet")
        if isinstance(out_t, SimdType) and out_t.total_width == w:
            v = emit_expr(expr)
            store = "store256" if w == 256 else "store128"
            lines.append(f"  {store}(out + {out_idx * words_per_vec}, {v});")
        elif isinstance(out_t, SimdType) and out_t.lane_width == 1:
            lanes = out_t.lanes
            if lanes > 64:
                raise CodegenError("mask outputs wider than 64 bits are not supported")
            m = emit_mask(expr)
            lines.append(f"  out[{out_idx * words_per_vec + 0}] = (uint64_t){m};")
            for i in range(1, words_per_vec):
                lines.append(f"  out[{out_idx * words_per_vec + i}] = 0ull;")
        else:
            raise CodegenError("unsupported output type for x86 avx512vl backend")

    lines.append("}")
    lines.append("")
    return "\n".join(lines)
