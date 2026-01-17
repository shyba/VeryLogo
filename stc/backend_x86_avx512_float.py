from __future__ import annotations

from dataclasses import dataclass

from stc.interp import infer_type
from stc.tick_ir import (
    Bitcast,
    Expr,
    SimdBlend,
    SimdConst,
    SimdFCmpEq,
    SimdFCmpLe,
    SimdFCmpLt,
    SimdFCmpNe,
    SimdFAbs,
    SimdFAdd,
    SimdFDiv,
    SimdFFma,
    SimdFNeg,
    SimdFSqrt,
    SimdFSub,
    SimdFMul,
    SimdType,
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


def _require_simd512_float(t: Type) -> SimdType:
    if not isinstance(t, SimdType):
        raise CodegenError("x86 avx512 float backend requires simd types")
    if t.total_width != 512:
        raise CodegenError("x86 avx512 float backend requires total_width=512")
    if t.lane_width not in {32, 64}:
        raise CodegenError("x86 avx512 float backend requires lane_width in {32,64}")
    return t


def _emit_simd_const(x: SimdConst) -> str:
    words = [(x.value >> (64 * i)) & ((1 << 64) - 1) for i in range(8)]
    return "load512_imm(" + ", ".join(f"{w}ull" for w in words) + ")"


def emit_x86_avx512_float_c(ir: TickIR) -> str:
    validate_tick_ir(ir)
    if ir.state:
        raise CodegenError("x86 avx512 float backend does not support state yet")

    types: dict[str, Type] = dict(ir.inputs)
    for t in ir.inputs.values():
        _require_simd512_float(t)

    input_order = sorted(ir.inputs.keys())
    output_order = sorted(ir.outputs.keys())

    lines: list[str] = []
    lines.append("#include <stdint.h>")
    lines.append("#include <immintrin.h>")
    lines.append("")
    lines.append(
        "typedef union { __m512i vi; __m512 vf; __m512d vd; uint64_t u64[8]; } u512;"
    )
    lines.append("")
    lines.append("static inline __m512i load512i(const uint64_t* p) {")
    lines.append("  u512 x;")
    for i in range(8):
        lines.append(f"  x.u64[{i}] = p[{i}];")
    lines.append("  return x.vi;")
    lines.append("}")
    lines.append("")
    lines.append(
        "static inline __m512i load512_imm(uint64_t w0, uint64_t w1, uint64_t w2, uint64_t w3, uint64_t w4, uint64_t w5, uint64_t w6, uint64_t w7) {"
    )
    lines.append("  u512 x;")
    for i in range(8):
        lines.append(f"  x.u64[{i}] = w{i};")
    lines.append("  return x.vi;")
    lines.append("}")
    lines.append("")
    lines.append("static inline __m512 load512ps(const uint64_t* p) {")
    lines.append("  return _mm512_castsi512_ps(load512i(p));")
    lines.append("}")
    lines.append("")
    lines.append("static inline __m512d load512pd(const uint64_t* p) {")
    lines.append("  return _mm512_castsi512_pd(load512i(p));")
    lines.append("}")
    lines.append("")
    lines.append("static inline void store512i(uint64_t* p, __m512i v) {")
    lines.append("  u512 x;")
    lines.append("  x.vi = v;")
    for i in range(8):
        lines.append(f"  p[{i}] = x.u64[{i}];")
    lines.append("}")
    lines.append("")
    lines.append("static inline void store512ps(uint64_t* p, __m512 v) {")
    lines.append("  store512i(p, _mm512_castps_si512(v));")
    lines.append("}")
    lines.append("")
    lines.append("static inline void store512pd(uint64_t* p, __m512d v) {")
    lines.append("  store512i(p, _mm512_castpd_si512(v));")
    lines.append("}")
    lines.append("")
    lines.append("static inline __m512 canon512_ps(__m512 x) {")
    lines.append("  __mmask16 m = _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q);")
    lines.append("  __m512 c = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FC00000u));")
    lines.append("  return _mm512_mask_mov_ps(x, m, c);")
    lines.append("}")
    lines.append("")
    lines.append("static inline __m512d canon512_pd(__m512d x) {")
    lines.append("  __mmask8 m = _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q);")
    lines.append(
        "  __m512d c = _mm512_castsi512_pd(_mm512_set1_epi64((long long)0x7FF8000000000000ull));"
    )
    lines.append("  return _mm512_mask_mov_pd(x, m, c);")
    lines.append("}")
    lines.append("")

    tmp_id = 0
    memo: dict[str, tuple[str, int]] = {}

    def emit_vec(expr: Expr) -> tuple[str, int]:
        nonlocal tmp_id
        key = repr(expr)
        if key in memo:
            return memo[key]

        if isinstance(expr, Var):
            t = _require_simd512_float(types[expr.name])
            name = _c_ident(expr.name)
            memo[key] = (name, t.lane_width)
            return name, t.lane_width

        if isinstance(expr, SimdConst):
            t = _require_simd512_float(
                SimdType(lane_width=expr.lane_width, lanes=expr.lanes)
            )
            name = f"t{tmp_id}"
            tmp_id += 1
            if t.lane_width == 32:
                lines.append(
                    f"  __m512 {name} = _mm512_castsi512_ps({_emit_simd_const(expr)});"
                )
            else:
                lines.append(
                    f"  __m512d {name} = _mm512_castsi512_pd({_emit_simd_const(expr)});"
                )
            memo[key] = (name, t.lane_width)
            return name, t.lane_width

        if isinstance(expr, Bitcast):
            dst_t = infer_type(expr, types)
            if not isinstance(dst_t, SimdType):
                raise CodegenError(
                    "x86 avx512 float backend supports only simd bitcasts"
                )
            _require_simd512_float(dst_t)
            src_name, src_w = emit_vec(expr.x)
            if dst_t.lane_width != src_w:
                raise CodegenError(
                    "x86 avx512 float backend does not support reinterpreting lane_width"
                )
            memo[key] = (src_name, src_w)
            return src_name, src_w

        if isinstance(expr, (SimdFAdd, SimdFSub, SimdFMul, SimdFDiv)):
            a, aw = emit_vec(expr.a)
            b, bw = emit_vec(expr.b)
            if aw != bw:
                raise CodegenError("simd float op requires matching lane_width")
            name = f"t{tmp_id}"
            tmp_id += 1
            if aw == 32:
                if isinstance(expr, SimdFAdd):
                    op = "_mm512_add_ps"
                elif isinstance(expr, SimdFSub):
                    op = "_mm512_sub_ps"
                elif isinstance(expr, SimdFMul):
                    op = "_mm512_mul_ps"
                else:
                    op = "_mm512_div_ps"
                lines.append(f"  __m512 {name} = {op}({a}, {b});")
                lines.append(f"  {name} = canon512_ps({name});")
            else:
                if isinstance(expr, SimdFAdd):
                    op = "_mm512_add_pd"
                elif isinstance(expr, SimdFSub):
                    op = "_mm512_sub_pd"
                elif isinstance(expr, SimdFMul):
                    op = "_mm512_mul_pd"
                else:
                    op = "_mm512_div_pd"
                lines.append(f"  __m512d {name} = {op}({a}, {b});")
                lines.append(f"  {name} = canon512_pd({name});")
            memo[key] = (name, aw)
            return name, aw

        if isinstance(expr, SimdFFma):
            a, aw = emit_vec(expr.a)
            b, bw = emit_vec(expr.b)
            c, cw = emit_vec(expr.c)
            if aw != bw or aw != cw:
                raise CodegenError("simd fma requires matching lane_width")
            name = f"t{tmp_id}"
            tmp_id += 1
            if aw == 32:
                lines.append(f"  __m512 {name} = _mm512_fmadd_ps({a}, {b}, {c});")
                lines.append(f"  {name} = canon512_ps({name});")
            else:
                lines.append(f"  __m512d {name} = _mm512_fmadd_pd({a}, {b}, {c});")
                lines.append(f"  {name} = canon512_pd({name});")
            memo[key] = (name, aw)
            return name, aw

        if isinstance(expr, SimdFSqrt):
            x, w = emit_vec(expr.x)
            name = f"t{tmp_id}"
            tmp_id += 1
            if w == 32:
                lines.append(f"  __m512 {name} = _mm512_sqrt_ps({x});")
                lines.append(f"  {name} = canon512_ps({name});")
            else:
                lines.append(f"  __m512d {name} = _mm512_sqrt_pd({x});")
                lines.append(f"  {name} = canon512_pd({name});")
            memo[key] = (name, w)
            return name, w

        if isinstance(expr, (SimdFNeg, SimdFAbs)):
            x, w = emit_vec(expr.x)
            name = f"t{tmp_id}"
            tmp_id += 1
            if w == 32:
                maski = f"t{tmp_id}_maski"
                xi = f"t{tmp_id}_xi"
                yi = f"t{tmp_id}_yi"
                tmp_id += 1
                if isinstance(expr, SimdFNeg):
                    lines.append(f"  __m512i {maski} = _mm512_set1_epi32(0x80000000u);")
                    lines.append(f"  __m512i {xi} = _mm512_castps_si512({x});")
                    lines.append(f"  __m512i {yi} = _mm512_xor_si512({xi}, {maski});")
                else:
                    lines.append(f"  __m512i {maski} = _mm512_set1_epi32(0x7FFFFFFFu);")
                    lines.append(f"  __m512i {xi} = _mm512_castps_si512({x});")
                    lines.append(f"  __m512i {yi} = _mm512_and_si512({xi}, {maski});")
                lines.append(f"  __m512 {name} = _mm512_castsi512_ps({yi});")
                lines.append(f"  {name} = canon512_ps({name});")
            else:
                maski = f"t{tmp_id}_maski"
                xi = f"t{tmp_id}_xi"
                yi = f"t{tmp_id}_yi"
                tmp_id += 1
                if isinstance(expr, SimdFNeg):
                    lines.append(
                        f"  __m512i {maski} = _mm512_set1_epi64((long long)0x8000000000000000ull);"
                    )
                    lines.append(f"  __m512i {xi} = _mm512_castpd_si512({x});")
                    lines.append(f"  __m512i {yi} = _mm512_xor_si512({xi}, {maski});")
                else:
                    lines.append(
                        f"  __m512i {maski} = _mm512_set1_epi64((long long)0x7FFFFFFFFFFFFFFFull);"
                    )
                    lines.append(f"  __m512i {xi} = _mm512_castpd_si512({x});")
                    lines.append(f"  __m512i {yi} = _mm512_and_si512({xi}, {maski});")
                lines.append(f"  __m512d {name} = _mm512_castsi512_pd({yi});")
                lines.append(f"  {name} = canon512_pd({name});")
            memo[key] = (name, w)
            return name, w

        if isinstance(expr, SimdBlend):
            a, aw = emit_vec(expr.a)
            b, bw = emit_vec(expr.b)
            if aw != bw:
                raise CodegenError("simd_blend requires matching lane_width")
            mask_name = emit_mask(expr.mask)
            name = f"t{tmp_id}"
            tmp_id += 1
            if aw == 32:
                lines.append(
                    f"  __m512 {name} = _mm512_mask_blend_ps({mask_name}, {a}, {b});"
                )
            else:
                lines.append(
                    f"  __m512d {name} = _mm512_mask_blend_pd({mask_name}, {a}, {b});"
                )
            memo[key] = (name, aw)
            return name, aw

        raise CodegenError("unsupported expression for x86 avx512 float backend")

    def emit_mask(expr: Expr) -> str:
        nonlocal tmp_id
        key = f"mask:{repr(expr)}"
        if key in memo:
            return memo[key][0]

        if not isinstance(expr, (SimdFCmpEq, SimdFCmpLt, SimdFCmpLe, SimdFCmpNe)):
            raise CodegenError(
                "unsupported mask expression for x86 avx512 float backend"
            )

        a, aw = emit_vec(expr.a)
        b, bw = emit_vec(expr.b)
        if aw != bw:
            raise CodegenError("simd float cmp requires matching lane_width")

        name = f"t{tmp_id}"
        tmp_id += 1
        if aw == 32:
            pred = (
                "_CMP_EQ_OQ"
                if isinstance(expr, SimdFCmpEq)
                else (
                    "_CMP_LT_OQ"
                    if isinstance(expr, SimdFCmpLt)
                    else "_CMP_LE_OQ" if isinstance(expr, SimdFCmpLe) else "_CMP_NEQ_UQ"
                )
            )
            lines.append(f"  __mmask16 {name} = _mm512_cmp_ps_mask({a}, {b}, {pred});")
        else:
            pred = (
                "_CMP_EQ_OQ"
                if isinstance(expr, SimdFCmpEq)
                else (
                    "_CMP_LT_OQ"
                    if isinstance(expr, SimdFCmpLt)
                    else "_CMP_LE_OQ" if isinstance(expr, SimdFCmpLe) else "_CMP_NEQ_UQ"
                )
            )
            lines.append(f"  __mmask8 {name} = _mm512_cmp_pd_mask({a}, {b}, {pred});")

        memo[key] = (name, 0)
        return name

    lines.append("void stc_eval(const uint64_t* in, uint64_t* out) {")
    for idx, name in enumerate(input_order):
        t = _require_simd512_float(ir.inputs[name])
        if t.lane_width == 32:
            lines.append(f"  __m512 {_c_ident(name)} = load512ps(in + {idx * 8});")
        else:
            lines.append(f"  __m512d {_c_ident(name)} = load512pd(in + {idx * 8});")

    for out_idx, name in enumerate(output_order):
        expr = ir.output_exprs[name]
        out_t = infer_type(expr, types)
        if (
            isinstance(out_t, SimdType)
            and out_t.total_width == 512
            and out_t.lane_width
            in {
                32,
                64,
            }
        ):
            v, vw = emit_vec(expr)
            if vw == 32:
                lines.append(f"  store512ps(out + {out_idx * 8}, {v});")
            else:
                lines.append(f"  store512pd(out + {out_idx * 8}, {v});")
        elif isinstance(out_t, SimdType) and out_t.lane_width == 1:
            lanes = out_t.lanes
            if lanes > 64:
                raise CodegenError("mask outputs wider than 64 bits are not supported")
            m = emit_mask(expr)
            lines.append(f"  out[{out_idx * 8 + 0}] = (uint64_t){m};")
            for i in range(1, 8):
                lines.append(f"  out[{out_idx * 8 + i}] = 0ull;")
        else:
            raise CodegenError("unsupported output type for x86 avx512 float backend")

    lines.append("}")
    lines.append("")
    return "\n".join(lines)
