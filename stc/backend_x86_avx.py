from __future__ import annotations

from dataclasses import dataclass

from stc.interp import infer_type
from stc.tick_ir import (
    Bitcast,
    Expr,
    SimdConst,
    SimdFCmpEq,
    SimdFCmpLe,
    SimdFCmpLt,
    SimdFCmpNe,
    SimdFAdd,
    SimdFAbs,
    SimdFDiv,
    SimdFFma,
    SimdFNeg,
    SimdFSqrt,
    SimdFSub,
    SimdFMul,
    SimdBlend,
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


def _require_simd256_float(t: Type) -> SimdType:
    if not isinstance(t, SimdType):
        raise CodegenError("x86 avx backend requires simd types")
    if t.total_width != 256:
        raise CodegenError("x86 avx backend requires total_width=256")
    if t.lane_width not in {32, 64}:
        raise CodegenError("x86 avx backend requires lane_width in {32,64}")
    if t.lanes * t.lane_width != 256:
        raise CodegenError("x86 avx backend requires full 256-bit vectors")
    return t


def _emit_simd_const(x: SimdConst) -> str:
    words = [(x.value >> (64 * i)) & ((1 << 64) - 1) for i in range(4)]
    return f"load256_imm({words[0]}ull, {words[1]}ull, {words[2]}ull, {words[3]}ull)"


def emit_x86_avx_c(ir: TickIR) -> str:
    validate_tick_ir(ir)
    if ir.state:
        raise CodegenError("x86 avx backend does not support state yet")

    types: dict[str, Type] = dict(ir.inputs)
    for t in ir.inputs.values():
        _require_simd256_float(t)

    input_order = sorted(ir.inputs.keys())
    output_order = sorted(ir.outputs.keys())

    lines: list[str] = []
    lines.append("#include <stdint.h>")
    lines.append("#include <immintrin.h>")
    lines.append("")
    lines.append(
        "typedef union { __m256i vi; __m256 vf; __m256d vd; uint64_t u64[4]; } u256;"
    )
    lines.append("")
    lines.append("static inline __m256i load256i(const uint64_t* p) {")
    lines.append("  u256 x;")
    lines.append("  x.u64[0] = p[0];")
    lines.append("  x.u64[1] = p[1];")
    lines.append("  x.u64[2] = p[2];")
    lines.append("  x.u64[3] = p[3];")
    lines.append("  return x.vi;")
    lines.append("}")
    lines.append("")
    lines.append(
        "static inline __m256i load256_imm(uint64_t w0, uint64_t w1, uint64_t w2, uint64_t w3) {"
    )
    lines.append("  u256 x;")
    lines.append("  x.u64[0] = w0;")
    lines.append("  x.u64[1] = w1;")
    lines.append("  x.u64[2] = w2;")
    lines.append("  x.u64[3] = w3;")
    lines.append("  return x.vi;")
    lines.append("}")
    lines.append("")
    lines.append("static inline __m256 load256ps(const uint64_t* p) {")
    lines.append("  return _mm256_castsi256_ps(load256i(p));")
    lines.append("}")
    lines.append("")
    lines.append("static inline __m256d load256pd(const uint64_t* p) {")
    lines.append("  return _mm256_castsi256_pd(load256i(p));")
    lines.append("}")
    lines.append("")
    lines.append("static inline void store256i(uint64_t* p, __m256i v) {")
    lines.append("  u256 x;")
    lines.append("  x.vi = v;")
    lines.append("  p[0] = x.u64[0];")
    lines.append("  p[1] = x.u64[1];")
    lines.append("  p[2] = x.u64[2];")
    lines.append("  p[3] = x.u64[3];")
    lines.append("}")
    lines.append("")
    lines.append("static inline void store256ps(uint64_t* p, __m256 v) {")
    lines.append("  store256i(p, _mm256_castps_si256(v));")
    lines.append("}")
    lines.append("")
    lines.append("static inline void store256pd(uint64_t* p, __m256d v) {")
    lines.append("  store256i(p, _mm256_castpd_si256(v));")
    lines.append("}")
    lines.append("")
    lines.append("static inline __m256 canon256_ps(__m256 x) {")
    lines.append("  __m256 m = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);")
    lines.append("  __m256 c = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FC00000u));")
    lines.append("  return _mm256_blendv_ps(x, c, m);")
    lines.append("}")
    lines.append("")
    lines.append("static inline __m256d canon256_pd(__m256d x) {")
    lines.append("  __m256d m = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);")
    lines.append(
        "  __m256d c = _mm256_castsi256_pd(_mm256_set1_epi64x((long long)0x7FF8000000000000ull));"
    )
    lines.append("  return _mm256_blendv_pd(x, c, m);")
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
            t = _require_simd256_float(types[expr.name])
            name = _c_ident(expr.name)
            memo[key] = (name, t.lane_width)
            return name, t.lane_width

        if isinstance(expr, SimdConst):
            t = _require_simd256_float(
                SimdType(lane_width=expr.lane_width, lanes=expr.lanes)
            )
            name = f"t{tmp_id}"
            tmp_id += 1
            if t.lane_width == 32:
                lines.append(
                    f"  __m256 {name} = _mm256_castsi256_ps({_emit_simd_const(expr)});"
                )
            else:
                lines.append(
                    f"  __m256d {name} = _mm256_castsi256_pd({_emit_simd_const(expr)});"
                )
            memo[key] = (name, t.lane_width)
            return name, t.lane_width

        if isinstance(expr, Bitcast):
            dst_t = infer_type(expr, types)
            if not isinstance(dst_t, SimdType):
                raise CodegenError("x86 avx backend supports only simd bitcasts")
            _require_simd256_float(dst_t)
            src_name, src_w = emit_vec(expr.x)
            if dst_t.lane_width != src_w:
                raise CodegenError(
                    "x86 avx backend does not support reinterpreting lane_width"
                )
            memo[key] = (src_name, src_w)
            return src_name, src_w

        if isinstance(expr, (SimdFAdd, SimdFMul)):
            a, aw = emit_vec(expr.a)
            b, bw = emit_vec(expr.b)
            if aw != bw:
                raise CodegenError("simd float op requires matching lane_width")
            name = f"t{tmp_id}"
            tmp_id += 1
            if aw == 32:
                op = "_mm256_add_ps" if isinstance(expr, SimdFAdd) else "_mm256_mul_ps"
                lines.append(f"  __m256 {name} = {op}({a}, {b});")
                lines.append(f"  {name} = canon256_ps({name});")
            else:
                op = "_mm256_add_pd" if isinstance(expr, SimdFAdd) else "_mm256_mul_pd"
                lines.append(f"  __m256d {name} = {op}({a}, {b});")
                lines.append(f"  {name} = canon256_pd({name});")
            memo[key] = (name, aw)
            return name, aw

        if isinstance(expr, (SimdFSub, SimdFDiv)):
            a, aw = emit_vec(expr.a)
            b, bw = emit_vec(expr.b)
            if aw != bw:
                raise CodegenError("simd float op requires matching lane_width")
            name = f"t{tmp_id}"
            tmp_id += 1
            if aw == 32:
                op = "_mm256_sub_ps" if isinstance(expr, SimdFSub) else "_mm256_div_ps"
                lines.append(f"  __m256 {name} = {op}({a}, {b});")
                lines.append(f"  {name} = canon256_ps({name});")
            else:
                op = "_mm256_sub_pd" if isinstance(expr, SimdFSub) else "_mm256_div_pd"
                lines.append(f"  __m256d {name} = {op}({a}, {b});")
                lines.append(f"  {name} = canon256_pd({name});")
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
                lines.append(f"  __m256 {name} = _mm256_fmadd_ps({a}, {b}, {c});")
                lines.append(f"  {name} = canon256_ps({name});")
            else:
                lines.append(f"  __m256d {name} = _mm256_fmadd_pd({a}, {b}, {c});")
                lines.append(f"  {name} = canon256_pd({name});")
            memo[key] = (name, aw)
            return name, aw

        if isinstance(expr, SimdFSqrt):
            x, w = emit_vec(expr.x)
            name = f"t{tmp_id}"
            tmp_id += 1
            if w == 32:
                lines.append(f"  __m256 {name} = _mm256_sqrt_ps({x});")
                lines.append(f"  {name} = canon256_ps({name});")
            else:
                lines.append(f"  __m256d {name} = _mm256_sqrt_pd({x});")
                lines.append(f"  {name} = canon256_pd({name});")
            memo[key] = (name, w)
            return name, w

        if isinstance(expr, (SimdFNeg, SimdFAbs)):
            x, w = emit_vec(expr.x)
            name = f"t{tmp_id}"
            tmp_id += 1
            if w == 32:
                sign = f"t{tmp_id}_sign"
                tmp_id += 1
                if isinstance(expr, SimdFNeg):
                    lines.append(
                        f"  __m256 {sign} = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000u));"
                    )
                    lines.append(f"  __m256 {name} = _mm256_xor_ps({x}, {sign});")
                    lines.append(f"  {name} = canon256_ps({name});")
                else:
                    lines.append(
                        f"  __m256 {sign} = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFFu));"
                    )
                    lines.append(f"  __m256 {name} = _mm256_and_ps({x}, {sign});")
                    lines.append(f"  {name} = canon256_ps({name});")
            else:
                sign = f"t{tmp_id}_sign"
                tmp_id += 1
                if isinstance(expr, SimdFNeg):
                    lines.append(
                        f"  __m256d {sign} = _mm256_castsi256_pd(_mm256_set1_epi64x((long long)0x8000000000000000ull));"
                    )
                    lines.append(f"  __m256d {name} = _mm256_xor_pd({x}, {sign});")
                    lines.append(f"  {name} = canon256_pd({name});")
                else:
                    lines.append(
                        f"  __m256d {sign} = _mm256_castsi256_pd(_mm256_set1_epi64x((long long)0x7FFFFFFFFFFFFFFFull));"
                    )
                    lines.append(f"  __m256d {name} = _mm256_and_pd({x}, {sign});")
                    lines.append(f"  {name} = canon256_pd({name});")
            memo[key] = (name, w)
            return name, w

        if isinstance(expr, SimdBlend):
            a, aw = emit_vec(expr.a)
            b, bw = emit_vec(expr.b)
            if aw != bw:
                raise CodegenError("simd_blend requires matching lane_width")
            mask_bits = emit_mask(expr.mask)

            lane_width = aw
            lanes = 256 // lane_width
            lane_mask = 0xFFFFFFFFFFFFFFFF if lane_width == 64 else 0xFFFFFFFF

            mw0 = f"t{tmp_id}_mw0"
            mw1 = f"t{tmp_id}_mw1"
            mw2 = f"t{tmp_id}_mw2"
            mw3 = f"t{tmp_id}_mw3"
            tmp_id += 1
            lines.append(f"  uint64_t {mw0} = 0ull;")
            lines.append(f"  uint64_t {mw1} = 0ull;")
            lines.append(f"  uint64_t {mw2} = 0ull;")
            lines.append(f"  uint64_t {mw3} = 0ull;")
            for i in range(lanes):
                fill = f"t{tmp_id}_f{i}"
                tmp_id += 1
                lines.append(
                    f"  uint64_t {fill} = 0ull - (({mask_bits} >> {i}) & 1ull);"
                )
                off = i * lane_width
                word = off // 64
                sh0 = off % 64
                if word == 0:
                    lines.append(f"  {mw0} |= ({fill} & {hex(lane_mask)}ull) << {sh0};")
                elif word == 1:
                    lines.append(f"  {mw1} |= ({fill} & {hex(lane_mask)}ull) << {sh0};")
                elif word == 2:
                    lines.append(f"  {mw2} |= ({fill} & {hex(lane_mask)}ull) << {sh0};")
                else:
                    lines.append(f"  {mw3} |= ({fill} & {hex(lane_mask)}ull) << {sh0};")

            maski = f"t{tmp_id}_maski"
            tmp_id += 1
            lines.append(
                f"  __m256i {maski} = load256_imm({mw0}, {mw1}, {mw2}, {mw3});"
            )
            name = f"t{tmp_id}"
            tmp_id += 1
            if aw == 32:
                maskv = f"t{tmp_id}_maskv"
                tmp_id += 1
                lines.append(f"  __m256 {maskv} = _mm256_castsi256_ps({maski});")
                lines.append(f"  __m256 {name} = _mm256_blendv_ps({a}, {b}, {maskv});")
            else:
                maskv = f"t{tmp_id}_maskv"
                tmp_id += 1
                lines.append(f"  __m256d {maskv} = _mm256_castsi256_pd({maski});")
                lines.append(f"  __m256d {name} = _mm256_blendv_pd({a}, {b}, {maskv});")
            memo[key] = (name, aw)
            return name, aw

        raise CodegenError("unsupported expression for x86 avx backend")

    def emit_mask(expr: Expr) -> str:
        nonlocal tmp_id
        key = f"mask:{repr(expr)}"
        if key in memo:
            return memo[key][0]

        if not isinstance(expr, (SimdFCmpEq, SimdFCmpLt, SimdFCmpLe, SimdFCmpNe)):
            raise CodegenError("unsupported mask expression for x86 avx backend")

        a, aw = emit_vec(expr.a)
        b, bw = emit_vec(expr.b)
        if aw != bw:
            raise CodegenError("simd float cmp requires matching lane_width")

        cmpv = f"t{tmp_id}_cmpv"
        tmp_id += 1
        if aw == 32:
            if isinstance(expr, SimdFCmpEq):
                pred = "_CMP_EQ_OQ"
            elif isinstance(expr, SimdFCmpLt):
                pred = "_CMP_LT_OQ"
            elif isinstance(expr, SimdFCmpLe):
                pred = "_CMP_LE_OQ"
            else:
                pred = "_CMP_NEQ_UQ"
            lines.append(f"  __m256 {cmpv} = _mm256_cmp_ps({a}, {b}, {pred});")
            mask = f"t{tmp_id}_mask"
            tmp_id += 1
            lines.append(f"  int {mask} = _mm256_movemask_ps({cmpv});")
        else:
            if isinstance(expr, SimdFCmpEq):
                pred = "_CMP_EQ_OQ"
            elif isinstance(expr, SimdFCmpLt):
                pred = "_CMP_LT_OQ"
            elif isinstance(expr, SimdFCmpLe):
                pred = "_CMP_LE_OQ"
            else:
                pred = "_CMP_NEQ_UQ"
            lines.append(f"  __m256d {cmpv} = _mm256_cmp_pd({a}, {b}, {pred});")
            mask = f"t{tmp_id}_mask"
            tmp_id += 1
            lines.append(f"  int {mask} = _mm256_movemask_pd({cmpv});")

        out = f"t{tmp_id}_out"
        tmp_id += 1
        lines.append(f"  uint64_t {out} = (uint64_t){mask};")
        memo[key] = (out, 0)
        return out

    lines.append("void stc_eval(const uint64_t* in, uint64_t* out) {")
    for idx, name in enumerate(input_order):
        t = _require_simd256_float(ir.inputs[name])
        if t.lane_width == 32:
            lines.append(f"  __m256 {_c_ident(name)} = load256ps(in + {idx * 4});")
        else:
            lines.append(f"  __m256d {_c_ident(name)} = load256pd(in + {idx * 4});")

    for out_idx, name in enumerate(output_order):
        expr = ir.output_exprs[name]
        out_t = infer_type(expr, types)
        if (
            isinstance(out_t, SimdType)
            and out_t.total_width == 256
            and out_t.lane_width in {32, 64}
        ):
            v, vw = emit_vec(expr)
            if vw == 32:
                lines.append(f"  store256ps(out + {out_idx * 4}, {v});")
            else:
                lines.append(f"  store256pd(out + {out_idx * 4}, {v});")
        elif isinstance(out_t, SimdType) and out_t.lane_width == 1:
            lanes = out_t.lanes
            if lanes > 64:
                raise CodegenError("mask outputs wider than 64 bits are not supported")
            m = emit_mask(expr)
            lines.append(f"  out[{out_idx * 4 + 0}] = {m};")
            lines.append(f"  out[{out_idx * 4 + 1}] = 0ull;")
            lines.append(f"  out[{out_idx * 4 + 2}] = 0ull;")
            lines.append(f"  out[{out_idx * 4 + 3}] = 0ull;")
        else:
            raise CodegenError("unsupported output type for x86 avx backend")

    lines.append("}")
    lines.append("")
    return "\n".join(lines)
