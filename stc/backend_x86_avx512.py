from __future__ import annotations

from dataclasses import dataclass

from stc.interp import infer_type
from stc.tick_ir import (
    BitVecConst,
    BitVecType,
    Bitcast,
    Expr,
    SimdAdd,
    SimdAddMasked,
    SimdAddSatS,
    SimdAddSatU,
    SimdAnd,
    SimdAShr,
    SimdBlend,
    SimdConst,
    SimdEq,
    SimdLShr,
    SimdMaddS16,
    SimdMaxS,
    SimdMaxU,
    SimdMaskPack,
    SimdMinS,
    SimdMinU,
    SimdPackSS16To8,
    SimdPackSS32To16,
    SimdPackUS16To8,
    SimdMulHiS,
    SimdMulHiU,
    SimdMulLo,
    SimdNot,
    SimdOr,
    SimdSExtLo,
    SimdShl,
    SimdShuffle,
    SimdSlt,
    SimdSub,
    SimdSubMasked,
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


def _require_simd512(t: Type) -> SimdType:
    if not isinstance(t, SimdType):
        raise CodegenError("x86 avx512 backend requires simd types")
    if t.total_width != 512:
        raise CodegenError("x86 avx512 backend requires total_width=512")
    return t


def _emit_simd_const(x: SimdConst) -> str:
    words = [(x.value >> (64 * i)) & ((1 << 64) - 1) for i in range(8)]
    return "load512_imm(" + ", ".join(f"{w}ull" for w in words) + ")"


def _emit_u512_imm_words(value: int) -> str:
    words = [(value >> (64 * i)) & ((1 << 64) - 1) for i in range(8)]
    return ", ".join(f"{w}ull" for w in words)


def _emit_sh_amount(expr: Expr, types: dict[str, Type]) -> int:
    if not isinstance(expr, BitVecConst):
        raise CodegenError("avx512 simd shifts require constant shift amount")
    if expr.value < 0:
        raise CodegenError("shift amount must be >= 0")
    return int(expr.value)


def emit_x86_avx512_c(ir: TickIR) -> str:
    validate_tick_ir(ir)
    if ir.state:
        raise CodegenError("x86 avx512 backend does not support state yet")

    types: dict[str, Type] = dict(ir.inputs)
    for t in ir.inputs.values():
        _require_simd512(t)

    input_order = sorted(ir.inputs.keys())
    output_order = sorted(ir.outputs.keys())

    lines: list[str] = []
    lines.append("#include <stdint.h>")
    lines.append("#include <immintrin.h>")
    lines.append("")
    lines.append(
        "typedef union { __m512i v; uint64_t u64[8]; uint32_t u32[16]; uint16_t u16[32]; uint8_t u8[64]; } u512;"
    )
    lines.append("")
    lines.append("static inline __m512i load512(const uint64_t* p) {")
    lines.append("  u512 x;")
    for i in range(8):
        lines.append(f"  x.u64[{i}] = p[{i}];")
    lines.append("  return x.v;")
    lines.append("}")
    lines.append("")
    lines.append(
        "static inline __m512i load512_imm(uint64_t w0, uint64_t w1, uint64_t w2, uint64_t w3, uint64_t w4, uint64_t w5, uint64_t w6, uint64_t w7) {"
    )
    lines.append("  u512 x;")
    for i in range(8):
        lines.append(f"  x.u64[{i}] = w{i};")
    lines.append("  return x.v;")
    lines.append("}")
    lines.append("")
    lines.append("static inline void store512(uint64_t* p, __m512i v) {")
    lines.append("  u512 x;")
    lines.append("  x.v = v;")
    for i in range(8):
        lines.append(f"  p[{i}] = x.u64[{i}];")
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
            lines.append(f"  __m512i {name} = {_emit_simd_const(expr)};")
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
            lines.append(
                f"  __m512i {name} = _mm512_xor_si512({x}, _mm512_set1_epi32(-1));"
            )
            memo[key] = name
            return name

        if isinstance(expr, (SimdAnd, SimdOr, SimdXor)):
            name = f"t{tmp_id}"
            tmp_id += 1
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            if isinstance(expr, SimdAnd):
                op = "_mm512_and_si512"
            elif isinstance(expr, SimdOr):
                op = "_mm512_or_si512"
            else:
                op = "_mm512_xor_si512"
            lines.append(f"  __m512i {name} = {op}({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdAdd, SimdSub)):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd512(infer_type(expr.a, types))
            name = f"t{tmp_id}"
            tmp_id += 1
            if t.lane_width == 8:
                op = (
                    "_mm512_add_epi8"
                    if isinstance(expr, SimdAdd)
                    else "_mm512_sub_epi8"
                )
            elif t.lane_width == 16:
                op = (
                    "_mm512_add_epi16"
                    if isinstance(expr, SimdAdd)
                    else "_mm512_sub_epi16"
                )
            elif t.lane_width == 32:
                op = (
                    "_mm512_add_epi32"
                    if isinstance(expr, SimdAdd)
                    else "_mm512_sub_epi32"
                )
            elif t.lane_width == 64:
                op = (
                    "_mm512_add_epi64"
                    if isinstance(expr, SimdAdd)
                    else "_mm512_sub_epi64"
                )
            else:
                raise CodegenError(
                    "avx512 backend supports add/sub for lane_width in {8,16,32,64}"
                )
            lines.append(f"  __m512i {name} = {op}({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdAddMasked, SimdSubMasked)):
            p = emit_expr(expr.passthru)
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd512(infer_type(expr.a, types))
            m = emit_mask(expr.mask)
            name = f"t{tmp_id}"
            tmp_id += 1

            if isinstance(expr, SimdAddMasked):
                if t.lane_width == 8:
                    op = "_mm512_mask_add_epi8"
                elif t.lane_width == 16:
                    op = "_mm512_mask_add_epi16"
                elif t.lane_width == 32:
                    op = "_mm512_mask_add_epi32"
                elif t.lane_width == 64:
                    op = "_mm512_mask_add_epi64"
                else:
                    raise CodegenError(
                        "simd_add_masked supports lane_width in {8,16,32,64}"
                    )
            else:
                if t.lane_width == 8:
                    op = "_mm512_mask_sub_epi8"
                elif t.lane_width == 16:
                    op = "_mm512_mask_sub_epi16"
                elif t.lane_width == 32:
                    op = "_mm512_mask_sub_epi32"
                elif t.lane_width == 64:
                    op = "_mm512_mask_sub_epi64"
                else:
                    raise CodegenError(
                        "simd_sub_masked supports lane_width in {8,16,32,64}"
                    )

            lines.append(f"  __m512i {name} = {op}({p}, {m}, {a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdUnpackLo, SimdUnpackHi)):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd512(infer_type(expr.a, types))
            if t.lanes % 2 != 0:
                raise CodegenError("simd unpack requires even lanes")
            half = t.lanes // 2

            ua = f"t{tmp_id}_ua"
            ub = f"t{tmp_id}_ub"
            uo = f"t{tmp_id}_uo"
            tmp_id += 1
            name = f"t{tmp_id}"
            tmp_id += 1

            lines.append(f"  u512 {ua}; {ua}.v = {a};")
            lines.append(f"  u512 {ub}; {ub}.v = {b};")
            lines.append(f"  u512 {uo};")
            src_expr = "i" if isinstance(expr, SimdUnpackLo) else f"(i + {half})"
            if t.lane_width == 8:
                lines.append(f"  for (int i = 0; i < {half}; i++) {{")
                lines.append(f"    {uo}.u8[2*i+0] = {ua}.u8[{src_expr}];")
                lines.append(f"    {uo}.u8[2*i+1] = {ub}.u8[{src_expr}];")
                lines.append("  }")
            elif t.lane_width == 16:
                lines.append(f"  for (int i = 0; i < {half}; i++) {{")
                lines.append(f"    {uo}.u16[2*i+0] = {ua}.u16[{src_expr}];")
                lines.append(f"    {uo}.u16[2*i+1] = {ub}.u16[{src_expr}];")
                lines.append("  }")
            elif t.lane_width == 32:
                lines.append(f"  for (int i = 0; i < {half}; i++) {{")
                lines.append(f"    {uo}.u32[2*i+0] = {ua}.u32[{src_expr}];")
                lines.append(f"    {uo}.u32[2*i+1] = {ub}.u32[{src_expr}];")
                lines.append("  }")
            elif t.lane_width == 64:
                lines.append(f"  for (int i = 0; i < {half}; i++) {{")
                lines.append(f"    {uo}.u64[2*i+0] = {ua}.u64[{src_expr}];")
                lines.append(f"    {uo}.u64[2*i+1] = {ub}.u64[{src_expr}];")
                lines.append("  }")
            else:
                raise CodegenError("simd unpack supports lane_width in {8,16,32,64}")

            lines.append(f"  __m512i {name} = {uo}.v;")
            memo[key] = name
            return name

        if isinstance(expr, (SimdPackSS16To8, SimdPackUS16To8)):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            a_t = _require_simd512(infer_type(expr.a, types))
            if a_t.lane_width != 16 or a_t.lanes != 32:
                raise CodegenError("simd pack16to8 requires simd[16,32] inputs")

            ua = f"t{tmp_id}_ua"
            ub = f"t{tmp_id}_ub"
            uo = f"t{tmp_id}_uo"
            tmp_id += 1
            name = f"t{tmp_id}"
            tmp_id += 1

            lines.append(f"  u512 {ua}; {ua}.v = {a};")
            lines.append(f"  u512 {ub}; {ub}.v = {b};")
            lines.append(f"  u512 {uo};")
            lines.append("  for (int i = 0; i < 64; i++) {")
            lines.append(
                f"    int v = (i < 32) ? (int)((int16_t){ua}.u16[i]) : (int)((int16_t){ub}.u16[i-32]);"
            )
            if isinstance(expr, SimdPackUS16To8):
                lines.append("    if (v < 0) v = 0;")
                lines.append("    if (v > 255) v = 255;")
                lines.append(f"    {uo}.u8[i] = (uint8_t)v;")
            else:
                lines.append("    if (v > 127) v = 127;")
                lines.append("    if (v < -128) v = -128;")
                lines.append(f"    {uo}.u8[i] = (uint8_t)(v & 0xFF);")
            lines.append("  }")
            lines.append(f"  __m512i {name} = {uo}.v;")
            memo[key] = name
            return name

        if isinstance(expr, SimdPackSS32To16):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            a_t = _require_simd512(infer_type(expr.a, types))
            if a_t.lane_width != 32 or a_t.lanes != 16:
                raise CodegenError("simd pack32to16 requires simd[32,16] inputs")

            ua = f"t{tmp_id}_ua"
            ub = f"t{tmp_id}_ub"
            uo = f"t{tmp_id}_uo"
            tmp_id += 1
            name = f"t{tmp_id}"
            tmp_id += 1

            lines.append(f"  u512 {ua}; {ua}.v = {a};")
            lines.append(f"  u512 {ub}; {ub}.v = {b};")
            lines.append(f"  u512 {uo};")
            lines.append("  for (int i = 0; i < 32; i++) {")
            lines.append(
                f"    int v = (i < 16) ? (int)((int32_t){ua}.u32[i]) : (int)((int32_t){ub}.u32[i-16]);"
            )
            lines.append("    if (v > 32767) v = 32767;")
            lines.append("    if (v < -32768) v = -32768;")
            lines.append(f"    {uo}.u16[i] = (uint16_t)(v & 0xFFFF);")
            lines.append("  }")
            lines.append(f"  __m512i {name} = {uo}.v;")
            memo[key] = name
            return name

        if isinstance(expr, (SimdAddSatU, SimdSubSatU, SimdAddSatS, SimdSubSatS)):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd512(infer_type(expr.a, types))
            name = f"t{tmp_id}"
            tmp_id += 1
            if t.lane_width not in {8, 16}:
                raise CodegenError(
                    "avx512 saturating add/sub supports lane_width in {8,16}"
                )

            if isinstance(expr, (SimdAddSatU, SimdSubSatU)):
                prefix = "adds" if isinstance(expr, SimdAddSatU) else "subs"
                suffix = "epu8" if t.lane_width == 8 else "epu16"
            else:
                prefix = "adds" if isinstance(expr, SimdAddSatS) else "subs"
                suffix = "epi8" if t.lane_width == 8 else "epi16"

            op = f"_mm512_{prefix}_{suffix}"
            lines.append(f"  __m512i {name} = {op}({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdMulLo, SimdMulHiU, SimdMulHiS, SimdMaddS16)):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd512(infer_type(expr.a, types))
            if t.lane_width != 16:
                raise CodegenError("avx512 mul/madd supports lane_width=16 only")
            name = f"t{tmp_id}"
            tmp_id += 1
            if isinstance(expr, SimdMulLo):
                op = "_mm512_mullo_epi16"
            elif isinstance(expr, SimdMulHiU):
                op = "_mm512_mulhi_epu16"
            elif isinstance(expr, SimdMulHiS):
                op = "_mm512_mulhi_epi16"
            else:
                op = "_mm512_madd_epi16"
            lines.append(f"  __m512i {name} = {op}({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdShl, SimdLShr, SimdAShr)):
            a = emit_expr(expr.a)
            t = _require_simd512(infer_type(expr.a, types))
            sh = _emit_sh_amount(expr.sh, types)
            name = f"t{tmp_id}"
            tmp_id += 1
            if t.lane_width == 16:
                if isinstance(expr, SimdShl):
                    op = "_mm512_slli_epi16"
                elif isinstance(expr, SimdLShr):
                    op = "_mm512_srli_epi16"
                else:
                    op = "_mm512_srai_epi16"
            elif t.lane_width == 32:
                if isinstance(expr, SimdShl):
                    op = "_mm512_slli_epi32"
                elif isinstance(expr, SimdLShr):
                    op = "_mm512_srli_epi32"
                else:
                    op = "_mm512_srai_epi32"
            elif t.lane_width == 64:
                if isinstance(expr, SimdAShr):
                    raise CodegenError("avx512 does not support ashr for 64-bit lanes")
                op = (
                    "_mm512_slli_epi64"
                    if isinstance(expr, SimdShl)
                    else "_mm512_srli_epi64"
                )
            else:
                raise CodegenError(
                    "avx512 backend supports shifts for lane_width in {16,32,64}"
                )
            lines.append(f"  __m512i {name} = {op}({a}, {sh});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdMinU, SimdMaxU, SimdMinS, SimdMaxS)):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd512(infer_type(expr.a, types))
            name = f"t{tmp_id}"
            tmp_id += 1

            if isinstance(expr, (SimdMinU, SimdMaxU)):
                prefix = "min" if isinstance(expr, SimdMinU) else "max"
                if t.lane_width == 8:
                    op = f"_mm512_{prefix}_epu8"
                elif t.lane_width == 16:
                    op = f"_mm512_{prefix}_epu16"
                elif t.lane_width == 32:
                    op = f"_mm512_{prefix}_epu32"
                elif t.lane_width == 64:
                    op = f"_mm512_{prefix}_epu64"
                else:
                    raise CodegenError(
                        "simd min/max requires lane_width in {8,16,32,64}"
                    )
            else:
                prefix = "min" if isinstance(expr, SimdMinS) else "max"
                if t.lane_width == 8:
                    op = f"_mm512_{prefix}_epi8"
                elif t.lane_width == 16:
                    op = f"_mm512_{prefix}_epi16"
                elif t.lane_width == 32:
                    op = f"_mm512_{prefix}_epi32"
                elif t.lane_width == 64:
                    op = f"_mm512_{prefix}_epi64"
                else:
                    raise CodegenError(
                        "simd min/max requires lane_width in {8,16,32,64}"
                    )

            lines.append(f"  __m512i {name} = {op}({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, SimdBlend):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd512(infer_type(expr.a, types))
            m = emit_mask(expr.mask)
            name = f"t{tmp_id}"
            tmp_id += 1

            if t.lane_width == 8:
                op = "_mm512_mask_blend_epi8"
            elif t.lane_width == 16:
                op = "_mm512_mask_blend_epi16"
            elif t.lane_width == 32:
                op = "_mm512_mask_blend_epi32"
            elif t.lane_width == 64:
                op = "_mm512_mask_blend_epi64"
            else:
                raise CodegenError("simd_blend supports lane_width in {8,16,32,64}")

            lines.append(f"  __m512i {name} = {op}({m}, {a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, SimdShuffle):
            x = emit_expr(expr.x)
            t = _require_simd512(infer_type(expr.x, types))
            if len(expr.indices) != t.lanes:
                raise CodegenError("simd_shuffle requires indices length == lanes")
            if any(i < 0 or i >= t.lanes for i in expr.indices):
                raise CodegenError("simd_shuffle index out of bounds")

            if t.lane_width not in {8, 32, 64}:
                raise CodegenError(
                    "avx512 simd_shuffle supports lane_width in {8,32,64}"
                )

            ctl_bits = 0
            for i, idx in enumerate(expr.indices):
                ctl_bits |= int(idx) << (t.lane_width * i)

            ctl = f"t{tmp_id}"
            tmp_id += 1
            name = f"t{tmp_id}"
            tmp_id += 1
            lines.append(
                f"  __m512i {ctl} = load512_imm({_emit_u512_imm_words(ctl_bits)});"
            )

            if t.lane_width == 8:
                op = "_mm512_permutexvar_epi8"
            elif t.lane_width == 32:
                op = "_mm512_permutexvar_epi32"
            else:
                op = "_mm512_permutexvar_epi64"

            lines.append(f"  __m512i {name} = {op}({ctl}, {x});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdZExtLo, SimdSExtLo)):
            src_t = _require_simd512(infer_type(expr.x, types))
            dst_t = expr.to
            if dst_t.total_width != 512:
                raise CodegenError("simd extend total_width mismatch")
            x = emit_expr(expr.x)
            xlo = f"t{tmp_id}"
            tmp_id += 1
            lines.append(f"  __m256i {xlo} = _mm512_castsi512_si256({x});")
            name = f"t{tmp_id}"
            tmp_id += 1

            if src_t.lane_width == 8 and dst_t.lane_width == 16 and dst_t.lanes == 32:
                op = (
                    "_mm512_cvtepu8_epi16"
                    if isinstance(expr, SimdZExtLo)
                    else "_mm512_cvtepi8_epi16"
                )
            elif (
                src_t.lane_width == 16 and dst_t.lane_width == 32 and dst_t.lanes == 16
            ):
                op = (
                    "_mm512_cvtepu16_epi32"
                    if isinstance(expr, SimdZExtLo)
                    else "_mm512_cvtepi16_epi32"
                )
            elif src_t.lane_width == 32 and dst_t.lane_width == 64 and dst_t.lanes == 8:
                op = (
                    "_mm512_cvtepu32_epi64"
                    if isinstance(expr, SimdZExtLo)
                    else "_mm512_cvtepi32_epi64"
                )
            else:
                raise CodegenError("avx512 simd extend supports 8->16, 16->32, 32->64")

            lines.append(f"  __m512i {name} = {op}({xlo});")
            memo[key] = name
            return name

        raise CodegenError("unsupported expression for x86 avx512 backend")

    def emit_mask(expr: Expr) -> str:
        nonlocal tmp_id
        key = f"mask:{repr(expr)}"
        if key in memo:
            return memo[key]

        if not isinstance(expr, (SimdEq, SimdSlt, SimdUlt, SimdMaskPack)):
            raise CodegenError("unsupported mask expression for x86 avx512 backend")

        if isinstance(expr, SimdMaskPack):
            src_t = _require_simd512(infer_type(expr.x, types))
            if src_t.lane_width not in {8, 16, 32, 64}:
                raise CodegenError(
                    "avx512 mask_pack supports lane_width in {8,16,32,64}"
                )
            a = emit_expr(expr.x)
            z = "_mm512_setzero_si512()"
            name = f"t{tmp_id}"
            tmp_id += 1
            if src_t.lane_width == 8:
                eq0 = f"t{tmp_id}"
                tmp_id += 1
                lines.append(f"  __mmask64 {eq0} = _mm512_cmpeq_epi8_mask({a}, {z});")
                lines.append(f"  __mmask64 {name} = (~{eq0});")
            elif src_t.lane_width == 16:
                eq0 = f"t{tmp_id}"
                tmp_id += 1
                lines.append(f"  __mmask32 {eq0} = _mm512_cmpeq_epi16_mask({a}, {z});")
                lines.append(f"  __mmask32 {name} = (~{eq0});")
            elif src_t.lane_width == 32:
                eq0 = f"t{tmp_id}"
                tmp_id += 1
                lines.append(f"  __mmask16 {eq0} = _mm512_cmpeq_epi32_mask({a}, {z});")
                lines.append(f"  __mmask16 {name} = (~{eq0});")
            else:
                eq0 = f"t{tmp_id}"
                tmp_id += 1
                lines.append(f"  __mmask8 {eq0} = _mm512_cmpeq_epi64_mask({a}, {z});")
                lines.append(f"  __mmask8 {name} = (~{eq0});")

            memo[key] = name
            return name

        at = _require_simd512(infer_type(expr.a, types))
        bt = _require_simd512(infer_type(expr.b, types))
        if at.lane_width != bt.lane_width or at.lanes != bt.lanes:
            raise CodegenError("mask compare requires matching simd operands")
        if at.lane_width not in {8, 16, 32, 64}:
            raise CodegenError(
                "avx512 backend supports compares for lane_width in {8,16,32,64}"
            )

        a = emit_expr(expr.a)
        b = emit_expr(expr.b)
        name = f"t{tmp_id}"
        tmp_id += 1
        if at.lane_width == 8:
            if isinstance(expr, SimdEq):
                lines.append(f"  __mmask64 {name} = _mm512_cmpeq_epi8_mask({a}, {b});")
            elif isinstance(expr, SimdSlt):
                lines.append(f"  __mmask64 {name} = _mm512_cmplt_epi8_mask({a}, {b});")
            else:
                lines.append(f"  __mmask64 {name} = _mm512_cmplt_epu8_mask({a}, {b});")
        elif at.lane_width == 16:
            if isinstance(expr, SimdEq):
                lines.append(f"  __mmask32 {name} = _mm512_cmpeq_epi16_mask({a}, {b});")
            elif isinstance(expr, SimdSlt):
                lines.append(f"  __mmask32 {name} = _mm512_cmplt_epi16_mask({a}, {b});")
            else:
                lines.append(f"  __mmask32 {name} = _mm512_cmplt_epu16_mask({a}, {b});")
        elif at.lane_width == 32:
            if isinstance(expr, SimdEq):
                lines.append(f"  __mmask16 {name} = _mm512_cmpeq_epi32_mask({a}, {b});")
            elif isinstance(expr, SimdSlt):
                lines.append(f"  __mmask16 {name} = _mm512_cmplt_epi32_mask({a}, {b});")
            else:
                lines.append(f"  __mmask16 {name} = _mm512_cmplt_epu32_mask({a}, {b});")
        else:
            if isinstance(expr, SimdEq):
                lines.append(f"  __mmask8 {name} = _mm512_cmpeq_epi64_mask({a}, {b});")
            elif isinstance(expr, SimdSlt):
                lines.append(f"  __mmask8 {name} = _mm512_cmplt_epi64_mask({a}, {b});")
            else:
                lines.append(f"  __mmask8 {name} = _mm512_cmplt_epu64_mask({a}, {b});")

        memo[key] = name
        return name

    lines.append("void stc_eval(const uint64_t* in, uint64_t* out) {")
    for idx, name in enumerate(input_order):
        lines.append(f"  __m512i {_c_ident(name)} = load512(in + {idx * 8});")

    for out_idx, name in enumerate(output_order):
        expr = ir.output_exprs[name]
        out_t = infer_type(expr, types)
        if isinstance(out_t, BitVecType):
            raise CodegenError("bitvec outputs are not supported yet")
        if isinstance(out_t, SimdType) and out_t.total_width == 512:
            v = emit_expr(expr)
            lines.append(f"  store512(out + {out_idx * 8}, {v});")
        elif isinstance(out_t, SimdType) and out_t.lane_width == 1:
            lanes = out_t.lanes
            if lanes > 64:
                raise CodegenError("mask outputs wider than 64 bits are not supported")
            m = emit_mask(expr)
            lines.append(f"  out[{out_idx * 8 + 0}] = (uint64_t){m};")
            for i in range(1, 8):
                lines.append(f"  out[{out_idx * 8 + i}] = 0ull;")
        else:
            raise CodegenError("unsupported output type for x86 avx512 backend")

    lines.append("}")
    lines.append("")
    return "\n".join(lines)
