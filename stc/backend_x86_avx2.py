from __future__ import annotations

from dataclasses import dataclass

from stc.interp import infer_type
from stc.tick_ir import (
    BitTranspose,
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
    SimdMaddS16,
    SimdMaskPack,
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
    SimdSExtLo,
    SimdSub,
    SimdSubSatS,
    SimdSubSatU,
    SimdType,
    SimdUlt,
    SimdUnpackHi,
    SimdUnpackLo,
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


def _require_simd256(t: Type) -> SimdType:
    if not isinstance(t, SimdType):
        raise CodegenError("x86 avx2 backend requires simd types")
    if t.total_width != 256:
        raise CodegenError("x86 avx2 backend requires total_width=256")
    return t


def _emit_simd_const(x: SimdConst) -> str:
    words = [(x.value >> (64 * i)) & ((1 << 64) - 1) for i in range(4)]
    return f"load256_imm({words[0]}ull, {words[1]}ull, {words[2]}ull, {words[3]}ull)"


def _emit_sh_amount(expr: Expr, types: dict[str, Type]) -> int:
    if not isinstance(expr, BitVecConst):
        raise CodegenError("avx2 simd shifts require constant shift amount")
    if expr.value < 0:
        raise CodegenError("shift amount must be >= 0")
    return int(expr.value)


def emit_x86_avx2_c(ir: TickIR) -> str:
    validate_tick_ir(ir)
    if ir.state:
        raise CodegenError("x86 avx2 backend does not support state yet")

    types: dict[str, Type] = dict(ir.inputs)
    for t in ir.inputs.values():
        _require_simd256(t)

    input_order = sorted(ir.inputs.keys())
    output_order = sorted(ir.outputs.keys())

    lines: list[str] = []
    lines.append("#include <stdint.h>")
    lines.append("#include <immintrin.h>")
    lines.append("")
    lines.append("typedef union { __m256i v; uint64_t u64[4]; } u256;")
    lines.append("")
    lines.append("static inline __m256i load256(const uint64_t* p) {")
    lines.append("  u256 x;")
    lines.append("  x.u64[0] = p[0];")
    lines.append("  x.u64[1] = p[1];")
    lines.append("  x.u64[2] = p[2];")
    lines.append("  x.u64[3] = p[3];")
    lines.append("  return x.v;")
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
    lines.append("  return x.v;")
    lines.append("}")
    lines.append("")
    lines.append("static inline void store256(uint64_t* p, __m256i v) {")
    lines.append("  u256 x;")
    lines.append("  x.v = v;")
    lines.append("  p[0] = x.u64[0];")
    lines.append("  p[1] = x.u64[1];")
    lines.append("  p[2] = x.u64[2];")
    lines.append("  p[3] = x.u64[3];")
    lines.append("}")
    lines.append("")
    lines.append("// Bit transpose: 32 bytes -> 8 32-bit words")
    lines.append("// Output word[i] contains bit i from all 32 input bytes")
    lines.append("static inline __m256i bit_transpose_8x32(__m256i x) {")
    lines.append("  u256 in, out;")
    lines.append("  in.v = x;")
    lines.append("  uint32_t planes[8] = {0};")
    lines.append("  for (int byte_idx = 0; byte_idx < 32; byte_idx++) {")
    lines.append("    int word = byte_idx / 8;")
    lines.append("    int shift = (byte_idx % 8) * 8;")
    lines.append("    uint8_t byte_val = (in.u64[word] >> shift) & 0xFF;")
    lines.append("    for (int bit = 0; bit < 8; bit++) {")
    lines.append("      if (byte_val & (1 << bit))")
    lines.append("        planes[bit] |= (1u << byte_idx);")
    lines.append("    }")
    lines.append("  }")
    lines.append("  out.u64[0] = (uint64_t)planes[0] | ((uint64_t)planes[1] << 32);")
    lines.append("  out.u64[1] = (uint64_t)planes[2] | ((uint64_t)planes[3] << 32);")
    lines.append("  out.u64[2] = (uint64_t)planes[4] | ((uint64_t)planes[5] << 32);")
    lines.append("  out.u64[3] = (uint64_t)planes[6] | ((uint64_t)planes[7] << 32);")
    lines.append("  return out.v;")
    lines.append("}")
    lines.append("")
    lines.append("// Bit transpose: 8 32-bit words -> 32 bytes")
    lines.append("// Output byte[j] contains bit j from all 8 input words")
    lines.append("static inline __m256i bit_transpose_32x8(__m256i x) {")
    lines.append("  u256 in, out;")
    lines.append("  in.v = x;")
    lines.append("  uint32_t planes[8];")
    lines.append("  planes[0] = in.u64[0] & 0xFFFFFFFF;")
    lines.append("  planes[1] = (in.u64[0] >> 32) & 0xFFFFFFFF;")
    lines.append("  planes[2] = in.u64[1] & 0xFFFFFFFF;")
    lines.append("  planes[3] = (in.u64[1] >> 32) & 0xFFFFFFFF;")
    lines.append("  planes[4] = in.u64[2] & 0xFFFFFFFF;")
    lines.append("  planes[5] = (in.u64[2] >> 32) & 0xFFFFFFFF;")
    lines.append("  planes[6] = in.u64[3] & 0xFFFFFFFF;")
    lines.append("  planes[7] = (in.u64[3] >> 32) & 0xFFFFFFFF;")
    lines.append("  uint8_t bytes[32] = {0};")
    lines.append("  for (int byte_idx = 0; byte_idx < 32; byte_idx++) {")
    lines.append("    for (int bit = 0; bit < 8; bit++) {")
    lines.append("      if (planes[bit] & (1u << byte_idx))")
    lines.append("        bytes[byte_idx] |= (1 << bit);")
    lines.append("    }")
    lines.append("  }")
    lines.append("  out.u64[0] = out.u64[1] = out.u64[2] = out.u64[3] = 0;")
    lines.append("  for (int i = 0; i < 8; i++) {")
    lines.append("    out.u64[0] |= ((uint64_t)bytes[i]) << (i * 8);")
    lines.append("    out.u64[1] |= ((uint64_t)bytes[8 + i]) << (i * 8);")
    lines.append("    out.u64[2] |= ((uint64_t)bytes[16 + i]) << (i * 8);")
    lines.append("    out.u64[3] |= ((uint64_t)bytes[24 + i]) << (i * 8);")
    lines.append("  }")
    lines.append("  return out.v;")
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
            lines.append(f"  __m256i {name} = {_emit_simd_const(expr)};")
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
                f"  __m256i {name} = _mm256_xor_si256({x}, _mm256_set1_epi32(-1));"
            )
            memo[key] = name
            return name

        if isinstance(expr, (SimdAnd, SimdOr, SimdXor)):
            name = f"t{tmp_id}"
            tmp_id += 1
            if isinstance(expr, SimdAnd):
                if isinstance(expr.a, SimdNot):
                    a0 = emit_expr(expr.a.x)
                    b0 = emit_expr(expr.b)
                    lines.append(f"  __m256i {name} = _mm256_andnot_si256({a0}, {b0});")
                    memo[key] = name
                    return name
                if isinstance(expr.b, SimdNot):
                    b0 = emit_expr(expr.b.x)
                    a0 = emit_expr(expr.a)
                    lines.append(f"  __m256i {name} = _mm256_andnot_si256({b0}, {a0});")
                    memo[key] = name
                    return name
                a = emit_expr(expr.a)
                b = emit_expr(expr.b)
                op = "_mm256_and_si256"
            elif isinstance(expr, SimdOr):
                a = emit_expr(expr.a)
                b = emit_expr(expr.b)
                op = "_mm256_or_si256"
            else:
                a = emit_expr(expr.a)
                b = emit_expr(expr.b)
                op = "_mm256_xor_si256"
            lines.append(f"  __m256i {name} = {op}({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdAdd, SimdSub)):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd256(infer_type(expr.a, types))
            name = f"t{tmp_id}"
            tmp_id += 1
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
                raise CodegenError("unsupported lane_width for add/sub")
            lines.append(f"  __m256i {name} = {op}({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdAddSatU, SimdSubSatU)):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd256(infer_type(expr.a, types))
            name = f"t{tmp_id}"
            tmp_id += 1
            if t.lane_width == 8:
                op = (
                    "_mm256_adds_epu8"
                    if isinstance(expr, SimdAddSatU)
                    else "_mm256_subs_epu8"
                )
            elif t.lane_width == 16:
                op = (
                    "_mm256_adds_epu16"
                    if isinstance(expr, SimdAddSatU)
                    else "_mm256_subs_epu16"
                )
            else:
                raise CodegenError(
                    "avx2 unsigned saturating ops require lane_width in {8,16}"
                )
            lines.append(f"  __m256i {name} = {op}({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdAddSatS, SimdSubSatS)):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd256(infer_type(expr.a, types))
            name = f"t{tmp_id}"
            tmp_id += 1
            if t.lane_width == 8:
                op = (
                    "_mm256_adds_epi8"
                    if isinstance(expr, SimdAddSatS)
                    else "_mm256_subs_epi8"
                )
            elif t.lane_width == 16:
                op = (
                    "_mm256_adds_epi16"
                    if isinstance(expr, SimdAddSatS)
                    else "_mm256_subs_epi16"
                )
            else:
                raise CodegenError(
                    "avx2 signed saturating ops require lane_width in {8,16}"
                )
            lines.append(f"  __m256i {name} = {op}({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdMulLo, SimdMulHiU, SimdMulHiS)):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd256(infer_type(expr.a, types))
            name = f"t{tmp_id}"
            tmp_id += 1
            if t.lane_width != 16:
                raise CodegenError("avx2 mul ops require lane_width=16")
            if isinstance(expr, SimdMulLo):
                op = "_mm256_mullo_epi16"
            elif isinstance(expr, SimdMulHiU):
                op = "_mm256_mulhi_epu16"
            else:
                op = "_mm256_mulhi_epi16"
            lines.append(f"  __m256i {name} = {op}({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, SimdMaddS16):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd256(infer_type(expr.a, types))
            name = f"t{tmp_id}"
            tmp_id += 1
            if t.lane_width != 16:
                raise CodegenError("avx2 madd requires lane_width=16")
            lines.append(f"  __m256i {name} = _mm256_madd_epi16({a}, {b});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdShl, SimdLShr, SimdAShr)):
            a = emit_expr(expr.a)
            t = _require_simd256(infer_type(expr.a, types))
            sh = _emit_sh_amount(expr.sh, types)
            name = f"t{tmp_id}"
            tmp_id += 1

            if t.lane_width == 16:
                if isinstance(expr, SimdShl):
                    op = "_mm256_slli_epi16"
                elif isinstance(expr, SimdLShr):
                    op = "_mm256_srli_epi16"
                else:
                    op = "_mm256_srai_epi16"
            elif t.lane_width == 32:
                if isinstance(expr, SimdShl):
                    op = "_mm256_slli_epi32"
                elif isinstance(expr, SimdLShr):
                    op = "_mm256_srli_epi32"
                else:
                    op = "_mm256_srai_epi32"
            elif t.lane_width == 64:
                if isinstance(expr, SimdAShr):
                    raise CodegenError("avx2 does not support ashr for 64-bit lanes")
                op = (
                    "_mm256_slli_epi64"
                    if isinstance(expr, SimdShl)
                    else "_mm256_srli_epi64"
                )
            else:
                raise CodegenError("unsupported lane_width for shifts")

            lines.append(f"  __m256i {name} = {op}({a}, {sh});")
            memo[key] = name
            return name

        if isinstance(expr, SimdBlend):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            t = _require_simd256(infer_type(expr.a, types))
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

            mw0 = f"t{tmp_id}_mw0"
            mw1 = f"t{tmp_id}_mw1"
            mw2 = f"t{tmp_id}_mw2"
            mw3 = f"t{tmp_id}_mw3"
            tmp_id += 1
            lines.append(f"  uint64_t {mw0} = 0ull;")
            lines.append(f"  uint64_t {mw1} = 0ull;")
            lines.append(f"  uint64_t {mw2} = 0ull;")
            lines.append(f"  uint64_t {mw3} = 0ull;")
            for i in range(t.lanes):
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

            maskv = f"t{tmp_id}_maskv"
            tmp_id += 1
            lines.append(
                f"  __m256i {maskv} = load256_imm({mw0}, {mw1}, {mw2}, {mw3});"
            )
            name = f"t{tmp_id}"
            tmp_id += 1
            lines.append(f"  __m256i {name} = _mm256_blendv_epi8({a}, {b}, {maskv});")
            memo[key] = name
            return name

        if isinstance(expr, (SimdZExtLo, SimdSExtLo)):
            x = emit_expr(expr.x)
            src_t = _require_simd256(infer_type(expr.x, types))
            dst_t = _require_simd256(infer_type(expr, types))
            if src_t.lane_width != dst_t.lane_width // 2:
                raise CodegenError("simd extend lane_width mismatch")
            if src_t.lanes != dst_t.lanes * 2:
                raise CodegenError("simd extend lanes mismatch")
            x128 = f"t{tmp_id}_x128"
            tmp_id += 1
            lines.append(f"  __m128i {x128} = _mm256_castsi256_si128({x});")
            if src_t.lane_width == 8 and dst_t.lane_width == 16:
                op = (
                    "_mm256_cvtepu8_epi16"
                    if isinstance(expr, SimdZExtLo)
                    else "_mm256_cvtepi8_epi16"
                )
            elif src_t.lane_width == 16 and dst_t.lane_width == 32:
                op = (
                    "_mm256_cvtepu16_epi32"
                    if isinstance(expr, SimdZExtLo)
                    else "_mm256_cvtepi16_epi32"
                )
            elif src_t.lane_width == 32 and dst_t.lane_width == 64:
                op = (
                    "_mm256_cvtepu32_epi64"
                    if isinstance(expr, SimdZExtLo)
                    else "_mm256_cvtepi32_epi64"
                )
            else:
                raise CodegenError("avx2 extend supports 8->16, 16->32, 32->64")
            name = f"t{tmp_id}"
            tmp_id += 1
            lines.append(f"  __m256i {name} = {op}({x128});")
            memo[key] = name
            return name

        if isinstance(expr, SimdShuffle):
            x = emit_expr(expr.x)
            src_t = _require_simd256(infer_type(expr.x, types))
            out_t = _require_simd256(infer_type(expr, types))
            if src_t.lane_width != out_t.lane_width:
                raise CodegenError("shuffle requires matching lane_width")

            if src_t.lane_width == 8 and src_t.lanes == 32 and out_t.lanes == 32:
                if len(expr.indices) != 32:
                    raise CodegenError("shuffle requires 32 indices for simd[8,32]")
                if any(i < 0 or i >= 32 for i in expr.indices):
                    raise CodegenError("shuffle index out of bounds")

                xlo = f"t{tmp_id}_xlo"
                xhi = f"t{tmp_id}_xhi"
                tmp_id += 1
                lines.append(
                    f"  __m256i {xlo} = _mm256_permute2x128_si256({x}, {x}, 0x00);"
                )
                lines.append(
                    f"  __m256i {xhi} = _mm256_permute2x128_si256({x}, {x}, 0x11);"
                )

                ctl_lo = f"t{tmp_id}_ctl_lo"
                ctl_hi = f"t{tmp_id}_ctl_hi"
                tmp_id += 1
                lo_args = ", ".join(
                    f"(char){int(i) if int(i) < 16 else 0x80}" for i in expr.indices
                )
                hi_args = ", ".join(
                    f"(char){int(i) - 16 if int(i) >= 16 else 0x80}"
                    for i in expr.indices
                )
                lines.append(f"  __m256i {ctl_lo} = _mm256_setr_epi8({lo_args});")
                lines.append(f"  __m256i {ctl_hi} = _mm256_setr_epi8({hi_args});")

                out_lo = f"t{tmp_id}_out_lo"
                out_hi = f"t{tmp_id}_out_hi"
                tmp_id += 1
                lines.append(
                    f"  __m256i {out_lo} = _mm256_shuffle_epi8({xlo}, {ctl_lo});"
                )
                lines.append(
                    f"  __m256i {out_hi} = _mm256_shuffle_epi8({xhi}, {ctl_hi});"
                )
                name = f"t{tmp_id}"
                tmp_id += 1
                lines.append(f"  __m256i {name} = _mm256_or_si256({out_lo}, {out_hi});")
                memo[key] = name
                return name

            if src_t.lane_width == 32 and src_t.lanes == 8 and out_t.lanes == 8:
                if len(expr.indices) != 8:
                    raise CodegenError("shuffle requires 8 indices for simd[32,8]")
                if any(i < 0 or i >= 8 for i in expr.indices):
                    raise CodegenError("shuffle index out of bounds")
                ctl = f"t{tmp_id}_ctl"
                tmp_id += 1
                args = ", ".join(str(int(i)) for i in expr.indices)
                lines.append(f"  __m256i {ctl} = _mm256_setr_epi32({args});")
                name = f"t{tmp_id}"
                tmp_id += 1
                lines.append(
                    f"  __m256i {name} = _mm256_permutevar8x32_epi32({x}, {ctl});"
                )
                memo[key] = name
                return name

            raise CodegenError("avx2 shuffle supports simd[8,32] and simd[32,8] only")

        if isinstance(
            expr,
            (
                SimdUnpackLo,
                SimdUnpackHi,
                SimdPackSS16To8,
                SimdPackUS16To8,
                SimdPackSS32To16,
            ),
        ):
            if isinstance(expr, (SimdUnpackLo, SimdUnpackHi)):
                a = emit_expr(expr.a)
                b = emit_expr(expr.b)
                t = _require_simd256(infer_type(expr, types))
                if t.lane_width not in {8, 16, 32, 64}:
                    raise CodegenError(
                        "simd_unpack supports lane_width in {8,16,32,64}"
                    )

                a0 = f"t{tmp_id}_a0"
                b0 = f"t{tmp_id}_b0"
                tmp_id += 1
                if isinstance(expr, SimdUnpackLo):
                    lines.append(f"  __m128i {a0} = _mm256_castsi256_si128({a});")
                    lines.append(f"  __m128i {b0} = _mm256_castsi256_si128({b});")
                else:
                    lines.append(f"  __m128i {a0} = _mm256_extracti128_si256({a}, 1);")
                    lines.append(f"  __m128i {b0} = _mm256_extracti128_si256({b}, 1);")

                lo = f"t{tmp_id}_lo"
                hi = f"t{tmp_id}_hi"
                tmp_id += 1
                if t.lane_width == 8:
                    op_lo = "_mm_unpacklo_epi8"
                    op_hi = "_mm_unpackhi_epi8"
                elif t.lane_width == 16:
                    op_lo = "_mm_unpacklo_epi16"
                    op_hi = "_mm_unpackhi_epi16"
                elif t.lane_width == 32:
                    op_lo = "_mm_unpacklo_epi32"
                    op_hi = "_mm_unpackhi_epi32"
                else:
                    op_lo = "_mm_unpacklo_epi64"
                    op_hi = "_mm_unpackhi_epi64"
                lines.append(f"  __m128i {lo} = {op_lo}({a0}, {b0});")
                lines.append(f"  __m128i {hi} = {op_hi}({a0}, {b0});")
                name = f"t{tmp_id}"
                tmp_id += 1
                lines.append(f"  __m256i {name} = _mm256_castsi128_si256({lo});")
                lines.append(f"  {name} = _mm256_inserti128_si256({name}, {hi}, 1);")
                memo[key] = name
                return name

            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            a_t = _require_simd256(infer_type(expr.a, types))
            if isinstance(expr, (SimdPackSS16To8, SimdPackUS16To8)):
                if a_t.lane_width != 16:
                    raise CodegenError("simd_pack_*16_to_8 requires lane_width=16")
                pack_op = (
                    "_mm_packs_epi16"
                    if isinstance(expr, SimdPackSS16To8)
                    else "_mm_packus_epi16"
                )
            else:
                if a_t.lane_width != 32:
                    raise CodegenError("simd_pack_ss32_to_16 requires lane_width=32")
                pack_op = "_mm_packs_epi32"

            a_lo = f"t{tmp_id}_alo"
            a_hi = f"t{tmp_id}_ahi"
            b_lo = f"t{tmp_id}_blo"
            b_hi = f"t{tmp_id}_bhi"
            tmp_id += 1
            lines.append(f"  __m128i {a_lo} = _mm256_castsi256_si128({a});")
            lines.append(f"  __m128i {a_hi} = _mm256_extracti128_si256({a}, 1);")
            lines.append(f"  __m128i {b_lo} = _mm256_castsi256_si128({b});")
            lines.append(f"  __m128i {b_hi} = _mm256_extracti128_si256({b}, 1);")

            pa = f"t{tmp_id}_pa"
            pb = f"t{tmp_id}_pb"
            tmp_id += 1
            lines.append(f"  __m128i {pa} = {pack_op}({a_lo}, {a_hi});")
            lines.append(f"  __m128i {pb} = {pack_op}({b_lo}, {b_hi});")
            name = f"t{tmp_id}"
            tmp_id += 1
            lines.append(f"  __m256i {name} = _mm256_castsi128_si256({pa});")
            lines.append(f"  {name} = _mm256_inserti128_si256({name}, {pb}, 1);")
            memo[key] = name
            return name

        if isinstance(expr, BitTranspose):
            x = emit_expr(expr.x)
            src_t = _require_simd256(infer_type(expr.x, types))
            name = f"t{tmp_id}"
            tmp_id += 1
            if src_t.lane_width == 8 and src_t.lanes == 32:
                if expr.lane_width != 8 or expr.lanes != 32:
                    raise CodegenError(
                        "BitTranspose simd[8,32] requires lane_width=8, lanes=32"
                    )
                lines.append(f"  __m256i {name} = bit_transpose_8x32({x});")
            elif src_t.lane_width == 32 and src_t.lanes == 8:
                if expr.lane_width != 32 or expr.lanes != 8:
                    raise CodegenError(
                        "BitTranspose simd[32,8] requires lane_width=32, lanes=8"
                    )
                lines.append(f"  __m256i {name} = bit_transpose_32x8({x});")
            else:
                raise CodegenError(
                    "avx2 BitTranspose only supports simd[8,32] and simd[32,8]"
                )
            memo[key] = name
            return name

        raise CodegenError("unsupported expression for x86 avx2 backend")

    def emit_mask(expr: Expr) -> str:
        nonlocal tmp_id
        key = f"mask:{repr(expr)}"
        if key in memo:
            return memo[key]

        if not isinstance(expr, (SimdEq, SimdSlt, SimdUlt, SimdMaskPack)):
            raise CodegenError("unsupported mask expression for x86 avx2 backend")

        if isinstance(expr, SimdMaskPack):
            src_t = _require_simd256(infer_type(expr.x, types))
            a = emit_expr(expr.x)
            b = None
            lane_width = src_t.lane_width
        else:
            at = _require_simd256(infer_type(expr.a, types))
            bt = _require_simd256(infer_type(expr.b, types))
            if at.lane_width != bt.lane_width or at.lanes != bt.lanes:
                raise CodegenError("mask compare requires matching simd operands")

            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            lane_width = at.lane_width
        if lane_width not in {8, 16, 32}:
            raise CodegenError("avx2 compare supports lane_width in {8,16,32}")

        cmp_name = f"t{tmp_id}"
        tmp_id += 1
        if isinstance(expr, SimdMaskPack):
            zero = f"t{tmp_id}"
            tmp_id += 1
            eq0 = f"t{tmp_id}"
            tmp_id += 1
            lines.append(f"  __m256i {zero} = _mm256_setzero_si256();")
            if lane_width == 8:
                op = "_mm256_cmpeq_epi8"
            elif lane_width == 16:
                op = "_mm256_cmpeq_epi16"
            else:
                op = "_mm256_cmpeq_epi32"
            lines.append(f"  __m256i {eq0} = {op}({a}, {zero});")
            lines.append(
                f"  __m256i {cmp_name} = _mm256_xor_si256({eq0}, _mm256_set1_epi32(-1));"
            )
        elif isinstance(expr, SimdEq):
            if lane_width == 8:
                op = "_mm256_cmpeq_epi8"
            elif lane_width == 16:
                op = "_mm256_cmpeq_epi16"
            else:
                op = "_mm256_cmpeq_epi32"
            lines.append(f"  __m256i {cmp_name} = {op}({a}, {b});")
        elif isinstance(expr, SimdSlt):
            if lane_width == 8:
                op = "_mm256_cmpgt_epi8"
            elif lane_width == 16:
                op = "_mm256_cmpgt_epi16"
            else:
                op = "_mm256_cmpgt_epi32"
            lines.append(f"  __m256i {cmp_name} = {op}({b}, {a});")
        else:
            if lane_width == 8:
                bias = "_mm256_set1_epi8((char)0x80)"
                op = "_mm256_cmpgt_epi8"
            elif lane_width == 16:
                bias = "_mm256_set1_epi16((short)0x8000)"
                op = "_mm256_cmpgt_epi16"
            else:
                bias = "_mm256_set1_epi32((int)0x80000000u)"
                op = "_mm256_cmpgt_epi32"
            ax = f"t{tmp_id}"
            tmp_id += 1
            bx = f"t{tmp_id}"
            tmp_id += 1
            lines.append(f"  __m256i {ax} = _mm256_xor_si256({a}, {bias});")
            lines.append(f"  __m256i {bx} = _mm256_xor_si256({b}, {bias});")
            lines.append(f"  __m256i {cmp_name} = {op}({bx}, {ax});")

        mask32 = f"t{tmp_id}"
        tmp_id += 1
        out = f"t{tmp_id}"
        tmp_id += 1
        group = lane_width // 8
        lanes = 32 // group
        lines.append(f"  int {mask32} = _mm256_movemask_epi8({cmp_name});")
        lines.append(f"  uint64_t {out} = 0ull;")
        for i in range(lanes):
            lines.append(
                f"  {out} |= (uint64_t)(({mask32} >> {i * group}) & 1) << {i};"
            )

        memo[key] = out
        return out

    lines.append("void stc_eval(const uint64_t* in, uint64_t* out) {")
    for idx, name in enumerate(input_order):
        lines.append(f"  __m256i {_c_ident(name)} = load256(in + {idx * 4});")

    for out_idx, name in enumerate(output_order):
        expr = ir.output_exprs[name]
        out_t = infer_type(expr, types)
        if isinstance(out_t, BitVecType):
            raise CodegenError("bitvec outputs are not supported yet")
        if isinstance(out_t, SimdType) and out_t.total_width == 256:
            v = emit_expr(expr)
            lines.append(f"  store256(out + {out_idx * 4}, {v});")
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
            raise CodegenError("unsupported output type for x86 avx2 backend")

    lines.append("}")
    lines.append("")
    return "\n".join(lines)
