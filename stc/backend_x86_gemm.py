"""Packed integer GEMM target lowering for x86-64 AVX-512 VNNI."""

from __future__ import annotations

from dataclasses import dataclass

from stc.interp import infer_type
from stc.tick_ir import BitVecType, Expr, GemmCall, Slice, TickIR, Type, Var
from stc.tick_ir_validate import validate_tick_ir


@dataclass(frozen=True)
class GemmCodegenError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def _c_ident(name: str) -> str:
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    if not out:
        return "_"
    return out if not out[0].isdigit() else "_" + out


def _output_root(expr: Expr) -> GemmCall | None:
    if isinstance(expr, GemmCall):
        return expr
    if isinstance(expr, Slice) and isinstance(expr.x, GemmCall):
        if expr.offset == 0 and expr.width == expr.x.result_width:
            return expr.x
    return None


def _words(width: int) -> int:
    return (int(width) + 63) // 64


def _emit_scalar_kernel(lines: list[str], call: GemmCall) -> None:
    lines.extend(
        [
            "  for (int i = 0; i < GEMM_M; ++i) {",
            "    for (int j = 0; j < GEMM_N; ++j) {",
            "      int64_t total = 0;",
            "      for (int q = 0; q < GEMM_K; ++q) {",
            "        int32_t av = (int32_t)A[i * GEMM_K + q];",
            "        int32_t bv = (int32_t)B[q * GEMM_N + j];",
        ]
    )
    if call.a_signed:
        lines.append("        av = (int32_t)(int8_t)av;")
    if call.b_signed:
        lines.append("        bv = (int32_t)(int8_t)bv;")
    lines.extend(
        [
            "        total += (int64_t)av * (int64_t)bv;",
            "      }",
            "      C[i * GEMM_N + j] = (int32_t)total;",
            "    }",
            "  }",
        ]
    )


def _emit_vnni_kernel(lines: list[str]) -> None:
    lines.extend(
        [
            "  for (int i = 0; i < GEMM_M; ++i) {",
            "    for (int j0 = 0; j0 < GEMM_N; j0 += 16) {",
            "      __m512i acc = _mm512_setzero_si512();",
            "      int8_t bpack[64];",
            "      for (int q0 = 0; q0 < GEMM_K; q0 += 4) {",
            "        uint32_t av = (uint32_t)A[i * GEMM_K + q0 + 0]",
            "          | ((uint32_t)A[i * GEMM_K + q0 + 1] << 8)"
            "          | ((uint32_t)A[i * GEMM_K + q0 + 2] << 16)"
            "          | ((uint32_t)A[i * GEMM_K + q0 + 3] << 24);",
            "        __m512i avec = _mm512_set1_epi32((int)av);",
            "        for (int lane = 0; lane < 16; ++lane) {",
            "          for (int q = 0; q < 4; ++q) {",
            "            bpack[4 * lane + q] = B[(q0 + q) * GEMM_N + j0 + lane];",
            "          }",
            "        }",
            "        __m512i bvec = _mm512_loadu_si512((const void*)bpack);",
            "        acc = _mm512_dpbusd_epi32(acc, avec, bvec);",
            "      }",
            "      int32_t lanes[16];",
            "      _mm512_storeu_si512((void*)lanes, acc);",
            "      for (int lane = 0; lane < 16; ++lane) {",
            "        C[i * GEMM_N + j0 + lane] = lanes[lane];",
            "      }",
            "    }",
            "  }",
        ]
    )


def emit_x86_gemm_c(ir: TickIR) -> str:
    """Emit ``stc_eval(const uint64_t*, uint64_t*)`` for one shaped GEMM.

    The packed ABI assigns each Tick-IR input/output one or more little-endian
    u64 words, in sorted port-name order.  Matrix elements inside those words
    are row-major bytes.  Shapes with the VNNI-friendly int8 contract use
    ``VPDPBUSD``; all other int8 shapes use the same ABI and a scalar fallback.
    """

    validate_tick_ir(ir)
    if ir.state:
        raise GemmCodegenError("x86 GEMM backend does not support state")
    if not ir.inputs or not ir.outputs:
        raise GemmCodegenError("x86 GEMM backend requires inputs and an output")

    roots = [_output_root(expr) for expr in ir.output_exprs.values()]
    call = next((root for root in roots if root is not None), None)
    if call is None:
        raise GemmCodegenError("x86 GEMM backend requires a GemmCall output")
    if any(root != call for root in roots):
        raise GemmCodegenError("x86 GEMM backend supports one GEMM result")
    from stc.tech import get_technology

    lowered = get_technology("x86-gemm").lower_expr(call, target="x86-vnni")
    if not isinstance(lowered, GemmCall):
        raise GemmCodegenError("x86 GEMM lowering dispatcher returned a non-GEMM node")
    call = lowered
    if call.layout != "row_major":
        raise GemmCodegenError("x86 GEMM backend supports row_major layout only")

    if not isinstance(call.a, Var) or not isinstance(call.b, Var):
        raise GemmCodegenError("x86 GEMM backend requires direct A/B input ports")
    if call.a.name not in ir.inputs or call.b.name not in ir.inputs:
        raise GemmCodegenError("GEMM operands must be Tick-IR inputs")
    a_t = ir.inputs[call.a.name]
    b_t = ir.inputs[call.b.name]
    if not isinstance(a_t, BitVecType) or a_t.width != call.a_total_width:
        raise GemmCodegenError("GEMM A input type does not match shape")
    if not isinstance(b_t, BitVecType) or b_t.width != call.b_total_width:
        raise GemmCodegenError("GEMM B input type does not match shape")
    if call.a_width != 8 or call.b_width != 8 or call.acc_width != 32:
        raise GemmCodegenError("x86 GEMM currently lowers 8-bit inputs to 32-bit accumulators")

    output_order = sorted(ir.outputs)
    if len(output_order) != 1:
        raise GemmCodegenError("x86 GEMM backend supports one output port")
    out_name = output_order[0]
    out_t = ir.outputs[out_name]
    if not isinstance(out_t, BitVecType) or out_t.width != call.result_width:
        raise GemmCodegenError("GEMM output type does not match shape")

    input_order = sorted(ir.inputs)
    input_word_offsets: dict[str, int] = {}
    cursor = 0
    for name in input_order:
        input_word_offsets[name] = cursor
        cursor += _words(ir.inputs[name].width if isinstance(ir.inputs[name], BitVecType) else 0)
    output_words = _words(out_t.width)

    lines = [
        "#include <stdint.h>",
        "#include <stddef.h>",
        "#include <string.h>",
        "#include <immintrin.h>",
        "",
        f"#define GEMM_M {call.m}",
        f"#define GEMM_N {call.n}",
        f"#define GEMM_K {call.k}",
        "",
        "void stc_eval(const uint64_t* in, uint64_t* out) {",
        "  uint8_t A[GEMM_M * GEMM_K];",
        "  uint8_t B[GEMM_K * GEMM_N];",
        "  int32_t C[GEMM_M * GEMM_N];",
    ]
    a_off = input_word_offsets[call.a.name] * 8
    b_off = input_word_offsets[call.b.name] * 8
    lines.append(f"  memcpy(A, ((const uint8_t*)in) + {a_off}, sizeof(A));")
    lines.append(f"  memcpy(B, ((const uint8_t*)in) + {b_off}, sizeof(B));")

    if (
        not call.a_signed
        and call.b_signed
        and call.n % 16 == 0
        and call.k % 4 == 0
    ):
        _emit_vnni_kernel(lines)
    else:
        _emit_scalar_kernel(lines, call)

    lines.extend(
        [
            f"  memset(out, 0, {output_words} * sizeof(uint64_t));",
            f"  memcpy(((uint8_t*)out) + 0, C, sizeof(C));",
            "}",
            "",
        ]
    )
    return "\n".join(lines)
