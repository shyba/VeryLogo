from __future__ import annotations

from dataclasses import dataclass

from stc.interp import infer_type
from stc.backend_x86_avx import emit_x86_avx_c
from stc.backend_x86_avx2 import emit_x86_avx2_c
from stc.backend_x86_avx512 import emit_x86_avx512_c
from stc.backend_x86_avx512_float import emit_x86_avx512_float_c
from stc.backend_x86_avx512vl import emit_x86_avx512vl_c
from stc.backend_x86_sse2 import emit_x86_sse2_c
from stc.tick_ir import (
    EXPR_CLASSES,
    SimdAShr,
    SimdAdd,
    SimdAddMasked,
    SimdAddSatS,
    SimdAddSatU,
    SimdBlend,
    SimdEq,
    SimdFAdd,
    SimdFAbs,
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
    SimdLShr,
    SimdMaddS16,
    SimdMaxS,
    SimdMaxU,
    SimdMaskPack,
    SimdMinS,
    SimdMinU,
    SimdMulHiS,
    SimdMulHiU,
    SimdMulLo,
    SimdSExtLo,
    SimdShuffle,
    SimdShl,
    SimdSub,
    SimdSubMasked,
    SimdSubSatS,
    SimdSubSatU,
    SimdType,
    SimdZExtLo,
    SimdSlt,
    SimdUlt,
    TickIR,
)
from stc.tick_ir_validate import validate_tick_ir
from stc.x86_features import has_flag


@dataclass(frozen=True)
class AutoBackendError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def _iter_child_exprs(expr) -> list:
    children: list = []
    for field_name in getattr(expr, "__dataclass_fields__", {}):
        value = getattr(expr, field_name)
        if isinstance(value, (tuple, list)):
            for v in value:
                if isinstance(v, EXPR_CLASSES):
                    children.append(v)
        elif isinstance(value, EXPR_CLASSES):
            children.append(value)
    return children


def _walk_expr(expr) -> list:
    out: list = []
    stack = [expr]
    while stack:
        cur = stack.pop()
        out.append(cur)
        stack.extend(_iter_child_exprs(cur))
    return out


def emit_x86_auto_c(ir: TickIR, *, flags: set[str] | None = None) -> str:
    validate_tick_ir(ir)
    if ir.state:
        raise AutoBackendError("x86 auto backend does not support state yet")
    if not ir.inputs:
        raise AutoBackendError("x86 auto backend requires at least one simd input")

    widths = set()
    for t in ir.inputs.values():
        if not isinstance(t, SimdType):
            raise AutoBackendError("x86 auto backend requires simd inputs")
        widths.add(t.total_width)
    if len(widths) != 1:
        raise AutoBackendError("x86 auto backend requires uniform simd total_width")
    (w,) = tuple(widths)

    if w == 128:
        need_ssse3 = False
        need_sse41 = False
        need_avx512vl = False
        for e in ir.output_exprs.values():
            for node in _walk_expr(e):
                if isinstance(node, SimdShuffle):
                    need_ssse3 = True
                if isinstance(
                    node,
                    (
                        SimdMinU,
                        SimdMaxU,
                        SimdMinS,
                        SimdMaxS,
                        SimdBlend,
                        SimdZExtLo,
                        SimdSExtLo,
                    ),
                ):
                    need_sse41 = True
                if isinstance(node, (SimdAddMasked, SimdSubMasked)):
                    need_avx512vl = True
        if need_ssse3 and not has_flag("ssse3", flags):
            raise AutoBackendError("ssse3 is not available on this CPU")
        if need_sse41 and not has_flag("sse4_1", flags):
            raise AutoBackendError("sse4_1 is not available on this CPU")
        if need_avx512vl:
            if not has_flag("avx512f", flags):
                raise AutoBackendError("avx512f is not available on this CPU")
            if not has_flag("avx512vl", flags):
                raise AutoBackendError("avx512vl is not available on this CPU")
            return emit_x86_avx512vl_c(ir)
        return emit_x86_sse2_c(ir)
    if w == 256:
        need_avx = False
        need_fma = False
        need_avx512vl = False
        need_avx512bw = False
        for e in ir.output_exprs.values():
            for node in _walk_expr(e):
                if isinstance(
                    node,
                    (
                        SimdFAdd,
                        SimdFSub,
                        SimdFMul,
                        SimdFDiv,
                        SimdFFma,
                        SimdFSqrt,
                        SimdFNeg,
                        SimdFAbs,
                        SimdFCmpEq,
                        SimdFCmpLt,
                        SimdFCmpLe,
                        SimdFCmpNe,
                    ),
                ):
                    need_avx = True
                if isinstance(node, SimdFFma):
                    need_fma = True
                if isinstance(node, (SimdAddMasked, SimdSubMasked)):
                    need_avx512vl = True
                    operand_t = infer_type(node, dict(ir.inputs))
                    if (
                        isinstance(operand_t, SimdType)
                        and operand_t.total_width == 256
                        and operand_t.lane_width in {8, 16}
                    ):
                        need_avx512bw = True
        if need_avx:
            if not has_flag("avx", flags):
                raise AutoBackendError("avx is not available on this CPU")
            if need_fma and not has_flag("fma", flags):
                raise AutoBackendError("fma is not available on this CPU")
            return emit_x86_avx_c(ir)
        if need_avx512vl:
            if not has_flag("avx512f", flags):
                raise AutoBackendError("avx512f is not available on this CPU")
            if not has_flag("avx512vl", flags):
                raise AutoBackendError("avx512vl is not available on this CPU")
            if need_avx512bw and not has_flag("avx512bw", flags):
                raise AutoBackendError("avx512bw is not available on this CPU")
            return emit_x86_avx512vl_c(ir)
        if not has_flag("avx2", flags):
            raise AutoBackendError("avx2 is not available on this CPU")
        return emit_x86_avx2_c(ir)
    if w == 512:
        if not has_flag("avx512f", flags):
            raise AutoBackendError("avx512f is not available on this CPU")

        need_avx512_float = False
        need_fma = False
        need_avx512bw = False
        need_avx512vbmi = False
        types = dict(ir.inputs)
        for e in ir.output_exprs.values():
            for node in _walk_expr(e):
                if isinstance(
                    node,
                    (
                        SimdFAdd,
                        SimdFSub,
                        SimdFMul,
                        SimdFDiv,
                        SimdFFma,
                        SimdFSqrt,
                        SimdFNeg,
                        SimdFAbs,
                        SimdFCmpEq,
                        SimdFCmpLt,
                        SimdFCmpLe,
                        SimdFCmpNe,
                    ),
                ):
                    need_avx512_float = True
                if isinstance(node, SimdFFma):
                    need_fma = True
                if isinstance(
                    node,
                    (
                        SimdAdd,
                        SimdSub,
                        SimdShl,
                        SimdLShr,
                        SimdAShr,
                        SimdEq,
                        SimdSlt,
                        SimdUlt,
                        SimdMaskPack,
                        SimdMinU,
                        SimdMaxU,
                        SimdMinS,
                        SimdMaxS,
                        SimdBlend,
                        SimdZExtLo,
                        SimdSExtLo,
                        SimdAddSatU,
                        SimdAddSatS,
                        SimdSubSatU,
                        SimdSubSatS,
                        SimdMulLo,
                        SimdMulHiU,
                        SimdMulHiS,
                        SimdMaddS16,
                    ),
                ):
                    if isinstance(node, (SimdEq, SimdSlt, SimdUlt)):
                        operand_t = infer_type(node.a, types)
                    elif isinstance(node, SimdMaskPack):
                        operand_t = infer_type(node.x, types)
                    elif isinstance(node, (SimdZExtLo, SimdSExtLo)):
                        operand_t = infer_type(node.x, types)
                    else:
                        operand_t = infer_type(node, types)
                    if isinstance(operand_t, SimdType) and operand_t.lane_width in {
                        8,
                        16,
                    }:
                        need_avx512bw = True

                if isinstance(node, SimdShuffle):
                    src_t = infer_type(node.x, types)
                    if (
                        isinstance(src_t, SimdType)
                        and src_t.total_width == 512
                        and src_t.lane_width == 8
                        and src_t.lanes == 64
                    ):
                        need_avx512vbmi = True

        if need_avx512_float:
            if need_fma and not has_flag("fma", flags):
                raise AutoBackendError("fma is not available on this CPU")
            return emit_x86_avx512_float_c(ir)

        if need_avx512bw and not has_flag("avx512bw", flags):
            raise AutoBackendError("avx512bw is not available on this CPU")
        if need_avx512vbmi and not has_flag("avx512vbmi", flags):
            raise AutoBackendError("avx512vbmi is not available on this CPU")

        return emit_x86_avx512_c(ir)

    raise AutoBackendError("unsupported simd total_width for x86 auto backend")
