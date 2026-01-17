import unittest

from stc.backend_ptx import emit_ptx
from stc.tick_ir import (
    Add,
    BitVecConst,
    BitVecType,
    BoolConst,
    BoolType,
    Concat,
    Eq,
    Lut8,
    Mux,
    Shl,
    Slice,
    TickIR,
    Uge,
    Ugt,
    Ule,
    Ult,
    Var,
)


class TestBackendPtx(unittest.TestCase):
    def test_emits_header_and_kernel_scaffold(self) -> None:
        ir = TickIR(
            name="t",
            inputs={"a": BitVecType(width=32)},
            outputs={"o": BitVecType(width=32)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Var("a")},
        )
        ptx = emit_ptx(ir, sm="sm_61")
        self.assertIn(".target sm_61", ptx)
        self.assertIn(".visible .entry stc_eval", ptx)
        self.assertIn("mov.u32", ptx)
        self.assertIn("%tid.x", ptx)
        self.assertIn("mad.lo.u32", ptx)
        self.assertIn("setp.ge.u32", ptx)

    def test_emits_add_eq_mux(self) -> None:
        t = BitVecType(width=32)
        ir = TickIR(
            name="t2",
            inputs={"a": t, "b": t, "sel": BoolType()},
            outputs={"add": t, "eq": BoolType(), "mux": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "add": Add(a=Var("a"), b=Var("b")),
                "eq": Eq(a=Var("a"), b=Var("b")),
                "mux": Mux(cond=Var("sel"), a=Var("a"), b=Var("b")),
            },
        )
        ptx = emit_ptx(ir)
        self.assertIn("add.u32", ptx)
        self.assertIn("setp.eq.u32", ptx)
        self.assertIn("selp.b32", ptx)

    def test_masks_narrow_bitvec_results(self) -> None:
        t5 = BitVecType(width=5)
        ir = TickIR(
            name="t3",
            inputs={"a": t5, "b": t5},
            outputs={"o": t5},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Add(a=Var("a"), b=Var("b"))},
        )
        ptx = emit_ptx(ir)
        self.assertIn("and.b32", ptx)
        self.assertIn("0x1f", ptx)

    def test_stateful_signature_includes_state_ptrs(self) -> None:
        t = BitVecType(width=8)
        ir = TickIR(
            name="t_state",
            inputs={"a": t},
            outputs={"o": t},
            state={"q": t},
            reset_state={"q": BitVecConst(width=8, value=0)},
            next_state={"q": Add(a=Var("q"), b=Var("a"))},
            output_exprs={"o": Var("q")},
        )
        ptx = emit_ptx(ir)
        self.assertIn(".param .u64 __state_in", ptx)
        self.assertIn(".param .u64 __state_out", ptx)

    def test_shift_zero_when_sh_ge_width(self) -> None:
        t = BitVecType(width=32)
        ir = TickIR(
            name="t4",
            inputs={"a": t, "sh": t},
            outputs={"o": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Shl(a=Var("a"), b=Var("sh"))},
        )
        ptx = emit_ptx(ir)
        self.assertIn("setp.ge.u32", ptx)
        self.assertIn("selp.b32", ptx)

    def test_emits_u64_ops_for_wide_bitvec(self) -> None:
        t = BitVecType(width=64)
        ir = TickIR(
            name="t64",
            inputs={"a": t, "b": t},
            outputs={"o": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Add(a=Var("a"), b=Var("b"))},
        )
        ptx = emit_ptx(ir)
        self.assertIn("add.cc.u32", ptx)
        self.assertIn("addc.u32", ptx)

    def test_emits_lut8_const_load(self) -> None:
        t = BitVecType(width=8)
        table = list(range(256))
        ir = TickIR(
            name="t_lut8",
            inputs={"x": t},
            outputs={"y": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": Lut8(x=Var("x"), table=table)},
        )
        ptx = emit_ptx(ir)
        self.assertIn(".const", ptx)
        self.assertIn("ld.const.u8", ptx)

    def test_masks_narrow_u64_results(self) -> None:
        t = BitVecType(width=40)
        ir = TickIR(
            name="t40",
            inputs={"a": t, "b": t},
            outputs={"o": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Add(a=Var("a"), b=Var("b"))},
        )
        ptx = emit_ptx(ir)
        self.assertIn("and.b32", ptx)
        self.assertIn("0xff", ptx.lower())

    def test_bool_consts(self) -> None:
        ir = TickIR(
            name="t5",
            inputs={},
            outputs={"o": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": BoolConst(value=True)},
        )
        ptx = emit_ptx(ir)
        self.assertIn("mov.pred", ptx)

    def test_bitvec_const(self) -> None:
        ir = TickIR(
            name="t6",
            inputs={},
            outputs={"o": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": BitVecConst(width=8, value=0xA5)},
        )
        ptx = emit_ptx(ir)
        self.assertIn("mov.u32", ptx)
        self.assertIn("0xa5", ptx.lower())

    def test_emits_unsigned_compare_ops(self) -> None:
        t = BitVecType(width=96)
        ir = TickIR(
            name="t_ucmp",
            inputs={"a": t, "b": t},
            outputs={
                "lt": BoolType(),
                "le": BoolType(),
                "gt": BoolType(),
                "ge": BoolType(),
            },
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "lt": Ult(a=Var("a"), b=Var("b")),
                "le": Ule(a=Var("a"), b=Var("b")),
                "gt": Ugt(a=Var("a"), b=Var("b")),
                "ge": Uge(a=Var("a"), b=Var("b")),
            },
        )
        ptx = emit_ptx(ir)
        self.assertIn("setp.lt.u32", ptx)
        self.assertIn("setp.gt.u32", ptx)

    def test_emits_concat_from_slices(self) -> None:
        t = BitVecType(width=4)
        ir = TickIR(
            name="t_concat",
            inputs={"a": t},
            outputs={"y": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "y": Concat(
                    parts=[
                        Slice(x=Var(name="a"), offset=3, width=1),
                        Slice(x=Var(name="a"), offset=2, width=1),
                        Slice(x=Var(name="a"), offset=1, width=1),
                        Slice(x=Var(name="a"), offset=0, width=1),
                    ]
                )
            },
        )
        ptx = emit_ptx(ir)
        self.assertIn("shl.b32", ptx)
        self.assertIn("or.b32", ptx)
