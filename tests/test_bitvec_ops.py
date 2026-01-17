import unittest

from stc.interp import eval_expr
from stc.tick_ir import (
    AShr,
    BitVecConst,
    BitVecType,
    LShr,
    Shl,
    Sub,
    Uge,
    Ugt,
    Ule,
    Ult,
    Var,
)


class TestBitVecOps(unittest.TestCase):
    def test_sub_wrap(self) -> None:
        t = BitVecType(width=4)
        types = {"x": t}
        expr = Sub(a=Var("x"), b=BitVecConst(width=4, value=1))
        self.assertEqual(eval_expr(expr, types, {"x": 0}), 0xF)

    def test_shifts(self) -> None:
        t = BitVecType(width=4)
        sh_t = BitVecType(width=4)
        types = {"x": t, "sh": sh_t}
        self.assertEqual(
            eval_expr(Shl(a=Var("x"), b=Var("sh")), types, {"x": 0x8, "sh": 1}), 0x0
        )
        self.assertEqual(
            eval_expr(LShr(a=Var("x"), b=Var("sh")), types, {"x": 0x8, "sh": 1}), 0x4
        )
        self.assertEqual(
            eval_expr(AShr(a=Var("x"), b=Var("sh")), types, {"x": 0x8, "sh": 1}), 0xC
        )

    def test_unsigned_compares(self) -> None:
        t = BitVecType(width=4)
        types = {"a": t, "b": t}
        self.assertEqual(
            eval_expr(Ult(a=Var("a"), b=Var("b")), types, {"a": 1, "b": 2}), True
        )
        self.assertEqual(
            eval_expr(Ule(a=Var("a"), b=Var("b")), types, {"a": 2, "b": 2}), True
        )
        self.assertEqual(
            eval_expr(Ugt(a=Var("a"), b=Var("b")), types, {"a": 3, "b": 2}), True
        )
        self.assertEqual(
            eval_expr(Uge(a=Var("a"), b=Var("b")), types, {"a": 2, "b": 2}), True
        )
