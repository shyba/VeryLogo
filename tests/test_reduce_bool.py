import unittest

from stc.reduce import reduce_expr
from stc.tick_ir import And, BoolConst, BoolType, Not, Var, Xor


class TestReduceBool(unittest.TestCase):
    def test_not_const(self) -> None:
        self.assertEqual(
            reduce_expr(Not(x=BoolConst(value=False)), {}), BoolConst(value=True)
        )

    def test_and_true(self) -> None:
        x = Var(name="x")
        self.assertEqual(
            reduce_expr(And(a=x, b=BoolConst(value=True)), {"x": BoolType()}), x
        )

    def test_xor_false(self) -> None:
        x = Var(name="x")
        self.assertEqual(
            reduce_expr(Xor(a=x, b=BoolConst(value=False)), {"x": BoolType()}), x
        )
