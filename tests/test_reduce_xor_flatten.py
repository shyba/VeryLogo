import unittest

from stc.reduce import reduce_expr
from stc.tick_ir import BitVecConst, BitVecType, Var, Xor


def _xor_terms(expr):
    out = []

    def rec(x):
        if isinstance(x, Xor):
            rec(x.a)
            rec(x.b)
        else:
            out.append(x)

    rec(expr)
    return out


class TestReduceXorFlatten(unittest.TestCase):
    def test_cancels_duplicates(self) -> None:
        t = BitVecType(width=8)
        types = {"a": t, "b": t, "c": t}
        expr = Xor(a=Xor(a=Var("a"), b=Var("b")), b=Xor(a=Var("b"), b=Var("c")))
        reduced = reduce_expr(expr, types)
        terms = sorted(_xor_terms(reduced), key=repr)
        self.assertEqual(terms, sorted([Var("a"), Var("c")], key=repr))

    def test_folds_constants(self) -> None:
        t = BitVecType(width=8)
        types = {"x": t}
        expr = Xor(
            a=BitVecConst(width=8, value=0xAA),
            b=Xor(a=BitVecConst(width=8, value=0x0F), b=Var("x")),
        )
        reduced = reduce_expr(expr, types)
        terms = _xor_terms(reduced)
        self.assertIn(Var("x"), terms)
        consts = [v for v in terms if isinstance(v, BitVecConst)]
        self.assertEqual(len(consts), 1)
        self.assertEqual(int(consts[0].value) & 0xFF, 0xA5)
