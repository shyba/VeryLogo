import unittest

from stc.reduce import reduce_expr
from stc.tick_ir import BitVecType, BoolType, Not, Var, Xor
from stc.z3_prove import prove_equiv


class TestReduceProofs(unittest.TestCase):
    def test_reduce_preserves_semantics_for_bool_identities(self) -> None:
        types = {"x": BoolType()}
        spec = Not(x=Not(x=Var("x")))
        cand = reduce_expr(spec, types)
        self.assertTrue(prove_equiv(spec, cand, types, timeout_ms=200))

    def test_reduce_preserves_semantics_for_xor_self(self) -> None:
        bv8 = BitVecType(width=8)
        types = {"x": bv8}
        spec = Xor(a=Var("x"), b=Var("x"))
        cand = reduce_expr(spec, types)
        self.assertTrue(prove_equiv(spec, cand, types, timeout_ms=200))
