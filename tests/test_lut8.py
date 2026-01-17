import unittest

import z3

from stc.interp import eval_expr, infer_type
from stc.reduce import reduce_expr
from stc.tick_ir import BitVecConst, BitVecType, Lut8, Var
from stc.z3_encode import encode_expr


class TestLut8(unittest.TestCase):
    def test_infer_type(self) -> None:
        table = list(range(256))
        expr = Lut8(x=Var("x"), table=table)
        t = infer_type(expr, {"x": BitVecType(width=8)})
        self.assertEqual(t, BitVecType(width=8))

    def test_eval_expr(self) -> None:
        table = [(i * 7 + 3) & 0xFF for i in range(256)]
        expr = Lut8(x=Var("x"), table=table)
        types = {"x": BitVecType(width=8)}
        for i in (0, 1, 2, 7, 13, 255):
            got = int(eval_expr(expr, types, {"x": i}))
            self.assertEqual(got, table[i])

    def test_reduce_constant_fold(self) -> None:
        table = [(i ^ 0xA5) & 0xFF for i in range(256)]
        expr = Lut8(x=BitVecConst(width=8, value=42), table=table)
        reduced = reduce_expr(expr, {})
        self.assertEqual(reduced, BitVecConst(width=8, value=table[42]))

    def test_z3_encode_constant_index(self) -> None:
        table = [(i * 17 + 9) & 0xFF for i in range(256)]
        expr = Lut8(x=BitVecConst(width=8, value=19), table=table)
        z = encode_expr(expr, {})
        self.assertIsInstance(z, z3.BitVecRef)
        s = z3.Solver()
        s.add(z != z3.BitVecVal(table[19], 8))
        self.assertEqual(s.check(), z3.unsat)
