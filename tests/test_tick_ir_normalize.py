import unittest

from stc.tick_ir import (
    Add,
    And,
    BitVecConst,
    BitVecType,
    BoolConst,
    Concat,
    LShr,
    Or,
    Rotl,
    Rotr,
    Shl,
    Slice,
    Sub,
    TickIR,
    Var,
)
from stc.tick_ir_normalize import normalize_expr, normalize_tick_ir


class TestRotateRecognition(unittest.TestCase):
    def test_recognize_rotl_const_amounts(self):
        types = {"x": BitVecType(width=32)}
        x = Var(name="x")
        k = BitVecConst(width=32, value=5)
        n_minus_k = BitVecConst(width=32, value=27)

        expr = Or(a=Shl(a=x, b=k), b=LShr(a=x, b=n_minus_k))
        result = normalize_expr(expr, types)

        self.assertIsInstance(result, Rotl)
        self.assertEqual(result.x, x)
        self.assertEqual(result.sh, k)

    def test_recognize_rotl_dynamic_amounts(self):
        types = {"x": BitVecType(width=32), "k": BitVecType(width=32)}
        x = Var(name="x")
        k = Var(name="k")
        n_minus_k = Sub(a=BitVecConst(width=32, value=32), b=k)

        expr = Or(a=Shl(a=x, b=k), b=LShr(a=x, b=n_minus_k))
        result = normalize_expr(expr, types)

        self.assertIsInstance(result, Rotl)
        self.assertEqual(result.x, x)
        self.assertEqual(result.sh, k)

    def test_recognize_rotr_dynamic_amounts(self):
        types = {"x": BitVecType(width=32), "k": BitVecType(width=32)}
        x = Var(name="x")
        k = Var(name="k")
        n_minus_k = Sub(a=BitVecConst(width=32, value=32), b=k)

        expr = Or(a=LShr(a=x, b=k), b=Shl(a=x, b=n_minus_k))
        result = normalize_expr(expr, types)

        self.assertIsInstance(result, Rotr)
        self.assertEqual(result.x, x)
        self.assertEqual(result.sh, k)

    def test_not_rotate_different_values(self):
        types = {"x": BitVecType(width=32), "y": BitVecType(width=32)}
        x = Var(name="x")
        y = Var(name="y")
        k = BitVecConst(width=32, value=5)

        expr = Or(a=Shl(a=x, b=k), b=LShr(a=y, b=k))
        result = normalize_expr(expr, types)

        self.assertIsInstance(result, Or)


class TestConcatSliceNormalization(unittest.TestCase):
    def test_flatten_nested_concat(self):
        types = {
            "a": BitVecType(width=8),
            "b": BitVecType(width=8),
            "c": BitVecType(width=8),
        }
        a = Var(name="a")
        b = Var(name="b")
        c = Var(name="c")

        expr = Concat(parts=[Concat(parts=[a, b]), c])
        result = normalize_expr(expr, types)

        self.assertIsInstance(result, Concat)
        self.assertEqual(len(result.parts), 3)
        self.assertEqual(result.parts[0], a)
        self.assertEqual(result.parts[1], b)
        self.assertEqual(result.parts[2], c)

    def test_slice_of_concat_full_part(self):
        types = {"a": BitVecType(width=8), "b": BitVecType(width=8)}
        a = Var(name="a")
        b = Var(name="b")

        concat = Concat(parts=[a, b])
        expr = Slice(x=concat, offset=0, width=8)
        result = normalize_expr(expr, types)

        self.assertEqual(result, b)


class TestMaskNormalization(unittest.TestCase):
    def test_and_with_all_ones_is_identity(self):
        types = {"x": BitVecType(width=8)}
        x = Var(name="x")
        mask = BitVecConst(width=8, value=0xFF)

        expr = And(a=x, b=mask)
        result = normalize_expr(expr, types)

        self.assertEqual(result, x)

    def test_and_with_zero_is_zero(self):
        types = {"x": BitVecType(width=8)}
        x = Var(name="x")
        zero = BitVecConst(width=8, value=0)

        expr = And(a=x, b=zero)
        result = normalize_expr(expr, types)

        self.assertEqual(result, zero)


class TestShiftNormalization(unittest.TestCase):
    def test_shift_beyond_width_becomes_zero(self):
        types = {"x": BitVecType(width=8)}
        x = Var(name="x")
        sh = BitVecConst(width=8, value=10)

        expr = Shl(a=x, b=sh)
        result = normalize_expr(expr, types)

        self.assertIsInstance(result, BitVecConst)
        self.assertEqual(result.value, 0)


class TestMuxNormalization(unittest.TestCase):
    def test_mux_equal_branches(self):
        types = {"c": BoolConst, "x": BitVecType(width=8)}
        c = Var(name="c")
        x = Var(name="x")

        from stc.tick_ir import Mux

        expr = Mux(cond=c, a=x, b=x)
        result = normalize_expr(expr, types)

        self.assertEqual(result, x)


class TestNormalizeTickIR(unittest.TestCase):
    def test_normalize_tick_ir_basic(self):
        ir = TickIR(
            name="test",
            inputs={"i": BitVecType(width=32)},
            outputs={"o": BitVecType(width=32)},
            state={"s": BitVecType(width=32)},
            reset_state={"s": BitVecConst(width=32, value=0)},
            next_state={
                "s": Or(
                    a=Shl(a=Var(name="s"), b=BitVecConst(width=32, value=1)),
                    b=LShr(a=Var(name="s"), b=BitVecConst(width=32, value=31)),
                )
            },
            output_exprs={"o": Var(name="s")},
        )

        result = normalize_tick_ir(ir)

        self.assertIsInstance(result.next_state["s"], Rotl)


if __name__ == "__main__":
    unittest.main()
