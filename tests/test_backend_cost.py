import unittest

from stc.tech import (
    PTXTechnology,
    AVX512Technology,
    GenericTechnology,
    get_technology,
)
from stc.tick_ir import And, Or, Xor, Not, Var, BoolType, TernaryLut
from stc.metrics import compute_depth, and_depth, multiplicative_depth


class TestBackendTechnologies(unittest.TestCase):
    def test_ptx_registration(self):
        """PTX technology is registered."""
        tech = get_technology("ptx")
        self.assertEqual(tech.name, "ptx")
        self.assertIsInstance(tech, PTXTechnology)

    def test_avx512_registration(self):
        """AVX-512 technology is registered."""
        tech = get_technology("x86-avx512")
        self.assertEqual(tech.name, "x86-avx512")
        self.assertIsInstance(tech, AVX512Technology)

    def test_ptx_primitives(self):
        """PTX has lop3 primitive."""
        tech = PTXTechnology()
        prims = {p.name for p in tech.primitives()}
        self.assertIn("lop3", prims)
        self.assertIn("and", prims)
        self.assertIn("xor", prims)

    def test_avx512_primitives(self):
        """AVX-512 has vpternlog primitive."""
        tech = AVX512Technology()
        prims = {p.name for p in tech.primitives()}
        self.assertIn("vpternlog", prims)
        self.assertIn("and", prims)


class TestBackendCostModels(unittest.TestCase):
    def test_ptx_ternary_lut_cost(self):
        """PTX cost model gives TernaryLut cost of 1."""
        tech = PTXTechnology()
        cost_model = tech.cost_model()

        a = Var("a")
        b = Var("b")
        c = Var("c")
        lut = TernaryLut(a, b, c, 0x96)

        cost = cost_model.expr_cost(lut)
        self.assertEqual(cost, 1.0)

    def test_avx512_ternary_lut_cost(self):
        """AVX-512 cost model gives TernaryLut cost of 1."""
        tech = AVX512Technology()
        cost_model = tech.cost_model()

        a = Var("a")
        b = Var("b")
        c = Var("c")
        lut = TernaryLut(a, b, c, 0x96)

        cost = cost_model.expr_cost(lut)
        self.assertEqual(cost, 1.0)

    def test_avx512_xor_weight(self):
        """AVX-512 has lower XOR weight."""
        tech = AVX512Technology()
        cost_model = tech.cost_model()

        self.assertEqual(cost_model.xor_weight(), 0.5)


class TestDepthComputation(unittest.TestCase):
    def test_compute_depth_generic(self):
        """Compute depth with generic depth model."""
        tech = GenericTechnology()
        depth_model = tech.depth_model()

        a = Var("a")
        b = Var("b")
        expr = And(a, b)

        depth = compute_depth(expr, depth_model)
        self.assertEqual(depth, 1)

    def test_compute_depth_chain(self):
        """Compute depth for chained operations."""
        tech = GenericTechnology()
        depth_model = tech.depth_model()

        a = Var("a")
        b = Var("b")
        c = Var("c")
        expr = And(And(a, b), c)

        depth = compute_depth(expr, depth_model)
        self.assertEqual(depth, 2)

    def test_and_depth(self):
        """AND-depth counts only AND gates."""
        a = Var("a")
        b = Var("b")
        c = Var("c")

        expr1 = Xor(Xor(a, b), c)
        self.assertEqual(and_depth(expr1), 0)

        expr2 = And(a, b)
        self.assertEqual(and_depth(expr2), 1)

        expr3 = And(And(a, b), c)
        self.assertEqual(and_depth(expr3), 2)

    def test_multiplicative_depth(self):
        """Multiplicative depth is same as AND-depth."""
        a = Var("a")
        b = Var("b")
        c = Var("c")

        expr = And(And(a, b), c)
        self.assertEqual(multiplicative_depth(expr), 2)
        self.assertEqual(multiplicative_depth(expr), and_depth(expr))

    def test_and_depth_mixed_ops(self):
        """AND-depth with mixed AND/XOR operations."""
        a = Var("a")
        b = Var("b")
        c = Var("c")
        d = Var("d")

        expr = And(Xor(a, b), And(c, d))
        depth = and_depth(expr)
        self.assertEqual(depth, 2)
