import unittest

from stc.tech import (
    GenericTechnology,
    get_technology,
    list_technologies,
    register_technology,
)


class TestTechnology(unittest.TestCase):
    def test_generic_registration(self):
        """Generic technology is registered by default."""
        tech = get_technology("generic")
        self.assertEqual(tech.name, "generic")

    def test_cost_model_protocol(self):
        """Cost model satisfies protocol."""
        tech = GenericTechnology()
        cm = tech.cost_model()
        self.assertIsInstance(cm.and_weight(), float)
        self.assertIsInstance(cm.xor_weight(), float)

    def test_depth_model_protocol(self):
        """Depth model satisfies protocol."""
        tech = GenericTechnology()
        dm = tech.depth_model()
        self.assertEqual(dm.op_depth("and"), 1)
        self.assertEqual(dm.op_depth("not"), 0)
        self.assertTrue(dm.is_free("not"))

    def test_list_technologies(self):
        """Can list registered technologies."""
        techs = list_technologies()
        self.assertIn("generic", techs)

    def test_unknown_technology(self):
        """Getting unknown technology raises ValueError."""
        with self.assertRaises(ValueError):
            get_technology("nonexistent")

    def test_primitives(self):
        """Generic technology has basic primitives."""
        tech = GenericTechnology()
        prims = tech.primitives()
        prim_names = {p.name for p in prims}
        self.assertIn("and", prim_names)
        self.assertIn("or", prim_names)
        self.assertIn("xor", prim_names)
        self.assertIn("not", prim_names)

    def test_is_legal(self):
        """Generic technology accepts all expressions."""
        from stc.tick_ir import And, BoolConst

        tech = GenericTechnology()
        expr = And(a=BoolConst(value=True), b=BoolConst(value=False))
        self.assertTrue(tech.is_legal(expr))
