import unittest

from stc.mapping.ternary import TernaryMappingPass, IMM8_XOR_ABC, IMM8_AND_AB
from stc.passmgr import PassContext
from stc.tech import PTXTechnology, AVX512Technology, GenericTechnology
from stc.tick_ir import TickIR, Xor, And, Or, Var, BoolType, TernaryLut


class TestTernaryMappingPass(unittest.TestCase):
    def test_pass_exists(self):
        """TernaryMappingPass can be instantiated."""
        pass_instance = TernaryMappingPass()
        self.assertEqual(pass_instance.name, "ternary-mapping")

    def test_should_run_with_ptx(self):
        """Pass should run with PTX technology."""
        tech = PTXTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
        )

        pass_instance = TernaryMappingPass()
        self.assertTrue(pass_instance.should_run(ctx))

    def test_should_run_with_avx512(self):
        """Pass should run with AVX-512 technology."""
        tech = AVX512Technology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
        )

        pass_instance = TernaryMappingPass()
        self.assertTrue(pass_instance.should_run(ctx))

    def test_should_not_run_with_generic(self):
        """Pass should not run with generic technology."""
        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
        )

        pass_instance = TernaryMappingPass()
        self.assertFalse(pass_instance.should_run(ctx))

    def test_maps_xor_chain(self):
        """Pass maps XOR chain to TernaryLut."""
        tech = PTXTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
        )

        a = Var("a")
        b = Var("b")
        c = Var("c")

        xor_chain = Xor(Xor(a, b), c)
        ir = TickIR(
            name="test",
            inputs={"a": BoolType(), "b": BoolType(), "c": BoolType()},
            outputs={"o": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": xor_chain},
        )

        pass_instance = TernaryMappingPass(min_gate_savings=1)
        new_ir, metrics = pass_instance.run(ir, ctx)

        self.assertTrue(metrics.changed)
        self.assertEqual(metrics.expressions_modified, 1)
        self.assertIsInstance(new_ir.output_exprs["o"], TernaryLut)

    def test_profitability_check(self):
        """Pass respects min_gate_savings profitability threshold."""
        tech = PTXTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
        )

        a = Var("a")
        b = Var("b")

        expr = And(a, b)
        ir = TickIR(
            name="test",
            inputs={"a": BoolType(), "b": BoolType()},
            outputs={"o": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )

        pass_instance = TernaryMappingPass(min_gate_savings=10)
        new_ir, metrics = pass_instance.run(ir, ctx)

        self.assertFalse(metrics.changed)

    def test_maps_multiple_outputs(self):
        """Pass maps multiple independent cones."""
        tech = PTXTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
        )

        a = Var("a")
        b = Var("b")
        c = Var("c")
        d = Var("d")

        out1 = Xor(Xor(a, b), c)
        out2 = And(And(d, c), b)

        ir = TickIR(
            name="test",
            inputs={
                "a": BoolType(),
                "b": BoolType(),
                "c": BoolType(),
                "d": BoolType(),
            },
            outputs={"o1": BoolType(), "o2": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o1": out1, "o2": out2},
        )

        pass_instance = TernaryMappingPass(min_gate_savings=1)
        new_ir, metrics = pass_instance.run(ir, ctx)

        self.assertTrue(metrics.changed)
        self.assertEqual(metrics.expressions_modified, 2)
        self.assertIsInstance(new_ir.output_exprs["o1"], TernaryLut)
        self.assertIsInstance(new_ir.output_exprs["o2"], TernaryLut)

    def test_preserves_semantics(self):
        """Pass preserves expression semantics."""
        from stc.interp import eval_expr

        tech = PTXTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
        )

        a = Var("a")
        b = Var("b")
        c = Var("c")

        original = Xor(Xor(a, b), c)
        ir = TickIR(
            name="test",
            inputs={"a": BoolType(), "b": BoolType(), "c": BoolType()},
            outputs={"o": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": original},
        )

        pass_instance = TernaryMappingPass()
        new_ir, metrics = pass_instance.run(ir, ctx)

        for a_val in [0, 1]:
            for b_val in [0, 1]:
                for c_val in [0, 1]:
                    env = {"a": a_val, "b": b_val, "c": c_val}
                    types = ir.inputs

                    orig_result = eval_expr(original, types, env)
                    new_result = eval_expr(new_ir.output_exprs["o"], types, env)

                    self.assertEqual(
                        orig_result,
                        new_result,
                        f"Mismatch for a={a_val}, b={b_val}, c={c_val}",
                    )
