import unittest

from stc.passes.balance import (
    BalanceAssociativePass,
    collect_assoc_leaves,
    build_balanced_tree,
)
from stc.passes.depth_resynth import DepthResynthesisPass
from stc.passes.depth_ternary import (
    DepthAwareTernaryPass,
    DepthBudgetPass,
    map_critical_path_to_ternary,
    map_circuit_critical_path_to_ternary,
    _expr_depth,
)
from stc.passmgr import PassContext
from stc.tech import GenericTechnology, PTXTechnology
from stc.tick_ir import TickIR, Xor, And, Or, Var, BoolType, TernaryLut
from stc.metrics import compute_depth
from stc.circuit_synth import CircuitState


class TestBalanceAssociativePass(unittest.TestCase):
    def test_collect_xor_leaves(self):
        """Collect leaves from XOR chain."""
        a = Var("a")
        b = Var("b")
        c = Var("c")
        d = Var("d")

        expr = Xor(Xor(Xor(a, b), c), d)
        leaves = collect_assoc_leaves(expr, Xor)

        self.assertEqual(len(leaves), 4)
        self.assertIn(a, leaves)
        self.assertIn(b, leaves)
        self.assertIn(c, leaves)
        self.assertIn(d, leaves)

    def test_build_balanced_tree(self):
        """Build balanced tree from leaves."""
        a = Var("a")
        b = Var("b")
        c = Var("c")
        d = Var("d")

        leaves = [a, b, c, d]
        balanced = build_balanced_tree(Xor, leaves)

        self.assertIsInstance(balanced, Xor)
        self.assertIsInstance(balanced.a, Xor)
        self.assertIsInstance(balanced.b, Xor)

    def test_balance_pass_chain(self):
        """Balance pass reduces depth of XOR chain."""
        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
        )

        a = Var("a")
        b = Var("b")
        c = Var("c")
        d = Var("d")

        unbalanced = Xor(Xor(Xor(a, b), c), d)
        ir = TickIR(
            name="test",
            inputs={"a": BoolType(), "b": BoolType(), "c": BoolType(), "d": BoolType()},
            outputs={"o": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": unbalanced},
        )

        pass_instance = BalanceAssociativePass()
        new_ir, metrics = pass_instance.run(ir, ctx)

        self.assertTrue(metrics.changed)
        self.assertEqual(metrics.expressions_modified, 1)

        original_depth = compute_depth(unbalanced, ctx.depth_model)
        new_depth = compute_depth(new_ir.output_exprs["o"], ctx.depth_model)

        self.assertLess(new_depth, original_depth)

    def test_balance_pass_no_change(self):
        """Balance pass doesn't change already balanced expressions."""
        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
        )

        a = Var("a")
        b = Var("b")

        expr = Xor(a, b)
        ir = TickIR(
            name="test",
            inputs={"a": BoolType(), "b": BoolType()},
            outputs={"o": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )

        pass_instance = BalanceAssociativePass()
        new_ir, metrics = pass_instance.run(ir, ctx)

        self.assertFalse(metrics.changed)


class TestDepthResynthesisPass(unittest.TestCase):
    def test_pass_exists(self):
        """DepthResynthesisPass can be instantiated."""
        pass_instance = DepthResynthesisPass()
        self.assertEqual(pass_instance.name, "depth-resynth")

    def test_should_not_run_without_depth_budget(self):
        """Pass should not run if no depth budget set."""
        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
            depth_budget=None,
        )

        pass_instance = DepthResynthesisPass()
        self.assertFalse(pass_instance.should_run(ctx))

    def test_should_run_with_depth_budget(self):
        """Pass should run if depth budget set."""
        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
            depth_budget=10,
        )

        pass_instance = DepthResynthesisPass()
        self.assertTrue(pass_instance.should_run(ctx))

    def test_no_change_placeholder(self):
        """Pass returns unchanged IR (placeholder implementation)."""
        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
            depth_budget=10,
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

        pass_instance = DepthResynthesisPass()
        new_ir, metrics = pass_instance.run(ir, ctx)

        self.assertFalse(metrics.changed)
        self.assertEqual(new_ir, ir)


class TestDepthAwareTernaryPass(unittest.TestCase):
    def test_pass_exists(self):
        """DepthAwareTernaryPass can be instantiated."""
        pass_instance = DepthAwareTernaryPass()
        self.assertEqual(pass_instance.name, "depth-ternary")

    def test_should_run_on_ptx(self):
        """Pass should run on PTX technology."""
        tech = PTXTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
        )

        pass_instance = DepthAwareTernaryPass()
        self.assertTrue(pass_instance.should_run(ctx))

    def test_should_not_run_on_generic(self):
        """Pass should not run on generic technology (no ternary support)."""
        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
        )

        pass_instance = DepthAwareTernaryPass()
        self.assertFalse(pass_instance.should_run(ctx))

    def test_map_critical_path_reduces_depth(self):
        """Mapping critical path to ternary reduces depth."""
        a = Var("a")
        b = Var("b")
        c = Var("c")

        deep_expr = And(And(a, b), c)
        original_depth = _expr_depth(deep_expr)

        new_expr, changed = map_critical_path_to_ternary(deep_expr)

        if changed:
            new_depth = _expr_depth(new_expr)
            self.assertLessEqual(new_depth, original_depth)

    def test_expr_depth_calculation(self):
        """Test expression depth calculation."""
        a = Var("a")
        b = Var("b")
        c = Var("c")

        self.assertEqual(_expr_depth(a), 0)

        simple = And(a, b)
        self.assertEqual(_expr_depth(simple), 1)

        chain = And(And(a, b), c)
        self.assertEqual(_expr_depth(chain), 2)

        lut = TernaryLut(a=a, b=b, c=c, imm8=0x80)
        self.assertEqual(_expr_depth(lut), 1)


class TestDepthBudgetPass(unittest.TestCase):
    def test_pass_exists(self):
        """DepthBudgetPass can be instantiated."""
        pass_instance = DepthBudgetPass()
        self.assertEqual(pass_instance.name, "depth-budget")

    def test_should_run_with_budget(self):
        """Pass should run when depth budget is set."""
        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
            depth_budget=5,
        )

        pass_instance = DepthBudgetPass()
        self.assertTrue(pass_instance.should_run(ctx))

    def test_should_not_run_without_budget(self):
        """Pass should not run when no depth budget."""
        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
            depth_budget=None,
        )

        pass_instance = DepthBudgetPass()
        self.assertFalse(pass_instance.should_run(ctx))

    def test_respects_depth_budget(self):
        """Pass allows IR within depth budget."""
        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
            depth_budget=10,
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

        pass_instance = DepthBudgetPass()
        new_ir, metrics = pass_instance.run(ir, ctx)

        self.assertFalse(metrics.changed)
        self.assertEqual(new_ir, ir)


class TestCircuitStateCriticalPath(unittest.TestCase):
    def test_critical_path_nodes_simple(self):
        """Test critical path identification on simple circuit."""
        circuit = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[("xor", 0, 1)],
            outputs=[(2, False)],
            gate_count=1,
        )

        critical = circuit.critical_path_nodes()
        self.assertIn(2, critical)

    def test_critical_path_nodes_chain(self):
        """Test critical path on chain of gates."""
        circuit = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("and", 3, 2),
            ],
            outputs=[(4, False)],
            gate_count=2,
        )

        critical = circuit.critical_path_nodes()
        self.assertIn(4, critical)
        self.assertIn(3, critical)

    def test_node_depths(self):
        """Test node depth computation."""
        circuit = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("and", 2, 0),
            ],
            outputs=[(3, False)],
            gate_count=2,
        )

        depths = circuit.node_depths()
        self.assertEqual(depths[0], 0)
        self.assertEqual(depths[1], 0)
        self.assertEqual(depths[2], 1)
        self.assertEqual(depths[3], 2)

    def test_optimize_for_depth_respects_constraint(self):
        """Test optimize_for_depth respects max_depth_increase."""
        circuit = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("and", 2, 2),
            ],
            outputs=[(3, False)],
            gate_count=2,
        )

        original_depth = circuit.depth
        optimized = circuit.optimize_for_depth(max_depth_increase=0)

        self.assertLessEqual(optimized.depth, original_depth)

    def test_depth_property(self):
        """Test depth property calculation."""
        circuit = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("and", 2, 0),
                ("xor", 3, 1),
            ],
            outputs=[(4, False)],
            gate_count=3,
        )

        self.assertEqual(circuit.depth, 3)

    def test_multiplicative_depth(self):
        """Test multiplicative depth (AND-only) calculation."""
        circuit = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("and", 2, 0),
                ("and", 3, 1),
            ],
            outputs=[(4, False)],
            gate_count=3,
        )

        self.assertEqual(circuit.multiplicative_depth, 2)

    def test_map_circuit_critical_path_to_ternary(self):
        """Test map_circuit_critical_path_to_ternary function."""
        circuit = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("and", 0, 1),
                ("xor", 3, 2),
            ],
            outputs=[(4, False)],
            gate_count=2,
        )

        original_depth = circuit.depth
        result = map_circuit_critical_path_to_ternary(circuit)

        self.assertLessEqual(result.depth, original_depth)

    def test_map_circuit_preserves_correctness(self):
        """Test that ternary mapping preserves circuit correctness."""
        circuit = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("and", 0, 1),
                ("xor", 3, 2),
            ],
            outputs=[(4, False)],
            gate_count=2,
        )

        result = map_circuit_critical_path_to_ternary(circuit)

        for x in range(8):
            expected = circuit.evaluate(x)
            actual = result.evaluate(x)
            self.assertEqual(expected, actual, f"Mismatch at input {x}")
