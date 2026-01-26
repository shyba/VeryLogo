import json
import tempfile
import unittest
from pathlib import Path

from stc.lowering_coordinator import (
    LoweringChoice,
    coordinate_lowering,
    write_lowering_choice_report,
)
from stc.packed_circuit import PackedCircuitState
from stc.tick_ir_to_packed_circuit_state import lower_tick_ir_to_packed_circuit_state
from stc.tick_ir import (
    BitVecConst,
    BitVecType,
    TickIR,
    Var,
    Xor,
)


class TestLoweringCoordinator(unittest.TestCase):
    def test_successful_packed_lowering(self) -> None:
        # Simple design that should succeed with packed lowering.
        ir = TickIR(
            name="simple_xor",
            inputs={"x": BitVecType(128)},
            outputs={"o": BitVecType(128)},
            state={"s": BitVecType(128)},
            reset_state={"s": BitVecConst(width=128, value=0)},
            next_state={"s": Xor(a=Var("x"), b=Var("s"))},
            output_exprs={"o": Xor(a=Var("x"), b=Var("s"))},
        )

        circuit, layout, choice = coordinate_lowering(ir)

        self.assertIsInstance(circuit, PackedCircuitState)
        self.assertEqual(choice.path, "packed")
        self.assertEqual(choice.unsupported, [])
        self.assertIn("Successfully lowered", choice.reason)
        self.assertIn("input_bits", choice.stats)
        self.assertEqual(choice.stats["input_bits"], 128)
        self.assertEqual(choice.stats["state_bits"], 128)
        self.assertEqual(choice.stats["output_bits"], 128)

    def test_packed_lowering_error_details(self) -> None:
        # Test that PackedLoweringError includes detailed context.
        from stc.tick_ir_to_packed_circuit_state import PackedLoweringError

        try:
            ir = TickIR(
                name="invalid_xor",
                inputs={"x": BitVecType(64)},
                outputs={"o": BitVecType(128)},
                state={"s": BitVecType(128)},
                reset_state={"s": BitVecConst(width=128, value=0)},
                next_state={"s": Xor(a=Var("x"), b=Var("s"))},
                output_exprs={"o": Xor(a=Var("x"), b=Var("s"))},
            )
            _circuit, _layout = lower_tick_ir_to_packed_circuit_state(ir)
            self.fail("Expected PackedLoweringError")
        except PackedLoweringError as e:
            self.assertIn("width mismatch", e.message)
            self.assertEqual(e.expression_type, "Xor")
            self.assertIsNotNone(e.subexpression_context)
            error_str = str(e)
            self.assertIn("Expression type:", error_str)
            self.assertIn("Suggestion:", error_str)

    def test_write_lowering_choice_report(self) -> None:
        choice = LoweringChoice(
            path="packed",
            reason="Test reason",
            unsupported=[],
            stats={
                "input_bits": 128,
                "state_bits": 64,
                "output_bits": 32,
                "gates": 100,
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            write_lowering_choice_report(choice, out_dir)

            report_path = out_dir / "lowering_choice.json"
            self.assertTrue(report_path.exists())

            with open(report_path, encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["path"], "packed")
            self.assertEqual(data["reason"], "Test reason")
            self.assertEqual(data["unsupported"], [])
            self.assertEqual(data["stats"]["input_bits"], 128)
            self.assertEqual(data["stats"]["gates"], 100)

    def test_lowering_choice_to_dict(self) -> None:
        choice = LoweringChoice(
            path="bit",
            reason="Fallback due to Add",
            unsupported=["Add", "Sub"],
            stats={"input_bits": 64},
        )

        data = choice.to_dict()

        self.assertEqual(data["path"], "bit")
        self.assertEqual(data["reason"], "Fallback due to Add")
        self.assertEqual(data["unsupported"], ["Add", "Sub"])
        self.assertEqual(data["stats"]["input_bits"], 64)

    def test_stats_include_all_fields(self) -> None:
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(128)},
            outputs={"o": BitVecType(128)},
            state={"s": BitVecType(128)},
            reset_state={"s": BitVecConst(width=128, value=0)},
            next_state={"s": Var("x")},
            output_exprs={"o": Var("s")},
        )

        _circuit, _layout, choice = coordinate_lowering(ir)

        required_stats = [
            "input_bits",
            "state_bits",
            "output_bits",
            "gates",
            "expr_nodes",
            "expr_depth",
        ]
        for field in required_stats:
            self.assertIn(field, choice.stats)
            self.assertIsInstance(choice.stats[field], int)


if __name__ == "__main__":
    unittest.main()
