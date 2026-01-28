import json
import tempfile
import unittest
from pathlib import Path

from stc.lowering_coordinator import coordinate_lowering, write_lowering_choice_report
from stc.tick_ir import BitVecConst, BitVecType, TickIR, Var, Xor


class TestCliLoweringReport(unittest.TestCase):
    def test_avx512_generates_lowering_report(self) -> None:
        # Create a simple TickIR design that should succeed with packed lowering.
        ir = TickIR(
            name="xor_test",
            inputs={"x": BitVecType(128)},
            outputs={"o": BitVecType(128)},
            state={"s": BitVecType(128)},
            reset_state={"s": BitVecConst(width=128, value=0)},
            next_state={"s": Xor(a=Var("x"), b=Var("s"))},
            output_exprs={"o": Var("s")},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "out"

            _circuit, _layout, choice = coordinate_lowering(ir, prefer_packed=True)
            write_lowering_choice_report(choice, out_dir)

            report_path = out_dir / "lowering_choice.json"
            self.assertTrue(
                report_path.exists(),
                "lowering_choice.json should be generated for AVX-512 backend",
            )

            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)

            self.assertIn("path", report)
            self.assertIn("reason", report)
            self.assertIn("unsupported", report)
            self.assertIn("stats", report)

            self.assertEqual(report["path"], "packed")
            self.assertIsInstance(report["unsupported"], list)
            self.assertEqual(len(report["unsupported"]), 0)

            stats = report["stats"]
            self.assertIn("input_bits", stats)
            self.assertIn("state_bits", stats)
            self.assertIn("output_bits", stats)
            self.assertIn("gates", stats)
            self.assertEqual(stats["input_bits"], 128)
            self.assertEqual(stats["state_bits"], 128)
            self.assertEqual(stats["output_bits"], 128)

    def test_report_format_validation(self) -> None:
        # Validate that the report JSON format matches the spec.
        ir = TickIR(
            name="simple",
            inputs={"x": BitVecType(64)},
            outputs={"o": BitVecType(64)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Var("x")},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "out"

            _circuit, _layout, choice = coordinate_lowering(ir, prefer_packed=True)
            write_lowering_choice_report(choice, out_dir)

            report_path = out_dir / "lowering_choice.json"
            self.assertTrue(report_path.exists())

            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)

            required_fields = ["path", "reason", "unsupported", "stats"]
            for field in required_fields:
                self.assertIn(field, report, f"Missing required field: {field}")

            self.assertIn(report["path"], ["packed", "bit"])
            self.assertIsInstance(report["reason"], str)
            self.assertIsInstance(report["unsupported"], list)
            self.assertIsInstance(report["stats"], dict)

            required_stats = [
                "input_bits",
                "state_bits",
                "output_bits",
                "gates",
                "expr_nodes",
                "expr_depth",
            ]
            for stat in required_stats:
                self.assertIn(stat, report["stats"], f"Missing stat field: {stat}")


if __name__ == "__main__":
    unittest.main()
