from __future__ import annotations

import unittest

from stc.tick_ir import (
    Add,
    BitVecConst,
    BitVecType,
    Shl,
    Sub,
    TickIR,
    Var,
)
from stc.tick_ir_classify_arith import classify_arithmetic


class TestClassifyArithmetic(unittest.TestCase):
    def test_add_zero_right(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8)},
            outputs={"y": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": Add(a=Var(name="x"), b=BitVecConst(width=8, value=0))},
        )
        new_ir, report = classify_arithmetic(ir)
        self.assertEqual(len(new_ir.output_exprs), 1)
        self.assertIsInstance(new_ir.output_exprs["y"], Var)
        self.assertEqual(new_ir.output_exprs["y"].name, "x")
        self.assertEqual(report["Add"], 0)
        self.assertEqual(report["Sub"], 0)

    def test_add_zero_left(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8)},
            outputs={"y": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": Add(a=BitVecConst(width=8, value=0), b=Var(name="x"))},
        )
        new_ir, report = classify_arithmetic(ir)
        self.assertEqual(len(new_ir.output_exprs), 1)
        self.assertIsInstance(new_ir.output_exprs["y"], Var)
        self.assertEqual(new_ir.output_exprs["y"].name, "x")
        self.assertEqual(report["Add"], 0)

    def test_sub_zero(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8)},
            outputs={"y": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": Sub(a=Var(name="x"), b=BitVecConst(width=8, value=0))},
        )
        new_ir, report = classify_arithmetic(ir)
        self.assertEqual(len(new_ir.output_exprs), 1)
        self.assertIsInstance(new_ir.output_exprs["y"], Var)
        self.assertEqual(new_ir.output_exprs["y"].name, "x")
        self.assertEqual(report["Sub"], 0)

    def test_sub_self(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8)},
            outputs={"y": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": Sub(a=Var(name="x"), b=Var(name="x"))},
        )
        new_ir, report = classify_arithmetic(ir)
        self.assertEqual(len(new_ir.output_exprs), 1)
        self.assertIsInstance(new_ir.output_exprs["y"], BitVecConst)
        self.assertEqual(new_ir.output_exprs["y"].width, 8)
        self.assertEqual(new_ir.output_exprs["y"].value, 0)
        self.assertEqual(report["Sub"], 0)

    def test_add_non_zero(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8)},
            outputs={"y": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": Add(a=Var(name="x"), b=BitVecConst(width=8, value=1))},
        )
        new_ir, report = classify_arithmetic(ir)
        self.assertEqual(len(new_ir.output_exprs), 1)
        self.assertIsInstance(new_ir.output_exprs["y"], Add)
        self.assertEqual(report["Add"], 1)

    def test_sub_different(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8), "y": BitVecType(width=8)},
            outputs={"z": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"z": Sub(a=Var(name="x"), b=Var(name="y"))},
        )
        new_ir, report = classify_arithmetic(ir)
        self.assertEqual(len(new_ir.output_exprs), 1)
        self.assertIsInstance(new_ir.output_exprs["z"], Sub)
        self.assertEqual(report["Sub"], 1)

    def test_nested_add_zero_simplification(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8)},
            outputs={"y": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "y": Add(
                    a=Add(a=Var(name="x"), b=BitVecConst(width=8, value=0)),
                    b=BitVecConst(width=8, value=0),
                )
            },
        )
        new_ir, report = classify_arithmetic(ir)
        self.assertEqual(len(new_ir.output_exprs), 1)
        self.assertIsInstance(new_ir.output_exprs["y"], Var)
        self.assertEqual(new_ir.output_exprs["y"].name, "x")
        self.assertEqual(report["Add"], 0)

    def test_multiple_operations(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8), "y": BitVecType(width=8)},
            outputs={
                "a": BitVecType(width=8),
                "b": BitVecType(width=8),
                "c": BitVecType(width=8),
            },
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "a": Add(a=Var(name="x"), b=BitVecConst(width=8, value=0)),
                "b": Sub(a=Var(name="y"), b=BitVecConst(width=8, value=0)),
                "c": Add(a=Var(name="x"), b=Var(name="y")),
            },
        )
        new_ir, report = classify_arithmetic(ir)
        self.assertEqual(len(new_ir.output_exprs), 3)
        self.assertIsInstance(new_ir.output_exprs["a"], Var)
        self.assertIsInstance(new_ir.output_exprs["b"], Var)
        self.assertIsInstance(new_ir.output_exprs["c"], Add)
        self.assertEqual(report["Add"], 1)
        self.assertEqual(report["Sub"], 0)

    def test_register_next_simplification(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8)},
            outputs={"y": BitVecType(width=8)},
            state={"counter": BitVecType(width=8)},
            reset_state={"counter": BitVecConst(width=8, value=0)},
            next_state={
                "counter": Add(a=Var(name="counter"), b=BitVecConst(width=8, value=0))
            },
            output_exprs={"y": Var(name="counter")},
        )
        new_ir, report = classify_arithmetic(ir)
        self.assertEqual(len(new_ir.next_state), 1)
        self.assertIsInstance(new_ir.next_state["counter"], Var)
        self.assertEqual(new_ir.next_state["counter"].name, "counter")
        self.assertEqual(report["Add"], 0)

    def test_wide_bitvec(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=128)},
            outputs={"y": BitVecType(width=128)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": Add(a=Var(name="x"), b=BitVecConst(width=128, value=0))},
        )
        new_ir, report = classify_arithmetic(ir)
        self.assertEqual(len(new_ir.output_exprs), 1)
        self.assertIsInstance(new_ir.output_exprs["y"], Var)
        self.assertEqual(report["Add"], 0)

    def test_preserves_non_arithmetic(self):
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8)},
            outputs={"y": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": Shl(a=Var(name="x"), b=BitVecConst(width=8, value=1))},
        )
        new_ir, report = classify_arithmetic(ir)
        self.assertEqual(len(new_ir.output_exprs), 1)
        self.assertIsInstance(new_ir.output_exprs["y"], Shl)
        self.assertEqual(report["Add"], 0)
        self.assertEqual(report["Sub"], 0)

    def test_classification_counts(self):
        ir = TickIR(
            name="test",
            inputs={
                "a": BitVecType(width=8),
                "b": BitVecType(width=8),
                "c": BitVecType(width=8),
            },
            outputs={
                "x": BitVecType(width=8),
                "y": BitVecType(width=8),
                "z": BitVecType(width=8),
            },
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "x": Add(a=Var(name="a"), b=Var(name="b")),
                "y": Add(a=Var(name="b"), b=Var(name="c")),
                "z": Sub(a=Var(name="a"), b=Var(name="c")),
            },
        )
        new_ir, report = classify_arithmetic(ir)
        self.assertEqual(report["Add"], 2)
        self.assertEqual(report["Sub"], 1)

    def test_simd_shift_operations_handled(self):
        from stc.tick_ir import SimdAShr, SimdLShr, SimdShl, SimdType

        ir = TickIR(
            name="test",
            inputs={
                "s1": SimdType(lane_width=32, lanes=4),
                "s2": SimdType(lane_width=32, lanes=4),
            },
            outputs={
                "out_shl": SimdType(lane_width=32, lanes=4),
                "out_lshr": SimdType(lane_width=32, lanes=4),
                "out_ashr": SimdType(lane_width=32, lanes=4),
            },
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "out_shl": SimdShl(a=Var(name="s1"), sh=Var(name="s2")),
                "out_lshr": SimdLShr(a=Var(name="s1"), sh=Var(name="s2")),
                "out_ashr": SimdAShr(a=Var(name="s1"), sh=Var(name="s2")),
            },
        )
        new_ir, report = classify_arithmetic(ir)
        self.assertIsInstance(new_ir.output_exprs["out_shl"], SimdShl)
        self.assertIsInstance(new_ir.output_exprs["out_lshr"], SimdLShr)
        self.assertIsInstance(new_ir.output_exprs["out_ashr"], SimdAShr)


if __name__ == "__main__":
    unittest.main()
