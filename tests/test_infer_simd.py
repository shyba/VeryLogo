import unittest

from stc.infer_simd import infer_simd_types
from stc.tick_ir import (
    Add,
    BitVecType,
    Concat,
    SimdType,
    Slice,
    TickIR,
    Var,
)


class TestInferSimd(unittest.TestCase):
    def test_infer_simd_from_concat_slices(self) -> None:
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8), "y": BitVecType(width=8)},
            outputs={"o": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "o": Concat(
                    parts=(
                        Add(
                            a=Slice(x=Var(name="x"), offset=4, width=4),
                            b=Slice(x=Var(name="y"), offset=4, width=4),
                        ),
                        Add(
                            a=Slice(x=Var(name="x"), offset=0, width=4),
                            b=Slice(x=Var(name="y"), offset=0, width=4),
                        ),
                    )
                )
            },
        )

        result = infer_simd_types(ir)

        self.assertEqual(result.inputs["x"], SimdType(lane_width=4, lanes=2))
        self.assertEqual(result.inputs["y"], SimdType(lane_width=4, lanes=2))
        self.assertEqual(result.outputs["o"], SimdType(lane_width=4, lanes=2))

    def test_infer_simd_preserves_non_simd_types(self) -> None:
        ir = TickIR(
            name="test",
            inputs={"x": BitVecType(width=8)},
            outputs={"o": BitVecType(width=8)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Var(name="x")},
        )

        result = infer_simd_types(ir)

        self.assertEqual(result.inputs["x"], BitVecType(width=8))
        self.assertEqual(result.outputs["o"], BitVecType(width=8))

    def test_collect_slices_handles_tuples(self) -> None:
        from stc.infer_simd import _collect_slices

        expr = Concat(
            parts=(
                Slice(x=Var(name="x"), offset=4, width=4),
                Slice(x=Var(name="x"), offset=0, width=4),
            )
        )

        slice_uses = {}
        _collect_slices(expr, slice_uses)

        self.assertEqual(slice_uses, {"x": [(4, 4), (0, 4)]})


if __name__ == "__main__":
    unittest.main()
