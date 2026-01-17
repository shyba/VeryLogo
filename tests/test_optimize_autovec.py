import unittest

from stc.reduce import optimize_tick_ir
from stc.tick_ir import (
    Add,
    Bitcast,
    BitVecType,
    Concat,
    SimdAdd,
    SimdConst,
    SimdType,
    Slice,
    TickIR,
    Var,
)


class TestOptimizeAutovec(unittest.TestCase):
    def test_optimize_can_autovec_outputs(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        bv8 = BitVecType(width=8)

        xb = Bitcast(to=bv8, x=Var("x"))
        yb = Bitcast(to=bv8, x=Var("y"))
        lo = Add(a=Slice(x=xb, offset=0, width=4), b=Slice(x=yb, offset=0, width=4))
        hi = Add(a=Slice(x=xb, offset=4, width=4), b=Slice(x=yb, offset=4, width=4))
        packed = Bitcast(to=simd, x=Concat(parts=[hi, lo]))

        ir = TickIR(
            name="t",
            inputs={"x": simd, "y": simd},
            outputs={"o": simd},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": packed},
        )
        out = optimize_tick_ir(ir, bound=1, autovec=True)
        self.assertEqual(out.output_exprs["o"], SimdAdd(a=Var("x"), b=Var("y")))
