import unittest

from stc.autovec_pass import autovectorize_tick_ir
from stc.tick_ir import (
    Add,
    Bitcast,
    BitVecType,
    Concat,
    SimdAdd,
    SimdType,
    Slice,
    TickIR,
    Var,
)
from stc.tick_ir_validate import validate_tick_ir


class TestAutovecPass(unittest.TestCase):
    def test_autovec_rewrites_tick_ir_outputs(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        bv8 = BitVecType(width=8)
        types = {"x": simd, "y": simd}

        xb = Bitcast(to=bv8, x=Var("x"))
        yb = Bitcast(to=bv8, x=Var("y"))
        lo = Add(a=Slice(x=xb, offset=0, width=4), b=Slice(x=yb, offset=0, width=4))
        hi = Add(a=Slice(x=xb, offset=4, width=4), b=Slice(x=yb, offset=4, width=4))
        packed = Bitcast(to=simd, x=Concat(parts=[hi, lo]))

        ir = TickIR(
            name="t",
            inputs=types,
            outputs={"o": simd},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": packed},
        )
        validate_tick_ir(ir)
        out = autovectorize_tick_ir(ir, timeout_ms=200)
        self.assertEqual(out.output_exprs["o"], SimdAdd(a=Var("x"), b=Var("y")))
