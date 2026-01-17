import unittest

import z3

from stc.interp import eval_expr
from stc.superopt import superopt_expr
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
from stc.tick_ir_to_verilog import emit_verilog
from stc.z3_encode import encode_expr


class TestSimdSemanticsAndSuperopt(unittest.TestCase):
    def test_simd_add_interpreter_lane_wrap(self) -> None:
        t = SimdType(lane_width=4, lanes=2)
        types = {"x": t, "y": t}
        expr = SimdAdd(a=Var(name="x"), b=Var(name="y"))
        out = eval_expr(expr, types, {"x": 0xF1, "y": 0x13})
        self.assertEqual(out, 0x04)

    def test_simd_add_zero_equiv_in_z3(self) -> None:
        t = SimdType(lane_width=4, lanes=2)
        types = {"x": t}
        spec = SimdAdd(a=Var(name="x"), b=SimdConst(lane_width=4, lanes=2, value=0))
        cand = Var(name="x")
        z3_vars: dict[str, z3.ExprRef] = {}
        spec_z = encode_expr(spec, types, z3_vars)
        cand_z = encode_expr(cand, types, z3_vars)
        s = z3.Solver()
        s.add(spec_z != cand_z)
        self.assertEqual(s.check(), z3.unsat)

    def test_superopt_finds_identity_for_simd_add_zero(self) -> None:
        t = SimdType(lane_width=4, lanes=2)
        types = {"x": t}
        spec = SimdAdd(a=Var(name="x"), b=SimdConst(lane_width=4, lanes=2, value=0))
        best = superopt_expr(
            spec,
            types,
            max_nodes=3,
            constants=[SimdConst(lane_width=4, lanes=2, value=0)],
        )
        self.assertEqual(best, Var(name="x"))

    def test_superopt_autovectorizes_scalar_lanes(self) -> None:
        simd = SimdType(lane_width=4, lanes=2)
        bv8 = BitVecType(width=8)
        bv4 = BitVecType(width=4)
        types = {"x": simd, "y": simd}

        xb = Bitcast(to=bv8, x=Var(name="x"))
        yb = Bitcast(to=bv8, x=Var(name="y"))
        lo = Add(a=Slice(x=xb, offset=0, width=4), b=Slice(x=yb, offset=0, width=4))
        hi = Add(a=Slice(x=xb, offset=4, width=4), b=Slice(x=yb, offset=4, width=4))
        packed = Concat(parts=[hi, lo])
        spec = Bitcast(to=simd, x=packed)

        best = superopt_expr(spec, types, max_nodes=3)
        self.assertEqual(best, SimdAdd(a=Var(name="x"), b=Var(name="y")))

    def test_verilog_emits_simd_ports(self) -> None:
        t = SimdType(lane_width=4, lanes=2)
        ir = TickIR(
            name="simd",
            inputs={"x": t},
            outputs={"o": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "o": SimdAdd(
                    a=Var(name="x"), b=SimdConst(lane_width=4, lanes=2, value=0)
                )
            },
        )
        v = emit_verilog(ir, module_name="tickir_impl")
        self.assertIn("input logic [7:0] x", v)
        self.assertIn("output logic [7:0] o", v)
        self.assertIn("assign o =", v)
