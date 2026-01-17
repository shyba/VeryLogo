import unittest

import z3

from stc.interp import eval_expr
from stc.tick_ir import (
    SimdAdd,
    SimdBlend,
    SimdSub,
    SimdType,
    SimdUlt,
    TickIR,
    Var,
)
from stc.tick_ir_validate import validate_tick_ir
from stc.z3_encode import encode_expr


class TestSimdMaskedPredication(unittest.TestCase):
    def test_masked_add_equals_blend_of_add(self) -> None:
        from stc.tick_ir import SimdAddMasked

        t = SimdType(lane_width=32, lanes=16)
        mask = SimdUlt(a=Var("x"), b=Var("y"))
        spec = SimdBlend(mask=mask, a=Var("x"), b=SimdAdd(a=Var("x"), b=Var("y")))
        cand = SimdAddMasked(mask=mask, passthru=Var("x"), a=Var("x"), b=Var("y"))

        types = {"x": t, "y": t}

        z3_vars: dict[str, z3.ExprRef] = {}
        sz = encode_expr(spec, types, z3_vars)
        cz = encode_expr(cand, types, z3_vars)
        s = z3.Solver()
        s.set(timeout=200)
        s.add(sz != cz)
        self.assertEqual(s.check(), z3.unsat)

        envs = [
            {"x": 0, "y": 0},
            {"x": int("0000000100000002" * 8, 16), "y": int("00000003" * 16, 16)},
            {"x": (1 << 512) - 1, "y": 0},
        ]
        for env in envs:
            self.assertEqual(
                int(eval_expr(spec, types, env)),
                int(eval_expr(cand, types, env)),
            )

    def test_masked_sub_equals_blend_of_sub(self) -> None:
        from stc.tick_ir import SimdSubMasked

        t = SimdType(lane_width=32, lanes=16)
        mask = SimdUlt(a=Var("x"), b=Var("y"))
        spec = SimdBlend(mask=mask, a=Var("x"), b=SimdSub(a=Var("x"), b=Var("y")))
        cand = SimdSubMasked(mask=mask, passthru=Var("x"), a=Var("x"), b=Var("y"))

        types = {"x": t, "y": t}

        z3_vars: dict[str, z3.ExprRef] = {}
        sz = encode_expr(spec, types, z3_vars)
        cz = encode_expr(cand, types, z3_vars)
        s = z3.Solver()
        s.set(timeout=200)
        s.add(sz != cz)
        self.assertEqual(s.check(), z3.unsat)

        envs = [
            {"x": 0, "y": 0},
            {"x": int("0000000100000002" * 8, 16), "y": int("00000003" * 16, 16)},
            {"x": 0, "y": (1 << 512) - 1},
        ]
        for env in envs:
            self.assertEqual(
                int(eval_expr(spec, types, env)),
                int(eval_expr(cand, types, env)),
            )

    def test_tick_ir_json_roundtrip(self) -> None:
        from stc.tick_ir import SimdAddMasked

        t = SimdType(lane_width=32, lanes=16)
        mask = SimdUlt(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="madd",
            inputs={"x": t, "y": t},
            outputs={"z": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "z": SimdAddMasked(mask=mask, passthru=Var("x"), a=Var("x"), b=Var("y"))
            },
        )
        validate_tick_ir(ir)
        rt = TickIR.from_dict(ir.to_dict())
        self.assertEqual(rt.to_dict(), ir.to_dict())
