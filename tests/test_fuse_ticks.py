import random
import re
import unittest

from stc.fuse_ticks import FuseBudget, fuse_ticks
from stc.testing.tickir_steps import step_tickir
from stc.tick_ir import Add, BitVecConst, BitVecType, BoolType, TickIR, Var
from stc.tick_ir_validate import validate_tick_ir


def _fused_steps(name: str) -> int:
    m = re.search(r"__fused(\d+)$", name)
    if not m:
        raise AssertionError(f"expected fused name suffix, got: {name}")
    return int(m.group(1))


class TestFuseTicks(unittest.TestCase):
    def test_equivalence_replicate_inputs(self) -> None:
        bv8 = BitVecType(width=8)
        ir = TickIR(
            name="accum",
            inputs={"i": bv8},
            outputs={"o": bv8},
            state={"s": bv8},
            reset_state={"s": BitVecConst(width=8, value=0)},
            next_state={"s": Add(a=Var("s"), b=Var("i"))},
            output_exprs={"o": Var("s")},
        )
        validate_tick_ir(ir)

        n = 6
        fused = fuse_ticks(
            ir,
            n,
            mode="final",
            input_policy="replicate",
            budget=FuseBudget(max_nodes=10_000, max_depth=10_000),
        )
        validate_tick_ir(fused)
        self.assertEqual(_fused_steps(fused.name), n)

        rng = random.Random(123)
        for _ in range(50):
            st = {"s": rng.randrange(256)}
            inputs_seq = [{"i": rng.randrange(256)} for _ in range(n)]

            cur = dict(st)
            last_out = None
            for k in range(n):
                step = step_tickir(ir, state=cur, inputs=inputs_seq[k])
                cur = step.next_state
                last_out = step.outputs
            assert last_out is not None

            fused_inputs = {f"i__t{k}": inputs_seq[k]["i"] for k in range(n)}
            fused_step = step_tickir(fused, state=st, inputs=fused_inputs)
            self.assertEqual(fused_step.next_state, cur)
            self.assertEqual(int(fused_step.outputs["o"]), int(last_out["o"]))

    def test_budget_stops_early(self) -> None:
        bv8 = BitVecType(width=8)
        ir = TickIR(
            name="accum_budget",
            inputs={"i": bv8},
            outputs={"o": bv8},
            state={"s": bv8},
            reset_state={"s": BitVecConst(width=8, value=0)},
            next_state={"s": Add(a=Var("s"), b=Var("i"))},
            output_exprs={"o": Var("s")},
        )
        validate_tick_ir(ir)

        fused = fuse_ticks(
            ir,
            20,
            mode="final",
            input_policy="replicate",
            budget=FuseBudget(max_nodes=20, max_depth=10_000, stop_on_budget_hit=True),
        )
        validate_tick_ir(fused)
        steps = _fused_steps(fused.name)
        self.assertGreaterEqual(steps, 1)
        self.assertLess(steps, 20)

    def test_deterministic(self) -> None:
        bv8 = BitVecType(width=8)
        ir = TickIR(
            name="accum_det",
            inputs={"i": bv8},
            outputs={"o": bv8},
            state={"s": bv8},
            reset_state={"s": BitVecConst(width=8, value=0)},
            next_state={"s": Add(a=Var("s"), b=Var("i"))},
            output_exprs={"o": Var("s")},
        )
        validate_tick_ir(ir)

        a = fuse_ticks(
            ir,
            8,
            mode="final",
            input_policy="replicate",
            budget=FuseBudget(max_nodes=10_000, max_depth=10_000),
        )
        b = fuse_ticks(
            ir,
            8,
            mode="final",
            input_policy="replicate",
            budget=FuseBudget(max_nodes=10_000, max_depth=10_000),
        )
        self.assertEqual(a, b)

    def test_equivalence_with_sync_reset(self) -> None:
        bv8 = BitVecType(width=8)
        ir = TickIR(
            name="accum_rst",
            inputs={"i": bv8, "reset": BoolType()},
            outputs={"o": bv8},
            state={"s": bv8},
            reset_state={"s": BitVecConst(width=8, value=0)},
            next_state={"s": Add(a=Var("s"), b=Var("i"))},
            output_exprs={"o": Var("s")},
        )
        validate_tick_ir(ir)

        n = 5
        fused = fuse_ticks(
            ir,
            n,
            mode="final",
            input_policy="replicate",
            budget=FuseBudget(max_nodes=50_000, max_depth=50_000),
        )
        validate_tick_ir(fused)

        rng = random.Random(999)
        for _ in range(50):
            st0 = {"s": rng.randrange(256)}
            inputs_seq = []
            for _k in range(n):
                inputs_seq.append(
                    {
                        "i": rng.randrange(256),
                        "reset": rng.choice([False, False, False, True]),
                    }
                )

            cur = dict(st0)
            last_out = None
            for k in range(n):
                step = step_tickir(ir, state=cur, inputs=inputs_seq[k])
                cur = step.next_state
                last_out = step.outputs
            assert last_out is not None

            fused_inputs = {f"i__t{k}": inputs_seq[k]["i"] for k in range(n)}
            fused_inputs.update(
                {f"reset__t{k}": inputs_seq[k]["reset"] for k in range(n)}
            )
            fused_step = step_tickir(fused, state=st0, inputs=fused_inputs)
            self.assertEqual(fused_step.next_state, cur)
            self.assertEqual(int(fused_step.outputs["o"]), int(last_out["o"]))
