import unittest

from stc.control_specialize import specialize_tick_ir_steps
from stc.interp import TickState, tick
from stc.tick_ir import (
    Add,
    BitVecConst,
    BitVecType,
    Eq,
    Expr,
    Mux,
    TickIR,
    Var,
)
from stc.tick_ir_validate import EXPR_CLASSES


def _count_mux(expr: Expr) -> int:
    if isinstance(expr, Mux):
        return 1 + _count_mux(expr.cond) + _count_mux(expr.a) + _count_mux(expr.b)

    if isinstance(expr, (Var, BitVecConst)):
        return 0

    child_exprs: list[Expr] = []
    for field_name in getattr(expr, "__dataclass_fields__", {}):
        value = getattr(expr, field_name)
        if isinstance(value, (tuple, list)):
            child_exprs.extend([v for v in value if isinstance(v, EXPR_CLASSES)])
        elif isinstance(value, EXPR_CLASSES):
            child_exprs.append(value)

    return sum(_count_mux(child) for child in child_exprs)


class TestControlSpecialize(unittest.TestCase):
    def test_specialize_toy_mux_fsm(self) -> None:
        ir = TickIR(
            name="toy",
            inputs={"x": BitVecType(width=8)},
            outputs={"y": BitVecType(width=8)},
            state={"round": BitVecType(width=4), "acc": BitVecType(width=8)},
            reset_state={
                "round": BitVecConst(width=4, value=0),
                "acc": BitVecConst(width=8, value=0),
            },
            next_state={
                "round": Add(a=Var(name="round"), b=BitVecConst(width=4, value=1)),
                "acc": Mux(
                    cond=Eq(a=Var(name="round"), b=BitVecConst(width=4, value=0)),
                    a=Var(name="x"),
                    b=Var(name="acc"),
                ),
            },
            output_exprs={
                "y": Mux(
                    cond=Eq(a=Var(name="round"), b=BitVecConst(width=4, value=0)),
                    a=Var(name="x"),
                    b=Add(a=Var(name="acc"), b=BitVecConst(width=8, value=1)),
                )
            },
        )

        spec = specialize_tick_ir_steps(ir, steps=2)
        self.assertEqual(spec.schedule.const_state_order, ["round"])
        self.assertEqual(spec.schedule.runtime_state_order, ["acc"])

        self.assertEqual(_count_mux(spec.output_exprs[0]["y"]), 0)
        self.assertEqual(_count_mux(spec.output_exprs[1]["y"]), 0)
        self.assertEqual(_count_mux(spec.next_state_exprs[0]["acc"]), 0)
        self.assertEqual(_count_mux(spec.next_state_exprs[1]["acc"]), 0)

        inputs = {"x": 0x3C}
        baseline = TickState(state={"round": 0, "acc": 0})
        baseline_outs: list[int] = []
        for _ in range(2):
            baseline, out = tick(ir, baseline, inputs)
            baseline_outs.append(int(out["y"]))

        steps: list[TickIR] = []
        for step_index in range(2):
            steps.append(
                TickIR(
                    name="toy_step",
                    inputs=ir.inputs,
                    outputs=ir.outputs,
                    state=ir.state,
                    reset_state=ir.reset_state,
                    next_state={
                        "round": ir.next_state["round"],
                        "acc": spec.next_state_exprs[step_index]["acc"],
                    },
                    output_exprs=spec.output_exprs[step_index],
                )
            )

        runtime_acc = 0
        spec_outs: list[int] = []
        for step_index in range(2):
            round_val = int(spec.schedule.const_state_steps[step_index]["round"])
            y = int(
                tick(
                    steps[step_index],
                    TickState(state={"round": round_val, "acc": runtime_acc}),
                    inputs,
                )[1]["y"]
            )
            spec_outs.append(y)
            runtime_acc = int(
                tick(
                    steps[step_index],
                    TickState(state={"round": round_val, "acc": runtime_acc}),
                    inputs,
                )[0].state["acc"]
            )

        self.assertEqual(spec_outs, baseline_outs)
        self.assertEqual(int(baseline.state["acc"]), runtime_acc)
