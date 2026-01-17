import unittest

from stc.dead_state import live_state_within_bound, remove_dead_state
from stc.tick_ir import BitVecConst, BitVecType, BoolConst, BoolType, TickIR, Var


class TestDeadState(unittest.TestCase):
    def test_remove_unused_reg(self) -> None:
        ir = TickIR(
            name="top",
            inputs={"i": BoolType()},
            outputs={"o": BoolType()},
            state={"used": BoolType(), "dead": BoolType()},
            reset_state={
                "used": BoolConst(value=False),
                "dead": BoolConst(value=False),
            },
            next_state={"used": Var(name="i"), "dead": BoolConst(value=True)},
            output_exprs={"o": Var(name="used")},
        )

        reduced = remove_dead_state(ir, bound=1)
        self.assertEqual(set(reduced.state.keys()), {"used"})

    def test_bound_affects_liveness(self) -> None:
        ir = TickIR(
            name="top",
            inputs={},
            outputs={"o": BoolType()},
            state={"s0": BitVecType(width=1), "s1": BitVecType(width=1)},
            reset_state={
                "s0": BitVecConst(width=1, value=1),
                "s1": BitVecConst(width=1, value=0),
            },
            next_state={
                "s0": Var(name="s0"),
                "s1": Var(name="s0"),
            },
            output_exprs={"o": Var(name="s1")},
        )

        live1 = live_state_within_bound(ir, 1)
        live2 = live_state_within_bound(ir, 2)
        self.assertEqual(live1, {"s1"})
        self.assertEqual(live2, {"s0", "s1"})
