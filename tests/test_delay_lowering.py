import unittest

from stc.delay_lower import lower_delays
from stc.interp import reset_state, tick
from stc.tick_ir import BitVecConst, BitVecType, Delay, TickIR, Var
from stc.tick_ir_validate import validate_tick_ir


class TestDelayLowering(unittest.TestCase):
    def test_delay_two_ticks_on_input(self) -> None:
        bv8 = BitVecType(width=8)
        ir = TickIR(
            name="t",
            inputs={"i": bv8},
            outputs={"o": bv8},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Delay(x=Var("i"), ticks=2)},
        )
        lowered = lower_delays(ir)
        validate_tick_ir(lowered)
        st = reset_state(lowered)

        inputs = [1, 2, 3, 4]
        outs = []
        for v in inputs:
            st, o = tick(lowered, st, {"i": v})
            outs.append(int(o["o"]))
        self.assertEqual(outs, [0, 0, 1, 2])
