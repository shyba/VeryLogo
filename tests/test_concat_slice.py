import unittest

from stc.interp import reset_state, tick
from stc.tick_ir import BitVecType, BoolType, Concat, Slice, TickIR, Var


class TestConcatSlice(unittest.TestCase):
    def test_slice_single_bit(self) -> None:
        ir = TickIR(
            name="top",
            inputs={"a": BitVecType(width=4)},
            outputs={"b1": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"b1": Slice(x=Var(name="a"), offset=1, width=1)},
        )

        state = reset_state(ir)
        _, out = tick(ir, state, {"a": 0b0010})
        self.assertEqual(out["b1"], True)

    def test_concat_roundtrip_bits(self) -> None:
        ir = TickIR(
            name="top",
            inputs={"a": BitVecType(width=4)},
            outputs={"y": BitVecType(width=4)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "y": Concat(
                    parts=[
                        Slice(x=Var(name="a"), offset=3, width=1),
                        Slice(x=Var(name="a"), offset=2, width=1),
                        Slice(x=Var(name="a"), offset=1, width=1),
                        Slice(x=Var(name="a"), offset=0, width=1),
                    ]
                )
            },
        )

        state = reset_state(ir)
        _, out = tick(ir, state, {"a": 0b1010})
        self.assertEqual(out["y"], 0b1010)
