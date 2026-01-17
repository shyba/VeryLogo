import json
import unittest

from stc.tick_ir import BoolType, TickIR, Var


class TestTickIRJson(unittest.TestCase):
    def test_round_trip(self) -> None:
        ir = TickIR(
            name="top",
            inputs={"i": BoolType()},
            outputs={"o": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Var(name="i")},
        )

        encoded = json.dumps(ir.to_dict(), sort_keys=True)
        decoded = TickIR.from_dict(json.loads(encoded))
        self.assertEqual(ir, decoded)
