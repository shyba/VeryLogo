import unittest

from stc.cases import all_cases, case_debounce_redundant_state
from stc.dead_state import remove_dead_state
from stc.interp import reset_state, tick
from stc.tick_ir import TickIR
from stc.tick_ir_validate import validate_tick_ir


class TestCases(unittest.TestCase):
    def test_all_cases_match_expected(self) -> None:
        for case in all_cases():
            validate_tick_ir(case.ir)
            st = reset_state(case.ir)
            outputs = []
            for ins in case.inputs:
                st, out = tick(case.ir, st, ins)
                outputs.append(out)
            self.assertEqual(outputs, case.expected_outputs)

    def test_debounce_case_dead_state_removed(self) -> None:
        case = case_debounce_redundant_state()
        reduced = remove_dead_state(case.ir, bound=2)
        self.assertEqual(set(reduced.state.keys()), {"o_reg"})

    def test_case_fixtures_round_trip(self) -> None:
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        fixture_dir = root / "fixtures" / "tick_ir"
        for path in sorted(fixture_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            ir = TickIR.from_dict(data)
            validate_tick_ir(ir)
