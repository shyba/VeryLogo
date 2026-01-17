import json
import tempfile
import unittest
from pathlib import Path

from stc.io_map import IoMap, default_io_map, load_io_map, validate_io_map
from stc.tick_ir import BitVecType, BoolType, TickIR


class TestIoMap(unittest.TestCase):
    def test_default_io_map_packs_inputs_then_outputs(self) -> None:
        ir = TickIR(
            name="t",
            inputs={"a": BoolType(), "b": BitVecType(width=2)},
            outputs={"y": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={},
        )
        m = default_io_map(ir)
        self.assertEqual(m.inputs["a"], {"lsb": 0, "width": 1})
        self.assertEqual(m.inputs["b"], {"lsb": 1, "width": 2})
        self.assertEqual(m.outputs["y"], {"lsb": 3, "width": 1})

    def test_io_map_round_trip(self) -> None:
        m = IoMap(
            port="B",
            inputs={"a": {"lsb": 0, "width": 1}},
            outputs={"y": {"lsb": 1, "width": 1}},
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "io_map.json"
            p.write_text(json.dumps(m.to_dict()), encoding="utf-8")
            m2 = load_io_map(p)
        self.assertEqual(m2, m)

    def test_validate_rejects_overlap(self) -> None:
        ir = TickIR(
            name="t",
            inputs={"a": BoolType(), "b": BoolType()},
            outputs={},
            state={},
            reset_state={},
            next_state={},
            output_exprs={},
        )
        m = IoMap(
            port="B",
            inputs={"a": {"lsb": 0, "width": 1}, "b": {"lsb": 0, "width": 1}},
            outputs={},
        )
        with self.assertRaises(ValueError):
            validate_io_map(ir, m)
