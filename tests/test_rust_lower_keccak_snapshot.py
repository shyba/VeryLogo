import json
import random
import subprocess
import unittest
from pathlib import Path

from stc.tick_ir import TickIR
from stc.tick_ir_to_circuit_state import (
    lower_tick_ir_to_circuit_state,
    rust_lower_tick_ir_to_circuit_state,
)
from stc.testing.circuitstate_eval import eval_circuitstate_bits


class TestRustLowerKeccakSnapshot(unittest.TestCase):
    def test_rust_lower_matches_python_on_keccak_snapshot(self) -> None:
        fixture = Path("fixtures/tick_ir/keccak_round_snapshot.json")
        if not fixture.exists():
            self.skipTest("keccak snapshot fixture missing")
        rust_result = rust_lower_tick_ir_to_circuit_state(
            TickIR.from_dict(json.loads(fixture.read_text()))
        )
        if rust_result is None:
            self._build_rust_lower()
            rust_result = rust_lower_tick_ir_to_circuit_state(
                TickIR.from_dict(json.loads(fixture.read_text()))
            )
            self.assertIsNotNone(
                rust_result, "rust tick_lower_rs binary unavailable after build"
            )
        assert rust_result is not None

        tick_ir = TickIR.from_dict(json.loads(fixture.read_text()))
        py_circuit, py_layout = lower_tick_ir_to_circuit_state(tick_ir)
        rs_circuit, rs_layout = rust_result

        self.assertEqual(py_layout.to_dict(), rs_layout.to_dict())
        self.assertEqual(py_circuit.input_bits, rs_circuit.input_bits)
        self.assertEqual(py_circuit.output_bits, rs_circuit.output_bits)

        rng = random.Random(0)
        for _ in range(3):
            inputs = [rng.randint(0, 1) for _ in range(py_circuit.input_bits)]
            py_out = eval_circuitstate_bits(py_circuit.to_dict(), inputs)
            rs_out = eval_circuitstate_bits(rs_circuit.to_dict(), inputs)
            self.assertEqual(py_out, rs_out)

    def _build_rust_lower(self) -> None:
        manifest = Path("rust/tick_lower_rs/Cargo.toml")
        if not manifest.exists():
            self.fail("rust/tick_lower_rs missing; cannot build rust lower")
        result = subprocess.run(
            ["cargo", "build", "--manifest-path", str(manifest)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.fail(
                "failed to build rust/tick_lower_rs:\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}\n"
            )
