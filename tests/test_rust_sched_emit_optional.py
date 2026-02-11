import os
import unittest
from pathlib import Path

from stc.backend_sched import _rust_generate_scheduled_code
from stc.circuit_synth import CircuitState


def _rust_bin_exists() -> bool:
    return any(
        Path(path).exists()
        for path in (
            "rust/sched_emit_rs/target/release/sched_emit_rs",
            "rust/sched_emit_rs/target/debug/sched_emit_rs",
        )
    )


def _make_tiny_circuit() -> CircuitState:
    # inputs: 0,1,2
    # gates: g0 = xor(0,1), g1 = and(1,2), g2 = ternary(0,1,2)
    gates = [
        ("xor", 0, 1),
        ("and", 1, 2),
        ("ternary", 0, 1, 2, 0xCA),
    ]
    outputs = [(3 + 2, True)]
    return CircuitState(
        input_bits=3,
        output_bits=1,
        gates=gates,
        outputs=outputs,
        gate_count=len(gates),
    )


@unittest.skipUnless(_rust_bin_exists(), "rust sched_emit_rs binary not built")
class TestRustSchedEmitOptional(unittest.TestCase):
    def setUp(self) -> None:
        self._old_env = os.environ.get("STC_RUST_SCHED_EMIT")
        os.environ["STC_RUST_SCHED_EMIT"] = "1"

    def tearDown(self) -> None:
        if self._old_env is None:
            os.environ.pop("STC_RUST_SCHED_EMIT", None)
        else:
            os.environ["STC_RUST_SCHED_EMIT"] = self._old_env
        os.environ.pop("STC_RUST_SCHED_FORMAT", None)

    def test_rust_emit_avx512_list(self) -> None:
        circuit = _make_tiny_circuit()
        os.environ["STC_RUST_SCHED_FORMAT"] = "bin"
        code = _rust_generate_scheduled_code(
            circuit=circuit,
            target="avx512",
            scheduler="list",
            function_name="circuit",
            io_split=None,
            max_live_pressure=None,
        )
        self.assertIsNotNone(code)
        assert code is not None
        self.assertIn("__m512i", code)
        self.assertIn("_mm512_ternarylogic_epi32", code)
        self.assertIn("_mm512_xor_si512", code)

    def test_rust_emit_avx2_pipelined(self) -> None:
        circuit = _make_tiny_circuit()
        os.environ["STC_RUST_SCHED_FORMAT"] = "bin"
        code = _rust_generate_scheduled_code(
            circuit=circuit,
            target="avx2",
            scheduler="pipelined",
            function_name="circuit",
            io_split=None,
            max_live_pressure=None,
        )
        self.assertIsNotNone(code)
        assert code is not None
        self.assertIn("__m256i", code)
        self.assertIn("_mm256_ternarylogic_epi32", code)
        self.assertIn("_mm256_xor_si256", code)
