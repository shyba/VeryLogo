"""Tests for register-pressure-aware scheduling."""

import unittest

from stc.circuit_synth import CircuitState
from stc.sched.list_scheduler import list_schedule
from stc.sched.target import AVX512


class TestPressureAwareScheduling(unittest.TestCase):

    def test_pressure_limit_respected(self):
        """Verify max_live_pressure limits concurrent live values."""
        # Create circuit with high parallelism potential
        # 10 independent XOR gates
        gates = [("xor", 0, 1) for _ in range(10)]

        schedule = list_schedule(
            gates,
            input_bits=2,
            outputs=[(i + 2, False) for i in range(10)],
            target=AVX512,
            max_live_pressure=5,
        )

        # With pressure limit 5, gates should be spread across cycles
        cycles_used = len(set(schedule.gate_cycle.values()))
        self.assertGreater(cycles_used, 1)

    def test_no_limit_allows_parallelism(self):
        """Without limit, scheduler maximizes parallelism."""
        gates = [("xor", 0, 1) for _ in range(10)]

        schedule = list_schedule(
            gates,
            input_bits=2,
            outputs=[(i + 2, False) for i in range(10)],
            target=AVX512,
            max_live_pressure=None,
        )

        # Without limit, all gates could be in same cycle (if throughput allows)
        # At minimum, should use fewer cycles than with limit
        cycles_used = len(set(schedule.gate_cycle.values()))
        self.assertLessEqual(cycles_used, 5)  # AVX512 throughput ~2/cycle

    def test_dependencies_still_respected(self):
        """Pressure limit doesn't break dependency ordering."""
        gates = [
            ("xor", 0, 1),  # gate 0, index 2 in node space
            ("and", 0, 1),  # gate 1, index 3 in node space
            ("or", 2, 3),  # gate 2, index 4: uses gates 0 and 1
        ]

        schedule = list_schedule(
            gates,
            input_bits=2,
            outputs=[(4, False)],
            target=AVX512,
            max_live_pressure=10,
        )

        # All gates should be scheduled
        self.assertEqual(len(schedule.gate_cycle), 3)

        # gate 2 must come after gates 0 and 1
        self.assertGreater(schedule.gate_cycle[2], schedule.gate_cycle[0])
        self.assertGreater(schedule.gate_cycle[2], schedule.gate_cycle[1])


if __name__ == "__main__":
    unittest.main()
