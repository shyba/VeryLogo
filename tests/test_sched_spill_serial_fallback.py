import unittest
from unittest.mock import patch

from stc.backend_sched import generate_scheduled_code
from stc.circuit_synth import CircuitState
from stc.sched import SSE2
from stc.sched.liveness import compute_live_ranges
from stc.sched.list_scheduler import list_schedule
from stc.sched.regalloc import allocate_registers
from stc.sched.schedule import Schedule


class TestSpillSerialFallback(unittest.TestCase):
    def test_serial_fallback_avoids_parallel_spill_hazards(self):
        # Force high register pressure: many independent nodes that are then
        # consumed late, so spilling is required.
        input_bits = 8
        gates = []
        # Create 40 nodes that depend on different inputs.
        for i in range(40):
            a = i % input_bits
            b = (i + 1) % input_bits
            gates.append(("xor", a, b))
        # Keep all intermediate values live by making them outputs.
        outputs = [(input_bits + i, False) for i in range(len(gates))]

        circuit = CircuitState(
            input_bits=input_bits,
            output_bits=len(outputs),
            gates=gates,
            outputs=outputs,
            gate_count=len(gates),
        )

        schedule = list_schedule(gates, input_bits, outputs, SSE2)
        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)
        alloc = allocate_registers(
            ranges,
            schedule,
            SSE2.registers,
            gates=gates,
            input_bits=input_bits,
            outputs=outputs,
        )
        self.assertGreater(alloc.num_spills, 0)

        # generate_scheduled_code should succeed even with spills+parallelism.
        code = generate_scheduled_code(circuit, target="avx2", scheduler="list")
        self.assertIn("void circuit", code)
        # In spill cases we fall back to naive emission (no schedule-driven stack traffic).
        self.assertNotIn("stack0", code)

    def test_ptx_keeps_parallel_schedule_under_spills(self):
        input_bits = 2
        gates = [("xor", 0, 1), ("and", 0, 1)]
        outputs = [(input_bits + i, False) for i in range(len(gates))]
        circuit = CircuitState(
            input_bits=input_bits,
            output_bits=len(outputs),
            gates=gates,
            outputs=outputs,
            gate_count=len(gates),
        )
        schedule = Schedule(gate_cycle={1: 0, 0: 0})  # two gates in one cycle
        mocked_alloc = allocate_registers(
            compute_live_ranges(schedule, gates, input_bits, outputs),
            schedule,
            1,  # force spills
            gates=gates,
            input_bits=input_bits,
            outputs=outputs,
        )
        self.assertGreater(mocked_alloc.num_spills, 0)

        with patch.dict(
            "stc.backend_sched.SCHEDULERS",
            {"list": lambda *_args, **_kwargs: schedule},
            clear=False,
        ):
            with patch(
                "stc.backend_sched.allocate_registers", return_value=mocked_alloc
            ) as alloc_mock:
                code = generate_scheduled_code(circuit, target="ptx", scheduler="list")

        self.assertEqual(alloc_mock.call_count, 1)
        self.assertIn(".visible .entry", code)

    def test_avx2_still_uses_serial_fallback_under_spills(self):
        input_bits = 2
        gates = [("xor", 0, 1), ("and", 0, 1)]
        outputs = [(input_bits + i, False) for i in range(len(gates))]
        circuit = CircuitState(
            input_bits=input_bits,
            output_bits=len(outputs),
            gates=gates,
            outputs=outputs,
            gate_count=len(gates),
        )
        schedule = Schedule(gate_cycle={1: 0, 0: 0})  # two gates in one cycle
        mocked_alloc = allocate_registers(
            compute_live_ranges(schedule, gates, input_bits, outputs),
            schedule,
            1,  # force spills
            gates=gates,
            input_bits=input_bits,
            outputs=outputs,
        )
        self.assertGreater(mocked_alloc.num_spills, 0)

        with patch.dict(
            "stc.backend_sched.SCHEDULERS",
            {"list": lambda *_args, **_kwargs: schedule},
            clear=False,
        ):
            with patch(
                "stc.backend_sched.allocate_registers", return_value=mocked_alloc
            ) as alloc_mock:
                generate_scheduled_code(circuit, target="avx2", scheduler="list")

        self.assertEqual(alloc_mock.call_count, 2)
