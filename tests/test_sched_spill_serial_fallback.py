import unittest

from stc.backend_sched import generate_scheduled_code
from stc.circuit_synth import CircuitState
from stc.sched import SSE2
from stc.sched.liveness import compute_live_ranges
from stc.sched.list_scheduler import list_schedule
from stc.sched.regalloc import allocate_registers


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
