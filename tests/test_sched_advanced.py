"""
Tests for advanced scheduling: register-pressure-aware scheduler and backend integration.

These tests verify:
- RegPressureScheduler produces valid schedules with reduced register pressure
- backend_sched generates correct target code from scheduled circuits
"""

import unittest
from dataclasses import dataclass

from stc.sched import (
    TargetModel,
    SSE2,
    AVX2,
    AVX512,
    PTX,
    Schedule,
    ListScheduler,
    PipelinedScheduler,
    list_schedule,
    pipelined_schedule,
    compute_live_ranges,
    max_live,
)
from stc.sched.regalloc import LinearScanAllocator, allocate_registers


def make_fan_out_circuit(width: int):
    """One gate feeds many - high register pressure."""
    gates = [("xor", 0, 1)]
    for i in range(width):
        gates.append(("and", 8, i + 2))
    outputs = [(8 + i + 1, False) for i in range(width)]
    return gates, 8, outputs


def make_chain_circuit(length: int):
    """Low register pressure - values die quickly."""
    gates = [("xor", 0, 1)]
    for i in range(1, length):
        gates.append(("xor", 8 + i - 1, (i + 2) % 8))
    return gates, 8, [(8 + length - 1, False)]


def make_bp_like_circuit():
    """Boyar-Peralta style circuit with mixed dependencies."""
    gates = []
    gates.append(("xor", 7, 4))
    gates.append(("xor", 7, 2))
    gates.append(("xor", 7, 1))
    gates.append(("xor", 4, 2))
    gates.append(("xor", 3, 1))
    gates.append(("xor", 8, 12))
    gates.append(("xor", 6, 5))
    gates.append(("xor", 0, 13))
    gates.append(("and", 8, 13))
    gates.append(("and", 9, 15))
    gates.append(("xor", 16, 17))
    outputs = [(18, False)]
    return gates, 8, outputs


@dataclass
class MockCircuit:
    """Minimal circuit representation for testing backend_sched."""

    input_bits: int
    output_bits: int
    gates: list
    outputs: list
    gate_count: int


def make_mock_circuit(gates, input_bits, outputs):
    """Create a MockCircuit from gates, input_bits, outputs."""
    return MockCircuit(
        input_bits=input_bits,
        output_bits=len(outputs),
        gates=gates,
        outputs=outputs,
        gate_count=len(gates),
    )


try:
    from stc.sched.regpressure_scheduler import (
        RegPressureScheduler,
        regpressure_schedule,
    )

    HAS_REGPRESSURE = True
except ImportError:
    HAS_REGPRESSURE = False
    RegPressureScheduler = None
    regpressure_schedule = None


try:
    from stc.backend_sched import (
        generate_scheduled_code,
        get_schedule_stats,
    )

    HAS_BACKEND_SCHED = True
except ImportError:
    HAS_BACKEND_SCHED = False
    generate_scheduled_code = None
    get_schedule_stats = None


@unittest.skipUnless(HAS_REGPRESSURE, "regpressure_scheduler not implemented yet")
class TestRegPressureScheduler(unittest.TestCase):
    """Tests for register-pressure-aware scheduling."""

    def test_regpressure_reduces_spills(self):
        """RegPressureScheduler should produce fewer spills than list scheduler on high-pressure circuits."""
        gates, input_bits, outputs = make_fan_out_circuit(12)

        list_sched = list_schedule(gates, input_bits, outputs, SSE2)
        list_ranges = compute_live_ranges(list_sched, gates, input_bits, outputs)
        list_alloc = allocate_registers(
            list_ranges,
            list_sched,
            SSE2.registers,
            gates=gates,
            input_bits=input_bits,
            outputs=outputs,
        )

        rp_sched = regpressure_schedule(
            gates, input_bits, outputs, SSE2, max_registers=SSE2.registers
        )
        rp_ranges = compute_live_ranges(rp_sched, gates, input_bits, outputs)
        rp_alloc = allocate_registers(
            rp_ranges,
            rp_sched,
            SSE2.registers,
            gates=gates,
            input_bits=input_bits,
            outputs=outputs,
        )

        self.assertLessEqual(
            rp_alloc.num_spills,
            list_alloc.num_spills,
            f"RegPressure should have <= spills: {rp_alloc.num_spills} vs list {list_alloc.num_spills}",
        )

    def test_regpressure_respects_dependencies(self):
        """Schedule produced by RegPressureScheduler must be valid."""
        gates, input_bits, outputs = make_fan_out_circuit(8)

        sched = regpressure_schedule(
            gates, input_bits, outputs, AVX2, max_registers=AVX2.registers
        )

        errors = sched.validate(gates, input_bits, AVX2.latencies)
        self.assertEqual(errors, [], f"Validation errors: {errors}")

    def test_regpressure_with_tight_limit(self):
        """RegPressureScheduler should work with very few registers."""
        gates, input_bits, outputs = make_chain_circuit(10)

        target = TargetModel(
            name="tiny",
            registers=4,
            issue_width=2,
            latencies={"xor": 1, "and": 1},
            throughput={"xor": 2, "and": 2},
        )

        sched = regpressure_schedule(
            gates, input_bits, outputs, target, max_registers=4
        )

        errors = sched.validate(gates, input_bits, target.latencies)
        self.assertEqual(errors, [], f"Validation errors: {errors}")

        self.assertEqual(len(sched.gate_cycle), len(gates))

    def test_regpressure_matches_target(self):
        """When max_registers matches target, there should be no spills."""
        gates, input_bits, outputs = make_chain_circuit(8)

        sched = regpressure_schedule(
            gates, input_bits, outputs, AVX512, max_registers=AVX512.registers
        )
        ranges = compute_live_ranges(sched, gates, input_bits, outputs)
        alloc = allocate_registers(
            ranges, sched, AVX512.registers, gates=gates, input_bits=input_bits, outputs=outputs
        )

        self.assertEqual(
            alloc.num_spills,
            0,
            f"Chain circuit on AVX512 should have 0 spills, got {alloc.num_spills}",
        )

    def test_regpressure_on_bp_circuit(self):
        """Test on realistic BP-like circuit."""
        gates, input_bits, outputs = make_bp_like_circuit()

        list_sched = list_schedule(gates, input_bits, outputs, AVX2)
        list_ranges = compute_live_ranges(list_sched, gates, input_bits, outputs)
        list_max = max_live(list_ranges)

        rp_sched = regpressure_schedule(
            gates, input_bits, outputs, AVX2, max_registers=AVX2.registers
        )
        rp_ranges = compute_live_ranges(rp_sched, gates, input_bits, outputs)
        rp_max = max_live(rp_ranges)

        errors = rp_sched.validate(gates, input_bits, AVX2.latencies)
        self.assertEqual(errors, [], f"Validation errors: {errors}")

        self.assertLessEqual(
            rp_max,
            list_max + 2,
            f"RegPressure max_live should not be much worse: {rp_max} vs list {list_max}",
        )


@unittest.skipUnless(HAS_BACKEND_SCHED, "backend_sched not implemented yet")
class TestBackendSchedIntegration(unittest.TestCase):
    """Tests for backend code generation from scheduled circuits."""

    def test_generate_scheduled_code_avx2(self):
        """Generate valid AVX2 code from scheduled circuit."""
        gates, input_bits, outputs = make_bp_like_circuit()
        circuit = make_mock_circuit(gates, input_bits, outputs)

        code = generate_scheduled_code(
            circuit,
            target="avx2",
            scheduler="list",
        )

        self.assertIsInstance(code, str)
        self.assertTrue(len(code) > 0)

        self.assertIn("_mm256", code)

    def test_generate_scheduled_code_ptx(self):
        """Generate valid PTX code from scheduled circuit."""
        gates, input_bits, outputs = make_bp_like_circuit()
        circuit = make_mock_circuit(gates, input_bits, outputs)

        code = generate_scheduled_code(
            circuit,
            target="ptx",
            scheduler="pipelined",
        )

        self.assertIsInstance(code, str)
        self.assertTrue(len(code) > 0)

        self.assertTrue(
            "xor" in code.lower() or "lop3" in code.lower(),
            "PTX code should contain xor or lop3 instructions",
        )

    def test_get_schedule_stats(self):
        """Get statistics about a scheduled circuit."""
        gates, input_bits, outputs = make_fan_out_circuit(8)
        circuit = make_mock_circuit(gates, input_bits, outputs)

        stats = get_schedule_stats(circuit, target="avx2", scheduler="list")

        self.assertIn("total_cycles", stats)
        self.assertIn("max_live", stats)
        self.assertIn("num_spills", stats)

        self.assertIsInstance(stats["total_cycles"], int)
        self.assertIsInstance(stats["max_live"], int)
        self.assertIsInstance(stats["num_spills"], int)

        self.assertGreater(stats["total_cycles"], 0)

    def test_different_schedulers(self):
        """All available schedulers produce valid output."""
        gates, input_bits, outputs = make_chain_circuit(6)
        circuit = make_mock_circuit(gates, input_bits, outputs)

        schedulers = ["list", "pipelined"]

        for sched_name in schedulers:
            with self.subTest(scheduler=sched_name):
                code = generate_scheduled_code(
                    circuit,
                    target="avx2",
                    scheduler=sched_name,
                )

                self.assertIsInstance(code, str)
                self.assertTrue(
                    len(code) > 0, f"Scheduler {sched_name} produced empty code"
                )


class TestSchedulerProtocol(unittest.TestCase):
    """Test that schedulers follow the protocol even without new modules."""

    def test_list_scheduler_protocol(self):
        """ListScheduler follows Scheduler protocol."""
        scheduler = ListScheduler(priority_fn="slack")

        self.assertTrue(hasattr(scheduler, "name"))
        self.assertTrue(hasattr(scheduler, "schedule"))

        self.assertEqual(scheduler.name, "list_slack")

    def test_pipelined_scheduler_protocol(self):
        """PipelinedScheduler follows Scheduler protocol."""
        scheduler = PipelinedScheduler()

        self.assertTrue(hasattr(scheduler, "name"))
        self.assertTrue(hasattr(scheduler, "schedule"))

        self.assertEqual(scheduler.name, "pipelined")

    @unittest.skipUnless(HAS_REGPRESSURE, "regpressure_scheduler not implemented yet")
    def test_regpressure_scheduler_protocol(self):
        """RegPressureScheduler follows Scheduler protocol."""
        scheduler = RegPressureScheduler(max_registers=16)

        self.assertTrue(hasattr(scheduler, "name"))
        self.assertTrue(hasattr(scheduler, "schedule"))

        self.assertEqual(scheduler.name, "regpressure_16")

    def test_all_schedulers_produce_valid_schedules(self):
        """All built-in schedulers produce valid schedules."""
        gates, input_bits, outputs = make_bp_like_circuit()

        schedulers = [
            ListScheduler(priority_fn="slack"),
            ListScheduler(priority_fn="depth"),
            ListScheduler(priority_fn="fifo"),
            PipelinedScheduler(),
        ]

        if HAS_REGPRESSURE:
            schedulers.append(RegPressureScheduler(max_registers=16))

        for scheduler in schedulers:
            with self.subTest(scheduler=scheduler.name):
                sched = scheduler.schedule(gates, input_bits, outputs, AVX2)

                errors = sched.validate(gates, input_bits, AVX2.latencies)
                self.assertEqual(
                    errors, [], f"{scheduler.name} validation errors: {errors}"
                )

                self.assertEqual(
                    len(sched.gate_cycle),
                    len(gates),
                    f"{scheduler.name} didn't schedule all gates",
                )


class TestRegisterAllocationWithSchedule(unittest.TestCase):
    """Test register allocation with different schedule qualities."""

    def test_fan_out_needs_many_registers(self):
        """Fan-out circuit has high register pressure."""
        gates, input_bits, outputs = make_fan_out_circuit(10)

        sched = list_schedule(gates, input_bits, outputs, AVX2)
        ranges = compute_live_ranges(sched, gates, input_bits, outputs)
        live = max_live(ranges)

        self.assertGreater(live, 8, "Fan-out should need many simultaneous registers")

    def test_chain_needs_fewer_registers_than_fan_out(self):
        """Chain circuit has lower register pressure than fan-out."""
        chain_gates, chain_input_bits, chain_outputs = make_chain_circuit(10)
        fan_gates, fan_input_bits, fan_outputs = make_fan_out_circuit(10)

        chain_sched = list_schedule(chain_gates, chain_input_bits, chain_outputs, AVX2)
        chain_ranges = compute_live_ranges(
            chain_sched, chain_gates, chain_input_bits, chain_outputs
        )
        chain_live = max_live(chain_ranges)

        fan_sched = list_schedule(fan_gates, fan_input_bits, fan_outputs, AVX2)
        fan_ranges = compute_live_ranges(
            fan_sched, fan_gates, fan_input_bits, fan_outputs
        )
        fan_live = max_live(fan_ranges)

        self.assertLess(
            chain_live,
            fan_live,
            f"Chain ({chain_live}) should need fewer registers than fan-out ({fan_live})",
        )

    def test_allocation_with_sse2_limited_regs(self):
        """SSE2 (8 regs) needs spills on high-pressure circuit."""
        gates, input_bits, outputs = make_fan_out_circuit(12)

        sched = list_schedule(gates, input_bits, outputs, SSE2)
        ranges = compute_live_ranges(sched, gates, input_bits, outputs)
        alloc = allocate_registers(
            ranges, sched, SSE2.registers, gates=gates, input_bits=input_bits, outputs=outputs
        )

        self.assertGreater(
            alloc.num_spills,
            0,
            "Fan-out on SSE2 should require spills",
        )

    def test_allocation_with_avx512_plenty_regs(self):
        """AVX512 (32 regs) should handle most circuits without spills."""
        gates, input_bits, outputs = make_bp_like_circuit()

        sched = list_schedule(gates, input_bits, outputs, AVX512)
        ranges = compute_live_ranges(sched, gates, input_bits, outputs)
        alloc = allocate_registers(
            ranges, sched, AVX512.registers, gates=gates, input_bits=input_bits, outputs=outputs
        )

        self.assertEqual(
            alloc.num_spills,
            0,
            f"BP circuit on AVX512 should have 0 spills, got {alloc.num_spills}",
        )


class TestTernaryGateLivenessAndAllocation(unittest.TestCase):
    """Tests for ternary gate (5-tuple) liveness and register allocation."""

    def test_ternary_live_ranges(self):
        gates = [
            ("xor", 0, 1),
            ("xor", 2, 3),
            ("xor", 4, 5),
            ("ternary", 8, 9, 10, 0xCA),
        ]
        input_bits = 8
        outputs = [(11, False)]

        sched = list_schedule(gates, input_bits, outputs, AVX2)
        ranges = compute_live_ranges(sched, gates, input_bits, outputs)

        self.assertIn(8, ranges)
        self.assertIn(9, ranges)
        self.assertIn(10, ranges)
        self.assertIn(11, ranges)

    def test_ternary_all_operands_tracked_for_liveness(self):
        gates = [
            ("xor", 0, 1),
            ("xor", 2, 3),
            ("xor", 4, 5),
            ("ternary", 8, 9, 10, 0xCA),
        ]
        input_bits = 8
        outputs = [(11, False)]

        sched = list_schedule(gates, input_bits, outputs, AVX2)
        ranges = compute_live_ranges(sched, gates, input_bits, outputs)

        ternary_cycle = sched.gate_cycle[3]
        self.assertGreaterEqual(ranges[8].end, ternary_cycle)
        self.assertGreaterEqual(ranges[9].end, ternary_cycle)
        self.assertGreaterEqual(ranges[10].end, ternary_cycle)

    def test_ternary_register_allocation(self):
        gates = [
            ("xor", 0, 1),
            ("xor", 2, 3),
            ("xor", 4, 5),
            ("ternary", 8, 9, 10, 0xCA),
        ]
        input_bits = 8
        outputs = [(11, False)]

        sched = list_schedule(gates, input_bits, outputs, AVX512)
        ranges = compute_live_ranges(sched, gates, input_bits, outputs)
        alloc = allocate_registers(
            ranges, sched, AVX512.registers, gates=gates, input_bits=input_bits, outputs=outputs
        )

        self.assertEqual(
            alloc.num_spills,
            0,
            f"Simple ternary circuit on AVX512 should have 0 spills, got {alloc.num_spills}",
        )

    def test_ternary_chain_scheduling(self):
        gates = [
            ("ternary", 0, 1, 2, 0xCA),
            ("ternary", 8, 3, 4, 0xE8),
            ("ternary", 9, 5, 6, 0x96),
        ]
        input_bits = 8
        outputs = [(10, False)]

        sched = list_schedule(gates, input_bits, outputs, AVX2)
        errors = sched.validate(gates, input_bits, AVX2.latencies)
        self.assertEqual(errors, [], f"Validation errors: {errors}")

        self.assertLess(sched.gate_cycle[0], sched.gate_cycle[1])
        self.assertLess(sched.gate_cycle[1], sched.gate_cycle[2])


if __name__ == "__main__":
    unittest.main()
