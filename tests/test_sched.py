"""
Tests for circuit scheduling.

These tests verify correctness and provide benchmarks for comparing
different scheduling strategies.
"""

import unittest
from stc.sched import (
    TargetModel,
    SSE2,
    SSE2_X64,
    AVX2,
    AVX512,
    PTX,
    Schedule,
    ListScheduler,
    list_schedule,
    compute_asap,
    compute_alap,
    compute_dependencies,
)
from stc.sched.analysis import compute_depth, compute_slack, critical_path


class TestTargetModel(unittest.TestCase):
    def test_sse2_has_8_regs(self):
        self.assertEqual(SSE2.registers, 8)

    def test_avx2_has_16_regs(self):
        self.assertEqual(AVX2.registers, 16)

    def test_ptx_has_255_regs(self):
        self.assertEqual(PTX.registers, 255)

    def test_latency_defaults_to_1(self):
        self.assertEqual(SSE2.latency("unknown_op"), 1)
        self.assertEqual(SSE2.latency("xor"), 1)

    def test_throughput_defaults_to_issue_width(self):
        target = TargetModel("test", registers=16, issue_width=4)
        self.assertEqual(target.max_per_cycle("foo"), 4)


class TestAnalysis(unittest.TestCase):
    def test_empty_circuit(self):
        asap = compute_asap([], 8, {"xor": 1})
        self.assertEqual(asap, {})

    def test_single_gate_asap(self):
        gates = [("xor", 0, 1)]
        asap = compute_asap(gates, 8, {"xor": 1})
        self.assertEqual(asap[0], 0)

    def test_chain_asap(self):
        gates = [
            ("xor", 0, 1),
            ("xor", 8, 2),
            ("xor", 9, 3),
        ]
        asap = compute_asap(gates, 8, {"xor": 1})
        self.assertEqual(asap[0], 0)
        self.assertEqual(asap[1], 1)
        self.assertEqual(asap[2], 2)

    def test_parallel_gates_same_asap(self):
        gates = [
            ("xor", 0, 1),
            ("xor", 2, 3),
            ("xor", 4, 5),
        ]
        asap = compute_asap(gates, 8, {"xor": 1})
        self.assertEqual(asap[0], 0)
        self.assertEqual(asap[1], 0)
        self.assertEqual(asap[2], 0)

    def test_diamond_asap(self):
        gates = [
            ("xor", 0, 1),
            ("xor", 2, 3),
            ("and", 8, 9),
        ]
        asap = compute_asap(gates, 8, {"xor": 1, "and": 1})
        self.assertEqual(asap[0], 0)
        self.assertEqual(asap[1], 0)
        self.assertEqual(asap[2], 1)

    def test_compute_depth_chain(self):
        gates = [
            ("xor", 0, 1),
            ("xor", 8, 2),
            ("xor", 9, 3),
        ]
        self.assertEqual(compute_depth(gates, 8), 3)

    def test_compute_depth_parallel(self):
        gates = [
            ("xor", 0, 1),
            ("xor", 2, 3),
        ]
        self.assertEqual(compute_depth(gates, 8), 1)

    def test_dependencies_chain(self):
        gates = [
            ("xor", 0, 1),
            ("xor", 8, 2),
        ]
        deps = compute_dependencies(gates, 8, [])
        self.assertEqual(deps.predecessors[0], set())
        self.assertEqual(deps.predecessors[1], {0})
        self.assertEqual(deps.successors[0], {1})

    def test_dependencies_diamond(self):
        gates = [
            ("xor", 0, 1),
            ("xor", 2, 3),
            ("and", 8, 9),
        ]
        deps = compute_dependencies(gates, 8, [])
        self.assertEqual(deps.predecessors[2], {0, 1})
        self.assertEqual(deps.successors[0], {2})
        self.assertEqual(deps.successors[1], {2})


class TestSchedule(unittest.TestCase):
    def test_empty_schedule(self):
        s = Schedule()
        self.assertEqual(s.total_cycles, 0)

    def test_single_gate_schedule(self):
        s = Schedule(gate_cycle={0: 0})
        self.assertEqual(s.total_cycles, 1)
        self.assertEqual(s.gates_at_cycle(0), [0])

    def test_validate_correct(self):
        gates = [("xor", 0, 1), ("xor", 8, 2)]
        s = Schedule(gate_cycle={0: 0, 1: 1})
        errors = s.validate(gates, 8, {"xor": 1})
        self.assertEqual(errors, [])

    def test_validate_dependency_violation(self):
        gates = [("xor", 0, 1), ("xor", 8, 2)]
        s = Schedule(gate_cycle={0: 1, 1: 0})
        errors = s.validate(gates, 8, {"xor": 1})
        self.assertTrue(len(errors) > 0)


class TestListScheduler(unittest.TestCase):
    def test_empty_circuit(self):
        s = list_schedule([], 8, [], AVX2)
        self.assertEqual(s.total_cycles, 0)

    def test_single_gate(self):
        gates = [("xor", 0, 1)]
        s = list_schedule(gates, 8, [(8, False)], AVX2)
        self.assertEqual(s.total_cycles, 1)
        self.assertEqual(s.gate_cycle[0], 0)

    def test_chain_respects_dependencies(self):
        gates = [
            ("xor", 0, 1),
            ("xor", 8, 2),
            ("xor", 9, 3),
        ]
        s = list_schedule(gates, 8, [(10, False)], AVX2)
        self.assertLess(s.gate_cycle[0], s.gate_cycle[1])
        self.assertLess(s.gate_cycle[1], s.gate_cycle[2])

    def test_parallel_gates_same_cycle(self):
        gates = [
            ("xor", 0, 1),
            ("xor", 2, 3),
            ("xor", 4, 5),
        ]
        s = list_schedule(gates, 8, [], AVX2)
        self.assertEqual(s.gate_cycle[0], 0)
        self.assertEqual(s.gate_cycle[1], 0)
        self.assertEqual(s.gate_cycle[2], 0)

    def test_throughput_limit(self):
        target = TargetModel(
            "limited",
            registers=16,
            issue_width=4,
            latencies={"xor": 1},
            throughput={"xor": 1},
        )
        gates = [
            ("xor", 0, 1),
            ("xor", 2, 3),
            ("xor", 4, 5),
        ]
        s = list_schedule(gates, 8, [], target)
        cycles = [s.gate_cycle[g] for g in range(3)]
        self.assertEqual(len(set(cycles)), 3)

    def test_schedule_validates(self):
        gates = [
            ("xor", 0, 1),
            ("and", 8, 2),
            ("xor", 9, 3),
        ]
        s = list_schedule(gates, 8, [(10, False)], AVX2)
        errors = s.validate(gates, 8, AVX2.latencies)
        self.assertEqual(errors, [], f"Validation errors: {errors}")

    def test_priority_slack(self):
        scheduler = ListScheduler(priority_fn="slack")
        self.assertEqual(scheduler.name, "list_slack")

    def test_priority_depth(self):
        scheduler = ListScheduler(priority_fn="depth")
        self.assertEqual(scheduler.name, "list_depth")


class TestSchedulerComparison(unittest.TestCase):
    """Compare scheduler performance across targets."""

    def _make_xor_chain(self, length: int) -> tuple[list, int, list]:
        gates = []
        for i in range(length):
            if i == 0:
                gates.append(("xor", 0, 1))
            else:
                gates.append(("xor", 8 + i - 1, (i + 2) % 8))
        output = (8 + length - 1, False)
        return gates, 8, [output]

    def _make_xor_tree(self, depth: int) -> tuple[list, int, list]:
        gates = []
        width = 2**depth
        input_bits = width

        level_start = input_bits
        level_width = width // 2

        for d in range(depth):
            for i in range(level_width):
                if d == 0:
                    left = 2 * i
                    right = 2 * i + 1
                else:
                    prev_start = level_start - (level_width * 2)
                    left = prev_start + 2 * i
                    right = prev_start + 2 * i + 1
                gates.append(("xor", left, right))
            level_start += level_width
            level_width //= 2

        output_idx = input_bits + len(gates) - 1
        return gates, input_bits, [(output_idx, False)]

    def test_chain_depth_equals_length(self):
        gates, input_bits, outputs = self._make_xor_chain(10)
        self.assertEqual(compute_depth(gates, input_bits), 10)

    def test_tree_depth_equals_log(self):
        gates, input_bits, outputs = self._make_xor_tree(4)
        self.assertEqual(compute_depth(gates, input_bits), 4)

    def test_chain_on_sse2(self):
        gates, input_bits, outputs = self._make_xor_chain(10)
        s = list_schedule(gates, input_bits, outputs, SSE2)
        self.assertEqual(s.total_cycles, 10)

    def test_tree_on_avx2(self):
        gates, input_bits, outputs = self._make_xor_tree(4)
        s = list_schedule(gates, input_bits, outputs, AVX2)
        depth = compute_depth(gates, input_bits)
        self.assertGreaterEqual(s.total_cycles, depth)
        errors = s.validate(gates, input_bits, AVX2.latencies)
        self.assertEqual(errors, [])

    def test_parallel_saturates_throughput(self):
        gates = [("xor", i % 8, (i + 1) % 8) for i in range(12)]
        s = list_schedule(gates, 8, [], AVX2)
        stats = s.compute_stats(gates, 8)
        self.assertLessEqual(s.total_cycles, 4)


class TestBPCircuitScheduling(unittest.TestCase):
    """Test scheduling with realistic BP-like circuits."""

    def _make_bp_like_circuit(self) -> tuple[list, int, list]:
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

    def test_bp_like_schedules(self):
        gates, input_bits, outputs = self._make_bp_like_circuit()
        s = list_schedule(gates, input_bits, outputs, AVX2)
        errors = s.validate(gates, input_bits, AVX2.latencies)
        self.assertEqual(errors, [], f"Validation errors: {errors}")

    def test_bp_like_depth(self):
        gates, input_bits, outputs = self._make_bp_like_circuit()
        depth = compute_depth(gates, input_bits)
        self.assertLessEqual(depth, 5)

    def test_bp_like_on_different_targets(self):
        gates, input_bits, outputs = self._make_bp_like_circuit()

        for target in [SSE2, AVX2, PTX]:
            s = list_schedule(gates, input_bits, outputs, target)
            errors = s.validate(gates, input_bits, target.latencies)
            self.assertEqual(errors, [], f"{target.name} validation errors: {errors}")


class TestPipelinedScheduler(unittest.TestCase):
    """Test pipelined scheduler for high-latency targets."""

    def test_empty_circuit(self):
        from stc.sched import pipelined_schedule, PTX

        s = pipelined_schedule([], 8, [], PTX)
        self.assertEqual(s.total_cycles, 0)

    def test_single_gate(self):
        from stc.sched import pipelined_schedule, PTX

        gates = [("xor", 0, 1)]
        s = pipelined_schedule(gates, 8, [(8, False)], PTX)
        self.assertEqual(s.total_cycles, 1)

    def test_chain_with_latency(self):
        from stc.sched import pipelined_schedule, TargetModel

        target = TargetModel(
            "test_latency",
            registers=32,
            issue_width=32,
            latencies={"xor": 4},
            throughput={"xor": 32},
        )

        gates = [
            ("xor", 0, 1),
            ("xor", 8, 2),
            ("xor", 9, 3),
        ]
        s = pipelined_schedule(gates, 8, [(10, False)], target)
        self.assertEqual(s.gate_cycle[0], 0)
        self.assertEqual(s.gate_cycle[1], 4)
        self.assertEqual(s.gate_cycle[2], 8)
        self.assertEqual(s.total_cycles, 9)

    def test_parallel_gates_same_cycle(self):
        from stc.sched import pipelined_schedule, PTX

        gates = [
            ("xor", 0, 1),
            ("xor", 2, 3),
            ("xor", 4, 5),
        ]
        s = pipelined_schedule(gates, 8, [], PTX)
        self.assertEqual(s.gate_cycle[0], 0)
        self.assertEqual(s.gate_cycle[1], 0)
        self.assertEqual(s.gate_cycle[2], 0)

    def test_pipelined_achieves_theoretical_minimum(self):
        from stc.sched import pipelined_schedule, TargetModel, compute_depth

        target = TargetModel(
            "high_latency",
            registers=255,
            issue_width=32,
            latencies={"xor": 4, "and": 4},
            throughput={"xor": 32, "and": 32},
        )

        gates = [
            ("xor", 0, 1),
            ("and", 8, 2),
            ("xor", 9, 3),
            ("and", 10, 4),
        ]
        s = pipelined_schedule(gates, 8, [(11, False)], target)

        depth = compute_depth(gates, 8)
        theoretical_min = (depth - 1) * 4 + 1

        self.assertEqual(s.total_cycles, theoretical_min)


class TestTernaryGateScheduling(unittest.TestCase):
    """Tests for ternary gate (5-tuple) scheduling."""

    def test_ternary_dependencies(self):
        gates = [
            ("xor", 0, 1),
            ("xor", 2, 3),
            ("xor", 4, 5),
            ("ternary", 8, 9, 10, 0xCA),
        ]
        deps = compute_dependencies(gates, 8, [])
        self.assertEqual(deps.predecessors[3], {0, 1, 2})
        self.assertEqual(deps.successors[0], {3})
        self.assertEqual(deps.successors[1], {3})
        self.assertEqual(deps.successors[2], {3})

    def test_ternary_asap(self):
        gates = [
            ("xor", 0, 1),
            ("xor", 2, 3),
            ("xor", 4, 5),
            ("ternary", 8, 9, 10, 0xCA),
        ]
        asap = compute_asap(gates, 8, {"xor": 1, "ternary": 1})
        self.assertEqual(asap[0], 0)
        self.assertEqual(asap[1], 0)
        self.assertEqual(asap[2], 0)
        self.assertEqual(asap[3], 1)

    def test_ternary_depth(self):
        gates = [
            ("xor", 0, 1),
            ("ternary", 8, 2, 3, 0xCA),
            ("ternary", 9, 4, 5, 0xE8),
        ]
        self.assertEqual(compute_depth(gates, 8), 3)

    def test_ternary_schedule_validates(self):
        gates = [
            ("xor", 0, 1),
            ("xor", 2, 3),
            ("ternary", 8, 9, 4, 0xCA),
        ]
        s = list_schedule(gates, 8, [(10, False)], AVX2)
        errors = s.validate(gates, 8, AVX2.latencies)
        self.assertEqual(errors, [], f"Validation errors: {errors}")

    def test_ternary_schedule_respects_all_operands(self):
        gates = [
            ("xor", 0, 1),
            ("xor", 2, 3),
            ("xor", 4, 5),
            ("ternary", 8, 9, 10, 0xCA),
        ]
        s = list_schedule(gates, 8, [(11, False)], AVX2)
        self.assertLess(s.gate_cycle[0], s.gate_cycle[3])
        self.assertLess(s.gate_cycle[1], s.gate_cycle[3])
        self.assertLess(s.gate_cycle[2], s.gate_cycle[3])
        errors = s.validate(gates, 8, AVX2.latencies)
        self.assertEqual(errors, [], f"Validation errors: {errors}")

    def test_ternary_chain(self):
        gates = [
            ("ternary", 0, 1, 2, 0xCA),
            ("ternary", 8, 3, 4, 0xE8),
            ("ternary", 9, 5, 6, 0x96),
        ]
        asap = compute_asap(gates, 8, {"ternary": 1})
        self.assertEqual(asap[0], 0)
        self.assertEqual(asap[1], 1)
        self.assertEqual(asap[2], 2)


if __name__ == "__main__":
    unittest.main()
