"""
Tests for liveness analysis and register allocation.

These tests verify correctness of:
- stc.sched.liveness: LiveRange, compute_live_ranges, live_at_cycle, max_live, interference_graph
- stc.sched.regalloc: LinearScanAllocator, RegAllocation, allocate_registers
"""

import unittest
from stc.sched import list_schedule, AVX2, SSE2
from stc.sched.liveness import (
    LiveRange,
    compute_live_ranges,
    live_at_cycle,
    max_live,
    interference_graph,
)
from stc.sched.regalloc import (
    LinearScanAllocator,
    RegAllocation,
    allocate_registers,
)


def make_chain(length: int) -> tuple[list, int, list]:
    """Create a chain of XOR gates."""
    gates = [("xor", 0, 1)]
    for i in range(1, length):
        gates.append(("xor", 8 + i - 1, (i + 2) % 8))
    return gates, 8, [(8 + length - 1, False)]


def make_fan_out(width: int) -> tuple[list, int, list]:
    """Create fan-out: one gate feeds many."""
    gates = [("xor", 0, 1)]
    for i in range(width):
        gates.append(("and", 8, i + 2))
    outputs = [(8 + i + 1, False) for i in range(width)]
    return gates, 8, outputs


def make_bp_circuit() -> tuple[list, int, list]:
    """Create a realistic BP-like circuit with 128 gates."""
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

    for i in range(11, 128):
        left = 8 + (i % 11)
        right = i % 8
        if i % 3 == 0:
            gates.append(("and", left, right))
        elif i % 3 == 1:
            gates.append(("xor", left, right))
        else:
            gates.append(("or", left, right))

    outputs = [(8 + 127, False)]
    return gates, 8, outputs


class TestLiveRange(unittest.TestCase):
    def test_live_range_creation(self):
        lr = LiveRange(node=10, start=0, end=5)
        self.assertEqual(lr.node, 10)
        self.assertEqual(lr.start, 0)
        self.assertEqual(lr.end, 5)

    def test_live_range_length(self):
        lr = LiveRange(node=10, start=2, end=7)
        self.assertEqual(lr.length, 6)

    def test_live_range_contains_cycle(self):
        lr = LiveRange(node=10, start=2, end=7)
        self.assertFalse(lr.contains(1))
        self.assertTrue(lr.contains(2))
        self.assertTrue(lr.contains(5))
        self.assertTrue(lr.contains(7))
        self.assertFalse(lr.contains(8))

    def test_live_range_overlaps(self):
        lr1 = LiveRange(node=10, start=0, end=5)
        lr2 = LiveRange(node=11, start=3, end=8)
        lr3 = LiveRange(node=12, start=6, end=10)
        self.assertTrue(lr1.overlaps(lr2))
        self.assertTrue(lr2.overlaps(lr1))
        self.assertFalse(lr1.overlaps(lr3))
        self.assertTrue(lr2.overlaps(lr3))


class TestComputeLiveRanges(unittest.TestCase):
    def test_single_gate_live_range(self):
        gates = [("xor", 0, 1)]
        input_bits = 8
        outputs = [(8, False)]
        schedule = list_schedule(gates, input_bits, outputs, AVX2)

        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

        gate_range = ranges[8]
        self.assertEqual(gate_range.start, schedule.gate_cycle[0])
        self.assertEqual(gate_range.end, schedule.gate_cycle[0])

    def test_chain_live_ranges(self):
        gates, input_bits, outputs = make_chain(3)
        schedule = list_schedule(gates, input_bits, outputs, AVX2)

        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

        lr0 = ranges[8]
        lr1 = ranges[9]
        lr2 = ranges[10]

        self.assertEqual(lr0.start, schedule.gate_cycle[0])
        self.assertEqual(lr0.end, schedule.gate_cycle[1])

        self.assertEqual(lr1.start, schedule.gate_cycle[1])
        self.assertEqual(lr1.end, schedule.gate_cycle[2])

        self.assertEqual(lr2.start, schedule.gate_cycle[2])
        self.assertEqual(lr2.end, schedule.gate_cycle[2])

    def test_parallel_gates_live_ranges(self):
        gates = [
            ("xor", 0, 1),
            ("xor", 2, 3),
            ("xor", 4, 5),
        ]
        input_bits = 8
        outputs = [(8, False), (9, False), (10, False)]
        schedule = list_schedule(gates, input_bits, outputs, AVX2)

        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

        self.assertEqual(ranges[8].start, schedule.gate_cycle[0])
        self.assertEqual(ranges[9].start, schedule.gate_cycle[1])
        self.assertEqual(ranges[10].start, schedule.gate_cycle[2])

    def test_input_live_ranges(self):
        gates = [("xor", 0, 1)]
        input_bits = 8
        outputs = [(8, False)]
        schedule = list_schedule(gates, input_bits, outputs, AVX2)

        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

        self.assertIn(0, ranges)
        self.assertIn(1, ranges)
        self.assertEqual(ranges[0].start, 0)
        self.assertEqual(ranges[1].start, 0)


class TestLiveAtCycle(unittest.TestCase):
    def test_live_at_cycle(self):
        gates, input_bits, outputs = make_chain(3)
        schedule = list_schedule(gates, input_bits, outputs, AVX2)
        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

        live_c0 = live_at_cycle(ranges, 0)
        live_c1 = live_at_cycle(ranges, 1)
        live_c2 = live_at_cycle(ranges, 2)

        self.assertIn(8, live_c0)
        self.assertIn(8, live_c1)
        self.assertNotIn(8, live_c2)

        self.assertNotIn(9, live_c0)
        self.assertIn(9, live_c1)
        self.assertIn(9, live_c2)

    def test_live_at_cycle_empty(self):
        ranges = {}
        live = live_at_cycle(ranges, 0)
        self.assertEqual(live, set())

    def test_live_at_cycle_includes_inputs(self):
        gates = [("xor", 0, 1)]
        input_bits = 8
        outputs = [(8, False)]
        schedule = list_schedule(gates, input_bits, outputs, AVX2)
        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

        live_c0 = live_at_cycle(ranges, 0)
        self.assertIn(0, live_c0)
        self.assertIn(1, live_c0)


class TestMaxLive(unittest.TestCase):
    def test_max_live_chain(self):
        gates, input_bits, outputs = make_chain(10)
        schedule = list_schedule(gates, input_bits, outputs, AVX2)
        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

        ml = max_live(ranges, schedule.total_cycles)

        self.assertLessEqual(ml, input_bits + 2)

    def test_max_live_fan_out(self):
        gates, input_bits, outputs = make_fan_out(8)
        schedule = list_schedule(gates, input_bits, outputs, AVX2)
        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

        ml = max_live(ranges, schedule.total_cycles)

        self.assertGreater(ml, input_bits)

    def test_max_live_empty(self):
        ranges = {}
        ml = max_live(ranges, 0)
        self.assertEqual(ml, 0)

    def test_max_live_single_gate(self):
        gates = [("xor", 0, 1)]
        input_bits = 8
        outputs = [(8, False)]
        schedule = list_schedule(gates, input_bits, outputs, AVX2)
        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

        ml = max_live(ranges, schedule.total_cycles)
        self.assertGreaterEqual(ml, 3)


class TestInterferenceGraph(unittest.TestCase):
    def test_interference_graph(self):
        gates, input_bits, outputs = make_chain(3)
        schedule = list_schedule(gates, input_bits, outputs, AVX2)
        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

        ig = interference_graph(ranges)

        self.assertIn(9, ig.get(8, set()))
        self.assertIn(8, ig.get(9, set()))

    def test_interference_graph_no_interference(self):
        lr1 = LiveRange(node=10, start=0, end=2)
        lr2 = LiveRange(node=11, start=5, end=7)
        ranges = {10: lr1, 11: lr2}

        ig = interference_graph(ranges)

        self.assertNotIn(11, ig.get(10, set()))
        self.assertNotIn(10, ig.get(11, set()))

    def test_interference_graph_mutual(self):
        gates, input_bits, outputs = make_fan_out(4)
        schedule = list_schedule(gates, input_bits, outputs, AVX2)
        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

        ig = interference_graph(ranges)

        for node_a, neighbors in ig.items():
            for node_b in neighbors:
                self.assertIn(node_a, ig.get(node_b, set()))

    def test_interference_graph_empty(self):
        ig = interference_graph({})
        self.assertEqual(ig, {})


class TestLinearScanAllocator(unittest.TestCase):
    def test_allocator_creation(self):
        allocator = LinearScanAllocator(num_registers=16)
        self.assertEqual(allocator.num_registers, 16)

    def test_allocate_single_gate(self):
        gates = [("xor", 0, 1)]
        input_bits = 8
        outputs = [(8, False)]
        schedule = list_schedule(gates, input_bits, outputs, AVX2)
        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

        allocator = LinearScanAllocator(num_registers=16)
        allocation = allocator.allocate(
            ranges, schedule, gates=gates, input_bits=input_bits, outputs=outputs
        )

        self.assertIsNotNone(allocation.reg_assignment.get(0))
        self.assertIsNotNone(allocation.reg_assignment.get(1))
        self.assertIsNotNone(allocation.reg_assignment.get(8))

        self.assertLessEqual(max(allocation.reg_assignment.values()) + 1, 16)

    def test_allocate_chain_reuses_registers(self):
        gates, input_bits, outputs = make_chain(20)
        schedule = list_schedule(gates, input_bits, outputs, AVX2)
        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

        allocator = LinearScanAllocator(num_registers=16)
        allocation = allocator.allocate(
            ranges, schedule, gates=gates, input_bits=input_bits, outputs=outputs
        )

        regs_used = set(allocation.reg_assignment.values())
        self.assertLessEqual(len(regs_used), 16)
        self.assertEqual(len(allocation.spills), 0)


class TestRegAllocation(unittest.TestCase):
    def test_reg_allocation_creation(self):
        alloc = RegAllocation(
            reg_assignment={0: 0, 1: 1, 8: 2},
            spills=[],
            loads=[],
            stores=[],
        )
        self.assertEqual(alloc.reg_assignment[0], 0)
        self.assertEqual(alloc.reg_assignment[8], 2)

    def test_reg_allocation_with_spills(self):
        alloc = RegAllocation(
            reg_assignment={0: 0, 1: 1},
            spills=[8],
            loads=[(8, 2, 5)],
            stores=[(8, 2, 3)],
        )
        self.assertEqual(len(alloc.spills), 1)
        self.assertEqual(len(alloc.loads), 1)
        self.assertEqual(len(alloc.stores), 1)

    def test_reg_allocation_num_spills(self):
        alloc = RegAllocation(
            reg_assignment={},
            spills=[8, 9, 10],
            loads=[],
            stores=[],
        )
        self.assertEqual(alloc.num_spills, 3)


class TestAllocateRegisters(unittest.TestCase):
    def test_allocate_enough_registers_no_spills(self):
        gates, input_bits, outputs = make_chain(5)
        schedule = list_schedule(gates, input_bits, outputs, AVX2)
        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

        ml = max_live(ranges, schedule.total_cycles)

        allocation = allocate_registers(
            ranges,
            schedule,
            ml + 5,
            gates=gates,
            input_bits=input_bits,
            outputs=outputs,
        )

        self.assertEqual(len(allocation.spills), 0)

    def test_allocate_insufficient_registers_spills(self):
        gates, input_bits, outputs = make_fan_out(10)
        schedule = list_schedule(gates, input_bits, outputs, AVX2)
        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

        ml = max_live(ranges, schedule.total_cycles)

        if ml > 4:
            allocation = allocate_registers(
                ranges, schedule, 4, gates=gates, input_bits=input_bits, outputs=outputs
            )
            self.assertGreater(len(allocation.spills), 0)

    def test_spill_store_before_overwrite(self):
        gates, input_bits, outputs = make_fan_out(10)
        schedule = list_schedule(gates, input_bits, outputs, AVX2)
        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

        ml = max_live(ranges, schedule.total_cycles)

        if ml > 4:
            allocation = allocate_registers(
                ranges, schedule, 4, gates=gates, input_bits=input_bits, outputs=outputs
            )

            for node, reg, store_cycle in allocation.stores:
                node_range = ranges.get(node)
                if node_range:
                    self.assertLessEqual(store_cycle, node_range.end)

    def test_spill_load_before_use(self):
        gates, input_bits, outputs = make_fan_out(10)
        schedule = list_schedule(gates, input_bits, outputs, AVX2)
        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

        ml = max_live(ranges, schedule.total_cycles)

        if ml > 4:
            allocation = allocate_registers(
                ranges, schedule, 4, gates=gates, input_bits=input_bits, outputs=outputs
            )

            for node, reg, load_cycle in allocation.loads:
                node_range = ranges.get(node)
                if node_range:
                    self.assertGreaterEqual(load_cycle, node_range.start)


class TestBPCircuitAllocation(unittest.TestCase):
    def test_bp_circuit_avx2(self):
        gates, input_bits, outputs = make_bp_circuit()
        schedule = list_schedule(gates, input_bits, outputs, AVX2)
        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

        allocation = allocate_registers(
            ranges,
            schedule,
            AVX2.registers,
            gates=gates,
            input_bits=input_bits,
            outputs=outputs,
        )

        for node, reg in allocation.reg_assignment.items():
            self.assertGreaterEqual(reg, 0)
            self.assertLess(reg, AVX2.registers)

    def test_bp_circuit_sse2(self):
        gates, input_bits, outputs = make_bp_circuit()
        schedule = list_schedule(gates, input_bits, outputs, SSE2)
        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

        allocation = allocate_registers(
            ranges,
            schedule,
            SSE2.registers,
            gates=gates,
            input_bits=input_bits,
            outputs=outputs,
        )

        for node, reg in allocation.reg_assignment.items():
            self.assertGreaterEqual(reg, 0)
            self.assertLess(reg, SSE2.registers)

    def test_bp_circuit_allocation_valid(self):
        gates, input_bits, outputs = make_bp_circuit()
        schedule = list_schedule(gates, input_bits, outputs, AVX2)
        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

        allocation = allocate_registers(
            ranges,
            schedule,
            AVX2.registers,
            gates=gates,
            input_bits=input_bits,
            outputs=outputs,
        )

        ig = interference_graph(ranges)
        for node_a, neighbors in ig.items():
            reg_a = allocation.reg_assignment.get(node_a)
            if reg_a is None:
                continue
            for node_b in neighbors:
                reg_b = allocation.reg_assignment.get(node_b)
                if reg_b is None:
                    continue
                if node_a not in allocation.spills and node_b not in allocation.spills:
                    self.assertNotEqual(
                        reg_a,
                        reg_b,
                        f"Nodes {node_a} and {node_b} interfere but share register {reg_a}",
                    )


class TestAllocationIntegration(unittest.TestCase):
    def test_allocation_covers_all_nodes(self):
        gates, input_bits, outputs = make_chain(10)
        schedule = list_schedule(gates, input_bits, outputs, AVX2)
        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

        allocation = allocate_registers(
            ranges, schedule, 16, gates=gates, input_bits=input_bits, outputs=outputs
        )

        for node in ranges:
            if node < input_bits:
                # Inputs are treated as fixed temporaries by emitters and are
                # excluded from register allocation.
                continue
            in_assignment = node in allocation.reg_assignment
            in_spills = node in allocation.spills
            self.assertTrue(
                in_assignment or in_spills,
                f"Node {node} not in assignment or spills",
            )

    def test_allocation_respects_interference(self):
        gates = [
            ("xor", 0, 1),
            ("xor", 2, 3),
            ("and", 8, 9),
        ]
        input_bits = 8
        outputs = [(10, False)]
        schedule = list_schedule(gates, input_bits, outputs, AVX2)
        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)

        allocation = allocate_registers(
            ranges, schedule, 16, gates=gates, input_bits=input_bits, outputs=outputs
        )

        ig = interference_graph(ranges)
        for node_a, neighbors in ig.items():
            for node_b in neighbors:
                reg_a = allocation.reg_assignment.get(node_a)
                reg_b = allocation.reg_assignment.get(node_b)
                if (
                    reg_a is not None
                    and reg_b is not None
                    and node_a not in allocation.spills
                    and node_b not in allocation.spills
                ):
                    self.assertNotEqual(reg_a, reg_b)


if __name__ == "__main__":
    unittest.main()
