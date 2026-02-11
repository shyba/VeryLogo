import unittest

from stc.circuit_synth import CircuitState
from stc.mir import Binary, Load, Store
from stc.mir.lower import circuit_to_mir
from stc.sched.regalloc import RegAllocation
from stc.sched.schedule import Schedule


class TestMirSpillOrder(unittest.TestCase):
    def test_load_store_follow_cycle_order(self) -> None:
        circuit = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("and", 0, 1),  # node 2 @ cycle 0
                ("xor", 2, 1),  # node 3 @ cycle 1
            ],
            outputs=[(3, False)],
            gate_count=2,
        )
        schedule = Schedule(gate_cycle={0: 0, 1: 1})
        allocation = RegAllocation(
            reg_assignment={2: 2, 3: 2},
            spills=[2],
            loads=[(2, 2, 1)],
            stores=[(2, 2, 1)],
        )

        mir = circuit_to_mir(circuit, schedule, allocation)
        kinds = [type(inst).__name__ for inst in mir.instructions]
        self.assertEqual(kinds, ["Binary", "Store", "Load", "Binary"])
        self.assertIsInstance(mir.instructions[1], Store)
        self.assertIsInstance(mir.instructions[2], Load)
        self.assertIsInstance(mir.instructions[3], Binary)

    def test_within_cycle_order_is_deterministic(self) -> None:
        circuit = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("and", 0, 1),  # gate 0 -> node 2
                ("xor", 0, 1),  # gate 1 -> node 3
            ],
            outputs=[(3, False)],
            gate_count=2,
        )
        # Intentionally unsorted gate insertion order for the same cycle.
        schedule = Schedule(gate_cycle={1: 0, 0: 0})
        allocation = RegAllocation(
            reg_assignment={2: 2, 3: 3},
            spills=[2, 3],
            # Intentionally unsorted by node/register.
            loads=[(3, 3, 0), (2, 2, 0)],
            stores=[(3, 3, 0), (2, 2, 0)],
        )

        mir = circuit_to_mir(circuit, schedule, allocation)
        kinds = [type(inst).__name__ for inst in mir.instructions]
        # Expect store(load) order by node id, then gate order by gate index.
        self.assertEqual(kinds, ["Store", "Store", "Load", "Load", "Binary", "Binary"])
        self.assertEqual(mir.instructions[0].src.id, 2)
        self.assertEqual(mir.instructions[1].src.id, 3)
        self.assertEqual(mir.instructions[2].dst.id, 2)
        self.assertEqual(mir.instructions[3].dst.id, 3)
        self.assertEqual(mir.instructions[4].dst.id, 2)  # gate 0 first
        self.assertEqual(mir.instructions[5].dst.id, 3)  # gate 1 second


if __name__ == "__main__":
    unittest.main()
