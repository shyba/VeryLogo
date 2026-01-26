import unittest

from stc.circuit_synth import CircuitState
from stc.regions import (
    RegionCaps,
    build_dep_graph,
    compute_region_interface,
    partition_into_regions,
)


class TestRegions(unittest.TestCase):
    def test_compute_region_interface(self) -> None:
        # inputs: 0,1
        # g0: xor(0,1) -> node2
        # g1: and(node2,0) -> node3
        circuit = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[("xor", 0, 1), ("and", 2, 0)],
            outputs=[(3, False)],
            gate_count=2,
        )
        graph = build_dep_graph(circuit)

        ins0, outs0 = compute_region_interface(graph, {0})
        self.assertEqual(ins0, (0, 1))
        self.assertEqual(outs0, (2,))

        ins1, outs1 = compute_region_interface(graph, {1})
        self.assertEqual(ins1, (0, 2))
        self.assertEqual(outs1, (3,))

    def test_partition_merge_respects_caps(self) -> None:
        circuit = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[("xor", 0, 1), ("and", 2, 0)],
            outputs=[(3, False)],
            gate_count=2,
        )
        graph = build_dep_graph(circuit)

        # Small max_nodes forces separate regions.
        regions = partition_into_regions(
            graph, RegionCaps(max_nodes=1, max_boundary=999)
        )
        self.assertEqual([r.gate_indices for r in regions], [(0,), (1,)])

        # Larger caps should merge into one region.
        regions2 = partition_into_regions(
            graph, RegionCaps(max_nodes=2, max_boundary=999)
        )
        self.assertEqual([r.gate_indices for r in regions2], [(0, 1)])
        self.assertEqual(regions2[0].inputs, (0, 1))
        self.assertEqual(regions2[0].outputs, (3,))

    def test_partition_deterministic(self) -> None:
        circuit = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[("xor", 0, 1), ("xor", 1, 2), ("and", 3, 4)],
            outputs=[(5, False)],
            gate_count=3,
        )
        graph = build_dep_graph(circuit)
        caps = RegionCaps(max_nodes=2, max_boundary=999)
        r1 = partition_into_regions(graph, caps)
        r2 = partition_into_regions(graph, caps)
        self.assertEqual(r1, r2)


if __name__ == "__main__":
    unittest.main()
