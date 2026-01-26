import unittest

from stc.packed_circuit import PackedCircuitState
from stc.packed_regions import (
    RegionCaps,
    build_dep_graph,
    compute_region_interface,
    partition_into_regions,
)


class TestPackedRegions(unittest.TestCase):
    def test_interface_and_partition(self) -> None:
        # inputs: 0,1 (words)
        # g0: xor(0,1) -> node2
        # g1: shl(node2, 1) -> node3
        # output: node3
        c = PackedCircuitState(
            word_bits=64,
            input_words=2,
            output_words=1,
            gates=(("xor", 0, 1), ("shl", 2, 1, 0)),
            outputs=((3, False),),
        )
        g = build_dep_graph(c)
        ins, outs = compute_region_interface(g, {0})
        self.assertEqual(ins, (0, 1))
        self.assertEqual(outs, (2,))

        regions = partition_into_regions(g, RegionCaps(max_gates=1, max_boundary=999))
        self.assertEqual([r.gate_indices for r in regions], [(0,), (1,)])


if __name__ == "__main__":
    unittest.main()
