import tempfile
import unittest
from pathlib import Path

from stc.packed_circuit import PackedCircuitState
from stc.packed_regions import RegionCaps, build_dep_graph, partition_into_regions
from stc.region_diagnostics import (
    generate_region_diagnostics,
    generate_region_dot,
    write_region_diagnostics,
    write_region_dot,
)


class TestRegionDiagnostics(unittest.TestCase):
    def test_empty_circuit(self):
        circuit = PackedCircuitState(
            word_bits=64,
            input_words=0,
            output_words=0,
            gates=(),
            outputs=(),
        )
        graph = build_dep_graph(circuit)
        regions = partition_into_regions(
            graph, RegionCaps(max_gates=10, max_boundary=5)
        )

        diagnostics = generate_region_diagnostics(regions, graph)

        self.assertEqual(diagnostics.regions_data["total_regions"], 0)
        self.assertEqual(diagnostics.regions_stats["total_gates"], 0)
        self.assertEqual(len(diagnostics.regions_data["regions"]), 0)

    def test_single_region(self):
        circuit = PackedCircuitState(
            word_bits=64,
            input_words=2,
            output_words=1,
            gates=(
                ("xor", 0, 1),
                ("and", 0, 2),
                ("or", 2, 3),
            ),
            outputs=((4, False),),
        )
        graph = build_dep_graph(circuit)
        regions = partition_into_regions(
            graph, RegionCaps(max_gates=100, max_boundary=50)
        )

        diagnostics = generate_region_diagnostics(regions, graph)

        self.assertEqual(diagnostics.regions_data["total_regions"], 1)
        self.assertEqual(diagnostics.regions_stats["total_gates"], 3)
        self.assertEqual(len(diagnostics.regions_data["regions"]), 1)

        region = diagnostics.regions_data["regions"][0]
        self.assertEqual(region["id"], 0)
        self.assertEqual(region["gates"], 3)
        self.assertEqual(region["outputs"], 1)
        self.assertGreater(region["boundary_nodes"], 0)

    def test_multi_region(self):
        gates = []
        for i in range(20):
            if i == 0:
                gates.append(("xor", 0, 1))
            else:
                gates.append(("and", i + 1, i + 2))

        circuit = PackedCircuitState(
            word_bits=64,
            input_words=2,
            output_words=1,
            gates=tuple(gates),
            outputs=((21, False),),
        )
        graph = build_dep_graph(circuit)
        regions = partition_into_regions(
            graph, RegionCaps(max_gates=5, max_boundary=10)
        )

        diagnostics = generate_region_diagnostics(regions, graph)

        self.assertGreater(diagnostics.regions_data["total_regions"], 1)
        self.assertEqual(diagnostics.regions_stats["total_gates"], 20)

        self.assertIn("min", diagnostics.regions_stats["gates_per_region"])
        self.assertIn("max", diagnostics.regions_stats["gates_per_region"])
        self.assertIn("mean", diagnostics.regions_stats["gates_per_region"])
        self.assertIn("median", diagnostics.regions_stats["gates_per_region"])

        self.assertIn("min", diagnostics.regions_stats["boundary_distribution"])
        self.assertIn("max", diagnostics.regions_stats["boundary_distribution"])
        self.assertIn("mean", diagnostics.regions_stats["boundary_distribution"])

        self.assertGreaterEqual(diagnostics.regions_stats["critical_path_estimate"], 0)

    def test_write_diagnostics(self):
        circuit = PackedCircuitState(
            word_bits=64,
            input_words=2,
            output_words=1,
            gates=(
                ("xor", 0, 1),
                ("and", 0, 2),
            ),
            outputs=((3, False),),
        )
        graph = build_dep_graph(circuit)
        regions = partition_into_regions(
            graph, RegionCaps(max_gates=100, max_boundary=50)
        )

        diagnostics = generate_region_diagnostics(regions, graph)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_region_diagnostics(diagnostics, output_dir)

            regions_path = output_dir / "regions.bin"
            stats_path = output_dir / "regions_stats.bin"

            self.assertTrue(regions_path.exists())
            self.assertTrue(stats_path.exists())

            from stc.region_diagnostics_bin import (
                read_regions_bin,
                read_regions_stats_bin,
            )

            regions_data = read_regions_bin(regions_path)
            self.assertEqual(regions_data["total_regions"], 1)
            self.assertEqual(regions_data["ordering"], "topological")

            stats_data = read_regions_stats_bin(stats_path)
            self.assertEqual(stats_data["total_gates"], 2)
            self.assertIn("gates_per_region", stats_data)

    def test_generate_dot(self):
        circuit = PackedCircuitState(
            word_bits=64,
            input_words=2,
            output_words=1,
            gates=(
                ("xor", 0, 1),
                ("and", 0, 2),
                ("or", 2, 3),
            ),
            outputs=((4, False),),
        )
        graph = build_dep_graph(circuit)
        regions = partition_into_regions(graph, RegionCaps(max_gates=2, max_boundary=5))

        dot = generate_region_dot(regions, graph)

        self.assertIn("digraph regions", dot)
        self.assertIn("rankdir", dot)
        for r in regions:
            self.assertIn(f"r{r.id}", dot)

    def test_write_dot(self):
        circuit = PackedCircuitState(
            word_bits=64,
            input_words=2,
            output_words=1,
            gates=(
                ("xor", 0, 1),
                ("and", 0, 2),
            ),
            outputs=((3, False),),
        )
        graph = build_dep_graph(circuit)
        regions = partition_into_regions(
            graph, RegionCaps(max_gates=100, max_boundary=50)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_region_dot(regions, graph, output_dir)

            dot_path = output_dir / "regions.dot"
            self.assertTrue(dot_path.exists())

            with open(dot_path) as f:
                dot_content = f.read()
                self.assertIn("digraph regions", dot_content)

    def test_region_input_references(self):
        circuit = PackedCircuitState(
            word_bits=64,
            input_words=2,
            output_words=1,
            gates=(
                ("xor", 0, 1),
                ("and", 0, 2),
                ("or", 3, 2),
            ),
            outputs=((4, False),),
        )
        graph = build_dep_graph(circuit)
        regions = partition_into_regions(
            graph, RegionCaps(max_gates=2, max_boundary=10)
        )

        diagnostics = generate_region_diagnostics(regions, graph)

        for region in diagnostics.regions_data["regions"]:
            for input_ref in region["inputs"]:
                self.assertTrue(
                    input_ref.startswith("input.")
                    or input_ref.startswith("r")
                    or input_ref.startswith("node.")
                )

    def test_critical_path_estimate(self):
        gates = []
        for i in range(10):
            if i == 0:
                gates.append(("xor", 0, 1))
            else:
                gates.append(("and", i + 1, i + 1))

        circuit = PackedCircuitState(
            word_bits=64,
            input_words=2,
            output_words=1,
            gates=tuple(gates),
            outputs=((11, False),),
        )
        graph = build_dep_graph(circuit)
        regions = partition_into_regions(
            graph, RegionCaps(max_gates=3, max_boundary=10)
        )

        diagnostics = generate_region_diagnostics(regions, graph)

        critical_path = diagnostics.regions_stats["critical_path_estimate"]
        self.assertGreater(critical_path, 0)
        self.assertLessEqual(critical_path, len(gates) * 2)


if __name__ == "__main__":
    unittest.main()
