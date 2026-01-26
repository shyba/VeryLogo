import unittest
from pathlib import Path
import tempfile
import json

from stc.autotune import (
    AutotuneConfig,
    AutotuneScores,
    _compute_composite_score,
    _generate_candidates,
    autotune_configuration,
    write_autotune_results,
)
from stc.circuit_synth import CircuitState
from stc.packed_circuit import PackedCircuitState


class TestAutotune(unittest.TestCase):
    def test_compute_composite_score(self) -> None:
        scores = AutotuneScores(
            max_live_estimate=1000,
            boundary_total=200,
            gate_count=5000,
            critical_path=50,
            compile_time_ms=123.4,
        )
        composite = _compute_composite_score(scores)
        self.assertIsInstance(composite, float)
        self.assertGreater(composite, 0.0)

    def test_compute_composite_score_lower_is_better(self) -> None:
        good_scores = AutotuneScores(
            max_live_estimate=100,
            boundary_total=50,
            gate_count=1000,
            critical_path=10,
            compile_time_ms=50.0,
        )
        bad_scores = AutotuneScores(
            max_live_estimate=1000,
            boundary_total=500,
            gate_count=10000,
            critical_path=100,
            compile_time_ms=500.0,
        )
        good_composite = _compute_composite_score(good_scores)
        bad_composite = _compute_composite_score(bad_scores)
        self.assertLess(good_composite, bad_composite)

    def test_generate_candidates_deterministic(self) -> None:
        candidates1 = _generate_candidates(num_candidates=8, seed=42, gate_count=1000)
        candidates2 = _generate_candidates(num_candidates=8, seed=42, gate_count=1000)
        self.assertEqual(candidates1, candidates2)

    def test_generate_candidates_count(self) -> None:
        candidates = _generate_candidates(num_candidates=10, seed=42, gate_count=1000)
        self.assertEqual(len(candidates), 10)

    def test_generate_candidates_different_seeds(self) -> None:
        candidates1 = _generate_candidates(num_candidates=8, seed=42, gate_count=1000)
        candidates2 = _generate_candidates(num_candidates=8, seed=99, gate_count=1000)
        self.assertNotEqual(candidates1, candidates2)

    def test_autotune_configuration_unpacked_circuit(self) -> None:
        circuit = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[("xor", 0, 1), ("and", 2, 0), ("xor", 3, 1)],
            outputs=[(4, False)],
            gate_count=3,
        )
        results, choice = autotune_configuration(
            circuit, seed=42, budget_ms=5000, num_candidates=4
        )
        self.assertEqual(len(results.candidates), 4)
        self.assertEqual(results.seed, 42)
        self.assertIn(choice.selected_id, range(4))
        self.assertIsInstance(choice.config, AutotuneConfig)
        self.assertGreater(len(choice.reason), 0)

    def test_autotune_configuration_packed_circuit(self) -> None:
        circuit = PackedCircuitState(
            word_bits=64,
            input_words=2,
            output_words=1,
            gates=(("xor", 0, 1), ("and", 2, 0)),
            outputs=((3, False),),
        )
        results, choice = autotune_configuration(
            circuit, seed=42, budget_ms=5000, num_candidates=3
        )
        self.assertLessEqual(len(results.candidates), 3)
        self.assertEqual(results.seed, 42)
        self.assertIn(choice.selected_id, range(len(results.candidates)))

    def test_write_autotune_results(self) -> None:
        circuit = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[("xor", 0, 1)],
            outputs=[(2, False)],
            gate_count=1,
        )
        results, choice = autotune_configuration(
            circuit, seed=42, budget_ms=2000, num_candidates=2
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            write_autotune_results(results, choice, out_dir)

            results_path = out_dir / "autotune_results.json"
            choice_path = out_dir / "autotune_choice.json"

            self.assertTrue(results_path.exists())
            self.assertTrue(choice_path.exists())

            results_data = json.loads(results_path.read_text())
            self.assertIn("candidates", results_data)
            self.assertIn("seed", results_data)
            self.assertEqual(results_data["seed"], 42)

            choice_data = json.loads(choice_path.read_text())
            self.assertIn("selected_id", choice_data)
            self.assertIn("config", choice_data)
            self.assertIn("reason", choice_data)
            self.assertIn("scores", choice_data)

    def test_autotune_respects_budget(self) -> None:
        circuit = CircuitState(
            input_bits=4,
            output_bits=2,
            gates=[
                ("xor", 0, 1),
                ("and", 2, 3),
                ("xor", 4, 5),
                ("and", 6, 1),
            ],
            outputs=[(7, False), (6, False)],
            gate_count=4,
        )
        results, choice = autotune_configuration(
            circuit, seed=42, budget_ms=100, num_candidates=20
        )
        self.assertGreater(len(results.candidates), 0)
        self.assertLessEqual(len(results.candidates), 20)


if __name__ == "__main__":
    unittest.main()
