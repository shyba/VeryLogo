from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stc.circuit_synth import CircuitState
from stc.packed_circuit import PackedCircuitState
from stc.packed_regions import (
    RegionCaps as PackedRegionCaps,
    build_dep_graph,
    partition_into_regions,
)
from stc.regions import (
    RegionCaps as UnpackedRegionCaps,
    build_dep_graph as build_dep_graph_unpacked,
    partition_into_regions as partition_into_regions_unpacked,
)
from stc.sched.analysis import compute_depth
from stc.sched.liveness import compute_live_ranges, max_live
from stc.sched import list_schedule, pipelined_schedule, AVX512


@dataclass(frozen=True)
class AutotuneConfig:
    region_max_gates: int
    region_max_boundary: int
    scheduler: str


@dataclass(frozen=True)
class AutotuneScores:
    max_live_estimate: int
    boundary_total: int
    gate_count: int
    critical_path: int
    compile_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AutotuneCandidate:
    id: int
    config: AutotuneConfig
    scores: AutotuneScores
    composite_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "config": asdict(self.config),
            "scores": self.scores.to_dict(),
            "composite_score": self.composite_score,
        }


@dataclass(frozen=True)
class AutotuneResults:
    candidates: list[AutotuneCandidate]
    seed: int
    budget_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "seed": self.seed,
            "budget_ms": self.budget_ms,
        }


@dataclass(frozen=True)
class AutotuneChoice:
    selected_id: int
    config: AutotuneConfig
    reason: str
    scores: AutotuneScores

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_id": self.selected_id,
            "config": asdict(self.config),
            "reason": self.reason,
            "scores": self.scores.to_dict(),
        }


def _compute_composite_score(scores: AutotuneScores) -> float:
    """
    Compute composite score from individual metrics.
    Lower is better. Weighted sum of raw metrics.
    """
    boundary_weight = 1.0
    max_live_weight = 2.0
    critical_path_weight = 1.5

    score = (
        boundary_weight * scores.boundary_total
        + max_live_weight * scores.max_live_estimate
        + critical_path_weight * scores.critical_path
    )

    return score


def _score_packed_circuit_config(
    circuit: PackedCircuitState, config: AutotuneConfig
) -> AutotuneScores:
    """Score a configuration for packed circuits."""
    start_time = time.perf_counter()

    caps = PackedRegionCaps(
        max_gates=config.region_max_gates, max_boundary=config.region_max_boundary
    )
    graph = build_dep_graph(circuit)
    regions = partition_into_regions(graph, caps)

    boundary_total = sum(len(r.inputs) + len(r.outputs) for r in regions)
    gate_count = len(circuit.gates)

    target_model = AVX512
    if config.scheduler == "list":
        schedule = list_schedule(
            list(circuit.gates),
            circuit.input_words,
            list(circuit.outputs),
            target_model,
        )
    else:
        schedule = pipelined_schedule(
            list(circuit.gates),
            circuit.input_words,
            list(circuit.outputs),
            target_model,
        )

    live_ranges = compute_live_ranges(
        schedule, list(circuit.gates), circuit.input_words, list(circuit.outputs)
    )
    max_live_estimate = max_live(live_ranges)

    critical_path = compute_depth(list(circuit.gates), circuit.input_words)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return AutotuneScores(
        max_live_estimate=max_live_estimate,
        boundary_total=boundary_total,
        gate_count=gate_count,
        critical_path=critical_path,
        compile_time_ms=elapsed_ms,
    )


def _score_unpacked_circuit_config(
    circuit: CircuitState, config: AutotuneConfig
) -> AutotuneScores:
    """Score a configuration for unpacked circuits."""
    start_time = time.perf_counter()

    caps = UnpackedRegionCaps(
        max_nodes=config.region_max_gates, max_boundary=config.region_max_boundary
    )
    graph = build_dep_graph_unpacked(circuit)
    regions = partition_into_regions_unpacked(graph, caps)

    boundary_total = sum(len(r.inputs) + len(r.outputs) for r in regions)
    gate_count = len(circuit.gates)

    target_model = AVX512
    if config.scheduler == "list":
        schedule = list_schedule(
            list(circuit.gates), circuit.input_bits, list(circuit.outputs), target_model
        )
    else:
        schedule = pipelined_schedule(
            list(circuit.gates), circuit.input_bits, list(circuit.outputs), target_model
        )

    live_ranges = compute_live_ranges(
        schedule, list(circuit.gates), circuit.input_bits, list(circuit.outputs)
    )
    max_live_estimate = max_live(live_ranges)

    critical_path = compute_depth(list(circuit.gates), circuit.input_bits)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return AutotuneScores(
        max_live_estimate=max_live_estimate,
        boundary_total=boundary_total,
        gate_count=gate_count,
        critical_path=critical_path,
        compile_time_ms=elapsed_ms,
    )


def _generate_candidates(
    num_candidates: int, seed: int, gate_count: int
) -> list[AutotuneConfig]:
    """Generate candidate configurations."""
    rng = random.Random(seed)
    candidates: list[AutotuneConfig] = []

    base_configs = [
        AutotuneConfig(
            region_max_gates=50000, region_max_boundary=8192, scheduler="list"
        ),
        AutotuneConfig(
            region_max_gates=100000, region_max_boundary=16384, scheduler="list"
        ),
        AutotuneConfig(
            region_max_gates=25000, region_max_boundary=4096, scheduler="list"
        ),
        AutotuneConfig(
            region_max_gates=50000, region_max_boundary=8192, scheduler="pipelined"
        ),
    ]

    candidates.extend(base_configs[:num_candidates])

    max_gates_choices = [10000, 25000, 50000, 75000, 100000, 150000]
    max_boundary_choices = [2048, 4096, 8192, 12288, 16384, 24576]
    scheduler_choices = ["list", "pipelined"]

    while len(candidates) < num_candidates:
        config = AutotuneConfig(
            region_max_gates=rng.choice(max_gates_choices),
            region_max_boundary=rng.choice(max_boundary_choices),
            scheduler=rng.choice(scheduler_choices),
        )
        if config not in candidates:
            candidates.append(config)

    return candidates[:num_candidates]


def _select_best_candidate(candidates: list[AutotuneCandidate]) -> AutotuneChoice:
    """Select the best candidate based on composite score."""
    if not candidates:
        raise ValueError("No candidates to select from")

    best = min(candidates, key=lambda c: c.composite_score)

    return AutotuneChoice(
        selected_id=best.id,
        config=best.config,
        reason=f"Best composite score ({best.composite_score:.3f}) with "
        f"lowest estimated register pressure ({best.scores.max_live_estimate})",
        scores=best.scores,
    )


def autotune_configuration(
    circuit: CircuitState | PackedCircuitState,
    *,
    seed: int = 42,
    budget_ms: int = 60000,
    num_candidates: int = 8,
) -> tuple[AutotuneResults, AutotuneChoice]:
    """
    Autotune compilation configuration for a circuit.

    Args:
        circuit: Circuit to optimize
        seed: Random seed for reproducibility
        budget_ms: Time budget in milliseconds
        num_candidates: Number of candidate configurations to try

    Returns:
        Tuple of (results, choice) with all candidates and selected configuration
    """
    start_time = time.perf_counter()

    gate_count = len(circuit.gates)
    configs = _generate_candidates(num_candidates, seed, gate_count)

    is_packed = isinstance(circuit, PackedCircuitState)

    candidates: list[AutotuneCandidate] = []
    for i, config in enumerate(configs):
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms >= budget_ms:
            break

        if is_packed:
            scores = _score_packed_circuit_config(circuit, config)
        else:
            scores = _score_unpacked_circuit_config(circuit, config)

        composite_score = _compute_composite_score(scores)

        candidates.append(
            AutotuneCandidate(
                id=i, config=config, scores=scores, composite_score=composite_score
            )
        )

    results = AutotuneResults(candidates=candidates, seed=seed, budget_ms=budget_ms)
    choice = _select_best_candidate(candidates)

    return results, choice


def write_autotune_results(
    results: AutotuneResults, choice: AutotuneChoice, out_dir: Path
) -> None:
    """Write autotune results and choice to output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)

    from stc.autotune_bin import write_autotune_choice_bin, write_autotune_results_bin

    write_autotune_results_bin(results, out_dir / "autotune_results.bin")
    write_autotune_choice_bin(choice, out_dir / "autotune_choice.bin")
