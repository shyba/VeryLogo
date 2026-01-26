from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from stc.packed_regions import Region as PackedRegion
from stc.packed_regions import DepGraph as PackedDepGraph
from stc.regions import Region as BitRegion
from stc.regions import DepGraph as BitDepGraph

Region = Union[PackedRegion, BitRegion]
DepGraph = Union[PackedDepGraph, BitDepGraph]


@dataclass(frozen=True)
class RegionDiagnostics:
    regions_data: dict
    regions_stats: dict


def _get_input_nodes(graph: DepGraph) -> int:
    if hasattr(graph, "input_nodes"):
        return graph.input_nodes
    elif hasattr(graph, "input_bits"):
        return graph.input_bits
    raise ValueError("Unknown graph type")


def _format_node_ref(node: int, graph: DepGraph) -> str:
    input_count = _get_input_nodes(graph)
    if node < input_count:
        return f"input.{node}"
    gate_idx = node - input_count
    if 0 <= gate_idx < len(graph.gates):
        return f"r?.out{gate_idx}"
    return f"node.{node}"


def _compute_critical_path_estimate(regions: list[Region], graph: DepGraph) -> int:
    if not regions:
        return 0

    input_count = _get_input_nodes(graph)
    region_by_id = {r.id: r for r in regions}
    gate_to_region = {}
    for r in regions:
        for gi in r.gate_indices:
            gate_to_region[gi] = r.id

    region_deps: dict[int, set[int]] = {r.id: set() for r in regions}
    for r in regions:
        for in_node in r.inputs:
            if in_node >= input_count:
                src_gate = in_node - input_count
                if src_gate in gate_to_region:
                    src_region = gate_to_region[src_gate]
                    if src_region != r.id:
                        region_deps[r.id].add(src_region)

    region_depth = {r.id: 0 for r in regions}
    for r in regions:
        for pred_id in region_deps[r.id]:
            region_depth[r.id] = max(region_depth[r.id], region_depth[pred_id] + 1)

    max_depth = max(region_depth.values()) if region_depth else 0
    avg_region_gates = (
        sum(len(r.gate_indices) for r in regions) // len(regions) if regions else 0
    )
    return (max_depth + 1) * avg_region_gates


def generate_region_diagnostics(
    regions: list[Region], graph: DepGraph
) -> RegionDiagnostics:
    if not regions:
        return RegionDiagnostics(
            regions_data={
                "regions": [],
                "total_regions": 0,
                "ordering": "topological",
            },
            regions_stats={
                "total_regions": 0,
                "total_gates": 0,
                "gates_per_region": {},
                "boundary_distribution": {},
                "critical_path_estimate": 0,
            },
        )

    input_count = _get_input_nodes(graph)
    regions_list = []
    gate_to_region = {}
    for r in regions:
        for gi in r.gate_indices:
            gate_to_region[gi] = r.id

    for r in regions:
        input_refs = []
        for node in r.inputs:
            if node < input_count:
                input_refs.append(f"input.{node}")
            else:
                src_gate = node - input_count
                if src_gate in gate_to_region:
                    src_region = gate_to_region[src_gate]
                    input_refs.append(f"r{src_region}.out{src_gate}")
                else:
                    input_refs.append(f"node.{node}")

        boundary_nodes = len(r.inputs) + len(r.outputs)

        regions_list.append(
            {
                "id": r.id,
                "gates": len(r.gate_indices),
                "inputs": input_refs,
                "outputs": len(r.outputs),
                "boundary_nodes": boundary_nodes,
            }
        )

    regions_data = {
        "regions": regions_list,
        "total_regions": len(regions),
        "ordering": "topological",
    }

    gate_counts = [len(r.gate_indices) for r in regions]
    boundary_counts = [len(r.inputs) + len(r.outputs) for r in regions]
    total_gates = sum(gate_counts)

    gates_per_region = {}
    if gate_counts:
        gates_per_region = {
            "min": min(gate_counts),
            "max": max(gate_counts),
            "mean": statistics.mean(gate_counts),
            "median": statistics.median(gate_counts),
        }

    boundary_distribution = {}
    if boundary_counts:
        boundary_distribution = {
            "min": min(boundary_counts),
            "max": max(boundary_counts),
            "mean": statistics.mean(boundary_counts),
        }

    critical_path = _compute_critical_path_estimate(regions, graph)

    regions_stats = {
        "total_regions": len(regions),
        "total_gates": total_gates,
        "gates_per_region": gates_per_region,
        "boundary_distribution": boundary_distribution,
        "critical_path_estimate": critical_path,
    }

    return RegionDiagnostics(
        regions_data=regions_data,
        regions_stats=regions_stats,
    )


def write_region_diagnostics(diagnostics: RegionDiagnostics, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    regions_path = output_dir / "regions.json"
    with open(regions_path, "w") as f:
        json.dump(diagnostics.regions_data, f, indent=2)

    stats_path = output_dir / "regions_stats.json"
    with open(stats_path, "w") as f:
        json.dump(diagnostics.regions_stats, f, indent=2)


def generate_region_dot(regions: list[Region], graph: DepGraph) -> str:
    lines = ["digraph regions {", '  rankdir="LR";', "  node [shape=box];", ""]

    input_count = _get_input_nodes(graph)
    gate_to_region = {}
    for r in regions:
        for gi in r.gate_indices:
            gate_to_region[gi] = r.id

    for r in regions:
        label = f"R{r.id}\\n{len(r.gate_indices)} gates\\n{len(r.inputs)} in / {len(r.outputs)} out"
        lines.append(f'  r{r.id} [label="{label}"];')

    lines.append("")

    region_deps: dict[int, set[int]] = {r.id: set() for r in regions}
    for r in regions:
        for in_node in r.inputs:
            if in_node >= input_count:
                src_gate = in_node - input_count
                if src_gate in gate_to_region:
                    src_region = gate_to_region[src_gate]
                    if src_region != r.id:
                        region_deps[r.id].add(src_region)

    for r_id, deps in region_deps.items():
        for dep_id in sorted(deps):
            lines.append(f"  r{dep_id} -> r{r_id};")

    lines.append("}")
    return "\n".join(lines)


def write_region_dot(regions: list[Region], graph: DepGraph, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dot_path = output_dir / "regions.dot"
    with open(dot_path, "w") as f:
        f.write(generate_region_dot(regions, graph))
