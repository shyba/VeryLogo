from __future__ import annotations

from dataclasses import dataclass

from stc.circuit_synth import CircuitState, Gate


@dataclass(frozen=True)
class RegionCaps:
    max_nodes: int
    max_boundary: int


@dataclass(frozen=True)
class Region:
    id: int
    gate_indices: tuple[int, ...]
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]


@dataclass(frozen=True)
class DepGraph:
    """
    Dependency graph for regioning at the CircuitState gate level.

    Nodes are gate indices (0..len(gates)-1). A directed edge i->j exists if
    gate i's output is used as an input to gate j.
    """

    input_bits: int
    gates: tuple[Gate, ...]
    outputs: tuple[tuple[int, bool], ...]
    succ: tuple[tuple[int, ...], ...]
    pred: tuple[tuple[int, ...], ...]
    gate_inputs: tuple[tuple[int, ...], ...]
    consumers: dict[int, tuple[int, ...]]  # node_idx -> consuming gate indices

    def gate_output_node(self, gate_idx: int) -> int:
        return self.input_bits + gate_idx


def _gate_operands(g: Gate) -> tuple[int, ...]:
    if len(g) == 3:
        _, a, b = g
        return (a, b)
    if len(g) == 5:
        _, a, b, c, _imm8 = g
        return (a, b, c)
    raise ValueError(f"unsupported gate tuple size: {len(g)}")


def build_dep_graph(circuit: CircuitState) -> DepGraph:
    gates = tuple(circuit.gates)
    input_bits = int(circuit.input_bits)
    outputs = tuple(circuit.outputs)

    gate_inputs: list[tuple[int, ...]] = []
    succ_sets: list[set[int]] = [set() for _ in range(len(gates))]
    pred_sets: list[set[int]] = [set() for _ in range(len(gates))]
    consumers: dict[int, set[int]] = {}

    for gi, g in enumerate(gates):
        ops = _gate_operands(g)
        gate_inputs.append(ops)
        for op in ops:
            consumers.setdefault(op, set()).add(gi)
            if op >= input_bits:
                src_gate = op - input_bits
                if 0 <= src_gate < len(gates):
                    succ_sets[src_gate].add(gi)
                    pred_sets[gi].add(src_gate)

    consumers_t = {k: tuple(sorted(v)) for k, v in consumers.items()}
    succ = tuple(tuple(sorted(s)) for s in succ_sets)
    pred = tuple(tuple(sorted(s)) for s in pred_sets)

    return DepGraph(
        input_bits=input_bits,
        gates=gates,
        outputs=outputs,
        succ=succ,
        pred=pred,
        gate_inputs=tuple(gate_inputs),
        consumers=consumers_t,
    )


def compute_region_interface(
    graph: DepGraph, gate_indices: set[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Compute region interface as node indices.

    - inputs: node indices used by region but defined outside region
    - outputs: node indices defined in region but used outside region or selected as primary outputs
    """
    produced_nodes = {graph.gate_output_node(gi) for gi in gate_indices}

    in_nodes: set[int] = set()
    out_nodes: set[int] = set()

    # Inputs: operands that are not produced by this region.
    for gi in gate_indices:
        for op in graph.gate_inputs[gi]:
            if op not in produced_nodes:
                in_nodes.add(op)

    # Outputs: produced nodes that escape.
    primary_out_nodes = {n for (n, _inv) in graph.outputs}
    for gi in gate_indices:
        node = graph.gate_output_node(gi)
        if node in primary_out_nodes:
            out_nodes.add(node)
            continue
        for cgi in graph.consumers.get(node, ()):
            if cgi not in gate_indices:
                out_nodes.add(node)
                break

    return (tuple(sorted(in_nodes)), tuple(sorted(out_nodes)))


def _scc_kosaraju(graph: DepGraph) -> list[list[int]]:
    """
    Deterministic SCC computation for the gate-level dependency graph.
    """
    n = len(graph.gates)
    visited = [False] * n
    order: list[int] = []

    def dfs1(v: int) -> None:
        visited[v] = True
        for w in graph.succ[v]:
            if not visited[w]:
                dfs1(w)
        order.append(v)

    for v in range(n):
        if not visited[v]:
            dfs1(v)

    visited = [False] * n
    sccs: list[list[int]] = []

    def dfs2(v: int, acc: list[int]) -> None:
        visited[v] = True
        acc.append(v)
        for w in graph.pred[v]:
            if not visited[w]:
                dfs2(w, acc)

    for v in reversed(order):
        if not visited[v]:
            acc: list[int] = []
            dfs2(v, acc)
            acc.sort()
            sccs.append(acc)

    # Stable ordering of SCCs: sort by minimum node index.
    sccs.sort(key=lambda xs: xs[0] if xs else -1)
    return sccs


def partition_into_regions(graph: DepGraph, caps: RegionCaps) -> list[Region]:
    """
    Partition the circuit into regions using SCCs and greedy merges with caps.

    Caps are currently enforced on:
    - region node count (gates)
    - region boundary size (unpacked, in node indices)
    """
    sccs = _scc_kosaraju(graph)
    regions: list[Region] = []

    cur_gate_set: set[int] = set()
    cur_sccs: list[int] = []

    def flush(region_id: int) -> None:
        nonlocal cur_gate_set, cur_sccs
        if not cur_sccs:
            return
        inputs, outputs = compute_region_interface(graph, cur_gate_set)
        regions.append(
            Region(
                id=region_id,
                gate_indices=tuple(sorted(cur_gate_set)),
                inputs=inputs,
                outputs=outputs,
            )
        )
        cur_gate_set = set()
        cur_sccs = []

    rid = 0
    for scc in sccs:
        trial = set(cur_gate_set)
        trial.update(scc)
        trial_inputs, trial_outputs = compute_region_interface(graph, trial)
        trial_boundary = len(trial_inputs) + len(trial_outputs)
        if (
            trial
            and (len(trial) > caps.max_nodes or trial_boundary > caps.max_boundary)
            and cur_gate_set
        ):
            flush(rid)
            rid += 1
            cur_gate_set.update(scc)
            cur_sccs.extend(scc)
        else:
            cur_gate_set = trial
            cur_sccs.extend(scc)

    flush(rid)
    return regions
