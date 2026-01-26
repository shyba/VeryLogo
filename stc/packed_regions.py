from __future__ import annotations

from dataclasses import dataclass

from stc.packed_circuit import PackedCircuitState, PackedGate


@dataclass(frozen=True)
class RegionCaps:
    max_gates: int
    max_boundary: int


@dataclass(frozen=True)
class Region:
    id: int
    gate_indices: tuple[int, ...]
    inputs: tuple[int, ...]  # node indices
    outputs: tuple[int, ...]  # node indices


@dataclass(frozen=True)
class DepGraph:
    input_nodes: int
    gates: tuple[PackedGate, ...]
    outputs: tuple[tuple[int, bool], ...]
    succ: tuple[tuple[int, ...], ...]
    pred: tuple[tuple[int, ...], ...]
    gate_inputs: tuple[tuple[int, ...], ...]
    consumers: dict[int, tuple[int, ...]]  # node_idx -> gate indices consuming it

    def gate_output_node(self, gate_idx: int) -> int:
        return self.input_nodes + gate_idx


def _gate_operands(g: PackedGate) -> tuple[int, ...]:
    op = g[0]
    if op == "const":
        return ()
    if op in ("not", "shl", "lshr"):
        return (int(g[1]),)
    # binary word ops
    return (int(g[1]), int(g[2]))


def build_dep_graph(circuit: PackedCircuitState) -> DepGraph:
    gates = tuple(circuit.gates)
    input_nodes = int(circuit.input_words)
    outputs = tuple(circuit.outputs)

    gate_inputs: list[tuple[int, ...]] = []
    succ_sets: list[set[int]] = [set() for _ in range(len(gates))]
    pred_sets: list[set[int]] = [set() for _ in range(len(gates))]
    consumers: dict[int, set[int]] = {}

    for gi, g in enumerate(gates):
        ops = _gate_operands(g)
        gate_inputs.append(ops)
        for opn in ops:
            consumers.setdefault(opn, set()).add(gi)
            if opn >= input_nodes:
                src_gate = opn - input_nodes
                if 0 <= src_gate < len(gates):
                    succ_sets[src_gate].add(gi)
                    pred_sets[gi].add(src_gate)

    return DepGraph(
        input_nodes=input_nodes,
        gates=gates,
        outputs=outputs,
        succ=tuple(tuple(sorted(s)) for s in succ_sets),
        pred=tuple(tuple(sorted(s)) for s in pred_sets),
        gate_inputs=tuple(gate_inputs),
        consumers={k: tuple(sorted(v)) for k, v in consumers.items()},
    )


def compute_region_interface(
    graph: DepGraph, gate_indices: set[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    produced_nodes = {graph.gate_output_node(gi) for gi in gate_indices}
    in_nodes: set[int] = set()
    out_nodes: set[int] = set()

    for gi in gate_indices:
        for op in graph.gate_inputs[gi]:
            if op not in produced_nodes:
                in_nodes.add(op)

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

    sccs.sort(key=lambda xs: xs[0] if xs else -1)
    return sccs


def partition_into_regions(graph: DepGraph, caps: RegionCaps) -> list[Region]:
    sccs = _scc_kosaraju(graph)
    regions: list[Region] = []

    cur: set[int] = set()

    def flush(rid: int) -> None:
        nonlocal cur
        if not cur:
            return
        ins, outs = compute_region_interface(graph, cur)
        regions.append(
            Region(
                id=rid,
                gate_indices=tuple(sorted(cur)),
                inputs=ins,
                outputs=outs,
            )
        )
        cur = set()

    rid = 0
    for scc in sccs:
        trial = set(cur)
        trial.update(scc)
        ins, outs = compute_region_interface(graph, trial)
        boundary = len(ins) + len(outs)
        if cur and (len(trial) > caps.max_gates or boundary > caps.max_boundary):
            flush(rid)
            rid += 1
            cur.update(scc)
        else:
            cur = trial

    flush(rid)
    return regions
