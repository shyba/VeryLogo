"""
Circuit analysis for scheduling: dependencies, ASAP/ALAP bounds.
"""

from dataclasses import dataclass


@dataclass
class DependencyInfo:
    """Dependency information for a circuit."""

    predecessors: dict[int, set[int]]
    successors: dict[int, set[int]]
    input_bits: int
    num_gates: int

    def topo_order(self) -> list[int]:
        in_degree = {
            g: len(self.predecessors.get(g, set())) for g in range(self.num_gates)
        }
        ready = [g for g in range(self.num_gates) if in_degree[g] == 0]
        order = []
        while ready:
            g = ready.pop(0)
            order.append(g)
            for succ in self.successors.get(g, set()):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    ready.append(succ)
        return order


def compute_dependencies(gates: list, input_bits: int, outputs: list) -> DependencyInfo:
    num_gates = len(gates)
    predecessors: dict[int, set[int]] = {g: set() for g in range(num_gates)}
    successors: dict[int, set[int]] = {g: set() for g in range(num_gates)}

    for g_idx, gate in enumerate(gates):
        op = gate[0]
        if op == "ternary":
            operands = gate[1:4]
        elif op in ("const", "not"):
            operands = [gate[1]]
        else:
            operands = gate[1:3]

        for operand in operands:
            if operand >= input_bits:
                pred_gate = operand - input_bits
                if 0 <= pred_gate < num_gates:
                    predecessors[g_idx].add(pred_gate)
                    successors[pred_gate].add(g_idx)

    return DependencyInfo(predecessors, successors, input_bits, num_gates)


def compute_asap(
    gates: list, input_bits: int, latencies: dict[str, int]
) -> dict[int, int]:
    """
    Compute As-Soon-As-Possible schedule: earliest cycle each gate can execute.

    Inputs are available at cycle 0. Gate can execute when all inputs are ready.
    """
    num_gates = len(gates)
    asap = {}

    deps = compute_dependencies(gates, input_bits, [])

    for g_idx in deps.topo_order():
        gate = gates[g_idx]
        op = gate[0]
        if op == "ternary":
            operands = gate[1:4]
        elif op in ("const", "not"):
            operands = [gate[1]]
        else:
            operands = gate[1:3]

        earliest = 0

        for operand in operands:
            if operand >= input_bits:
                pred = operand - input_bits
                if pred in asap:
                    pred_op = gates[pred][0]
                    pred_latency = latencies.get(pred_op, 1)
                    earliest = max(earliest, asap[pred] + pred_latency)

        asap[g_idx] = earliest

    return asap


def compute_alap(
    gates: list,
    input_bits: int,
    outputs: list,
    latencies: dict[str, int],
    max_cycles: int,
) -> dict[int, int]:
    """
    Compute As-Late-As-Possible schedule: latest cycle each gate can execute
    while still meeting the max_cycles deadline.

    Works backwards from outputs.
    """
    num_gates = len(gates)
    alap = {}

    deps = compute_dependencies(gates, input_bits, outputs)

    output_gates = set()
    for out_idx, _ in outputs:
        if out_idx >= input_bits:
            output_gates.add(out_idx - input_bits)

    for g_idx in range(num_gates):
        if not deps.successors.get(g_idx) or g_idx in output_gates:
            op = gates[g_idx][0]
            latency = latencies.get(op, 1)
            alap[g_idx] = max_cycles - latency

    for g_idx in reversed(deps.topo_order()):
        if g_idx in alap:
            continue

        latest = max_cycles
        op = gates[g_idx][0]
        latency = latencies.get(op, 1)

        for succ in deps.successors.get(g_idx, set()):
            if succ in alap:
                latest = min(latest, alap[succ] - latency)

        alap[g_idx] = max(0, latest)

    return alap


def compute_slack(asap: dict[int, int], alap: dict[int, int]) -> dict[int, int]:
    """Slack = ALAP - ASAP. Lower slack = more critical."""
    return {g: alap.get(g, 0) - asap.get(g, 0) for g in asap}


def compute_depth(gates: list, input_bits: int) -> int:
    """Compute circuit depth (longest path from input to output)."""
    latencies = {
        op: 1 for op in ["xor", "and", "or", "not", "const", "ternary", "andn"]
    }
    asap = compute_asap(gates, input_bits, latencies)
    if not asap:
        return 0
    return max(asap.values()) + 1


def critical_path(
    gates: list, input_bits: int, outputs: list, latencies: dict[str, int]
) -> list[int]:
    """Return gate indices on the critical path."""
    asap = compute_asap(gates, input_bits, latencies)
    if not asap:
        return []

    max_depth = max(asap.values()) + max(latencies.values(), default=1)
    alap = compute_alap(gates, input_bits, outputs, latencies, max_depth)
    slack = compute_slack(asap, alap)

    return [g for g in range(len(gates)) if slack.get(g, 999) == 0]
