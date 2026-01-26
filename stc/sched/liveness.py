"""
Liveness analysis for scheduling: live ranges, interference graphs.

This module computes which values are live (needed for future use) at each
cycle of a schedule. This information is essential for register allocation
and spill cost estimation.

A value's live range spans from when it is produced (cycle 0 for inputs,
scheduled cycle for gates) to when it is last used.
"""

from dataclasses import dataclass

from stc.sched.schedule import Schedule


@dataclass
class LiveRange:
    """
    Live range for a node's value.

    A live range spans from when a value is produced until the last cycle
    where it is consumed by another gate or appears in the circuit outputs.
    """

    node: int
    start: int
    end: int
    is_input: bool = False

    def overlaps(self, other: "LiveRange") -> bool:
        """Check if this live range overlaps with another."""
        return self.start <= other.end and other.start <= self.end

    def contains(self, cycle: int) -> bool:
        """Check if this range contains the given cycle."""
        return self.start <= cycle <= self.end

    @property
    def length(self) -> int:
        """Number of cycles this value is live."""
        return self.end - self.start + 1


def compute_live_ranges(
    schedule: Schedule,
    gates: list,
    input_bits: int,
    outputs: list,
) -> dict[int, LiveRange]:
    """
    Compute live ranges for all nodes in a scheduled circuit.

    Args:
        schedule: Gate cycle assignments
        gates: List of (op, left, right) tuples
        input_bits: Number of input bits (node indices 0..input_bits-1)
        outputs: List of (node_idx, inverted) pairs

    Returns:
        Dict mapping node index to LiveRange for all nodes with non-empty ranges
    """
    if not gates and not outputs:
        return {}

    total_cycles = schedule.total_cycles
    final_cycle = max(total_cycles - 1, 0)

    last_use: dict[int, int] = {}

    for g_idx, gate in enumerate(gates):
        cycle = schedule.gate_cycle.get(g_idx, 0)
        op = gate[0]

        if op == "ternary":
            operands = gate[1:4]
        elif op in ("const", "not", "shl", "lshr"):
            operands = [gate[1]]
        else:
            operands = gate[1:3]

        for operand in operands:
            if operand >= 0:
                last_use[operand] = max(last_use.get(operand, 0), cycle)

    for out_idx, _ in outputs:
        last_use[out_idx] = max(last_use.get(out_idx, 0), final_cycle)

    ranges: dict[int, LiveRange] = {}

    for i in range(input_bits):
        if i in last_use:
            ranges[i] = LiveRange(
                node=i,
                start=0,
                end=last_use[i],
                is_input=True,
            )

    for g_idx in range(len(gates)):
        node_idx = input_bits + g_idx
        start = schedule.gate_cycle.get(g_idx, 0)
        end = last_use.get(node_idx, start)
        ranges[node_idx] = LiveRange(
            node=node_idx,
            start=start,
            end=end,
            is_input=False,
        )

    return ranges


def live_at_cycle(live_ranges: dict[int, LiveRange], cycle: int) -> set[int]:
    """
    Return the set of node indices that are live at a given cycle.

    A value is live at cycle c if c is within its live range [start, end].

    Args:
        live_ranges: Dict mapping node index to LiveRange objects
        cycle: The cycle to query

    Returns:
        Set of node indices whose values are live at the given cycle
    """
    return {lr.node for lr in live_ranges.values() if lr.start <= cycle <= lr.end}


def max_live(live_ranges: dict[int, LiveRange], total_cycles: int = 0) -> int:
    """
    Compute the maximum number of simultaneously live values.

    This is a lower bound on the number of registers needed.

    Args:
        live_ranges: Dict mapping node index to LiveRange objects
        total_cycles: Optional total cycles (unused, for API compatibility)

    Returns:
        Maximum number of values live at any cycle
    """
    if not live_ranges:
        return 0

    ranges_list = list(live_ranges.values())
    min_start = min(lr.start for lr in ranges_list)
    max_end = max(lr.end for lr in ranges_list)

    max_count = 0
    for cycle in range(min_start, max_end + 1):
        count = sum(1 for lr in ranges_list if lr.start <= cycle <= lr.end)
        max_count = max(max_count, count)

    return max_count


def interference_graph(live_ranges: dict[int, LiveRange]) -> dict[int, set[int]]:
    """
    Build an interference graph from live ranges.

    Two nodes interfere (cannot share a register) if their live ranges overlap.
    The graph is represented as an adjacency list: node -> set of interfering nodes.

    Args:
        live_ranges: Dict mapping node index to LiveRange objects

    Returns:
        Adjacency list mapping each node to its set of interfering nodes
    """
    graph: dict[int, set[int]] = {}

    ranges_list = list(live_ranges.values())

    for lr in ranges_list:
        if lr.node not in graph:
            graph[lr.node] = set()

    for i, lr1 in enumerate(ranges_list):
        for lr2 in ranges_list[i + 1 :]:
            if lr1.overlaps(lr2):
                graph[lr1.node].add(lr2.node)
                graph[lr2.node].add(lr1.node)

    return graph
