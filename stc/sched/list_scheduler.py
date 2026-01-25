"""
List scheduling: simple, fast, reasonably good.

This is the "tracer bullet" scheduler - simple enough to understand and debug,
good enough to be useful, and a baseline for more sophisticated strategies.

Algorithm:
1. Compute ASAP for each gate (earliest possible cycle)
2. Compute priority (lower ALAP - ASAP = more critical = higher priority)
3. For each cycle, greedily schedule ready gates by priority
4. Respect throughput limits per operation type
"""

from stc.sched.target import TargetModel
from stc.sched.schedule import Schedule
from stc.sched.scheduler import BaseScheduler
from stc.sched.analysis import (
    compute_asap,
    compute_alap,
    compute_slack,
    compute_dependencies,
)


class ListScheduler(BaseScheduler):
    """
    Priority-based list scheduling.

    Simple and fast. Schedules gates in priority order, respecting
    dependencies and throughput constraints.
    """

    def __init__(self, priority_fn: str = "slack"):
        """
        Args:
            priority_fn: How to prioritize gates. Options:
                - "slack": ALAP - ASAP (lower = higher priority)
                - "depth": Schedule deeper gates first
                - "fifo": Topological order
        """
        self._priority_fn = priority_fn

    @property
    def name(self) -> str:
        return f"list_{self._priority_fn}"

    def schedule(
        self,
        gates: list,
        input_bits: int,
        outputs: list,
        target: TargetModel,
    ) -> Schedule:
        if not gates:
            return Schedule()

        num_gates = len(gates)
        latencies = target.latencies

        asap = compute_asap(gates, input_bits, latencies)
        max_depth = (
            max(asap.values()) + max(latencies.values(), default=1) if asap else 1
        )
        alap = compute_alap(gates, input_bits, outputs, latencies, max_depth)
        slack = compute_slack(asap, alap)

        deps = compute_dependencies(gates, input_bits, outputs)

        priority = self._compute_priority(gates, asap, alap, slack, deps)

        schedule = Schedule()
        scheduled = set()
        ready_at = {}

        for g in range(num_gates):
            if not deps.predecessors[g]:
                ready_at[g] = 0

        cycle = 0
        max_cycles = max(max_depth * 2, num_gates + 10)

        while len(scheduled) < num_gates and cycle < max_cycles:
            ready = [
                g
                for g in range(num_gates)
                if g not in scheduled and g in ready_at and ready_at[g] <= cycle
            ]

            ready.sort(key=lambda g: priority.get(g, 0))

            op_counts: dict[str, int] = {}

            for g in ready:
                op = gates[g][0]
                max_throughput = target.max_per_cycle(op)
                current = op_counts.get(op, 0)

                if current < max_throughput:
                    schedule.gate_cycle[g] = cycle
                    scheduled.add(g)
                    op_counts[op] = current + 1

                    latency = latencies.get(op, 1)
                    result_ready = cycle + latency

                    for succ in deps.successors.get(g, set()):
                        if succ not in ready_at:
                            all_preds_scheduled = all(
                                p in scheduled for p in deps.predecessors[succ]
                            )
                            if all_preds_scheduled:
                                pred_ready = 0
                                for p in deps.predecessors[succ]:
                                    p_cycle = schedule.gate_cycle[p]
                                    p_op = gates[p][0]
                                    p_latency = latencies.get(p_op, 1)
                                    pred_ready = max(pred_ready, p_cycle + p_latency)
                                ready_at[succ] = pred_ready
                        else:
                            node_idx = input_bits + g
                            succ_left = gates[succ][1]
                            succ_right = gates[succ][2] if len(gates[succ]) > 2 else -1
                            if succ_left == node_idx or succ_right == node_idx:
                                ready_at[succ] = max(ready_at[succ], result_ready)

            cycle += 1

        return schedule

    def _compute_priority(
        self,
        gates: list,
        asap: dict[int, int],
        alap: dict[int, int],
        slack: dict[int, int],
        deps,
    ) -> dict[int, int]:
        if self._priority_fn == "slack":
            return slack
        elif self._priority_fn == "depth":
            return {g: -asap.get(g, 0) for g in range(len(gates))}
        elif self._priority_fn == "fifo":
            return {g: g for g in range(len(gates))}
        else:
            return slack


def list_schedule(
    gates: list,
    input_bits: int,
    outputs: list,
    target: TargetModel,
    priority: str = "slack",
) -> Schedule:
    """Convenience function for list scheduling."""
    scheduler = ListScheduler(priority_fn=priority)
    return scheduler.schedule(gates, input_bits, outputs, target)


def schedule_circuit(circuit, target: TargetModel, priority: str = "slack") -> Schedule:
    """Schedule a CircuitState for a target."""
    return list_schedule(
        list(circuit.gates),
        circuit.input_bits,
        list(circuit.outputs),
        target,
        priority,
    )
