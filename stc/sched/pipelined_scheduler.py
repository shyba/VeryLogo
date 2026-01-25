"""
Pipelined scheduler: models instruction pipelining for high-latency targets.

On GPUs and modern CPUs, instructions are pipelined:
- Issue an instruction every cycle (if operands ready)
- Result arrives `latency` cycles later
- Multiple instructions can be "in flight" simultaneously

This scheduler properly models this, achieving much better throughput
on targets like PTX where latency is high but so is throughput.
"""

from stc.sched.target import TargetModel
from stc.sched.schedule import Schedule
from stc.sched.scheduler import BaseScheduler
from stc.sched.analysis import compute_dependencies


class PipelinedScheduler(BaseScheduler):
    """
    Scheduler that models instruction pipelining.

    Key insight: a gate can be issued at cycle C if its operands
    were issued at cycle <= C - latency. Multiple gates can be
    in-flight simultaneously.
    """

    @property
    def name(self) -> str:
        return "pipelined"

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

        deps = compute_dependencies(gates, input_bits, outputs)

        schedule = Schedule()
        scheduled = set()

        result_ready_at: dict[int, int] = {}
        for i in range(input_bits):
            result_ready_at[i] = 0

        ready_at: dict[int, int] = {}
        for g in range(num_gates):
            if not deps.predecessors[g]:
                ready_at[g] = 0
            else:
                ready_at[g] = -1

        cycle = 0
        max_cycles = num_gates * max(latencies.values(), default=1) + 100

        while len(scheduled) < num_gates and cycle < max_cycles:
            for g in range(num_gates):
                if g in scheduled:
                    continue
                if ready_at[g] >= 0:
                    continue

                all_ready = True
                earliest = 0
                for pred in deps.predecessors[g]:
                    pred_node = input_bits + pred
                    if pred_node not in result_ready_at:
                        all_ready = False
                        break
                    earliest = max(earliest, result_ready_at[pred_node])

                if all_ready:
                    ready_at[g] = earliest

            ready = [
                g
                for g in range(num_gates)
                if g not in scheduled
                and ready_at.get(g, -1) >= 0
                and ready_at[g] <= cycle
            ]

            ready.sort(key=lambda g: (ready_at[g], g))

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
                    node_idx = input_bits + g
                    result_ready_at[node_idx] = cycle + latency

            cycle += 1

        return schedule


def pipelined_schedule(
    gates: list,
    input_bits: int,
    outputs: list,
    target: TargetModel,
) -> Schedule:
    """Convenience function for pipelined scheduling."""
    scheduler = PipelinedScheduler()
    return scheduler.schedule(gates, input_bits, outputs, target)
