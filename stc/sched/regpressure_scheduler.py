"""
Register-pressure-aware scheduling: minimize spills on limited-register targets.

Problem: ListScheduler ignores register pressure, leading to many spills on
targets with few registers (e.g., AVR with 32 general-purpose registers).

Solution: Modify list scheduling to track the live set during scheduling and
prefer gates that reduce register pressure when pressure is high.

Key insight: A gate that consumes the last use of its inputs is "register-neutral"
or even "register-reducing". Prefer these gates when pressure is high.
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


def compute_last_use(gates: list, input_bits: int, outputs: list) -> dict[int, int]:
    """
    Compute when each node's value is last used.

    Returns a dict mapping node index to the gate index that last uses it.
    For outputs, maps to -1 to indicate the value must survive to the end.
    """
    last_use: dict[int, int] = {}

    for g_idx, gate in enumerate(gates):
        op = gate[0]
        if op == "ternary":
            operands = gate[1:4]
        elif op in ("const", "not", "shl", "lshr"):
            operands = [gate[1]]
        else:
            operands = gate[1:3]

        for operand in operands:
            if operand >= 0:
                last_use[operand] = g_idx

    for out_idx, _ in outputs:
        last_use[out_idx] = -1

    return last_use


def compute_use_counts(gates: list, input_bits: int, outputs: list) -> dict[int, int]:
    """
    Compute how many times each node's value is used.

    This helps track when values can be freed during scheduling.
    """
    use_counts: dict[int, int] = {}

    for g_idx, gate in enumerate(gates):
        op = gate[0]
        if op == "ternary":
            operands = gate[1:4]
        elif op in ("const", "not", "shl", "lshr"):
            operands = [gate[1]]
        else:
            operands = gate[1:3]

        for operand in operands:
            if operand >= 0:
                use_counts[operand] = use_counts.get(operand, 0) + 1

    for out_idx, _ in outputs:
        use_counts[out_idx] = use_counts.get(out_idx, 0) + 1

    return use_counts


class RegPressureScheduler(BaseScheduler):
    """
    Scheduler that minimizes peak register pressure.

    Modification to list scheduling:
    1. Track live set at each cycle during scheduling
    2. When multiple gates are ready, prefer gates that:
       a) Have inputs that will die (freeing registers)
       b) Don't increase pressure above target register count
    3. May delay gates if scheduling them would exceed register limit

    Modes:
    - "balanced": Standard pressure-aware scheduling
    - "use-eager": Aggressively prioritize freeing registers when pressure is high
    """

    def __init__(self, max_registers: int, mode: str = "balanced"):
        self._max_registers = max_registers
        self._mode = mode
        if mode not in ("balanced", "use-eager"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'balanced' or 'use-eager'")

    @property
    def name(self) -> str:
        return f"regpressure_{self._max_registers}"

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

        use_counts = compute_use_counts(gates, input_bits, outputs)
        remaining_uses: dict[int, int] = dict(use_counts)

        live_set: set[int] = set()
        for i in range(input_bits):
            if remaining_uses.get(i, 0) > 0:
                live_set.add(i)

        schedule = Schedule()
        scheduled = set()
        ready_at: dict[int, int] = {}

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

            if not ready:
                cycle += 1
                continue

            ready_with_priority = []
            for g in ready:
                base_priority = slack.get(g, 0)
                freed = self._count_freed_registers(
                    g, gates, input_bits, remaining_uses
                )
                pressure_delta = 1 - freed

                current_pressure = len(live_set)
                penalty = 0
                if current_pressure + pressure_delta > self._max_registers:
                    penalty = (
                        current_pressure + pressure_delta - self._max_registers
                    ) * 1000

                if self._mode == "use-eager":
                    pressure_threshold = self._max_registers * 0.7
                    if current_pressure > pressure_threshold:
                        freed_weight = -100
                    else:
                        freed_weight = -10
                else:
                    freed_weight = -10

                combined_priority = base_priority + freed * freed_weight + penalty
                ready_with_priority.append((combined_priority, g))

            ready_with_priority.sort(key=lambda x: x[0])

            op_counts: dict[str, int] = {}
            scheduled_this_cycle = []

            for _, g in ready_with_priority:
                op = gates[g][0]
                max_throughput = target.max_per_cycle(op)
                current = op_counts.get(op, 0)

                if current >= max_throughput:
                    continue

                freed = self._count_freed_registers(
                    g, gates, input_bits, remaining_uses
                )
                pressure_delta = 1 - freed
                new_pressure = len(live_set) + pressure_delta

                if new_pressure > self._max_registers and slack.get(g, 0) > 0:
                    continue

                schedule.gate_cycle[g] = cycle
                scheduled.add(g)
                op_counts[op] = current + 1
                scheduled_this_cycle.append(g)

                self._update_live_set(g, gates, input_bits, remaining_uses, live_set)

                latency = latencies.get(op, 1)
                result_ready = cycle + latency

                for succ in deps.successors.get(g, set()):
                    if succ in scheduled:
                        continue
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
                    elif succ in ready_at:
                        node_idx = input_bits + g
                        succ_gate = gates[succ]
                        succ_op = succ_gate[0]
                        if succ_op == "ternary":
                            succ_operands = succ_gate[1:4]
                        elif succ_op in ("const", "not", "shl", "lshr"):
                            succ_operands = [succ_gate[1]]
                        else:
                            succ_operands = succ_gate[1:3]
                        if node_idx in succ_operands:
                            ready_at[succ] = max(ready_at[succ], result_ready)

            cycle += 1

        return schedule

    def _count_freed_registers(
        self,
        gate_idx: int,
        gates: list,
        input_bits: int,
        remaining_uses: dict[int, int],
    ) -> int:
        """Count how many registers will be freed by scheduling this gate."""
        gate = gates[gate_idx]
        op = gate[0]
        if op == "ternary":
            operands = gate[1:4]
        elif op in ("const", "not"):
            operands = [gate[1]]
        else:
            operands = gate[1:3]

        freed = 0
        seen = set()
        for operand in operands:
            if operand >= 0 and operand not in seen:
                seen.add(operand)
                if remaining_uses.get(operand, 0) == 1:
                    freed += 1
        return freed

    def _update_live_set(
        self,
        gate_idx: int,
        gates: list,
        input_bits: int,
        remaining_uses: dict[int, int],
        live_set: set[int],
    ) -> None:
        """Update live set after scheduling a gate."""
        gate = gates[gate_idx]
        op = gate[0]
        if op == "ternary":
            operands = gate[1:4]
        elif op in ("const", "not"):
            operands = [gate[1]]
        else:
            operands = gate[1:3]

        seen = set()
        for operand in operands:
            if operand >= 0 and operand not in seen:
                seen.add(operand)
                remaining_uses[operand] = remaining_uses.get(operand, 1) - 1
                if remaining_uses[operand] <= 0:
                    live_set.discard(operand)

        node_idx = input_bits + gate_idx
        if remaining_uses.get(node_idx, 0) > 0:
            live_set.add(node_idx)


def regpressure_schedule(
    gates: list,
    input_bits: int,
    outputs: list,
    target: TargetModel,
    max_registers: int,
    mode: str = "balanced",
) -> Schedule:
    """Convenience function for register-pressure-aware scheduling."""
    scheduler = RegPressureScheduler(max_registers=max_registers, mode=mode)
    return scheduler.schedule(gates, input_bits, outputs, target)


def schedule_circuit(
    circuit, target: TargetModel, max_registers: int, mode: str = "balanced"
) -> Schedule:
    """Schedule a CircuitState for a target with register pressure awareness."""
    return regpressure_schedule(
        list(circuit.gates),
        circuit.input_bits,
        list(circuit.outputs),
        target,
        max_registers,
        mode,
    )
