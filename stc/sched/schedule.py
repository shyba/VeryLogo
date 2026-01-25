"""
Schedule representation: mapping gates to cycles and registers.
"""

from dataclasses import dataclass, field


@dataclass
class ScheduleStats:
    """Statistics about a schedule's quality."""

    total_cycles: int
    max_live: int
    spills: int
    gates_per_cycle: list[int]

    @property
    def avg_parallelism(self) -> float:
        if self.total_cycles == 0:
            return 0.0
        return sum(self.gates_per_cycle) / self.total_cycles

    def __str__(self) -> str:
        return (
            f"cycles={self.total_cycles}, max_live={self.max_live}, "
            f"spills={self.spills}, avg_parallel={self.avg_parallelism:.1f}"
        )


@dataclass
class Schedule:
    """
    Complete schedule: gate assignments to cycles and registers.

    gate_cycle[g] = cycle when gate g is scheduled
    gate_reg[g] = physical register holding gate g's output
    input_reg[i] = physical register holding input i

    Cycles are 0-indexed. A gate scheduled at cycle c has its result
    available at cycle c + latency.
    """

    gate_cycle: dict[int, int] = field(default_factory=dict)
    gate_reg: dict[int, int] = field(default_factory=dict)
    input_reg: dict[int, int] = field(default_factory=dict)
    spill_loads: list[tuple[int, int, int]] = field(default_factory=list)
    spill_stores: list[tuple[int, int, int]] = field(default_factory=list)

    def cycle_of(self, gate_idx: int) -> int:
        return self.gate_cycle.get(gate_idx, -1)

    def reg_of(self, node_idx: int, input_bits: int) -> int:
        if node_idx < input_bits:
            return self.input_reg.get(node_idx, -1)
        return self.gate_reg.get(node_idx - input_bits, -1)

    def gates_at_cycle(self, cycle: int) -> list[int]:
        return [g for g, c in self.gate_cycle.items() if c == cycle]

    @property
    def total_cycles(self) -> int:
        if not self.gate_cycle:
            return 0
        return max(self.gate_cycle.values()) + 1

    def compute_stats(self, gates: list, input_bits: int) -> ScheduleStats:
        if not self.gate_cycle:
            return ScheduleStats(0, 0, 0, [])

        total_cycles = self.total_cycles
        gates_per_cycle = [0] * total_cycles
        for c in self.gate_cycle.values():
            gates_per_cycle[c] += 1

        live_at_cycle = self._compute_liveness(gates, input_bits, total_cycles)
        max_live = max(live_at_cycle) if live_at_cycle else 0

        spills = len(self.spill_loads)

        return ScheduleStats(total_cycles, max_live, spills, gates_per_cycle)

    def _compute_liveness(
        self, gates: list, input_bits: int, total_cycles: int
    ) -> list[int]:
        last_use = {}

        for g_idx, gate in enumerate(gates):
            op = gate[0]
            cycle = self.gate_cycle.get(g_idx, 0)

            if op == "ternary":
                operands = gate[1:4]
            elif op in ("const", "not"):
                operands = [gate[1]]
            else:
                operands = gate[1:3]

            for operand in operands:
                if operand >= 0:
                    last_use[operand] = max(last_use.get(operand, 0), cycle)

        live_at = [0] * (total_cycles + 1)

        for i in range(input_bits):
            start = 0
            end = last_use.get(i, 0)
            for c in range(start, min(end + 1, total_cycles + 1)):
                live_at[c] += 1

        for g_idx in range(len(gates)):
            node_idx = input_bits + g_idx
            start = self.gate_cycle.get(g_idx, 0)
            end = last_use.get(node_idx, start)
            for c in range(start, min(end + 1, total_cycles + 1)):
                live_at[c] += 1

        return live_at

    def validate(
        self, gates: list, input_bits: int, latencies: dict[str, int]
    ) -> list[str]:
        errors = []

        for g_idx, gate in enumerate(gates):
            op = gate[0]
            if g_idx not in self.gate_cycle:
                errors.append(f"Gate {g_idx} not scheduled")
                continue

            my_cycle = self.gate_cycle[g_idx]

            if op == "ternary":
                operands = gate[1:4]
            elif op in ("const", "not"):
                operands = [gate[1]]
            else:
                operands = gate[1:3]

            for operand in operands:
                if operand >= input_bits:
                    pred_gate = operand - input_bits
                    if pred_gate in self.gate_cycle:
                        pred_cycle = self.gate_cycle[pred_gate]
                        pred_op = gates[pred_gate][0]
                        ready = pred_cycle + latencies.get(pred_op, 1)
                        if my_cycle < ready:
                            errors.append(
                                f"Gate {g_idx} at cycle {my_cycle} uses gate {pred_gate} "
                                f"ready at cycle {ready}"
                            )

        return errors
