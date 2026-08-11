from __future__ import annotations

import time

import z3


class CpuBudget:
    def __init__(self, budget_ms: float) -> None:
        self._cpu_start = time.process_time()
        self._wall_start = time.time()
        self._budget = budget_ms

    def remaining_ms(self) -> float:
        return self._budget - (time.process_time() - self._cpu_start) * 1000.0

    def exhausted(self) -> bool:
        return self.remaining_ms() <= 0

    def load_factor(self) -> float:
        cpu = time.process_time() - self._cpu_start
        wall = time.time() - self._wall_start
        return wall / max(cpu, 1e-6)


def make_solver(*, timeout_ms: int | None = None) -> z3.Solver:
    solver = z3.Solver()
    solver.set("random_seed", 0)
    if timeout_ms is not None:
        solver.set("timeout", timeout_ms)
    return solver


def check_with_budget(
    solver: z3.Solver,
    budget: CpuBudget,
    *,
    start_wall_ms: int = 50,
    max_wall_ms: int = 10000,
) -> z3.CheckSatResult:
    wall = start_wall_ms
    while not budget.exhausted():
        remaining_cpu = budget.remaining_ms()
        wall = min(wall, max(int(remaining_cpu * max(budget.load_factor(), 1.0)), 1))
        solver.set("timeout", wall)
        result = solver.check()
        if result != z3.unknown:
            return result
        wall = min(wall * 2, max_wall_ms)
    return z3.unknown
