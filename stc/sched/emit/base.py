from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from stc.sched.schedule import Schedule
from stc.sched.regalloc import RegAllocation


@runtime_checkable
class Emitter(Protocol):
    @property
    def name(self) -> str: ...

    def emit(
        self,
        schedule: Schedule,
        allocation: RegAllocation,
        gates: list,
        input_bits: int,
        outputs: list,
        function_name: str = "circuit",
    ) -> str: ...


class BaseEmitter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def emit(
        self,
        schedule: Schedule,
        allocation: RegAllocation,
        gates: list,
        input_bits: int,
        outputs: list,
        function_name: str = "circuit",
    ) -> str:
        pass

    def emit_circuit(
        self,
        circuit,
        schedule: Schedule,
        allocation: RegAllocation,
        function_name: str = "circuit",
    ) -> str:
        return self.emit(
            schedule,
            allocation,
            list(circuit.gates),
            circuit.input_bits,
            list(circuit.outputs),
            function_name,
        )

    def _gates_by_cycle(
        self, schedule: Schedule, num_gates: int
    ) -> dict[int, list[int]]:
        result: dict[int, list[int]] = {}
        for g_idx, cycle in schedule.gate_cycle.items():
            if cycle not in result:
                result[cycle] = []
            result[cycle].append(g_idx)
        return result
