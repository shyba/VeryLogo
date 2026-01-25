"""
Abstract scheduler interface.

All scheduling strategies implement this protocol, making them
interchangeable and testable against the same benchmarks.
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from stc.sched.target import TargetModel
from stc.sched.schedule import Schedule


@runtime_checkable
class Scheduler(Protocol):
    """Protocol for scheduling strategies."""

    @property
    def name(self) -> str:
        """Human-readable name for this scheduler."""
        ...

    def schedule(
        self,
        gates: list,
        input_bits: int,
        outputs: list,
        target: TargetModel,
    ) -> Schedule:
        """
        Schedule gates for the given target.

        Args:
            gates: List of (op, left, right) tuples
            input_bits: Number of input bits (node indices 0..input_bits-1)
            outputs: List of (node_idx, inverted) pairs
            target: Hardware constraints

        Returns:
            Schedule mapping gates to cycles and registers
        """
        ...


class BaseScheduler(ABC):
    """Base class for schedulers with common utilities."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def schedule(
        self,
        gates: list,
        input_bits: int,
        outputs: list,
        target: TargetModel,
    ) -> Schedule:
        pass

    def schedule_circuit(self, circuit, target: TargetModel) -> Schedule:
        """Convenience method to schedule a CircuitState directly."""
        return self.schedule(
            list(circuit.gates),
            circuit.input_bits,
            list(circuit.outputs),
            target,
        )
