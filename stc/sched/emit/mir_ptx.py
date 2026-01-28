"""PTX emitter using Machine IR layer."""

from __future__ import annotations

from stc.circuit_synth import CircuitState
from stc.mir.emit_ptx import emit_ptx
from stc.mir.lower import circuit_to_mir
from stc.sched.emit.base import BaseEmitter
from stc.sched.regalloc import RegAllocation
from stc.sched.schedule import Schedule


class MIRPTXEmitter(BaseEmitter):
    """PTX emitter via Machine IR layer."""

    @property
    def name(self) -> str:
        return "ptx_mir"

    def emit(
        self,
        schedule: Schedule,
        allocation: RegAllocation,
        gates: list,
        input_bits: int,
        outputs: list,
        function_name: str = "circuit",
        io_split: tuple[int, int] | None = None,
    ) -> str:
        if io_split is not None:
            from stc.sched.emit.ptx import PTXEmitter

            fallback = PTXEmitter()
            return fallback.emit(
                schedule,
                allocation,
                gates,
                input_bits,
                outputs,
                function_name,
                io_split,
            )

        circuit = CircuitState(
            input_bits=input_bits,
            output_bits=len(outputs),
            gates=gates,
            outputs=outputs,
            gate_count=len(gates),
        )

        mir = circuit_to_mir(circuit, schedule, allocation)
        return emit_ptx(mir, allocation, function_name)
