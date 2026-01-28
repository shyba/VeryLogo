"""AVX-512 emitter using Machine IR layer."""

from __future__ import annotations

from stc.circuit_synth import CircuitState
from stc.mir.emit_avx512 import emit_avx512
from stc.mir.lower import circuit_to_mir
from stc.sched.emit.base import BaseEmitter
from stc.sched.regalloc import RegAllocation
from stc.sched.schedule import Schedule


class MIRAVX512Emitter(BaseEmitter):
    """AVX-512 emitter via Machine IR layer."""

    @property
    def name(self) -> str:
        return "avx512_mir"

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
            from stc.sched.emit import AVX512Emitter

            fallback = AVX512Emitter()
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
        return emit_avx512(mir, allocation, function_name)
