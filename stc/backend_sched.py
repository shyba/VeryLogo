from __future__ import annotations

from stc.sched import (
    list_schedule,
    pipelined_schedule,
    AVX2,
    AVX512,
    PTX,
    TargetModel,
)
from stc.sched.liveness import compute_live_ranges, max_live
from stc.sched.regalloc import allocate_registers
from stc.sched.emit import AVX2Emitter
from stc.sched.emit import AVX512Emitter
from stc.sched.emit.ptx import PTXEmitter
from stc.sched.emit.avx512_u64 import AVX512U64Emitter


SCHEDULERS = {
    "list": lambda g, i, o, t, max_live_pressure=None: list_schedule(
        g, i, o, t, "slack", max_live_pressure=max_live_pressure
    ),
    "pipelined": pipelined_schedule,
}

TARGETS = {
    "avx2": AVX2,
    "avx512": AVX512,
    "avx512_u64": AVX512,
    "ptx": PTX,
}

EMITTERS = {
    "avx2": AVX2Emitter,
    "avx512": AVX512Emitter,
    "avx512_u64": AVX512U64Emitter,
    "ptx": PTXEmitter,
}


def generate_scheduled_code(
    circuit,
    target: str = "avx2",
    scheduler: str = "list",
    function_name: str = "circuit",
    io_split: tuple[int, int] | None = None,
    max_live_pressure: int | None = None,
) -> str:
    gates = list(circuit.gates)
    input_bits = circuit.input_bits
    outputs = list(circuit.outputs)

    target_model = TARGETS[target]
    schedule_fn = SCHEDULERS[scheduler]
    emitter_cls = EMITTERS[target]

    if scheduler == "list" and max_live_pressure is not None:
        schedule = schedule_fn(
            gates, input_bits, outputs, target_model, max_live_pressure
        )
    else:
        schedule = schedule_fn(gates, input_bits, outputs, target_model)

    live_ranges = compute_live_ranges(schedule, gates, input_bits, outputs)
    allocation = allocate_registers(
        live_ranges,
        schedule,
        target_model.registers,
        gates=gates,
        input_bits=input_bits,
        outputs=outputs,
    )

    # Conservative correctness fallback: if spills occur and the schedule has
    # multiple gates in the same cycle, spill/load placement becomes sensitive
    # to within-cycle ordering. Re-schedule serially (one gate per cycle) which
    # guarantees a total order consistent with dependencies (gate indices are
    # topologically sorted in CircuitState).
    if allocation.num_spills:
        gates_per_cycle: dict[int, int] = {}
        for c in schedule.gate_cycle.values():
            gates_per_cycle[c] = gates_per_cycle.get(c, 0) + 1
        if any(n > 1 for n in gates_per_cycle.values()):
            from stc.sched.schedule import Schedule

            serial = Schedule(gate_cycle={g: g for g in range(len(gates))})
            live_ranges = compute_live_ranges(serial, gates, input_bits, outputs)
            allocation = allocate_registers(
                live_ranges,
                serial,
                target_model.registers,
                gates=gates,
                input_bits=input_bits,
                outputs=outputs,
            )
            schedule = serial

    emitter = emitter_cls()
    return emitter.emit(
        schedule,
        allocation,
        gates,
        input_bits,
        outputs,
        function_name,
        io_split,
    )


def get_schedule_stats(circuit, target: str, scheduler: str) -> dict:
    gates = list(circuit.gates)
    input_bits = circuit.input_bits
    outputs = list(circuit.outputs)

    target_model = TARGETS[target]
    schedule_fn = SCHEDULERS[scheduler]

    schedule = schedule_fn(gates, input_bits, outputs, target_model)
    live_ranges = compute_live_ranges(schedule, gates, input_bits, outputs)
    allocation = allocate_registers(
        live_ranges,
        schedule,
        target_model.registers,
        gates=gates,
        input_bits=input_bits,
        outputs=outputs,
    )

    return {
        "total_cycles": schedule.total_cycles,
        "max_live": max_live(live_ranges),
        "num_spills": allocation.num_spills,
        "gates": len(gates),
        "inputs": input_bits,
        "outputs": len(outputs),
        "target": target,
        "scheduler": scheduler,
    }
