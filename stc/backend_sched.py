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


SCHEDULERS = {
    "list": lambda g, i, o, t: list_schedule(g, i, o, t, "slack"),
    "pipelined": pipelined_schedule,
}

TARGETS = {
    "avx2": AVX2,
    "avx512": AVX512,
    "ptx": PTX,
}

EMITTERS = {
    "avx2": AVX2Emitter,
    "avx512": AVX512Emitter,
    "ptx": PTXEmitter,
}


def generate_scheduled_code(
    circuit,
    target: str = "avx2",
    scheduler: str = "list",
    function_name: str = "circuit",
) -> str:
    gates = list(circuit.gates)
    input_bits = circuit.input_bits
    outputs = list(circuit.outputs)

    target_model = TARGETS[target]
    schedule_fn = SCHEDULERS[scheduler]
    emitter_cls = EMITTERS[target]

    schedule = schedule_fn(gates, input_bits, outputs, target_model)

    live_ranges = compute_live_ranges(schedule, gates, input_bits, outputs)
    allocation = allocate_registers(live_ranges, schedule, target_model.registers)

    emitter = emitter_cls()
    return emitter.emit(schedule, allocation, gates, input_bits, outputs, function_name)


def get_schedule_stats(circuit, target: str, scheduler: str) -> dict:
    gates = list(circuit.gates)
    input_bits = circuit.input_bits
    outputs = list(circuit.outputs)

    target_model = TARGETS[target]
    schedule_fn = SCHEDULERS[scheduler]

    schedule = schedule_fn(gates, input_bits, outputs, target_model)
    live_ranges = compute_live_ranges(schedule, gates, input_bits, outputs)
    allocation = allocate_registers(live_ranges, schedule, target_model.registers)

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
