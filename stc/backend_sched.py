from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

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
from stc.sched.emit.mir_avx512 import MIRAVX512Emitter
from stc.sched.emit.mir_ptx import MIRPTXEmitter


def _serial_schedule(gates, _input_bits, _outputs, _target, *_args, **_kwargs):
    from stc.sched.schedule import Schedule

    return Schedule(gate_cycle={g: g for g in range(len(gates))})


SCHEDULERS = {
    "list": lambda g, i, o, t, max_live_pressure=None: list_schedule(
        g, i, o, t, "slack", max_live_pressure=max_live_pressure
    ),
    "pipelined": pipelined_schedule,
    "serial": _serial_schedule,
}

TARGETS = {
    "avx2": AVX2,
    "avx512": AVX512,
    "avx512_u64": AVX512,
    "avx512_mir": AVX512,
    "ptx": PTX,
    "ptx_mir": PTX,
    "ptx_legacy": PTX,
}

EMITTERS = {
    "avx2": AVX2Emitter,
    "avx512": AVX512Emitter,
    "avx512_u64": AVX512U64Emitter,
    "avx512_mir": MIRAVX512Emitter,
    "ptx": MIRPTXEmitter,
    "ptx_mir": MIRPTXEmitter,
    "ptx_legacy": PTXEmitter,
}

def _find_rust_sched_emit_bin() -> Path | None:
    candidates = [
        Path("rust/sched_emit_rs/target/release/sched_emit_rs"),
        Path("rust/sched_emit_rs/target/debug/sched_emit_rs"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _rust_generate_scheduled_code(
    circuit,
    target: str,
    scheduler: str,
    function_name: str,
    io_split: tuple[int, int] | None,
    max_live_pressure: int | None,
) -> str | None:
    if target not in {"avx512", "avx2"} or scheduler not in {
        "list",
        "pipelined",
        "serial",
    }:
        return None
    if os.environ.get("STC_RUST_SCHED_EMIT", "0").lower() in {"0", "false", "no"}:
        return None
    bin_path = _find_rust_sched_emit_bin()
    if bin_path is None:
        return None
    fmt = os.environ.get("STC_RUST_SCHED_FORMAT", "bin").lower()
    if fmt != "bin":
        return None
    timing_enabled = os.environ.get("STC_TIMING", "0").lower() not in {
        "0",
        "false",
        "no",
    }

    def _log(label: str, start: float) -> None:
        if timing_enabled:
            elapsed = time.perf_counter() - start
            print(f"[timing] rust_sched_emit:{label}: {elapsed:.3f}s", flush=True)
    if timing_enabled:
        print(
            f"[timing] rust_sched_emit:enabled target={target} scheduler={scheduler}",
            flush=True,
        )
    with tempfile.TemporaryDirectory(prefix="stc_rust_sched_") as td:
        td_path = Path(td)
        input_path = td_path / "circuit.bin"
        output_path = td_path / "circuit.c"
        try:
            from stc.circuit_state_bin import write_circuit_state_bin
        except Exception:
            return None
        t0 = time.perf_counter()
        write_circuit_state_bin(circuit, input_path)
        _log("write_input_bin", t0)
        cmd = [
            str(bin_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--target",
            target,
            "--scheduler",
            scheduler,
            "--function-name",
            function_name,
            "--format",
            fmt,
        ]
        if timing_enabled:
            cmd.append("--timing")
        if io_split is not None:
            cmd += [
                "--input-io-bits",
                str(io_split[0]),
                "--output-io-bits",
                str(io_split[1]),
            ]
        if max_live_pressure is not None:
            cmd += ["--max-live-pressure", str(max_live_pressure)]
        try:
            t0 = time.perf_counter()
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _log("subprocess", t0)
        except subprocess.CalledProcessError:
            return None
        t0 = time.perf_counter()
        code = output_path.read_text(encoding="utf-8")
        _log("read_output", t0)
        return code


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

    if os.environ.get("STC_EMIT_NAIVE", "0").lower() not in {"0", "false", "no"}:
        emitter = emitter_cls()
        return emitter._emit_naive(gates, input_bits, outputs, function_name, io_split)

    rust_code = _rust_generate_scheduled_code(
        circuit, target, scheduler, function_name, io_split, max_live_pressure
    )
    if rust_code is not None:
        return rust_code
    if os.environ.get("STC_TIMING", "0").lower() not in {"0", "false", "no"}:
        print(
            f"[timing] rust_sched_emit:disabled target={target} scheduler={scheduler}",
            flush=True,
        )

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

    # Conservative correctness fallback for CPU targets: if spills occur and the
    # schedule has multiple gates in the same cycle, spill/load placement can be
    # sensitive to within-cycle ordering. PTX emitters now enforce deterministic
    # within-cycle order directly, so keep the parallel schedule on PTX targets.
    if allocation.num_spills and target not in {"ptx", "ptx_mir", "ptx_legacy"}:
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
