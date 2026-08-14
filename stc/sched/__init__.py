"""
Circuit scheduling for SIMD/GPU targets.

This module provides:
- Target models describing hardware constraints
- Schedule representation (gate -> cycle, register assignments)
- Pluggable scheduling strategies
- Register allocation

Usage:
    from stc.sched import SSE2, AVX2, PTX, list_schedule

    schedule = list_schedule(circuit, AVX2)
    print(f"Cycles: {schedule.total_cycles}, Regs: {schedule.max_live}")
"""

from stc.sched.cpu_model import (
    CpuModel,
    CpuModelError,
    InstructionForm,
    RegisterFileSpec,
    ResourceSpec,
    cpu_from_agner_csv,
    load_agner_forms,
    parse_pipe_set,
)
from stc.sched.target import TargetModel, SSE2, SSE2_X64, AVX2, AVX512, PTX
from stc.sched.schedule import Schedule, ScheduleStats
from stc.sched.scheduler import Scheduler
from stc.sched.list_scheduler import ListScheduler, list_schedule
from stc.sched.pipelined_scheduler import PipelinedScheduler, pipelined_schedule
from stc.sched.regpressure_scheduler import RegPressureScheduler, regpressure_schedule
from stc.sched.analysis import (
    compute_asap,
    compute_alap,
    compute_dependencies,
    compute_depth,
)
from stc.sched.liveness import (
    LiveRange,
    compute_live_ranges,
    live_at_cycle,
    max_live,
    interference_graph,
)
from stc.sched.floor_planner import (
    FloorOp,
    FloorPlanner,
    FloorPlannerError,
    FloorProgram,
    FloorPlacement,
    FloorSchedule,
    RegisterPlan,
    plan_floor,
)
from stc.sched.emit_x86_asm import emit_x86_64_asm
from stc.sched.floor_backend import (
    circuit_to_floor_program,
    emit_circuit_x86_64_asm,
    plan_circuit_floor,
)

__all__ = [
    "TargetModel",
    "CpuModel",
    "CpuModelError",
    "InstructionForm",
    "RegisterFileSpec",
    "ResourceSpec",
    "cpu_from_agner_csv",
    "load_agner_forms",
    "parse_pipe_set",
    "SSE2",
    "SSE2_X64",
    "AVX2",
    "AVX512",
    "PTX",
    "Schedule",
    "ScheduleStats",
    "Scheduler",
    "ListScheduler",
    "list_schedule",
    "PipelinedScheduler",
    "pipelined_schedule",
    "RegPressureScheduler",
    "regpressure_schedule",
    "compute_asap",
    "compute_alap",
    "compute_dependencies",
    "compute_depth",
    "LiveRange",
    "compute_live_ranges",
    "live_at_cycle",
    "max_live",
    "interference_graph",
    "FloorOp",
    "FloorPlanner",
    "FloorPlannerError",
    "FloorProgram",
    "FloorPlacement",
    "FloorSchedule",
    "RegisterPlan",
    "plan_floor",
    "emit_x86_64_asm",
    "circuit_to_floor_program",
    "emit_circuit_x86_64_asm",
    "plan_circuit_floor",
]
