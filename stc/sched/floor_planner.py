"""A small, target-independent floor planner over explicit machine forms.

The legacy circuit scheduler intentionally remains a gate scheduler.  This
module is the first machine-level path: callers provide a value DAG whose
operations already name a logical instruction family and an operand form.  The
planner resolves that form against :class:`~stc.sched.cpu_model.CpuModel`,
tracks dependency latency, and reserves eligible execution resources.

This is deliberately not an instruction selector or a spill scheduler yet.
Those decisions must be explicit before they can be optimized.  A v1 plan
therefore rejects ambiguous forms and register pressure that cannot fit in the
declared register file instead of silently producing an optimistic schedule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from stc.sched.cpu_model import CpuModel, CpuModelError, InstructionForm


class FloorPlannerError(ValueError):
    """Raised when a floor plan cannot be represented or scheduled safely."""


@dataclass(frozen=True)
class FloorOp:
    """One operation in a value DAG.

    ``inputs`` and ``output`` are value identifiers.  Values listed in
    ``FloorProgram.inputs`` are available at cycle zero; every other input
    must be produced by exactly one operation.  ``operands`` is the exact
    Agner operand spelling (for example ``"v,v,v"``), and is mandatory when a
    family has more than one row.

    ``asm_mnemonic`` is an optional target-emitter spelling.  Keeping it
    separate from Agner's grouped instruction-family name lets the planner
    remain descriptive while a direct emitter uses a concrete opcode.
    """

    id: int
    family: str
    inputs: tuple[int, ...]
    output: int
    operands: str | None = None
    asm_mnemonic: str | None = None
    immediate: int | None = None
    form: InstructionForm | None = None

    def __post_init__(self) -> None:
        if self.id < 0:
            raise FloorPlannerError("operation id must be non-negative")
        if not self.family and self.form is None:
            raise FloorPlannerError("operation needs a family or explicit form")
        if self.output < 0:
            raise FloorPlannerError("operation output must be non-negative")
        if any(value < 0 for value in self.inputs):
            raise FloorPlannerError("operation inputs must be non-negative")
        if self.immediate is not None and not 0 <= self.immediate <= 0xFF:
            raise FloorPlannerError("operation immediate must fit in 8 bits")


@dataclass(frozen=True)
class FloorProgram:
    """A closed value DAG suitable for machine scheduling."""

    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    operations: tuple[FloorOp, ...]

    def __post_init__(self) -> None:
        input_set = set(self.inputs)
        if len(input_set) != len(self.inputs):
            raise FloorPlannerError("program inputs must be unique")
        if any(value < 0 for value in input_set):
            raise FloorPlannerError("program inputs must be non-negative")
        if any(value < 0 for value in self.outputs):
            raise FloorPlannerError("program outputs must be non-negative")
        op_ids = [op.id for op in self.operations]
        if len(op_ids) != len(set(op_ids)):
            raise FloorPlannerError("operation ids must be unique")
        produced: dict[int, int] = {}
        for op in self.operations:
            if op.output in input_set:
                raise FloorPlannerError(
                    f"operation {op.id} overwrites input value {op.output}"
                )
            if op.output in produced:
                raise FloorPlannerError(
                    f"value {op.output} has multiple producers "
                    f"({produced[op.output]} and {op.id})"
                )
            produced[op.output] = op.id
        for op in self.operations:
            for value in op.inputs:
                if value not in input_set and value not in produced:
                    raise FloorPlannerError(
                        f"operation {op.id} uses undefined value {value}"
                    )
        for value in self.outputs:
            if value not in input_set and value not in produced:
                raise FloorPlannerError(f"program output {value} is undefined")

    @property
    def input_set(self) -> frozenset[int]:
        return frozenset(self.inputs)

    @property
    def producer_by_value(self) -> dict[int, int]:
        return {op.output: op.id for op in self.operations}

    @property
    def operation_by_id(self) -> dict[int, FloorOp]:
        return {op.id: op for op in self.operations}

    def predecessors(self) -> dict[int, frozenset[int]]:
        producers = self.producer_by_value
        return {
            op.id: frozenset(producers[value] for value in op.inputs if value in producers)
            for op in self.operations
        }

    def successors(self) -> dict[int, frozenset[int]]:
        result: dict[int, set[int]] = {op.id: set() for op in self.operations}
        producers = self.producer_by_value
        for op in self.operations:
            for value in op.inputs:
                producer = producers.get(value)
                if producer is not None:
                    result[producer].add(op.id)
        return {op_id: frozenset(users) for op_id, users in result.items()}


@dataclass(frozen=True)
class FloorPlacement:
    """Cycle, selected form, and concrete resource assignment for one op."""

    cycle: int
    form: InstructionForm
    resources: tuple[str, ...]


@dataclass(frozen=True)
class RegisterPlan:
    """Non-spilling value-to-register assignment for a floor plan."""

    value_to_register: Mapping[int, int]
    register_file: str

    @property
    def registers_used(self) -> int:
        return max(self.value_to_register.values(), default=-1) + 1


@dataclass(frozen=True)
class FloorSchedule:
    """A validated schedule plus its register assignment."""

    placements: Mapping[int, FloorPlacement]
    registers: RegisterPlan
    total_cycles: int

    def placement(self, op_id: int) -> FloorPlacement:
        try:
            return self.placements[op_id]
        except KeyError as exc:
            raise FloorPlannerError(f"operation {op_id} is not scheduled") from exc

    def operations_at(self, cycle: int) -> tuple[int, ...]:
        return tuple(
            op_id
            for op_id, placement in sorted(self.placements.items())
            if placement.cycle == cycle
        )

    def resource_usage(self, cycle: int) -> dict[str, int]:
        usage: dict[str, int] = {}
        for op_id in self.operations_at(cycle):
            for resource in self.placements[op_id].resources:
                usage[resource] = usage.get(resource, 0) + 1
        return usage


@dataclass(frozen=True)
class _ResolvedOp:
    op: FloorOp
    form: InstructionForm
    latency: int
    issue_limit: int


class FloorPlanner:
    """Dependency/resource list scheduler for explicit machine operations.

    The planner is intentionally conservative.  It reserves one eligible
    resource per uop, enforces the form's reciprocal-throughput ceiling, and
    treats fractional table values as a per-cycle floor.  This produces a
    valid lower-bound schedule without pretending that a fractional issue rate
    can be achieved in a single cycle.
    """

    def __init__(self, cpu: CpuModel, *, register_file: str | None = None):
        self.cpu = cpu
        self.register_file = register_file

    def plan(self, program: FloorProgram) -> FloorSchedule:
        resolved = self._resolve(program)
        predecessors = program.predecessors()
        successors = program.successors()
        critical_path = self._critical_paths(program, resolved, successors)
        unscheduled = set(resolved)
        placements: dict[int, FloorPlacement] = {}
        cycle = 0

        while unscheduled:
            ready = [
                op_id
                for op_id in unscheduled
                if all(pred in placements for pred in predecessors[op_id])
                and self._ready_cycle(op_id, predecessors, placements, resolved) <= cycle
            ]
            ready.sort(
                key=lambda op_id: (
                    -critical_path[op_id],
                    len(resolved[op_id].form.pipes),
                    op_id,
                )
            )

            remaining_issue = self._issue_capacity()
            remaining_resources = self._resource_capacities()
            selected = self._select_cycle(
                ready,
                resolved,
                remaining_issue=remaining_issue,
                remaining_resources=remaining_resources,
            )
            placed_this_cycle = bool(selected)

            for op_id, assignment in selected:
                candidate = resolved[op_id]
                placements[op_id] = FloorPlacement(
                    cycle=cycle,
                    form=candidate.form,
                    resources=assignment,
                )
                unscheduled.remove(op_id)
                placed_this_cycle = True

            if not placed_this_cycle:
                blocked = [
                    op_id
                    for op_id in unscheduled
                    if all(pred in placements for pred in predecessors[op_id])
                ]
                if not blocked:
                    raise FloorPlannerError(
                        "operation graph contains a cycle or unsatisfied dependency"
                    )
                for op_id in blocked:
                    candidate = resolved[op_id]
                    if candidate.form.uops and candidate.form.uops > self._issue_capacity():
                        raise FloorPlannerError(
                            f"operation {op_id} needs {candidate.form.uops} uops, "
                            f"but CPU issue width is {self._issue_capacity()}"
                        )
                # If resources were the only blocker, advancing the cycle is
                # correct; the next cycle has fresh port and issue capacity.
            cycle += 1

        total_cycles = max((placement.cycle for placement in placements.values()), default=-1) + 1
        registers = self._allocate_registers(program, placements, total_cycles)
        schedule = FloorSchedule(placements, registers, total_cycles)
        self.validate(program, schedule)
        return schedule

    def validate(self, program: FloorProgram, schedule: FloorSchedule) -> None:
        """Validate dependencies, latency, throughput, and port capacity."""

        resolved = self._resolve(program)
        predecessors = program.predecessors()
        if set(schedule.placements) != set(resolved):
            raise FloorPlannerError("schedule does not cover every operation")
        for op_id, candidate in resolved.items():
            placement = schedule.placements[op_id]
            if placement.form != candidate.form:
                raise FloorPlannerError(f"operation {op_id} changed instruction form")
            if len(placement.resources) != (candidate.form.uops or 0):
                raise FloorPlannerError(f"operation {op_id} has the wrong uop assignment")
            if any(resource not in candidate.form.pipes for resource in placement.resources):
                raise FloorPlannerError(f"operation {op_id} uses an ineligible resource")
            for pred in predecessors[op_id]:
                if schedule.placements[pred].cycle + resolved[pred].latency > placement.cycle:
                    raise FloorPlannerError(
                        f"operation {op_id} violates dependency latency from {pred}"
                    )

        for cycle in range(schedule.total_cycles):
            usage = schedule.resource_usage(cycle)
            for resource, used in usage.items():
                if used > self.cpu.resource(resource).capacity:
                    raise FloorPlannerError(
                        f"resource {resource} oversubscribed at cycle {cycle}"
                    )
            issue = sum(
                (resolved[op_id].form.uops or 0)
                for op_id in schedule.operations_at(cycle)
            )
            if issue > self._issue_capacity():
                raise FloorPlannerError(f"issue width exceeded at cycle {cycle}")
            forms: dict[tuple[str, str], int] = {}
            for op_id in schedule.operations_at(cycle):
                key = self._form_key(resolved[op_id].form)
                forms[key] = forms.get(key, 0) + 1
            for key, count in forms.items():
                limit = resolved[next(
                    op_id for op_id in schedule.operations_at(cycle)
                    if self._form_key(resolved[op_id].form) == key
                )].issue_limit
                if count > limit:
                    raise FloorPlannerError(
                        f"form {key} exceeds issue limit at cycle {cycle}"
                    )

    def _resolve(self, program: FloorProgram) -> dict[int, _ResolvedOp]:
        result: dict[int, _ResolvedOp] = {}
        for op in program.operations:
            if op.form is not None:
                form = op.form
            else:
                try:
                    form = self.cpu.select_form(op.family, operands=op.operands)
                except CpuModelError as exc:
                    raise FloorPlannerError(f"operation {op.id}: {exc}") from exc
            if form.latency is None:
                raise FloorPlannerError(
                    f"operation {op.id} uses form with unknown latency ({form.source})"
                )
            if form.reciprocal_throughput is None:
                raise FloorPlannerError(
                    f"operation {op.id} uses form with unknown throughput ({form.source})"
                )
            for resource in form.pipes:
                try:
                    self.cpu.resource(resource)
                except CpuModelError as exc:
                    raise FloorPlannerError(
                        f"operation {op.id} references undeclared resource {resource}"
                    ) from exc
            if not form.pipes:
                raise FloorPlannerError(f"operation {op.id} has no eligible resources")
            if form.uops is None:
                raise FloorPlannerError(
                    f"operation {op.id} has no uop count in {form.source}"
                )
            total_resource_capacity = sum(
                math.floor(self.cpu.resource(resource).capacity + 1e-9)
                for resource in set(form.pipes)
            )
            if total_resource_capacity < form.uops:
                raise FloorPlannerError(
                    f"operation {op.id} needs {form.uops} resources, but its "
                    f"eligible resources provide {total_resource_capacity}"
                )
            issue_limit = max(1, math.floor((1.0 / form.reciprocal_throughput) + 1e-9))
            result[op.id] = _ResolvedOp(
                op=op,
                form=form,
                latency=max(1, math.ceil(form.latency)),
                issue_limit=issue_limit,
            )
        return result

    def _critical_paths(
        self,
        program: FloorProgram,
        resolved: Mapping[int, _ResolvedOp],
        successors: Mapping[int, Iterable[int]],
    ) -> dict[int, int]:
        result: dict[int, int] = {}
        visiting: set[int] = set()

        def visit(op_id: int) -> int:
            if op_id in result:
                return result[op_id]
            if op_id in visiting:
                raise FloorPlannerError("operation graph contains a cycle")
            visiting.add(op_id)
            users = tuple(successors[op_id])
            result[op_id] = resolved[op_id].latency + max(
                (visit(user) for user in users), default=0
            )
            visiting.remove(op_id)
            return result[op_id]

        for op in program.operations:
            visit(op.id)
        return result

    @staticmethod
    def _ready_cycle(
        op_id: int,
        predecessors: Mapping[int, Iterable[int]],
        placements: Mapping[int, FloorPlacement],
        resolved: Mapping[int, _ResolvedOp],
    ) -> int:
        return max(
            (
                placements[pred].cycle + resolved[pred].latency
                for pred in predecessors[op_id]
            ),
            default=0,
        )

    def _issue_capacity(self) -> int:
        if self.cpu.issue_width is None:
            raise FloorPlannerError("CPU issue width is required for floor planning")
        capacity = math.floor(self.cpu.issue_width + 1e-9)
        if capacity < 1:
            raise FloorPlannerError("CPU issue width must be at least one")
        return capacity

    def _resource_capacities(self) -> dict[str, int]:
        capacities: dict[str, int] = {}
        for resource in self.cpu.resources:
            capacity = math.floor(resource.capacity + 1e-9)
            if capacity < 1:
                raise FloorPlannerError(
                    f"resource {resource.name} has no schedulable integer capacity"
                )
            capacities[resource.name] = capacity
        return capacities

    @staticmethod
    def _form_key(form: InstructionForm) -> tuple[str, str]:
        return form.family, form.operands

    def _select_cycle(
        self,
        ready: list[int],
        resolved: Mapping[int, _ResolvedOp],
        *,
        remaining_issue: int,
        remaining_resources: dict[str, int],
    ) -> list[tuple[int, tuple[str, ...]]]:
        """Choose a feasible ready subset and reserve concrete resources.

        Most Agner rows in the current table are one-uop forms.  For those,
        use augmenting-path matching so a broad ``P01`` operation cannot steal
        a narrow ``P0`` slot from a later operation.  Multi-uop rows use the
        deterministic greedy fallback until their richer resource-demand
        grammar is introduced.
        """

        if all((resolved[op_id].form.uops or 0) == 1 for op_id in ready):
            return self._select_single_uop_cycle(
                ready,
                resolved,
                remaining_issue=remaining_issue,
                remaining_resources=remaining_resources,
            )

        selected: list[tuple[int, tuple[str, ...]]] = []
        form_counts: dict[tuple[str, str], int] = {}
        for op_id in ready:
            candidate = resolved[op_id]
            uops = candidate.form.uops or 0
            key = self._form_key(candidate.form)
            if uops > remaining_issue or form_counts.get(key, 0) >= candidate.issue_limit:
                continue
            assignment = self._assign_resources(candidate.form, remaining_resources)
            if assignment is None:
                continue
            for resource in assignment:
                remaining_resources[resource] -= 1
            remaining_issue -= uops
            form_counts[key] = form_counts.get(key, 0) + 1
            selected.append((op_id, assignment))
        return selected

    def _select_single_uop_cycle(
        self,
        ready: list[int],
        resolved: Mapping[int, _ResolvedOp],
        *,
        remaining_issue: int,
        remaining_resources: dict[str, int],
    ) -> list[tuple[int, tuple[str, ...]]]:
        resource_users: dict[str, list[int]] = {
            resource: [] for resource in remaining_resources
        }
        assignments: dict[int, str] = {}
        form_counts: dict[tuple[str, str], int] = {}
        selected: list[int] = []

        def augment(op_id: int, seen_resources: set[str], seen_ops: set[int]) -> bool:
            form = resolved[op_id].form
            for resource in form.pipes:
                if resource in seen_resources:
                    continue
                seen_resources.add(resource)
                occupants = resource_users[resource]
                if len(occupants) < remaining_resources[resource]:
                    old_resource = assignments.get(op_id)
                    if old_resource is not None:
                        resource_users[old_resource].remove(op_id)
                    assignments[op_id] = resource
                    occupants.append(op_id)
                    return True
                for occupant in list(occupants):
                    if occupant in seen_ops:
                        continue
                    seen_ops.add(occupant)
                    occupants.remove(occupant)
                    previous = assignments.pop(occupant)
                    if augment(occupant, seen_resources, seen_ops):
                        assignments[op_id] = resource
                        occupants.append(op_id)
                        return True
                    assignments[occupant] = previous
                    occupants.append(occupant)
            return False

        for op_id in ready:
            if len(selected) >= remaining_issue:
                break
            candidate = resolved[op_id]
            key = self._form_key(candidate.form)
            if form_counts.get(key, 0) >= candidate.issue_limit:
                continue
            if augment(op_id, set(), {op_id}):
                selected.append(op_id)
                form_counts[key] = form_counts.get(key, 0) + 1

        return [(op_id, (assignments[op_id],)) for op_id in selected]

    @staticmethod
    def _assign_resources(
        form: InstructionForm, remaining: Mapping[str, int]
    ) -> tuple[str, ...] | None:
        if form.uops is None:
            return None
        available = dict(remaining)
        assignment: list[str] = []
        for _ in range(form.uops):
            eligible = [
                resource
                for resource in form.pipes
                if available.get(resource, 0) > 0
            ]
            if not eligible:
                return None
            # Use the least-loaded eligible resource first.  The available
            # count is the inverse of load for equal-capacity resources, and
            # lexical order keeps plans reproducible.
            chosen = max(eligible, key=lambda resource: (available[resource], resource))
            available[chosen] -= 1
            assignment.append(chosen)
        return tuple(assignment)

    def _allocate_registers(
        self,
        program: FloorProgram,
        placements: Mapping[int, FloorPlacement],
        total_cycles: int,
    ) -> RegisterPlan:
        file_name, register_count = self._register_file_capacity()
        producers = program.producer_by_value
        users: dict[int, list[int]] = {value: [] for value in program.inputs}
        for op in program.operations:
            users.setdefault(op.output, [])
            for value in op.inputs:
                users.setdefault(value, []).append(op.id)
        for value in program.outputs:
            users.setdefault(value, []).append(None)  # type: ignore[arg-type]

        intervals: list[tuple[int, int, int]] = []
        for value, value_users in users.items():
            if value in producers:
                start = placements[producers[value]].cycle
            else:
                start = 0
            end = start
            for user in value_users:
                if user is None:
                    end = max(end, total_cycles)
                else:
                    end = max(end, placements[user].cycle)
            intervals.append((start, end, value))

        assignments: dict[int, int] = {}
        active: list[tuple[int, int, int]] = []
        for start, end, value in sorted(intervals, key=lambda item: (item[0], item[1], item[2])):
            active = [item for item in active if item[0] >= start]
            used = {item[1] for item in active}
            available = next(
                (register for register in range(register_count) if register not in used),
                None,
            )
            if available is None:
                raise FloorPlannerError(
                    f"register pressure exceeds {register_count} registers at cycle {start}"
                )
            assignments[value] = available
            active.append((end, available, value))
        return RegisterPlan(assignments, file_name)

    def _register_file_capacity(self) -> tuple[str, int]:
        if not self.cpu.register_files:
            raise FloorPlannerError("CPU register file is required for floor planning")
        if self.register_file is not None:
            try:
                spec = self.cpu.register_file(self.register_file)
            except CpuModelError as exc:
                raise FloorPlannerError(str(exc)) from exc
            return spec.name, spec.count
        spec = self.cpu.register_files[0]
        return spec.name, spec.count


def plan_floor(
    program: FloorProgram,
    cpu: CpuModel,
    *,
    register_file: str | None = None,
) -> FloorSchedule:
    """Convenience wrapper for the v1 floor planner."""

    return FloorPlanner(cpu, register_file=register_file).plan(program)
