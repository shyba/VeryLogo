"""CPU and instruction-resource descriptions derived from Agner tables.

The scheduler historically received only a small ``TargetModel`` containing
integer latency and throughput maps.  That is enough for a gate-level
baseline, but it is not a floor model: an instruction form can have several
operand variants, can use a set of eligible execution pipes, and can have
unknown timing for a memory form.

This module is deliberately descriptive.  It preserves every row from an
Agner CSV and keeps the machine-wide facts (issue width, resource capacities,
and register files) separate from the instruction rows.  A future scheduler
can therefore choose an instruction form and reserve resources without
silently inheriting the old ``one family -> one row`` approximation.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class CpuModelError(ValueError):
    """Raised when an Agner table cannot be represented safely."""


@dataclass(frozen=True)
class ResourceSpec:
    """A machine resource with a steady-state capacity in units/cycle."""

    name: str
    capacity: float = 1.0
    kind: str = "execution"

    def __post_init__(self) -> None:
        if not self.name:
            raise CpuModelError("resource name must not be empty")
        if not math.isfinite(self.capacity) or self.capacity <= 0:
            raise CpuModelError("resource capacity must be finite and positive")


@dataclass(frozen=True)
class RegisterFileSpec:
    """A register file visible to a target-specific allocator."""

    name: str
    count: int
    width_bits: int

    def __post_init__(self) -> None:
        if not self.name:
            raise CpuModelError("register-file name must not be empty")
        if self.count < 1:
            raise CpuModelError("register-file count must be positive")
        if self.width_bits < 1:
            raise CpuModelError("register-file width must be positive")


@dataclass(frozen=True)
class InstructionForm:
    """One operand form from an Agner instruction-table row.

    ``pipes`` are *eligible* resources, not an already assigned reservation.
    That distinction is important: a floor planner must choose among those
    resources while respecting contention between otherwise different ops.
    """

    mnemonic: str
    operands: str
    uops: int | None
    latency: float | None
    reciprocal_throughput: float | None
    pipes: tuple[str, ...]
    notes: str = ""
    family: str = ""
    features: frozenset[str] = frozenset()
    source: str = "agner"

    def __post_init__(self) -> None:
        if not self.mnemonic:
            raise CpuModelError("instruction mnemonic must not be empty")
        if self.uops is not None and self.uops < 1:
            raise CpuModelError("instruction uop count must be positive")
        for name, value in (
            ("latency", self.latency),
            ("reciprocal throughput", self.reciprocal_throughput),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise CpuModelError(f"instruction {name} must be positive")

    @property
    def issue_rate(self) -> float | None:
        """The table-implied instruction rate, when reciprocal throughput exists."""

        if self.reciprocal_throughput is None:
            return None
        return 1.0 / self.reciprocal_throughput

    @property
    def minimum_independent_chains(self) -> int | None:
        """Chains needed to reach the table-implied rate for this form.

        This is a dependency lower bound, not a complete schedule.  It is
        ``ceil(latency / reciprocal_throughput)`` and intentionally remains
        available to future modulo/resource schedulers rather than being
        baked into any GEMM-specific code.
        """

        if self.latency is None or self.reciprocal_throughput is None:
            return None
        return max(1, math.ceil(self.latency / self.reciprocal_throughput))


@dataclass(frozen=True)
class CpuModel:
    """A machine-level floor description plus all source instruction forms."""

    name: str
    instruction_forms: tuple[InstructionForm, ...]
    issue_width: float | None = None
    resources: tuple[ResourceSpec, ...] = ()
    register_files: tuple[RegisterFileSpec, ...] = ()
    features: frozenset[str] = frozenset()
    source: str = "agner"

    def __post_init__(self) -> None:
        if not self.name:
            raise CpuModelError("CPU name must not be empty")
        if self.issue_width is not None and (
            not math.isfinite(self.issue_width) or self.issue_width <= 0
        ):
            raise CpuModelError("CPU issue width must be finite and positive")
        resource_names = [resource.name for resource in self.resources]
        if len(resource_names) != len(set(resource_names)):
            raise CpuModelError("CPU resource names must be unique")
        register_names = [reg.name for reg in self.register_files]
        if len(register_names) != len(set(register_names)):
            raise CpuModelError("CPU register-file names must be unique")

    @property
    def families(self) -> tuple[str, ...]:
        """Distinct family names in source order."""

        seen: set[str] = set()
        result: list[str] = []
        for form in self.instruction_forms:
            if form.family and form.family not in seen:
                seen.add(form.family)
                result.append(form.family)
        return tuple(result)

    @property
    def resource_names(self) -> tuple[str, ...]:
        """Declared resources, or pipe names inferred from the table rows."""

        if self.resources:
            return tuple(resource.name for resource in self.resources)
        seen: set[str] = set()
        result: list[str] = []
        for form in self.instruction_forms:
            for pipe in form.pipes:
                if pipe not in seen:
                    seen.add(pipe)
                    result.append(pipe)
        return tuple(result)

    def forms_for(self, family_or_mnemonic: str) -> tuple[InstructionForm, ...]:
        """Return every form matching a family or exact mnemonic."""

        key = family_or_mnemonic.casefold()
        return tuple(
            form
            for form in self.instruction_forms
            if form.family.casefold() == key or form.mnemonic.casefold() == key
        )

    def select_form(
        self, family_or_mnemonic: str, *, operands: str | None = None
    ) -> InstructionForm:
        """Select one form, requiring an operand spelling when ambiguous."""

        forms = self.forms_for(family_or_mnemonic)
        if operands is not None:
            wanted = _normalize_operands(operands)
            forms = tuple(
                form for form in forms if _normalize_operands(form.operands) == wanted
            )
        if not forms:
            suffix = f" with operands {operands!r}" if operands is not None else ""
            raise CpuModelError(
                f"no instruction form for {family_or_mnemonic!r}{suffix}"
            )
        if len(forms) > 1:
            choices = ", ".join(form.operands or "<none>" for form in forms)
            raise CpuModelError(
                f"ambiguous instruction form {family_or_mnemonic!r}; "
                f"select operands explicitly ({choices})"
            )
        return forms[0]

    def resource(self, name: str) -> ResourceSpec:
        """Return a declared resource by name."""

        for resource in self.resources:
            if resource.name == name:
                return resource
        raise CpuModelError(f"unknown CPU resource {name!r}")

    def register_file(self, name: str) -> RegisterFileSpec:
        """Return a declared register file by name."""

        for register_file in self.register_files:
            if register_file.name == name:
                return register_file
        raise CpuModelError(f"unknown register file {name!r}")


_REQUIRED_COLUMNS = {
    "instruction",
    "operands",
    "ops",
    "latency",
    "rt_cycles",
    "pipes",
    "notes",
    "family",
}


def _normalize_operands(value: str) -> str:
    return ",".join(part.strip() for part in value.split(","))


def parse_pipe_set(value: str) -> tuple[str, ...]:
    """Parse compact Agner pipe notation such as ``P0123`` into ``P0``..."""

    text = value.strip().upper()
    if not text:
        return ()

    ports: list[str] = []
    compact = re.fullmatch(r"(?:P\d+)+", text)
    if compact:
        for match in re.finditer(r"P(\d+)", text):
            ports.extend(f"P{digit}" for digit in match.group(1))
    else:
        for token in re.split(r"[,/;\s]+", text):
            if not token:
                continue
            match = re.fullmatch(r"P(\d+)", token)
            if match is None:
                raise CpuModelError(f"unsupported Agner pipe notation {value!r}")
            ports.extend(f"P{digit}" for digit in match.group(1))

    return tuple(dict.fromkeys(ports))


def _optional_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    parsed = float(value)
    if not parsed.is_integer():
        raise CpuModelError(f"expected integer value, got {value!r}")
    return int(parsed)


def _optional_float(value: str) -> float | None:
    value = value.strip()
    return None if not value else float(value)


def _feature_tokens(notes: str) -> frozenset[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]*", notes)
    prefixes = ("avx", "sse", "fma", "bmi", "amx", "sha")
    return frozenset(
        token.casefold() for token in tokens if token.casefold().startswith(prefixes)
    )


def load_agner_forms(path: str | Path) -> tuple[InstructionForm, ...]:
    """Load every instruction-form row from an Agner CSV."""

    csv_path = Path(path)
    try:
        handle = csv_path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise CpuModelError(f"cannot read Agner table {csv_path}") from exc

    with handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(_REQUIRED_COLUMNS - columns)
        if missing:
            raise CpuModelError(
                f"Agner table {csv_path} is missing columns: {', '.join(missing)}"
            )

        forms: list[InstructionForm] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                instruction = (row.get("instruction") or "").strip()
                family = (row.get("family") or "").strip()
                notes = (row.get("notes") or "").strip()
                forms.append(
                    InstructionForm(
                        mnemonic=instruction,
                        operands=(row.get("operands") or "").strip(),
                        uops=_optional_int(row.get("ops") or ""),
                        latency=_optional_float(row.get("latency") or ""),
                        reciprocal_throughput=_optional_float(
                            row.get("rt_cycles") or ""
                        ),
                        pipes=parse_pipe_set(row.get("pipes") or ""),
                        notes=notes,
                        family=family,
                        features=_feature_tokens(notes),
                        source=f"agner:{csv_path}:{row_number}",
                    )
                )
            except (TypeError, ValueError, CpuModelError) as exc:
                raise CpuModelError(
                    f"invalid Agner row {csv_path}:{row_number}: {exc}"
                ) from exc
        return tuple(forms)


def cpu_from_agner_csv(
    path: str | Path,
    *,
    name: str | None = None,
    issue_width: float | None = None,
    resources: Iterable[ResourceSpec] | None = None,
    register_files: Iterable[RegisterFileSpec] = (),
    features: Iterable[str] = (),
) -> CpuModel:
    """Construct a CPU description from an Agner table and machine manifest.

    Agner supplies instruction forms and timing/resource eligibility.  It does
    not, by itself, define the complete architectural register file or global
    issue width, so those facts are explicit arguments rather than guessed from
    the CSV.  When no resources are supplied, pipe names are exposed with a
    neutral capacity of one unit/cycle; callers should provide measured or
    vendor-backed capacities before using them for scheduling.
    """

    csv_path = Path(path)
    forms = load_agner_forms(csv_path)
    if resources is None:
        resources = tuple(ResourceSpec(name=pipe) for pipe in _pipe_names(forms))
    resource_tuple = tuple(resources)
    register_tuple = tuple(register_files)
    return CpuModel(
        name=name or csv_path.stem,
        instruction_forms=forms,
        issue_width=issue_width,
        resources=resource_tuple,
        register_files=register_tuple,
        # Form feature tokens describe requirements; only the explicit
        # manifest describes what this CPU actually supports.
        features=frozenset(feature.casefold() for feature in features),
        source=f"agner:{csv_path}",
    )


def _pipe_names(forms: Iterable[InstructionForm]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for form in forms:
        for pipe in form.pipes:
            if pipe not in seen:
                seen.add(pipe)
                result.append(pipe)
    return tuple(result)
