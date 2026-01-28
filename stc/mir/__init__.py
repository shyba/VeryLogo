"""Machine IR for VeryLogo code generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class VReg:
    """Virtual register (before physical allocation)."""

    id: int

    def __repr__(self) -> str:
        return f"v{self.id}"


@dataclass(frozen=True)
class PReg:
    """Physical register with target-specific name."""

    id: int
    name: str

    def __repr__(self) -> str:
        return self.name


Reg = Union[VReg, PReg]


@dataclass
class MInst:
    """Base class for machine instructions."""

    dst: Reg | None


@dataclass
class Binary(MInst):
    """Binary operation (AND, OR, XOR, etc.)."""

    op: str
    a: Reg
    b: Reg


@dataclass
class Unary(MInst):
    """Unary operation (NOT, etc.)."""

    op: str
    a: Reg


@dataclass
class Ternary(MInst):
    """Ternary operation (VPTERNLOG/lop3)."""

    a: Reg
    b: Reg
    c: Reg
    imm8: int


@dataclass
class Mux(MInst):
    """Multiplexer (select ? a : b)."""

    select: Reg
    a: Reg
    b: Reg


@dataclass
class Copy(MInst):
    """Copy operation (dst = src)."""

    src: Reg


@dataclass
class Const(MInst):
    """Constant load (dst = value)."""

    value: int


@dataclass
class Load(MInst):
    """Load from spill slot."""

    slot: int


@dataclass
class Store(MInst):
    """Store to spill slot."""

    src: Reg
    slot: int


@dataclass
class MIRFunction:
    """Machine IR function representation."""

    input_regs: list[VReg]
    output_regs: list[VReg]
    instructions: list[MInst]
    num_virtual_regs: int

    def __repr__(self) -> str:
        lines = [f"MIRFunction(inputs={self.input_regs}, outputs={self.output_regs})"]
        for i, inst in enumerate(self.instructions):
            lines.append(f"  {i}: {inst}")
        return "\n".join(lines)
