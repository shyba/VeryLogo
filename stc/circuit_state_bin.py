from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from stc.circuit_synth import CircuitState, Gate


MAGIC = b"CSB1"
VERSION = 1

OP_XOR = 0
OP_AND = 1
OP_OR = 2
OP_NOT = 3
OP_CONST = 4
OP_TERNARY = 5
OP_ANDNOT = 6
OP_ORNOT = 7

OPCODE_TO_NAME = {
    OP_XOR: "xor",
    OP_AND: "and",
    OP_OR: "or",
    OP_NOT: "not",
    OP_CONST: "const",
    OP_TERNARY: "ternary",
    OP_ANDNOT: "andn",
    OP_ORNOT: "ornot",
}

NAME_TO_OPCODE = {v: k for k, v in OPCODE_TO_NAME.items()}
NAME_TO_OPCODE["andnot"] = OP_ANDNOT


class _BinWriter:
    def __init__(self) -> None:
        self.buf = bytearray()

    def write_u8(self, v: int) -> None:
        self.buf.append(v & 0xFF)

    def write_u32(self, v: int) -> None:
        self.buf.extend(int(v).to_bytes(4, "little", signed=False))

    def write_bytes(self, b: bytes) -> None:
        self.buf.extend(b)


class _BinReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.off = 0

    def read_u8(self) -> int:
        v = self.data[self.off]
        self.off += 1
        return v

    def read_u32(self) -> int:
        v = int.from_bytes(self.data[self.off : self.off + 4], "little", signed=False)
        self.off += 4
        return v

    def read_bytes(self, n: int) -> bytes:
        b = self.data[self.off : self.off + n]
        self.off += n
        return b


def write_circuit_state_bin(circuit: CircuitState, path: str | Path) -> None:
    w = _BinWriter()
    w.write_bytes(MAGIC)
    w.write_u8(VERSION)
    w.write_u32(circuit.input_bits)
    w.write_u32(circuit.output_bits)
    w.write_u32(len(circuit.gates))
    w.write_u32(len(circuit.outputs))
    for gate in circuit.gates:
        if len(gate) == 5:
            op, a, b, c, imm8 = gate
            if op != "ternary":
                raise ValueError(f"unsupported 5-elem gate op: {op}")
            w.write_u8(OP_TERNARY)
            w.write_u32(a)
            w.write_u32(b)
            w.write_u32(c)
            w.write_u8(int(imm8) & 0xFF)
            continue
        if len(gate) != 3:
            raise ValueError(f"unsupported gate length: {len(gate)}")
        op, a, b = gate
        if op == "const":
            w.write_u8(OP_CONST)
            w.write_u32(int(a))
            w.write_u32(int(b))
        elif op == "not":
            w.write_u8(OP_NOT)
            w.write_u32(int(a))
        else:
            opcode = NAME_TO_OPCODE.get(op)
            if opcode is None:
                raise ValueError(f"unsupported gate op: {op}")
            w.write_u8(opcode)
            w.write_u32(int(a))
            w.write_u32(int(b))
    for node_idx, inv in circuit.outputs:
        w.write_u32(int(node_idx))
        w.write_u8(1 if inv else 0)
    Path(path).write_bytes(w.buf)


def read_circuit_state_bin(path: str | Path) -> CircuitState:
    data = Path(path).read_bytes()
    r = _BinReader(data)
    magic = r.read_bytes(4)
    if magic != MAGIC:
        raise ValueError("bad circuit_state bin magic")
    version = r.read_u8()
    if version != VERSION:
        raise ValueError(f"unsupported circuit_state bin version {version}")
    input_bits = r.read_u32()
    output_bits = r.read_u32()
    gate_count = r.read_u32()
    output_count = r.read_u32()

    gates: list[Gate] = []
    for _ in range(gate_count):
        opcode = r.read_u8()
        if opcode == OP_TERNARY:
            a = r.read_u32()
            b = r.read_u32()
            c = r.read_u32()
            imm8 = r.read_u8()
            gates.append(("ternary", a, b, c, imm8))
        elif opcode == OP_CONST:
            value = r.read_u32()
            width = r.read_u32()
            gates.append(("const", value, width))
        elif opcode == OP_NOT:
            a = r.read_u32()
            gates.append(("not", a, 0))
        else:
            op = OPCODE_TO_NAME.get(opcode)
            if op is None:
                raise ValueError(f"unknown opcode {opcode}")
            a = r.read_u32()
            b = r.read_u32()
            gates.append((op, a, b))

    outputs: list[tuple[int, bool]] = []
    for _ in range(output_count):
        idx = r.read_u32()
        inv = r.read_u8() != 0
        outputs.append((idx, inv))

    return CircuitState(
        input_bits=input_bits,
        output_bits=output_bits,
        gates=gates,
        outputs=outputs,
        gate_count=len(gates),
    )
