from __future__ import annotations

from pathlib import Path

from stc.packed_circuit import PackedCircuitState, PackedGate


MAGIC = b"PCB1"
VERSION = 1

OP_XOR = 0
OP_AND = 1
OP_OR = 2
OP_ADD = 3
OP_SUB = 4
OP_ULT = 5
OP_NOT = 6
OP_CONST = 7
OP_SHL = 8
OP_LSHR = 9
OP_ANDNOT = 10
OP_TERNARY = 11

NAME_TO_OPCODE = {
    "xor": OP_XOR,
    "and": OP_AND,
    "or": OP_OR,
    "add": OP_ADD,
    "sub": OP_SUB,
    "ult": OP_ULT,
    "not": OP_NOT,
    "const": OP_CONST,
    "shl": OP_SHL,
    "lshr": OP_LSHR,
    "andnot": OP_ANDNOT,
    "ternary": OP_TERNARY,
}

OPCODE_TO_NAME = {v: k for k, v in NAME_TO_OPCODE.items()}


class _BinWriter:
    def __init__(self) -> None:
        self.buf = bytearray()

    def write_u8(self, v: int) -> None:
        self.buf.append(v & 0xFF)

    def write_u32(self, v: int) -> None:
        self.buf.extend(int(v).to_bytes(4, "little", signed=False))

    def write_u64(self, v: int) -> None:
        self.buf.extend(int(v).to_bytes(8, "little", signed=False))


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

    def read_u64(self) -> int:
        v = int.from_bytes(self.data[self.off : self.off + 8], "little", signed=False)
        self.off += 8
        return v


def write_packed_circuit_bin(circuit: PackedCircuitState, path: str | Path) -> None:
    w = _BinWriter()
    w.buf.extend(MAGIC)
    w.write_u8(VERSION)
    w.write_u8(circuit.word_bits)
    w.write_u32(circuit.input_words)
    w.write_u32(circuit.output_words)
    w.write_u32(len(circuit.gates))
    w.write_u32(len(circuit.outputs))
    for gate in circuit.gates:
        op = gate[0]
        opcode = NAME_TO_OPCODE.get(op)
        if opcode is None:
            raise ValueError(f"unsupported packed gate op: {op}")
        w.write_u8(opcode)
        if op in {"xor", "and", "or", "add", "sub", "ult", "andnot"}:
            _, a, b = gate
            w.write_u32(int(a))
            w.write_u32(int(b))
        elif op == "ternary":
            _, a, b, c, imm8 = gate
            w.write_u32(int(a))
            w.write_u32(int(b))
            w.write_u32(int(c))
            w.write_u8(int(imm8) & 0xFF)
        elif op == "not":
            _, a, _ = gate
            w.write_u32(int(a))
        elif op == "const":
            _, imm, _ = gate
            w.write_u64(int(imm))
        elif op in {"shl", "lshr"}:
            if len(gate) == 4:
                _, a, imm, _ = gate
            else:
                _, a, imm = gate
            w.write_u32(int(a))
            w.write_u8(int(imm) & 0xFF)
        else:
            raise ValueError(f"unsupported packed gate op: {op}")
    for node_idx, inv in circuit.outputs:
        w.write_u32(int(node_idx))
        w.write_u8(1 if inv else 0)
    Path(path).write_bytes(w.buf)


def read_packed_circuit_bin(path: str | Path) -> PackedCircuitState:
    data = Path(path).read_bytes()
    if data[:4] != MAGIC:
        raise ValueError("bad packed circuit bin magic")
    r = _BinReader(data[4:])
    version = r.read_u8()
    if version != VERSION:
        raise ValueError(f"unsupported packed circuit bin version {version}")
    word_bits = r.read_u8()
    input_words = r.read_u32()
    output_words = r.read_u32()
    gate_count = r.read_u32()
    output_count = r.read_u32()

    gates: list[PackedGate] = []
    for _ in range(gate_count):
        opcode = r.read_u8()
        op = OPCODE_TO_NAME.get(opcode)
        if op is None:
            raise ValueError(f"unknown packed gate opcode {opcode}")
        if op in {"xor", "and", "or", "add", "sub", "ult", "andnot"}:
            a = r.read_u32()
            b = r.read_u32()
            gates.append((op, a, b))
        elif op == "ternary":
            a = r.read_u32()
            b = r.read_u32()
            c = r.read_u32()
            imm8 = r.read_u8()
            gates.append((op, a, b, c, imm8))
        elif op == "not":
            a = r.read_u32()
            gates.append((op, a, 0))
        elif op == "const":
            imm = r.read_u64()
            gates.append((op, imm, 0))
        elif op in {"shl", "lshr"}:
            a = r.read_u32()
            imm = r.read_u8()
            gates.append((op, a, imm, 0))
        else:
            raise ValueError(f"unsupported packed gate op {op}")

    outputs: list[tuple[int, bool]] = []
    for _ in range(output_count):
        idx = r.read_u32()
        inv = r.read_u8() != 0
        outputs.append((idx, inv))

    return PackedCircuitState(
        word_bits=int(word_bits),
        input_words=int(input_words),
        output_words=int(output_words),
        gates=tuple(gates),
        outputs=tuple(outputs),
    )


def write_packed_circuit_bin_file(
    circuit: PackedCircuitState, path: str | Path
) -> None:
    write_packed_circuit_bin(circuit, path)


def read_packed_circuit_bin_file(path: str | Path) -> PackedCircuitState:
    return read_packed_circuit_bin(path)
