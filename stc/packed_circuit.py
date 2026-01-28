from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PackedGate = tuple[str, int, int] | tuple[str, int, int, int]
"""
Packed gate encoding (word-level).

Node indices:
- Inputs are nodes 0..input_words-1
- Each gate produces one output node at index input_words + gate_index

Gate forms:
- (op, a, b): binary ops ("xor","and","or","add","sub")
- (op, a, imm): unary ops with immediate ("shl","lshr") where imm in [0,63]
- ("not", a, 0): unary not encoded as 3-tuple for simplicity
- ("const", imm, 0): constant word node; imm is a 64-bit value
- ("ult", a, b): unsigned compare; returns 1 if a<b else 0 (boolean-in-word form)
"""


@dataclass(frozen=True)
class PackedCircuitState:
    """
    Word-level circuit suitable for regioning and word-wise code generation.

    Semantics:
    - Each node is a 64-bit word (uint64) value.
    - Operations are word-wise (bitwise ops, shifts, and optionally add/sub).

    This is intended as the primary lowering target for sequential designs,
    with the existing bit-level CircuitState retained as a fallback/debug path.
    """

    word_bits: int
    input_words: int
    output_words: int
    gates: tuple[PackedGate, ...]
    outputs: tuple[tuple[int, bool], ...]

    def __post_init__(self) -> None:
        if self.word_bits != 64:
            raise ValueError("PackedCircuitState currently supports word_bits=64 only")
        if self.input_words < 0 or self.output_words < 0:
            raise ValueError("invalid input/output word counts")

    def to_dict(self) -> dict[str, Any]:
        return {
            "word_bits": self.word_bits,
            "input_words": self.input_words,
            "output_words": self.output_words,
            "gates": [list(g) for g in self.gates],
            "outputs": [list(o) for o in self.outputs],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "PackedCircuitState":
        return PackedCircuitState(
            word_bits=int(data["word_bits"]),
            input_words=int(data["input_words"]),
            output_words=int(data["output_words"]),
            gates=tuple(tuple(g) for g in data["gates"]),
            outputs=tuple((int(n), bool(inv)) for (n, inv) in data["outputs"]),
        )

    @property
    def input_bits(self) -> int:
        # Scheduler/emit infrastructure uses `input_bits` as the count of input nodes.
        return self.input_words

    @property
    def output_bits(self) -> int:
        # Kept for symmetry with CircuitState.
        return self.output_words

    @property
    def gate_count(self) -> int:
        return len(self.gates)


def eval_packed_circuit_words(
    circuit: PackedCircuitState, inputs: list[int]
) -> list[int]:
    """
    Evaluate PackedCircuitState on concrete 64-bit inputs.

    Args:
        circuit: word-level circuit
        inputs: list of uint64 words of length circuit.input_words

    Returns:
        output words of length circuit.output_words
    """
    if len(inputs) != circuit.input_words:
        raise ValueError("input word length mismatch")

    mask = (1 << 64) - 1
    nodes: list[int] = [x & mask for x in inputs]

    for g in circuit.gates:
        if len(g) == 3:
            op, a, b = g
            if op == "xor":
                v = nodes[a] ^ nodes[b]
            elif op == "and":
                v = nodes[a] & nodes[b]
            elif op == "or":
                v = nodes[a] | nodes[b]
            elif op == "add":
                v = (nodes[a] + nodes[b]) & mask
            elif op == "sub":
                v = (nodes[a] - nodes[b]) & mask
            elif op == "ult":
                v = 1 if (nodes[a] & mask) < (nodes[b] & mask) else 0
            elif op == "not":
                v = (~nodes[a]) & mask
            elif op == "const":
                v = a & mask
            else:
                raise ValueError(f"unsupported packed gate op: {op}")
        else:
            op, a, imm, _unused = g
            if op == "shl":
                v = (nodes[a] << int(imm)) & mask
            elif op == "lshr":
                v = (nodes[a] >> int(imm)) & mask
            else:
                raise ValueError(f"unsupported packed gate op: {op}")
        nodes.append(v)

    outs: list[int] = []
    for node_idx, inv in circuit.outputs:
        v = nodes[node_idx]
        if inv:
            v = (~v) & mask
        outs.append(v)

    if len(outs) != circuit.output_words:
        raise ValueError("output_words does not match outputs list length")
    return outs
