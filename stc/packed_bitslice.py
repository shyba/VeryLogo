from __future__ import annotations

from stc.circuit_synth import CircuitState
from stc.packed_circuit import PackedCircuitState, PackedGate
from stc.tick_ir_to_circuit_state import PackedLayout
from stc.tick_ir_to_packed_circuit_state import PackedWordLayout


def bitslice_circuit_to_packed(
    circuit: CircuitState, layout: PackedLayout
) -> tuple[PackedCircuitState, PackedWordLayout]:
    """
    Convert a bit-level CircuitState into a word-level PackedCircuitState
    where each word bit is an independent instance (bit-sliced across bits).

    Layout mapping:
      - word index == bit index in the packed bit layout
      - PackedWordLayout uses lsw=lsb and width_bits=width
    """

    supported_ops = {
        "xor",
        "and",
        "andnot",
        "andn",
        "or",
        "not",
        "add",
        "sub",
        "ult",
        "shl",
        "lshr",
        "const",
        "ternary",
    }
    remapped_gates: list[PackedGate] = []
    for g in circuit.gates:
        op = g[0]
        if op not in supported_ops:
            raise ValueError(f"unsupported gate op for packed bitslice: {op}")
        if op == "andn":
            if len(g) != 3:
                raise ValueError("andn gate expects 3-tuple")
            _, a, b = g
            remapped_gates.append(("andnot", int(a), int(b)))
            continue
        remapped_gates.append(g)

    packed = PackedCircuitState(
        word_bits=64,
        input_words=circuit.input_bits,
        output_words=len(circuit.outputs),
        gates=tuple(remapped_gates),
        outputs=tuple(circuit.outputs),
    )

    def _convert_map(m: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
        return {k: {"lsw": v["lsb"], "width_bits": v["width"]} for k, v in m.items()}

    layout_w = PackedWordLayout(
        inputs=_convert_map(layout.inputs),
        state=_convert_map(layout.state),
        outputs=_convert_map(layout.outputs),
        next_state=_convert_map(layout.next_state),
        input_words=layout.input_bits,
        output_words=layout.output_bits,
        mode="bitslice",
    )
    return packed, layout_w
