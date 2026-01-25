from __future__ import annotations

from typing import Iterable


def eval_circuitstate_bits(circuit_state: dict, in_bits: Iterable[int]) -> list[int]:
    """Evaluate a serialized CircuitState on boolean (0/1) packed-bit inputs.

    This is a reference evaluator for tests and user self-checks.
    """
    input_bits = int(circuit_state["input_bits"])
    gates = circuit_state["gates"]
    outputs = circuit_state["outputs"]
    bits = [int(b) & 1 for b in in_bits]
    if len(bits) != input_bits:
        raise ValueError(f"expected {input_bits} input bits, got {len(bits)}")

    nodes = list(bits)
    for gate in gates:
        if len(gate) == 5:
            op, a, b, c, imm8 = gate
            if op != "ternary":
                raise ValueError("unexpected 5-tuple gate")
            ia = nodes[int(a)] & 1
            ib = nodes[int(b)] & 1
            ic = nodes[int(c)] & 1
            idx = (ic << 2) | (ib << 1) | ia
            nodes.append((int(imm8) >> idx) & 1)
            continue

        op = gate[0]
        if op == "const":
            nodes.append(int(gate[1]) & 1)
        elif op == "not":
            nodes.append((nodes[int(gate[1])] ^ 1) & 1)
        elif op == "xor":
            nodes.append((nodes[int(gate[1])] ^ nodes[int(gate[2])]) & 1)
        elif op == "and":
            nodes.append((nodes[int(gate[1])] & nodes[int(gate[2])]) & 1)
        elif op == "or":
            nodes.append((nodes[int(gate[1])] | nodes[int(gate[2])]) & 1)
        else:
            raise ValueError(f"unknown gate op: {op}")

    out: list[int] = []
    for idx, inv in outputs:
        v = nodes[int(idx)] & 1
        if inv:
            v ^= 1
        out.append(v)
    return out

