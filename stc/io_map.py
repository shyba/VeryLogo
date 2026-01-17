from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from stc.tick_ir import BitVecType, BoolType, TickIR, Type

SCHEMA_VERSION = 1


def _width(t: Type) -> int:
    if isinstance(t, BoolType):
        return 1
    if isinstance(t, BitVecType):
        return t.width
    raise ValueError("io mapping does not support simd types")


@dataclass(frozen=True)
class IoMap:
    port: str
    inputs: dict[str, dict[str, int]]
    outputs: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "port": self.port,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }


def load_io_map(path: Path) -> IoMap:
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError("unsupported io_map schema_version")
    port = data.get("port")
    inputs = data.get("inputs")
    outputs = data.get("outputs")
    if not isinstance(port, str) or not port:
        raise ValueError("invalid port")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise ValueError("invalid inputs/outputs")
    for m in (inputs, outputs):
        for name, spec in m.items():
            if not isinstance(name, str) or not name:
                raise ValueError("invalid signal name")
            if not isinstance(spec, dict):
                raise ValueError("invalid mapping spec")
            if not isinstance(spec.get("lsb"), int) or not isinstance(
                spec.get("width"), int
            ):
                raise ValueError("invalid mapping fields")
    return IoMap(
        port=port,
        inputs={
            str(k): {"lsb": int(v["lsb"]), "width": int(v["width"])}
            for k, v in inputs.items()
        },
        outputs={
            str(k): {"lsb": int(v["lsb"]), "width": int(v["width"])}
            for k, v in outputs.items()
        },
    )


def validate_io_map(ir: TickIR, m: IoMap) -> None:
    if m.port != "B":
        raise ValueError("only PORTB is supported in the MVP")

    if set(m.inputs.keys()) != set(ir.inputs.keys()):
        raise ValueError("io_map inputs must match Tick-IR inputs")
    if set(m.outputs.keys()) != set(ir.outputs.keys()):
        raise ValueError("io_map outputs must match Tick-IR outputs")

    used: set[int] = set()

    for name, t in ir.inputs.items():
        want_w = _width(t)
        spec = m.inputs[name]
        lsb = int(spec["lsb"])
        w = int(spec["width"])
        if w != want_w:
            raise ValueError("io_map width mismatch")
        if lsb < 0 or w < 1 or lsb + w > 8:
            raise ValueError("io_map out of range")
        for b in range(lsb, lsb + w):
            if b in used:
                raise ValueError("io_map overlaps")
            used.add(b)

    for name, t in ir.outputs.items():
        want_w = _width(t)
        spec = m.outputs[name]
        lsb = int(spec["lsb"])
        w = int(spec["width"])
        if w != want_w:
            raise ValueError("io_map width mismatch")
        if lsb < 0 or w < 1 or lsb + w > 8:
            raise ValueError("io_map out of range")
        for b in range(lsb, lsb + w):
            if b in used:
                raise ValueError("io_map overlaps")
            used.add(b)


def default_io_map(ir: TickIR) -> IoMap:
    pin = 0
    inputs: dict[str, dict[str, int]] = {}
    outputs: dict[str, dict[str, int]] = {}

    for name in sorted(ir.inputs.keys()):
        w = _width(ir.inputs[name])
        if pin + w > 8:
            raise ValueError("GPIO mapping exceeds PORTB width")
        inputs[name] = {"lsb": pin, "width": w}
        pin += w

    for name in sorted(ir.outputs.keys()):
        w = _width(ir.outputs[name])
        if pin + w > 8:
            raise ValueError("GPIO mapping exceeds PORTB width")
        outputs[name] = {"lsb": pin, "width": w}
        pin += w

    m = IoMap(port="B", inputs=inputs, outputs=outputs)
    validate_io_map(ir, m)
    return m
