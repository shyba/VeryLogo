from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class YosysPort:
    name: str
    direction: str
    bits: list[int]


@dataclass(frozen=True)
class YosysCell:
    name: str
    type: str
    port_directions: dict[str, str]
    connections: dict[str, list[int]]
    parameters: dict[str, str]
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class YosysModule:
    name: str
    ports: dict[str, YosysPort]
    cells: dict[str, YosysCell]
    attributes: dict[str, str] = field(default_factory=dict)
    parameter_defaults: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class YosysDesign:
    top: str
    modules: dict[str, YosysModule]


def module_hdl_name(module: YosysModule) -> str:
    """Return the source HDL name for a (possibly parameterized) module."""

    raw = module.attributes.get("hdlname", module.name)
    return raw.lstrip("\\").rsplit("\\", 1)[-1]


def is_gemm_module(module: YosysModule) -> bool:
    return module_hdl_name(module).lower().split(".")[-1] == "gemm"


def is_gemm_cell(design: YosysDesign, cell: YosysCell) -> bool:
    module = design.modules.get(cell.type)
    if module is not None and is_gemm_module(module):
        return True
    # Hand-authored normalized fixtures often use the source module name
    # directly rather than Yosys' $paramod...\gemm spelling.
    return cell.type.lstrip("\\").rsplit("\\", 1)[-1].lower() == "gemm"


def load_design(path: str | Path, *, top: str | None = None) -> YosysDesign:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    modules_raw = data.get("modules")
    if not isinstance(modules_raw, dict) or not modules_raw:
        raise ValueError("normalized.json has no modules")

    if top is None:
        if len(modules_raw) != 1:
            raise ValueError("top module must be specified when multiple modules exist")
        top_name = next(iter(modules_raw.keys()))
    else:
        top_name = top
        if top_name not in modules_raw:
            raise ValueError("top module not found")

    def parse_module(name: str, m: Any) -> YosysModule:
        if not isinstance(m, dict):
            raise ValueError("invalid module entry")

        def parse_bit(b: Any) -> int:
            if isinstance(b, int):
                return int(b)
            if isinstance(b, str) and b in {"0", "1"}:
                return int(b, 10)
            if isinstance(b, str) and b.lower() in {"x", "z"}:
                return 0
            raise ValueError("invalid bit")

        ports_raw = m.get("ports", {})
        ports: dict[str, YosysPort] = {}
        for port_name, port_data in ports_raw.items():
            direction = port_data.get("direction")
            bits = port_data.get("bits")
            if direction not in {"input", "output"}:
                raise ValueError("unsupported port direction")
            if not isinstance(bits, list):
                raise ValueError("invalid port bits")
            try:
                bits_parsed = [parse_bit(b) for b in bits]
            except ValueError as e:
                raise ValueError("invalid port bits") from e
            ports[port_name] = YosysPort(
                name=port_name, direction=direction, bits=list(bits_parsed)
            )

        cells_raw = m.get("cells", {})
        if not isinstance(cells_raw, dict):
            raise ValueError("invalid cells map")

        cells: dict[str, YosysCell] = {}
        for cell_name, cell_data in cells_raw.items():
            if not isinstance(cell_data, dict):
                raise ValueError("invalid cell entry")
            cell_type = cell_data.get("type")
            if not isinstance(cell_type, str) or not cell_type:
                raise ValueError("invalid cell type")
            port_directions = cell_data.get("port_directions", {})
            if not isinstance(port_directions, dict) or not all(
                isinstance(k, str) and isinstance(v, str)
                for k, v in port_directions.items()
            ):
                raise ValueError("invalid cell port_directions")
            if not set(port_directions.values()).issubset({"input", "output"}):
                raise ValueError("invalid cell port direction")
            connections = cell_data.get("connections", {})
            if not isinstance(connections, dict):
                raise ValueError("invalid cell connections")
            conns: dict[str, list[int]] = {}
            for port, bits in connections.items():
                if not isinstance(port, str):
                    raise ValueError("invalid connection port")
                if not isinstance(bits, list):
                    raise ValueError("invalid connection bits")
                try:
                    bits_parsed = [parse_bit(b) for b in bits]
                except ValueError as e:
                    raise ValueError("invalid connection bits") from e
                conns[port] = list(bits_parsed)
            params_raw = cell_data.get("parameters", {})
            if not isinstance(params_raw, dict) or not all(
                isinstance(k, str) and isinstance(v, (str, int))
                for k, v in params_raw.items()
            ):
                raise ValueError("invalid cell parameters")
            cells[cell_name] = YosysCell(
                name=cell_name,
                type=cell_type,
                port_directions=dict(port_directions),
                connections=conns,
                parameters={k: str(v) for k, v in params_raw.items()},
                attributes={
                    str(k): str(v)
                    for k, v in (cell_data.get("attributes", {}) or {}).items()
                    if isinstance(k, str) and isinstance(v, (str, int))
                },
            )

        attrs_raw = m.get("attributes", {})
        if not isinstance(attrs_raw, dict):
            raise ValueError("invalid module attributes")
        parameter_defaults_raw = m.get("parameter_default_values", {})
        if not isinstance(parameter_defaults_raw, dict):
            raise ValueError("invalid module parameter defaults")
        return YosysModule(
            name=name,
            ports=ports,
            cells=cells,
            attributes={
                str(k): str(v)
                for k, v in attrs_raw.items()
                if isinstance(k, str) and isinstance(v, (str, int))
            },
            parameter_defaults={
                str(k): str(v)
                for k, v in parameter_defaults_raw.items()
                if isinstance(k, str) and isinstance(v, (str, int))
            },
        )

    modules: dict[str, YosysModule] = {}
    for name, m in modules_raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError("invalid module name")
        modules[name] = parse_module(name, m)

    return YosysDesign(top=top_name, modules=modules)
