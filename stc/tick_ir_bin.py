from __future__ import annotations

from typing import Any

from stc.tick_ir import TickIR


def _require_msgpack() -> Any:
    try:
        import msgpack  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("msgpack is required for binary TickIR") from exc
    return msgpack


def write_tick_ir_bin(ir: TickIR, path: str) -> None:
    msgpack = _require_msgpack()
    data = ir.to_dict()
    packed = msgpack.packb(data, use_bin_type=True)
    with open(path, "wb") as f:
        f.write(packed)


def read_tick_ir_bin(path: str) -> TickIR:
    msgpack = _require_msgpack()
    with open(path, "rb") as f:
        data = msgpack.unpackb(f.read(), raw=False)
    if not isinstance(data, dict):
        raise ValueError("invalid TickIR binary payload")
    return TickIR.from_dict(data)
