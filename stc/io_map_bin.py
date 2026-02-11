from __future__ import annotations

from pathlib import Path

from stc.io_map import IoMap


MAGIC = b"IOM1"
VERSION = 1


class _BinWriter:
    def __init__(self) -> None:
        self.buf = bytearray()

    def write_u8(self, v: int) -> None:
        self.buf.append(v & 0xFF)

    def write_u32(self, v: int) -> None:
        self.buf.extend(int(v).to_bytes(4, "little", signed=False))

    def write_str(self, s: str) -> None:
        b = s.encode("utf-8")
        self.write_u32(len(b))
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

    def read_str(self) -> str:
        n = self.read_u32()
        b = self.data[self.off : self.off + n]
        self.off += n
        return b.decode("utf-8")


def write_io_map_bin(io_map: IoMap, path: str | Path) -> None:
    w = _BinWriter()
    w.buf.extend(MAGIC)
    w.write_u8(VERSION)
    w.write_str(io_map.port)
    w.write_u32(len(io_map.inputs))
    for name, spec in io_map.inputs.items():
        w.write_str(name)
        w.write_u32(int(spec["lsb"]))
        w.write_u32(int(spec["width"]))
    w.write_u32(len(io_map.outputs))
    for name, spec in io_map.outputs.items():
        w.write_str(name)
        w.write_u32(int(spec["lsb"]))
        w.write_u32(int(spec["width"]))
    Path(path).write_bytes(w.buf)


def read_io_map_bin(path: str | Path) -> IoMap:
    data = Path(path).read_bytes()
    if data[:4] != MAGIC:
        raise ValueError("bad io_map bin magic")
    r = _BinReader(data[4:])
    version = r.read_u8()
    if version != VERSION:
        raise ValueError(f"unsupported io_map bin version {version}")
    port = r.read_str()
    inputs_count = r.read_u32()
    inputs: dict[str, dict[str, int]] = {}
    for _ in range(inputs_count):
        name = r.read_str()
        lsb = r.read_u32()
        width = r.read_u32()
        inputs[name] = {"lsb": int(lsb), "width": int(width)}
    outputs_count = r.read_u32()
    outputs: dict[str, dict[str, int]] = {}
    for _ in range(outputs_count):
        name = r.read_str()
        lsb = r.read_u32()
        width = r.read_u32()
        outputs[name] = {"lsb": int(lsb), "width": int(width)}
    return IoMap(port=port, inputs=inputs, outputs=outputs)
