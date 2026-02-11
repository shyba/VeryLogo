from __future__ import annotations

from pathlib import Path


MAGIC = b"KVB1"
VERSION = 1


class _BinWriter:
    def __init__(self) -> None:
        self.buf = bytearray()

    def write_u8(self, v: int) -> None:
        self.buf.append(v & 0xFF)

    def write_u32(self, v: int) -> None:
        self.buf.extend(int(v).to_bytes(4, "little", signed=False))

    def write_f64(self, v: float) -> None:
        import struct

        self.buf.extend(struct.pack("<d", float(v)))

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

    def read_f64(self) -> float:
        import struct

        v = struct.unpack("<d", self.data[self.off : self.off + 8])[0]
        self.off += 8
        return float(v)

    def read_str(self) -> str:
        n = self.read_u32()
        b = self.data[self.off : self.off + n]
        self.off += n
        return b.decode("utf-8")


def write_kv_bin(data: dict[str, float | int], path: str | Path) -> None:
    w = _BinWriter()
    w.buf.extend(MAGIC)
    w.write_u8(VERSION)
    w.write_u32(len(data))
    for k, v in data.items():
        w.write_str(str(k))
        w.write_f64(float(v))
    Path(path).write_bytes(w.buf)


def read_kv_bin(path: str | Path) -> dict[str, float]:
    data = Path(path).read_bytes()
    if data[:4] != MAGIC:
        raise ValueError("bad kv bin magic")
    r = _BinReader(data[4:])
    version = r.read_u8()
    if version != VERSION:
        raise ValueError(f"unsupported kv bin version {version}")
    count = r.read_u32()
    out: dict[str, float] = {}
    for _ in range(count):
        key = r.read_str()
        out[key] = r.read_f64()
    return out
