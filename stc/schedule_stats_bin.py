from __future__ import annotations

from pathlib import Path


MAGIC = b"SSB1"
VERSION = 1


class _BinWriter:
    def __init__(self) -> None:
        self.buf = bytearray()

    def write_u8(self, v: int) -> None:
        self.buf.append(v & 0xFF)

    def write_u32(self, v: int) -> None:
        self.buf.extend(int(v).to_bytes(4, "little", signed=False))

    def write_u64(self, v: int) -> None:
        self.buf.extend(int(v).to_bytes(8, "little", signed=False))

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

    def read_u64(self) -> int:
        v = int.from_bytes(self.data[self.off : self.off + 8], "little", signed=False)
        self.off += 8
        return v

    def read_str(self) -> str:
        n = self.read_u32()
        b = self.data[self.off : self.off + n]
        self.off += n
        return b.decode("utf-8")


def write_schedule_stats_bin(stats: dict, path: str | Path) -> None:
    w = _BinWriter()
    w.buf.extend(MAGIC)
    w.write_u8(VERSION)
    # encode known numeric keys
    numeric_keys = [
        "total_cycles",
        "max_live",
        "num_spills",
        "gates",
        "inputs",
        "outputs",
    ]
    w.write_u32(len(numeric_keys))
    for key in numeric_keys:
        w.write_str(key)
        w.write_u64(int(stats.get(key, 0)))
    # strings
    w.write_str(str(stats.get("target", "")))
    w.write_str(str(stats.get("scheduler", "")))
    Path(path).write_bytes(w.buf)


def read_schedule_stats_bin(path: str | Path) -> dict:
    data = Path(path).read_bytes()
    if data[:4] != MAGIC:
        raise ValueError("bad schedule stats bin magic")
    r = _BinReader(data[4:])
    version = r.read_u8()
    if version != VERSION:
        raise ValueError(f"unsupported schedule stats bin version {version}")
    num = r.read_u32()
    stats: dict[str, int | str] = {}
    for _ in range(num):
        key = r.read_str()
        stats[key] = int(r.read_u64())
    stats["target"] = r.read_str()
    stats["scheduler"] = r.read_str()
    return stats
