from __future__ import annotations

from pathlib import Path

from stc.lowering_coordinator import LoweringChoice


MAGIC = b"LCB1"
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

    def read_u64(self) -> int:
        v = int.from_bytes(self.data[self.off : self.off + 8], "little", signed=False)
        self.off += 8
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


def write_lowering_choice_bin(choice: LoweringChoice, path: str | Path) -> None:
    w = _BinWriter()
    w.buf.extend(MAGIC)
    w.write_u8(VERSION)
    w.write_str(choice.path)
    w.write_str(choice.reason)
    w.write_u32(len(choice.unsupported))
    for s in choice.unsupported:
        w.write_str(s)
    w.write_u32(len(choice.stats))
    for k, v in choice.stats.items():
        w.write_str(k)
        w.write_u64(int(v))
    if choice.gate_comparison is None:
        w.write_u8(0)
    else:
        w.write_u8(1)
        w.write_u32(len(choice.gate_comparison))
        for k, v in choice.gate_comparison.items():
            w.write_str(k)
            w.write_f64(float(v))
    Path(path).write_bytes(w.buf)


def read_lowering_choice_bin(path: str | Path) -> LoweringChoice:
    data = Path(path).read_bytes()
    if data[:4] != MAGIC:
        raise ValueError("bad lowering choice bin magic")
    r = _BinReader(data[4:])
    version = r.read_u8()
    if version != VERSION:
        raise ValueError(f"unsupported lowering choice bin version {version}")
    path_str = r.read_str()
    reason = r.read_str()
    unsupported = [r.read_str() for _ in range(r.read_u32())]
    stats_count = r.read_u32()
    stats: dict[str, int] = {}
    for _ in range(stats_count):
        key = r.read_str()
        stats[key] = int(r.read_u64())
    gate_comparison = None
    if r.read_u8() == 1:
        comp_count = r.read_u32()
        gate_comparison = {}
        for _ in range(comp_count):
            key = r.read_str()
            gate_comparison[key] = float(r.read_f64())
    return LoweringChoice(
        path=path_str,
        reason=reason,
        unsupported=unsupported,
        stats=stats,
        gate_comparison=gate_comparison,
    )
