from __future__ import annotations

from pathlib import Path

from stc.region_diagnostics import RegionDiagnostics
from stc.kv_bin import write_kv_bin, read_kv_bin


MAGIC = b"RDB1"
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


def write_regions_bin(diagnostics: RegionDiagnostics, path: str | Path) -> None:
    w = _BinWriter()
    w.buf.extend(MAGIC)
    w.write_u8(VERSION)
    regions = diagnostics.regions_data.get("regions", [])
    w.write_u32(len(regions))
    for r in regions:
        w.write_u32(int(r.get("id", 0)))
        w.write_u32(int(r.get("gates", 0)))
        w.write_u32(int(r.get("outputs", 0)))
        w.write_u32(int(r.get("boundary_nodes", 0)))
        inputs = r.get("inputs", [])
        w.write_u32(len(inputs))
        for inp in inputs:
            w.write_str(str(inp))
    Path(path).write_bytes(w.buf)


def read_regions_bin(path: str | Path) -> dict:
    data = Path(path).read_bytes()
    if data[:4] != MAGIC:
        raise ValueError("bad regions bin magic")
    r = _BinReader(data[4:])
    version = r.read_u8()
    if version != VERSION:
        raise ValueError(f"unsupported regions bin version {version}")
    count = r.read_u32()
    regions = []
    for _ in range(count):
        rid = r.read_u32()
        gates = r.read_u32()
        outputs = r.read_u32()
        boundary_nodes = r.read_u32()
        inputs_count = r.read_u32()
        inputs = [r.read_str() for _ in range(inputs_count)]
        regions.append(
            {
                "id": int(rid),
                "gates": int(gates),
                "outputs": int(outputs),
                "boundary_nodes": int(boundary_nodes),
                "inputs": inputs,
            }
        )
    return {
        "regions": regions,
        "total_regions": len(regions),
        "ordering": "topological",
    }


def write_regions_stats_bin(diagnostics: RegionDiagnostics, path: str | Path) -> None:
    stats = diagnostics.regions_stats
    flat: dict[str, float | int] = {}
    for key in ("total_regions", "total_gates", "critical_path_estimate"):
        if key in stats:
            flat[key] = float(stats[key])
    gates = stats.get("gates_per_region", {})
    for key in ("min", "max", "mean", "median"):
        if key in gates:
            flat[f"gates_per_region.{key}"] = float(gates[key])
    boundaries = stats.get("boundary_distribution", {})
    for key in ("min", "max", "mean"):
        if key in boundaries:
            flat[f"boundary_distribution.{key}"] = float(boundaries[key])
    write_kv_bin(flat, path)


def read_regions_stats_bin(path: str | Path) -> dict:
    flat = read_kv_bin(path)
    stats: dict[str, object] = {}
    for key, value in flat.items():
        if key.startswith("gates_per_region."):
            stats.setdefault("gates_per_region", {})
            stats["gates_per_region"][key.split(".", 1)[1]] = value
        elif key.startswith("boundary_distribution."):
            stats.setdefault("boundary_distribution", {})
            stats["boundary_distribution"][key.split(".", 1)[1]] = value
        else:
            stats[key] = value
    return stats
