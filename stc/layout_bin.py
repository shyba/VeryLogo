from __future__ import annotations

from pathlib import Path

from stc.tick_ir_to_circuit_state import PackedLayout
from stc.tick_ir_to_packed_circuit_state import PackedWordLayout


MAGIC_BIT = b"PLB1"
MAGIC_WORD = b"PWB1"
VERSION_BIT = 1
VERSION_WORD = 2


class _BinWriter:
    def __init__(self) -> None:
        self.buf = bytearray()

    def write_u8(self, v: int) -> None:
        self.buf.append(v & 0xFF)

    def write_u32(self, v: int) -> None:
        self.buf.extend(int(v).to_bytes(4, "little", signed=False))

    def write_bytes(self, b: bytes) -> None:
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

    def read_bytes(self) -> bytes:
        n = self.read_u32()
        b = self.data[self.off : self.off + n]
        self.off += n
        return b


def _write_string_table(w: _BinWriter, names: list[str]) -> dict[str, int]:
    w.write_u32(len(names))
    index: dict[str, int] = {}
    for i, name in enumerate(names):
        index[name] = i
        w.write_bytes(name.encode("utf-8"))
    return index


def _read_string_table(r: _BinReader) -> list[str]:
    count = r.read_u32()
    names = []
    for _ in range(count):
        names.append(r.read_bytes().decode("utf-8"))
    return names


def write_packed_layout_bin(layout: PackedLayout, path: str | Path) -> None:
    w = _BinWriter()
    w.buf.extend(MAGIC_BIT)
    w.write_u8(VERSION_BIT)
    names = sorted(
        set(layout.inputs)
        | set(layout.state)
        | set(layout.outputs)
        | set(layout.next_state)
    )
    index = _write_string_table(w, names)
    w.write_u32(layout.input_bits)
    w.write_u32(layout.output_bits)
    for mapping in (layout.inputs, layout.state, layout.outputs, layout.next_state):
        w.write_u32(len(mapping))
        for name, info in mapping.items():
            w.write_u32(index[name])
            w.write_u32(int(info["lsb"]))
            w.write_u32(int(info["width"]))
    Path(path).write_bytes(w.buf)


def read_packed_layout_bin(path: str | Path) -> PackedLayout:
    data = Path(path).read_bytes()
    if data[:4] != MAGIC_BIT:
        raise ValueError("bad packed layout bin magic")
    r = _BinReader(data[4:])
    version = r.read_u8()
    if version != VERSION_BIT:
        raise ValueError(f"unsupported packed layout bin version {version}")
    names = _read_string_table(r)
    input_bits = r.read_u32()
    output_bits = r.read_u32()

    def read_map() -> dict[str, dict[str, int]]:
        count = r.read_u32()
        mapping: dict[str, dict[str, int]] = {}
        for _ in range(count):
            name_idx = r.read_u32()
            lsb = r.read_u32()
            width = r.read_u32()
            name = names[name_idx]
            mapping[name] = {"lsb": int(lsb), "width": int(width)}
        return mapping

    inputs = read_map()
    state = read_map()
    outputs = read_map()
    next_state = read_map()
    return PackedLayout(
        inputs=inputs,
        state=state,
        outputs=outputs,
        next_state=next_state,
        input_bits=int(input_bits),
        output_bits=int(output_bits),
    )


def write_packed_word_layout_bin(layout: PackedWordLayout, path: str | Path) -> None:
    w = _BinWriter()
    w.buf.extend(MAGIC_WORD)
    w.write_u8(VERSION_WORD)
    mode = getattr(layout, "mode", "packed")
    w.write_u8(1 if mode == "bitslice" else 0)
    names = sorted(
        set(layout.inputs)
        | set(layout.state)
        | set(layout.outputs)
        | set(layout.next_state)
    )
    index = _write_string_table(w, names)
    w.write_u32(layout.input_words)
    w.write_u32(layout.output_words)
    for mapping in (layout.inputs, layout.state, layout.outputs, layout.next_state):
        w.write_u32(len(mapping))
        for name, info in mapping.items():
            w.write_u32(index[name])
            w.write_u32(int(info["lsw"]))
            w.write_u32(int(info["width_bits"]))
    Path(path).write_bytes(w.buf)


def read_packed_word_layout_bin(path: str | Path) -> PackedWordLayout:
    data = Path(path).read_bytes()
    if data[:4] != MAGIC_WORD:
        raise ValueError("bad packed word layout bin magic")
    r = _BinReader(data[4:])
    version = r.read_u8()
    if version not in {1, VERSION_WORD}:
        raise ValueError(f"unsupported packed word layout bin version {version}")
    if version == 1:
        mode = "packed"
    else:
        mode = "bitslice" if r.read_u8() != 0 else "packed"
    names = _read_string_table(r)
    input_words = r.read_u32()
    output_words = r.read_u32()

    def read_map() -> dict[str, dict[str, int]]:
        count = r.read_u32()
        mapping: dict[str, dict[str, int]] = {}
        for _ in range(count):
            name_idx = r.read_u32()
            lsw = r.read_u32()
            width_bits = r.read_u32()
            name = names[name_idx]
            mapping[name] = {"lsw": int(lsw), "width_bits": int(width_bits)}
        return mapping

    inputs = read_map()
    state = read_map()
    outputs = read_map()
    next_state = read_map()
    return PackedWordLayout(
        inputs=inputs,
        state=state,
        outputs=outputs,
        next_state=next_state,
        input_words=int(input_words),
        output_words=int(output_words),
        mode=mode,
    )
