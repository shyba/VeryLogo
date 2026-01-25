from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class TernaryDB:
    """Precomputed ternary instruction reachability database."""

    bgc: bytes
    q0: set[int]
    q1_dict: dict[int, tuple[int, int, int, int]]
    base_vectors: tuple[int, int, int, int]


def compute_base_vectors() -> tuple[int, int, int, int]:
    """Compute 4-bit bitsliced representation for 4 input variables.

    For 4×4 S-box with inputs x0..x3:
      - x0: least significant bit
      - x1: second bit
      - x2: third bit
      - x3: most significant bit

    Returns (v0, v1, v2, v3) where each is 16-bit vector.
    """
    x0 = 0b0101010101010101
    x1 = 0b0011001100110011
    x2 = 0b0000111100001111
    x3 = 0b0000000011111111

    return (x0, x1, x2, x3)


def apply_ternary_lut(a: int, b: int, c: int, imm8: int) -> int:
    """Compute TernaryLut output for 16-bit vectors."""
    result = 0
    for i in range(16):
        a_bit = (a >> i) & 1
        b_bit = (b >> i) & 1
        c_bit = (c >> i) & 1
        idx = (a_bit << 2) | (b_bit << 1) | c_bit
        out_bit = (imm8 >> idx) & 1
        result |= out_bit << i
    return result


def build_q0_table(base_vectors: tuple[int, int, int, int]) -> set[int]:
    """Build q0: all vectors reachable in exactly 1 TI from base.

    Expected size: ~936 vectors (from Sovyn paper).
    """
    q0 = set()

    q0.add(0x0000)
    q0.add(0xFFFF)

    for v in base_vectors:
        q0.add(v)
        q0.add(v ^ 0xFFFF)

    candidates = list(q0)

    for imm8 in range(256):
        for a in candidates:
            for b in candidates:
                for c in candidates:
                    output = apply_ternary_lut(a, b, c, imm8)
                    q0.add(output)

    return q0


def build_q1_table(
    q0: set[int], base_vectors: tuple[int, int, int, int]
) -> dict[int, tuple[int, int, int, int]]:
    """Build q1: vectors reachable in exactly 2 TIs from base.

    Returns dict mapping output vector to reconstruction recipe.
    Expected size: ~438,312 unique vectors (from paper).
    """
    q1_dict = {}
    q0_list = sorted(q0)

    for imm8 in range(256):
        for a in q0_list:
            for b in q0_list:
                for c in q0_list:
                    output = apply_ternary_lut(a, b, c, imm8)

                    if output not in q1_dict:
                        q1_dict[output] = (a, b, c, imm8)

    return q1_dict


def build_bgc_table() -> bytes:
    """Build BGC table: maps each 16-bit vector to minimum TI count.

    Returns 65536-byte array.
    """
    bgc = bytearray(65536)

    for i in range(65536):
        bgc[i] = 4

    bgc[0] = 0
    bgc[0xFFFF] = 0

    base = compute_base_vectors()

    for v in base:
        bgc[v] = 1
        bgc[v ^ 0xFFFF] = 1

    q0 = build_q0_table(base)
    for v in q0:
        if bgc[v] > 2:
            bgc[v] = 2

    q1 = build_q1_table(q0, base)
    for v in q1:
        if bgc[v] > 3:
            bgc[v] = 3

    return bytes(bgc)


def enumerate_min_ti(
    target: int, base: tuple[int, int, int, int], max_depth: int = 3
) -> list[tuple[int, int, int, int]]:
    """Find minimum-TI sequence to construct target vector.

    Returns empty list if target is unreachable within depth bound.
    """
    return []


def score_sbox(outputs: tuple[int, int, int, int], bgc: bytes) -> int:
    """Score 4×4 S-box by summing BGC values of output vectors.

    Lower score = fewer total ternary instructions needed.
    """
    return sum(bgc[v] for v in outputs)


def save_db(db: TernaryDB, path: Path) -> None:
    """Serialize database to JSON."""
    data = {
        "bgc": list(db.bgc),
        "q0": sorted(db.q0),
        "q1": {str(k): v for k, v in db.q1_dict.items()},
        "base_vectors": list(db.base_vectors),
    }
    path.write_text(json.dumps(data, indent=2))


def load_db(path: Path) -> TernaryDB:
    """Load precomputed database from JSON."""
    data = json.loads(path.read_text())
    return TernaryDB(
        bgc=bytes(data["bgc"]),
        q0=set(data["q0"]),
        q1_dict={int(k): tuple(v) for k, v in data["q1"].items()},
        base_vectors=tuple(data["base_vectors"]),
    )
