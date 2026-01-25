from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KeccakError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


# Rotation offsets (rho) for Keccak-f[1600].
_R = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]

# Round constants.
_RC = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
]


def _rol64(x: int, n: int) -> int:
    n &= 63
    return ((x << n) | (x >> (64 - n))) & 0xFFFFFFFFFFFFFFFF


def keccak_f1600(state: list[int]) -> None:
    """In-place Keccak-f[1600] permutation.

    `state` is 25 lanes of 64-bit, indexed as state[x + 5*y].
    """
    if len(state) != 25:
        raise KeccakError("state must have 25 lanes")

    for rc in _RC:
        # theta
        c = [0] * 5
        for x in range(5):
            c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20]
        d = [0] * 5
        for x in range(5):
            d[x] = c[(x - 1) % 5] ^ _rol64(c[(x + 1) % 5], 1)
        for x in range(5):
            for y in range(5):
                state[x + 5 * y] ^= d[x]

        # rho + pi
        b = [0] * 25
        for x in range(5):
            for y in range(5):
                b[y + 5 * ((2 * x + 3 * y) % 5)] = _rol64(state[x + 5 * y], _R[x][y])

        # chi
        for x in range(5):
            for y in range(5):
                state[x + 5 * y] = b[x + 5 * y] ^ ((~b[((x + 1) % 5) + 5 * y]) & b[((x + 2) % 5) + 5 * y])
                state[x + 5 * y] &= 0xFFFFFFFFFFFFFFFF

        # iota
        state[0] ^= rc


def keccak_512(message: bytes, *, pad_byte: int = 0x01) -> bytes:
    """Keccak-512 sponge with rate=576 bits, capacity=1024 bits.

    `pad_byte=0x01` matches Keccak padding; SHA3-512 uses 0x06.
    """
    rate_bytes = 72  # 576 bits
    out_bytes = 64

    state = [0] * 25

    # absorb with pad10*1 using given domain byte
    data = bytearray(message)
    data.append(pad_byte & 0xFF)
    while (len(data) % rate_bytes) != rate_bytes - 1:
        data.append(0)
    data.append(0x80)

    for off in range(0, len(data), rate_bytes):
        block = data[off : off + rate_bytes]
        # XOR into state lanes, little-endian within lanes.
        for i in range(rate_bytes // 8):
            lane = int.from_bytes(block[i * 8 : (i + 1) * 8], byteorder="little")
            state[i] ^= lane
        keccak_f1600(state)

    # squeeze
    out = bytearray()
    while len(out) < out_bytes:
        for i in range(rate_bytes // 8):
            out.extend(int(state[i]).to_bytes(8, byteorder="little"))
            if len(out) >= out_bytes:
                break
        if len(out) >= out_bytes:
            break
        keccak_f1600(state)

    return bytes(out[:out_bytes])

