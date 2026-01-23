from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from stc.tick_ir import (
    And,
    BitTranspose,
    BitVecConst,
    BitVecType,
    Concat,
    Expr,
    Lut8,
    Not,
    Slice,
    Type,
    Var,
    Xor,
)
from stc.circuit_synth import synthesize_single_output


@dataclass
class BatchedLut8:
    source_var: str
    table: list[int]
    outputs: list[tuple[str, int]]


def find_batched_lut8(
    exprs: dict[str, Expr], types: dict[str, Type]
) -> list[BatchedLut8]:
    """
    Find groups of Lut8 expressions that can be batched together.

    A batch consists of Lut8 nodes that:
    1. Use the same table
    2. Have inputs that are 8-bit slices from the same source variable
    """
    candidates: dict[tuple[tuple[int, ...], str], list[tuple[str, int]]] = {}

    for name, expr in exprs.items():
        if not isinstance(expr, Lut8):
            continue
        if not isinstance(expr.x, Slice):
            continue
        if expr.x.width != 8:
            continue
        if not isinstance(expr.x.x, Var):
            continue

        source_var = expr.x.x.name
        offset = expr.x.offset
        table_key = tuple(expr.table)
        key = (table_key, source_var)

        if key not in candidates:
            candidates[key] = []
        candidates[key].append((name, offset))

    batches = []
    for (table_key, source_var), outputs in candidates.items():
        outputs_sorted = sorted(outputs, key=lambda x: x[1])
        batches.append(
            BatchedLut8(
                source_var=source_var,
                table=list(table_key),
                outputs=outputs_sorted,
            )
        )

    return batches


def bitslice_aes_sbox_ir(input_expr: Expr, num_bytes: int) -> Expr:
    """
    Generate Tick-IR for bitsliced AES S-box on num_bytes bytes.

    Input: expression representing num_bytes * 8 bits (e.g., 128 bits for 16 bytes)
    Output: expression representing the S-box applied to each byte
    """
    total_bits = num_bytes * 8
    plane_width = num_bytes

    transposed = BitTranspose(x=input_expr, lane_width=8, lanes=num_bytes)

    def plane(i: int) -> Expr:
        return Slice(x=transposed, offset=i * plane_width, width=plane_width)

    U = [plane(i) for i in range(8)]
    mask = BitVecConst(width=plane_width, value=(1 << plane_width) - 1)

    def xor2(a: Expr, b: Expr) -> Expr:
        return Xor(a=a, b=b)

    def and2(a: Expr, b: Expr) -> Expr:
        return And(a=a, b=b)

    def not1(a: Expr) -> Expr:
        return Xor(a=a, b=mask)

    T1 = xor2(U[7], U[4])
    T2 = xor2(U[7], U[2])
    T3 = xor2(U[7], U[1])
    T4 = xor2(U[4], U[2])
    T5 = xor2(U[3], U[1])
    T6 = xor2(T1, T5)
    T7 = xor2(U[6], U[5])
    T8 = xor2(U[0], T6)
    T9 = xor2(U[0], T7)
    T10 = xor2(T6, T7)
    T11 = xor2(U[6], U[2])
    T12 = xor2(U[5], U[2])
    T13 = xor2(T3, T4)
    T14 = xor2(T6, T11)
    T15 = xor2(T5, T11)
    T16 = xor2(T5, T12)
    T17 = xor2(T9, T16)
    T18 = xor2(U[4], U[0])
    T19 = xor2(T7, T18)
    T20 = xor2(T1, T19)
    T21 = xor2(U[1], U[0])
    T22 = xor2(T7, T21)
    T23 = xor2(T2, T22)
    T24 = xor2(T2, T10)
    T25 = xor2(T20, T17)
    T26 = xor2(T3, T16)
    T27 = xor2(T1, T12)

    M1 = and2(T13, T6)
    M2 = and2(T23, T8)
    M3 = xor2(T14, M1)
    M4 = and2(T19, U[0])
    M5 = xor2(M4, M1)
    M6 = and2(T3, T16)
    M7 = and2(T22, T9)
    M8 = xor2(T26, M6)
    M9 = and2(T20, T17)
    M10 = xor2(M9, M6)
    M11 = and2(T1, T15)
    M12 = and2(T4, T27)
    M13 = xor2(M12, M11)
    M14 = and2(T2, T10)
    M15 = xor2(M14, M11)
    M16 = xor2(M3, M2)
    M17 = xor2(M5, T24)
    M18 = xor2(M8, M7)
    M19 = xor2(M10, M15)
    M20 = xor2(M16, M13)
    M21 = xor2(M17, M15)
    M22 = xor2(M18, M13)
    M23 = xor2(M19, T25)
    M24 = xor2(M22, M23)
    M25 = and2(M22, M20)
    M26 = xor2(M21, M25)
    M27 = xor2(M20, M21)
    M28 = xor2(M23, M25)
    M29 = and2(M28, M27)
    M30 = and2(M26, M24)
    M31 = and2(M20, M23)
    M32 = and2(M27, M31)
    M33 = xor2(M27, M25)
    M34 = and2(M21, M22)
    M35 = and2(M24, M34)
    M36 = xor2(M24, M25)
    M37 = xor2(M21, M29)
    M38 = xor2(M32, M33)
    M39 = xor2(M23, M30)
    M40 = xor2(M35, M36)
    M41 = xor2(M38, M40)
    M42 = xor2(M37, M39)
    M43 = xor2(M37, M38)
    M44 = xor2(M39, M40)
    M45 = xor2(M42, M41)

    M46 = and2(M44, T6)
    M47 = and2(M40, T8)
    M48 = and2(M39, U[0])
    M49 = and2(M43, T16)
    M50 = and2(M38, T9)
    M51 = and2(M37, T17)
    M52 = and2(M42, T15)
    M53 = and2(M45, T27)
    M54 = and2(M41, T10)
    M55 = and2(M44, T13)
    M56 = and2(M40, T23)
    M57 = and2(M39, T19)
    M58 = and2(M43, T3)
    M59 = and2(M38, T22)
    M60 = and2(M37, T20)
    M61 = and2(M42, T1)
    M62 = and2(M45, T4)
    M63 = and2(M41, T2)

    L0 = xor2(M61, M62)
    L1 = xor2(M50, M56)
    L2 = xor2(M46, M48)
    L3 = xor2(M47, M55)
    L4 = xor2(M54, M58)
    L5 = xor2(M49, M61)
    L6 = xor2(M62, L5)
    L7 = xor2(M46, L3)
    L8 = xor2(M51, M59)
    L9 = xor2(M52, M53)
    L10 = xor2(M53, L4)
    L11 = xor2(M60, L2)
    L12 = xor2(M48, M51)
    L13 = xor2(M50, L0)
    L14 = xor2(M52, M61)
    L15 = xor2(M55, L1)
    L16 = xor2(M56, L0)
    L17 = xor2(M57, L1)
    L18 = xor2(M58, L8)
    L19 = xor2(M63, L4)
    L20 = xor2(L0, L1)
    L21 = xor2(L1, L7)
    L22 = xor2(L3, L12)
    L23 = xor2(L18, L2)
    L24 = xor2(L15, L9)
    L25 = xor2(L6, L10)
    L26 = xor2(L7, L9)
    L27 = xor2(L8, L10)
    L28 = xor2(L11, L14)
    L29 = xor2(L11, L17)

    S = [None] * 8
    S[7] = xor2(L6, L24)
    S[6] = not1(xor2(L16, L26))
    S[5] = not1(xor2(L19, L28))
    S[4] = xor2(L6, L21)
    S[3] = xor2(L20, L22)
    S[2] = xor2(L25, L29)
    S[1] = not1(xor2(L13, L27))
    S[0] = not1(xor2(L6, L23))

    output_transposed = Concat(parts=list(reversed(S)))
    output = BitTranspose(x=output_transposed, lane_width=num_bytes, lanes=8)

    return output


def synthesize_bitslice_circuit(
    table: Sequence[int],
    num_bytes: int,
    max_gates_per_bit: int = 20,
    timeout_ms_per_bit: int = 30000,
) -> Expr | None:
    """
    Synthesize a bitsliced circuit for any 8-bit to 8-bit lookup table.

    Uses Z3-based circuit synthesis to find a boolean circuit for each output bit,
    then assembles them into a bitsliced form that can process num_bytes in parallel.

    Args:
        table: 256-entry lookup table (8-bit input -> 8-bit output)
        num_bytes: Number of bytes to process in parallel
        max_gates_per_bit: Maximum gates to search for per output bit
        timeout_ms_per_bit: Timeout per output bit synthesis

    Returns:
        Tick-IR expression, or None if synthesis fails
    """
    assert len(table) == 256

    input_var = Var("x")
    transposed = BitTranspose(x=input_var, lane_width=8, lanes=num_bytes)
    plane_width = num_bytes

    output_bits: list[Expr] = []

    for out_bit in range(8):
        bit_table = [(table[i] >> out_bit) & 1 for i in range(256)]

        result = synthesize_single_output(
            bit_table,
            input_bits=8,
            max_gates=max_gates_per_bit,
            timeout_ms=timeout_ms_per_bit,
        )

        if result is None:
            return None

        single_bit_expr, _ = result

        parallel_expr = _widen_expr_for_bitslice(
            single_bit_expr, transposed, plane_width
        )
        output_bits.append(parallel_expr)

    output_transposed = Concat(parts=list(reversed(output_bits)))
    output = BitTranspose(x=output_transposed, lane_width=num_bytes, lanes=8)

    return output


def _widen_expr_for_bitslice(expr: Expr, transposed: Expr, plane_width: int) -> Expr:
    """
    Transform a single-bit circuit to operate on plane_width-bit parallel lanes.

    Replaces Slice(Var("x"), offset=i, width=1) with Slice(transposed, offset=i*plane_width, width=plane_width)
    and adjusts all operations to work on the wider type.
    """
    mask = BitVecConst(width=plane_width, value=(1 << plane_width) - 1)

    def transform(e: Expr) -> Expr:
        if (
            isinstance(e, Slice)
            and isinstance(e.x, Var)
            and e.x.name == "x"
            and e.width == 1
        ):
            return Slice(x=transposed, offset=e.offset * plane_width, width=plane_width)
        elif isinstance(e, Xor):
            return Xor(a=transform(e.a), b=transform(e.b))
        elif isinstance(e, And):
            return And(a=transform(e.a), b=transform(e.b))
        elif isinstance(e, Not):
            return Xor(a=transform(e.x), b=mask)
        elif isinstance(e, BitVecConst):
            if e.width == 1:
                return BitVecConst(
                    width=plane_width, value=(1 << plane_width) - 1 if e.value else 0
                )
            return e
        else:
            return e

    return transform(expr)


def transpose_to_bitplanes(bytes_in: Sequence[int], byte_width: int = 8) -> list[int]:
    """
    Transpose bytes to bit-planes.

    Input: N bytes, each byte_width bits wide
    Output: byte_width bit-planes, each N bits wide

    Byte j, bit i goes to plane i, position j.
    """
    n = len(bytes_in)
    planes = [0] * byte_width
    for j, byte_val in enumerate(bytes_in):
        for i in range(byte_width):
            if (byte_val >> i) & 1:
                planes[i] |= 1 << j
    return planes


def transpose_from_bitplanes(
    planes: Sequence[int], byte_width: int, count: int
) -> list[int]:
    """
    Transpose bit-planes back to bytes.

    Input: byte_width bit-planes, each at least count bits wide
    Output: count bytes, each byte_width bits wide

    Plane i, position j goes to byte j, bit i.
    """
    bytes_out = [0] * count
    for j in range(count):
        for i in range(byte_width):
            if (planes[i] >> j) & 1:
                bytes_out[j] |= 1 << i
    return bytes_out


AES_SBOX_TABLE: list[int] = [
    0x63,
    0x7C,
    0x77,
    0x7B,
    0xF2,
    0x6B,
    0x6F,
    0xC5,
    0x30,
    0x01,
    0x67,
    0x2B,
    0xFE,
    0xD7,
    0xAB,
    0x76,
    0xCA,
    0x82,
    0xC9,
    0x7D,
    0xFA,
    0x59,
    0x47,
    0xF0,
    0xAD,
    0xD4,
    0xA2,
    0xAF,
    0x9C,
    0xA4,
    0x72,
    0xC0,
    0xB7,
    0xFD,
    0x93,
    0x26,
    0x36,
    0x3F,
    0xF7,
    0xCC,
    0x34,
    0xA5,
    0xE5,
    0xF1,
    0x71,
    0xD8,
    0x31,
    0x15,
    0x04,
    0xC7,
    0x23,
    0xC3,
    0x18,
    0x96,
    0x05,
    0x9A,
    0x07,
    0x12,
    0x80,
    0xE2,
    0xEB,
    0x27,
    0xB2,
    0x75,
    0x09,
    0x83,
    0x2C,
    0x1A,
    0x1B,
    0x6E,
    0x5A,
    0xA0,
    0x52,
    0x3B,
    0xD6,
    0xB3,
    0x29,
    0xE3,
    0x2F,
    0x84,
    0x53,
    0xD1,
    0x00,
    0xED,
    0x20,
    0xFC,
    0xB1,
    0x5B,
    0x6A,
    0xCB,
    0xBE,
    0x39,
    0x4A,
    0x4C,
    0x58,
    0xCF,
    0xD0,
    0xEF,
    0xAA,
    0xFB,
    0x43,
    0x4D,
    0x33,
    0x85,
    0x45,
    0xF9,
    0x02,
    0x7F,
    0x50,
    0x3C,
    0x9F,
    0xA8,
    0x51,
    0xA3,
    0x40,
    0x8F,
    0x92,
    0x9D,
    0x38,
    0xF5,
    0xBC,
    0xB6,
    0xDA,
    0x21,
    0x10,
    0xFF,
    0xF3,
    0xD2,
    0xCD,
    0x0C,
    0x13,
    0xEC,
    0x5F,
    0x97,
    0x44,
    0x17,
    0xC4,
    0xA7,
    0x7E,
    0x3D,
    0x64,
    0x5D,
    0x19,
    0x73,
    0x60,
    0x81,
    0x4F,
    0xDC,
    0x22,
    0x2A,
    0x90,
    0x88,
    0x46,
    0xEE,
    0xB8,
    0x14,
    0xDE,
    0x5E,
    0x0B,
    0xDB,
    0xE0,
    0x32,
    0x3A,
    0x0A,
    0x49,
    0x06,
    0x24,
    0x5C,
    0xC2,
    0xD3,
    0xAC,
    0x62,
    0x91,
    0x95,
    0xE4,
    0x79,
    0xE7,
    0xC8,
    0x37,
    0x6D,
    0x8D,
    0xD5,
    0x4E,
    0xA9,
    0x6C,
    0x56,
    0xF4,
    0xEA,
    0x65,
    0x7A,
    0xAE,
    0x08,
    0xBA,
    0x78,
    0x25,
    0x2E,
    0x1C,
    0xA6,
    0xB4,
    0xC6,
    0xE8,
    0xDD,
    0x74,
    0x1F,
    0x4B,
    0xBD,
    0x8B,
    0x8A,
    0x70,
    0x3E,
    0xB5,
    0x66,
    0x48,
    0x03,
    0xF6,
    0x0E,
    0x61,
    0x35,
    0x57,
    0xB9,
    0x86,
    0xC1,
    0x1D,
    0x9E,
    0xE1,
    0xF8,
    0x98,
    0x11,
    0x69,
    0xD9,
    0x8E,
    0x94,
    0x9B,
    0x1E,
    0x87,
    0xE9,
    0xCE,
    0x55,
    0x28,
    0xDF,
    0x8C,
    0xA1,
    0x89,
    0x0D,
    0xBF,
    0xE6,
    0x42,
    0x68,
    0x41,
    0x99,
    0x2D,
    0x0F,
    0xB0,
    0x54,
    0xBB,
    0x16,
]


def aes_sbox_bitslice(U: Sequence[int], mask: int = 1) -> list[int]:
    """
    Boyar-Peralta AES S-box circuit.

    Input: U[0..7] where U[i] is bit i of the input byte(s).
           Each U[i] can be a multi-bit word for vectorized operation.
           mask: all-ones mask for the word width (e.g., 1 for single bit,
                 0xFFFFFFFFFFFFFFFF for 64-bit words)
    Output: S[0..7] where S[i] is bit i of the output byte(s).

    This circuit uses 115 XOR gates and 36 AND gates.
    Reference: http://cs-www.cs.yale.edu/homes/peralta/CircuitStuff/CMT.html
    """
    T1 = U[7] ^ U[4]
    T2 = U[7] ^ U[2]
    T3 = U[7] ^ U[1]
    T4 = U[4] ^ U[2]
    T5 = U[3] ^ U[1]
    T6 = T1 ^ T5
    T7 = U[6] ^ U[5]
    T8 = U[0] ^ T6
    T9 = U[0] ^ T7
    T10 = T6 ^ T7
    T11 = U[6] ^ U[2]
    T12 = U[5] ^ U[2]
    T13 = T3 ^ T4
    T14 = T6 ^ T11
    T15 = T5 ^ T11
    T16 = T5 ^ T12
    T17 = T9 ^ T16
    T18 = U[4] ^ U[0]
    T19 = T7 ^ T18
    T20 = T1 ^ T19
    T21 = U[1] ^ U[0]
    T22 = T7 ^ T21
    T23 = T2 ^ T22
    T24 = T2 ^ T10
    T25 = T20 ^ T17
    T26 = T3 ^ T16
    T27 = T1 ^ T12

    M1 = T13 & T6
    M2 = T23 & T8
    M3 = T14 ^ M1
    M4 = T19 & U[0]
    M5 = M4 ^ M1
    M6 = T3 & T16
    M7 = T22 & T9
    M8 = T26 ^ M6
    M9 = T20 & T17
    M10 = M9 ^ M6
    M11 = T1 & T15
    M12 = T4 & T27
    M13 = M12 ^ M11
    M14 = T2 & T10
    M15 = M14 ^ M11
    M16 = M3 ^ M2
    M17 = M5 ^ T24
    M18 = M8 ^ M7
    M19 = M10 ^ M15
    M20 = M16 ^ M13
    M21 = M17 ^ M15
    M22 = M18 ^ M13
    M23 = M19 ^ T25
    M24 = M22 ^ M23
    M25 = M22 & M20
    M26 = M21 ^ M25
    M27 = M20 ^ M21
    M28 = M23 ^ M25
    M29 = M28 & M27
    M30 = M26 & M24
    M31 = M20 & M23
    M32 = M27 & M31
    M33 = M27 ^ M25
    M34 = M21 & M22
    M35 = M24 & M34
    M36 = M24 ^ M25
    M37 = M21 ^ M29
    M38 = M32 ^ M33
    M39 = M23 ^ M30
    M40 = M35 ^ M36
    M41 = M38 ^ M40
    M42 = M37 ^ M39
    M43 = M37 ^ M38
    M44 = M39 ^ M40
    M45 = M42 ^ M41

    M46 = M44 & T6
    M47 = M40 & T8
    M48 = M39 & U[0]
    M49 = M43 & T16
    M50 = M38 & T9
    M51 = M37 & T17
    M52 = M42 & T15
    M53 = M45 & T27
    M54 = M41 & T10
    M55 = M44 & T13
    M56 = M40 & T23
    M57 = M39 & T19
    M58 = M43 & T3
    M59 = M38 & T22
    M60 = M37 & T20
    M61 = M42 & T1
    M62 = M45 & T4
    M63 = M41 & T2

    L0 = M61 ^ M62
    L1 = M50 ^ M56
    L2 = M46 ^ M48
    L3 = M47 ^ M55
    L4 = M54 ^ M58
    L5 = M49 ^ M61
    L6 = M62 ^ L5
    L7 = M46 ^ L3
    L8 = M51 ^ M59
    L9 = M52 ^ M53
    L10 = M53 ^ L4
    L11 = M60 ^ L2
    L12 = M48 ^ M51
    L13 = M50 ^ L0
    L14 = M52 ^ M61
    L15 = M55 ^ L1
    L16 = M56 ^ L0
    L17 = M57 ^ L1
    L18 = M58 ^ L8
    L19 = M63 ^ L4
    L20 = L0 ^ L1
    L21 = L1 ^ L7
    L22 = L3 ^ L12
    L23 = L18 ^ L2
    L24 = L15 ^ L9
    L25 = L6 ^ L10
    L26 = L7 ^ L9
    L27 = L8 ^ L10
    L28 = L11 ^ L14
    L29 = L11 ^ L17

    S = [0] * 8
    S[7] = L6 ^ L24
    S[6] = (L16 ^ L26) ^ mask
    S[5] = (L19 ^ L28) ^ mask
    S[4] = L6 ^ L21
    S[3] = L20 ^ L22
    S[2] = L25 ^ L29
    S[1] = (L13 ^ L27) ^ mask
    S[0] = (L6 ^ L23) ^ mask

    return S
