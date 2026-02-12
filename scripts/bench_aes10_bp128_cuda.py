#!/usr/bin/env python3
"""
Benchmark AES-128 (10 rounds) on CUDA using BP128 S-box logic.

Modes:
- legacy / legacy_tuned:
  Historical fixed-input benchmark path (fast baseline).
- replacement / replacement_tuned:
  Runtime IO path with input/output buffers + runtime round-key upload.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bench_lop3_cuda import _compute_cone_imm8
from scripts.benchmark_ternary_sbox import find_3input_cones
from scripts.bp_circuit_sbox import build_bp_sbox
from stc.circuit_synth import CircuitState


ROUND_KEYS_HEX = [
    "000102030405060708090a0b0c0d0e0f",
    "d6aa74fdd2af72fadaa678f1d6ab76fe",
    "b692cf0b643dbdf1be9bc5006830b3fe",
    "b6ff744ed2c2c9bf6c590cbf0469bf41",
    "47f7f7bc95353e03f96c32bcfd058dfd",
    "3caaa3e8a99f9deb50f3af57adf622aa",
    "5e390f7df7a69296a7553dc10aa31f6b",
    "14f9701ae35fe28c440adf4d4ea9c026",
    "47438735a41c65b9e016baf4aebf7ad2",
    "549932d1f08557681093ed9cbe2c974e",
    "13111d7fe3944a17f307a78b4d2b30c5",
]

ROUND_KEY_BYTES_PER_THREAD = 11 * 16
MASTER_KEY_BYTES_PER_THREAD = 16

AES_SBOX_BYTES = [
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


def _rk_bits_c_initializer() -> str:
    rows: list[str] = []
    for rk_hex in ROUND_KEYS_HEX:
        rk = bytes.fromhex(rk_hex)
        byte_rows: list[str] = []
        for byte in rk:
            bits = ", ".join(
                "0xffffffffu" if ((byte >> b) & 1) else "0u" for b in range(8)
            )
            byte_rows.append("{ " + bits + " }")
        rows.append("  { " + ", ".join(byte_rows) + " }")
    return ",\n".join(rows)


def _rk_bytes_c_initializer() -> str:
    rows: list[str] = []
    for rk_hex in ROUND_KEYS_HEX:
        rk = bytes.fromhex(rk_hex)
        elems = ", ".join(f"0x{b:02x}" for b in rk)
        rows.append("  { " + elems + " }")
    return ",\n".join(rows)


def _aes_sbox_c_initializer() -> str:
    rows: list[str] = []
    for i in range(0, len(AES_SBOX_BYTES), 16):
        chunk = ", ".join(f"0x{b:02x}" for b in AES_SBOX_BYTES[i : i + 16])
        rows.append("  " + chunk)
    return ",\n".join(rows)


def pack_rk_soa_packed_shared_key(rk_bytes_11x16: bytes, threads: int) -> bytes:
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if len(rk_bytes_11x16) != ROUND_KEY_BYTES_PER_THREAD:
        raise ValueError(
            f"rk_bytes_11x16 must be {ROUND_KEY_BYTES_PER_THREAD} bytes "
            f"(got {len(rk_bytes_11x16)})"
        )
    out = bytearray(ROUND_KEY_BYTES_PER_THREAD * threads)
    for key_idx, key_byte in enumerate(rk_bytes_11x16):
        base = key_idx * threads
        out[base : base + threads] = bytes([key_byte]) * threads
    return bytes(out)


def pack_rk_soa_packed_thread_keys(rk_bytes_per_thread: bytes, threads: int) -> bytes:
    if threads <= 0:
        raise ValueError("threads must be > 0")
    expected = ROUND_KEY_BYTES_PER_THREAD * threads
    if len(rk_bytes_per_thread) != expected:
        raise ValueError(
            f"rk_bytes_per_thread must be {expected} bytes "
            f"(got {len(rk_bytes_per_thread)})"
        )
    out = bytearray(expected)
    src = memoryview(rk_bytes_per_thread)
    for key_idx in range(ROUND_KEY_BYTES_PER_THREAD):
        dst_base = key_idx * threads
        src_off = key_idx
        for tid in range(threads):
            out[dst_base + tid] = src[tid * ROUND_KEY_BYTES_PER_THREAD + src_off]
    return bytes(out)


def pack_masterkey_soa_shared_key(master_key_16: bytes, threads: int) -> bytes:
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if len(master_key_16) != MASTER_KEY_BYTES_PER_THREAD:
        raise ValueError(
            f"master_key_16 must be {MASTER_KEY_BYTES_PER_THREAD} bytes "
            f"(got {len(master_key_16)})"
        )
    out = bytearray(MASTER_KEY_BYTES_PER_THREAD * threads)
    for key_idx, key_byte in enumerate(master_key_16):
        base = key_idx * threads
        out[base : base + threads] = bytes([key_byte]) * threads
    return bytes(out)


def pack_masterkey_soa_thread_keys(
    master_keys_per_thread: bytes, threads: int
) -> bytes:
    if threads <= 0:
        raise ValueError("threads must be > 0")
    expected = MASTER_KEY_BYTES_PER_THREAD * threads
    if len(master_keys_per_thread) != expected:
        raise ValueError(
            f"master_keys_per_thread must be {expected} bytes "
            f"(got {len(master_keys_per_thread)})"
        )
    out = bytearray(expected)
    src = memoryview(master_keys_per_thread)
    for key_idx in range(MASTER_KEY_BYTES_PER_THREAD):
        dst_base = key_idx * threads
        src_off = key_idx
        for tid in range(threads):
            out[dst_base + tid] = src[tid * MASTER_KEY_BYTES_PER_THREAD + src_off]
    return bytes(out)


def _build_bp128_mapped(
    max_ternary_cones: int | None = None,
) -> tuple[CircuitState, int]:
    bp = build_bp_sbox()
    cones = find_3input_cones(bp)
    if max_ternary_cones is not None:
        cones = cones[: max(0, max_ternary_cones)]

    new_gates = list(bp.gates)
    for cone in cones:
        root = cone["root"]
        leaves = cone["leaves"]
        imm8 = _compute_cone_imm8(bp, root, leaves)
        new_gates[root] = ("ternary", leaves[0], leaves[1], leaves[2], imm8)

    mapped = CircuitState(
        input_bits=bp.input_bits,
        output_bits=bp.output_bits,
        gates=new_gates,
        outputs=list(bp.outputs),
        gate_count=len(new_gates),
    ).eliminate_dead_code()
    lop3_count = sum(1 for g in mapped.gates if g[0] == "ternary")
    return mapped, lop3_count


def _emit_sbox_inline_cuda(
    mapped: CircuitState,
    *,
    func_name: str = "sbox_bp128_lop3_inline",
    noinline: bool = False,
) -> str:
    lines: list[str] = []
    qualifiers = "__device__"
    if noinline:
        qualifiers += " __noinline__"
    else:
        qualifiers += " __forceinline__"
    lines.append(
        f"{qualifiers} void {func_name}("
        "const uint32_t in_bits[8], uint32_t out_bits[8]) {"
    )
    for i in range(mapped.input_bits):
        lines.append(f"  uint32_t n{i} = in_bits[{i}];")
    for gate_idx, gate in enumerate(mapped.gates):
        dst = mapped.input_bits + gate_idx
        op = gate[0]
        if op == "xor":
            a, b = int(gate[1]), int(gate[2])
            lines.append(f"  uint32_t n{dst} = n{a} ^ n{b};")
        elif op == "and":
            a, b = int(gate[1]), int(gate[2])
            lines.append(f"  uint32_t n{dst} = n{a} & n{b};")
        elif op == "ternary":
            a, b, c, imm8 = int(gate[1]), int(gate[2]), int(gate[3]), int(gate[4])
            lines.append(f"  uint32_t n{dst};")
            lines.append(
                f'  asm("lop3.b32 %0, %1, %2, %3, {imm8};" : "=r"(n{dst}) : "r"(n{a}), "r"(n{b}), "r"(n{c}));'
            )
        else:
            raise RuntimeError(f"Unsupported BP128 op in CUDA emitter: {op}")

    for out_idx, (node, inv) in enumerate(mapped.outputs):
        expr = f"n{int(node)}"
        if inv:
            expr = f"~({expr})"
        lines.append(f"  out_bits[{out_idx}] = {expr};")
    lines.append("}")
    return "\n".join(lines)


def _kernel_cu_source_legacy(sbox_inline_cuda: str, *, input_mode: str) -> str:
    if input_mode not in {"const", "tid-xor"}:
        raise ValueError(f"unsupported input_mode: {input_mode}")
    input_expr = "pt[b]" if input_mode == "const" else "(uint8_t)(pt[b] ^ (uint8_t)tid)"
    rk_init = _rk_bits_c_initializer()

    return f"""
#include <stdint.h>

{sbox_inline_cuda}

__device__ __constant__ uint32_t RK_BITS[11][16][8] = {{
{rk_init}
}};

__device__ __forceinline__ void xor8(
    const uint32_t a[8], const uint32_t b[8], uint32_t o[8]
) {{
    #pragma unroll
    for (int i = 0; i < 8; i++) o[i] = a[i] ^ b[i];
}}

__device__ __forceinline__ void xtime8(const uint32_t a[8], uint32_t o[8]) {{
    const uint32_t b7 = a[7];
    o[7] = a[6];
    o[6] = a[5];
    o[5] = a[4];
    o[4] = a[3] ^ b7;
    o[3] = a[2] ^ b7;
    o[2] = a[1];
    o[1] = a[0] ^ b7;
    o[0] = b7;
}}

__device__ __forceinline__ void mix_shifted_col_addkey(
    const uint32_t s0[8], const uint32_t s1[8], const uint32_t s2[8], const uint32_t s3[8],
    const uint32_t rk0[8], const uint32_t rk1[8], const uint32_t rk2[8], const uint32_t rk3[8],
    uint32_t d0[8], uint32_t d1[8], uint32_t d2[8], uint32_t d3[8]
) {{
    uint32_t t[8], x01[8], x12[8], x23[8], x30[8], xt[8];

    #pragma unroll
    for (int bit = 0; bit < 8; bit++) {{
        t[bit] = s0[bit] ^ s1[bit] ^ s2[bit] ^ s3[bit];
        x01[bit] = s0[bit] ^ s1[bit];
        x12[bit] = s1[bit] ^ s2[bit];
        x23[bit] = s2[bit] ^ s3[bit];
        x30[bit] = s3[bit] ^ s0[bit];
    }}

    xtime8(x01, xt);
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) d0[bit] = s0[bit] ^ t[bit] ^ xt[bit] ^ rk0[bit];

    xtime8(x12, xt);
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) d1[bit] = s1[bit] ^ t[bit] ^ xt[bit] ^ rk1[bit];

    xtime8(x23, xt);
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) d2[bit] = s2[bit] ^ t[bit] ^ xt[bit] ^ rk2[bit];

    xtime8(x30, xt);
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) d3[bit] = s3[bit] ^ t[bit] ^ xt[bit] ^ rk3[bit];
}}

extern "C" __global__ void aes10_bp128_kernel(
    uint32_t* out_ptr,
    uint32_t n_threads
) {{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;

    uint32_t st[16][8];
    uint32_t sb[16][8];
    uint32_t sr[16][8];
    uint32_t mc[16][8];
    const uint8_t pt[16] = {{
        0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,
        0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff
    }};

    #pragma unroll
    for (int b = 0; b < 16; b++) {{
        uint8_t pb = {input_expr};
        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {{
            st[b][bit] = ((pb >> bit) & 1) ? 0xffffffffu : 0u;
        }}
    }}

    #pragma unroll
    for (int b = 0; b < 16; b++) {{
        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {{
            st[b][bit] ^= RK_BITS[0][b][bit];
        }}
    }}

    for (int round = 1; round <= 9; round++) {{
        #pragma unroll
        for (int b = 0; b < 16; b++) {{
            sbox_bp128_lop3_inline(st[b], sb[b]);
        }}

        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {{
            sr[0][bit]  = sb[0][bit];
            sr[1][bit]  = sb[5][bit];
            sr[2][bit]  = sb[10][bit];
            sr[3][bit]  = sb[15][bit];
            sr[4][bit]  = sb[4][bit];
            sr[5][bit]  = sb[9][bit];
            sr[6][bit]  = sb[14][bit];
            sr[7][bit]  = sb[3][bit];
            sr[8][bit]  = sb[8][bit];
            sr[9][bit]  = sb[13][bit];
            sr[10][bit] = sb[2][bit];
            sr[11][bit] = sb[7][bit];
            sr[12][bit] = sb[12][bit];
            sr[13][bit] = sb[1][bit];
            sr[14][bit] = sb[6][bit];
            sr[15][bit] = sb[11][bit];
        }}

        #pragma unroll
        for (int c = 0; c < 4; c++) {{
            int i0 = c * 4 + 0;
            int i1 = c * 4 + 1;
            int i2 = c * 4 + 2;
            int i3 = c * 4 + 3;
            uint32_t t[8], u[8], x01[8], x12[8], x23[8], x30[8], xt[8];

            #pragma unroll
            for (int i = 0; i < 8; i++) {{
                t[i] = sr[i0][i] ^ sr[i1][i] ^ sr[i2][i] ^ sr[i3][i];
                u[i] = sr[i0][i];
            }}
            xor8(sr[i0], sr[i1], x01);
            xor8(sr[i1], sr[i2], x12);
            xor8(sr[i2], sr[i3], x23);
            xor8(sr[i3], u, x30);

            xtime8(x01, xt);
            #pragma unroll
            for (int i = 0; i < 8; i++) mc[i0][i] = sr[i0][i] ^ t[i] ^ xt[i];

            xtime8(x12, xt);
            #pragma unroll
            for (int i = 0; i < 8; i++) mc[i1][i] = sr[i1][i] ^ t[i] ^ xt[i];

            xtime8(x23, xt);
            #pragma unroll
            for (int i = 0; i < 8; i++) mc[i2][i] = sr[i2][i] ^ t[i] ^ xt[i];

            xtime8(x30, xt);
            #pragma unroll
            for (int i = 0; i < 8; i++) mc[i3][i] = sr[i3][i] ^ t[i] ^ xt[i];
        }}

        #pragma unroll
        for (int b = 0; b < 16; b++) {{
            #pragma unroll
            for (int bit = 0; bit < 8; bit++) {{
                st[b][bit] = mc[b][bit] ^ RK_BITS[round][b][bit];
            }}
        }}
    }}

    #pragma unroll
    for (int b = 0; b < 16; b++) {{
        sbox_bp128_lop3_inline(st[b], sb[b]);
    }}
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) {{
        sr[0][bit]  = sb[0][bit];
        sr[1][bit]  = sb[5][bit];
        sr[2][bit]  = sb[10][bit];
        sr[3][bit]  = sb[15][bit];
        sr[4][bit]  = sb[4][bit];
        sr[5][bit]  = sb[9][bit];
        sr[6][bit]  = sb[14][bit];
        sr[7][bit]  = sb[3][bit];
        sr[8][bit]  = sb[8][bit];
        sr[9][bit]  = sb[13][bit];
        sr[10][bit] = sb[2][bit];
        sr[11][bit] = sb[7][bit];
        sr[12][bit] = sb[12][bit];
        sr[13][bit] = sb[1][bit];
        sr[14][bit] = sb[6][bit];
        sr[15][bit] = sb[11][bit];
    }}
    #pragma unroll
    for (int b = 0; b < 16; b++) {{
        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {{
            st[b][bit] = sr[b][bit] ^ RK_BITS[10][b][bit];
        }}
    }}

    size_t base = (size_t)tid * 128;
    #pragma unroll
    for (int b = 0; b < 16; b++) {{
        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {{
            out_ptr[base + b * 8 + bit] = st[b][bit];
        }}
    }}
}}
"""


def _kernel_cu_source_legacy_streamed_inplace(
    sbox_inline_cuda: str, *, input_mode: str
) -> str:
    if input_mode not in {"const", "tid-xor"}:
        raise ValueError(f"unsupported input_mode: {input_mode}")
    input_expr = "pt[b]" if input_mode == "const" else "(uint8_t)(pt[b] ^ (uint8_t)tid)"
    rk_init = _rk_bits_c_initializer()

    return f"""
#include <stdint.h>

{sbox_inline_cuda}

__device__ __constant__ uint32_t RK_BITS[11][16][8] = {{
{rk_init}
}};

__device__ __forceinline__ void xtime8(const uint32_t a[8], uint32_t o[8]) {{
    const uint32_t b7 = a[7];
    o[7] = a[6];
    o[6] = a[5];
    o[5] = a[4];
    o[4] = a[3] ^ b7;
    o[3] = a[2] ^ b7;
    o[2] = a[1];
    o[1] = a[0] ^ b7;
    o[0] = b7;
}}

__device__ __forceinline__ void mix_shifted_col_addkey(
    const uint32_t s0[8], const uint32_t s1[8], const uint32_t s2[8], const uint32_t s3[8],
    const uint32_t rk0[8], const uint32_t rk1[8], const uint32_t rk2[8], const uint32_t rk3[8],
    uint32_t d0[8], uint32_t d1[8], uint32_t d2[8], uint32_t d3[8]
) {{
    uint32_t t[8], x01[8], x12[8], x23[8], x30[8], xt[8];

    #pragma unroll
    for (int bit = 0; bit < 8; bit++) {{
        t[bit] = s0[bit] ^ s1[bit] ^ s2[bit] ^ s3[bit];
        x01[bit] = s0[bit] ^ s1[bit];
        x12[bit] = s1[bit] ^ s2[bit];
        x23[bit] = s2[bit] ^ s3[bit];
        x30[bit] = s3[bit] ^ s0[bit];
    }}

    xtime8(x01, xt);
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) d0[bit] = s0[bit] ^ t[bit] ^ xt[bit] ^ rk0[bit];

    xtime8(x12, xt);
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) d1[bit] = s1[bit] ^ t[bit] ^ xt[bit] ^ rk1[bit];

    xtime8(x23, xt);
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) d2[bit] = s2[bit] ^ t[bit] ^ xt[bit] ^ rk2[bit];

    xtime8(x30, xt);
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) d3[bit] = s3[bit] ^ t[bit] ^ xt[bit] ^ rk3[bit];
}}

__device__ __forceinline__ void subbyte_addkey_store(
    const uint32_t inb[8], const uint32_t rk[8], uint32_t* outw
) {{
    uint32_t tmp[8];
    sbox_bp128_lop3_inline(inb, tmp);
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) outw[bit] = tmp[bit] ^ rk[bit];
}}

extern "C" __global__ void aes10_bp128_kernel(
    uint32_t* out_ptr,
    uint32_t n_threads
) {{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;

    uint32_t st[16][8];
    const uint8_t pt[16] = {{
        0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,
        0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff
    }};

    #pragma unroll
    for (int b = 0; b < 16; b++) {{
        uint8_t pb = {input_expr};
        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {{
            st[b][bit] = (((uint32_t)pb >> bit) & 1u) ? 0xffffffffu : 0u;
            st[b][bit] ^= RK_BITS[0][b][bit];
        }}
    }}

    for (int round = 1; round <= 9; round++) {{
        uint32_t sv1[8], sv2[8], svx[8], svy[8];
        uint32_t s0[8], s1[8], s2[8], s3[8];

        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {{
            sv1[bit] = st[1][bit];
            sv2[bit] = st[2][bit];
            svx[bit] = st[3][bit];
        }}

        sbox_bp128_lop3_inline(st[0], s0);
        sbox_bp128_lop3_inline(st[5], s1);
        sbox_bp128_lop3_inline(st[10], s2);
        sbox_bp128_lop3_inline(st[15], s3);
        mix_shifted_col_addkey(
            s0, s1, s2, s3,
            RK_BITS[round][0], RK_BITS[round][1], RK_BITS[round][2], RK_BITS[round][3],
            st[0], st[1], st[2], st[3]
        );

        sbox_bp128_lop3_inline(st[4], s0);
        sbox_bp128_lop3_inline(st[9], s1);
        sbox_bp128_lop3_inline(st[14], s2);
        sbox_bp128_lop3_inline(svx, s3);
        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {{
            svx[bit] = st[6][bit];
            svy[bit] = st[7][bit];
        }}
        mix_shifted_col_addkey(
            s0, s1, s2, s3,
            RK_BITS[round][4], RK_BITS[round][5], RK_BITS[round][6], RK_BITS[round][7],
            st[4], st[5], st[6], st[7]
        );

        sbox_bp128_lop3_inline(st[8], s0);
        sbox_bp128_lop3_inline(st[13], s1);
        sbox_bp128_lop3_inline(sv2, s2);
        sbox_bp128_lop3_inline(svy, s3);
        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {{
            svy[bit] = st[11][bit];
        }}
        mix_shifted_col_addkey(
            s0, s1, s2, s3,
            RK_BITS[round][8], RK_BITS[round][9], RK_BITS[round][10], RK_BITS[round][11],
            st[8], st[9], st[10], st[11]
        );

        sbox_bp128_lop3_inline(st[12], s0);
        sbox_bp128_lop3_inline(sv1, s1);
        sbox_bp128_lop3_inline(svx, s2);
        sbox_bp128_lop3_inline(svy, s3);
        mix_shifted_col_addkey(
            s0, s1, s2, s3,
            RK_BITS[round][12], RK_BITS[round][13], RK_BITS[round][14], RK_BITS[round][15],
            st[12], st[13], st[14], st[15]
        );
    }}

    size_t base = (size_t)tid * 128u;
    subbyte_addkey_store(st[0], RK_BITS[10][0], out_ptr + base + 0u * 8u);
    subbyte_addkey_store(st[5], RK_BITS[10][1], out_ptr + base + 1u * 8u);
    subbyte_addkey_store(st[10], RK_BITS[10][2], out_ptr + base + 2u * 8u);
    subbyte_addkey_store(st[15], RK_BITS[10][3], out_ptr + base + 3u * 8u);
    subbyte_addkey_store(st[4], RK_BITS[10][4], out_ptr + base + 4u * 8u);
    subbyte_addkey_store(st[9], RK_BITS[10][5], out_ptr + base + 5u * 8u);
    subbyte_addkey_store(st[14], RK_BITS[10][6], out_ptr + base + 6u * 8u);
    subbyte_addkey_store(st[3], RK_BITS[10][7], out_ptr + base + 7u * 8u);
    subbyte_addkey_store(st[8], RK_BITS[10][8], out_ptr + base + 8u * 8u);
    subbyte_addkey_store(st[13], RK_BITS[10][9], out_ptr + base + 9u * 8u);
    subbyte_addkey_store(st[2], RK_BITS[10][10], out_ptr + base + 10u * 8u);
    subbyte_addkey_store(st[7], RK_BITS[10][11], out_ptr + base + 11u * 8u);
    subbyte_addkey_store(st[12], RK_BITS[10][12], out_ptr + base + 12u * 8u);
    subbyte_addkey_store(st[1], RK_BITS[10][13], out_ptr + base + 13u * 8u);
    subbyte_addkey_store(st[6], RK_BITS[10][14], out_ptr + base + 14u * 8u);
    subbyte_addkey_store(st[11], RK_BITS[10][15], out_ptr + base + 15u * 8u);
}}
"""


def _kernel_cu_source_replacement(sbox_inline_cuda: str) -> str:
    return f"""
#include <stdint.h>

{sbox_inline_cuda}

__device__ __constant__ uint32_t RK_BITS[11][16][8];

__device__ __forceinline__ void xor8(
    const uint32_t a[8], const uint32_t b[8], uint32_t o[8]
) {{
    #pragma unroll
    for (int i = 0; i < 8; i++) o[i] = a[i] ^ b[i];
}}

__device__ __forceinline__ void xtime8(const uint32_t a[8], uint32_t o[8]) {{
    const uint32_t b7 = a[7];
    o[7] = a[6];
    o[6] = a[5];
    o[5] = a[4];
    o[4] = a[3] ^ b7;
    o[3] = a[2] ^ b7;
    o[2] = a[1];
    o[1] = a[0] ^ b7;
    o[0] = b7;
}}

extern "C" __global__ void aes10_bp128_kernel(
    const uint32_t* __restrict__ in_ptr,
    uint32_t* __restrict__ out_ptr,
    uint32_t n_threads
) {{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;

    uint32_t st[16][8];
    uint32_t sb[16][8];
    uint32_t sr[16][8];
    uint32_t mc[16][8];
    size_t base = (size_t)tid * 128;

    #pragma unroll
    for (int b = 0; b < 16; b++) {{
        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {{
            st[b][bit] = in_ptr[base + b * 8 + bit] ^ RK_BITS[0][b][bit];
        }}
    }}

    for (int round = 1; round <= 9; round++) {{
        #pragma unroll
        for (int b = 0; b < 16; b++) {{
            sbox_bp128_lop3_inline(st[b], sb[b]);
        }}

        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {{
            sr[0][bit]  = sb[0][bit];
            sr[1][bit]  = sb[5][bit];
            sr[2][bit]  = sb[10][bit];
            sr[3][bit]  = sb[15][bit];
            sr[4][bit]  = sb[4][bit];
            sr[5][bit]  = sb[9][bit];
            sr[6][bit]  = sb[14][bit];
            sr[7][bit]  = sb[3][bit];
            sr[8][bit]  = sb[8][bit];
            sr[9][bit]  = sb[13][bit];
            sr[10][bit] = sb[2][bit];
            sr[11][bit] = sb[7][bit];
            sr[12][bit] = sb[12][bit];
            sr[13][bit] = sb[1][bit];
            sr[14][bit] = sb[6][bit];
            sr[15][bit] = sb[11][bit];
        }}

        #pragma unroll
        for (int c = 0; c < 4; c++) {{
            int i0 = c * 4 + 0;
            int i1 = c * 4 + 1;
            int i2 = c * 4 + 2;
            int i3 = c * 4 + 3;
            uint32_t t[8], u[8], x01[8], x12[8], x23[8], x30[8], xt[8];

            #pragma unroll
            for (int i = 0; i < 8; i++) {{
                t[i] = sr[i0][i] ^ sr[i1][i] ^ sr[i2][i] ^ sr[i3][i];
                u[i] = sr[i0][i];
            }}
            xor8(sr[i0], sr[i1], x01);
            xor8(sr[i1], sr[i2], x12);
            xor8(sr[i2], sr[i3], x23);
            xor8(sr[i3], u, x30);

            xtime8(x01, xt);
            #pragma unroll
            for (int i = 0; i < 8; i++) mc[i0][i] = sr[i0][i] ^ t[i] ^ xt[i];

            xtime8(x12, xt);
            #pragma unroll
            for (int i = 0; i < 8; i++) mc[i1][i] = sr[i1][i] ^ t[i] ^ xt[i];

            xtime8(x23, xt);
            #pragma unroll
            for (int i = 0; i < 8; i++) mc[i2][i] = sr[i2][i] ^ t[i] ^ xt[i];

            xtime8(x30, xt);
            #pragma unroll
            for (int i = 0; i < 8; i++) mc[i3][i] = sr[i3][i] ^ t[i] ^ xt[i];
        }}

        #pragma unroll
        for (int b = 0; b < 16; b++) {{
            #pragma unroll
            for (int bit = 0; bit < 8; bit++) {{
                st[b][bit] = mc[b][bit] ^ RK_BITS[round][b][bit];
            }}
        }}
    }}

    #pragma unroll
    for (int b = 0; b < 16; b++) {{
        sbox_bp128_lop3_inline(st[b], sb[b]);
    }}
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) {{
        sr[0][bit]  = sb[0][bit];
        sr[1][bit]  = sb[5][bit];
        sr[2][bit]  = sb[10][bit];
        sr[3][bit]  = sb[15][bit];
        sr[4][bit]  = sb[4][bit];
        sr[5][bit]  = sb[9][bit];
        sr[6][bit]  = sb[14][bit];
        sr[7][bit]  = sb[3][bit];
        sr[8][bit]  = sb[8][bit];
        sr[9][bit]  = sb[13][bit];
        sr[10][bit] = sb[2][bit];
        sr[11][bit] = sb[7][bit];
        sr[12][bit] = sb[12][bit];
        sr[13][bit] = sb[1][bit];
        sr[14][bit] = sb[6][bit];
        sr[15][bit] = sb[11][bit];
    }}
    #pragma unroll
    for (int b = 0; b < 16; b++) {{
        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {{
            out_ptr[base + b * 8 + bit] = sr[b][bit] ^ RK_BITS[10][b][bit];
        }}
    }}
}}
"""


def _kernel_cu_source_replacement_streamed_inplace(sbox_inline_cuda: str) -> str:
    return f"""
#include <stdint.h>

{sbox_inline_cuda}

__device__ __constant__ uint32_t RK_BITS[11][16][8];

__device__ __forceinline__ void xtime8(const uint32_t a[8], uint32_t o[8]) {{
    const uint32_t b7 = a[7];
    o[7] = a[6];
    o[6] = a[5];
    o[5] = a[4];
    o[4] = a[3] ^ b7;
    o[3] = a[2] ^ b7;
    o[2] = a[1];
    o[1] = a[0] ^ b7;
    o[0] = b7;
}}

__device__ __forceinline__ void mix_shifted_col_addkey(
    const uint32_t s0[8], const uint32_t s1[8], const uint32_t s2[8], const uint32_t s3[8],
    const uint32_t rk0[8], const uint32_t rk1[8], const uint32_t rk2[8], const uint32_t rk3[8],
    uint32_t d0[8], uint32_t d1[8], uint32_t d2[8], uint32_t d3[8]
) {{
    uint32_t t[8], x01[8], x12[8], x23[8], x30[8], xt[8];

    #pragma unroll
    for (int bit = 0; bit < 8; bit++) {{
        t[bit] = s0[bit] ^ s1[bit] ^ s2[bit] ^ s3[bit];
        x01[bit] = s0[bit] ^ s1[bit];
        x12[bit] = s1[bit] ^ s2[bit];
        x23[bit] = s2[bit] ^ s3[bit];
        x30[bit] = s3[bit] ^ s0[bit];
    }}

    xtime8(x01, xt);
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) d0[bit] = s0[bit] ^ t[bit] ^ xt[bit] ^ rk0[bit];

    xtime8(x12, xt);
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) d1[bit] = s1[bit] ^ t[bit] ^ xt[bit] ^ rk1[bit];

    xtime8(x23, xt);
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) d2[bit] = s2[bit] ^ t[bit] ^ xt[bit] ^ rk2[bit];

    xtime8(x30, xt);
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) d3[bit] = s3[bit] ^ t[bit] ^ xt[bit] ^ rk3[bit];
}}

__device__ __forceinline__ void subbyte_addkey_store(
    const uint32_t inb[8], const uint32_t rk[8], uint32_t* outw
) {{
    uint32_t tmp[8];
    sbox_bp128_lop3_inline(inb, tmp);
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) outw[bit] = tmp[bit] ^ rk[bit];
}}

extern "C" __global__ void aes10_bp128_kernel(
    const uint32_t* __restrict__ in_ptr,
    uint32_t* __restrict__ out_ptr,
    uint32_t n_threads
) {{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;

    uint32_t st[16][8];
    size_t base = (size_t)tid * 128u;

    #pragma unroll
    for (int b = 0; b < 16; b++) {{
        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {{
            st[b][bit] = in_ptr[base + (size_t)b * 8u + (size_t)bit] ^ RK_BITS[0][b][bit];
        }}
    }}

    for (int round = 1; round <= 9; round++) {{
        uint32_t sv1[8], sv2[8], svx[8], svy[8];
        uint32_t s0[8], s1[8], s2[8], s3[8];

        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {{
            sv1[bit] = st[1][bit];
            sv2[bit] = st[2][bit];
            svx[bit] = st[3][bit];
        }}

        sbox_bp128_lop3_inline(st[0], s0);
        sbox_bp128_lop3_inline(st[5], s1);
        sbox_bp128_lop3_inline(st[10], s2);
        sbox_bp128_lop3_inline(st[15], s3);
        mix_shifted_col_addkey(
            s0, s1, s2, s3,
            RK_BITS[round][0], RK_BITS[round][1], RK_BITS[round][2], RK_BITS[round][3],
            st[0], st[1], st[2], st[3]
        );

        sbox_bp128_lop3_inline(st[4], s0);
        sbox_bp128_lop3_inline(st[9], s1);
        sbox_bp128_lop3_inline(st[14], s2);
        sbox_bp128_lop3_inline(svx, s3);
        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {{
            svx[bit] = st[6][bit];
            svy[bit] = st[7][bit];
        }}
        mix_shifted_col_addkey(
            s0, s1, s2, s3,
            RK_BITS[round][4], RK_BITS[round][5], RK_BITS[round][6], RK_BITS[round][7],
            st[4], st[5], st[6], st[7]
        );

        sbox_bp128_lop3_inline(st[8], s0);
        sbox_bp128_lop3_inline(st[13], s1);
        sbox_bp128_lop3_inline(sv2, s2);
        sbox_bp128_lop3_inline(svy, s3);
        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {{
            svy[bit] = st[11][bit];
        }}
        mix_shifted_col_addkey(
            s0, s1, s2, s3,
            RK_BITS[round][8], RK_BITS[round][9], RK_BITS[round][10], RK_BITS[round][11],
            st[8], st[9], st[10], st[11]
        );

        sbox_bp128_lop3_inline(st[12], s0);
        sbox_bp128_lop3_inline(sv1, s1);
        sbox_bp128_lop3_inline(svx, s2);
        sbox_bp128_lop3_inline(svy, s3);
        mix_shifted_col_addkey(
            s0, s1, s2, s3,
            RK_BITS[round][12], RK_BITS[round][13], RK_BITS[round][14], RK_BITS[round][15],
            st[12], st[13], st[14], st[15]
        );
    }}

    subbyte_addkey_store(st[0], RK_BITS[10][0], out_ptr + base + 0u * 8u);
    subbyte_addkey_store(st[5], RK_BITS[10][1], out_ptr + base + 1u * 8u);
    subbyte_addkey_store(st[10], RK_BITS[10][2], out_ptr + base + 2u * 8u);
    subbyte_addkey_store(st[15], RK_BITS[10][3], out_ptr + base + 3u * 8u);
    subbyte_addkey_store(st[4], RK_BITS[10][4], out_ptr + base + 4u * 8u);
    subbyte_addkey_store(st[9], RK_BITS[10][5], out_ptr + base + 5u * 8u);
    subbyte_addkey_store(st[14], RK_BITS[10][6], out_ptr + base + 6u * 8u);
    subbyte_addkey_store(st[3], RK_BITS[10][7], out_ptr + base + 7u * 8u);
    subbyte_addkey_store(st[8], RK_BITS[10][8], out_ptr + base + 8u * 8u);
    subbyte_addkey_store(st[13], RK_BITS[10][9], out_ptr + base + 9u * 8u);
    subbyte_addkey_store(st[2], RK_BITS[10][10], out_ptr + base + 10u * 8u);
    subbyte_addkey_store(st[7], RK_BITS[10][11], out_ptr + base + 11u * 8u);
    subbyte_addkey_store(st[12], RK_BITS[10][12], out_ptr + base + 12u * 8u);
    subbyte_addkey_store(st[1], RK_BITS[10][13], out_ptr + base + 13u * 8u);
    subbyte_addkey_store(st[6], RK_BITS[10][14], out_ptr + base + 14u * 8u);
    subbyte_addkey_store(st[11], RK_BITS[10][15], out_ptr + base + 15u * 8u);
}}
"""


def _kernel_cu_source_replacement_coalesced(sbox_inline_cuda: str) -> str:
    return f"""
#include <stdint.h>

{sbox_inline_cuda}

__device__ __constant__ uint32_t RK_BITS[11][16][8];

__device__ __forceinline__ void xor8(
    const uint32_t a[8], const uint32_t b[8], uint32_t o[8]
) {{
    #pragma unroll
    for (int i = 0; i < 8; i++) o[i] = a[i] ^ b[i];
}}

__device__ __forceinline__ void xtime8(const uint32_t a[8], uint32_t o[8]) {{
    const uint32_t b7 = a[7];
    o[7] = a[6];
    o[6] = a[5];
    o[5] = a[4];
    o[4] = a[3] ^ b7;
    o[3] = a[2] ^ b7;
    o[2] = a[1];
    o[1] = a[0] ^ b7;
    o[0] = b7;
}}

extern "C" __global__ void aes10_bp128_kernel(
    const uint32_t* __restrict__ in_ptr,
    uint32_t* __restrict__ out_ptr,
    uint32_t n_threads
) {{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;

    uint32_t st[16][8];
    uint32_t sb[16][8];
    uint32_t sr[16][8];
    uint32_t mc[16][8];
    const size_t threads = (size_t)n_threads;
    const size_t t = (size_t)tid;

    #pragma unroll
    for (int b = 0; b < 16; b++) {{
        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {{
            size_t plane = (size_t)b * 8u + (size_t)bit;
            st[b][bit] = in_ptr[plane * threads + t] ^ RK_BITS[0][b][bit];
        }}
    }}

    for (int round = 1; round <= 9; round++) {{
        #pragma unroll
        for (int b = 0; b < 16; b++) {{
            sbox_bp128_lop3_inline(st[b], sb[b]);
        }}

        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {{
            sr[0][bit]  = sb[0][bit];
            sr[1][bit]  = sb[5][bit];
            sr[2][bit]  = sb[10][bit];
            sr[3][bit]  = sb[15][bit];
            sr[4][bit]  = sb[4][bit];
            sr[5][bit]  = sb[9][bit];
            sr[6][bit]  = sb[14][bit];
            sr[7][bit]  = sb[3][bit];
            sr[8][bit]  = sb[8][bit];
            sr[9][bit]  = sb[13][bit];
            sr[10][bit] = sb[2][bit];
            sr[11][bit] = sb[7][bit];
            sr[12][bit] = sb[12][bit];
            sr[13][bit] = sb[1][bit];
            sr[14][bit] = sb[6][bit];
            sr[15][bit] = sb[11][bit];
        }}

        #pragma unroll
        for (int c = 0; c < 4; c++) {{
            int i0 = c * 4 + 0;
            int i1 = c * 4 + 1;
            int i2 = c * 4 + 2;
            int i3 = c * 4 + 3;
            uint32_t t8[8], u[8], x01[8], x12[8], x23[8], x30[8], xt[8];

            #pragma unroll
            for (int i = 0; i < 8; i++) {{
                t8[i] = sr[i0][i] ^ sr[i1][i] ^ sr[i2][i] ^ sr[i3][i];
                u[i] = sr[i0][i];
            }}
            xor8(sr[i0], sr[i1], x01);
            xor8(sr[i1], sr[i2], x12);
            xor8(sr[i2], sr[i3], x23);
            xor8(sr[i3], u, x30);

            xtime8(x01, xt);
            #pragma unroll
            for (int i = 0; i < 8; i++) mc[i0][i] = sr[i0][i] ^ t8[i] ^ xt[i];

            xtime8(x12, xt);
            #pragma unroll
            for (int i = 0; i < 8; i++) mc[i1][i] = sr[i1][i] ^ t8[i] ^ xt[i];

            xtime8(x23, xt);
            #pragma unroll
            for (int i = 0; i < 8; i++) mc[i2][i] = sr[i2][i] ^ t8[i] ^ xt[i];

            xtime8(x30, xt);
            #pragma unroll
            for (int i = 0; i < 8; i++) mc[i3][i] = sr[i3][i] ^ t8[i] ^ xt[i];
        }}

        #pragma unroll
        for (int b = 0; b < 16; b++) {{
            #pragma unroll
            for (int bit = 0; bit < 8; bit++) {{
                st[b][bit] = mc[b][bit] ^ RK_BITS[round][b][bit];
            }}
        }}
    }}

    #pragma unroll
    for (int b = 0; b < 16; b++) {{
        sbox_bp128_lop3_inline(st[b], sb[b]);
    }}
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) {{
        sr[0][bit]  = sb[0][bit];
        sr[1][bit]  = sb[5][bit];
        sr[2][bit]  = sb[10][bit];
        sr[3][bit]  = sb[15][bit];
        sr[4][bit]  = sb[4][bit];
        sr[5][bit]  = sb[9][bit];
        sr[6][bit]  = sb[14][bit];
        sr[7][bit]  = sb[3][bit];
        sr[8][bit]  = sb[8][bit];
        sr[9][bit]  = sb[13][bit];
        sr[10][bit] = sb[2][bit];
        sr[11][bit] = sb[7][bit];
        sr[12][bit] = sb[12][bit];
        sr[13][bit] = sb[1][bit];
        sr[14][bit] = sb[6][bit];
        sr[15][bit] = sb[11][bit];
    }}
    #pragma unroll
    for (int b = 0; b < 16; b++) {{
        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {{
            size_t plane = (size_t)b * 8u + (size_t)bit;
            out_ptr[plane * threads + t] = sr[b][bit] ^ RK_BITS[10][b][bit];
        }}
    }}
}}
"""


def _kernel_cu_source_replacement_coalesced4(sbox_inline_cuda: str) -> str:
    return f"""
#include <stdint.h>

{sbox_inline_cuda}

__device__ __constant__ uint32_t RK_BITS[11][16][8];

__device__ __forceinline__ void xor8(
    const uint32_t a[8], const uint32_t b[8], uint32_t o[8]
) {{
    #pragma unroll
    for (int i = 0; i < 8; i++) o[i] = a[i] ^ b[i];
}}

__device__ __forceinline__ void xtime8(const uint32_t a[8], uint32_t o[8]) {{
    const uint32_t b7 = a[7];
    o[7] = a[6];
    o[6] = a[5];
    o[5] = a[4];
    o[4] = a[3] ^ b7;
    o[3] = a[2] ^ b7;
    o[2] = a[1];
    o[1] = a[0] ^ b7;
    o[0] = b7;
}}

extern "C" __global__ void aes10_bp128_kernel(
    const uint32_t* __restrict__ in_ptr,
    uint32_t* __restrict__ out_ptr,
    uint32_t n_threads
) {{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;

    uint32_t st[16][8];
    uint32_t sb[16][8];
    uint32_t sr[16][8];
    uint32_t mc[16][8];
    const size_t threads = (size_t)n_threads;
    const size_t t = (size_t)tid;
    const uint4* in4 = (const uint4*)in_ptr;
    uint4* out4 = (uint4*)out_ptr;

    #pragma unroll
    for (int b = 0; b < 16; b++) {{
        size_t idx0 = ((size_t)b * 2u + 0u) * threads + t;
        size_t idx1 = ((size_t)b * 2u + 1u) * threads + t;
        uint4 v0 = in4[idx0];
        uint4 v1 = in4[idx1];
        st[b][0] = v0.x ^ RK_BITS[0][b][0];
        st[b][1] = v0.y ^ RK_BITS[0][b][1];
        st[b][2] = v0.z ^ RK_BITS[0][b][2];
        st[b][3] = v0.w ^ RK_BITS[0][b][3];
        st[b][4] = v1.x ^ RK_BITS[0][b][4];
        st[b][5] = v1.y ^ RK_BITS[0][b][5];
        st[b][6] = v1.z ^ RK_BITS[0][b][6];
        st[b][7] = v1.w ^ RK_BITS[0][b][7];
    }}

    for (int round = 1; round <= 9; round++) {{
        #pragma unroll
        for (int b = 0; b < 16; b++) {{
            sbox_bp128_lop3_inline(st[b], sb[b]);
        }}

        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {{
            sr[0][bit]  = sb[0][bit];
            sr[1][bit]  = sb[5][bit];
            sr[2][bit]  = sb[10][bit];
            sr[3][bit]  = sb[15][bit];
            sr[4][bit]  = sb[4][bit];
            sr[5][bit]  = sb[9][bit];
            sr[6][bit]  = sb[14][bit];
            sr[7][bit]  = sb[3][bit];
            sr[8][bit]  = sb[8][bit];
            sr[9][bit]  = sb[13][bit];
            sr[10][bit] = sb[2][bit];
            sr[11][bit] = sb[7][bit];
            sr[12][bit] = sb[12][bit];
            sr[13][bit] = sb[1][bit];
            sr[14][bit] = sb[6][bit];
            sr[15][bit] = sb[11][bit];
        }}

        #pragma unroll
        for (int c = 0; c < 4; c++) {{
            int i0 = c * 4 + 0;
            int i1 = c * 4 + 1;
            int i2 = c * 4 + 2;
            int i3 = c * 4 + 3;
            uint32_t t8[8], u[8], x01[8], x12[8], x23[8], x30[8], xt[8];

            #pragma unroll
            for (int i = 0; i < 8; i++) {{
                t8[i] = sr[i0][i] ^ sr[i1][i] ^ sr[i2][i] ^ sr[i3][i];
                u[i] = sr[i0][i];
            }}
            xor8(sr[i0], sr[i1], x01);
            xor8(sr[i1], sr[i2], x12);
            xor8(sr[i2], sr[i3], x23);
            xor8(sr[i3], u, x30);

            xtime8(x01, xt);
            #pragma unroll
            for (int i = 0; i < 8; i++) mc[i0][i] = sr[i0][i] ^ t8[i] ^ xt[i];

            xtime8(x12, xt);
            #pragma unroll
            for (int i = 0; i < 8; i++) mc[i1][i] = sr[i1][i] ^ t8[i] ^ xt[i];

            xtime8(x23, xt);
            #pragma unroll
            for (int i = 0; i < 8; i++) mc[i2][i] = sr[i2][i] ^ t8[i] ^ xt[i];

            xtime8(x30, xt);
            #pragma unroll
            for (int i = 0; i < 8; i++) mc[i3][i] = sr[i3][i] ^ t8[i] ^ xt[i];
        }}

        #pragma unroll
        for (int b = 0; b < 16; b++) {{
            #pragma unroll
            for (int bit = 0; bit < 8; bit++) {{
                st[b][bit] = mc[b][bit] ^ RK_BITS[round][b][bit];
            }}
        }}
    }}

    #pragma unroll
    for (int b = 0; b < 16; b++) {{
        sbox_bp128_lop3_inline(st[b], sb[b]);
    }}
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) {{
        sr[0][bit]  = sb[0][bit];
        sr[1][bit]  = sb[5][bit];
        sr[2][bit]  = sb[10][bit];
        sr[3][bit]  = sb[15][bit];
        sr[4][bit]  = sb[4][bit];
        sr[5][bit]  = sb[9][bit];
        sr[6][bit]  = sb[14][bit];
        sr[7][bit]  = sb[3][bit];
        sr[8][bit]  = sb[8][bit];
        sr[9][bit]  = sb[13][bit];
        sr[10][bit] = sb[2][bit];
        sr[11][bit] = sb[7][bit];
        sr[12][bit] = sb[12][bit];
        sr[13][bit] = sb[1][bit];
        sr[14][bit] = sb[6][bit];
        sr[15][bit] = sb[11][bit];
    }}
    #pragma unroll
    for (int b = 0; b < 16; b++) {{
        size_t idx0 = ((size_t)b * 2u + 0u) * threads + t;
        size_t idx1 = ((size_t)b * 2u + 1u) * threads + t;
        uint4 o0;
        uint4 o1;
        o0.x = sr[b][0] ^ RK_BITS[10][b][0];
        o0.y = sr[b][1] ^ RK_BITS[10][b][1];
        o0.z = sr[b][2] ^ RK_BITS[10][b][2];
        o0.w = sr[b][3] ^ RK_BITS[10][b][3];
        o1.x = sr[b][4] ^ RK_BITS[10][b][4];
        o1.y = sr[b][5] ^ RK_BITS[10][b][5];
        o1.z = sr[b][6] ^ RK_BITS[10][b][6];
        o1.w = sr[b][7] ^ RK_BITS[10][b][7];
        out4[idx0] = o0;
        out4[idx1] = o1;
    }}
}}
"""


def _host_bench_cu_source_legacy() -> str:
    return r"""
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void ck(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) {
    const char* s = 0;
    cuGetErrorString(r, &s);
    fprintf(stderr, "%s failed: %d (%s)\n", what, (int)r, s ? s : "?");
    exit(1);
  }
}

int main(int argc, char** argv) {
  const char* cubin = argv[1];
  int threads = atoi(argv[2]);
  int block = atoi(argv[3]);
  int reps = atoi(argv[4]);
  int do_check = (argc > 5) ? atoi(argv[5]) : 0;

  ck(cuInit(0), "cuInit");
  CUdevice dev;
  ck(cuDeviceGet(&dev, 0), "cuDeviceGet");
  CUcontext ctx;
  ck(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");

  CUmodule mod;
  ck(cuModuleLoad(&mod, cubin), "cuModuleLoad");
  CUfunction fn;
  ck(cuModuleGetFunction(&fn, mod, "aes10_bp128_kernel"), "cuModuleGetFunction");

  size_t out_words = (size_t)threads * 128;
  size_t out_bytes = out_words * sizeof(uint32_t);
  CUdeviceptr d_out;
  ck(cuMemAlloc(&d_out, out_bytes), "cuMemAlloc(out)");
  ck(cuMemsetD8(d_out, 0, out_bytes), "cuMemsetD8(out)");

  int grid = (threads + block - 1) / block;
  void* params[] = { &d_out, &threads };

  if (do_check) {
    ck(cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0, params, 0), "cuLaunchKernel(check)");
    ck(cuCtxSynchronize(), "cuCtxSynchronize(check)");

    uint32_t out0[128];
    ck(cuMemcpyDtoH(out0, d_out, sizeof(out0)), "cuMemcpyDtoH(check)");
    uint8_t got[16] = {0};
    for (int byte_idx = 0; byte_idx < 16; byte_idx++) {
      uint8_t v = 0;
      for (int bit = 0; bit < 8; bit++) {
        uint32_t w = out0[byte_idx * 8 + bit];
        v |= (uint8_t)((w & 1u) << bit);
      }
      got[byte_idx] = v;
    }
    const uint8_t exp[16] = {
      0x69,0xc4,0xe0,0xd8,0x6a,0x7b,0x04,0x30,
      0xd8,0xcd,0xb7,0x80,0x70,0xb4,0xc5,0x5a
    };
    int errors = 0;
    for (int i = 0; i < 16; i++) {
      if (got[i] != exp[i]) errors++;
    }
    if (errors) {
      fprintf(stderr, "FAIL: AES KAT mismatch\n  got=");
      for (int i = 0; i < 16; i++) fprintf(stderr, "%02x", got[i]);
      fprintf(stderr, "\n  exp=");
      for (int i = 0; i < 16; i++) fprintf(stderr, "%02x", exp[i]);
      fprintf(stderr, "\n");
      return 1;
    }
    printf("PASS: AES-128 known-answer check matched\n");
  }

  ck(cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0, params, 0), "cuLaunchKernel(warmup)");
  ck(cuCtxSynchronize(), "cuCtxSynchronize(warmup)");

  CUevent e0, e1;
  ck(cuEventCreate(&e0, 0), "cuEventCreate(start)");
  ck(cuEventCreate(&e1, 0), "cuEventCreate(end)");
  ck(cuEventRecord(e0, 0), "cuEventRecord(start)");
  for (int i = 0; i < reps; i++) {
    ck(cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0, params, 0), "cuLaunchKernel");
  }
  ck(cuEventRecord(e1, 0), "cuEventRecord(end)");
  ck(cuEventSynchronize(e1), "cuEventSynchronize");

  float ms = 0.0f;
  ck(cuEventElapsedTime(&ms, e0, e1), "cuEventElapsedTime");

  double seconds = (double)ms / 1000.0;
  double total_evals = (double)threads * 32.0 * (double)reps;
  double evals_per_sec = total_evals / seconds;
  double ns_per_eval = (seconds / total_evals) * 1e9;
  double mib_s = evals_per_sec * 16.0 / (1024.0 * 1024.0);

  printf("threads=%d block=%d reps=%d\n", threads, block, reps);
  printf("elapsed=%.3f ms\n", ms);
  printf("aes10_bp128_ptx: %.3f ns/eval, %.3fB evals/sec, %.2f MiB/s\n",
         ns_per_eval, evals_per_sec / 1e9, mib_s);

  cuMemFree(d_out);
  cuModuleUnload(mod);
  cuCtxDestroy(ctx);
  return 0;
}
"""


def _host_bench_cu_source_replacement(layout: str = "thread-major") -> str:
    if layout not in {"thread-major", "plane-major", "plane-major4"}:
        raise ValueError(f"unsupported replacement layout: {layout}")

    if layout == "thread-major":
        seed_fn = r"""
static void seed_plaintext_bitplanes(uint32_t* h_in, int threads) {
  const uint8_t pt[16] = {
    0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,
    0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff
  };
  for (int tid = 0; tid < threads; tid++) {
    size_t base = (size_t)tid * 128u;
    for (int b = 0; b < 16; b++) {
      uint8_t pb = pt[b];
      for (int bit = 0; bit < 8; bit++) {
        h_in[base + (size_t)b * 8u + (size_t)bit] = ((pb >> bit) & 1u) ? 0xffffffffu : 0u;
      }
    }
  }
}
"""
        decode_fn = r"""
    for (int byte_idx = 0; byte_idx < 16; byte_idx++) {
      uint8_t v = 0;
      for (int bit = 0; bit < 8; bit++) {
        uint32_t w = h_out[(size_t)byte_idx * 8u + (size_t)bit];
        v |= (uint8_t)((w & 1u) << bit);
      }
      got[byte_idx] = v;
    }
"""
    elif layout == "plane-major":
        seed_fn = r"""
static void seed_plaintext_bitplanes(uint32_t* h_in, int threads) {
  const uint8_t pt[16] = {
    0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,
    0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff
  };
  for (int tid = 0; tid < threads; tid++) {
    for (int b = 0; b < 16; b++) {
      uint8_t pb = pt[b];
      for (int bit = 0; bit < 8; bit++) {
        size_t plane = (size_t)b * 8u + (size_t)bit;
        h_in[plane * (size_t)threads + (size_t)tid] = ((pb >> bit) & 1u) ? 0xffffffffu : 0u;
      }
    }
  }
}
"""
        decode_fn = r"""
    for (int byte_idx = 0; byte_idx < 16; byte_idx++) {
      uint8_t v = 0;
      for (int bit = 0; bit < 8; bit++) {
        size_t plane = (size_t)byte_idx * 8u + (size_t)bit;
        uint32_t w = h_out[plane * (size_t)threads];
        v |= (uint8_t)((w & 1u) << bit);
      }
      got[byte_idx] = v;
    }
"""
    else:
        seed_fn = r"""
static void seed_plaintext_bitplanes(uint32_t* h_in, int threads) {
  const uint8_t pt[16] = {
    0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,
    0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff
  };
  uint4* in4 = (uint4*)h_in;
  for (int tid = 0; tid < threads; tid++) {
    for (int b = 0; b < 16; b++) {
      uint8_t pb = pt[b];
      uint4 v0;
      uint4 v1;
      v0.x = ((pb >> 0) & 1u) ? 0xffffffffu : 0u;
      v0.y = ((pb >> 1) & 1u) ? 0xffffffffu : 0u;
      v0.z = ((pb >> 2) & 1u) ? 0xffffffffu : 0u;
      v0.w = ((pb >> 3) & 1u) ? 0xffffffffu : 0u;
      v1.x = ((pb >> 4) & 1u) ? 0xffffffffu : 0u;
      v1.y = ((pb >> 5) & 1u) ? 0xffffffffu : 0u;
      v1.z = ((pb >> 6) & 1u) ? 0xffffffffu : 0u;
      v1.w = ((pb >> 7) & 1u) ? 0xffffffffu : 0u;
      size_t idx0 = ((size_t)b * 2u + 0u) * (size_t)threads + (size_t)tid;
      size_t idx1 = ((size_t)b * 2u + 1u) * (size_t)threads + (size_t)tid;
      in4[idx0] = v0;
      in4[idx1] = v1;
    }
  }
}
"""
        decode_fn = r"""
    const uint4* out4 = (const uint4*)h_out;
    for (int byte_idx = 0; byte_idx < 16; byte_idx++) {
      uint8_t v = 0;
      for (int bit = 0; bit < 8; bit++) {
        size_t group = (size_t)byte_idx * 2u + (size_t)(bit >> 2);
        size_t idx = group * (size_t)threads;
        uint4 w4 = out4[idx];
        uint32_t w = 0u;
        if ((bit & 3) == 0) w = w4.x;
        else if ((bit & 3) == 1) w = w4.y;
        else if ((bit & 3) == 2) w = w4.z;
        else w = w4.w;
        v |= (uint8_t)((w & 1u) << bit);
      }
      got[byte_idx] = v;
    }
"""

    rk_bytes = _rk_bytes_c_initializer()
    return (
        r"""
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void ck(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) {
    const char* s = 0;
    cuGetErrorString(r, &s);
    fprintf(stderr, "%s failed: %d (%s)\n", what, (int)r, s ? s : "?");
    exit(1);
  }
}
"""
        + seed_fn
        + r"""

static void build_rk_bits(uint32_t rk_bits[11][16][8]) {
  static const uint8_t rk_bytes[11][16] = {
"""
        + rk_bytes
        + r"""
  };
  for (int r = 0; r < 11; r++) {
    for (int b = 0; b < 16; b++) {
      uint8_t x = rk_bytes[r][b];
      for (int bit = 0; bit < 8; bit++) {
        rk_bits[r][b][bit] = ((x >> bit) & 1u) ? 0xffffffffu : 0u;
      }
    }
  }
}

int main(int argc, char** argv) {
  const char* cubin = argv[1];
  int threads = atoi(argv[2]);
  int block = atoi(argv[3]);
  int reps = atoi(argv[4]);
  int do_check = (argc > 5) ? atoi(argv[5]) : 0;

  ck(cuInit(0), "cuInit");
  CUdevice dev;
  ck(cuDeviceGet(&dev, 0), "cuDeviceGet");
  CUcontext ctx;
  ck(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");

  CUmodule mod;
  ck(cuModuleLoad(&mod, cubin), "cuModuleLoad");
  CUfunction fn;
  ck(cuModuleGetFunction(&fn, mod, "aes10_bp128_kernel"), "cuModuleGetFunction");

  CUdeviceptr d_rk = 0;
  size_t rk_nbytes = 0;
  ck(cuModuleGetGlobal(&d_rk, &rk_nbytes, mod, "RK_BITS"), "cuModuleGetGlobal(RK_BITS)");
  uint32_t h_rk_bits[11][16][8];
  build_rk_bits(h_rk_bits);
  if (rk_nbytes < sizeof(h_rk_bits)) {
    fprintf(stderr, "RK_BITS symbol too small: %zu < %zu\n", rk_nbytes, sizeof(h_rk_bits));
    return 2;
  }
  ck(cuMemcpyHtoD(d_rk, h_rk_bits, sizeof(h_rk_bits)), "cuMemcpyHtoD(RK_BITS)");

  size_t words = (size_t)threads * 128u;
  size_t bytes = words * sizeof(uint32_t);
  CUdeviceptr d_in = 0, d_out = 0;
  ck(cuMemAlloc(&d_in, bytes), "cuMemAlloc(in)");
  ck(cuMemAlloc(&d_out, bytes), "cuMemAlloc(out)");

  uint32_t* h_in = (uint32_t*)malloc(bytes);
  if (!h_in) {
    fprintf(stderr, "malloc failed for h_in\n");
    return 2;
  }
  seed_plaintext_bitplanes(h_in, threads);
  ck(cuMemcpyHtoD(d_in, h_in, bytes), "cuMemcpyHtoD(in)");
  ck(cuMemsetD8(d_out, 0, bytes), "cuMemsetD8(out)");

  int grid = (threads + block - 1) / block;
  void* params[] = { &d_in, &d_out, &threads };

  if (do_check) {
    ck(cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0, params, 0), "cuLaunchKernel(check)");
    ck(cuCtxSynchronize(), "cuCtxSynchronize(check)");

    uint32_t* h_out = (uint32_t*)malloc(bytes);
    if (!h_out) {
      fprintf(stderr, "malloc failed for h_out\n");
      return 2;
    }
    ck(cuMemcpyDtoH(h_out, d_out, bytes), "cuMemcpyDtoH(check)");
    uint8_t got[16] = {0};
"""
        + decode_fn
        + r"""
    const uint8_t exp[16] = {
      0x69,0xc4,0xe0,0xd8,0x6a,0x7b,0x04,0x30,
      0xd8,0xcd,0xb7,0x80,0x70,0xb4,0xc5,0x5a
    };
    int errors = 0;
    for (int i = 0; i < 16; i++) {
      if (got[i] != exp[i]) errors++;
    }
    free(h_out);
    if (errors) {
      fprintf(stderr, "FAIL: AES KAT mismatch\n  got=");
      for (int i = 0; i < 16; i++) fprintf(stderr, "%02x", got[i]);
      fprintf(stderr, "\n  exp=");
      for (int i = 0; i < 16; i++) fprintf(stderr, "%02x", exp[i]);
      fprintf(stderr, "\n");
      return 1;
    }
    printf("PASS: AES-128 known-answer check matched\n");
  }

  ck(cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0, params, 0), "cuLaunchKernel(warmup)");
  ck(cuCtxSynchronize(), "cuCtxSynchronize(warmup)");

  CUevent e0, e1;
  ck(cuEventCreate(&e0, 0), "cuEventCreate(start)");
  ck(cuEventCreate(&e1, 0), "cuEventCreate(end)");
  ck(cuEventRecord(e0, 0), "cuEventRecord(start)");
  for (int i = 0; i < reps; i++) {
    ck(cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0, params, 0), "cuLaunchKernel");
  }
  ck(cuEventRecord(e1, 0), "cuEventRecord(end)");
  ck(cuEventSynchronize(e1), "cuEventSynchronize");

  float ms = 0.0f;
  ck(cuEventElapsedTime(&ms, e0, e1), "cuEventElapsedTime");

  double seconds = (double)ms / 1000.0;
  double total_evals = (double)threads * 32.0 * (double)reps;
  double evals_per_sec = total_evals / seconds;
  double ns_per_eval = (seconds / total_evals) * 1e9;
  double mib_s = evals_per_sec * 16.0 / (1024.0 * 1024.0);

  printf("threads=%d block=%d reps=%d\n", threads, block, reps);
  printf("elapsed=%.3f ms\n", ms);
  printf("aes10_bp128_ptx: %.3f ns/eval, %.3fB evals/sec, %.2f MiB/s\n",
         ns_per_eval, evals_per_sec / 1e9, mib_s);

  free(h_in);
  cuMemFree(d_in);
  cuMemFree(d_out);
  cuModuleUnload(mod);
  cuCtxDestroy(ctx);
  return 0;
}
"""
    )


def _kernel_cu_source_replacement_coalesced4_paramrk(sbox_inline_cuda: str) -> str:
    src = _kernel_cu_source_replacement_coalesced4(sbox_inline_cuda)
    src = src.replace("__device__ __constant__ uint32_t RK_BITS[11][16][8];\n\n", "")
    sig_old = (
        'extern "C" __global__ void aes10_bp128_kernel(\n'
        "    const uint32_t* __restrict__ in_ptr,\n"
        "    uint32_t* __restrict__ out_ptr,\n"
        "    uint32_t n_threads\n"
        ") {"
    )
    sig_new = (
        'extern "C" __global__ void aes10_bp128_kernel(\n'
        "    const uint32_t* __restrict__ in_ptr,\n"
        "    uint32_t* __restrict__ out_ptr,\n"
        "    const uint32_t* __restrict__ rk_bits,\n"
        "    uint32_t n_threads\n"
        ") {"
    )
    src = src.replace(sig_old, sig_new)
    src = src.replace(
        "    const size_t threads = (size_t)n_threads;\n"
        "    const size_t t = (size_t)tid;\n"
        "    const uint4* in4 = (const uint4*)in_ptr;\n"
        "    uint4* out4 = (uint4*)out_ptr;\n",
        "    const size_t threads = (size_t)n_threads;\n"
        "    const size_t t = (size_t)tid;\n"
        "    const uint4* in4 = (const uint4*)in_ptr;\n"
        "    uint4* out4 = (uint4*)out_ptr;\n"
        "    #define RK_AT(r,b,bit) rk_bits[(((size_t)(r) * 16u + (size_t)(b)) * 8u + (size_t)(bit))]\n",
    )
    src = re.sub(
        r"RK_BITS\[([^\]]+)\]\[([^\]]+)\]\[([^\]]+)\]",
        r"RK_AT(\1,\2,\3)",
        src,
    )
    src = src.replace('\n}}\n"""\n', '\n    #undef RK_AT\n}}\n"""\n')
    return src


def _kernel_cu_source_replacement_coalesced4_paramrk_shared(
    sbox_inline_cuda: str,
) -> str:
    src = _kernel_cu_source_replacement_coalesced4_paramrk(sbox_inline_cuda)
    src = src.replace(
        "    const size_t threads = (size_t)n_threads;\n"
        "    const size_t t = (size_t)tid;\n"
        "    const uint4* in4 = (const uint4*)in_ptr;\n"
        "    uint4* out4 = (uint4*)out_ptr;\n"
        "    #define RK_AT(r,b,bit) rk_bits[(((size_t)(r) * 16u + (size_t)(b)) * 8u + (size_t)(bit))]\n",
        "    const size_t threads = (size_t)n_threads;\n"
        "    const size_t t = (size_t)tid;\n"
        "    const uint4* in4 = (const uint4*)in_ptr;\n"
        "    uint4* out4 = (uint4*)out_ptr;\n"
        "    const uint4* rk4_global = (const uint4*)rk_bits;\n"
        "    extern __shared__ uint4 rk4_shared[];\n"
        "    const size_t rk4_words = (11u * 16u * 8u) / 4u;\n"
        "    for (size_t i = (size_t)threadIdx.x; i < rk4_words; i += (size_t)blockDim.x) {\n"
        "        rk4_shared[i] = rk4_global[i];\n"
        "    }\n"
        "    __syncthreads();\n"
        "    const uint32_t* rk_shared = (const uint32_t*)rk4_shared;\n"
        "    #define RK_AT(r,b,bit) rk_shared[(((size_t)(r) * 16u + (size_t)(b)) * 8u + (size_t)(bit))]\n",
    )
    return src


def _kernel_cu_source_replacement_coalesced4_paramrk_soa(
    sbox_inline_cuda: str,
) -> str:
    src = _kernel_cu_source_replacement_coalesced4_paramrk(sbox_inline_cuda)
    src = src.replace(
        "    #define RK_AT(r,b,bit) rk_bits[(((size_t)(r) * 16u + (size_t)(b)) * 8u + (size_t)(bit))]\n",
        "    #define RK_AT(r,b,bit) rk_bits[(((((size_t)(r) * 16u + (size_t)(b)) * 8u + (size_t)(bit)) * threads) + t)]\n",
    )
    return src


def _kernel_cu_source_replacement_coalesced4_paramrk_soa_packed(
    sbox_inline_cuda: str,
) -> str:
    src = _kernel_cu_source_replacement_coalesced4_paramrk(sbox_inline_cuda)
    src = src.replace(
        "const uint32_t* __restrict__ rk_bits",
        "const uint8_t* __restrict__ rk_bytes",
    )
    src = src.replace(
        "    #define RK_AT(r,b,bit) rk_bits[(((size_t)(r) * 16u + (size_t)(b)) * 8u + (size_t)(bit))]\n",
        "    #define RK_BYTE_AT(r,b) rk_bytes[((((size_t)(r) * 16u + (size_t)(b)) * threads) + t)]\n"
        "    #define RK_AT(r,b,bit) ((((uint32_t)(RK_BYTE_AT((r),(b))) >> (bit)) & 1u) ? 0xffffffffu : 0u)\n",
    )
    src = src.replace(
        "    #undef RK_AT\n",
        "    #undef RK_AT\n    #undef RK_BYTE_AT\n",
    )
    return src


def _kernel_cu_source_replacement_coalesced4_masterkey_soa(
    sbox_inline_cuda: str,
) -> str:
    return f"""
#include <stdint.h>

{sbox_inline_cuda}

__device__ __constant__ uint8_t AES_SBOX_KS[256] = {{
{_aes_sbox_c_initializer()}
}};

__device__ __forceinline__ uint8_t aes_xtime_u8(uint8_t x) {{
    uint8_t hi = (uint8_t)(x >> 7);
    return (uint8_t)(((uint8_t)(x << 1)) ^ (hi ? 0x1bu : 0x00u));
}}

__device__ __forceinline__ void aes128_expand_round_key_u8(
    uint8_t rk[16], uint8_t rcon
) {{
    uint8_t t0 = AES_SBOX_KS[rk[13]];
    uint8_t t1 = AES_SBOX_KS[rk[14]];
    uint8_t t2 = AES_SBOX_KS[rk[15]];
    uint8_t t3 = AES_SBOX_KS[rk[12]];
    t0 ^= rcon;
    rk[0] ^= t0;
    rk[1] ^= t1;
    rk[2] ^= t2;
    rk[3] ^= t3;
    #pragma unroll
    for (int i = 4; i < 16; i++) {{
        rk[i] ^= rk[i - 4];
    }}
}}

__device__ __forceinline__ void xor8(
    const uint32_t a[8], const uint32_t b[8], uint32_t o[8]
) {{
    #pragma unroll
    for (int i = 0; i < 8; i++) o[i] = a[i] ^ b[i];
}}

__device__ __forceinline__ void xtime8(const uint32_t a[8], uint32_t o[8]) {{
    const uint32_t b7 = a[7];
    o[7] = a[6];
    o[6] = a[5];
    o[5] = a[4];
    o[4] = a[3] ^ b7;
    o[3] = a[2] ^ b7;
    o[2] = a[1];
    o[1] = a[0] ^ b7;
    o[0] = b7;
}}

extern "C" __global__ void aes10_bp128_kernel(
    const uint32_t* __restrict__ in_ptr,
    uint32_t* __restrict__ out_ptr,
    const uint8_t* __restrict__ key_bytes,
    uint32_t n_threads
) {{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;

    uint32_t st[16][8];
    uint32_t sb[16][8];
    uint32_t sr[16][8];
    uint32_t mc[16][8];
    uint8_t rk[16];
    const size_t threads = (size_t)n_threads;
    const size_t t = (size_t)tid;
    const uint4* in4 = (const uint4*)in_ptr;
    uint4* out4 = (uint4*)out_ptr;
    #define RK_MASK(k,bit) ((((uint32_t)(k) >> (bit)) & 1u) ? 0xffffffffu : 0u)

    #pragma unroll
    for (int b = 0; b < 16; b++) {{
        rk[b] = key_bytes[(size_t)b * threads + t];
    }}

    #pragma unroll
    for (int b = 0; b < 16; b++) {{
        size_t idx0 = ((size_t)b * 2u + 0u) * threads + t;
        size_t idx1 = ((size_t)b * 2u + 1u) * threads + t;
        uint4 v0 = in4[idx0];
        uint4 v1 = in4[idx1];
        uint8_t rk0 = rk[b];
        st[b][0] = v0.x ^ RK_MASK(rk0,0);
        st[b][1] = v0.y ^ RK_MASK(rk0,1);
        st[b][2] = v0.z ^ RK_MASK(rk0,2);
        st[b][3] = v0.w ^ RK_MASK(rk0,3);
        st[b][4] = v1.x ^ RK_MASK(rk0,4);
        st[b][5] = v1.y ^ RK_MASK(rk0,5);
        st[b][6] = v1.z ^ RK_MASK(rk0,6);
        st[b][7] = v1.w ^ RK_MASK(rk0,7);
    }}

    uint8_t rcon = 0x01u;
    for (int round = 1; round <= 9; round++) {{
        #pragma unroll
        for (int b = 0; b < 16; b++) {{
            sbox_bp128_lop3_inline(st[b], sb[b]);
        }}

        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {{
            sr[0][bit]  = sb[0][bit];
            sr[1][bit]  = sb[5][bit];
            sr[2][bit]  = sb[10][bit];
            sr[3][bit]  = sb[15][bit];
            sr[4][bit]  = sb[4][bit];
            sr[5][bit]  = sb[9][bit];
            sr[6][bit]  = sb[14][bit];
            sr[7][bit]  = sb[3][bit];
            sr[8][bit]  = sb[8][bit];
            sr[9][bit]  = sb[13][bit];
            sr[10][bit] = sb[2][bit];
            sr[11][bit] = sb[7][bit];
            sr[12][bit] = sb[12][bit];
            sr[13][bit] = sb[1][bit];
            sr[14][bit] = sb[6][bit];
            sr[15][bit] = sb[11][bit];
        }}

        #pragma unroll
        for (int c = 0; c < 4; c++) {{
            int i0 = c * 4 + 0;
            int i1 = c * 4 + 1;
            int i2 = c * 4 + 2;
            int i3 = c * 4 + 3;
            uint32_t t8[8], u[8], x01[8], x12[8], x23[8], x30[8], xt[8];

            #pragma unroll
            for (int i = 0; i < 8; i++) {{
                t8[i] = sr[i0][i] ^ sr[i1][i] ^ sr[i2][i] ^ sr[i3][i];
                u[i] = sr[i0][i];
            }}
            xor8(sr[i0], sr[i1], x01);
            xor8(sr[i1], sr[i2], x12);
            xor8(sr[i2], sr[i3], x23);
            xor8(sr[i3], u, x30);

            xtime8(x01, xt);
            #pragma unroll
            for (int i = 0; i < 8; i++) mc[i0][i] = sr[i0][i] ^ t8[i] ^ xt[i];

            xtime8(x12, xt);
            #pragma unroll
            for (int i = 0; i < 8; i++) mc[i1][i] = sr[i1][i] ^ t8[i] ^ xt[i];

            xtime8(x23, xt);
            #pragma unroll
            for (int i = 0; i < 8; i++) mc[i2][i] = sr[i2][i] ^ t8[i] ^ xt[i];

            xtime8(x30, xt);
            #pragma unroll
            for (int i = 0; i < 8; i++) mc[i3][i] = sr[i3][i] ^ t8[i] ^ xt[i];
        }}

        aes128_expand_round_key_u8(rk, rcon);
        rcon = aes_xtime_u8(rcon);

        #pragma unroll
        for (int b = 0; b < 16; b++) {{
            uint8_t rkb = rk[b];
            st[b][0] = mc[b][0] ^ RK_MASK(rkb,0);
            st[b][1] = mc[b][1] ^ RK_MASK(rkb,1);
            st[b][2] = mc[b][2] ^ RK_MASK(rkb,2);
            st[b][3] = mc[b][3] ^ RK_MASK(rkb,3);
            st[b][4] = mc[b][4] ^ RK_MASK(rkb,4);
            st[b][5] = mc[b][5] ^ RK_MASK(rkb,5);
            st[b][6] = mc[b][6] ^ RK_MASK(rkb,6);
            st[b][7] = mc[b][7] ^ RK_MASK(rkb,7);
        }}
    }}

    #pragma unroll
    for (int b = 0; b < 16; b++) {{
        sbox_bp128_lop3_inline(st[b], sb[b]);
    }}
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) {{
        sr[0][bit]  = sb[0][bit];
        sr[1][bit]  = sb[5][bit];
        sr[2][bit]  = sb[10][bit];
        sr[3][bit]  = sb[15][bit];
        sr[4][bit]  = sb[4][bit];
        sr[5][bit]  = sb[9][bit];
        sr[6][bit]  = sb[14][bit];
        sr[7][bit]  = sb[3][bit];
        sr[8][bit]  = sb[8][bit];
        sr[9][bit]  = sb[13][bit];
        sr[10][bit] = sb[2][bit];
        sr[11][bit] = sb[7][bit];
        sr[12][bit] = sb[12][bit];
        sr[13][bit] = sb[1][bit];
        sr[14][bit] = sb[6][bit];
        sr[15][bit] = sb[11][bit];
    }}

    aes128_expand_round_key_u8(rk, rcon);

    #pragma unroll
    for (int b = 0; b < 16; b++) {{
        size_t idx0 = ((size_t)b * 2u + 0u) * threads + t;
        size_t idx1 = ((size_t)b * 2u + 1u) * threads + t;
        uint4 o0;
        uint4 o1;
        uint8_t rkb = rk[b];
        o0.x = sr[b][0] ^ RK_MASK(rkb,0);
        o0.y = sr[b][1] ^ RK_MASK(rkb,1);
        o0.z = sr[b][2] ^ RK_MASK(rkb,2);
        o0.w = sr[b][3] ^ RK_MASK(rkb,3);
        o1.x = sr[b][4] ^ RK_MASK(rkb,4);
        o1.y = sr[b][5] ^ RK_MASK(rkb,5);
        o1.z = sr[b][6] ^ RK_MASK(rkb,6);
        o1.w = sr[b][7] ^ RK_MASK(rkb,7);
        out4[idx0] = o0;
        out4[idx1] = o1;
    }}

    #undef RK_MASK
}}
"""


def _host_bench_cu_source_replacement_paramrk(layout: str = "plane-major4") -> str:
    if layout != "plane-major4":
        raise ValueError(
            "paramrk host path currently supports plane-major4 layout only"
        )

    src = _host_bench_cu_source_replacement(layout=layout)
    old_rk_block = (
        "  CUdeviceptr d_rk = 0;\n"
        "  size_t rk_nbytes = 0;\n"
        '  ck(cuModuleGetGlobal(&d_rk, &rk_nbytes, mod, "RK_BITS"), "cuModuleGetGlobal(RK_BITS)");\n'
        "  uint32_t h_rk_bits[11][16][8];\n"
        "  build_rk_bits(h_rk_bits);\n"
        "  if (rk_nbytes < sizeof(h_rk_bits)) {\n"
        '    fprintf(stderr, "RK_BITS symbol too small: %zu < %zu\\n", rk_nbytes, sizeof(h_rk_bits));\n'
        "    return 2;\n"
        "  }\n"
        '  ck(cuMemcpyHtoD(d_rk, h_rk_bits, sizeof(h_rk_bits)), "cuMemcpyHtoD(RK_BITS)");\n'
    )
    new_rk_block = (
        "  uint32_t h_rk_bits[11][16][8];\n"
        "  build_rk_bits(h_rk_bits);\n"
        "  CUdeviceptr d_rk = 0;\n"
        '  ck(cuMemAlloc(&d_rk, sizeof(h_rk_bits)), "cuMemAlloc(rk_bits)");\n'
        '  ck(cuMemcpyHtoD(d_rk, h_rk_bits, sizeof(h_rk_bits)), "cuMemcpyHtoD(rk_bits)");\n'
    )
    src = src.replace(old_rk_block, new_rk_block)
    src = src.replace(
        "  void* params[] = { &d_in, &d_out, &threads };\n",
        "  void* params[] = { &d_in, &d_out, &d_rk, &threads };\n",
    )
    src = src.replace(
        "  free(h_in);\n  cuMemFree(d_in);\n",
        "  free(h_in);\n  cuMemFree(d_rk);\n  cuMemFree(d_in);\n",
    )
    return src


def _host_bench_cu_source_replacement_paramrk_shared(
    layout: str = "plane-major4",
) -> str:
    src = _host_bench_cu_source_replacement_paramrk(layout=layout)
    src = src.replace(
        "  int grid = (threads + block - 1) / block;\n"
        "  void* params[] = { &d_in, &d_out, &d_rk, &threads };\n",
        "  int grid = (threads + block - 1) / block;\n"
        "  const unsigned int rk_shared_bytes = (unsigned int)sizeof(h_rk_bits);\n"
        "  void* params[] = { &d_in, &d_out, &d_rk, &threads };\n",
    )
    src = src.replace(
        ", 0, 0, params, 0)",
        ", rk_shared_bytes, 0, params, 0)",
    )
    return src


def _host_bench_cu_source_replacement_paramrk_soa(layout: str = "plane-major4") -> str:
    if layout != "plane-major4":
        raise ValueError(
            "paramrk_soa host path currently supports plane-major4 layout only"
        )

    src = _host_bench_cu_source_replacement(layout=layout)
    old_rk_block = (
        "  CUdeviceptr d_rk = 0;\n"
        "  size_t rk_nbytes = 0;\n"
        '  ck(cuModuleGetGlobal(&d_rk, &rk_nbytes, mod, "RK_BITS"), "cuModuleGetGlobal(RK_BITS)");\n'
        "  uint32_t h_rk_bits[11][16][8];\n"
        "  build_rk_bits(h_rk_bits);\n"
        "  if (rk_nbytes < sizeof(h_rk_bits)) {\n"
        '    fprintf(stderr, "RK_BITS symbol too small: %zu < %zu\\n", rk_nbytes, sizeof(h_rk_bits));\n'
        "    return 2;\n"
        "  }\n"
        '  ck(cuMemcpyHtoD(d_rk, h_rk_bits, sizeof(h_rk_bits)), "cuMemcpyHtoD(RK_BITS)");\n'
    )
    new_rk_block = (
        "  uint32_t h_rk_bits[11][16][8];\n"
        "  build_rk_bits(h_rk_bits);\n"
        "  const size_t rk_planes = (size_t)11u * 16u * 8u;\n"
        "  const size_t rk_words = rk_planes * (size_t)threads;\n"
        "  const size_t rk_bytes = rk_words * sizeof(uint32_t);\n"
        "  uint32_t* h_rk_soa = (uint32_t*)malloc(rk_bytes);\n"
        "  if (!h_rk_soa) {\n"
        '    fprintf(stderr, "malloc failed for h_rk_soa\\n");\n'
        "    return 2;\n"
        "  }\n"
        "  for (int r = 0; r < 11; r++) {\n"
        "    for (int b = 0; b < 16; b++) {\n"
        "      for (int bit = 0; bit < 8; bit++) {\n"
        "        size_t plane = ((size_t)r * 16u + (size_t)b) * 8u + (size_t)bit;\n"
        "        uint32_t v = h_rk_bits[r][b][bit];\n"
        "        size_t base = plane * (size_t)threads;\n"
        "        for (int tid = 0; tid < threads; tid++) {\n"
        "          h_rk_soa[base + (size_t)tid] = v;\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "  CUdeviceptr d_rk = 0;\n"
        '  ck(cuMemAlloc(&d_rk, rk_bytes), "cuMemAlloc(rk_bits_soa)");\n'
        '  ck(cuMemcpyHtoD(d_rk, h_rk_soa, rk_bytes), "cuMemcpyHtoD(rk_bits_soa)");\n'
    )
    src = src.replace(old_rk_block, new_rk_block)
    src = src.replace(
        "  void* params[] = { &d_in, &d_out, &threads };\n",
        "  void* params[] = { &d_in, &d_out, &d_rk, &threads };\n",
    )
    src = src.replace(
        "  free(h_in);\n  cuMemFree(d_in);\n",
        "  free(h_in);\n  free(h_rk_soa);\n  cuMemFree(d_rk);\n  cuMemFree(d_in);\n",
    )
    return src


def _host_bench_cu_source_replacement_paramrk_soa_packed(
    layout: str = "plane-major4",
) -> str:
    if layout != "plane-major4":
        raise ValueError(
            "paramrk_soa_packed host path currently supports plane-major4 layout only"
        )

    src = _host_bench_cu_source_replacement(layout=layout)
    old_rk_block = (
        "  CUdeviceptr d_rk = 0;\n"
        "  size_t rk_nbytes = 0;\n"
        '  ck(cuModuleGetGlobal(&d_rk, &rk_nbytes, mod, "RK_BITS"), "cuModuleGetGlobal(RK_BITS)");\n'
        "  uint32_t h_rk_bits[11][16][8];\n"
        "  build_rk_bits(h_rk_bits);\n"
        "  if (rk_nbytes < sizeof(h_rk_bits)) {\n"
        '    fprintf(stderr, "RK_BITS symbol too small: %zu < %zu\\n", rk_nbytes, sizeof(h_rk_bits));\n'
        "    return 2;\n"
        "  }\n"
        '  ck(cuMemcpyHtoD(d_rk, h_rk_bits, sizeof(h_rk_bits)), "cuMemcpyHtoD(RK_BITS)");\n'
    )
    new_rk_block = (
        "  static const uint8_t rk_bytes[11][16] = {\n"
        + _rk_bytes_c_initializer()
        + "\n"
        "  };\n"
        "  const size_t rk_planes = (size_t)11u * 16u;\n"
        "  const size_t rk_words = rk_planes * (size_t)threads;\n"
        "  const size_t rk_bytes_len = rk_words * sizeof(uint8_t);\n"
        "  uint8_t* h_rk_soa_packed = (uint8_t*)malloc(rk_bytes_len);\n"
        "  if (!h_rk_soa_packed) {\n"
        '    fprintf(stderr, "malloc failed for h_rk_soa_packed\\n");\n'
        "    return 2;\n"
        "  }\n"
        "  for (int r = 0; r < 11; r++) {\n"
        "    for (int b = 0; b < 16; b++) {\n"
        "      uint8_t kv = rk_bytes[r][b];\n"
        "      size_t plane = (size_t)r * 16u + (size_t)b;\n"
        "      size_t base = plane * (size_t)threads;\n"
        "      for (int tid = 0; tid < threads; tid++) {\n"
        "        h_rk_soa_packed[base + (size_t)tid] = kv;\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "  CUdeviceptr d_rk = 0;\n"
        '  ck(cuMemAlloc(&d_rk, rk_bytes_len), "cuMemAlloc(rk_bytes_soa)");\n'
        '  ck(cuMemcpyHtoD(d_rk, h_rk_soa_packed, rk_bytes_len), "cuMemcpyHtoD(rk_bytes_soa)");\n'
    )
    src = src.replace(old_rk_block, new_rk_block)
    src = src.replace(
        "  void* params[] = { &d_in, &d_out, &threads };\n",
        "  void* params[] = { &d_in, &d_out, &d_rk, &threads };\n",
    )
    src = src.replace(
        "  free(h_in);\n  cuMemFree(d_in);\n",
        "  free(h_in);\n  free(h_rk_soa_packed);\n  cuMemFree(d_rk);\n  cuMemFree(d_in);\n",
    )
    return src


def _host_bench_cu_source_replacement_masterkey_soa(
    layout: str = "plane-major4",
) -> str:
    if layout != "plane-major4":
        raise ValueError(
            "masterkey_soa host path currently supports plane-major4 layout only"
        )

    src = _host_bench_cu_source_replacement(layout=layout)
    old_rk_block = (
        "  CUdeviceptr d_rk = 0;\n"
        "  size_t rk_nbytes = 0;\n"
        '  ck(cuModuleGetGlobal(&d_rk, &rk_nbytes, mod, "RK_BITS"), "cuModuleGetGlobal(RK_BITS)");\n'
        "  uint32_t h_rk_bits[11][16][8];\n"
        "  build_rk_bits(h_rk_bits);\n"
        "  if (rk_nbytes < sizeof(h_rk_bits)) {\n"
        '    fprintf(stderr, "RK_BITS symbol too small: %zu < %zu\\n", rk_nbytes, sizeof(h_rk_bits));\n'
        "    return 2;\n"
        "  }\n"
        '  ck(cuMemcpyHtoD(d_rk, h_rk_bits, sizeof(h_rk_bits)), "cuMemcpyHtoD(RK_BITS)");\n'
    )
    new_rk_block = (
        "  static const uint8_t key_bytes_ref[16] = {\n"
        "    0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,\n"
        "    0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f\n"
        "  };\n"
        "  const size_t key_planes = (size_t)16u;\n"
        "  const size_t key_words = key_planes * (size_t)threads;\n"
        "  const size_t key_bytes_len = key_words * sizeof(uint8_t);\n"
        "  uint8_t* h_key_soa = (uint8_t*)malloc(key_bytes_len);\n"
        "  if (!h_key_soa) {\n"
        '    fprintf(stderr, "malloc failed for h_key_soa\\n");\n'
        "    return 2;\n"
        "  }\n"
        "  for (int b = 0; b < 16; b++) {\n"
        "    uint8_t kv = key_bytes_ref[b];\n"
        "    size_t base = (size_t)b * (size_t)threads;\n"
        "    for (int tid = 0; tid < threads; tid++) {\n"
        "      h_key_soa[base + (size_t)tid] = kv;\n"
        "    }\n"
        "  }\n"
        "  CUdeviceptr d_key = 0;\n"
        '  ck(cuMemAlloc(&d_key, key_bytes_len), "cuMemAlloc(key_bytes_soa)");\n'
        '  ck(cuMemcpyHtoD(d_key, h_key_soa, key_bytes_len), "cuMemcpyHtoD(key_bytes_soa)");\n'
    )
    src = src.replace(old_rk_block, new_rk_block)
    src = src.replace(
        "  void* params[] = { &d_in, &d_out, &threads };\n",
        "  void* params[] = { &d_in, &d_out, &d_key, &threads };\n",
    )
    src = src.replace(
        "  free(h_in);\n  cuMemFree(d_in);\n",
        "  free(h_in);\n  free(h_key_soa);\n  cuMemFree(d_key);\n  cuMemFree(d_in);\n",
    )
    return src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sm", default="sm_61")
    ap.add_argument("--threads", type=int, default=65_536)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument(
        "--autotune-blocks",
        default="",
        help="Comma-separated block sizes to sweep (e.g. 64,128,256,512)",
    )
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--check", action="store_true", help="Run AES known-answer check")
    ap.add_argument(
        "--maxrregcount",
        type=int,
        default=None,
        help="Pass --maxrregcount to nvcc/ptxas for occupancy tuning",
    )
    ap.add_argument(
        "--max-ternary-cones",
        type=int,
        default=None,
        help="Map at most N ternary cones in the BP128 S-box",
    )
    ap.add_argument(
        "--input-mode",
        choices=("const", "tid-xor"),
        default="const",
        help="Input pattern per thread: const reproduces historical runs, tid-xor avoids identical-thread data",
    )
    ap.add_argument(
        "--ptxas-report",
        action="store_true",
        help="Print ptxas -v resource report for the generated kernel PTX",
    )
    ap.add_argument(
        "--kernel-mode",
        choices=(
            "legacy",
            "legacy_tuned",
            "legacy_streamed",
            "legacy_streamed_tuned",
            "replacement",
            "replacement_tuned",
            "replacement_coalesced",
            "replacement_coalesced_tuned",
            "replacement_coalesced4",
            "replacement_coalesced4_tuned",
            "replacement_coalesced4_paramrk",
            "replacement_coalesced4_paramrk_tuned",
            "replacement_coalesced4_paramrk_shared",
            "replacement_coalesced4_paramrk_shared_tuned",
            "replacement_coalesced4_paramrk_soa",
            "replacement_coalesced4_paramrk_soa_tuned",
            "replacement_coalesced4_paramrk_soa_packed",
            "replacement_coalesced4_paramrk_soa_packed_tuned",
            "replacement_coalesced4_masterkey_soa",
            "replacement_coalesced4_masterkey_soa_tuned",
            "replacement_streamed",
            "replacement_streamed_tuned",
        ),
        default="legacy",
        help=(
            "legacy keeps historical behavior; replacement is runtime in/out + runtime round-key "
            "upload; *_coalesced uses plane-major runtime IO layout; *_coalesced4 uses uint4 "
            "plane-major groups; *_streamed uses column-streamed in-place rounds"
        ),
    )
    ap.add_argument("--out", default="out/aes10_bp128_lop3")
    args = ap.parse_args()

    if args.kernel_mode.startswith("replacement") and args.input_mode != "const":
        print("warning: replacement mode ignores --input-mode and uses runtime in_ptr")

    mapped, lop3_count = _build_bp128_mapped(args.max_ternary_cones)
    print(f"lop3.b32 count in BP128 S-box function: {lop3_count}")
    if args.kernel_mode in {"legacy", "legacy_tuned"}:
        sbox_cuda = _emit_sbox_inline_cuda(
            mapped, func_name="sbox_bp128_lop3_inline", noinline=False
        )
        kernel_src = _kernel_cu_source_legacy(sbox_cuda, input_mode=args.input_mode)
        host_src = _host_bench_cu_source_legacy()
    elif args.kernel_mode in {"legacy_streamed", "legacy_streamed_tuned"}:
        sbox_cuda = _emit_sbox_inline_cuda(
            mapped, func_name="sbox_bp128_lop3_inline", noinline=False
        )
        kernel_src = _kernel_cu_source_legacy_streamed_inplace(
            sbox_cuda, input_mode=args.input_mode
        )
        host_src = _host_bench_cu_source_legacy()
    elif args.kernel_mode in {"replacement_streamed", "replacement_streamed_tuned"}:
        sbox_cuda = _emit_sbox_inline_cuda(
            mapped, func_name="sbox_bp128_lop3_inline", noinline=False
        )
        kernel_src = _kernel_cu_source_replacement_streamed_inplace(sbox_cuda)
        host_src = _host_bench_cu_source_replacement()
    elif args.kernel_mode in {"replacement_coalesced", "replacement_coalesced_tuned"}:
        sbox_cuda = _emit_sbox_inline_cuda(
            mapped, func_name="sbox_bp128_lop3_inline", noinline=False
        )
        kernel_src = _kernel_cu_source_replacement_coalesced(sbox_cuda)
        host_src = _host_bench_cu_source_replacement(layout="plane-major")
    elif args.kernel_mode in {"replacement_coalesced4", "replacement_coalesced4_tuned"}:
        sbox_cuda = _emit_sbox_inline_cuda(
            mapped, func_name="sbox_bp128_lop3_inline", noinline=False
        )
        kernel_src = _kernel_cu_source_replacement_coalesced4(sbox_cuda)
        host_src = _host_bench_cu_source_replacement(layout="plane-major4")
    elif args.kernel_mode in {
        "replacement_coalesced4_paramrk",
        "replacement_coalesced4_paramrk_tuned",
    }:
        sbox_cuda = _emit_sbox_inline_cuda(
            mapped, func_name="sbox_bp128_lop3_inline", noinline=False
        )
        kernel_src = _kernel_cu_source_replacement_coalesced4_paramrk(sbox_cuda)
        host_src = _host_bench_cu_source_replacement_paramrk(layout="plane-major4")
    elif args.kernel_mode in {
        "replacement_coalesced4_paramrk_shared",
        "replacement_coalesced4_paramrk_shared_tuned",
    }:
        sbox_cuda = _emit_sbox_inline_cuda(
            mapped, func_name="sbox_bp128_lop3_inline", noinline=False
        )
        kernel_src = _kernel_cu_source_replacement_coalesced4_paramrk_shared(sbox_cuda)
        host_src = _host_bench_cu_source_replacement_paramrk_shared(
            layout="plane-major4"
        )
    elif args.kernel_mode in {
        "replacement_coalesced4_paramrk_soa",
        "replacement_coalesced4_paramrk_soa_tuned",
    }:
        sbox_cuda = _emit_sbox_inline_cuda(
            mapped, func_name="sbox_bp128_lop3_inline", noinline=False
        )
        kernel_src = _kernel_cu_source_replacement_coalesced4_paramrk_soa(sbox_cuda)
        host_src = _host_bench_cu_source_replacement_paramrk_soa(layout="plane-major4")
    elif args.kernel_mode in {
        "replacement_coalesced4_paramrk_soa_packed",
        "replacement_coalesced4_paramrk_soa_packed_tuned",
    }:
        sbox_cuda = _emit_sbox_inline_cuda(
            mapped, func_name="sbox_bp128_lop3_inline", noinline=False
        )
        kernel_src = _kernel_cu_source_replacement_coalesced4_paramrk_soa_packed(
            sbox_cuda
        )
        host_src = _host_bench_cu_source_replacement_paramrk_soa_packed(
            layout="plane-major4"
        )
    elif args.kernel_mode in {
        "replacement_coalesced4_masterkey_soa",
        "replacement_coalesced4_masterkey_soa_tuned",
    }:
        sbox_cuda = _emit_sbox_inline_cuda(
            mapped, func_name="sbox_bp128_lop3_inline", noinline=False
        )
        kernel_src = _kernel_cu_source_replacement_coalesced4_masterkey_soa(sbox_cuda)
        host_src = _host_bench_cu_source_replacement_masterkey_soa(
            layout="plane-major4"
        )
    else:
        sbox_cuda = _emit_sbox_inline_cuda(
            mapped, func_name="sbox_bp128_lop3_inline", noinline=False
        )
        kernel_src = _kernel_cu_source_replacement(sbox_cuda)
        host_src = _host_bench_cu_source_replacement()

    with tempfile.TemporaryDirectory(prefix="aes10_bp128_cuda_") as td:
        tdp = Path(td)
        kernel_cu = tdp / "aes10_kernel.cu"
        kernel_ptx = tdp / "aes10_kernel.ptx"
        out_prefix = Path(args.out)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        kernel_ptx_out = out_prefix.with_suffix(".kernel.ptx")
        cubin_out = out_prefix.with_suffix(".cubin")
        cubin = tdp / "aes10_kernel.cubin"
        host_cu = tdp / "bench_host.cu"
        host_bin = tdp / "bench_host"

        kernel_cu.write_text(kernel_src, encoding="utf-8")
        host_cu.write_text(host_src, encoding="utf-8")

        nvcc_codegen_flags = ["-O3", f"-arch={args.sm}"]
        if args.maxrregcount is not None and args.maxrregcount > 0:
            nvcc_codegen_flags.append(f"--maxrregcount={args.maxrregcount}")

        res = subprocess.run(
            [
                "nvcc",
                "-ptx",
                *nvcc_codegen_flags,
                "-o",
                str(kernel_ptx),
                str(kernel_cu),
            ],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            print(res.stderr.strip()[:4000])
            return res.returncode

        kernel_ptx_text = kernel_ptx.read_text(encoding="utf-8")
        kernel_ptx_out.write_text(kernel_ptx_text, encoding="utf-8")

        res = subprocess.run(
            [
                "nvcc",
                "-cubin",
                *nvcc_codegen_flags,
                "-o",
                str(cubin),
                str(kernel_cu),
            ],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            print(res.stderr.strip()[:4000])
            return res.returncode

        cubin_out.write_bytes(cubin.read_bytes())

        if args.ptxas_report:
            rep = subprocess.run(
                [
                    "ptxas",
                    f"-arch={args.sm}",
                    "-v",
                    str(kernel_ptx),
                    "-o",
                    str(tdp / "tmp.cubin"),
                ],
                capture_output=True,
                text=True,
            )
            if rep.stdout.strip():
                print(rep.stdout.strip())
            if rep.stderr.strip():
                print(rep.stderr.strip())

        res = subprocess.run(
            ["nvcc", "-O3", "-o", str(host_bin), str(host_cu), "-lcuda"],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            print(res.stderr.strip()[:4000])
            return res.returncode

        blocks = [args.block]
        if args.autotune_blocks.strip():
            blocks = [
                int(x.strip()) for x in args.autotune_blocks.split(",") if x.strip()
            ]
        elif args.kernel_mode in {
            "legacy_tuned",
            "legacy_streamed_tuned",
            "replacement_tuned",
            "replacement_coalesced_tuned",
            "replacement_coalesced4_tuned",
            "replacement_coalesced4_paramrk_tuned",
            "replacement_coalesced4_paramrk_shared_tuned",
            "replacement_coalesced4_paramrk_soa_tuned",
            "replacement_coalesced4_paramrk_soa_packed_tuned",
            "replacement_coalesced4_masterkey_soa_tuned",
            "replacement_streamed_tuned",
        }:
            # Pascal-friendly defaults for this kernel shape.
            blocks = [64, 128, 256]

        best_eval_b = -1.0
        best_block = blocks[0]
        for idx, block in enumerate(blocks):
            do_check = "1" if args.check and idx == 0 else "0"
            run = subprocess.run(
                [
                    str(host_bin),
                    str(cubin),
                    str(args.threads),
                    str(block),
                    str(args.reps),
                    do_check,
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            print(run.stdout.strip())
            if run.returncode != 0:
                if run.stderr:
                    print(run.stderr.strip()[:2000])
                if len(blocks) == 1:
                    return run.returncode
                print(f"skipping block={block} due to launch/runtime failure")
                continue
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)B evals/sec", run.stdout)
            if match:
                eval_b = float(match.group(1))
                if eval_b > best_eval_b:
                    best_eval_b = eval_b
                    best_block = block

        if best_eval_b < 0.0:
            print("no successful CUDA launches for the selected mode/config")
            return 3

        if len(blocks) > 1 and best_eval_b >= 0.0:
            print(
                f"autotune best block={best_block} " f"({best_eval_b:.3f}B evals/sec)"
            )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
