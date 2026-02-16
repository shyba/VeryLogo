from __future__ import annotations

import ctypes
import json
import random
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from stc.cuda_driver import Cuda, CudaError

KeySource = Literal["masterkey_soa", "expanded_rk_soa", "const_key"]
PostOp = Literal["store", "xor_accumulate"]
IoLayout = Literal["plane-major4", "bytes"]
OutputLayout = Literal["bitplanes", "words", "bytes"]

KEY_BITS_OPTIONS = (128, 192, 256)
CTR_GROUP_OPTIONS = (1, 2, 4)
KEY_SOURCE_OPTIONS: tuple[KeySource, ...] = (
    "masterkey_soa",
    "expanded_rk_soa",
    "const_key",
)
IO_LAYOUT_OPTIONS: tuple[IoLayout, ...] = ("plane-major4", "bytes")

_BYTES_TO_PLANES4_KERNEL = "stc_bytes_to_planes4_kernel"
_PLANES4_TO_BYTES_KERNEL = "stc_planes4_to_bytes_kernel"

_AES_NR_BY_KEY_BITS = {128: 10, 192: 12, 256: 14}
_AES_SBOX = (
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
)
_AES_RCON = (0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36, 0x6C)


@dataclass(frozen=True)
class Bp128VariantMetadata:
    variant_id: str
    kernel_name: str
    key_bits: int
    rounds: int
    ctr_group: int
    key_source: KeySource
    io_layout: IoLayout
    supported: bool
    notes: str


@dataclass(frozen=True)
class Bp128DispatchRequest:
    key_bits: int
    key_source: KeySource = "masterkey_soa"
    io_layout: IoLayout = "plane-major4"
    max_ctr_group: int = 4


def _aes_rounds(key_bits: int) -> int:
    if key_bits not in _AES_NR_BY_KEY_BITS:
        raise ValueError(f"unsupported key_bits={key_bits}")
    return _AES_NR_BY_KEY_BITS[key_bits]


def variant_id(
    key_bits: int, ctr_group: int, key_source: KeySource, io_layout: IoLayout
) -> str:
    return f"bp128_aes{key_bits}_g{ctr_group}_{key_source}_{io_layout}"


def kernel_name_from_variant_id(v_id: str) -> str:
    safe_id = re.sub(r"[^0-9A-Za-z_]", "_", v_id)
    return f"stc_{safe_id}"


def _support_notes(
    key_bits: int, key_source: KeySource, io_layout: IoLayout
) -> tuple[bool, str]:
    if io_layout == "bytes":
        return (
            True,
            "supported via byte<->plane-major4 converter kernels (tail-aware ABI)",
        )
    if io_layout != "plane-major4":
        return (False, f"unsupported io_layout={io_layout}")
    if key_source == "masterkey_soa" and key_bits != 128:
        return (
            True,
            "supported via host-expanded round-key fallback for AES-192/256",
        )
    if key_source == "const_key" and key_bits != 128:
        return (
            True,
            "supported via const expanded-round-key kernel for AES-192/256",
        )
    return (True, "supported")


def build_variant_metadata(
    key_bits: int, ctr_group: int, key_source: KeySource, io_layout: IoLayout
) -> Bp128VariantMetadata:
    rounds = _aes_rounds(key_bits)
    v_id = variant_id(key_bits, ctr_group, key_source, io_layout)
    supported, notes = _support_notes(key_bits, key_source, io_layout)
    return Bp128VariantMetadata(
        variant_id=v_id,
        kernel_name=kernel_name_from_variant_id(v_id),
        key_bits=key_bits,
        rounds=rounds,
        ctr_group=ctr_group,
        key_source=key_source,
        io_layout=io_layout,
        supported=supported,
        notes=notes,
    )


def all_variant_metadata() -> list[Bp128VariantMetadata]:
    out: list[Bp128VariantMetadata] = []
    for key_bits in KEY_BITS_OPTIONS:
        for ctr_group in CTR_GROUP_OPTIONS:
            for key_source in KEY_SOURCE_OPTIONS:
                for io_layout in IO_LAYOUT_OPTIONS:
                    out.append(
                        build_variant_metadata(
                            key_bits=key_bits,
                            ctr_group=ctr_group,
                            key_source=key_source,
                            io_layout=io_layout,
                        )
                    )
    return out


def family_manifest_json(indent: int = 2) -> str:
    payload = {
        "family": "bp128-fast",
        "variants": [
            {
                "variant_id": item.variant_id,
                "kernel_name": item.kernel_name,
                "key_bits": item.key_bits,
                "rounds": item.rounds,
                "ctr_group": item.ctr_group,
                "key_source": item.key_source,
                "io_layout": item.io_layout,
                "supported": item.supported,
                "notes": item.notes,
            }
            for item in all_variant_metadata()
        ],
        "abi": {
            "kernel_signature": (
                "masterkey_soa/expanded_rk_soa: "
                "void kernel(const uint32_t* in_ptr, uint32_t* out_ptr, "
                "const uint8_t* key_or_rk_soa, uint32_t n_threads, "
                "uint32_t out_len_bytes, uint32_t tail_bytes); "
                "const_key: "
                "void kernel(const uint32_t* in_ptr, uint32_t* out_ptr, "
                "uint32_t n_threads, uint32_t out_len_bytes, uint32_t tail_bytes)"
            ),
            "layout": (
                "input plane-major4 or bytes; output bitplanes/words/bytes via "
                "runtime output_layout selection"
            ),
            "n_threads": (
                "effective threads = logical_threads * ctr_group "
                "(host replicates per-thread key material per group)"
            ),
            "tail": (
                "out_len_bytes/tail_bytes enable tail-aware output handling; "
                "core plane-major kernels keep register behavior and treat them "
                "as ABI fields."
            ),
        },
    }
    return json.dumps(payload, indent=indent, sort_keys=True)


class Bp128Dispatcher:
    def __init__(self, variants: list[Bp128VariantMetadata] | None = None) -> None:
        self._variants = variants[:] if variants is not None else all_variant_metadata()

    def select(self, req: Bp128DispatchRequest) -> Bp128VariantMetadata:
        candidates = [item for item in self._variants if item.key_bits == req.key_bits]
        if not candidates:
            raise ValueError(f"no variants for key_bits={req.key_bits}")

        scored: list[tuple[int, Bp128VariantMetadata]] = []
        for item in candidates:
            if item.ctr_group > req.max_ctr_group:
                continue
            score = 0
            if item.supported:
                score += 1000
            if item.key_source == req.key_source:
                score += 300
            if item.io_layout == req.io_layout:
                score += 200
            if item.key_source == "masterkey_soa":
                score += 100
            score += item.ctr_group * 10
            scored.append((score, item))
        if not scored:
            raise ValueError("no candidate variants match dispatch request")
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[0][1]


def _require_nvcc() -> str:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc not found on PATH")
    return nvcc


def _load_bench_module():
    try:
        from scripts import bench_aes10_bp128_cuda as bench
    except Exception as exc:  # pragma: no cover - dependency/runtime environment
        raise RuntimeError(
            "failed to import scripts.bench_aes10_bp128_cuda; "
            "run with project venv (PYTHONPATH=. .venv/bin/python ...)"
        ) from exc
    return bench


@lru_cache(maxsize=1)
def _bp128_sbox_inline_cuda() -> str:
    bench = _load_bench_module()
    mapped, _lop3_count = bench._build_bp128_mapped()
    return bench._emit_sbox_inline_cuda(
        mapped, func_name="sbox_bp128_lop3_inline", noinline=False
    )


def _patch_round_count_expanded_source(src: str, rounds: int) -> str:
    if rounds == 10:
        return src
    src = src.replace(
        "for (int round = 1; round <= 9; round++) {",
        f"for (int round = 1; round <= {rounds - 1}; round++) {{",
    )
    src = src.replace("RK_AT(10,b,", f"RK_AT({rounds},b,")
    return src


def _rename_kernel(src: str, kernel_name: str) -> str:
    return src.replace("aes10_bp128_kernel", kernel_name)


def _patch_tail_abi(src: str) -> str:
    sig_keyed = re.compile(
        r'(extern "C" __global__ (?:__launch_bounds__\([^)]*\)\s+)?void [^\n]+\(\n'
        r"\s*const uint32_t\* __restrict__ in_ptr,\n"
        r"\s*uint32_t\* __restrict__ out_ptr,\n"
        r"\s*const uint8_t\* __restrict__ [A-Za-z0-9_]+,\n"
        r"\s*)uint32_t n_threads\n(\) \{)",
        re.MULTILINE,
    )
    src, keyed_count = sig_keyed.subn(
        r"\1uint32_t n_threads,\n"
        r"    uint32_t out_len_bytes,\n"
        r"    uint32_t tail_bytes\n\2",
        src,
        count=1,
    )

    sig_const = re.compile(
        r'(extern "C" __global__ (?:__launch_bounds__\([^)]*\)\s+)?void [^\n]+\(\n'
        r"\s*const uint32_t\* __restrict__ in_ptr,\n"
        r"\s*uint32_t\* __restrict__ out_ptr,\n"
        r"\s*)uint32_t n_threads\n(\) \{)",
        re.MULTILINE,
    )
    src, const_count = sig_const.subn(
        r"\1uint32_t n_threads,\n"
        r"    uint32_t out_len_bytes,\n"
        r"    uint32_t tail_bytes\n\2",
        src,
        count=1,
    )
    if keyed_count + const_count != 1:
        raise RuntimeError("failed to patch kernel signature for tail-aware ABI")

    marker = "    if (tid >= n_threads) return;\n"
    if marker not in src:
        raise RuntimeError("failed to patch kernel body for tail-aware ABI")
    src = src.replace(
        marker,
        marker + "    (void)out_len_bytes;\n" + "    (void)tail_bytes;\n",
        1,
    )
    return src


def _bytes_converter_cuda_source(post_op: PostOp) -> str:
    if post_op not in {"store", "xor_accumulate"}:
        raise ValueError(f"unsupported post_op={post_op}")
    xor_enabled = "1" if post_op == "xor_accumulate" else "0"
    return f"""
extern "C" __global__ void {_BYTES_TO_PLANES4_KERNEL}(
    const uint4* __restrict__ in_blocks4,
    uint4* __restrict__ out_planes4,
    uint32_t n_threads
) {{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t lane = gid & 31u;
    uint32_t tid = gid >> 5;
    if (tid >= n_threads) return;

    uint4 blk = in_blocks4[(size_t)tid * 32u + (size_t)lane];
    uint8_t by[16];
    by[0] = (uint8_t)(blk.x & 0xffu);
    by[1] = (uint8_t)((blk.x >> 8) & 0xffu);
    by[2] = (uint8_t)((blk.x >> 16) & 0xffu);
    by[3] = (uint8_t)((blk.x >> 24) & 0xffu);
    by[4] = (uint8_t)(blk.y & 0xffu);
    by[5] = (uint8_t)((blk.y >> 8) & 0xffu);
    by[6] = (uint8_t)((blk.y >> 16) & 0xffu);
    by[7] = (uint8_t)((blk.y >> 24) & 0xffu);
    by[8] = (uint8_t)(blk.z & 0xffu);
    by[9] = (uint8_t)((blk.z >> 8) & 0xffu);
    by[10] = (uint8_t)((blk.z >> 16) & 0xffu);
    by[11] = (uint8_t)((blk.z >> 24) & 0xffu);
    by[12] = (uint8_t)(blk.w & 0xffu);
    by[13] = (uint8_t)((blk.w >> 8) & 0xffu);
    by[14] = (uint8_t)((blk.w >> 16) & 0xffu);
    by[15] = (uint8_t)((blk.w >> 24) & 0xffu);

    #pragma unroll
    for (int b = 0; b < 16; b++) {{
        uint8_t pb = by[b];
        uint32_t m0 = __ballot_sync(0xffffffffu, (pb >> 0) & 1u);
        uint32_t m1 = __ballot_sync(0xffffffffu, (pb >> 1) & 1u);
        uint32_t m2 = __ballot_sync(0xffffffffu, (pb >> 2) & 1u);
        uint32_t m3 = __ballot_sync(0xffffffffu, (pb >> 3) & 1u);
        uint32_t m4 = __ballot_sync(0xffffffffu, (pb >> 4) & 1u);
        uint32_t m5 = __ballot_sync(0xffffffffu, (pb >> 5) & 1u);
        uint32_t m6 = __ballot_sync(0xffffffffu, (pb >> 6) & 1u);
        uint32_t m7 = __ballot_sync(0xffffffffu, (pb >> 7) & 1u);
        if (lane == 0u) {{
            size_t idx0 = ((size_t)b * 2u + 0u) * (size_t)n_threads + (size_t)tid;
            size_t idx1 = ((size_t)b * 2u + 1u) * (size_t)n_threads + (size_t)tid;
            out_planes4[idx0] = make_uint4(m0, m1, m2, m3);
            out_planes4[idx1] = make_uint4(m4, m5, m6, m7);
        }}
    }}
}}

extern "C" __global__ void {_PLANES4_TO_BYTES_KERNEL}(
    const uint4* __restrict__ in_planes4,
    uint4* __restrict__ out_blocks4,
    uint32_t n_threads,
    uint32_t out_len_bytes,
    uint32_t tail_bytes
) {{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t lane = gid & 31u;
    uint32_t tid = gid >> 5;
    if (tid >= n_threads) return;

    uint64_t default_len = (uint64_t)n_threads * 32ull * 16ull;
    uint64_t total_bytes = out_len_bytes ? (uint64_t)out_len_bytes : default_len;
    (void)tail_bytes;

    uint8_t by[16];
    #pragma unroll
    for (int b = 0; b < 16; b++) {{
        uint32_t m0 = 0u, m1 = 0u, m2 = 0u, m3 = 0u;
        uint32_t m4 = 0u, m5 = 0u, m6 = 0u, m7 = 0u;
        if (lane == 0u) {{
            size_t idx0 = ((size_t)b * 2u + 0u) * (size_t)n_threads + (size_t)tid;
            size_t idx1 = ((size_t)b * 2u + 1u) * (size_t)n_threads + (size_t)tid;
            uint4 v0 = in_planes4[idx0];
            uint4 v1 = in_planes4[idx1];
            m0 = v0.x;
            m1 = v0.y;
            m2 = v0.z;
            m3 = v0.w;
            m4 = v1.x;
            m5 = v1.y;
            m6 = v1.z;
            m7 = v1.w;
        }}
        m0 = __shfl_sync(0xffffffffu, m0, 0);
        m1 = __shfl_sync(0xffffffffu, m1, 0);
        m2 = __shfl_sync(0xffffffffu, m2, 0);
        m3 = __shfl_sync(0xffffffffu, m3, 0);
        m4 = __shfl_sync(0xffffffffu, m4, 0);
        m5 = __shfl_sync(0xffffffffu, m5, 0);
        m6 = __shfl_sync(0xffffffffu, m6, 0);
        m7 = __shfl_sync(0xffffffffu, m7, 0);

        uint8_t pb = 0u;
        pb |= (uint8_t)(((m0 >> lane) & 1u) << 0);
        pb |= (uint8_t)(((m1 >> lane) & 1u) << 1);
        pb |= (uint8_t)(((m2 >> lane) & 1u) << 2);
        pb |= (uint8_t)(((m3 >> lane) & 1u) << 3);
        pb |= (uint8_t)(((m4 >> lane) & 1u) << 4);
        pb |= (uint8_t)(((m5 >> lane) & 1u) << 5);
        pb |= (uint8_t)(((m6 >> lane) & 1u) << 6);
        pb |= (uint8_t)(((m7 >> lane) & 1u) << 7);
        by[b] = pb;
    }}

    uint64_t block_idx = (uint64_t)tid * 32ull + (uint64_t)lane;
    uint64_t block_byte_base = block_idx * 16ull;
    if (block_byte_base >= total_bytes) {{
        return;
    }}
    uint32_t valid_bytes = (block_byte_base + 16ull <= total_bytes)
        ? 16u
        : (uint32_t)(total_bytes - block_byte_base);

    size_t out_idx = (size_t)block_idx;
    uint4 outv = make_uint4(
        ((uint32_t)by[0]) | ((uint32_t)by[1] << 8) | ((uint32_t)by[2] << 16) | ((uint32_t)by[3] << 24),
        ((uint32_t)by[4]) | ((uint32_t)by[5] << 8) | ((uint32_t)by[6] << 16) | ((uint32_t)by[7] << 24),
        ((uint32_t)by[8]) | ((uint32_t)by[9] << 8) | ((uint32_t)by[10] << 16) | ((uint32_t)by[11] << 24),
        ((uint32_t)by[12]) | ((uint32_t)by[13] << 8) | ((uint32_t)by[14] << 16) | ((uint32_t)by[15] << 24)
    );

    if (valid_bytes == 16u) {{
        if ({xor_enabled}) {{
            uint4 prev = out_blocks4[out_idx];
            outv.x ^= prev.x;
            outv.y ^= prev.y;
            outv.z ^= prev.z;
            outv.w ^= prev.w;
        }}
        out_blocks4[out_idx] = outv;
        return;
    }}

    uint4 prev = out_blocks4[out_idx];
    uint8_t prev_by[16];
    prev_by[0] = (uint8_t)(prev.x & 0xffu);
    prev_by[1] = (uint8_t)((prev.x >> 8) & 0xffu);
    prev_by[2] = (uint8_t)((prev.x >> 16) & 0xffu);
    prev_by[3] = (uint8_t)((prev.x >> 24) & 0xffu);
    prev_by[4] = (uint8_t)(prev.y & 0xffu);
    prev_by[5] = (uint8_t)((prev.y >> 8) & 0xffu);
    prev_by[6] = (uint8_t)((prev.y >> 16) & 0xffu);
    prev_by[7] = (uint8_t)((prev.y >> 24) & 0xffu);
    prev_by[8] = (uint8_t)(prev.z & 0xffu);
    prev_by[9] = (uint8_t)((prev.z >> 8) & 0xffu);
    prev_by[10] = (uint8_t)((prev.z >> 16) & 0xffu);
    prev_by[11] = (uint8_t)((prev.z >> 24) & 0xffu);
    prev_by[12] = (uint8_t)(prev.w & 0xffu);
    prev_by[13] = (uint8_t)((prev.w >> 8) & 0xffu);
    prev_by[14] = (uint8_t)((prev.w >> 16) & 0xffu);
    prev_by[15] = (uint8_t)((prev.w >> 24) & 0xffu);

    for (uint32_t i = 0; i < valid_bytes; i++) {{
        if ({xor_enabled}) {{
            prev_by[i] ^= by[i];
        }} else {{
            prev_by[i] = by[i];
        }}
    }}
    out_blocks4[out_idx] = make_uint4(
        ((uint32_t)prev_by[0]) | ((uint32_t)prev_by[1] << 8) | ((uint32_t)prev_by[2] << 16) | ((uint32_t)prev_by[3] << 24),
        ((uint32_t)prev_by[4]) | ((uint32_t)prev_by[5] << 8) | ((uint32_t)prev_by[6] << 16) | ((uint32_t)prev_by[7] << 24),
        ((uint32_t)prev_by[8]) | ((uint32_t)prev_by[9] << 8) | ((uint32_t)prev_by[10] << 16) | ((uint32_t)prev_by[11] << 24),
        ((uint32_t)prev_by[12]) | ((uint32_t)prev_by[13] << 8) | ((uint32_t)prev_by[14] << 16) | ((uint32_t)prev_by[15] << 24)
    );
}}
"""


def _core_kernel_name(meta: Bp128VariantMetadata) -> str:
    return meta.kernel_name


def _patch_const_key_from_expanded_source(src: str, rounds: int) -> str:
    sig_old = (
        'extern "C" __global__ void aes10_bp128_kernel(\n'
        "    const uint32_t* __restrict__ in_ptr,\n"
        "    uint32_t* __restrict__ out_ptr,\n"
        "    const uint8_t* __restrict__ rk_bytes,\n"
        "    uint32_t n_threads\n"
        ") {"
    )
    sig_new = (
        'extern "C" __global__ void aes10_bp128_kernel(\n'
        "    const uint32_t* __restrict__ in_ptr,\n"
        "    uint32_t* __restrict__ out_ptr,\n"
        "    uint32_t n_threads\n"
        ") {"
    )
    if sig_old not in src:
        raise RuntimeError("failed to patch expanded signature for const_key")
    src = src.replace(sig_old, sig_new)

    macro_old = (
        "    #define RK_BYTE_AT(r,b) rk_bytes[((((size_t)(r) * 16u + (size_t)(b)) * "
        "threads) + t)]\n"
    )
    macro_new = "    #define RK_BYTE_AT(r,b) RK_BYTES_CONST[(r)][(b)]\n"
    if macro_old not in src:
        raise RuntimeError("failed to patch RK_BYTE_AT macro for const_key")
    src = src.replace(macro_old, macro_new)

    decl = f"__device__ __constant__ uint8_t RK_BYTES_CONST[{rounds + 1}][16];\n\n"
    marker = 'extern "C" __global__ void aes10_bp128_kernel('
    pos = src.find(marker)
    if pos < 0:
        raise RuntimeError("failed to inject RK_BYTES_CONST declaration")
    src = src[:pos] + decl + src[pos:]
    return src


def _optimize_packed_rk_loads(src: str) -> str:
    src = re.sub(
        r"(^\s*#define\s+RK_BYTE_AT\(r,b\)\s+)rk_bytes\[(.+)\]$",
        r"\1__ldg(&rk_bytes[\2])",
        src,
        flags=re.MULTILINE,
    )

    if "#define RK_MASK(k,bit)" not in src:
        src, count = re.subn(
            r"(^\s*#define\s+RK_AT\([^\n]+\)\n)",
            (
                r"\1"
                "    #define RK_MASK(k,bit) ((((uint32_t)(k) >> (bit)) & 1u) ? "
                "0xffffffffu : 0u)\n"
            ),
            src,
            count=1,
            flags=re.MULTILINE,
        )
        if count == 0:
            raise RuntimeError("failed to insert RK_MASK macro")

    old_init = (
        "        st[b][0] = v0.x ^ RK_AT(0,b,0);\n"
        "        st[b][1] = v0.y ^ RK_AT(0,b,1);\n"
        "        st[b][2] = v0.z ^ RK_AT(0,b,2);\n"
        "        st[b][3] = v0.w ^ RK_AT(0,b,3);\n"
        "        st[b][4] = v1.x ^ RK_AT(0,b,4);\n"
        "        st[b][5] = v1.y ^ RK_AT(0,b,5);\n"
        "        st[b][6] = v1.z ^ RK_AT(0,b,6);\n"
        "        st[b][7] = v1.w ^ RK_AT(0,b,7);"
    )
    new_init = (
        "        uint8_t rk0 = RK_BYTE_AT(0,b);\n"
        "        st[b][0] = v0.x ^ RK_MASK(rk0,0);\n"
        "        st[b][1] = v0.y ^ RK_MASK(rk0,1);\n"
        "        st[b][2] = v0.z ^ RK_MASK(rk0,2);\n"
        "        st[b][3] = v0.w ^ RK_MASK(rk0,3);\n"
        "        st[b][4] = v1.x ^ RK_MASK(rk0,4);\n"
        "        st[b][5] = v1.y ^ RK_MASK(rk0,5);\n"
        "        st[b][6] = v1.z ^ RK_MASK(rk0,6);\n"
        "        st[b][7] = v1.w ^ RK_MASK(rk0,7);"
    )
    src = src.replace(old_init, new_init)

    old_round = (
        "            #pragma unroll\n"
        "            for (int bit = 0; bit < 8; bit++) {\n"
        "                st[b][bit] = mc[b][bit] ^ RK_AT(round,b,bit);\n"
        "            }"
    )
    new_round = (
        "            uint8_t rkb = RK_BYTE_AT(round,b);\n"
        "            #pragma unroll\n"
        "            for (int bit = 0; bit < 8; bit++) {\n"
        "                st[b][bit] = mc[b][bit] ^ RK_MASK(rkb,bit);\n"
        "            }"
    )
    src = src.replace(old_round, new_round)

    final_pattern = re.compile(
        r"        o0\.x = sr\[b\]\[0\] \^ RK_AT\((\d+),b,0\);\n"
        r"        o0\.y = sr\[b\]\[1\] \^ RK_AT\(\1,b,1\);\n"
        r"        o0\.z = sr\[b\]\[2\] \^ RK_AT\(\1,b,2\);\n"
        r"        o0\.w = sr\[b\]\[3\] \^ RK_AT\(\1,b,3\);\n"
        r"        o1\.x = sr\[b\]\[4\] \^ RK_AT\(\1,b,4\);\n"
        r"        o1\.y = sr\[b\]\[5\] \^ RK_AT\(\1,b,5\);\n"
        r"        o1\.z = sr\[b\]\[6\] \^ RK_AT\(\1,b,6\);\n"
        r"        o1\.w = sr\[b\]\[7\] \^ RK_AT\(\1,b,7\);"
    )
    src = final_pattern.sub(
        (
            "        uint8_t rkf = RK_BYTE_AT(\\1,b);\n"
            "        o0.x = sr[b][0] ^ RK_MASK(rkf,0);\n"
            "        o0.y = sr[b][1] ^ RK_MASK(rkf,1);\n"
            "        o0.z = sr[b][2] ^ RK_MASK(rkf,2);\n"
            "        o0.w = sr[b][3] ^ RK_MASK(rkf,3);\n"
            "        o1.x = sr[b][4] ^ RK_MASK(rkf,4);\n"
            "        o1.y = sr[b][5] ^ RK_MASK(rkf,5);\n"
            "        o1.z = sr[b][6] ^ RK_MASK(rkf,6);\n"
            "        o1.w = sr[b][7] ^ RK_MASK(rkf,7);"
        ),
        src,
        count=1,
    )

    src = src.replace(
        "    #undef RK_AT\n    #undef RK_BYTE_AT\n",
        "    #undef RK_AT\n    #undef RK_MASK\n    #undef RK_BYTE_AT\n",
    )
    return src


def _apply_post_op(src: str, post_op: PostOp) -> str:
    if post_op == "store":
        return src
    if post_op != "xor_accumulate":
        raise ValueError(f"unsupported post_op={post_op}")

    old_candidates = [
        "        out4[idx0] = o0;\n        out4[idx1] = o1;",
        "    out4[idx0] = o0;\n    out4[idx1] = o1;",
    ]
    new = (
        "        uint4 prev0 = out4[idx0];\n"
        "        uint4 prev1 = out4[idx1];\n"
        "        o0.x ^= prev0.x;\n"
        "        o0.y ^= prev0.y;\n"
        "        o0.z ^= prev0.z;\n"
        "        o0.w ^= prev0.w;\n"
        "        o1.x ^= prev1.x;\n"
        "        o1.y ^= prev1.y;\n"
        "        o1.z ^= prev1.z;\n"
        "        o1.w ^= prev1.w;\n"
        "        out4[idx0] = o0;\n"
        "        out4[idx1] = o1;"
    )
    for old in old_candidates:
        if old in src:
            return src.replace(old, new)
    raise RuntimeError("failed to patch output stores for xor_accumulate post-op")


def _generate_plane_major4_core_source(
    meta: Bp128VariantMetadata, kernel_name: str, post_op: PostOp
) -> str:
    bench = _load_bench_module()
    sbox_inline_cuda = _bp128_sbox_inline_cuda()
    if meta.key_source == "masterkey_soa":
        if meta.key_bits == 128:
            src = bench._kernel_cu_source_replacement_coalesced4_masterkey_soa_streamed(
                sbox_inline_cuda,
                xor_accumulate=False,
                launch_bounds=(128, 3),
            )
        else:
            src = bench._kernel_cu_source_replacement_coalesced4_paramrk_soa_packed(
                sbox_inline_cuda
            )
            src = _patch_round_count_expanded_source(src, meta.rounds)
            src = _optimize_packed_rk_loads(src)
    elif meta.key_source == "expanded_rk_soa":
        src = bench._kernel_cu_source_replacement_coalesced4_paramrk_soa_packed(
            sbox_inline_cuda
        )
        src = _patch_round_count_expanded_source(src, meta.rounds)
        src = _optimize_packed_rk_loads(src)
    else:
        if meta.key_bits == 128:
            src = bench._kernel_cu_source_replacement_coalesced4(sbox_inline_cuda)
        else:
            src = bench._kernel_cu_source_replacement_coalesced4_paramrk_soa_packed(
                sbox_inline_cuda
            )
            src = _patch_round_count_expanded_source(src, meta.rounds)
            src = _patch_const_key_from_expanded_source(src, meta.rounds)
            src = _optimize_packed_rk_loads(src)
    src = _rename_kernel(src, kernel_name)
    src = _patch_tail_abi(src)
    src = _apply_post_op(src, post_op)
    return src


def generate_cuda_source(meta: Bp128VariantMetadata, post_op: PostOp = "store") -> str:
    if not meta.supported:
        raise ValueError(f"unsupported variant: {meta.variant_id} ({meta.notes})")

    if meta.io_layout == "plane-major4":
        return _generate_plane_major4_core_source(
            meta=meta,
            kernel_name=meta.kernel_name,
            post_op=post_op,
        )

    if meta.io_layout != "bytes":
        raise ValueError(f"unsupported io_layout={meta.io_layout}")

    plane_meta = build_variant_metadata(
        key_bits=meta.key_bits,
        ctr_group=meta.ctr_group,
        key_source=meta.key_source,
        io_layout="plane-major4",
    )
    core_name = meta.kernel_name
    core_src = _generate_plane_major4_core_source(
        meta=plane_meta,
        kernel_name=core_name,
        post_op="store",
    )
    return core_src + "\n" + _bytes_converter_cuda_source(post_op=post_op)


@lru_cache(maxsize=64)
def compile_variant_to_ptx(
    meta: Bp128VariantMetadata, sm: str = "sm_61", post_op: PostOp = "store"
) -> str:
    nvcc = _require_nvcc()
    source = generate_cuda_source(meta, post_op=post_op)
    with tempfile.TemporaryDirectory(prefix="bp128_variant_") as td:
        work_dir = Path(td)
        cu_path = work_dir / "kernel.cu"
        ptx_path = work_dir / "kernel.ptx"
        cu_path.write_text(source, encoding="utf-8")
        cmd = [
            nvcc,
            "-ptx",
            "-O3",
            "-std=c++17",
            f"-arch={sm}",
            str(cu_path),
            "-o",
            str(ptx_path),
        ]
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError(
                "nvcc failed generating PTX\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{run.stdout}\nSTDERR:\n{run.stderr}"
            )
        return ptx_path.read_text(encoding="utf-8")


@lru_cache(maxsize=64)
def compile_variant_to_cubin(
    meta: Bp128VariantMetadata, sm: str = "sm_61", post_op: PostOp = "store"
) -> bytes:
    nvcc = _require_nvcc()
    source = generate_cuda_source(meta, post_op=post_op)
    with tempfile.TemporaryDirectory(prefix="bp128_variant_") as td:
        work_dir = Path(td)
        cu_path = work_dir / "kernel.cu"
        cubin_path = work_dir / "kernel.cubin"
        cu_path.write_text(source, encoding="utf-8")
        cmd = [
            nvcc,
            "-cubin",
            "-O3",
            "-std=c++17",
            f"-arch={sm}",
            str(cu_path),
            "-o",
            str(cubin_path),
        ]
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError(
                "nvcc failed generating CUBIN\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{run.stdout}\nSTDERR:\n{run.stderr}"
            )
        return cubin_path.read_bytes()


def _expand_round_keys(master_key: bytes, key_bits: int) -> bytes:
    rounds = _aes_rounds(key_bits)
    nk = key_bits // 32
    total_words = 4 * (rounds + 1)
    if len(master_key) != key_bits // 8:
        raise ValueError(f"master key length mismatch for key_bits={key_bits}")

    words = [0] * total_words
    for i in range(nk):
        j = i * 4
        words[i] = (
            (master_key[j + 0] << 24)
            | (master_key[j + 1] << 16)
            | (master_key[j + 2] << 8)
            | master_key[j + 3]
        )

    for i in range(nk, total_words):
        temp = words[i - 1]
        if i % nk == 0:
            temp = ((temp << 8) & 0xFFFFFFFF) | (temp >> 24)
            temp = (
                (_AES_SBOX[(temp >> 24) & 0xFF] << 24)
                | (_AES_SBOX[(temp >> 16) & 0xFF] << 16)
                | (_AES_SBOX[(temp >> 8) & 0xFF] << 8)
                | (_AES_SBOX[temp & 0xFF])
            )
            temp ^= _AES_RCON[i // nk] << 24
        elif nk > 6 and (i % nk) == 4:
            temp = (
                (_AES_SBOX[(temp >> 24) & 0xFF] << 24)
                | (_AES_SBOX[(temp >> 16) & 0xFF] << 16)
                | (_AES_SBOX[(temp >> 8) & 0xFF] << 8)
                | (_AES_SBOX[temp & 0xFF])
            )
        words[i] = words[i - nk] ^ temp

    out = bytearray((rounds + 1) * 16)
    for i, word in enumerate(words):
        out[i * 4 + 0] = (word >> 24) & 0xFF
        out[i * 4 + 1] = (word >> 16) & 0xFF
        out[i * 4 + 2] = (word >> 8) & 0xFF
        out[i * 4 + 3] = word & 0xFF
    return bytes(out)


def _pack_key_soa(
    key_bytes_per_thread: list[bytes], byte_len: int, ctr_group: int
) -> ctypes.Array[ctypes.c_uint8]:
    expanded: list[bytes] = []
    for key in key_bytes_per_thread:
        for _ in range(ctr_group):
            expanded.append(key)
    threads_eff = len(expanded)
    out = (ctypes.c_uint8 * (threads_eff * byte_len))()
    for byte_idx in range(byte_len):
        base = byte_idx * threads_eff
        for tid in range(threads_eff):
            out[base + tid] = expanded[tid][byte_idx]
    return out


def make_key_material_soa(
    meta: Bp128VariantMetadata,
    logical_threads: int,
    seed: int = 0xBEEFC0DE,
    shared_key: bool = True,
) -> ctypes.Array[ctypes.c_uint8]:
    if meta.key_source == "const_key":
        return (ctypes.c_uint8 * 0)()
    rng = random.Random(seed)
    key_len = meta.key_bits // 8
    if meta.key_source == "masterkey_soa":
        if shared_key:
            key0 = bytes(rng.getrandbits(8) for _ in range(key_len))
            master_keys = [key0 for _ in range(logical_threads)]
        else:
            master_keys = [
                bytes(rng.getrandbits(8) for _ in range(key_len))
                for _ in range(logical_threads)
            ]
        if meta.key_bits == 128:
            return _pack_key_soa(master_keys, byte_len=16, ctr_group=meta.ctr_group)
        round_keys = [_expand_round_keys(mk, meta.key_bits) for mk in master_keys]
        return _pack_key_soa(
            round_keys,
            byte_len=(meta.rounds + 1) * 16,
            ctr_group=meta.ctr_group,
        )
    if shared_key:
        key0 = bytes(rng.getrandbits(8) for _ in range(key_len))
        master_keys = [key0 for _ in range(logical_threads)]
    else:
        master_keys = [
            bytes(rng.getrandbits(8) for _ in range(key_len))
            for _ in range(logical_threads)
        ]
    round_keys = [_expand_round_keys(mk, meta.key_bits) for mk in master_keys]
    return _pack_key_soa(
        round_keys,
        byte_len=(meta.rounds + 1) * 16,
        ctr_group=meta.ctr_group,
    )


def _aes_shift_rows(state: list[int]) -> list[int]:
    out = [0] * 16
    out[0] = state[0]
    out[1] = state[5]
    out[2] = state[10]
    out[3] = state[15]
    out[4] = state[4]
    out[5] = state[9]
    out[6] = state[14]
    out[7] = state[3]
    out[8] = state[8]
    out[9] = state[13]
    out[10] = state[2]
    out[11] = state[7]
    out[12] = state[12]
    out[13] = state[1]
    out[14] = state[6]
    out[15] = state[11]
    return out


def _aes_xtime(x: int) -> int:
    x &= 0xFF
    x2 = x << 1
    if x & 0x80:
        x2 ^= 0x1B
    return x2 & 0xFF


def _aes_mul3(x: int) -> int:
    return _aes_xtime(x) ^ (x & 0xFF)


def _aes_mix_columns(state: list[int]) -> list[int]:
    out = state[:]
    for col in range(4):
        i = col * 4
        s0 = state[i + 0]
        s1 = state[i + 1]
        s2 = state[i + 2]
        s3 = state[i + 3]
        out[i + 0] = _aes_xtime(s0) ^ _aes_mul3(s1) ^ s2 ^ s3
        out[i + 1] = s0 ^ _aes_xtime(s1) ^ _aes_mul3(s2) ^ s3
        out[i + 2] = s0 ^ s1 ^ _aes_xtime(s2) ^ _aes_mul3(s3)
        out[i + 3] = _aes_mul3(s0) ^ s1 ^ s2 ^ _aes_xtime(s3)
        out[i + 0] &= 0xFF
        out[i + 1] &= 0xFF
        out[i + 2] &= 0xFF
        out[i + 3] &= 0xFF
    return out


def _aes_add_round_key(state: list[int], rk_bytes: bytes, round_idx: int) -> list[int]:
    base = round_idx * 16
    return [(state[i] ^ rk_bytes[base + i]) & 0xFF for i in range(16)]


def _aes_encrypt_block_ref(block16: bytes, key: bytes, key_bits: int) -> bytes:
    if len(block16) != 16:
        raise ValueError("AES block must be 16 bytes")
    rk_bytes = _expand_round_keys(key, key_bits)
    rounds = _aes_rounds(key_bits)

    state = [x for x in block16]
    state = _aes_add_round_key(state, rk_bytes, 0)
    for round_idx in range(1, rounds):
        state = [_AES_SBOX[x] for x in state]
        state = _aes_shift_rows(state)
        state = _aes_mix_columns(state)
        state = _aes_add_round_key(state, rk_bytes, round_idx)
    state = [_AES_SBOX[x] for x in state]
    state = _aes_shift_rows(state)
    state = _aes_add_round_key(state, rk_bytes, rounds)
    return bytes(state)


def _seed_plaintext_planes4(
    words: ctypes.Array[ctypes.c_uint32], threads_eff: int, block16: bytes
) -> None:
    if len(block16) != 16:
        raise ValueError("seed block must be 16 bytes")
    for tid in range(threads_eff):
        for b in range(16):
            pb = block16[b]
            for bit in range(8):
                group = b * 2 + (bit >> 2)
                lane = bit & 3
                word_idx = (group * threads_eff + tid) * 4 + lane
                words[word_idx] = 0xFFFFFFFF if ((pb >> bit) & 1) else 0


def _decode_first_lane_planes4(
    words: ctypes.Array[ctypes.c_uint32], threads_eff: int
) -> bytes:
    out = bytearray(16)
    for b in range(16):
        v = 0
        for bit in range(8):
            group = b * 2 + (bit >> 2)
            lane = bit & 3
            word_idx = (group * threads_eff + 0) * 4 + lane
            bitval = words[word_idx] & 1
            v |= bitval << bit
        out[b] = v
    return bytes(out)


def _seed_plaintext_bytes(
    data: ctypes.Array[ctypes.c_uint8], blocks_total: int, block16: bytes
) -> None:
    if len(block16) != 16:
        raise ValueError("seed block must be 16 bytes")
    for blk in range(blocks_total):
        base = blk * 16
        for i in range(16):
            data[base + i] = block16[i]


def _decode_first_block_bytes(data: ctypes.Array[ctypes.c_uint8]) -> bytes:
    return bytes(int(data[i]) for i in range(16))


def _seed_bitplanes_words(
    words: ctypes.Array[ctypes.c_uint32], threads_eff: int, block16: bytes
) -> None:
    if len(block16) != 16:
        raise ValueError("seed block must be 16 bytes")
    for tid in range(threads_eff):
        for b in range(16):
            pb = block16[b]
            for bit in range(8):
                plane = b * 8 + bit
                words[plane * threads_eff + tid] = (
                    0xFFFFFFFF if ((pb >> bit) & 1) else 0
                )


def _decode_first_lane_words(
    words: ctypes.Array[ctypes.c_uint32], threads_eff: int
) -> bytes:
    out = bytearray(16)
    for b in range(16):
        v = 0
        for bit in range(8):
            plane = b * 8 + bit
            bitval = words[plane * threads_eff + 0] & 1
            v |= bitval << bit
        out[b] = v
    return bytes(out)


@lru_cache(maxsize=8)
def _compile_coalesced_words_cubin(sm: str, post_op: PostOp) -> bytes:
    bench = _load_bench_module()
    mapped, _lop3_count = bench._build_bp128_mapped()
    sbox_cuda = bench._emit_sbox_inline_cuda(
        mapped, func_name="sbox_bp128_lop3_inline", noinline=False
    )
    src = bench._kernel_cu_source_replacement_coalesced(sbox_cuda)
    src = _patch_tail_abi(src)
    if post_op == "xor_accumulate":
        old = "            out_ptr[plane * threads + t] = sr[b][bit] ^ RK_BITS[10][b][bit];"
        new = (
            "            uint32_t outv = sr[b][bit] ^ RK_BITS[10][b][bit];\n"
            "            out_ptr[plane * threads + t] ^= outv;"
        )
        if old not in src:
            raise RuntimeError(
                "failed to patch coalesced words store for xor_accumulate"
            )
        src = src.replace(old, new)
    elif post_op != "store":
        raise ValueError(f"unsupported post_op={post_op}")
    src = _rename_kernel(src, "stc_bp128_words_kernel")
    with tempfile.TemporaryDirectory(prefix="bp128_words_cubin_") as td:
        work_dir = Path(td)
        cu_path = work_dir / "kernel.cu"
        cubin_path = work_dir / "kernel.cubin"
        cu_path.write_text(src, encoding="utf-8")
        cmd = [
            _require_nvcc(),
            "-cubin",
            "-O3",
            "-std=c++17",
            f"-arch={sm}",
            str(cu_path),
            "-o",
            str(cubin_path),
        ]
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError(
                "nvcc failed for coalesced words kernel\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{run.stdout}\nSTDERR:\n{run.stderr}"
            )
        return cubin_path.read_bytes()


def _module_get_global_ptr(
    cuda: Cuda, mod: ctypes.c_void_p, symbol: str
) -> tuple[int, int]:
    fn_name = "cuModuleGetGlobal_v2"
    if not hasattr(cuda.lib, fn_name):
        fn_name = "cuModuleGetGlobal"
    fn = getattr(cuda.lib, fn_name)
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    fn.restype = ctypes.c_int

    dptr = ctypes.c_uint64()
    nbytes = ctypes.c_size_t()
    code = fn(ctypes.byref(dptr), ctypes.byref(nbytes), mod, symbol.encode("utf-8"))
    if code != 0:
        raise CudaError(f"cuModuleGetGlobal({symbol}) failed", code=code)
    return int(dptr.value), int(nbytes.value)


def _build_const_rk_bits_u32(key_bits: int = 128) -> ctypes.Array[ctypes.c_uint32]:
    if key_bits != 128:
        raise ValueError("RK_BITS constant path only supports AES-128")
    key = bytes(range(16))
    rk = _expand_round_keys(key, key_bits)
    out = (ctypes.c_uint32 * (11 * 16 * 8))()
    idx = 0
    for r in range(11):
        for b in range(16):
            kv = rk[r * 16 + b]
            for bit in range(8):
                out[idx] = 0xFFFFFFFF if ((kv >> bit) & 1) else 0
                idx += 1
    return out


def _upload_const_rk_bits(
    cuda: Cuda, mod: ctypes.c_void_p, key_bits: int = 128
) -> None:
    dptr, nbytes = _module_get_global_ptr(cuda, mod, "RK_BITS")
    rk_bits = _build_const_rk_bits_u32(key_bits=key_bits)
    rk_nbytes = ctypes.sizeof(rk_bits)
    if nbytes < rk_nbytes:
        raise RuntimeError(f"RK_BITS symbol too small: {nbytes} < {rk_nbytes}")
    cuda.memcpy_htod(dptr, rk_bits, rk_nbytes)


def _upload_const_rk_bytes(cuda: Cuda, mod: ctypes.c_void_p, key_bits: int) -> None:
    dptr, nbytes = _module_get_global_ptr(cuda, mod, "RK_BYTES_CONST")
    key = bytes(range(key_bits // 8))
    rk = _expand_round_keys(key, key_bits)
    rk_arr = (ctypes.c_uint8 * len(rk))()
    for i, byte in enumerate(rk):
        rk_arr[i] = byte
    rk_nbytes = ctypes.sizeof(rk_arr)
    if nbytes < rk_nbytes:
        raise RuntimeError(f"RK_BYTES_CONST symbol too small: {nbytes} < {rk_nbytes}")
    cuda.memcpy_htod(dptr, rk_arr, rk_nbytes)


def _upload_const_keys(
    cuda: Cuda, mod: ctypes.c_void_p, meta: Bp128VariantMetadata
) -> None:
    if meta.key_source != "const_key":
        return
    if meta.key_bits == 128:
        _upload_const_rk_bits(cuda, mod, key_bits=128)
    else:
        _upload_const_rk_bytes(cuda, mod, key_bits=meta.key_bits)


def _build_check_key_material(
    meta: Bp128VariantMetadata, threads: int
) -> tuple[bytes, ctypes.Array[ctypes.c_uint8]]:
    key_bytes = bytes(range(meta.key_bits // 8))
    if meta.key_source == "const_key":
        return key_bytes, (ctypes.c_uint8 * 0)()
    if meta.key_source == "masterkey_soa":
        if meta.key_bits == 128:
            key_soa = _pack_key_soa(
                [key_bytes for _ in range(threads)],
                byte_len=len(key_bytes),
                ctr_group=meta.ctr_group,
            )
            return key_bytes, key_soa
        expanded = _expand_round_keys(key_bytes, meta.key_bits)
        key_soa = _pack_key_soa(
            [expanded for _ in range(threads)],
            byte_len=len(expanded),
            ctr_group=meta.ctr_group,
        )
        return key_bytes, key_soa
    expanded = _expand_round_keys(key_bytes, meta.key_bits)
    key_soa = _pack_key_soa(
        [expanded for _ in range(threads)],
        byte_len=len(expanded),
        ctr_group=meta.ctr_group,
    )
    return key_bytes, key_soa


def check_variant_correctness(
    meta: Bp128VariantMetadata,
    threads: int = 256,
    block: int = 64,
    sm: str = "sm_61",
    post_op: PostOp = "store",
    output_layout: OutputLayout | None = None,
    out_len_bytes: int | None = None,
    tail_bytes: int = 0,
) -> tuple[bool, str]:
    if not meta.supported:
        return (False, f"unsupported variant: {meta.variant_id}")
    if output_layout is None:
        output_layout = "bytes" if meta.io_layout == "bytes" else "bitplanes"

    threads_eff = threads * meta.ctr_group
    blocks_total = threads_eff * 32
    if out_len_bytes is None:
        out_len_bytes_eff = blocks_total * 16
    else:
        out_len_bytes_eff = int(out_len_bytes)
    tail_bytes_eff = tail_bytes if tail_bytes > 0 else (out_len_bytes_eff % 16)

    if meta.io_layout == "bytes" and output_layout != "bytes":
        return (
            False,
            f"io_layout={meta.io_layout} requires output_layout=bytes (got {output_layout})",
        )
    if meta.io_layout == "plane-major4" and output_layout == "bytes":
        return (
            False,
            "output_layout=bytes requires io_layout=bytes in this family",
        )
    if output_layout == "words" and meta.io_layout != "plane-major4":
        return (
            False,
            "output_layout=words requires io_layout=plane-major4",
        )

    pt_block = bytes(
        (
            0x00,
            0x11,
            0x22,
            0x33,
            0x44,
            0x55,
            0x66,
            0x77,
            0x88,
            0x99,
            0xAA,
            0xBB,
            0xCC,
            0xDD,
            0xEE,
            0xFF,
        )
    )
    if post_op == "store":
        out_seed = bytes(16)
    elif post_op == "xor_accumulate":
        out_seed = bytes(
            (
                0xA5,
                0x5A,
                0xC3,
                0x3C,
                0x96,
                0x69,
                0x0F,
                0xF0,
                0x12,
                0x21,
                0x34,
                0x43,
                0x56,
                0x65,
                0x78,
                0x87,
            )
        )
    else:
        raise ValueError(f"unsupported post_op={post_op}")

    exp_key, key_soa = _build_check_key_material(meta, threads)

    expected_store = _aes_encrypt_block_ref(pt_block, exp_key, meta.key_bits)
    if post_op == "store":
        expected = expected_store
    else:
        expected = bytes((out_seed[i] ^ expected_store[i]) & 0xFF for i in range(16))

    use_words_fastpath = (
        output_layout == "words"
        and meta.io_layout == "plane-major4"
        and meta.key_source == "const_key"
        and meta.key_bits == 128
    )

    if use_words_fastpath:
        cubin = _compile_coalesced_words_cubin(sm=sm, post_op=post_op)
        words = threads_eff * 128
        host_in = (ctypes.c_uint32 * words)()
        host_out = (ctypes.c_uint32 * words)()
        _seed_bitplanes_words(host_in, threads_eff, pt_block)
        if post_op == "store":
            for i in range(words):
                host_out[i] = 0
        else:
            _seed_bitplanes_words(host_out, threads_eff, out_seed)

        cuda = Cuda()
        cuda.init()
        dev = cuda.device(0)
        ctx = cuda.ctx_create(dev)
        try:
            mod = cuda.module_load_data(cubin)
            _upload_const_rk_bits(cuda, mod, key_bits=128)
            fn = cuda.module_get_function(mod, "stc_bp128_words_kernel")
            d_in = cuda.mem_alloc(ctypes.sizeof(host_in))
            d_out = cuda.mem_alloc(ctypes.sizeof(host_out))
            try:
                cuda.memcpy_htod(d_in, host_in, ctypes.sizeof(host_in))
                cuda.memcpy_htod(d_out, host_out, ctypes.sizeof(host_out))
                arg_in = ctypes.c_uint64(d_in)
                arg_out = ctypes.c_uint64(d_out)
                arg_threads = ctypes.c_uint32(threads_eff)
                arg_out_len = ctypes.c_uint32(out_len_bytes_eff)
                arg_tail = ctypes.c_uint32(tail_bytes_eff)
                args = [
                    ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_threads), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_out_len), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_tail), ctypes.c_void_p),
                ]
                grid = ((threads_eff + block - 1) // block, 1, 1)
                cuda.launch_async(fn, grid=grid, block=(block, 1, 1), args=args)
                cuda.synchronize()
                cuda.memcpy_dtoh(host_out, d_out, ctypes.sizeof(host_out))
            finally:
                cuda.mem_free(d_in)
                cuda.mem_free(d_out)
        finally:
            cuda.ctx_destroy(ctx)
            got = _decode_first_lane_words(host_out, threads_eff)
        ok = got == expected
        return (
            ok,
            (
                f"expected={expected.hex()} got={got.hex()}"
                if not ok
                else f"ciphertext={got.hex()}"
            ),
        )

    cubin = compile_variant_to_cubin(meta, sm=sm, post_op=post_op)
    cuda = Cuda()
    cuda.init()
    dev = cuda.device(0)
    ctx = cuda.ctx_create(dev)
    try:
        mod = cuda.module_load_data(cubin)
        _upload_const_keys(cuda, mod, meta)
        core_fn = cuda.module_get_function(mod, _core_kernel_name(meta))

        if meta.io_layout == "bytes":
            if (block & 31) != 0:
                return (False, "bytes io_layout requires block multiple of 32")
            conv_in_fn = cuda.module_get_function(mod, _BYTES_TO_PLANES4_KERNEL)
            conv_out_fn = cuda.module_get_function(mod, _PLANES4_TO_BYTES_KERNEL)
            bytes_io = blocks_total * 16
            planes_words = threads_eff * 128
            planes_bytes = ctypes.sizeof(ctypes.c_uint32) * planes_words
            host_in_bytes = (ctypes.c_uint8 * bytes_io)()
            host_out_bytes = (ctypes.c_uint8 * bytes_io)()
            _seed_plaintext_bytes(host_in_bytes, blocks_total, pt_block)
            if post_op == "store":
                for i in range(bytes_io):
                    host_out_bytes[i] = 0
            else:
                _seed_plaintext_bytes(host_out_bytes, blocks_total, out_seed)

            d_in_bytes = cuda.mem_alloc(ctypes.sizeof(host_in_bytes))
            d_out_bytes = cuda.mem_alloc(ctypes.sizeof(host_out_bytes))
            d_planes0 = cuda.mem_alloc(planes_bytes)
            d_planes1 = cuda.mem_alloc(planes_bytes)
            d_key = 0
            try:
                cuda.memcpy_htod(
                    d_in_bytes, host_in_bytes, ctypes.sizeof(host_in_bytes)
                )
                cuda.memcpy_htod(
                    d_out_bytes, host_out_bytes, ctypes.sizeof(host_out_bytes)
                )
                arg_threads = ctypes.c_uint32(threads_eff)
                arg_out_len = ctypes.c_uint32(out_len_bytes_eff)
                arg_tail = ctypes.c_uint32(tail_bytes_eff)
                arg_in_b = ctypes.c_uint64(d_in_bytes)
                arg_out_b = ctypes.c_uint64(d_out_bytes)
                arg_p0 = ctypes.c_uint64(d_planes0)
                arg_p1 = ctypes.c_uint64(d_planes1)
                conv_in_args = [
                    ctypes.cast(ctypes.byref(arg_in_b), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_p0), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_threads), ctypes.c_void_p),
                ]
                conv_out_args = [
                    ctypes.cast(ctypes.byref(arg_p1), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_out_b), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_threads), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_out_len), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_tail), ctypes.c_void_p),
                ]
                if meta.key_source == "const_key":
                    aes_args = [
                        ctypes.cast(ctypes.byref(arg_p0), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_p1), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_threads), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_out_len), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_tail), ctypes.c_void_p),
                    ]
                else:
                    d_key = cuda.mem_alloc(ctypes.sizeof(key_soa))
                    cuda.memcpy_htod(d_key, key_soa, ctypes.sizeof(key_soa))
                    arg_key = ctypes.c_uint64(d_key)
                    aes_args = [
                        ctypes.cast(ctypes.byref(arg_p0), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_p1), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_key), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_threads), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_out_len), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_tail), ctypes.c_void_p),
                    ]

                grid_aes = ((threads_eff + block - 1) // block, 1, 1)
                conv_threads = threads_eff * 32
                grid_conv = ((conv_threads + block - 1) // block, 1, 1)
                cuda.launch_async(
                    conv_in_fn,
                    grid=grid_conv,
                    block=(block, 1, 1),
                    args=conv_in_args,
                )
                cuda.launch_async(
                    core_fn, grid=grid_aes, block=(block, 1, 1), args=aes_args
                )
                cuda.launch_async(
                    conv_out_fn,
                    grid=grid_conv,
                    block=(block, 1, 1),
                    args=conv_out_args,
                )
                cuda.synchronize()
                cuda.memcpy_dtoh(
                    host_out_bytes, d_out_bytes, ctypes.sizeof(host_out_bytes)
                )
            finally:
                if d_key:
                    cuda.mem_free(d_key)
                cuda.mem_free(d_in_bytes)
                cuda.mem_free(d_out_bytes)
                cuda.mem_free(d_planes0)
                cuda.mem_free(d_planes1)
            got = _decode_first_block_bytes(host_out_bytes)
        else:
            words = threads_eff * 128
            host_in = (ctypes.c_uint32 * words)()
            host_out = (ctypes.c_uint32 * words)()
            _seed_plaintext_planes4(host_in, threads_eff, pt_block)
            if post_op == "store":
                for i in range(words):
                    host_out[i] = 0
            else:
                _seed_plaintext_planes4(host_out, threads_eff, out_seed)

            d_in = cuda.mem_alloc(ctypes.sizeof(host_in))
            d_out = cuda.mem_alloc(ctypes.sizeof(host_out))
            d_key = 0
            try:
                cuda.memcpy_htod(d_in, host_in, ctypes.sizeof(host_in))
                cuda.memcpy_htod(d_out, host_out, ctypes.sizeof(host_out))
                arg_in = ctypes.c_uint64(d_in)
                arg_out = ctypes.c_uint64(d_out)
                arg_threads = ctypes.c_uint32(threads_eff)
                arg_out_len = ctypes.c_uint32(out_len_bytes_eff)
                arg_tail = ctypes.c_uint32(tail_bytes_eff)
                if meta.key_source == "const_key":
                    args = [
                        ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_threads), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_out_len), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_tail), ctypes.c_void_p),
                    ]
                else:
                    d_key = cuda.mem_alloc(ctypes.sizeof(key_soa))
                    cuda.memcpy_htod(d_key, key_soa, ctypes.sizeof(key_soa))
                    arg_key = ctypes.c_uint64(d_key)
                    args = [
                        ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_key), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_threads), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_out_len), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_tail), ctypes.c_void_p),
                    ]
                grid = ((threads_eff + block - 1) // block, 1, 1)
                cuda.launch_async(core_fn, grid=grid, block=(block, 1, 1), args=args)
                cuda.synchronize()
                cuda.memcpy_dtoh(host_out, d_out, ctypes.sizeof(host_out))
            finally:
                if d_key:
                    cuda.mem_free(d_key)
                cuda.mem_free(d_in)
                cuda.mem_free(d_out)
            got = _decode_first_lane_planes4(host_out, threads_eff)
    finally:
        cuda.ctx_destroy(ctx)

    ok = got == expected
    return (
        ok,
        (
            f"expected={expected.hex()} got={got.hex()}"
            if not ok
            else f"ciphertext={got.hex()}"
        ),
    )


def benchmark_variant(
    meta: Bp128VariantMetadata,
    threads: int = 65_536,
    block: int = 128,
    reps: int = 200,
    sm: str = "sm_61",
    seed: int = 0x12345678,
    shared_key: bool = True,
    post_op: PostOp = "store",
    output_layout: OutputLayout | None = None,
    out_len_bytes: int | None = None,
    tail_bytes: int = 0,
) -> tuple[float, float]:
    if not meta.supported:
        raise ValueError(f"unsupported variant: {meta.variant_id}")
    if output_layout is None:
        output_layout = "bytes" if meta.io_layout == "bytes" else "bitplanes"

    threads_eff = threads * meta.ctr_group
    blocks_total = threads_eff * 32
    if out_len_bytes is None:
        out_len_bytes_eff = blocks_total * 16
    else:
        out_len_bytes_eff = int(out_len_bytes)
    tail_bytes_eff = tail_bytes if tail_bytes > 0 else (out_len_bytes_eff % 16)

    if meta.io_layout == "bytes" and output_layout != "bytes":
        raise ValueError(
            f"io_layout={meta.io_layout} requires output_layout=bytes (got {output_layout})"
        )
    if meta.io_layout == "plane-major4" and output_layout == "bytes":
        raise ValueError("output_layout=bytes requires io_layout=bytes")
    if output_layout == "words" and meta.io_layout != "plane-major4":
        raise ValueError(
            "output_layout=words requires io_layout=plane-major4"
        )

    rng = random.Random(seed)
    if post_op not in {"store", "xor_accumulate"}:
        raise ValueError(f"unsupported post_op={post_op}")
    key_soa = make_key_material_soa(
        meta,
        logical_threads=threads,
        seed=seed ^ 0xABCDEF,
        shared_key=shared_key,
    )

    use_words_fastpath = (
        output_layout == "words"
        and meta.io_layout == "plane-major4"
        and meta.key_source == "const_key"
        and meta.key_bits == 128
    )

    if use_words_fastpath:
        cubin = _compile_coalesced_words_cubin(sm=sm, post_op=post_op)
        words = threads_eff * 128
        host_in = (ctypes.c_uint32 * words)()
        host_out = (ctypes.c_uint32 * words)()
        for i in range(words):
            host_in[i] = rng.getrandbits(32)
            host_out[i] = 0 if post_op == "store" else rng.getrandbits(32)
        cuda = Cuda()
        cuda.init()
        dev = cuda.device(0)
        ctx = cuda.ctx_create(dev)
        try:
            mod = cuda.module_load_data(cubin)
            _upload_const_rk_bits(cuda, mod, key_bits=128)
            fn = cuda.module_get_function(mod, "stc_bp128_words_kernel")
            d_in = cuda.mem_alloc(ctypes.sizeof(host_in))
            d_out = cuda.mem_alloc(ctypes.sizeof(host_out))
            try:
                cuda.memcpy_htod(d_in, host_in, ctypes.sizeof(host_in))
                cuda.memcpy_htod(d_out, host_out, ctypes.sizeof(host_out))
                arg_in = ctypes.c_uint64(d_in)
                arg_out = ctypes.c_uint64(d_out)
                arg_threads = ctypes.c_uint32(threads_eff)
                arg_out_len = ctypes.c_uint32(out_len_bytes_eff)
                arg_tail = ctypes.c_uint32(tail_bytes_eff)
                args = [
                    ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_threads), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_out_len), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_tail), ctypes.c_void_p),
                ]
                grid = ((threads_eff + block - 1) // block, 1, 1)
                for _ in range(20):
                    cuda.launch_async(fn, grid=grid, block=(block, 1, 1), args=args)
                cuda.synchronize()
                start = cuda.event_create()
                end = cuda.event_create()
                try:
                    cuda.event_record(start)
                    for _ in range(reps):
                        cuda.launch_async(fn, grid=grid, block=(block, 1, 1), args=args)
                    cuda.event_record(end)
                    cuda.event_synchronize(end)
                    ms = cuda.event_elapsed_ms(start, end)
                finally:
                    cuda.event_destroy(start)
                    cuda.event_destroy(end)
            finally:
                cuda.mem_free(d_in)
                cuda.mem_free(d_out)
        finally:
            cuda.ctx_destroy(ctx)

        seconds = ms / 1000.0
        evals_per_sec = (threads_eff * 32 * reps) / seconds
        mib_s = evals_per_sec * 16.0 / (1024.0 * 1024.0)
        return evals_per_sec / 1e9, mib_s

    cubin = compile_variant_to_cubin(meta, sm=sm, post_op=post_op)
    cuda = Cuda()
    cuda.init()
    dev = cuda.device(0)
    ctx = cuda.ctx_create(dev)
    try:
        mod = cuda.module_load_data(cubin)
        _upload_const_keys(cuda, mod, meta)
        core_fn = cuda.module_get_function(mod, _core_kernel_name(meta))

        if meta.io_layout == "bytes":
            if (block & 31) != 0:
                raise ValueError("bytes io_layout requires block multiple of 32")
            conv_in_fn = cuda.module_get_function(mod, _BYTES_TO_PLANES4_KERNEL)
            conv_out_fn = cuda.module_get_function(mod, _PLANES4_TO_BYTES_KERNEL)
            bytes_io = blocks_total * 16
            planes_words = threads_eff * 128
            planes_bytes = ctypes.sizeof(ctypes.c_uint32) * planes_words
            host_in_bytes = (ctypes.c_uint8 * bytes_io)()
            host_out_bytes = (ctypes.c_uint8 * bytes_io)()
            for i in range(bytes_io):
                host_in_bytes[i] = rng.getrandbits(8)
                host_out_bytes[i] = 0 if post_op == "store" else rng.getrandbits(8)
            d_in_bytes = cuda.mem_alloc(ctypes.sizeof(host_in_bytes))
            d_out_bytes = cuda.mem_alloc(ctypes.sizeof(host_out_bytes))
            d_planes0 = cuda.mem_alloc(planes_bytes)
            d_planes1 = cuda.mem_alloc(planes_bytes)
            d_key = 0
            try:
                cuda.memcpy_htod(
                    d_in_bytes, host_in_bytes, ctypes.sizeof(host_in_bytes)
                )
                cuda.memcpy_htod(
                    d_out_bytes, host_out_bytes, ctypes.sizeof(host_out_bytes)
                )
                arg_threads = ctypes.c_uint32(threads_eff)
                arg_out_len = ctypes.c_uint32(out_len_bytes_eff)
                arg_tail = ctypes.c_uint32(tail_bytes_eff)
                arg_in_b = ctypes.c_uint64(d_in_bytes)
                arg_out_b = ctypes.c_uint64(d_out_bytes)
                arg_p0 = ctypes.c_uint64(d_planes0)
                arg_p1 = ctypes.c_uint64(d_planes1)
                conv_in_args = [
                    ctypes.cast(ctypes.byref(arg_in_b), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_p0), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_threads), ctypes.c_void_p),
                ]
                conv_out_args = [
                    ctypes.cast(ctypes.byref(arg_p1), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_out_b), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_threads), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_out_len), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_tail), ctypes.c_void_p),
                ]
                if meta.key_source == "const_key":
                    aes_args = [
                        ctypes.cast(ctypes.byref(arg_p0), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_p1), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_threads), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_out_len), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_tail), ctypes.c_void_p),
                    ]
                else:
                    d_key = cuda.mem_alloc(ctypes.sizeof(key_soa))
                    cuda.memcpy_htod(d_key, key_soa, ctypes.sizeof(key_soa))
                    arg_key = ctypes.c_uint64(d_key)
                    aes_args = [
                        ctypes.cast(ctypes.byref(arg_p0), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_p1), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_key), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_threads), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_out_len), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_tail), ctypes.c_void_p),
                    ]

                grid_aes = ((threads_eff + block - 1) // block, 1, 1)
                conv_threads = threads_eff * 32
                grid_conv = ((conv_threads + block - 1) // block, 1, 1)
                for _ in range(20):
                    cuda.launch_async(
                        conv_in_fn,
                        grid=grid_conv,
                        block=(block, 1, 1),
                        args=conv_in_args,
                    )
                    cuda.launch_async(
                        core_fn, grid=grid_aes, block=(block, 1, 1), args=aes_args
                    )
                    cuda.launch_async(
                        conv_out_fn,
                        grid=grid_conv,
                        block=(block, 1, 1),
                        args=conv_out_args,
                    )
                cuda.synchronize()
                start = cuda.event_create()
                end = cuda.event_create()
                try:
                    cuda.event_record(start)
                    for _ in range(reps):
                        cuda.launch_async(
                            conv_in_fn,
                            grid=grid_conv,
                            block=(block, 1, 1),
                            args=conv_in_args,
                        )
                        cuda.launch_async(
                            core_fn, grid=grid_aes, block=(block, 1, 1), args=aes_args
                        )
                        cuda.launch_async(
                            conv_out_fn,
                            grid=grid_conv,
                            block=(block, 1, 1),
                            args=conv_out_args,
                        )
                    cuda.event_record(end)
                    cuda.event_synchronize(end)
                    ms = cuda.event_elapsed_ms(start, end)
                finally:
                    cuda.event_destroy(start)
                    cuda.event_destroy(end)
            finally:
                if d_key:
                    cuda.mem_free(d_key)
                cuda.mem_free(d_in_bytes)
                cuda.mem_free(d_out_bytes)
                cuda.mem_free(d_planes0)
                cuda.mem_free(d_planes1)
        else:
            words = threads_eff * 128
            host_in = (ctypes.c_uint32 * words)()
            host_out = (ctypes.c_uint32 * words)()
            for i in range(words):
                host_in[i] = rng.getrandbits(32)
                host_out[i] = 0 if post_op == "store" else rng.getrandbits(32)
            d_in = cuda.mem_alloc(ctypes.sizeof(host_in))
            d_out = cuda.mem_alloc(ctypes.sizeof(host_out))
            d_key = 0
            try:
                cuda.memcpy_htod(d_in, host_in, ctypes.sizeof(host_in))
                cuda.memcpy_htod(d_out, host_out, ctypes.sizeof(host_out))

                arg_in = ctypes.c_uint64(d_in)
                arg_out = ctypes.c_uint64(d_out)
                arg_threads = ctypes.c_uint32(threads_eff)
                arg_out_len = ctypes.c_uint32(out_len_bytes_eff)
                arg_tail = ctypes.c_uint32(tail_bytes_eff)
                if meta.key_source == "const_key":
                    args = [
                        ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_threads), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_out_len), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_tail), ctypes.c_void_p),
                    ]
                else:
                    d_key = cuda.mem_alloc(ctypes.sizeof(key_soa))
                    cuda.memcpy_htod(d_key, key_soa, ctypes.sizeof(key_soa))
                    arg_key = ctypes.c_uint64(d_key)
                    args = [
                        ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_key), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_threads), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_out_len), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_tail), ctypes.c_void_p),
                    ]
                grid = ((threads_eff + block - 1) // block, 1, 1)
                for _ in range(20):
                    cuda.launch_async(
                        core_fn, grid=grid, block=(block, 1, 1), args=args
                    )
                cuda.synchronize()
                start = cuda.event_create()
                end = cuda.event_create()
                try:
                    cuda.event_record(start)
                    for _ in range(reps):
                        cuda.launch_async(
                            core_fn, grid=grid, block=(block, 1, 1), args=args
                        )
                    cuda.event_record(end)
                    cuda.event_synchronize(end)
                    ms = cuda.event_elapsed_ms(start, end)
                finally:
                    cuda.event_destroy(start)
                    cuda.event_destroy(end)
            finally:
                cuda.mem_free(d_in)
                cuda.mem_free(d_out)
                if d_key:
                    cuda.mem_free(d_key)
    finally:
        cuda.ctx_destroy(ctx)

    seconds = ms / 1000.0
    evals_per_sec = (threads_eff * 32 * reps) / seconds
    mib_s = evals_per_sec * 16.0 / (1024.0 * 1024.0)
    return evals_per_sec / 1e9, mib_s


__all__ = [
    "KEY_BITS_OPTIONS",
    "CTR_GROUP_OPTIONS",
    "KEY_SOURCE_OPTIONS",
    "IO_LAYOUT_OPTIONS",
    "KeySource",
    "IoLayout",
    "PostOp",
    "Bp128VariantMetadata",
    "Bp128DispatchRequest",
    "Bp128Dispatcher",
    "variant_id",
    "kernel_name_from_variant_id",
    "build_variant_metadata",
    "all_variant_metadata",
    "family_manifest_json",
    "generate_cuda_source",
    "compile_variant_to_ptx",
    "compile_variant_to_cubin",
    "make_key_material_soa",
    "check_variant_correctness",
    "benchmark_variant",
]
