#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import bench_aes10_bp128_cuda as base
from stc.aes_bp128_variant_family import _aes_encrypt_block_ref
from stc.cuda_driver import Cuda, CudaError

Mode = Literal[
    "masterkey_old",
    "masterkey_streamed",
    "masterkey_streamed_xor",
    "paramrk_shared",
]
ArgKind = Literal["masterkey", "rk_shared"]


@dataclass
class ModeSpec:
    mode: Mode
    arg_kind: ArgKind
    xor_accumulate: bool
    kernel_cubin: bytes
    regs: int


@dataclass
class ModeResult:
    mode: Mode
    regs: int
    best_block: int
    block_eval_b: float
    ns_per_eval: float
    ctx_eval_m: float
    ns_per_ctx: float
    mib_s: float
    occupancy: float
    check_ok: bool


def _require_nvcc() -> str:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc not found on PATH")
    return nvcc


def _build_lane_plaintexts() -> list[bytes]:
    prefix = bytes.fromhex("00112233445566778899aabb")
    blocks: list[bytes] = []
    for lane in range(32):
        ctr = lane.to_bytes(4, "big")
        blocks.append(prefix + ctr)
    return blocks


def _build_lane_masks(lane_blocks: list[bytes]) -> list[list[int]]:
    masks = [[0 for _ in range(8)] for _ in range(16)]
    for lane, blk in enumerate(lane_blocks):
        lane_bit = 1 << lane
        for byte_idx in range(16):
            pb = blk[byte_idx]
            for bit in range(8):
                if (pb >> bit) & 1:
                    masks[byte_idx][bit] |= lane_bit
    return masks


def _seed_ctr_planes4(words: ctypes.Array[ctypes.c_uint32], threads: int) -> None:
    lane_blocks = _build_lane_plaintexts()
    masks = _build_lane_masks(lane_blocks)
    for byte_idx in range(16):
        for bit in range(8):
            group = byte_idx * 2 + (bit >> 2)
            lane4 = bit & 3
            mask = masks[byte_idx][bit]
            base_idx = (group * threads) * 4 + lane4
            for tid in range(threads):
                words[base_idx + tid * 4] = mask


def _decode_block_from_planes4(
    words: ctypes.Array[ctypes.c_uint32], threads: int, tid: int, lane: int
) -> bytes:
    out = bytearray(16)
    for byte_idx in range(16):
        value = 0
        for bit in range(8):
            group = byte_idx * 2 + (bit >> 2)
            lane4 = bit & 3
            idx = (group * threads + tid) * 4 + lane4
            bitval = (int(words[idx]) >> lane) & 1
            value |= bitval << bit
        out[byte_idx] = value
    return bytes(out)


def _extract_context_bytes(
    words: ctypes.Array[ctypes.c_uint32],
    threads: int,
    tid: int,
    context_index: int,
    context_bytes: int,
    blocks_per_ctx: int,
) -> bytes:
    lane_base = context_index * blocks_per_ctx
    out = bytearray()
    for lane in range(lane_base, lane_base + blocks_per_ctx):
        out.extend(_decode_block_from_planes4(words, threads=threads, tid=tid, lane=lane))
    return bytes(out[:context_bytes])


def _build_masterkey_soa(threads: int, shared_key: bool) -> tuple[bytes, bytes]:
    key0 = bytes(range(16))
    if shared_key:
        packed = base.pack_masterkey_soa_shared_key(key0, threads)
        return key0, packed

    key_bytes = bytearray(threads * 16)
    for tid in range(threads):
        mask = (tid * 131) & 0xFF
        off = tid * 16
        for i in range(16):
            key_bytes[off + i] = key0[i] ^ mask
    packed = base.pack_masterkey_soa_thread_keys(bytes(key_bytes), threads)
    return bytes(key_bytes[0:16]), packed


def _build_rk_bits_u32() -> ctypes.Array[ctypes.c_uint32]:
    out = (ctypes.c_uint32 * (11 * 16 * 8))()
    idx = 0
    for rk_hex in base.ROUND_KEYS_HEX:
        rk = bytes.fromhex(rk_hex)
        for byte_idx in range(16):
            kv = rk[byte_idx]
            for bit in range(8):
                out[idx] = 0xFFFFFFFF if ((kv >> bit) & 1) else 0
                idx += 1
    return out


def _compile_kernel(mode: Mode, sm: str, maxrregcount: int | None) -> ModeSpec:
    mapped, _lop3_count = base._build_bp128_mapped()
    sbox_cuda = base._emit_sbox_inline_cuda(
        mapped, func_name="sbox_bp128_lop3_inline", noinline=False
    )
    if mode == "masterkey_old":
        src = base._kernel_cu_source_replacement_coalesced4_masterkey_soa(sbox_cuda)
        arg_kind: ArgKind = "masterkey"
        xor_accumulate = False
    elif mode == "masterkey_streamed":
        src = base._kernel_cu_source_replacement_coalesced4_masterkey_soa_streamed(
            sbox_cuda, xor_accumulate=False, launch_bounds=(128, 3)
        )
        arg_kind = "masterkey"
        xor_accumulate = False
    elif mode == "masterkey_streamed_xor":
        src = base._kernel_cu_source_replacement_coalesced4_masterkey_soa_streamed(
            sbox_cuda, xor_accumulate=True, launch_bounds=(128, 3)
        )
        arg_kind = "masterkey"
        xor_accumulate = True
    elif mode == "paramrk_shared":
        src = base._kernel_cu_source_replacement_coalesced4_paramrk_shared(sbox_cuda)
        arg_kind = "rk_shared"
        xor_accumulate = False
    else:
        raise ValueError(f"unknown mode {mode}")

    nvcc = _require_nvcc()
    with tempfile.TemporaryDirectory(prefix=f"bp128_faest_{mode}_") as td:
        work = Path(td)
        cu_path = work / "kernel.cu"
        ptx_path = work / "kernel.ptx"
        cubin_path = work / "kernel.cubin"
        cu_path.write_text(src, encoding="utf-8")

        codegen_flags = ["-O3", f"-arch={sm}"]
        if maxrregcount is not None and maxrregcount > 0:
            codegen_flags.append(f"--maxrregcount={maxrregcount}")

        run_ptx = subprocess.run(
            [nvcc, "-ptx", *codegen_flags, str(cu_path), "-o", str(ptx_path)],
            capture_output=True,
            text=True,
        )
        if run_ptx.returncode != 0:
            raise RuntimeError(
                f"nvcc -ptx failed for {mode}\nSTDOUT:\n{run_ptx.stdout}\nSTDERR:\n{run_ptx.stderr}"
            )

        run_cubin = subprocess.run(
            [
                nvcc,
                "-cubin",
                *codegen_flags,
                "-Xptxas",
                "-v",
                str(cu_path),
                "-o",
                str(cubin_path),
            ],
            capture_output=True,
            text=True,
        )
        if run_cubin.returncode != 0:
            raise RuntimeError(
                f"nvcc -cubin failed for {mode}\nSTDOUT:\n{run_cubin.stdout}\nSTDERR:\n{run_cubin.stderr}"
            )
        report = (run_cubin.stdout or "") + "\n" + (run_cubin.stderr or "")
        m = re.search(r"Used\s+(\d+)\s+registers", report)
        if not m:
            raise RuntimeError(f"failed to parse register count for {mode}\n{report}")
        regs = int(m.group(1))
        src = cubin_path.read_bytes()

    return ModeSpec(
        mode=mode,
        arg_kind=arg_kind,
        xor_accumulate=xor_accumulate,
        kernel_cubin=src,
        regs=regs,
    )


def _arch_limits(sm: str) -> tuple[int, int, int, int]:
    if sm.startswith("sm_61") or sm.startswith("sm_60"):
        return (65536, 2048, 64, 32)
    if sm.startswith("sm_75"):
        return (65536, 1024, 32, 16)
    if sm.startswith("sm_80") or sm.startswith("sm_86") or sm.startswith("sm_89"):
        return (65536, 1536, 48, 16)
    return (65536, 2048, 64, 32)


def _estimate_occupancy(sm: str, regs_per_thread: int, block: int) -> float:
    regs_sm, max_threads_sm, max_warps_sm, max_blocks_sm = _arch_limits(sm)
    if regs_per_thread <= 0 or block <= 0:
        return 0.0
    blocks_by_regs = regs_sm // (regs_per_thread * block)
    blocks_by_threads = max_threads_sm // block
    active_blocks = min(blocks_by_regs, blocks_by_threads, max_blocks_sm)
    if active_blocks <= 0:
        return 0.0
    warps_per_block = (block + 31) // 32
    active_warps = min(active_blocks * warps_per_block, max_warps_sm)
    return float(active_warps) / float(max_warps_sm)


def _launch_async(
    cuda: Cuda,
    fn: ctypes.c_void_p,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    args: list[ctypes.c_void_p],
    shared_bytes: int = 0,
) -> None:
    arr_t = ctypes.c_void_p * len(args)
    params = arr_t(*args)
    code = cuda.lib.cuLaunchKernel(
        fn,
        int(grid[0]),
        int(grid[1]),
        int(grid[2]),
        int(block[0]),
        int(block[1]),
        int(block[2]),
        int(shared_bytes),
        None,
        params,
        None,
    )
    if code != 0:
        raise CudaError("cuLaunchKernel failed", code=code)


def _as_u8_array(data: bytes) -> ctypes.Array[ctypes.c_uint8]:
    arr = (ctypes.c_uint8 * len(data))()
    for i, b in enumerate(data):
        arr[i] = b
    return arr


def _run_mode(
    spec: ModeSpec,
    threads: int,
    blocks: list[int],
    reps: int,
    sm: str,
    context_bytes: int,
    shared_key: bool,
    do_check: bool,
) -> ModeResult:
    blocks_per_ctx = (context_bytes + 15) // 16
    ctx_per_thread = 32 // blocks_per_ctx
    if ctx_per_thread <= 0:
        raise ValueError(
            f"context_bytes={context_bytes} too large for 32-lane BP128 groups"
        )

    words = threads * 128
    host_in = (ctypes.c_uint32 * words)()
    _seed_ctr_planes4(host_in, threads)

    host_out_seed = (ctypes.c_uint32 * words)()
    if spec.xor_accumulate:
        state = 0x243F6A88
        for i in range(words):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            host_out_seed[i] = state
    else:
        for i in range(words):
            host_out_seed[i] = 0
    host_out = (ctypes.c_uint32 * words)()

    lane_blocks = _build_lane_plaintexts()
    if spec.arg_kind == "masterkey":
        key0, key_soa_bytes = _build_masterkey_soa(threads=threads, shared_key=shared_key)
        key_soa_arr = _as_u8_array(key_soa_bytes)
    else:
        key0 = bytes(range(16))
        key_soa_arr = (ctypes.c_uint8 * 0)()
    expected_blocks = [_aes_encrypt_block_ref(blk, key0, 128) for blk in lane_blocks]

    cuda = Cuda()
    cuda.init()
    dev = cuda.device(0)
    ctx = cuda.ctx_create(dev)
    try:
        mod = cuda.module_load_data(spec.kernel_cubin)
        fn = cuda.module_get_function(mod, "aes10_bp128_kernel")
        d_in = cuda.mem_alloc(ctypes.sizeof(host_in))
        d_out = cuda.mem_alloc(ctypes.sizeof(host_out_seed))
        d_key = 0
        d_rk = 0
        try:
            cuda.memcpy_htod(d_in, host_in, ctypes.sizeof(host_in))
            if spec.arg_kind == "masterkey":
                d_key = cuda.mem_alloc(ctypes.sizeof(key_soa_arr))
                cuda.memcpy_htod(d_key, key_soa_arr, ctypes.sizeof(key_soa_arr))
            else:
                rk_bits = _build_rk_bits_u32()
                d_rk = cuda.mem_alloc(ctypes.sizeof(rk_bits))
                cuda.memcpy_htod(d_rk, rk_bits, ctypes.sizeof(rk_bits))

            best_block = blocks[0]
            best_ctx_eval = -1.0
            best_block_eval = 0.0
            best_ns_eval = 0.0
            best_ns_ctx = 0.0
            best_mib_s = 0.0

            check_ok = not do_check

            for block in blocks:
                grid = ((threads + block - 1) // block, 1, 1)
                cuda.memcpy_htod(d_out, host_out_seed, ctypes.sizeof(host_out_seed))
                shared_bytes = (11 * 16 * 8 * 4) if spec.arg_kind == "rk_shared" else 0

                arg_in = ctypes.c_uint64(d_in)
                arg_out = ctypes.c_uint64(d_out)
                arg_threads = ctypes.c_uint32(threads)
                if spec.arg_kind == "masterkey":
                    arg_key = ctypes.c_uint64(d_key)
                    args = [
                        ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_key), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_threads), ctypes.c_void_p),
                    ]
                else:
                    arg_rk = ctypes.c_uint64(d_rk)
                    args = [
                        ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_rk), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_threads), ctypes.c_void_p),
                    ]

                if do_check and not check_ok:
                    _launch_async(
                        cuda,
                        fn,
                        grid=grid,
                        block=(block, 1, 1),
                        args=args,
                        shared_bytes=shared_bytes,
                    )
                    cuda.synchronize()
                    cuda.memcpy_dtoh(host_out, d_out, ctypes.sizeof(host_out))
                    for context_index in range(ctx_per_thread):
                        got = _extract_context_bytes(
                            host_out,
                            threads=threads,
                            tid=0,
                            context_index=context_index,
                            context_bytes=context_bytes,
                            blocks_per_ctx=blocks_per_ctx,
                        )
                        expected_concat = bytearray()
                        lane_base = context_index * blocks_per_ctx
                        for lane in range(lane_base, lane_base + blocks_per_ctx):
                            expected_concat.extend(expected_blocks[lane])
                        expected = bytes(expected_concat[:context_bytes])
                        if spec.xor_accumulate:
                            seed_bytes = _extract_context_bytes(
                                host_out_seed,
                                threads=threads,
                                tid=0,
                                context_index=context_index,
                                context_bytes=context_bytes,
                                blocks_per_ctx=blocks_per_ctx,
                            )
                            expected = bytes(
                                (seed_bytes[i] ^ expected[i]) & 0xFF
                                for i in range(context_bytes)
                            )
                        if got != expected:
                            raise RuntimeError(
                                f"{spec.mode} check failed at context {context_index}: "
                                f"expected={expected[:32].hex()} got={got[:32].hex()}"
                            )
                    check_ok = True
                    cuda.memcpy_htod(d_out, host_out_seed, ctypes.sizeof(host_out_seed))

                for _ in range(20):
                    _launch_async(
                        cuda,
                        fn,
                        grid=grid,
                        block=(block, 1, 1),
                        args=args,
                        shared_bytes=shared_bytes,
                    )
                cuda.synchronize()

                start = cuda.event_create()
                end = cuda.event_create()
                try:
                    cuda.event_record(start)
                    for _ in range(reps):
                        _launch_async(
                            cuda,
                            fn,
                            grid=grid,
                            block=(block, 1, 1),
                            args=args,
                            shared_bytes=shared_bytes,
                        )
                    cuda.event_record(end)
                    cuda.event_synchronize(end)
                    ms = cuda.event_elapsed_ms(start, end)
                finally:
                    cuda.event_destroy(start)
                    cuda.event_destroy(end)

                seconds = ms / 1000.0
                block_total = threads * 32 * reps
                block_eval_per_sec = block_total / seconds
                ns_per_eval = (seconds / block_total) * 1e9
                ctx_total = threads * ctx_per_thread * reps
                ctx_eval_per_sec = ctx_total / seconds
                ns_per_ctx = (seconds / ctx_total) * 1e9
                mib_s = ctx_eval_per_sec * context_bytes / (1024.0 * 1024.0)

                if ctx_eval_per_sec > best_ctx_eval:
                    best_ctx_eval = ctx_eval_per_sec
                    best_block = block
                    best_block_eval = block_eval_per_sec
                    best_ns_eval = ns_per_eval
                    best_ns_ctx = ns_per_ctx
                    best_mib_s = mib_s

        finally:
            if d_key:
                cuda.mem_free(d_key)
            if d_rk:
                cuda.mem_free(d_rk)
            cuda.mem_free(d_in)
            cuda.mem_free(d_out)
    finally:
        cuda.ctx_destroy(ctx)

    occ = _estimate_occupancy(sm=sm, regs_per_thread=spec.regs, block=best_block)
    return ModeResult(
        mode=spec.mode,
        regs=spec.regs,
        best_block=best_block,
        block_eval_b=best_block_eval / 1e9,
        ns_per_eval=best_ns_eval,
        ctx_eval_m=best_ctx_eval / 1e6,
        ns_per_ctx=best_ns_ctx,
        mib_s=best_mib_s,
        occupancy=occ,
        check_ok=check_ok,
    )


def _parse_modes(text: str) -> list[Mode]:
    out: list[Mode] = []
    for token in text.split(","):
        mode = token.strip()
        if not mode:
            continue
        if mode not in {
            "masterkey_old",
            "masterkey_streamed",
            "masterkey_streamed_xor",
            "paramrk_shared",
        }:
            raise ValueError(f"unsupported mode: {mode}")
        out.append(mode)  # type: ignore[arg-type]
    if not out:
        raise ValueError("no modes selected")
    return out


def _parse_blocks(text: str) -> list[int]:
    out = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("block sizes must be > 0")
        out.append(value)
    if not out:
        raise ValueError("no block sizes selected")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="FAEST-shaped BP128 AES benchmark for master-key and shared-RK modes."
    )
    ap.add_argument("--sm", default="sm_61")
    ap.add_argument("--threads", type=int, default=65_536)
    ap.add_argument("--reps", type=int, default=120)
    ap.add_argument("--blocks", default="64,128,256")
    ap.add_argument("--context-bytes", type=int, default=210)
    ap.add_argument(
        "--modes",
        default="masterkey_old,masterkey_streamed,masterkey_streamed_xor,paramrk_shared",
    )
    ap.add_argument(
        "--shared-key",
        action="store_true",
        help="Use one master key for all threads (default uses per-thread keys for masterkey modes).",
    )
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--maxrregcount", type=int, default=None)
    args = ap.parse_args()

    smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if smi.returncode != 0:
        print("nvidia-smi -L failed; GPU/NVML not ready")
        if smi.stderr.strip():
            print(smi.stderr.strip())
        return 2

    modes = _parse_modes(args.modes)
    blocks = _parse_blocks(args.blocks)

    specs: list[ModeSpec] = []
    for mode in modes:
        spec = _compile_kernel(mode=mode, sm=args.sm, maxrregcount=args.maxrregcount)
        specs.append(spec)

    print(
        f"sm={args.sm} threads={args.threads} reps={args.reps} "
        f"context_bytes={args.context_bytes} blocks={blocks}"
    )
    print(
        "mode order:",
        ", ".join(modes),
    )

    results: list[ModeResult] = []
    for spec in specs:
        result = _run_mode(
            spec=spec,
            threads=args.threads,
            blocks=blocks,
            reps=args.reps,
            sm=args.sm,
            context_bytes=args.context_bytes,
            shared_key=args.shared_key,
            do_check=args.check,
        )
        results.append(result)

    print("| mode | regs | occ@best | best block | B eval/s | ns/eval | M ctx/s | ns/ctx | MiB/s | check |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in results:
        print(
            f"| `{row.mode}` | {row.regs} | {row.occupancy * 100.0:.1f}% | "
            f"{row.best_block} | {row.block_eval_b:.3f} | {row.ns_per_eval:.3f} | "
            f"{row.ctx_eval_m:.3f} | {row.ns_per_ctx:.3f} | {row.mib_s:.2f} | "
            f"{'PASS' if row.check_ok else 'FAIL'} |"
        )

    by_mode = {row.mode: row for row in results}
    if "masterkey_old" in by_mode and "masterkey_streamed" in by_mode:
        base_row = by_mode["masterkey_old"]
        new_row = by_mode["masterkey_streamed"]
        if base_row.ctx_eval_m > 0:
            speedup = new_row.ctx_eval_m / base_row.ctx_eval_m
            print(
                f"masterkey_streamed vs masterkey_old: {speedup:.2f}x "
                f"(ctx/s), regs {base_row.regs}->{new_row.regs}"
            )
    if "masterkey_streamed" in by_mode and "paramrk_shared" in by_mode:
        a = by_mode["masterkey_streamed"]
        b = by_mode["paramrk_shared"]
        if b.ctx_eval_m > 0:
            gap = a.ctx_eval_m / b.ctx_eval_m
            print(
                f"masterkey_streamed vs paramrk_shared: {gap:.2f}x "
                f"(ctx/s), regs {a.regs} vs {b.regs}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
