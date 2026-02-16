#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stc.aes_bp128_variant_family import (
    _expand_round_keys,
    benchmark_variant,
    build_variant_metadata,
    check_variant_correctness,
)
from stc.cuda_driver import Cuda, CudaError

KEY_BITS = (128, 192, 256)
CTR_GROUP = (1, 2, 4)
KEY_SOURCE = ("masterkey_soa", "expanded_rk_soa", "const_key")
IN_LAYOUT = ("bytes", "bitplanes")
OUT_LAYOUT = ("bytes", "words", "bitplanes")
POST_OP = ("store", "xor-accumulate")


@dataclass
class AxisResult:
    key_bits: int
    ctr_group: int
    key_source: str
    in_layout: str
    out_layout: str
    post_op: str
    supported: bool
    correctness: str
    eval_b: float | None
    mib_s: float | None
    best_block: int | None
    notes: str


def _support_for_combo(
    key_bits: int,
    ctr_group: int,
    key_source: str,
    in_layout: str,
    out_layout: str,
    post_op: str,
) -> tuple[bool, str]:
    if (
        key_bits == 128
        and key_source == "const_key"
        and in_layout == "bytes"
        and out_layout == "bytes"
        and post_op == "store"
    ):
        return (True, "supported via bench_aes10_bp128_byteio_cuda.py")
    if (
        key_bits == 128
        and key_source == "const_key"
        and in_layout == "bitplanes"
        and out_layout == "words"
        and post_op in {"store", "xor-accumulate"}
    ):
        return (True, "supported via replacement_coalesced_tuned")

    if in_layout != "bitplanes":
        return (
            False,
            "bytes input not wired in dynamic family (requires conversion wrapper)",
        )
    if out_layout != "bitplanes":
        return (
            False,
            "non-bitplane output not wired in dynamic family (words/bytes wrapper missing)",
        )
    if post_op not in {"store", "xor-accumulate"}:
        return (False, f"unsupported post_op {post_op}")

    meta = build_variant_metadata(
        key_bits=key_bits,
        ctr_group=ctr_group,
        key_source=key_source,  # type: ignore[arg-type]
        io_layout="plane-major4",
    )
    if not meta.supported:
        return (False, meta.notes)
    return (True, "supported")


def _parse_best_eval_and_mib(
    text: str, metric_tag: str
) -> tuple[float | None, float | None]:
    eval_candidates = [
        float(x) for x in re.findall(r"([0-9]+(?:\.[0-9]+)?)B evals/sec", text)
    ]
    best_eval = max(eval_candidates) if eval_candidates else None
    mib_candidates = [
        float(x)
        for x in re.findall(
            rf"{re.escape(metric_tag)}:[^\n]*?([0-9]+(?:\.[0-9]+)?) MiB/s", text
        )
    ]
    best_mib = max(mib_candidates) if mib_candidates else None
    if best_mib is None and best_eval is not None:
        best_mib = best_eval * 1e9 * 16.0 / (1024.0 * 1024.0)
    return best_eval, best_mib


def _load_bench_module():
    from scripts import bench_aes10_bp128_cuda as bench

    return bench


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


def _build_const_rk_bits_u32() -> ctypes.Array[ctypes.c_uint32]:
    key = bytes(range(16))
    rk = _expand_round_keys(key, 128)
    out = (ctypes.c_uint32 * (11 * 16 * 8))()
    idx = 0
    for r in range(11):
        for b in range(16):
            kv = rk[r * 16 + b]
            for bit in range(8):
                out[idx] = 0xFFFFFFFF if ((kv >> bit) & 1) else 0
                idx += 1
    return out


def _upload_rk_bits_symbol(cuda: Cuda, mod: ctypes.c_void_p) -> None:
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
    code = fn(ctypes.byref(dptr), ctypes.byref(nbytes), mod, b"RK_BITS")
    if code != 0:
        raise CudaError("cuModuleGetGlobal(RK_BITS) failed", code=code)
    rk_bits = _build_const_rk_bits_u32()
    rk_nbytes = ctypes.sizeof(rk_bits)
    if int(nbytes.value) < rk_nbytes:
        raise RuntimeError(
            f"RK_BITS symbol too small: {int(nbytes.value)} < {rk_nbytes}"
        )
    cuda.memcpy_htod(int(dptr.value), rk_bits, rk_nbytes)


@lru_cache(maxsize=8)
def _compile_coalesced_words_cubin(sm: str, post_op: str) -> bytes:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc not found on PATH")

    bench = _load_bench_module()
    mapped, _lop3_count = bench._build_bp128_mapped()
    sbox_cuda = bench._emit_sbox_inline_cuda(
        mapped, func_name="sbox_bp128_lop3_inline", noinline=False
    )
    src = bench._kernel_cu_source_replacement_coalesced(sbox_cuda)
    if post_op == "xor-accumulate":
        old = "            out_ptr[plane * threads + t] = sr[b][bit] ^ RK_BITS[10][b][bit];"
        new = (
            "            uint32_t outv = sr[b][bit] ^ RK_BITS[10][b][bit];\n"
            "            out_ptr[plane * threads + t] ^= outv;"
        )
        if old not in src:
            raise RuntimeError(
                "failed to patch coalesced output store for xor-accumulate"
            )
        src = src.replace(old, new)
    elif post_op != "store":
        raise ValueError(f"unsupported post_op={post_op}")

    with tempfile.TemporaryDirectory(prefix="bp128_words_cubin_") as td:
        work_dir = Path(td)
        cu_path = work_dir / "kernel.cu"
        cubin_path = work_dir / "kernel.cubin"
        cu_path.write_text(src, encoding="utf-8")
        run = subprocess.run(
            [
                nvcc,
                "-cubin",
                "-O3",
                "-std=c++17",
                f"-arch={sm}",
                str(cu_path),
                "-o",
                str(cubin_path),
            ],
            capture_output=True,
            text=True,
        )
        if run.returncode != 0:
            raise RuntimeError(
                "nvcc failed for coalesced words kernel\n"
                f"STDOUT:\n{run.stdout}\nSTDERR:\n{run.stderr}"
            )
        return cubin_path.read_bytes()


def _run_external_const_words(
    threads: int, ctr_group: int, block: int, reps: int, sm: str, post_op: str
) -> tuple[str, float | None, float | None, str]:
    threads_eff = threads * ctr_group
    words = threads_eff * 128
    host_in = (ctypes.c_uint32 * words)()
    host_out = (ctypes.c_uint32 * words)()

    pt = bytes.fromhex("00112233445566778899aabbccddeeff")
    ct = bytes.fromhex("69c4e0d86a7b0430d8cdb78070b4c55a")
    seed = bytes.fromhex("a55ac33c96690ff01221344356657887")
    _seed_bitplanes_words(host_in, threads_eff, pt)
    if post_op == "store":
        _seed_bitplanes_words(host_out, threads_eff, bytes(16))
        expected = ct
    elif post_op == "xor-accumulate":
        _seed_bitplanes_words(host_out, threads_eff, seed)
        expected = bytes((seed[i] ^ ct[i]) & 0xFF for i in range(16))
    else:
        return ("FAIL", None, None, f"unsupported post_op={post_op}")

    try:
        cubin = _compile_coalesced_words_cubin(sm=sm, post_op=post_op)
    except Exception as exc:
        return ("FAIL", None, None, f"cubin build failed: {exc}")

    cuda = Cuda()
    cuda.init()
    dev = cuda.device(0)
    ctx = cuda.ctx_create(dev)
    try:
        mod = cuda.module_load_data(cubin)
        _upload_rk_bits_symbol(cuda, mod)
        fn = cuda.module_get_function(mod, "aes10_bp128_kernel")
        d_in = cuda.mem_alloc(ctypes.sizeof(host_in))
        d_out = cuda.mem_alloc(ctypes.sizeof(host_out))
        try:
            cuda.memcpy_htod(d_in, host_in, ctypes.sizeof(host_in))
            cuda.memcpy_htod(d_out, host_out, ctypes.sizeof(host_out))

            arg_in = ctypes.c_uint64(d_in)
            arg_out = ctypes.c_uint64(d_out)
            arg_threads = ctypes.c_uint32(threads_eff)
            args = [
                ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(arg_threads), ctypes.c_void_p),
            ]
            grid = ((threads_eff + block - 1) // block, 1, 1)

            cuda.launch_async(fn, grid=grid, block=(block, 1, 1), args=args)
            cuda.synchronize()
            cuda.memcpy_dtoh(host_out, d_out, ctypes.sizeof(host_out))
            got = _decode_first_lane_words(host_out, threads_eff)
            if got != expected:
                return (
                    "FAIL",
                    None,
                    None,
                    f"coalesced words mismatch expected={expected.hex()} got={got.hex()}",
                )

            cuda.memcpy_htod(d_out, host_out, ctypes.sizeof(host_out))
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
    except Exception as exc:
        return ("FAIL", None, None, f"runtime failed: {exc}")
    finally:
        cuda.ctx_destroy(ctx)

    seconds = ms / 1000.0
    evals_per_sec = (threads_eff * 32 * reps) / seconds
    mib_s = evals_per_sec * 16.0 / (1024.0 * 1024.0)
    return (
        "PASS",
        evals_per_sec / 1e9,
        mib_s,
        f"replacement_coalesced_tuned ({post_op}, bitplanes->words)",
    )


def _parse_block_candidates(block: int, autotune_blocks: str) -> tuple[int, ...]:
    if autotune_blocks.strip():
        vals = []
        for token in autotune_blocks.split(","):
            token = token.strip()
            if not token:
                continue
            vals.append(int(token))
        if vals:
            uniq = list(dict.fromkeys(vals))
            return tuple(uniq)
    return (block,)


def _run_external_const_byteio(
    threads: int, ctr_group: int, reps: int, sm: str
) -> tuple[str, float | None, float | None, str]:
    threads_eff = threads * ctr_group
    cmd = [
        sys.executable,
        "scripts/bench_aes10_bp128_byteio_cuda.py",
        "--threads",
        str(threads_eff),
        "--block",
        "128",
        "--reps",
        str(reps),
        "--sm",
        sm,
        "--check",
    ]
    run = subprocess.run(cmd, capture_output=True, text=True)
    text = (run.stdout or "") + "\n" + (run.stderr or "")
    if run.returncode != 0:
        return ("FAIL", None, None, f"external runner failed: rc={run.returncode}")
    ok = "PASS: AES-128 known-answer check matched" in text
    eval_b, mib_s = _parse_best_eval_and_mib(text, "full_pipeline")
    return (
        "PASS" if ok else "FAIL",
        eval_b,
        mib_s,
        "bench_aes10_bp128_byteio_cuda.py full_pipeline (bytes->bytes)",
    )


def _run_one(
    key_bits: int,
    ctr_group: int,
    key_source: str,
    in_layout: str,
    out_layout: str,
    post_op: str,
    threads: int,
    blocks: tuple[int, ...],
    reps: int,
    sm: str,
) -> AxisResult:
    supported, notes = _support_for_combo(
        key_bits=key_bits,
        ctr_group=ctr_group,
        key_source=key_source,
        in_layout=in_layout,
        out_layout=out_layout,
        post_op=post_op,
    )
    if not supported:
        return AxisResult(
            key_bits=key_bits,
            ctr_group=ctr_group,
            key_source=key_source,
            in_layout=in_layout,
            out_layout=out_layout,
            post_op=post_op,
            supported=False,
            correctness="SKIP",
            eval_b=None,
            mib_s=None,
            best_block=None,
            notes=notes,
        )

    block = blocks[0]
    if (
        key_bits == 128
        and key_source == "const_key"
        and in_layout == "bitplanes"
        and out_layout == "words"
        and post_op in {"store", "xor-accumulate"}
    ):
        best_eval = None
        best_mib = None
        best_block = None
        best_notes = None
        best_correctness = "FAIL"
        for cand_block in blocks:
            correctness, eval_b, mib_s, extra_notes = _run_external_const_words(
                threads=threads,
                ctr_group=ctr_group,
                block=cand_block,
                reps=reps,
                sm=sm,
                post_op=post_op,
            )
            if correctness != "PASS" or eval_b is None:
                if best_notes is None:
                    best_notes = extra_notes
                continue
            if best_eval is None or eval_b > best_eval:
                best_eval = eval_b
                best_mib = mib_s
                best_block = cand_block
                best_notes = extra_notes
                best_correctness = correctness
        return AxisResult(
            key_bits=key_bits,
            ctr_group=ctr_group,
            key_source=key_source,
            in_layout=in_layout,
            out_layout=out_layout,
            post_op=post_op,
            supported=True,
            correctness=best_correctness,
            eval_b=best_eval,
            mib_s=best_mib,
            best_block=best_block,
            notes=best_notes or "all block candidates failed",
        )

    if (
        key_bits == 128
        and key_source == "const_key"
        and in_layout == "bytes"
        and out_layout == "bytes"
        and post_op == "store"
    ):
        correctness, eval_b, mib_s, extra_notes = _run_external_const_byteio(
            threads=threads,
            ctr_group=ctr_group,
            reps=reps,
            sm=sm,
        )
        return AxisResult(
            key_bits=key_bits,
            ctr_group=ctr_group,
            key_source=key_source,
            in_layout=in_layout,
            out_layout=out_layout,
            post_op=post_op,
            supported=True,
            correctness=correctness,
            eval_b=eval_b,
            mib_s=mib_s,
            best_block=None,
            notes=extra_notes,
        )

    meta = build_variant_metadata(
        key_bits=key_bits,
        ctr_group=ctr_group,
        key_source=key_source,  # type: ignore[arg-type]
        io_layout="plane-major4",
    )
    post_mode = "xor_accumulate" if post_op == "xor-accumulate" else "store"
    ok, check_details = check_variant_correctness(
        meta=meta,
        threads=max(threads // 8, 256),
        block=block,
        sm=sm,
        post_op=post_mode,
    )
    eval_b = None
    mib_s = None
    best_block = None
    if ok:
        for cand_block in blocks:
            try:
                cand_eval, cand_mib = benchmark_variant(
                    meta=meta,
                    threads=threads,
                    block=cand_block,
                    reps=reps,
                    sm=sm,
                    shared_key=True,
                    post_op=post_mode,
                )
            except Exception:
                continue
            if eval_b is None or cand_eval > eval_b:
                eval_b = cand_eval
                mib_s = cand_mib
                best_block = cand_block
        if eval_b is None:
            ok = False
            check_details = "benchmark failed for all block candidates"

    return AxisResult(
        key_bits=key_bits,
        ctr_group=ctr_group,
        key_source=key_source,
        in_layout=in_layout,
        out_layout=out_layout,
        post_op=post_op,
        supported=True,
        correctness="PASS" if ok else "FAIL",
        eval_b=eval_b,
        mib_s=mib_s,
        best_block=best_block,
        notes=check_details,
    )


def _write_markdown(results: list[AxisResult], path: Path) -> None:
    supported = [row for row in results if row.supported]
    passed = [row for row in supported if row.correctness == "PASS"]
    failed = [row for row in supported if row.correctness == "FAIL"]
    unsupported = [row for row in results if not row.supported]

    lines = []
    lines.append("# BP128 AES Axis Sweep\n")
    lines.append(
        f"- total combinations: {len(results)}"
        f"\n- supported: {len(supported)}"
        f"\n- correctness PASS: {len(passed)}"
        f"\n- correctness FAIL: {len(failed)}"
        f"\n- unsupported: {len(unsupported)}\n"
    )
    lines.append(
        "| key_bits | ctr_group | key_source | in_layout | out_layout | post_op | supported | correctness | best_block | eval_B/s | MiB/s | notes |"
    )
    lines.append("|---:|---:|---|---|---|---|---|---|---:|---:|---:|---|")
    for row in results:
        eval_txt = "" if row.eval_b is None else f"{row.eval_b:.3f}"
        mib_txt = "" if row.mib_s is None else f"{row.mib_s:.2f}"
        blk_txt = "" if row.best_block is None else str(row.best_block)
        lines.append(
            f"| {row.key_bits} | {row.ctr_group} | {row.key_source} | "
            f"{row.in_layout} | {row.out_layout} | {row.post_op} | "
            f"{'yes' if row.supported else 'no'} | {row.correctness} | "
            f"{blk_txt} | {eval_txt} | {mib_txt} | {row.notes.replace('|', '/')} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=65536)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument(
        "--autotune-blocks",
        default="64,128,256",
        help="Comma-separated block sizes to sweep per supported mode; empty disables sweep.",
    )
    ap.add_argument("--reps", type=int, default=120)
    ap.add_argument("--sm", default="sm_61")
    ap.add_argument("--json-out", default="out/bp128_axes_results.json")
    ap.add_argument("--md-out", default="out/bp128_axes_results.md")
    args = ap.parse_args()

    rows: list[AxisResult] = []
    block_candidates = _parse_block_candidates(args.block, args.autotune_blocks)
    for key_bits in KEY_BITS:
        for ctr_group in CTR_GROUP:
            for key_source in KEY_SOURCE:
                for in_layout in IN_LAYOUT:
                    for out_layout in OUT_LAYOUT:
                        for post_op in POST_OP:
                            row = _run_one(
                                key_bits=key_bits,
                                ctr_group=ctr_group,
                                key_source=key_source,
                                in_layout=in_layout,
                                out_layout=out_layout,
                                post_op=post_op,
                                threads=args.threads,
                                blocks=block_candidates,
                                reps=args.reps,
                                sm=args.sm,
                            )
                            rows.append(row)
                            status = (
                                f"{row.correctness} {row.eval_b:.3f}B"
                                if row.eval_b is not None
                                else row.correctness
                            )
                            if row.best_block is not None:
                                status = f"{status} blk={row.best_block}"
                            print(
                                f"{row.key_bits}/g{row.ctr_group}/{row.key_source}/"
                                f"{row.in_layout}->{row.out_layout}/{row.post_op}: "
                                f"{'supported' if row.supported else 'unsupported'} {status}"
                            )

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps([asdict(row) for row in rows], indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(rows, Path(args.md_out))

    supported = sum(1 for row in rows if row.supported)
    passed = sum(1 for row in rows if row.correctness == "PASS")
    failed = sum(1 for row in rows if row.correctness == "FAIL")
    print(
        f"done: total={len(rows)} supported={supported} pass={passed} fail={failed} "
        f"json={args.json_out} md={args.md_out}"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
