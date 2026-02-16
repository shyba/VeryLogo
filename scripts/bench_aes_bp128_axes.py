#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stc.aes_bp128_variant_family import (
    benchmark_variant,
    build_variant_metadata,
    check_variant_correctness,
)

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
    if post_op not in {"store", "xor-accumulate"}:
        return (False, f"unsupported post_op {post_op}")

    if in_layout == "bitplanes":
        io_layout = "plane-major4"
    elif in_layout == "bytes":
        io_layout = "bytes"
    else:
        return (False, f"unsupported in_layout {in_layout}")

    meta = build_variant_metadata(
        key_bits=key_bits,
        ctr_group=ctr_group,
        key_source=key_source,  # type: ignore[arg-type]
        io_layout=io_layout,  # type: ignore[arg-type]
    )
    if not meta.supported:
        return (False, meta.notes)

    if out_layout == "bitplanes":
        if io_layout != "plane-major4":
            return (
                False,
                "bitplanes output requires in_layout=bitplanes (plane-major4 kernel)",
            )
    elif out_layout == "words":
        if io_layout != "plane-major4":
            return (False, "words output requires in_layout=bitplanes")
    elif out_layout == "bytes":
        if io_layout != "bytes":
            return (False, "bytes output requires in_layout=bytes")
    else:
        return (False, f"unsupported out_layout {out_layout}")

    return (True, "supported")


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

    io_layout = "bytes" if in_layout == "bytes" else "plane-major4"
    block = blocks[0]

    meta = build_variant_metadata(
        key_bits=key_bits,
        ctr_group=ctr_group,
        key_source=key_source,  # type: ignore[arg-type]
        io_layout=io_layout,  # type: ignore[arg-type]
    )
    post_mode = "xor_accumulate" if post_op == "xor-accumulate" else "store"
    try:
        ok, check_details = check_variant_correctness(
            meta=meta,
            threads=max(threads // 8, 256),
            block=block,
            sm=sm,
            post_op=post_mode,
            output_layout=out_layout,  # type: ignore[arg-type]
        )
    except Exception as exc:
        return AxisResult(
            key_bits=key_bits,
            ctr_group=ctr_group,
            key_source=key_source,
            in_layout=in_layout,
            out_layout=out_layout,
            post_op=post_op,
            supported=True,
            correctness="FAIL",
            eval_b=None,
            mib_s=None,
            best_block=None,
            notes=f"correctness check failed: {exc}",
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
                    output_layout=out_layout,  # type: ignore[arg-type]
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
