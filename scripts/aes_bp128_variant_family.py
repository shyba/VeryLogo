#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stc.aes_bp128_variant_family import (
    Bp128DispatchRequest,
    Bp128Dispatcher,
    benchmark_variant,
    build_variant_metadata,
    check_variant_correctness,
    compile_variant_to_ptx,
    family_manifest_json,
)


def _add_variant_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--key-bits", type=int, choices=[128, 192, 256], required=True)
    parser.add_argument("--ctr-group", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument(
        "--key-source",
        choices=["masterkey_soa", "expanded_rk_soa", "const_key"],
        default="masterkey_soa",
    )
    parser.add_argument(
        "--io-layout", choices=["plane-major4", "bytes"], default="plane-major4"
    )


def _add_post_op_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--post-op",
        choices=["store", "xor-accumulate"],
        default="store",
    )


def _cmd_manifest(_: argparse.Namespace) -> int:
    print(family_manifest_json(indent=2))
    return 0


def _cmd_dispatch(args: argparse.Namespace) -> int:
    dispatcher = Bp128Dispatcher()
    req = Bp128DispatchRequest(
        key_bits=args.key_bits,
        key_source=args.key_source,
        io_layout=args.io_layout,
        max_ctr_group=args.max_ctr_group,
    )
    picked = dispatcher.select(req)
    print(
        "selected:",
        picked.variant_id,
        f"kernel={picked.kernel_name}",
        f"supported={int(picked.supported)}",
        f"notes={picked.notes}",
    )
    return 0


def _cmd_emit_ptx(args: argparse.Namespace) -> int:
    meta = build_variant_metadata(
        key_bits=args.key_bits,
        ctr_group=args.ctr_group,
        key_source=args.key_source,
        io_layout=args.io_layout,
    )
    if not meta.supported:
        raise SystemExit(f"unsupported variant: {meta.variant_id} ({meta.notes})")
    post_op = "xor_accumulate" if args.post_op == "xor-accumulate" else "store"
    ptx = compile_variant_to_ptx(meta, sm=args.sm, post_op=post_op)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(ptx, encoding="utf-8")
    print(f"wrote {out_path} ({len(ptx)} bytes)")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    meta = build_variant_metadata(
        key_bits=args.key_bits,
        ctr_group=args.ctr_group,
        key_source=args.key_source,
        io_layout=args.io_layout,
    )
    if not meta.supported:
        raise SystemExit(f"unsupported variant: {meta.variant_id} ({meta.notes})")
    eval_b, mib_s = benchmark_variant(
        meta=meta,
        threads=args.threads,
        block=args.block,
        reps=args.reps,
        sm=args.sm,
        shared_key=not args.per_thread_keys,
        post_op="xor_accumulate" if args.post_op == "xor-accumulate" else "store",
    )
    print(
        f"{meta.variant_id}: {eval_b:.3f}B eval/s, {mib_s:.2f} MiB/s "
        f"(threads={args.threads} ctr_group={args.ctr_group} block={args.block} "
        f"reps={args.reps} per_thread_keys={int(args.per_thread_keys)} "
        f"post_op={args.post_op})"
    )
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    meta = build_variant_metadata(
        key_bits=args.key_bits,
        ctr_group=args.ctr_group,
        key_source=args.key_source,
        io_layout=args.io_layout,
    )
    if not meta.supported:
        raise SystemExit(f"unsupported variant: {meta.variant_id} ({meta.notes})")
    ok, details = check_variant_correctness(
        meta=meta,
        threads=args.threads,
        block=args.block,
        sm=args.sm,
        post_op="xor_accumulate" if args.post_op == "xor-accumulate" else "store",
    )
    print(f"{meta.variant_id}: {'PASS' if ok else 'FAIL'} {details}")
    return 0 if ok else 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dynamic variant generator from fast BP128 CUDA kernel templates."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_manifest = sub.add_parser("manifest", help="Print variant metadata JSON")
    p_manifest.set_defaults(run=_cmd_manifest)

    p_dispatch = sub.add_parser("dispatch", help="Select best variant for a request")
    p_dispatch.add_argument(
        "--key-bits", type=int, choices=[128, 192, 256], required=True
    )
    p_dispatch.add_argument(
        "--key-source",
        choices=["masterkey_soa", "expanded_rk_soa", "const_key"],
        default="masterkey_soa",
    )
    p_dispatch.add_argument(
        "--io-layout", choices=["plane-major4", "bytes"], default="plane-major4"
    )
    p_dispatch.add_argument("--max-ctr-group", type=int, choices=[1, 2, 4], default=4)
    p_dispatch.set_defaults(run=_cmd_dispatch)

    p_emit = sub.add_parser("emit-ptx", help="Generate and compile one variant to PTX")
    _add_variant_args(p_emit)
    _add_post_op_arg(p_emit)
    p_emit.add_argument("--sm", default="sm_61")
    p_emit.add_argument("--out", required=True)
    p_emit.set_defaults(run=_cmd_emit_ptx)

    p_bench = sub.add_parser("bench", help="Benchmark one supported variant")
    _add_variant_args(p_bench)
    _add_post_op_arg(p_bench)
    p_bench.add_argument("--sm", default="sm_61")
    p_bench.add_argument("--threads", type=int, default=65536)
    p_bench.add_argument("--block", type=int, default=64)
    p_bench.add_argument("--reps", type=int, default=200)
    p_bench.add_argument(
        "--per-thread-keys",
        action="store_true",
        help="Use unique key material per thread (default is one shared key for peak throughput)",
    )
    p_bench.set_defaults(run=_cmd_bench)

    p_check = sub.add_parser("check", help="Run AES correctness check for one variant")
    _add_variant_args(p_check)
    _add_post_op_arg(p_check)
    p_check.add_argument("--sm", default="sm_61")
    p_check.add_argument("--threads", type=int, default=256)
    p_check.add_argument("--block", type=int, default=64)
    p_check.set_defaults(run=_cmd_check)

    args = parser.parse_args()
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
