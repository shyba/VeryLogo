import argparse
import ctypes
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stc.backend_ptx import emit_ptx_steps
from stc.cuda_driver import Cuda
from stc.extract import extract_tick_ir
from stc.tick_ir import BitVecType, BoolType
from stc.yosys_json import load_design


def _words_for_type(t: object) -> int:
    if isinstance(t, BoolType):
        return 1
    if isinstance(t, BitVecType):
        return (t.width + 31) // 32
    raise TypeError("unsupported type")


def _port_offsets(
    ts: dict[str, object], order: list[str]
) -> tuple[dict[str, int], int]:
    off = 0
    offsets: dict[str, int] = {}
    for name in order:
        offsets[name] = off
        off += _words_for_type(ts[name])
    return offsets, off


def _pack_u32_words(x: int, words: int) -> list[int]:
    return [(x >> (32 * i)) & 0xFFFFFFFF for i in range(words)]


def _compile_yosys_json(out_json: Path) -> None:
    v = Path("fixtures/verilog/aes128_fixedkey_seq_lut.v")
    top = "aes128_fixedkey_seq_lut"
    script = (
        f"read_verilog -sv {v}; hierarchy -check -top {top}; proc; opt; opt_clean; "
        f"write_json {out_json}"
    )
    subprocess.run(["yosys", "-q", "-p", script], check=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1 << 18)
    p.add_argument("--ticks", type=int, default=11)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--mode", choices=["baseline", "fused"], default="fused")
    p.add_argument(
        "--runtime-loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Emit a true PTX loop for multi-step fused mode instead of compile-time "
            "step unrolling."
        ),
    )
    p.add_argument(
        "--assume-reset-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable const-state specialization for fused mode. "
            "Disable if specialization triggers type mismatch on a design."
        ),
    )
    p.add_argument("--pt", type=str, default="00112233445566778899aabbccddeeff")
    args = p.parse_args()

    pt = int(args.pt, 16)
    exp = int("69c4e0d86a7b0430d8cdb78070b4c55a", 16)

    tmp = Path("tmp_out")
    tmp.mkdir(parents=True, exist_ok=True)
    yosys_json = tmp / "aes128_fixedkey_seq_lut.json"
    _compile_yosys_json(yosys_json)
    design = load_design(yosys_json, top="aes128_fixedkey_seq_lut")
    ir = extract_tick_ir(design)

    cuda = Cuda()
    cuda.init()
    dev = cuda.device(0)
    major, minor = cuda.compute_capability(dev)
    sm = f"sm_{major}{minor}"

    if args.mode == "baseline":
        ptx = emit_ptx_steps(ir, sm=sm, steps=1)
    else:
        assume_reset = bool(args.assume_reset_state and not args.runtime_loop)
        ptx = emit_ptx_steps(
            ir,
            sm=sm,
            steps=int(args.ticks),
            assume_reset_state=assume_reset,
            runtime_step_loop=bool(args.runtime_loop),
        )

    input_order = sorted(ir.inputs.keys())
    output_order = sorted(ir.outputs.keys())
    state_order = sorted(ir.state.keys())
    in_offs, in_stride = _port_offsets(ir.inputs, input_order)
    _, out_stride = _port_offsets(ir.outputs, output_order)
    st_offs, st_stride = _port_offsets(ir.state, state_order)

    if input_order != ["pt"] or output_order != ["ct"]:
        raise RuntimeError("unexpected aes module interface")
    if in_stride != 4 or out_stride != 4:
        raise RuntimeError("unexpected aes block width")

    st128 = [
        k for k, t in ir.state.items() if isinstance(t, BitVecType) and t.width == 128
    ]
    if len(st128) != 1:
        raise RuntimeError("expected exactly one 128-bit state reg")
    st128_word_off = st_offs[st128[0]]

    ctx = cuda.ctx_create(dev)
    try:
        mod = cuda.module_load_ptx(ptx)
        fn = cuda.module_get_function(mod, "stc_eval")

        n = int(args.n)
        h_in = (ctypes.c_uint32 * (n * in_stride))()
        h_out = (ctypes.c_uint32 * (n * out_stride))()
        h_st0 = (ctypes.c_uint32 * (n * st_stride))()
        h_st1 = (ctypes.c_uint32 * (n * st_stride))()

        pt_words = _pack_u32_words(pt, 4)
        for i in range(n):
            base = i * in_stride + in_offs["pt"]
            for w in range(4):
                h_in[base + w] = ctypes.c_uint32(pt_words[w])

        d_in = cuda.mem_alloc(ctypes.sizeof(h_in))
        d_out = cuda.mem_alloc(ctypes.sizeof(h_out))
        d_st0 = cuda.mem_alloc(ctypes.sizeof(h_st0))
        d_st1 = cuda.mem_alloc(ctypes.sizeof(h_st1))
        try:
            cuda.memcpy_htod(d_in, h_in, ctypes.sizeof(h_in))
            cuda.memcpy_htod(d_out, h_out, ctypes.sizeof(h_out))
            cuda.memcpy_htod(d_st0, h_st0, ctypes.sizeof(h_st0))
            cuda.memcpy_htod(d_st1, h_st1, ctypes.sizeof(h_st1))

            block = (128, 1, 1)
            grid = ((n + block[0] - 1) // block[0], 1, 1)

            def launch(d_state_in: int, d_state_out: int) -> None:
                arg_in = ctypes.c_uint64(d_in)
                arg_st_in = ctypes.c_uint64(d_state_in)
                arg_out = ctypes.c_uint64(d_out)
                arg_st_out = ctypes.c_uint64(d_state_out)
                arg_n = ctypes.c_uint32(n)
                kernel_args = [
                    ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_st_in), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_st_out), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_n), ctypes.c_void_p),
                ]
                cuda.launch_async(fn, grid, block, kernel_args)

            if args.mode == "baseline":
                st_in = d_st0
                st_out = d_st1

                for _ in range(2):
                    launch(st_in, st_out)
                    st_in, st_out = st_out, st_in
                cuda.synchronize()

                t0 = time.perf_counter()
                for _ in range(int(args.ticks)):
                    launch(st_in, st_out)
                    st_in, st_out = st_out, st_in
                cuda.synchronize()
                t1 = time.perf_counter()

                sec = t1 - t0
                bytes_total = n * 16
                mib_s = (bytes_total / sec) / (1024 * 1024)
                print(
                    f"gpu_tick_aes128 mode=baseline n={n} ticks={args.ticks} sm={sm} time_s={sec:.6f} throughput_mib_s={mib_s:.2f}"
                )
                ct_ptr = st_in
            else:
                for _ in range(2):
                    launch(d_st0, d_st1)
                cuda.synchronize()

                reps = int(args.reps)
                t0 = time.perf_counter()
                for _ in range(reps):
                    launch(d_st0, d_st1)
                cuda.synchronize()
                t1 = time.perf_counter()

                sec = t1 - t0
                bytes_total = n * 16 * reps
                mib_s = (bytes_total / sec) / (1024 * 1024)
                print(
                    f"gpu_tick_aes128 mode=fused n={n} ticks={args.ticks} reps={reps} sm={sm} time_s={sec:.6f} throughput_mib_s={mib_s:.2f}"
                )
                ct_ptr = d_st1

            h_ct = (ctypes.c_uint32 * 4)()
            cuda.memcpy_dtoh(
                h_ct,
                ct_ptr + (st128_word_off * 4),
                ctypes.sizeof(h_ct),
            )
            got = 0
            for i, w in enumerate(h_ct):
                got |= int(w) << (32 * i)
            if got != exp:
                raise RuntimeError(f"cipher mismatch: got={got:032x} exp={exp:032x}")
        finally:
            cuda.mem_free(d_in)
            cuda.mem_free(d_out)
            cuda.mem_free(d_st0)
            cuda.mem_free(d_st1)
    finally:
        cuda.ctx_destroy(ctx)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
