#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

from stc.extract import extract_tick_ir
from stc.gate_ternary_synth import apply_ternary_synthesis
from stc.tick_ir_to_circuit_state import (
    lower_tick_ir_to_circuit_state,
    rust_lower_tick_ir_to_circuit_state,
)
from stc.yosys_json import load_design
from stc.sched import PTX, list_schedule, pipelined_schedule
from stc.sched.liveness import compute_live_ranges, max_live
from stc.sched.regalloc import allocate_registers
from stc.sched.target import TargetModel
from stc.mir.lower import circuit_to_mir
from stc.mir.emit_spirv_proper import emit_vulkan_compute


def _run_yosys_flatten(
    verilog: Path, extra_verilog: Path, top: str, out_json: Path
) -> None:
    script = (
        f"read_verilog -sv {verilog} {extra_verilog}; hierarchy -top {top}; "
        "proc; flatten; opt; opt_clean; write_json " + str(out_json)
    )
    subprocess.run(["yosys", "-q", "-p", script], check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--verilog",
        type=Path,
        default=Path("fixtures/verilog/aes128_fixedkey_seq_modular.v"),
    )
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument(
        "--kernel-loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run rounds in a true in-kernel loop (single host dispatch round). "
            "Enabled by default for AES mode when rounds > 1."
        ),
    )
    ap.add_argument(
        "--loop-state-words",
        type=int,
        default=0,
        help="Override loop-carried state words (0 = derive from `st` width).",
    )
    ap.add_argument(
        "--loop-output-state-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "In loop mode, store only final state words (reduces output traffic). "
            "Currently experimental on some Vulkan drivers."
        ),
    )
    ap.add_argument("--threads", type=int, default=65536)
    ap.add_argument("--local-x", type=int, default=64)
    ap.add_argument("--local-y", type=int, default=1)
    ap.add_argument("--local-z", type=int, default=1)
    ap.add_argument(
        "--tid-mode",
        choices=["full", "xy", "xonly"],
        default="full",
        help="Linear thread-id mode used in SPIR-V address calculation",
    )
    ap.add_argument(
        "--registers",
        type=int,
        default=0,
        help="Force register budget for MIR alloc (0 = auto peak_live+8)",
    )
    ap.add_argument(
        "--allow-spills",
        action="store_true",
        default=False,
        help="Allow MIR spills (SPIR-V emitter now supports Load/Store spills).",
    )
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--bytes-per-eval", type=int, default=16)
    ap.add_argument(
        "--mode",
        choices=["aes", "passthrough", "xor"],
        default="aes",
        help="Circuit mode: aes (round pair) or passthrough (rk/st concat)",
    )
    ap.add_argument(
        "--serial-schedule",
        action="store_true",
        help="Force single-issue scheduling to debug correctness",
    )
    ap.add_argument(
        "--scheduler",
        choices=["list", "pipelined", "serial"],
        default="list",
        help="Scheduler policy (default: list). --serial-schedule still forces serial.",
    )
    ap.add_argument(
        "--max-live-pressure",
        type=int,
        default=None,
        help="Optional live-pressure cap for list scheduler.",
    )
    ap.add_argument(
        "--dump-words",
        type=int,
        default=0,
        help="Dump first N output words per buffer when checking",
    )
    ap.add_argument(
        "--dump-offset",
        type=int,
        default=0,
        help="Start offset for dump in output words",
    )
    ap.add_argument(
        "--check-threads",
        type=int,
        default=0,
        help="Threads to validate in correctness mode (0 = all effective threads)",
    )
    ap.add_argument(
        "--ternary-synth",
        action="store_true",
        default=False,
        help="Apply local ternary synthesis before scheduling.",
    )
    args = ap.parse_args()

    if not args.verilog.exists():
        raise SystemExit(f"Missing verilog: {args.verilog}")

    with tempfile.TemporaryDirectory(prefix="aes_round_vk_") as td:
        td = Path(td)
        round_v = td / "aes_round.v"
        if args.mode == "aes":
            round_v.write_text(
                """
module aes_round_pair(
  input  logic [127:0] st,
  input  logic [127:0] rk,
  output logic [255:0] out
);
  wire [127:0] sb;
  wire [127:0] sr;
  wire [127:0] mc;
  aes_sub_bytes u_sb(.s(st), .o(sb));
  aes_shift_rows u_sr(.s(sb), .o(sr));
  aes_mix_columns u_mc(.s(sr), .o(mc));
  // out[127:0] = state, out[255:128] = round key (pass-through)
  assign out = {rk, mc ^ rk};
endmodule
""",
                encoding="utf-8",
            )
            extra_verilog = args.verilog
        elif args.mode == "passthrough":
            round_v.write_text(
                """
module aes_round_pair(
  input  logic [127:0] st,
  input  logic [127:0] rk,
  output logic [255:0] out
);
  // passthrough: out[127:0] = st, out[255:128] = rk
  assign out = {rk, st};
endmodule
""",
                encoding="utf-8",
            )
            extra_verilog = None
        else:
            round_v.write_text(
                """
module aes_round_pair(
  input  logic [127:0] st,
  input  logic [127:0] rk,
  output logic [255:0] out
);
  // simple xor: out[127:0] = st ^ rk, out[255:128] = rk
  assign out = {rk, st ^ rk};
endmodule
""",
                encoding="utf-8",
            )
            extra_verilog = None
        json_path = td / "aes_round.json"

        t0 = time.perf_counter()
        if extra_verilog is None:
            script = (
                f"read_verilog -sv {round_v}; hierarchy -top aes_round_pair; "
                f"proc; flatten; opt; opt_clean; write_json {json_path}"
            )
            subprocess.run(["yosys", "-q", "-p", script], check=True)
        else:
            _run_yosys_flatten(extra_verilog, round_v, "aes_round_pair", json_path)
        print(f"[aes_round] yosys: {time.perf_counter()-t0:.2f}s", flush=True)

        design = load_design(json_path, top="aes_round_pair")
        t0 = time.perf_counter()
        tick_ir = extract_tick_ir(design)
        print(f"[aes_round] extract: {time.perf_counter()-t0:.2f}s", flush=True)

        t0 = time.perf_counter()
        rust_result = rust_lower_tick_ir_to_circuit_state(tick_ir)
        if rust_result is None:
            circuit, layout = lower_tick_ir_to_circuit_state(tick_ir)
            lower_path = "python"
        else:
            circuit, layout = rust_result
            lower_path = "rust"
        print(
            f"[aes_round] lower({lower_path}): {time.perf_counter()-t0:.2f}s "
            f"gates={len(circuit.gates)}",
            flush=True,
        )
        st_info = layout.inputs.get("st") if layout else None
        rk_info = layout.inputs.get("rk") if layout else None
        out_info = layout.outputs.get("out") if layout else None
        if args.ternary_synth:
            t0 = time.perf_counter()
            circuit, ts = apply_ternary_synthesis(circuit)
            print(
                f"[aes_round] ternary_synth: {time.perf_counter()-t0:.2f}s "
                f"gates={ts.gates_before}->{ts.gates_after} "
                f"ternary={ts.ternary_gates_created}",
                flush=True,
            )

        gates = list(circuit.gates)
        input_bits = circuit.input_bits
        outputs = list(circuit.outputs)

        t0 = time.perf_counter()
        sched_target = PTX
        sched_name = "serial" if args.serial_schedule else args.scheduler
        if sched_name == "serial":
            sched_target = TargetModel(
                name="vulkan_serial",
                registers=PTX.registers,
                issue_width=1,
                latencies=PTX.latencies,
                throughput={},
            )
            schedule = list_schedule(gates, input_bits, outputs, sched_target)
        elif sched_name == "pipelined":
            schedule = pipelined_schedule(gates, input_bits, outputs, sched_target)
        else:
            schedule = list_schedule(
                gates,
                input_bits,
                outputs,
                sched_target,
                max_live_pressure=args.max_live_pressure,
            )
        live_ranges = compute_live_ranges(schedule, gates, input_bits, outputs)
        peak = max_live(live_ranges)
        reg_budget = args.registers if args.registers > 0 else max(peak + 8, 32)
        target = TargetModel(
            name="vulkan",
            registers=reg_budget,
            issue_width=sched_target.issue_width,
            latencies=sched_target.latencies,
            throughput=sched_target.throughput,
        )
        allocation = allocate_registers(
            live_ranges,
            schedule,
            target.registers,
            gates=gates,
            input_bits=input_bits,
            outputs=outputs,
        )
        if allocation.spills:
            if not args.allow_spills:
                raise SystemExit(
                    f"Vulkan MIR has spills ({len(allocation.spills)}). "
                    "Use --allow-spills or increase --registers."
                )
            print(
                f"[aes_round] warning: using {len(allocation.spills)} spill slots",
                flush=True,
            )
        print(
            f"[aes_round] schedule+alloc: {time.perf_counter()-t0:.2f}s "
            f"peak_live={peak} regs={target.registers} spills={len(allocation.spills)}",
            flush=True,
        )

        tid_mode = args.tid_mode
        if tid_mode == "xonly" and (args.local_y != 1 or args.local_z != 1):
            print(
                "[aes_round] warning: xonly requires local_y=local_z=1; "
                "falling back to tid_mode=full",
                flush=True,
            )
            tid_mode = "full"
        if tid_mode == "xy" and args.local_z != 1:
            print(
                "[aes_round] warning: xy requires local_z=1; "
                "falling back to tid_mode=full",
                flush=True,
            )
            tid_mode = "full"

        t0 = time.perf_counter()
        mir = circuit_to_mir(circuit, schedule, allocation)
        kernel_loop = (
            bool(args.kernel_loop)
            and args.mode == "aes"
            and args.rounds > 1
            and st_info is not None
            and out_info is not None
        )
        if kernel_loop:
            state_words = (
                int(args.loop_state_words)
                if args.loop_state_words > 0
                else int(st_info.get("width", 0))
            )
            if state_words <= 0:
                raise SystemExit("loop mode could not determine `st` width")
            state_in_off = int(st_info["lsb"])
            state_out_off = int(out_info["lsb"])
            output_words = (
                state_words
                if bool(args.loop_output_state_only)
                else len(mir.output_regs)
            )
            kernel_steps = int(args.rounds)
            host_rounds = 1
        else:
            state_words = 0
            state_in_off = 0
            state_out_off = 0
            output_words = len(mir.output_regs)
            kernel_steps = 1
            host_rounds = int(args.rounds)

        spirv_asm = emit_vulkan_compute(
            mir,
            allocation,
            local_size=(args.local_x, args.local_y, args.local_z),
            input_stride=len(mir.input_regs),
            output_stride=output_words,
            tid_linear_mode=tid_mode,
            step_count=kernel_steps,
            loop_state_words=state_words,
            loop_state_input_offset=state_in_off,
            loop_state_output_offset=state_out_off,
            loop_output_state_only=bool(args.loop_output_state_only),
        )
        if kernel_loop:
            print(
                "[aes_round] kernel loop: "
                f"steps={kernel_steps} state_words={state_words} "
                f"state_in_off={state_in_off} state_out_off={state_out_off} "
                f"output_words={output_words} state_only={args.loop_output_state_only}",
                flush=True,
            )
        print(f"[aes_round] emit spirv: {time.perf_counter()-t0:.2f}s", flush=True)

        spvasm_path = td / "aes_round.spvasm"
        spv_path = td / "aes_round.spv"
        spvasm_path.write_text(spirv_asm, encoding="utf-8")
        subprocess.run(
            ["spirv-as", str(spvasm_path), "-o", str(spv_path)],
            check=True,
        )

        bench_cpp = Path("scripts/bench_circuit_vulkan.cpp")
        if not bench_cpp.exists():
            raise SystemExit("Missing scripts/bench_circuit_vulkan.cpp")
        bench_bin = td / "bench_vulkan"
        subprocess.run(
            [
                "g++",
                "-O3",
                "-std=c++17",
                "-I.",
                "-o",
                str(bench_bin),
                str(bench_cpp),
                "-lvulkan",
            ],
            check=True,
        )

        if rk_info is None or out_info is None:
            raise SystemExit("Missing rk/out layout info for correctness check")
        rk_off = int(rk_info["lsb"])
        out_off = int(out_info["lsb"]) + 128
        st_off = int(st_info["lsb"]) if (args.mode == "aes" and st_info is not None) else -1
        state_out_off = int(out_info["lsb"]) if args.mode == "aes" else -1
        check_flag = 1 if (args.iters == 1 and host_rounds == 1 and kernel_steps == 1) else 0
        if check_flag == 0:
            print(
                "[aes_round] correctness check disabled (requires one host round and one kernel step)",
                flush=True,
            )

        cmd = [
            str(bench_bin),
            str(spv_path),
            str(args.iters),
            str(args.threads),
            str(args.local_x),
            str(args.local_y),
            str(args.local_z),
            str(len(mir.input_regs)),
            str(output_words),
            str(host_rounds),
            str(args.lanes),
            str(args.bytes_per_eval),
            str(check_flag),
            str(rk_off),
            str(out_off),
            "128",
            str(args.check_threads),
            str(args.dump_words),
            str(args.dump_offset),
            str(kernel_steps),
            str(0 if kernel_loop else 1),
            str(st_off),
            str(state_out_off),
        ]
        try:
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            print(result.stdout)
        except subprocess.CalledProcessError as exc:
            print(exc.stdout)
            print(exc.stderr)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
