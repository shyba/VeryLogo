from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from stc.autotune import autotune_configuration, write_autotune_results
from stc.backend_avr import emit_avr_c
from stc.backend_futhark import (
    compute_futhark_source_metrics,
    emit_futhark,
    emit_futhark_manifest,
    resolve_futhark_mode,
)
from stc.avr_project import emit_avr_project
from stc.backend_sched import generate_scheduled_code, get_schedule_stats
from stc.circuit_synth import CircuitState
from stc.extract import extract_tick_ir
from stc.infer_simd import infer_simd_types
from stc.io_map import default_io_map, load_io_map, validate_io_map
from stc.lowering_coordinator import (
    coordinate_lowering,
    LoweringChoice,
    write_lowering_choice_report,
)
from stc.metrics import compute_metrics
from stc.packed_circuit import PackedCircuitState
from stc.reduce import optimize_tick_ir
from stc.tick_ir_to_circuit_state import lower_tick_ir_to_circuit_state
from stc.tick_ir_to_packed_circuit_state import (
    PackedLoweringError,
    PackedWordLayout,
    lower_tick_ir_to_packed_circuit_state,
)
from stc.tick_ir import SimdType, TickIR
from stc.tick_ir_validate import validate_tick_ir
from stc.yosys_json import load_design
from stc.yosys_frontend import run_yosys


def _has_simd_types(ir: TickIR) -> bool:
    for t in (
        list(ir.inputs.values()) + list(ir.outputs.values()) + list(ir.state.values())
    ):
        if isinstance(t, SimdType):
            return True
    return False


def run_pipeline(
    input_path: Path,
    out_dir: Path,
    *,
    top: str | None = None,
    bound: int = 8,
    infer_simd: bool = False,
    autovec: bool = False,
    autovec_timeout_ms: int = 200,
    superopt: bool = False,
    superopt_max_nodes: int = 6,
    superopt_timeout_ms: int = 200,
    no_backend: bool = False,
    use_synth: bool = False,
    use_abc_lut3: bool = False,
    abc_lut3_depth: int | None = None,
    use_abc_lut3_aggressive: bool = False,
    abc_lut3_aggressive_threshold: int | None = None,
    abc_lut3_aggressive_max_live_pressure: int | None = None,
    io_map: Path | None = None,
    avr_project: bool = False,
    backend: str = "generic",
    futhark_mode: str = "auto",
    ternary_mapping: bool | None = None,
    fuse_ticks: int = 1,
    fuse_mode: str = "final",
    fuse_input_policy: str = "shared",
    fuse_budget_max_nodes: int = 1 << 60,
    fuse_budget_max_depth: int = 1 << 60,
    fuse_budget_max_step_ms: int | None = None,
    use_regions: bool = False,
    region_max_gates: int = 50000,
    force_bitsliced: bool = False,
    force_packed: bool = False,
    packed_bitslice: bool = False,
    region_max_boundary: int = 8192,
    autotune: bool = False,
    autotune_seed: int = 42,
    autotune_budget_ms: int = 60000,
    autotune_candidates: int = 8,
    arith_classify: bool = True,
    mul_div_max_width: int = 32,
    dump_arith_report: Path | None = None,
    max_live_pressure: int | None = None,
    scheduler: str | None = None,
    bounded_state_opt: bool | None = None,
) -> None:
    timing_enabled = os.environ.get("STC_TIMING", "0").lower() not in {
        "0",
        "false",
        "no",
    }

    def _log_timing(label: str, start: float) -> None:
        if timing_enabled:
            elapsed = time.perf_counter() - start
            print(f"[timing] {label}: {elapsed:.3f}s")

    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.suffix == ".v" or input_path.is_dir():
        if use_synth and use_abc_lut3:
            raise ValueError("Cannot use both --use-synth and --abc-lut3")
        if use_abc_lut3_aggressive and use_synth:
            raise ValueError("Cannot use both --use-synth and --abc-lut3-aggressive")
        if use_abc_lut3_aggressive and use_abc_lut3:
            raise ValueError("Cannot use both --abc-lut3 and --abc-lut3-aggressive")
        normalized_json = out_dir / "normalized.json"
        abc_script = None
        if use_abc_lut3_aggressive:
            abc_script = out_dir / "abc_lut3_aggressive.script"
            abc_script.write_text(
                "\n".join(
                    [
                        "strash",
                        "&get -n",
                        "&fraig -x",
                        "&put",
                        "scorr",
                        "dc2",
                        "dretime",
                        "strash",
                        "dch -f",
                        "if -K 3",
                        "mfs2",
                        "&get -n",
                        "&fraig -x",
                        "&put",
                        "scorr",
                        "dc2",
                        "dretime",
                        "strash",
                        "dch -f",
                        "if -K 3",
                        "mfs2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        t0 = time.perf_counter()
        run_yosys(
            input_path,
            normalized_json,
            top=top,
            output_script=out_dir / "normalized.ys",
            use_synth=use_synth,
            use_abc_lut3=use_abc_lut3,
            abc_lut3_depth=abc_lut3_depth,
            abc_script=abc_script,
        )
        _log_timing("yosys", t0)
    else:
        normalized_json = input_path

    t0 = time.perf_counter()
    design = load_design(normalized_json, top=top)
    _log_timing("load_design", t0)
    t0 = time.perf_counter()
    tick_ir = extract_tick_ir(design)
    _log_timing("extract_tick_ir", t0)
    t0 = time.perf_counter()
    validate_tick_ir(tick_ir)
    _log_timing("validate_tick_ir", t0)
    if (
        (input_path.suffix == ".v" or input_path.is_dir())
        and not use_synth
        and not use_abc_lut3
        and not use_abc_lut3_aggressive
    ):
        threshold = abc_lut3_aggressive_threshold
        if threshold is None:
            env_val = os.environ.get("STC_ABC_LUT3_AGGRESSIVE_THRESHOLD")
            if env_val:
                try:
                    threshold = int(env_val)
                except ValueError:
                    threshold = None
        if threshold is not None and threshold > 0:
            metrics = compute_metrics(tick_ir)
            if metrics.ops_total >= threshold:
                abc_script = out_dir / "abc_lut3_aggressive.script"
                abc_script.write_text(
                    "\n".join(
                        [
                            "strash",
                            "&get -n",
                            "&fraig -x",
                            "&put",
                            "scorr",
                            "dc2",
                            "dretime",
                            "strash",
                            "dch -f",
                            "if -K 3",
                            "mfs2",
                            "&get -n",
                            "&fraig -x",
                            "&put",
                            "scorr",
                            "dc2",
                            "dretime",
                            "strash",
                            "dch -f",
                            "if -K 3",
                            "mfs2",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                t0 = time.perf_counter()
                run_yosys(
                    input_path,
                    normalized_json,
                    top=top,
                    output_script=out_dir / "normalized.ys",
                    use_synth=False,
                    use_abc_lut3=False,
                    abc_lut3_depth=None,
                    abc_script=abc_script,
                )
                _log_timing("yosys(abc_lut3_aggressive)", t0)
                t0 = time.perf_counter()
                design = load_design(normalized_json, top=top)
                _log_timing("load_design(abc_lut3_aggressive)", t0)
                t0 = time.perf_counter()
                tick_ir = extract_tick_ir(design)
                _log_timing("extract_tick_ir(abc_lut3_aggressive)", t0)
                t0 = time.perf_counter()
                validate_tick_ir(tick_ir)
                _log_timing("validate_tick_ir(abc_lut3_aggressive)", t0)
                if (
                    abc_lut3_aggressive_max_live_pressure is not None
                    and max_live_pressure is None
                ):
                    max_live_pressure = abc_lut3_aggressive_max_live_pressure
    if infer_simd or autovec:
        t0 = time.perf_counter()
        tick_ir = infer_simd_types(tick_ir)
        _log_timing("infer_simd_types", t0)
        t0 = time.perf_counter()
        validate_tick_ir(tick_ir)
        _log_timing("validate_tick_ir", t0)
    t0 = time.perf_counter()
    effective_bounded_state_opt = (
        bounded_state_opt
        if bounded_state_opt is not None
        else backend not in {"x86-avx2", "x86-avx512", "ptx", "futhark"}
    )
    reduced, arith_report = optimize_tick_ir(
        tick_ir,
        bound=bound,
        autovec=autovec,
        autovec_timeout_ms=autovec_timeout_ms,
        superopt=superopt,
        superopt_max_nodes=superopt_max_nodes,
        superopt_timeout_ms=superopt_timeout_ms,
        backend=backend,
        ternary_mapping=ternary_mapping,
        # Bounded state pruning/const-prop is useful for small bounded analyses,
        # but is not semantics-preserving for long-horizon sequential designs.
        bounded_state_opt=effective_bounded_state_opt,
        fuse_ticks=fuse_ticks,
        fuse_mode=fuse_mode,
        fuse_input_policy=fuse_input_policy,
        fuse_budget_max_nodes=fuse_budget_max_nodes,
        fuse_budget_max_depth=fuse_budget_max_depth,
        fuse_budget_max_step_ms=fuse_budget_max_step_ms,
        arith_classify=arith_classify,
        mul_div_max_width=mul_div_max_width,
    )
    _log_timing("optimize_tick_ir", t0)
    t0 = time.perf_counter()
    validate_tick_ir(reduced)
    _log_timing("validate_tick_ir(reduced)", t0)

    if dump_arith_report is not None and arith_report is not None:
        from stc.kv_bin import write_kv_bin

        write_kv_bin({k: float(v) for k, v in arith_report.items()}, dump_arith_report)

    write_bin = os.environ.get("STC_SKIP_ARTIFACTS", "0") not in {
        "1",
        "true",
        "yes",
    }
    if write_bin:
        from stc.tick_ir_bin2 import write_tick_ir_bin
        from stc.metrics_bin import write_metrics_bin

        t0 = time.perf_counter()
        write_tick_ir_bin(tick_ir, str(out_dir / "tick_ir.bin"))
        write_tick_ir_bin(reduced, str(out_dir / "reduced_tick_ir.bin"))
        write_metrics_bin(compute_metrics(tick_ir), out_dir / "metrics.bin")
        write_metrics_bin(compute_metrics(reduced), out_dir / "reduced_metrics.bin")
        _log_timing("write_artifacts_bin", t0)
    # `io_map.bin` and AVR artifacts only apply to the AVR backend. When
    # `--no-backend` is set, users may be targeting non-AVR outputs and the
    # reduced IR may have many outputs that cannot fit the default PORTB map.
    if no_backend:
        return

    if backend == "futhark":
        t0 = time.perf_counter()
        selected_futhark_mode, fallback_reason = resolve_futhark_mode(
            reduced, futhark_mode
        )
        fut_source = emit_futhark(
            reduced, module_name=reduced.name, mode=selected_futhark_mode
        )
        (out_dir / "circuit_futhark.fut").write_text(fut_source, encoding="utf-8")
        manifest = emit_futhark_manifest(
            reduced,
            module_name=reduced.name,
            mode=selected_futhark_mode,
            fallback_reason=fallback_reason,
        )
        (out_dir / "futhark_io_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        (out_dir / "futhark_source_metrics.json").write_text(
            json.dumps(compute_futhark_source_metrics(fut_source), indent=2) + "\n",
            encoding="utf-8",
        )
        _log_timing("emit_futhark", t0)
        return

    # Non-AVR scheduled backends: lower TickIR to CircuitState and emit scheduled code.
    # This enables full Verilog-derived designs (including wide state) to target AVX-512/PTX.
    if backend in {"x86-avx512", "x86-avx2", "ptx"}:
        target = {"x86-avx512": "avx512", "x86-avx2": "avx2", "ptx": "ptx"}[backend]
        if packed_bitslice and backend != "x86-avx512":
            raise ValueError("--packed-bitslice is only supported for x86-avx512")
        if backend == "ptx" and force_packed:
            raise ValueError("--force-packed is only supported for x86-avx512")
        if packed_bitslice and (force_packed or force_bitsliced):
            raise ValueError(
                "Cannot combine --packed-bitslice with --force-packed/--force-bitsliced"
            )
        if force_packed and force_bitsliced:
            raise ValueError(
                "Cannot use both --force-packed and --force-bitsliced simultaneously"
            )

        packed_layout: PackedWordLayout | None = None
        if packed_bitslice:
            from stc.packed_bitslice import bitslice_circuit_to_packed

            t0 = time.perf_counter()
            circuit_bits, layout_bits, base_choice = coordinate_lowering(
                reduced, prefer_packed=False, force_choice=True
            )
            packed, layout_w = bitslice_circuit_to_packed(circuit_bits, layout_bits)
            choice = LoweringChoice(
                path="packed_bitslice",
                reason="Bit-level lowering followed by packed bitslice conversion",
                unsupported=base_choice.unsupported,
                stats=base_choice.stats,
                gate_comparison=None,
            )
            _log_timing("coordinate_lowering(bitslice)", t0)
            write_lowering_choice_report(choice, out_dir, write_bin=write_bin)
            circuit = packed
            packed_layout = layout_w
        else:
            if force_packed:
                prefer_packed = True
            elif force_bitsliced:
                prefer_packed = False
            else:
                prefer_packed = backend == "x86-avx512"

            t0 = time.perf_counter()
            circuit, layout, choice = coordinate_lowering(
                reduced,
                prefer_packed=prefer_packed,
                force_choice=force_packed or force_bitsliced,
            )
            _log_timing("coordinate_lowering", t0)
            write_lowering_choice_report(choice, out_dir, write_bin=write_bin)

            if isinstance(circuit, PackedCircuitState):
                packed_layout = layout

        if packed_layout is not None:
            layout_w = packed_layout
            if write_bin:
                from stc.packed_circuit_bin import write_packed_circuit_bin
                from stc.layout_bin import write_packed_word_layout_bin

                write_packed_circuit_bin(circuit, out_dir / "packed_circuit_state.bin")
                write_packed_word_layout_bin(
                    layout_w, out_dir / "packed_word_layout.bin"
                )
            input_io_words, output_io_words = layout_w.io_words()

            if use_regions:
                from stc.packed_region_emit import (
                    EmitRegionsConfig,
                    emit_avx512_u64_regions,
                )
                from stc.packed_regions import RegionCaps

                if autotune:
                    results, choice = autotune_configuration(
                        circuit,
                        seed=autotune_seed,
                        budget_ms=autotune_budget_ms,
                        num_candidates=autotune_candidates,
                    )
                    write_autotune_results(results, choice, out_dir)
                    region_max_gates = choice.config.region_max_gates
                    region_max_boundary = choice.config.region_max_boundary

                t0 = time.perf_counter()
                code = emit_avx512_u64_regions(
                    circuit,
                    layout_w,
                    function_name="circuit",
                    config=EmitRegionsConfig(
                        caps=RegionCaps(
                            max_gates=region_max_gates,
                            max_boundary=region_max_boundary,
                        ),
                        diagnostics_dir=out_dir,
                    ),
                )
                (out_dir / "circuit_avx512_u64_regions.c").write_text(
                    code, encoding="utf-8"
                )
                _log_timing("emit_avx512_u64_regions", t0)
            else:
                t0 = time.perf_counter()
                code = generate_scheduled_code(
                    circuit,
                    target="avx512_u64",
                    scheduler="list",
                    function_name="circuit",
                    io_split=(input_io_words, output_io_words),
                    max_live_pressure=max_live_pressure,
                )
                (out_dir / "circuit_avx512_u64.c").write_text(code, encoding="utf-8")
                stats = get_schedule_stats(circuit, "avx512_u64", "list")
                if write_bin:
                    from stc.schedule_stats_bin import write_schedule_stats_bin

                    write_schedule_stats_bin(stats, out_dir / "schedule_stats.bin")
                _log_timing("emit_avx512_u64", t0)
            return
        else:
            t0 = time.perf_counter()
            if write_bin:
                from stc.circuit_state_bin import write_circuit_state_bin
                from stc.layout_bin import write_packed_layout_bin

                write_circuit_state_bin(circuit, out_dir / "circuit_state.bin")
                write_packed_layout_bin(layout, out_dir / "io_layout.bin")
            _log_timing("write_circuit_state", t0)

            if target in ("avx512", "avx2", "ptx"):
                if os.environ.get("STC_SKIP_LINEAR_OPT", "0").lower() in {
                    "0",
                    "false",
                    "no",
                }:
                    if os.environ.get("STC_RUST_LINEAR_OPT", "0").lower() in {
                        "0",
                        "false",
                        "no",
                    }:
                        rust_linear = False
                    else:
                        rust_linear = True
                    max_gates = int(os.environ.get("STC_LINEAR_OPT_MAX_GATES", "50000"))
                    force_linear = os.environ.get(
                        "STC_FORCE_LINEAR_OPT", "0"
                    ).lower() not in {"0", "false", "no"}
                    if not rust_linear and (
                        force_linear or circuit.gate_count <= max_gates
                    ):
                        t0 = time.perf_counter()
                        circuit = circuit.optimize_linear_layers()
                        if write_bin:
                            from stc.circuit_state_bin import write_circuit_state_bin

                            write_circuit_state_bin(
                                circuit, out_dir / "circuit_state_linear.bin"
                            )
                        _log_timing("optimize_linear_layers", t0)

                if os.environ.get("STC_SKIP_GATE_TERNARY", "0").lower() in {
                    "0",
                    "false",
                    "no",
                } and os.environ.get("STC_RUST_GATE_TERNARY", "0").lower() in {
                    "0",
                    "false",
                    "no",
                }:
                    max_gates = int(
                        os.environ.get("STC_GATE_TERNARY_MAX_GATES", "50000")
                    )
                    force_gate_ternary = os.environ.get(
                        "STC_FORCE_GATE_TERNARY", "0"
                    ).lower() not in {"0", "false", "no"}
                    if force_gate_ternary or circuit.gate_count <= max_gates:
                        t0 = time.perf_counter()
                        from stc.gate_ternary_synth import (
                            apply_andnot_optimization,
                            apply_ternary_synthesis,
                            eliminate_double_nots,
                        )

                        gates_initial = len(circuit.gates)

                        circuit, dnot_stats = eliminate_double_nots(circuit)

                        circuit, andnot_stats = apply_andnot_optimization(circuit)

                        circuit, ternary_stats = apply_ternary_synthesis(circuit)

                        if write_bin:
                            from stc.circuit_state_bin import write_circuit_state_bin
                            from stc.kv_bin import write_kv_bin

                            patterns_found = ternary_stats.patterns_found
                            if isinstance(patterns_found, (dict, list, tuple, set)):
                                patterns_found_metric = float(len(patterns_found))
                            else:
                                try:
                                    patterns_found_metric = float(patterns_found)
                                except (TypeError, ValueError):
                                    patterns_found_metric = 0.0

                            write_circuit_state_bin(
                                circuit, out_dir / "circuit_state_ternary.bin"
                            )
                            write_kv_bin(
                                {
                                    "double_nots_eliminated": dnot_stats.get(
                                        "double_nots_eliminated", 0
                                    ),
                                    "andnot_created": andnot_stats.get(
                                        "andnot_created", 0
                                    ),
                                    "ternary_gates_created": ternary_stats.ternary_gates_created,
                                    "patterns_found": patterns_found_metric,
                                    "total_gates_before": gates_initial,
                                    "total_gates_after": ternary_stats.gates_after,
                                    "total_reduction_percent": (
                                        round(
                                            100
                                            * (
                                                1
                                                - ternary_stats.gates_after
                                                / gates_initial
                                            ),
                                            1,
                                        )
                                        if gates_initial > 0
                                        else 0
                                    ),
                                },
                                out_dir / "gate_opt_stats.bin",
                            )
                        _log_timing("gate_ternary_synth", t0)

            input_io_bits = sum(int(v["width"]) for v in layout.inputs.values())
            output_io_bits = sum(int(v["width"]) for v in layout.outputs.values())

            t0 = time.perf_counter()
            sched = scheduler or "list"
            emit_io_split = (
                None
                if target in {"ptx", "ptx_mir"}
                else (input_io_bits, output_io_bits)
            )
            code = generate_scheduled_code(
                circuit,
                target=target,
                scheduler=sched,
                function_name="circuit",
                io_split=emit_io_split,
                max_live_pressure=max_live_pressure,
            )
            ext = "ptx" if target.startswith("ptx") else "c"
            (out_dir / f"circuit_{target}.{ext}").write_text(code, encoding="utf-8")
            skip_stats_env = os.environ.get(
                "STC_SKIP_SCHEDULE_STATS", "0"
            ).lower() not in {
                "0",
                "false",
                "no",
            }
            max_stats_gates = int(
                os.environ.get("STC_SCHEDULE_STATS_MAX_GATES", "200000")
            )
            skip_stats = skip_stats_env or len(circuit.gates) > max_stats_gates
            if not skip_stats:
                stats = get_schedule_stats(circuit, target, sched)
                if write_bin:
                    from stc.schedule_stats_bin import write_schedule_stats_bin

                    write_schedule_stats_bin(stats, out_dir / "schedule_stats.bin")
            _log_timing("emit_bitsliced_schedule", t0)
            return

    if not _has_simd_types(reduced):
        if io_map is None:
            iom = default_io_map(reduced)
        else:
            iom = load_io_map(io_map)
            validate_io_map(reduced, iom)
        if write_bin:
            from stc.io_map_bin import write_io_map_bin

            write_io_map_bin(iom, out_dir / "io_map.bin")

        if not no_backend:
            t0 = time.perf_counter()
            (out_dir / "avr.c").write_text(
                emit_avr_c(reduced, io_map=iom), encoding="utf-8"
            )
            _log_timing("emit_avr_c", t0)
            if avr_project:
                t0 = time.perf_counter()
                for name, content in emit_avr_project(reduced, io_map=iom).items():
                    (out_dir / name).write_text(content, encoding="utf-8")
                _log_timing("emit_avr_project", t0)
    elif not no_backend:
        raise ValueError("avr backend does not support simd types; use --no-backend")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="stc")
    p.add_argument("input", type=Path)
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument("--top", type=str, default=None)
    p.add_argument("--bound", type=int, default=8)
    p.add_argument("--infer-simd", action="store_true", default=False)
    p.add_argument("--autovec", action="store_true", default=False)
    p.add_argument("--autovec-timeout-ms", type=int, default=200)
    p.add_argument("--superopt", action="store_true", default=False)
    p.add_argument("--superopt-max-nodes", type=int, default=6)
    p.add_argument("--superopt-timeout-ms", type=int, default=200)
    p.add_argument("--no-backend", action="store_true", default=False)
    p.add_argument(
        "--use-synth",
        action="store_true",
        default=False,
        help="Use Yosys synth pass with ABC optimization (generates techmap cells)",
    )
    p.add_argument(
        "--abc-lut3",
        action="store_true",
        default=False,
        help="Use Yosys+ABC LUT3 mapping (targets lop3/vpternlog)",
    )
    p.add_argument(
        "--abc-lut3-depth",
        type=int,
        default=None,
        help="Depth target for ABC LUT3 mapping (depth-first then size recovery)",
    )
    p.add_argument(
        "--abc-lut3-aggressive",
        action="store_true",
        default=False,
        help="Use aggressive ABC LUT3 script (extra FRAIG/DCH/MFS2 passes)",
    )
    p.add_argument(
        "--abc-lut3-aggressive-auto-threshold",
        type=int,
        default=None,
        help="Auto-enable aggressive ABC LUT3 when gate count exceeds threshold",
    )
    p.add_argument(
        "--abc-lut3-aggressive-auto-max-live-pressure",
        type=int,
        default=None,
        help="Set max-live-pressure when aggressive ABC auto-enables",
    )
    p.add_argument("--io-map", type=Path, default=None)
    p.add_argument("--avr-project", action="store_true", default=False)
    p.add_argument(
        "--backend",
        "-b",
        choices=["generic", "avr", "ptx", "x86-avx2", "x86-avx512", "futhark"],
        default="generic",
        help="Target backend for optimization",
    )
    p.add_argument(
        "--futhark-mode",
        choices=["auto", "combinational_fast", "step_legacy"],
        default="auto",
        help="Futhark emission mode (auto picks combinational_fast unless sequential feedback is detected).",
    )
    p.add_argument(
        "--force-bitsliced",
        action="store_true",
        default=False,
        help="Force bit-level lowering even for AVX-512 (workaround for packed lowering gate explosion)",
    )
    p.add_argument(
        "--packed-bitslice",
        action="store_true",
        default=False,
        help="Lower to bit-level CircuitState, then emit packed u64 with one word per bit (bit-sliced instances)",
    )
    p.add_argument(
        "--force-packed",
        action="store_true",
        default=False,
        help="Force packed word-level lowering for AVX-512 (for testing packed lowering)",
    )
    p.add_argument(
        "--ternary-mapping",
        action="store_true",
        dest="ternary_mapping",
        default=None,
        help="Enable ternary LUT mapping (auto-enabled for ptx, x86-avx512)",
    )
    p.add_argument(
        "--no-ternary-mapping",
        action="store_false",
        dest="ternary_mapping",
        help="Disable ternary LUT mapping",
    )
    p.add_argument(
        "--depth-budget", type=int, default=None, help="Maximum allowed circuit depth"
    )
    p.add_argument(
        "--anytime",
        action="store_true",
        default=False,
        help="Enable anytime optimization with checkpointing",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Optimization timeout in seconds (default: 300)",
    )
    p.add_argument(
        "--no-improvement-timeout",
        type=float,
        default=60.0,
        help="Stop if no improvement for this many seconds (default: 60)",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints (default: output_dir/checkpoints)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic optimization",
    )
    p.add_argument(
        "--scheduler",
        choices=["list", "pipelined", "serial", "auto"],
        default=None,
        help="Scheduling algorithm for code generation",
    )
    p.add_argument(
        "--emit-target",
        choices=["avx2", "avx512", "avx512_mir", "ptx", "ptx_mir", "ptx_legacy"],
        default=None,
        help="Target for scheduled code emission (MIR variants use Machine IR layer)",
    )
    p.add_argument(
        "--max-live-pressure",
        type=int,
        default=None,
        help="Maximum live values during scheduling (default: unlimited). "
        "Set to ~28 for AVX-512 to avoid spills.",
    )
    p.add_argument(
        "--fuse-ticks",
        type=int,
        default=1,
        help="Fuse/unroll N sequential ticks in TickIR before lowering (default: 1, disabled).",
    )
    p.add_argument(
        "--fuse-mode",
        choices=["final", "all", "state-only"],
        default="final",
        help="Fusion output mode (default: final).",
    )
    p.add_argument(
        "--fuse-input-policy",
        choices=["shared", "replicate"],
        default="shared",
        help="Input policy for fusion (default: shared).",
    )
    p.add_argument(
        "--fuse-budget-max-nodes",
        type=int,
        default=1 << 60,
        help="Fusion budget: maximum expression nodes (default: effectively unlimited).",
    )
    p.add_argument(
        "--fuse-budget-max-depth",
        type=int,
        default=1 << 60,
        help="Fusion budget: maximum expression depth (default: effectively unlimited).",
    )
    p.add_argument(
        "--fuse-budget-max-step-ms",
        type=int,
        default=None,
        help="Fusion budget: maximum wall time per fused step in ms (default: no limit).",
    )
    p.add_argument(
        "--use-regions",
        action="store_true",
        default=False,
        help="Enable region decomposition and region-fused emission (currently for packed AVX-512 u64 path).",
    )
    p.add_argument(
        "--region-max-gates",
        type=int,
        default=50000,
        help="Region cap: maximum gates per region (packed path).",
    )
    p.add_argument(
        "--region-max-boundary",
        type=int,
        default=8192,
        help="Region cap: maximum boundary nodes per region (packed path).",
    )
    p.add_argument(
        "--autotune",
        action="store_true",
        default=False,
        help="Enable autotuning mode to find optimal compilation configuration.",
    )
    p.add_argument(
        "--autotune-seed",
        type=int,
        default=42,
        help="Random seed for autotune reproducibility (default: 42).",
    )
    p.add_argument(
        "--autotune-budget-ms",
        type=int,
        default=60000,
        help="Time budget for autotuning in milliseconds (default: 60000).",
    )
    p.add_argument(
        "--autotune-candidates",
        type=int,
        default=8,
        help="Number of candidate configurations to try (default: 8).",
    )
    p.add_argument(
        "--no-arith-classify",
        action="store_false",
        dest="arith_classify",
        default=True,
        help="Disable arithmetic classification pass (enabled by default).",
    )
    p.add_argument(
        "--mul-div-max-width",
        type=int,
        default=32,
        help="Maximum bit width for mul/div lowering (default: 32).",
    )
    p.add_argument(
        "--dump-arith-report",
        type=Path,
        default=None,
        help="Output path for arithmetic classification report binary.",
    )
    p.add_argument(
        "--bin-artifacts",
        action="store_true",
        default=False,
        help="Write binary artifacts for large outputs.",
    )
    return p.parse_args(argv)


def _load_circuit_state(path: Path) -> CircuitState | None:
    if path.suffix != ".bin":
        return None
    from stc.circuit_state_bin import read_circuit_state_bin

    return read_circuit_state_bin(path)


def _select_scheduler(scheduler: str, target: str) -> str:
    if scheduler == "auto":
        return "pipelined" if target == "ptx" else "list"
    return scheduler


def run_scheduled_backend(
    input_path: Path,
    out_dir: Path,
    *,
    scheduler: str,
    emit_target: str,
    function_name: str = "circuit",
    max_live_pressure: int | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_bin = os.environ.get("STC_BIN_ARTIFACTS", "0") not in {
        "0",
        "false",
        "no",
    }

    circuit = _load_circuit_state(input_path)
    if circuit is None:
        raise ValueError(f"Input must be a CircuitState BIN file: {input_path}")

    scheduler = _select_scheduler(scheduler, emit_target)

    code = generate_scheduled_code(
        circuit,
        target=emit_target,
        scheduler=scheduler,
        function_name=function_name,
        max_live_pressure=max_live_pressure,
    )

    stats = get_schedule_stats(circuit, emit_target, scheduler)

    ext = "ptx" if emit_target in ("ptx", "ptx_mir", "ptx_legacy") else "c"
    (out_dir / f"{function_name}.{ext}").write_text(code, encoding="utf-8")
    if write_bin:
        from stc.schedule_stats_bin import write_schedule_stats_bin

        write_schedule_stats_bin(stats, out_dir / "schedule_stats.bin")


def main(argv: list[str] | None = None) -> int:
    ns = parse_args([] if argv is None else argv)

    if ns.bin_artifacts:
        os.environ["STC_BIN_ARTIFACTS"] = "1"
        os.environ.setdefault("STC_RUST_LOWER_OUTPUT", "bin")
        os.environ.setdefault("STC_RUST_SCHED_FORMAT", "bin")

    if ns.scheduler is not None and ns.emit_target is not None:
        run_scheduled_backend(
            ns.input,
            ns.out,
            scheduler=ns.scheduler,
            emit_target=ns.emit_target,
            max_live_pressure=ns.max_live_pressure,
        )
    else:
        run_pipeline(
            ns.input,
            ns.out,
            top=ns.top,
            bound=ns.bound,
            infer_simd=ns.infer_simd,
            autovec=ns.autovec,
            autovec_timeout_ms=ns.autovec_timeout_ms,
            superopt=ns.superopt,
            superopt_max_nodes=ns.superopt_max_nodes,
            superopt_timeout_ms=ns.superopt_timeout_ms,
            no_backend=ns.no_backend,
            use_synth=ns.use_synth,
            use_abc_lut3=ns.abc_lut3,
            abc_lut3_depth=ns.abc_lut3_depth,
            use_abc_lut3_aggressive=ns.abc_lut3_aggressive,
            abc_lut3_aggressive_threshold=ns.abc_lut3_aggressive_auto_threshold,
            abc_lut3_aggressive_max_live_pressure=ns.abc_lut3_aggressive_auto_max_live_pressure,
            io_map=ns.io_map,
            avr_project=ns.avr_project,
            backend=ns.backend,
            futhark_mode=ns.futhark_mode,
            ternary_mapping=ns.ternary_mapping,
            fuse_ticks=ns.fuse_ticks,
            fuse_mode=ns.fuse_mode,
            fuse_input_policy=ns.fuse_input_policy,
            fuse_budget_max_nodes=ns.fuse_budget_max_nodes,
            fuse_budget_max_depth=ns.fuse_budget_max_depth,
            fuse_budget_max_step_ms=ns.fuse_budget_max_step_ms,
            use_regions=ns.use_regions,
            region_max_gates=ns.region_max_gates,
            region_max_boundary=ns.region_max_boundary,
            autotune=ns.autotune,
            autotune_seed=ns.autotune_seed,
            autotune_budget_ms=ns.autotune_budget_ms,
            autotune_candidates=ns.autotune_candidates,
            force_bitsliced=ns.force_bitsliced,
            force_packed=ns.force_packed,
            packed_bitslice=ns.packed_bitslice,
            arith_classify=ns.arith_classify,
            mul_div_max_width=ns.mul_div_max_width,
            dump_arith_report=ns.dump_arith_report,
            max_live_pressure=ns.max_live_pressure,
            scheduler=ns.scheduler,
        )
    return 0
