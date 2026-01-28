from __future__ import annotations

import argparse
import json
from pathlib import Path

from stc.autotune import autotune_configuration, write_autotune_results
from stc.backend_avr import emit_avr_c
from stc.avr_project import emit_avr_project
from stc.backend_sched import generate_scheduled_code, get_schedule_stats
from stc.circuit_synth import CircuitState
from stc.extract import extract_tick_ir
from stc.infer_simd import infer_simd_types
from stc.io_map import default_io_map, load_io_map, validate_io_map
from stc.lowering_coordinator import (
    coordinate_lowering,
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


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    io_map: Path | None = None,
    avr_project: bool = False,
    backend: str = "generic",
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
    region_max_boundary: int = 8192,
    autotune: bool = False,
    autotune_seed: int = 42,
    autotune_budget_ms: int = 60000,
    autotune_candidates: int = 8,
    arith_classify: bool = True,
    mul_div_max_width: int = 32,
    dump_arith_report: Path | None = None,
    max_live_pressure: int | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.suffix == ".v":
        normalized_json = out_dir / "normalized.json"
        run_yosys(
            input_path,
            normalized_json,
            top=top,
            output_script=out_dir / "normalized.ys",
            use_synth=use_synth,
        )
    else:
        normalized_json = input_path

    design = load_design(normalized_json, top=top)
    tick_ir = extract_tick_ir(design)
    validate_tick_ir(tick_ir)
    if infer_simd or autovec:
        tick_ir = infer_simd_types(tick_ir)
        validate_tick_ir(tick_ir)
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
        bounded_state_opt=backend not in {"x86-avx2", "x86-avx512"},
        fuse_ticks=fuse_ticks,
        fuse_mode=fuse_mode,
        fuse_input_policy=fuse_input_policy,
        fuse_budget_max_nodes=fuse_budget_max_nodes,
        fuse_budget_max_depth=fuse_budget_max_depth,
        fuse_budget_max_step_ms=fuse_budget_max_step_ms,
        arith_classify=arith_classify,
        mul_div_max_width=mul_div_max_width,
    )
    validate_tick_ir(reduced)

    if dump_arith_report is not None and arith_report is not None:
        _write_json(dump_arith_report, arith_report)

    _write_json(out_dir / "tick_ir.json", tick_ir.to_dict())
    _write_json(out_dir / "reduced_tick_ir.json", reduced.to_dict())
    _write_json(out_dir / "metrics.json", compute_metrics(tick_ir).to_dict())
    _write_json(out_dir / "reduced_metrics.json", compute_metrics(reduced).to_dict())
    # `io_map.json` and AVR artifacts only apply to the AVR backend. When
    # `--no-backend` is set, users may be targeting non-AVR outputs and the
    # reduced IR may have many outputs that cannot fit the default PORTB map.
    if no_backend:
        return

    # Non-AVR scheduled backends: lower TickIR to CircuitState and emit scheduled code.
    # This enables full Verilog-derived designs (including wide state) to target AVX-512.
    if backend in {"x86-avx512", "x86-avx2"}:
        target = "avx512" if backend == "x86-avx512" else "avx2"
        if force_packed and force_bitsliced:
            raise ValueError(
                "Cannot use both --force-packed and --force-bitsliced simultaneously"
            )
        if force_packed:
            prefer_packed = True
        elif force_bitsliced:
            prefer_packed = False
        else:
            prefer_packed = backend == "x86-avx512"

        circuit, layout, choice = coordinate_lowering(
            reduced,
            prefer_packed=prefer_packed,
            force_choice=force_packed or force_bitsliced,
        )
        write_lowering_choice_report(choice, out_dir)

        if isinstance(circuit, PackedCircuitState):
            layout_w = layout
            _write_json(out_dir / "packed_circuit_state.json", circuit.to_dict())
            _write_json(out_dir / "packed_word_layout.json", layout_w.to_dict())

            input_io_words = sum(
                (int(v["width_bits"]) + 63) // 64 for v in layout_w.inputs.values()
            )
            output_io_words = sum(
                (int(v["width_bits"]) + 63) // 64 for v in layout_w.outputs.values()
            )

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
            else:
                code = generate_scheduled_code(
                    circuit,
                    target="avx512_u64",
                    scheduler="list",
                    function_name="circuit",
                    io_split=(input_io_words, output_io_words),
                    max_live_pressure=max_live_pressure,
                )
                (out_dir / "circuit_avx512_u64.c").write_text(code, encoding="utf-8")
                _write_json(
                    out_dir / "schedule_stats.json",
                    get_schedule_stats(circuit, "avx512_u64", "list"),
                )
            return
        else:
            _write_json(out_dir / "circuit_state.json", circuit.to_dict())
            _write_json(out_dir / "io_layout.json", layout.to_dict())

            if target in ("avx512", "avx2"):
                from stc.gate_ternary_synth import (
                    apply_andnot_optimization,
                    apply_ternary_synthesis,
                    eliminate_double_nots,
                )

                gates_initial = len(circuit.gates)

                circuit, dnot_stats = eliminate_double_nots(circuit)

                circuit, andnot_stats = apply_andnot_optimization(circuit)

                circuit, ternary_stats = apply_ternary_synthesis(circuit)

                _write_json(out_dir / "circuit_state_ternary.json", circuit.to_dict())
                _write_json(
                    out_dir / "gate_opt_stats.json",
                    {
                        "double_nots_eliminated": dnot_stats.get(
                            "double_nots_eliminated", 0
                        ),
                        "andnot_created": andnot_stats.get("andnot_created", 0),
                        "ternary_gates_created": ternary_stats.ternary_gates_created,
                        "patterns_found": ternary_stats.patterns_found,
                        "total_gates_before": gates_initial,
                        "total_gates_after": ternary_stats.gates_after,
                        "total_reduction_percent": (
                            round(
                                100 * (1 - ternary_stats.gates_after / gates_initial),
                                1,
                            )
                            if gates_initial > 0
                            else 0
                        ),
                    },
                )

            input_io_bits = sum(int(v["width"]) for v in layout.inputs.values())
            output_io_bits = sum(int(v["width"]) for v in layout.outputs.values())

            code = generate_scheduled_code(
                circuit,
                target=target,
                scheduler="list",
                function_name="circuit",
                io_split=(input_io_bits, output_io_bits),
                max_live_pressure=max_live_pressure,
            )
            (out_dir / f"circuit_{target}.c").write_text(code, encoding="utf-8")
            _write_json(
                out_dir / "schedule_stats.json",
                get_schedule_stats(circuit, target, "list"),
            )
            return

    if not _has_simd_types(reduced):
        if io_map is None:
            iom = default_io_map(reduced)
        else:
            iom = load_io_map(io_map)
            validate_io_map(reduced, iom)
        _write_json(out_dir / "io_map.json", iom.to_dict())

        if not no_backend:
            (out_dir / "avr.c").write_text(
                emit_avr_c(reduced, io_map=iom), encoding="utf-8"
            )
            if avr_project:
                for name, content in emit_avr_project(reduced, io_map=iom).items():
                    (out_dir / name).write_text(content, encoding="utf-8")
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
    p.add_argument("--io-map", type=Path, default=None)
    p.add_argument("--avr-project", action="store_true", default=False)
    p.add_argument(
        "--backend",
        "-b",
        choices=["generic", "avr", "ptx", "x86-avx2", "x86-avx512"],
        default="generic",
        help="Target backend for optimization",
    )
    p.add_argument(
        "--force-bitsliced",
        action="store_true",
        default=False,
        help="Force bit-level lowering even for AVX-512 (workaround for packed lowering gate explosion)",
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
        choices=["list", "pipelined", "auto"],
        default=None,
        help="Scheduling algorithm for code generation",
    )
    p.add_argument(
        "--emit-target",
        choices=["avx2", "avx512", "avx512_mir", "ptx", "ptx_mir"],
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
        help="Output path for arithmetic classification report JSON.",
    )
    return p.parse_args(argv)


def _load_circuit_state(path: Path) -> CircuitState | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if all(k in data for k in ("input_bits", "output_bits", "gates", "outputs")):
            return CircuitState.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return None


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

    circuit = _load_circuit_state(input_path)
    if circuit is None:
        raise ValueError(f"Input must be a CircuitState JSON file: {input_path}")

    scheduler = _select_scheduler(scheduler, emit_target)

    code = generate_scheduled_code(
        circuit,
        target=emit_target,
        scheduler=scheduler,
        function_name=function_name,
        max_live_pressure=max_live_pressure,
    )

    stats = get_schedule_stats(circuit, emit_target, scheduler)

    ext = "ptx" if emit_target in ("ptx", "ptx_mir") else "c"
    (out_dir / f"{function_name}.{ext}").write_text(code, encoding="utf-8")
    _write_json(out_dir / "schedule_stats.json", stats)


def main(argv: list[str] | None = None) -> int:
    ns = parse_args([] if argv is None else argv)

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
            io_map=ns.io_map,
            avr_project=ns.avr_project,
            backend=ns.backend,
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
            arith_classify=ns.arith_classify,
            mul_div_max_width=ns.mul_div_max_width,
            dump_arith_report=ns.dump_arith_report,
            max_live_pressure=ns.max_live_pressure,
        )
    return 0
