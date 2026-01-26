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
    region_max_boundary: int = 8192,
    autotune: bool = False,
    autotune_seed: int = 42,
    autotune_budget_ms: int = 60000,
    autotune_candidates: int = 8,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.suffix == ".v":
        normalized_json = out_dir / "normalized.json"
        run_yosys(
            input_path,
            normalized_json,
            top=top,
            output_script=out_dir / "normalized.ys",
        )
    else:
        normalized_json = input_path

    design = load_design(normalized_json, top=top)
    tick_ir = extract_tick_ir(design)
    validate_tick_ir(tick_ir)
    if infer_simd or autovec:
        tick_ir = infer_simd_types(tick_ir)
        validate_tick_ir(tick_ir)
    reduced = optimize_tick_ir(
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
    )
    validate_tick_ir(reduced)

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
        if backend == "x86-avx512":
            circuit, layout, choice = coordinate_lowering(reduced)
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
                    )
                    (out_dir / "circuit_avx512_u64.c").write_text(
                        code, encoding="utf-8"
                    )
                    _write_json(
                        out_dir / "schedule_stats.json",
                        get_schedule_stats(circuit, "avx512_u64", "list"),
                    )
                return
            else:
                _write_json(out_dir / "circuit_state.json", circuit.to_dict())
                _write_json(out_dir / "io_layout.json", layout.to_dict())

                input_io_bits = sum(int(v["width"]) for v in layout.inputs.values())
                output_io_bits = sum(int(v["width"]) for v in layout.outputs.values())

                code = generate_scheduled_code(
                    circuit,
                    target=target,
                    scheduler="list",
                    function_name="circuit",
                    io_split=(input_io_bits, output_io_bits),
                )
                (out_dir / f"circuit_{target}.c").write_text(code, encoding="utf-8")
                _write_json(
                    out_dir / "schedule_stats.json",
                    get_schedule_stats(circuit, target, "list"),
                )
                return

        circuit, layout = lower_tick_ir_to_circuit_state(reduced)
        _write_json(out_dir / "circuit_state.json", circuit.to_dict())
        _write_json(out_dir / "io_layout.json", layout.to_dict())

        input_io_bits = sum(int(v["width"]) for v in layout.inputs.values())
        output_io_bits = sum(int(v["width"]) for v in layout.outputs.values())

        code = generate_scheduled_code(
            circuit,
            target=target,
            scheduler="list",
            function_name="circuit",
            io_split=(input_io_bits, output_io_bits),
        )
        (out_dir / f"circuit_{target}.c").write_text(code, encoding="utf-8")
        _write_json(
            out_dir / "schedule_stats.json", get_schedule_stats(circuit, target, "list")
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
        choices=["avx2", "avx512", "ptx"],
        default=None,
        help="Target for scheduled code emission",
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
    )

    stats = get_schedule_stats(circuit, emit_target, scheduler)

    ext = "ptx" if emit_target == "ptx" else "c"
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
        )
    return 0
