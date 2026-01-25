from __future__ import annotations

import argparse
import json
from pathlib import Path

from stc.backend_avr import emit_avr_c
from stc.avr_project import emit_avr_project
from stc.backend_sched import generate_scheduled_code, get_schedule_stats
from stc.circuit_synth import CircuitState
from stc.extract import extract_tick_ir
from stc.infer_simd import infer_simd_types
from stc.io_map import default_io_map, load_io_map, validate_io_map
from stc.metrics import compute_metrics
from stc.reduce import optimize_tick_ir
from stc.tick_ir_to_circuit_state import lower_tick_ir_to_circuit_state
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
        circuit, layout = lower_tick_ir_to_circuit_state(reduced)
        _write_json(out_dir / "circuit_state.json", circuit.to_dict())
        _write_json(out_dir / "io_layout.json", layout.to_dict())

        code = generate_scheduled_code(
            circuit,
            target=target,
            scheduler="list",
            function_name="circuit",
        )
        (out_dir / f"circuit_{target}.c").write_text(code, encoding="utf-8")
        _write_json(out_dir / "schedule_stats.json", get_schedule_stats(circuit, target, "list"))
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
        )
    return 0
