#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from stc.cli import run_pipeline
from stc.interp import TickState, tick
from stc.tick_ir import BitVecType, BoolType, TickIR, Type
from stc.tick_ir_bin2 import read_tick_ir_bin


def _mask(width: int) -> int:
    if width < 1:
        raise ValueError(f"invalid bit width: {width}")
    return (1 << width) - 1


def _chunks_for_width(width: int) -> int:
    if width < 1:
        raise ValueError(f"invalid bit width: {width}")
    return (width + 63) // 64


def _split_top_level(text: str) -> list[str]:
    items: list[str] = []
    depth_round = 0
    depth_square = 0
    start = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            depth_round += 1
        elif ch == ")":
            depth_round -= 1
        elif ch == "[":
            depth_square += 1
        elif ch == "]":
            depth_square -= 1
        elif ch == "," and depth_round == 0 and depth_square == 0:
            items.append(text[start:idx].strip())
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        items.append(tail)
    return items


def _split_top_level_values(text: str) -> list[str]:
    items: list[str] = []
    depth_round = 0
    depth_square = 0
    start = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            depth_round += 1
        elif ch == ")":
            depth_round -= 1
        elif ch == "[":
            depth_square += 1
        elif ch == "]":
            depth_square -= 1
        elif ch == "\n" and depth_round == 0 and depth_square == 0:
            part = text[start:idx].strip()
            if part:
                items.append(part)
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        items.append(tail)
    return items


def _parse_futhark_value(text: str) -> Any:
    src = text.strip()
    if src.startswith("(") and src.endswith(")"):
        inner = src[1:-1].strip()
        if not inner:
            return tuple()
        return tuple(_parse_futhark_value(part) for part in _split_top_level(inner))
    if src.startswith("[") and src.endswith("]"):
        inner = src[1:-1].strip()
        if not inner:
            return []
        return [_parse_futhark_value(part) for part in _split_top_level(inner)]
    if src == "true":
        return True
    if src == "false":
        return False
    for suffix in ("u64", "i64", "i32", "u32"):
        if src.endswith(suffix):
            return int(src[: -len(suffix)], 0)
    if src.startswith("0x") or src.startswith("-0x"):
        return int(src, 16)
    return int(src, 10)


def _format_scalar(value: int | bool, t: Type) -> str:
    if isinstance(t, BoolType):
        return "true" if bool(value) else "false"
    assert isinstance(t, BitVecType)
    masked = int(value) & _mask(t.width)
    if t.width <= 64:
        mask64 = _mask(t.width) & 0xFFFFFFFFFFFFFFFF
        return f"{masked & mask64}u64"
    chunks = _chunks_for_width(t.width)
    words = []
    for idx in range(chunks):
        word = (masked >> (64 * idx)) & 0xFFFFFFFFFFFFFFFF
        words.append(f"{word}u64")
    return "[" + ", ".join(words) + "]"


def _format_array(values: list[int | bool], t: Type) -> str:
    return "[" + ", ".join(_format_scalar(v, t) for v in values) + "]"


def _format_array2(values: list[list[int | bool]], t: Type) -> str:
    return "[" + ", ".join(_format_array(row, t) for row in values) + "]"


def _normalize_lane_value(value: int | bool, t: Type) -> int | bool:
    if isinstance(t, BoolType):
        return bool(value)
    assert isinstance(t, BitVecType)
    if t.width <= 64:
        mask64 = _mask(t.width) & 0xFFFFFFFFFFFFFFFF
        return int(value) & mask64
    if isinstance(value, list):
        chunks = _chunks_for_width(t.width)
        if len(value) != chunks:
            raise ValueError(
                f"invalid chunked value length for width {t.width}: "
                f"expected {chunks}, got {len(value)}"
            )
        acc = 0
        for idx, word in enumerate(value):
            acc |= (int(word) & 0xFFFFFFFFFFFFFFFF) << (64 * idx)
        return acc & _mask(t.width)
    return int(value) & _mask(t.width)


def _normalize_array(values: list[Any], t: Type) -> list[int | bool]:
    return [_normalize_lane_value(v, t) for v in values]


def _unpack_result(parsed: Any, count: int) -> list[Any]:
    if count == 0:
        return []
    if count == 1:
        return [parsed]
    if not isinstance(parsed, tuple) or len(parsed) != count:
        raise ValueError(
            f"unexpected futhark output shape: expected tuple[{count}], got {parsed!r}"
        )
    return list(parsed)


def _run_futhark(
    fut_path: Path, entry: str, args_lines: list[str], *, quiet: bool = False
) -> Any:
    proc = subprocess.run(
        ["futhark", "run", "-e", entry, str(fut_path)],
        input="\n".join(args_lines) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"futhark run failed for entry {entry}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    result_text = proc.stdout.strip()
    if not result_text:
        raise RuntimeError(f"futhark run produced no output for entry {entry}")
    top_values = _split_top_level_values(result_text)
    if not quiet:
        if len(top_values) == 1:
            print(f"[futhark] {entry} output: {top_values[0]}")
        else:
            print(f"[futhark] {entry} output: " + "\n".join(top_values))
    if len(top_values) == 1:
        return _parse_futhark_value(top_values[0])
    return tuple(_parse_futhark_value(v) for v in top_values)


def _compile_futhark_target(fut_path: Path, target: str) -> None:
    if target not in {"cuda", "opencl", "c"}:
        raise ValueError(f"unsupported futhark target: {target}")
    proc = subprocess.run(
        ["futhark", target, str(fut_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"futhark {target} compile failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _generate_random_soa(
    names: list[str], types: dict[str, Type], *, n: int, rng: random.Random
) -> dict[str, list[int | bool]]:
    out: dict[str, list[int | bool]] = {}
    for name in names:
        t = types[name]
        if isinstance(t, BoolType):
            out[name] = [bool(rng.getrandbits(1)) for _ in range(n)]
        elif isinstance(t, BitVecType):
            out[name] = [rng.getrandbits(t.width) & _mask(t.width) for _ in range(n)]
        else:
            raise ValueError(f"unsupported type in parity script: {name}:{t}")
    return out


def _interp_step_batch(
    ir: TickIR,
    *,
    n: int,
    state_soa: dict[str, list[int | bool]],
    input_soa: dict[str, list[int | bool]],
) -> tuple[dict[str, list[int | bool]], dict[str, list[int | bool]]]:
    state_names = sorted(ir.state.keys())
    output_names = sorted(ir.outputs.keys())

    next_state_soa: dict[str, list[int | bool]] = {name: [] for name in state_names}
    output_soa: dict[str, list[int | bool]] = {name: [] for name in output_names}
    for lane in range(n):
        cur_state = {
            name: _normalize_lane_value(state_soa[name][lane], ir.state[name])
            for name in state_names
        }
        cur_inputs = {
            name: _normalize_lane_value(input_soa[name][lane], ir.inputs[name])
            for name in sorted(ir.inputs.keys())
        }
        nxt, outs = tick(ir, TickState(state=cur_state), cur_inputs)
        for name in state_names:
            next_state_soa[name].append(
                _normalize_lane_value(nxt.state[name], ir.state[name])
            )
        for name in output_names:
            output_soa[name].append(_normalize_lane_value(outs[name], ir.outputs[name]))
    return next_state_soa, output_soa


def _interp_run_steps_batch(
    ir: TickIR,
    *,
    steps: int,
    n: int,
    state_soa: dict[str, list[int | bool]],
    input_seq_soa: dict[str, list[list[int | bool]]],
) -> tuple[dict[str, list[int | bool]], dict[str, list[int | bool]]]:
    state_names = sorted(ir.state.keys())
    output_names = sorted(ir.outputs.keys())
    cur_state = {
        name: [
            _normalize_lane_value(state_soa[name][lane], ir.state[name])
            for lane in range(n)
        ]
        for name in state_names
    }
    out_last = {name: [False] * n for name in output_names}
    for s in range(steps):
        in_soa = {name: input_seq_soa[name][s] for name in sorted(ir.inputs.keys())}
        cur_state, out_last = _interp_step_batch(
            ir, n=n, state_soa=cur_state, input_soa=in_soa
        )
    return cur_state, out_last


def _assert_equal_soa(
    expected: dict[str, list[int | bool]],
    actual: dict[str, list[int | bool]],
    types: dict[str, Type],
) -> None:
    for name in sorted(expected.keys()):
        exp = _normalize_array(expected[name], types[name])
        got = _normalize_array(actual[name], types[name])
        if exp != got:
            raise AssertionError(f"mismatch for {name}: expected={exp} got={got}")


def _ensure_supported_types(ir: TickIR) -> None:
    for scope_name, mapping in (
        ("input", ir.inputs),
        ("state", ir.state),
        ("output", ir.outputs),
    ):
        for name, t in mapping.items():
            if isinstance(t, BoolType):
                continue
            if isinstance(t, BitVecType) and t.width >= 1:
                continue
            raise ValueError(
                f"unsupported {scope_name} type for parity check: {name}:{t} "
                "(only bool/bitvec)"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="Verilog file/dir or normalized.json")
    ap.add_argument("--out", type=Path, default=None, help="Output dir (default: temp)")
    ap.add_argument("--top", type=str, default=None)
    ap.add_argument("--bound", type=int, default=8)
    ap.add_argument(
        "--futhark-mode",
        choices=["auto", "combinational_fast", "step_legacy"],
        default="auto",
        help="Futhark backend mode to use during pipeline emission",
    )
    ap.add_argument("--batch", type=int, default=16, help="Number of lanes for parity")
    ap.add_argument("--steps", type=int, default=3, help="Steps for run_steps parity")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--compile-targets",
        type=str,
        default="cuda,opencl,c",
        help="Comma-separated futhark compile targets (subset of cuda,opencl,c)",
    )
    ap.add_argument("--skip-compile", action="store_true", default=False)
    ap.add_argument(
        "--no-bounded-state-opt",
        action="store_true",
        default=False,
        help="Disable bounded-state optimization in run_pipeline",
    )
    ap.add_argument("--quiet", action="store_true", default=False)
    ns = ap.parse_args()

    if shutil.which("futhark") is None:
        raise SystemExit("futhark not found on PATH")

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if ns.out is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="stc_futhark_parity_")
        out_dir = Path(temp_dir.name)
    else:
        out_dir = ns.out
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_pipeline(
            ns.input,
            out_dir,
            top=ns.top,
            bound=ns.bound,
            backend="futhark",
            futhark_mode=ns.futhark_mode,
            bounded_state_opt=False if ns.no_bounded_state_opt else None,
        )
        fut_path = out_dir / "circuit_futhark.fut"
        reduced_path = out_dir / "reduced_tick_ir.bin"
        manifest_path = out_dir / "futhark_io_manifest.json"
        if not fut_path.exists():
            raise SystemExit(f"missing generated futhark file: {fut_path}")
        if not reduced_path.exists():
            raise SystemExit(f"missing reduced Tick-IR file: {reduced_path}")
        if not manifest_path.exists():
            raise SystemExit(f"missing futhark manifest file: {manifest_path}")

        ir = read_tick_ir_bin(reduced_path)
        _ensure_supported_types(ir)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("entries", {})
        step_entry = entries.get("eval_batch", entries.get("step_batch"))
        xor_entry = entries.get("eval_batch_xor")
        run_entry = entries.get("run_steps_batch")
        if step_entry is None:
            raise SystemExit("manifest is missing eval/step entry for parity checks")

        subprocess.run(
            ["futhark", "check", str(fut_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if not ns.quiet:
            print(f"[ok] futhark check: {fut_path}")

        if not ns.skip_compile:
            targets = [t.strip() for t in ns.compile_targets.split(",") if t.strip()]
            for target in targets:
                _compile_futhark_target(fut_path, target)
                if not ns.quiet:
                    print(f"[ok] futhark {target} compile")

        rng = random.Random(ns.seed)
        n = int(ns.batch)
        if n < 1:
            raise SystemExit("--batch must be >= 1")

        state_names = sorted(ir.state.keys())
        input_names = sorted(ir.inputs.keys())
        output_names = sorted(ir.outputs.keys())

        state_soa = _generate_random_soa(state_names, ir.state, n=n, rng=rng)
        input_soa = _generate_random_soa(input_names, ir.inputs, n=n, rng=rng)

        step_lines = [f"{n}i64"]
        step_lines.extend(
            _format_array(state_soa[name], ir.state[name]) for name in state_names
        )
        step_lines.extend(
            _format_array(input_soa[name], ir.inputs[name]) for name in input_names
        )
        step_result = _run_futhark(fut_path, step_entry, step_lines, quiet=ns.quiet)

        step_fields = _unpack_result(step_result, len(state_names) + len(output_names))
        actual_next = {name: step_fields[idx] for idx, name in enumerate(state_names)}
        actual_out = {
            name: step_fields[len(state_names) + idx]
            for idx, name in enumerate(output_names)
        }
        expected_next, expected_out = _interp_step_batch(
            ir, n=n, state_soa=state_soa, input_soa=input_soa
        )
        _assert_equal_soa(expected_next, actual_next, ir.state)
        _assert_equal_soa(expected_out, actual_out, ir.outputs)
        print(f"[ok] {step_entry} parity")

        if xor_entry is not None:
            acc_soa = _generate_random_soa(output_names, ir.outputs, n=n, rng=rng)
            xor_lines = list(step_lines)
            xor_lines.extend(
                _format_array(acc_soa[name], ir.outputs[name]) for name in output_names
            )
            xor_result = _run_futhark(fut_path, xor_entry, xor_lines, quiet=ns.quiet)
            xor_fields = _unpack_result(
                xor_result, len(state_names) + len(output_names)
            )
            actual_xor_state = {
                name: xor_fields[idx] for idx, name in enumerate(state_names)
            }
            actual_xor_out = {
                name: xor_fields[len(state_names) + idx]
                for idx, name in enumerate(output_names)
            }

            expected_xor_out: dict[str, list[int | bool]] = {}
            for name in output_names:
                t = ir.outputs[name]
                values: list[int | bool] = []
                for lane in range(n):
                    a = _normalize_lane_value(expected_out[name][lane], t)
                    b = _normalize_lane_value(acc_soa[name][lane], t)
                    if isinstance(t, BoolType):
                        values.append(bool(a) ^ bool(b))
                    else:
                        assert isinstance(t, BitVecType)
                        values.append((int(a) ^ int(b)) & _mask(t.width))
                expected_xor_out[name] = values
            _assert_equal_soa(expected_next, actual_xor_state, ir.state)
            _assert_equal_soa(expected_xor_out, actual_xor_out, ir.outputs)
            print(f"[ok] {xor_entry} parity")

        steps = int(ns.steps)
        if steps < 1:
            raise SystemExit("--steps must be >= 1")
        input_seq_soa = {
            name: [
                _generate_random_soa([name], ir.inputs, n=n, rng=rng)[name]
                for _ in range(steps)
            ]
            for name in input_names
        }

        run_lines = [f"{steps}i64", f"{n}i64"]
        run_lines.extend(
            _format_array(state_soa[name], ir.state[name]) for name in state_names
        )
        run_lines.extend(
            _format_array2(input_seq_soa[name], ir.inputs[name]) for name in input_names
        )
        if run_entry is not None:
            run_result = _run_futhark(fut_path, run_entry, run_lines, quiet=ns.quiet)
            run_fields = _unpack_result(
                run_result, len(state_names) + len(output_names)
            )
            actual_final_state = {
                name: run_fields[idx] for idx, name in enumerate(state_names)
            }
            actual_final_out = {
                name: run_fields[len(state_names) + idx]
                for idx, name in enumerate(output_names)
            }
        else:
            cur_state = {name: list(state_soa[name]) for name in state_names}
            actual_final_out = {name: [False] * n for name in output_names}
            for s in range(steps):
                step_lines_loop = [f"{n}i64"]
                step_lines_loop.extend(
                    _format_array(cur_state[name], ir.state[name])
                    for name in state_names
                )
                step_lines_loop.extend(
                    _format_array(input_seq_soa[name][s], ir.inputs[name])
                    for name in input_names
                )
                step_loop_result = _run_futhark(
                    fut_path, step_entry, step_lines_loop, quiet=True
                )
                step_loop_fields = _unpack_result(
                    step_loop_result, len(state_names) + len(output_names)
                )
                cur_state = {
                    name: step_loop_fields[idx] for idx, name in enumerate(state_names)
                }
                actual_final_out = {
                    name: step_loop_fields[len(state_names) + idx]
                    for idx, name in enumerate(output_names)
                }
            actual_final_state = cur_state
        expected_final_state, expected_final_out = _interp_run_steps_batch(
            ir, steps=steps, n=n, state_soa=state_soa, input_seq_soa=input_seq_soa
        )
        _assert_equal_soa(expected_final_state, actual_final_state, ir.state)
        _assert_equal_soa(expected_final_out, actual_final_out, ir.outputs)
        if run_entry is not None:
            print(f"[ok] {run_entry} parity")
        else:
            print(f"[ok] {step_entry}-loop parity ({steps} steps)")

        print("[success] futhark backend compile + parity checks passed")
        return 0
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
