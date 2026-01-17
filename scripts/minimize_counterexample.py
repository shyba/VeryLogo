from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from stc.backend_x86_auto import emit_x86_auto_c
from stc.backend_x86_avx import emit_x86_avx_c
from stc.backend_x86_avx2 import emit_x86_avx2_c
from stc.backend_x86_avx512 import emit_x86_avx512_c
from stc.backend_x86_avx512_float import emit_x86_avx512_float_c
from stc.backend_x86_sse2 import emit_x86_sse2_c
from stc.interp import eval_expr
from stc.tick_ir import BitVecType, SimdType, TickIR
from stc.tick_ir_validate import validate_tick_ir


def _load_tick_ir(path: Path) -> TickIR:
    data = json.loads(path.read_text(encoding="utf-8"))
    return TickIR.from_dict(data)


def _emit_c(ir: TickIR, backend: str) -> str:
    if backend == "sse2":
        return emit_x86_sse2_c(ir)
    if backend == "avx":
        return emit_x86_avx_c(ir)
    if backend == "avx2":
        return emit_x86_avx2_c(ir)
    if backend == "avx512":
        return emit_x86_avx512_c(ir)
    if backend == "avx512_float":
        return emit_x86_avx512_float_c(ir)
    if backend == "auto":
        return emit_x86_auto_c(ir)
    raise ValueError("unknown backend")


def _default_cflags(backend: str) -> list[str]:
    if backend == "sse2":
        return ["-msse2"]
    if backend == "avx":
        return ["-mavx"]
    if backend == "avx2":
        return ["-mavx2"]
    if backend == "avx512":
        return ["-mavx512f"]
    if backend == "avx512_float":
        return ["-mavx512f"]
    return []


def _words_per_value(ir: TickIR, backend: str) -> int:
    if backend == "sse2":
        return 2
    if backend == "avx":
        return 4
    if backend == "avx2":
        return 4
    if backend == "avx512":
        return 8
    if backend == "avx512_float":
        return 8
    widths = {t.total_width for t in ir.inputs.values() if isinstance(t, SimdType)}
    if len(widths) != 1:
        raise ValueError("auto backend requires uniform simd total_width inputs")
    (w,) = tuple(widths)
    if w % 64 != 0:
        raise ValueError("simd total_width must be divisible by 64")
    return w // 64


def _mask(width: int) -> int:
    if width <= 0:
        return 0
    return (1 << width) - 1


def _parse_int(v) -> int:
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s.startswith("0x"):
            return int(s, 16)
        return int(s, 0)
    raise TypeError("unsupported value type")


def _to_words(x: int, word_count: int) -> list[int]:
    return [(x >> (64 * i)) & ((1 << 64) - 1) for i in range(word_count)]


def _words_to_int(words: list[int]) -> int:
    out = 0
    for i, w in enumerate(words):
        out |= int(w) << (64 * i)
    return out


def _emit_runner_c(ir: TickIR, backend: str) -> str:
    validate_tick_ir(ir)
    word_count = _words_per_value(ir, backend)
    input_order = sorted(ir.inputs.keys())
    output_order = sorted(ir.outputs.keys())
    in_words = len(input_order) * word_count
    out_words = len(output_order) * word_count

    lines: list[str] = []
    lines.append("#include <stdint.h>")
    lines.append("#include <stdio.h>")
    lines.append("")
    lines.append("void stc_eval(const uint64_t* in, uint64_t* out);")
    lines.append("")
    lines.append("int main(void) {")
    lines.append(f"  uint64_t in[{in_words}];")
    lines.append(f"  uint64_t out[{out_words}];")
    lines.append("  unsigned long long tmp;")
    lines.append(f"  for (int i = 0; i < {in_words}; i++) {{")
    lines.append('    if (scanf("%llx", &tmp) != 1) {')
    lines.append("      return 2;")
    lines.append("    }")
    lines.append("    in[i] = (uint64_t)tmp;")
    lines.append("  }")
    lines.append("  stc_eval(in, out);")
    lines.append(f"  for (int i = 0; i < {out_words}; i++) {{")
    lines.append(
        '    printf("%016llx%s", (unsigned long long)out[i], (i == '
        f"{out_words - 1}"
        ') ? "\\n" : " ");'
    )
    lines.append("  }")
    lines.append("  return 0;")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ir", type=Path, required=True)
    p.add_argument(
        "--backend",
        choices=["sse2", "avx", "avx2", "avx512", "avx512_float", "auto"],
        required=True,
    )
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--cc", default="cc")
    p.add_argument("--cflags", action="append", default=[])
    p.add_argument("--max_passes", type=int, default=3)
    args = p.parse_args()

    ir = _load_tick_ir(args.ir)
    validate_tick_ir(ir)
    if ir.state:
        raise SystemExit("x86 minimizer supports combinational Tick-IR only")

    types = dict(ir.inputs)
    input_order = sorted(ir.inputs.keys())
    output_order = sorted(ir.outputs.keys())
    word_count = _words_per_value(ir, args.backend)

    case_data = json.loads(args.case.read_text(encoding="utf-8"))
    env: dict[str, int] = {}
    for name in input_order:
        if name not in case_data:
            raise SystemExit(f"missing input: {name}")
        t = types[name]
        if isinstance(t, SimdType):
            env[name] = _parse_int(case_data[name]) & _mask(t.total_width)
        elif isinstance(t, BitVecType):
            env[name] = _parse_int(case_data[name]) & _mask(t.width)
        else:
            raise SystemExit("unsupported input type")

    c = _emit_c(ir, args.backend)
    runner_c = _emit_runner_c(ir, args.backend)

    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d)
        impl_c = out_dir / "impl.c"
        impl_c.write_text(c, encoding="utf-8")
        main_c = out_dir / "main.c"
        main_c.write_text(runner_c, encoding="utf-8")
        exe = out_dir / "runner"
        cflags = ["-std=c99", "-O2", *_default_cflags(args.backend), *args.cflags]
        subprocess.run(
            [args.cc, *cflags, "-o", str(exe), str(main_c), str(impl_c)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        def backend_eval(cur_env: dict[str, int]) -> dict[str, int]:
            words: list[int] = []
            for name in input_order:
                t = types[name]
                if isinstance(t, SimdType):
                    words.extend(_to_words(cur_env[name], word_count))
                elif isinstance(t, BitVecType):
                    words.extend(_to_words(cur_env[name], word_count))
                else:
                    raise SystemExit("unsupported input type")
            stdin = " ".join(f"{w:x}" for w in words) + "\n"
            p0 = subprocess.run(
                [str(exe)],
                input=stdin.encode("utf-8"),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            out_words = [int(w, 16) for w in p0.stdout.decode("utf-8").strip().split()]
            if len(out_words) != len(output_order) * word_count:
                raise SystemExit("runner output word count mismatch")
            out: dict[str, int] = {}
            for idx, name in enumerate(output_order):
                out[name] = _words_to_int(
                    out_words[idx * word_count : (idx + 1) * word_count]
                )
            return out

        def interp_eval(cur_env: dict[str, int]) -> dict[str, int]:
            out: dict[str, int] = {}
            for name in output_order:
                out[name] = int(eval_expr(ir.output_exprs[name], types, cur_env))
            return out

        def is_failing(cur_env: dict[str, int]) -> bool:
            return backend_eval(cur_env) != interp_eval(cur_env)

        if not is_failing(env):
            raise SystemExit("case does not fail")

        for _ in range(args.max_passes):
            for name in input_order:
                t = types[name]
                if isinstance(t, SimdType):
                    width = t.total_width
                elif isinstance(t, BitVecType):
                    width = t.width
                else:
                    continue

                v = env[name] & _mask(width)
                if v == 0:
                    continue

                step = max(1, width // 2)
                while step >= 1:
                    changed = False
                    for off in range(0, width, step):
                        chunk = ((_mask(step)) << off) & _mask(width)
                        cand = v & ~chunk
                        if cand == v:
                            continue
                        env2 = dict(env)
                        env2[name] = cand
                        if is_failing(env2):
                            v = cand
                            env[name] = cand
                            changed = True
                    if not changed:
                        step //= 2

        out_case: dict[str, str] = {}
        for name in input_order:
            out_case[name] = hex(env[name])
        print(json.dumps(out_case, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
