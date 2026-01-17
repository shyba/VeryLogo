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
from stc.tick_ir import SimdType, TickIR
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
    lines.append("  for (;;) {")
    lines.append(f"    for (int i = 0; i < {in_words}; i++) {{")
    lines.append('      if (scanf("%llx", &tmp) != 1) {')
    lines.append("        return 0;")
    lines.append("      }")
    lines.append("      in[i] = (uint64_t)tmp;")
    lines.append("    }")
    lines.append("    stc_eval(in, out);")
    lines.append(f"    for (int i = 0; i < {out_words}; i++) {{")
    lines.append(
        '      printf("%016llx%s", (unsigned long long)out[i], (i == '
        f"{out_words - 1}"
        ') ? "\\n" : " ");'
    )
    lines.append("    }")
    lines.append("  }")
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
    p.add_argument("--out", type=Path)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--cc", default="cc")
    p.add_argument("--cflags", action="append", default=[])
    p.add_argument("--exe", type=Path)
    args = p.parse_args()

    ir = _load_tick_ir(args.ir)
    validate_tick_ir(ir)

    c = _emit_c(ir, args.backend)
    if args.out:
        args.out.write_text(c, encoding="utf-8")
    else:
        print(c)

    if not args.compile and not args.run:
        return

    runner_c = _emit_runner_c(ir, args.backend)
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d)
        impl_c = out_dir / "impl.c"
        impl_c.write_text(c, encoding="utf-8")
        main_c = out_dir / "main.c"
        main_c.write_text(runner_c, encoding="utf-8")

        exe = args.exe if args.exe else out_dir / "runner"
        cflags = ["-std=c99", "-O2", *_default_cflags(args.backend), *args.cflags]
        subprocess.run(
            [args.cc, *cflags, "-o", str(exe), str(main_c), str(impl_c)],
            check=True,
        )
        if args.run:
            subprocess.run([str(exe)], check=False)


if __name__ == "__main__":
    main()
