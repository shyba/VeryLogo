#!/usr/bin/env python3
"""
Benchmark BP-128 S-box compiled to PTX using `lop3.b32`.

This script:
1) Builds BP-128 circuit and maps some 3-input cones to ternary gates
2) Emits PTX via the scheduler PTX emitter (uses `lop3.b32` for ternary gates)
3) Wraps the emitted `.func` in a `.entry` kernel that runs one instance per thread
4) Assembles with `ptxas`
5) Compiles a small CUDA driver program that loads the cubin and benchmarks it

Interpretation:
- Each thread processes one 8x32-bit "bit-plane" instance.
- Each 32-bit bit-plane represents 32 independent S-box evaluations.
- So total evals = threads * 32 per kernel invocation.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bench_lop3_cuda import _compute_cone_imm8
from scripts.benchmark_ternary_sbox import find_3input_cones
from scripts.bp_circuit_sbox import build_bp_sbox
from stc.backend_sched import generate_scheduled_code
from stc.circuit_synth import CircuitState


def _wrap_ptx_as_kernel(ptx_func: str, func_name: str, sm: str) -> str:
    header = "\n".join(
        [
            ".version 7.0",
            f".target {sm}",
            ".address_size 64",
            "",
        ]
    )

    # Kernel expects:
    #   in_ptr:  pointer to uint32_t[N*8]
    #   out_ptr: pointer to uint32_t[N*8]
    # Each thread uses base = ptr + tid*8*4.
    kernel = f"""
.visible .entry {func_name}_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n_threads
) {{
    .reg .b32 %rK<8>;
    .reg .b64 %rdK<8>;
    .reg .pred %p<1>;
    .param .u64 __p_in;
    .param .u64 __p_out;

    ld.param.u64 %rdK0, [in_ptr];
    ld.param.u64 %rdK1, [out_ptr];
    ld.param.u32 %rK0, [n_threads];

    // tid = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32 %rK1, %ctaid.x;
    mov.u32 %rK2, %ntid.x;
    mov.u32 %rK3, %tid.x;
    mad.lo.u32 %rK4, %rK1, %rK2, %rK3;

    setp.ge.u32 %p0, %rK4, %rK0;
    @%p0 bra DONE;

    // byte_offset = tid * (8 * 4)
    mul.lo.u32 %rK5, %rK4, 32;
    cvt.u64.u32 %rdK2, %rK5;
    add.u64 %rdK3, %rdK0, %rdK2;
    add.u64 %rdK4, %rdK1, %rdK2;

    st.param.u64 [__p_in], %rdK3;
    st.param.u64 [__p_out], %rdK4;
    call.uni (), {func_name}, (__p_in, __p_out);

DONE:
    ret;
}}
"""

    # The PTX emitter currently emits `.visible .func <name>(...) { ... }` without
    # a `.version`/`.target` header, so prepend our own header and append kernel.
    return header + ptx_func + "\n" + kernel


def _emit_cpu_ref_c(mapped: CircuitState) -> str:
    gates = list(mapped.gates)
    input_bits = mapped.input_bits
    outputs = list(mapped.outputs)

    def reg_expr(node: int) -> str:
        if node < input_bits:
            return f"in[{node}]"
        return f"t[{node - input_bits}]"

    lines: list[str] = []
    lines.append(
        "static inline uint32_t ternary_u32(uint32_t a, uint32_t b, uint32_t c, uint8_t imm8) {"
    )
    lines.append(
        "  // imm8 bit i corresponds to i = (a<<2)|(b<<1)|c with a/b/c as 0/1 bits."
    )
    lines.append("  uint32_t na = ~a, nb = ~b, nc = ~c;")
    lines.append("  uint32_t r = 0;")
    for i in range(8):
        conds = []
        conds.append("a" if ((i >> 2) & 1) else "na")
        conds.append("b" if ((i >> 1) & 1) else "nb")
        conds.append("c" if (i & 1) else "nc")
        term = " & ".join(conds)
        lines.append(f"  if (imm8 & 0x{1<<i:02x}) r |= ({term});")
    lines.append("  return r;")
    lines.append("}")
    lines.append("")

    lines.append("static inline void cpu_sbox(uint32_t in[8], uint32_t out[8]) {")
    lines.append(f"  uint32_t t[{len(gates)}];")
    for g_idx, gate in enumerate(gates):
        if len(gate) == 5:
            _, a, b, c, imm8 = gate
            lines.append(
                f"  t[{g_idx}] = ternary_u32({reg_expr(a)}, {reg_expr(b)}, {reg_expr(c)}, 0x{imm8:02x});"
            )
            continue
        op, a, b = gate
        if op == "xor":
            lines.append(f"  t[{g_idx}] = {reg_expr(a)} ^ {reg_expr(b)};")
        elif op == "and":
            lines.append(f"  t[{g_idx}] = {reg_expr(a)} & {reg_expr(b)};")
        elif op == "or":
            lines.append(f"  t[{g_idx}] = {reg_expr(a)} | {reg_expr(b)};")
        elif op == "not":
            lines.append(f"  t[{g_idx}] = ~{reg_expr(a)};")
        elif op == "const":
            lines.append(f"  t[{g_idx}] = {0xFFFFFFFF if a else 0};")
        else:
            lines.append(f"  t[{g_idx}] = 0;")

    for out_i, (node, inv) in enumerate(outputs):
        expr = reg_expr(node)
        if inv:
            lines.append(f"  out[{out_i}] = ~{expr};")
        else:
            lines.append(f"  out[{out_i}] = {expr};")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sm", default="sm_61")
    ap.add_argument("--threads", type=int, default=1_048_576)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument(
        "--check", action="store_true", help="Run a small correctness check"
    )
    ap.add_argument("--out", default="out/bp128_lop3_bench.ptx")
    args = ap.parse_args()

    bp = build_bp_sbox()
    cones = find_3input_cones(bp)
    new_gates = list(bp.gates)
    for cone in cones:
        root = cone["root"]
        leaves = cone["leaves"]
        imm8 = _compute_cone_imm8(bp, root, leaves)
        new_gates[root] = ("ternary", leaves[0], leaves[1], leaves[2], imm8)

    mapped = CircuitState(
        input_bits=bp.input_bits,
        output_bits=bp.output_bits,
        gates=new_gates,
        outputs=list(bp.outputs),
        gate_count=len(new_gates),
    ).eliminate_dead_code()

    func_name = "sbox_bp128_lop3"
    ptx_func = generate_scheduled_code(mapped, target="ptx", function_name=func_name)
    lop3_count = len(re.findall(r"\blop3\.b32\b", ptx_func))
    print(f"lop3.b32 count in function: {lop3_count}")

    ptx = _wrap_ptx_as_kernel(ptx_func, func_name, args.sm)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(ptx)

    cpu_ref = _emit_cpu_ref_c(mapped)

    with tempfile.TemporaryDirectory() as td:
        cubin = os.path.join(td, "kern.cubin")
        res = subprocess.run(
            ["ptxas", f"-arch={args.sm}", args.out, "-o", cubin],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            print(res.stderr.strip()[:4000])
            return res.returncode

        host_cc = (
            r"""
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void ck(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) {
    const char* s = 0;
    cuGetErrorString(r, &s);
    fprintf(stderr, "%s failed: %d (%s)\n", what, (int)r, s ? s : "?");
    exit(1);
  }
}

"""
            + cpu_ref
            + r"""
int main(int argc, char** argv) {
  const char* cubin = argv[1];
  int threads = atoi(argv[2]);
  int block = atoi(argv[3]);
  int reps = atoi(argv[4]);
  int do_check = (argc > 5) ? atoi(argv[5]) : 0;

  ck(cuInit(0), "cuInit");
  CUdevice dev;
  ck(cuDeviceGet(&dev, 0), "cuDeviceGet");
  CUcontext ctx;
  ck(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");

  CUmodule mod;
  ck(cuModuleLoad(&mod, cubin), "cuModuleLoad");
  CUfunction fn;
  ck(cuModuleGetFunction(&fn, mod, "sbox_bp128_lop3_kernel"), "cuModuleGetFunction");

  size_t words = (size_t)threads * 8;
  size_t bytes = words * sizeof(uint32_t);
  CUdeviceptr d_in, d_out;
  ck(cuMemAlloc(&d_in, bytes), "cuMemAlloc(in)");
  ck(cuMemAlloc(&d_out, bytes), "cuMemAlloc(out)");

  uint32_t* h_in = (uint32_t*)malloc(bytes);
  uint32_t* h_out = (uint32_t*)malloc(bytes);
  uint32_t* h_exp = (uint32_t*)malloc(bytes);
  if (!h_in || !h_out || !h_exp) { fprintf(stderr, "malloc failed\n"); return 2; }

  for (int t = 0; t < threads; t++) {
    for (int i = 0; i < 8; i++) {
      uint32_t x = (uint32_t)(0x9E3779B9u * (uint32_t)(t + 1)) ^ (uint32_t)(0xA5A5A5A5u + 0x11111111u * i);
      // Mix it a bit.
      x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
      h_in[(size_t)t * 8 + i] = x;
    }
  }

  ck(cuMemcpyHtoD(d_in, h_in, bytes), "cuMemcpyHtoD(in)");
  ck(cuMemsetD8(d_out, 0, bytes), "cuMemsetD8(out)");

  int grid = (threads + block - 1) / block;
  void* params[] = { &d_in, &d_out, &threads };

  if (do_check) {
    // One run
    ck(cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0, params, 0), "cuLaunchKernel(check)");
    ck(cuCtxSynchronize(), "cuCtxSynchronize(check)");
    ck(cuMemcpyDtoH(h_out, d_out, bytes), "cuMemcpyDtoH(out)");

    int errors = 0;
    for (int t = 0; t < threads; t++) {
      uint32_t in_planes[8], out_planes[8];
      for (int i = 0; i < 8; i++) in_planes[i] = h_in[(size_t)t * 8 + i];
      cpu_sbox(in_planes, out_planes);
      for (int i = 0; i < 8; i++) h_exp[(size_t)t * 8 + i] = out_planes[i];
    }

    for (size_t i = 0; i < words; i++) {
      if (h_out[i] != h_exp[i]) {
        if (errors < 8) {
          fprintf(stderr, "mismatch[%zu]: got=0x%08x exp=0x%08x\n", i, h_out[i], h_exp[i]);
        }
        errors++;
      }
    }
    if (errors) {
      fprintf(stderr, "FAIL: %d mismatches\n", errors);
      return 1;
    }
    printf("PASS: output matches CPU reference\n");
  }

  // Warmup for timing
  ck(cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0, params, 0), "cuLaunchKernel(warmup)");
  ck(cuCtxSynchronize(), "cuCtxSynchronize");

  CUevent e0, e1;
  ck(cuEventCreate(&e0, 0), "cuEventCreate");
  ck(cuEventCreate(&e1, 0), "cuEventCreate");
  ck(cuEventRecord(e0, 0), "cuEventRecord(start)");
  for (int i = 0; i < reps; i++) {
    ck(cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0, params, 0), "cuLaunchKernel");
  }
  ck(cuEventRecord(e1, 0), "cuEventRecord(end)");
  ck(cuEventSynchronize(e1), "cuEventSynchronize");

  float ms = 0.0f;
  ck(cuEventElapsedTime(&ms, e0, e1), "cuEventElapsedTime");

  double seconds = (double)ms / 1000.0;
  double total_evals = (double)threads * 32.0 * (double)reps;
  double evals_per_sec = total_evals / seconds;
  double ns_per_eval = (seconds / total_evals) * 1e9;
  printf("threads=%d block=%d reps=%d\n", threads, block, reps);
  printf("elapsed=%.3f ms\n", ms);
  printf("bp128_lop3: %.3f ns/eval, %.3fB evals/sec\n", ns_per_eval, evals_per_sec / 1e9);

  free(h_in);
  free(h_out);
  free(h_exp);
  cuMemFree(d_in);
  cuMemFree(d_out);
  cuModuleUnload(mod);
  cuCtxDestroy(ctx);
  return 0;
}
"""
        )
        host_path = os.path.join(td, "bench.cu")
        exe = os.path.join(td, "bench")
        Path(host_path).write_text(host_cc)
        res = subprocess.run(
            ["nvcc", "-O3", "-o", exe, host_path, "-lcuda"],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            print(res.stderr.strip()[:4000])
            return res.returncode

        do_check = "1" if args.check else "0"
        res = subprocess.run(
            [exe, cubin, str(args.threads), str(args.block), str(args.reps), do_check],
            capture_output=True,
            text=True,
            timeout=300,
        )
        print(res.stdout.strip())
        if res.returncode != 0 and res.stderr:
            print(res.stderr.strip()[:2000])
        return res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
