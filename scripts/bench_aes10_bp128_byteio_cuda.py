from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

from scripts import bench_aes10_bp128_cuda as base


def _converter_kernels_source() -> str:
    return r"""
extern "C" __global__ void bytes_to_planes4_kernel(
    const uint4* __restrict__ in_blocks4,
    uint4* __restrict__ out_planes4,
    uint32_t n_threads
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t lane = gid & 31u;
    uint32_t tid = gid >> 5;
    if (tid >= n_threads) return;

    uint4 blk = in_blocks4[(size_t)tid * 32u + (size_t)lane];
    uint8_t by[16];
    by[0] = (uint8_t)(blk.x & 0xffu);
    by[1] = (uint8_t)((blk.x >> 8) & 0xffu);
    by[2] = (uint8_t)((blk.x >> 16) & 0xffu);
    by[3] = (uint8_t)((blk.x >> 24) & 0xffu);
    by[4] = (uint8_t)(blk.y & 0xffu);
    by[5] = (uint8_t)((blk.y >> 8) & 0xffu);
    by[6] = (uint8_t)((blk.y >> 16) & 0xffu);
    by[7] = (uint8_t)((blk.y >> 24) & 0xffu);
    by[8] = (uint8_t)(blk.z & 0xffu);
    by[9] = (uint8_t)((blk.z >> 8) & 0xffu);
    by[10] = (uint8_t)((blk.z >> 16) & 0xffu);
    by[11] = (uint8_t)((blk.z >> 24) & 0xffu);
    by[12] = (uint8_t)(blk.w & 0xffu);
    by[13] = (uint8_t)((blk.w >> 8) & 0xffu);
    by[14] = (uint8_t)((blk.w >> 16) & 0xffu);
    by[15] = (uint8_t)((blk.w >> 24) & 0xffu);

    #pragma unroll
    for (int b = 0; b < 16; b++) {
        uint8_t pb = by[b];
        uint32_t m0 = __ballot_sync(0xffffffffu, (pb >> 0) & 1u);
        uint32_t m1 = __ballot_sync(0xffffffffu, (pb >> 1) & 1u);
        uint32_t m2 = __ballot_sync(0xffffffffu, (pb >> 2) & 1u);
        uint32_t m3 = __ballot_sync(0xffffffffu, (pb >> 3) & 1u);
        uint32_t m4 = __ballot_sync(0xffffffffu, (pb >> 4) & 1u);
        uint32_t m5 = __ballot_sync(0xffffffffu, (pb >> 5) & 1u);
        uint32_t m6 = __ballot_sync(0xffffffffu, (pb >> 6) & 1u);
        uint32_t m7 = __ballot_sync(0xffffffffu, (pb >> 7) & 1u);
        if (lane == 0u) {
            size_t idx0 = ((size_t)b * 2u + 0u) * (size_t)n_threads + (size_t)tid;
            size_t idx1 = ((size_t)b * 2u + 1u) * (size_t)n_threads + (size_t)tid;
            out_planes4[idx0] = make_uint4(m0, m1, m2, m3);
            out_planes4[idx1] = make_uint4(m4, m5, m6, m7);
        }
    }
}

extern "C" __global__ void planes4_to_bytes_kernel(
    const uint4* __restrict__ in_planes4,
    uint4* __restrict__ out_blocks4,
    uint32_t n_threads
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t lane = gid & 31u;
    uint32_t tid = gid >> 5;
    if (tid >= n_threads) return;

    uint8_t by[16];
    #pragma unroll
    for (int b = 0; b < 16; b++) {
        uint32_t m0 = 0u, m1 = 0u, m2 = 0u, m3 = 0u;
        uint32_t m4 = 0u, m5 = 0u, m6 = 0u, m7 = 0u;
        if (lane == 0u) {
            size_t idx0 = ((size_t)b * 2u + 0u) * (size_t)n_threads + (size_t)tid;
            size_t idx1 = ((size_t)b * 2u + 1u) * (size_t)n_threads + (size_t)tid;
            uint4 v0 = in_planes4[idx0];
            uint4 v1 = in_planes4[idx1];
            m0 = v0.x;
            m1 = v0.y;
            m2 = v0.z;
            m3 = v0.w;
            m4 = v1.x;
            m5 = v1.y;
            m6 = v1.z;
            m7 = v1.w;
        }
        m0 = __shfl_sync(0xffffffffu, m0, 0);
        m1 = __shfl_sync(0xffffffffu, m1, 0);
        m2 = __shfl_sync(0xffffffffu, m2, 0);
        m3 = __shfl_sync(0xffffffffu, m3, 0);
        m4 = __shfl_sync(0xffffffffu, m4, 0);
        m5 = __shfl_sync(0xffffffffu, m5, 0);
        m6 = __shfl_sync(0xffffffffu, m6, 0);
        m7 = __shfl_sync(0xffffffffu, m7, 0);

        uint8_t pb = 0u;
        pb |= (uint8_t)(((m0 >> lane) & 1u) << 0);
        pb |= (uint8_t)(((m1 >> lane) & 1u) << 1);
        pb |= (uint8_t)(((m2 >> lane) & 1u) << 2);
        pb |= (uint8_t)(((m3 >> lane) & 1u) << 3);
        pb |= (uint8_t)(((m4 >> lane) & 1u) << 4);
        pb |= (uint8_t)(((m5 >> lane) & 1u) << 5);
        pb |= (uint8_t)(((m6 >> lane) & 1u) << 6);
        pb |= (uint8_t)(((m7 >> lane) & 1u) << 7);
        by[b] = pb;
    }

    uint32_t w0 = ((uint32_t)by[0]) | ((uint32_t)by[1] << 8) | ((uint32_t)by[2] << 16) |
                  ((uint32_t)by[3] << 24);
    uint32_t w1 = ((uint32_t)by[4]) | ((uint32_t)by[5] << 8) | ((uint32_t)by[6] << 16) |
                  ((uint32_t)by[7] << 24);
    uint32_t w2 = ((uint32_t)by[8]) | ((uint32_t)by[9] << 8) | ((uint32_t)by[10] << 16) |
                  ((uint32_t)by[11] << 24);
    uint32_t w3 = ((uint32_t)by[12]) | ((uint32_t)by[13] << 8) | ((uint32_t)by[14] << 16) |
                  ((uint32_t)by[15] << 24);
    out_blocks4[(size_t)tid * 32u + (size_t)lane] = make_uint4(w0, w1, w2, w3);
}
"""


def _host_bench_cu_source() -> str:
    rk_bytes = base._rk_bytes_c_initializer()
    return (
        r"""
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

static void ck(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) {
    const char* s = 0;
    cuGetErrorString(r, &s);
    fprintf(stderr, "%s failed: %d (%s)\n", what, (int)r, s ? s : "?");
    exit(1);
  }
}

static void seed_plaintext_blocks(uint8_t* in_bytes, int threads) {
  const uint8_t pt[16] = {
    0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,
    0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff
  };
  for (int tid = 0; tid < threads; tid++) {
    for (int lane = 0; lane < 32; lane++) {
      uint8_t* dst = in_bytes + (((size_t)tid * 32u + (size_t)lane) * 16u);
      for (int i = 0; i < 16; i++) dst[i] = pt[i];
    }
  }
}

static void build_rk_bits(uint32_t rk_bits[11][16][8]) {
  static const uint8_t rk_bytes[11][16] = {
"""
        + rk_bytes
        + r"""
  };
  for (int r = 0; r < 11; r++) {
    for (int b = 0; b < 16; b++) {
      uint8_t x = rk_bytes[r][b];
      for (int bit = 0; bit < 8; bit++) {
        rk_bits[r][b][bit] = ((x >> bit) & 1u) ? 0xffffffffu : 0u;
      }
    }
  }
}

int main(int argc, char** argv) {
  const char* cubin = argv[1];
  int threads = atoi(argv[2]);
  int block = atoi(argv[3]);
  int reps = atoi(argv[4]);
  int do_check = (argc > 5) ? atoi(argv[5]) : 0;
  if ((block & 31) != 0) {
    fprintf(stderr, "block must be multiple of 32 for converter kernels\n");
    return 2;
  }

  ck(cuInit(0), "cuInit");
  CUdevice dev;
  ck(cuDeviceGet(&dev, 0), "cuDeviceGet");
  CUcontext ctx;
  ck(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");

  CUmodule mod;
  ck(cuModuleLoad(&mod, cubin), "cuModuleLoad");
  CUfunction fn_in, fn_aes, fn_out;
  ck(cuModuleGetFunction(&fn_in, mod, "bytes_to_planes4_kernel"), "cuModuleGetFunction(conv_in)");
  ck(cuModuleGetFunction(&fn_aes, mod, "aes10_bp128_kernel"), "cuModuleGetFunction(aes)");
  ck(cuModuleGetFunction(&fn_out, mod, "planes4_to_bytes_kernel"), "cuModuleGetFunction(conv_out)");

  CUdeviceptr d_rk = 0;
  size_t rk_nbytes = 0;
  ck(cuModuleGetGlobal(&d_rk, &rk_nbytes, mod, "RK_BITS"), "cuModuleGetGlobal(RK_BITS)");
  uint32_t h_rk_bits[11][16][8];
  build_rk_bits(h_rk_bits);
  if (rk_nbytes < sizeof(h_rk_bits)) {
    fprintf(stderr, "RK_BITS symbol too small: %zu < %zu\n", rk_nbytes, sizeof(h_rk_bits));
    return 2;
  }
  ck(cuMemcpyHtoD(d_rk, h_rk_bits, sizeof(h_rk_bits)), "cuMemcpyHtoD(RK_BITS)");

  size_t blocks32 = (size_t)threads * 32u;
  size_t bytes_io = blocks32 * 16u;
  size_t planes_bytes = (size_t)threads * 32u * sizeof(uint4);

  CUdeviceptr d_in_bytes = 0, d_out_bytes = 0, d_planes0 = 0, d_planes1 = 0;
  ck(cuMemAlloc(&d_in_bytes, bytes_io), "cuMemAlloc(in_bytes)");
  ck(cuMemAlloc(&d_out_bytes, bytes_io), "cuMemAlloc(out_bytes)");
  ck(cuMemAlloc(&d_planes0, planes_bytes), "cuMemAlloc(planes0)");
  ck(cuMemAlloc(&d_planes1, planes_bytes), "cuMemAlloc(planes1)");

  uint8_t* h_in = (uint8_t*)malloc(bytes_io);
  if (!h_in) {
    fprintf(stderr, "malloc failed for input\n");
    return 2;
  }
  seed_plaintext_blocks(h_in, threads);
  ck(cuMemcpyHtoD(d_in_bytes, h_in, bytes_io), "cuMemcpyHtoD(input)");
  ck(cuMemsetD8(d_out_bytes, 0, bytes_io), "cuMemsetD8(out_bytes)");

  int grid_aes = (threads + block - 1) / block;
  int conv_threads = threads * 32;
  int grid_conv = (conv_threads + block - 1) / block;
  void* p_in[] = { &d_in_bytes, &d_planes0, &threads };
  void* p_aes[] = { &d_planes0, &d_planes1, &threads };
  void* p_out[] = { &d_planes1, &d_out_bytes, &threads };

  if (do_check) {
    ck(cuLaunchKernel(fn_in, grid_conv, 1, 1, block, 1, 1, 0, 0, p_in, 0), "cuLaunchKernel(conv_in check)");
    ck(cuLaunchKernel(fn_aes, grid_aes, 1, 1, block, 1, 1, 0, 0, p_aes, 0), "cuLaunchKernel(aes check)");
    ck(cuLaunchKernel(fn_out, grid_conv, 1, 1, block, 1, 1, 0, 0, p_out, 0), "cuLaunchKernel(conv_out check)");
    ck(cuCtxSynchronize(), "cuCtxSynchronize(check)");

    uint8_t out0[16];
    ck(cuMemcpyDtoH(out0, d_out_bytes, sizeof(out0)), "cuMemcpyDtoH(check)");
    const uint8_t exp[16] = {
      0x69,0xc4,0xe0,0xd8,0x6a,0x7b,0x04,0x30,
      0xd8,0xcd,0xb7,0x80,0x70,0xb4,0xc5,0x5a
    };
    int errors = 0;
    for (int i = 0; i < 16; i++) if (out0[i] != exp[i]) errors++;
    if (errors) {
      fprintf(stderr, "FAIL: AES KAT mismatch\n  got=");
      for (int i = 0; i < 16; i++) fprintf(stderr, "%02x", out0[i]);
      fprintf(stderr, "\n  exp=");
      for (int i = 0; i < 16; i++) fprintf(stderr, "%02x", exp[i]);
      fprintf(stderr, "\n");
      return 1;
    }
    printf("PASS: AES-128 known-answer check matched\n");
  }

  ck(cuLaunchKernel(fn_in, grid_conv, 1, 1, block, 1, 1, 0, 0, p_in, 0), "cuLaunchKernel(conv_in warmup)");
  ck(cuLaunchKernel(fn_aes, grid_aes, 1, 1, block, 1, 1, 0, 0, p_aes, 0), "cuLaunchKernel(aes warmup)");
  ck(cuLaunchKernel(fn_out, grid_conv, 1, 1, block, 1, 1, 0, 0, p_out, 0), "cuLaunchKernel(conv_out warmup)");
  ck(cuCtxSynchronize(), "cuCtxSynchronize(warmup)");

  CUevent e0, e1;
  ck(cuEventCreate(&e0, 0), "cuEventCreate(start)");
  ck(cuEventCreate(&e1, 0), "cuEventCreate(end)");

  ck(cuEventRecord(e0, 0), "cuEventRecord(start full)");
  for (int i = 0; i < reps; i++) {
    ck(cuLaunchKernel(fn_in, grid_conv, 1, 1, block, 1, 1, 0, 0, p_in, 0), "cuLaunchKernel(conv_in)");
    ck(cuLaunchKernel(fn_aes, grid_aes, 1, 1, block, 1, 1, 0, 0, p_aes, 0), "cuLaunchKernel(aes)");
    ck(cuLaunchKernel(fn_out, grid_conv, 1, 1, block, 1, 1, 0, 0, p_out, 0), "cuLaunchKernel(conv_out)");
  }
  ck(cuEventRecord(e1, 0), "cuEventRecord(end full)");
  ck(cuEventSynchronize(e1), "cuEventSynchronize(full)");
  float full_ms = 0.0f;
  ck(cuEventElapsedTime(&full_ms, e0, e1), "cuEventElapsedTime(full)");

  ck(cuEventRecord(e0, 0), "cuEventRecord(start conv)");
  for (int i = 0; i < reps; i++) {
    ck(cuLaunchKernel(fn_in, grid_conv, 1, 1, block, 1, 1, 0, 0, p_in, 0), "cuLaunchKernel(conv_in only)");
    ck(cuLaunchKernel(fn_out, grid_conv, 1, 1, block, 1, 1, 0, 0, p_out, 0), "cuLaunchKernel(conv_out only)");
  }
  ck(cuEventRecord(e1, 0), "cuEventRecord(end conv)");
  ck(cuEventSynchronize(e1), "cuEventSynchronize(conv)");
  float conv_ms = 0.0f;
  ck(cuEventElapsedTime(&conv_ms, e0, e1), "cuEventElapsedTime(conv)");

  ck(cuLaunchKernel(fn_in, grid_conv, 1, 1, block, 1, 1, 0, 0, p_in, 0), "cuLaunchKernel(conv_in pre aes)");
  ck(cuCtxSynchronize(), "cuCtxSynchronize(pre aes)");
  ck(cuEventRecord(e0, 0), "cuEventRecord(start aes)");
  for (int i = 0; i < reps; i++) {
    ck(cuLaunchKernel(fn_aes, grid_aes, 1, 1, block, 1, 1, 0, 0, p_aes, 0), "cuLaunchKernel(aes only)");
  }
  ck(cuEventRecord(e1, 0), "cuEventRecord(end aes)");
  ck(cuEventSynchronize(e1), "cuEventSynchronize(aes)");
  float aes_ms = 0.0f;
  ck(cuEventElapsedTime(&aes_ms, e0, e1), "cuEventElapsedTime(aes)");

  double full_s = (double)full_ms / 1000.0;
  double conv_s = (double)conv_ms / 1000.0;
  double aes_s = (double)aes_ms / 1000.0;
  double total_evals = (double)threads * 32.0 * (double)reps;

  double full_eps = total_evals / full_s;
  double conv_eps = total_evals / conv_s;
  double aes_eps = total_evals / aes_s;

  printf("threads=%d block=%d reps=%d\n", threads, block, reps);
  printf("full_pipeline: %.3f ms, %.3fB evals/sec, %.2f MiB/s\n",
         full_ms, full_eps / 1e9, full_eps * 16.0 / (1024.0 * 1024.0));
  printf("converter_only: %.3f ms, %.3fB evals/sec-equivalent\n",
         conv_ms, conv_eps / 1e9);
  printf("aes_only: %.3f ms, %.3fB evals/sec, %.2f MiB/s\n",
         aes_ms, aes_eps / 1e9, aes_eps * 16.0 / (1024.0 * 1024.0));

  free(h_in);
  cuMemFree(d_in_bytes);
  cuMemFree(d_out_bytes);
  cuMemFree(d_planes0);
  cuMemFree(d_planes1);
  cuModuleUnload(mod);
  cuCtxDestroy(ctx);
  return 0;
}
"""
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sm", default="sm_61")
    ap.add_argument("--threads", type=int, default=65_536)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--check", action="store_true")
    ap.add_argument(
        "--autotune-blocks",
        default="64,128,256",
        help="Comma-separated block sizes for sweeps (must be multiples of 32)",
    )
    ap.add_argument("--out", default="out/aes10_bp128_byteio")
    args = ap.parse_args()

    mapped, lop3_count = base._build_bp128_mapped(None)
    print(f"lop3.b32 count in BP128 S-box function: {lop3_count}")
    sbox_cuda = base._emit_sbox_inline_cuda(
        mapped, func_name="sbox_bp128_lop3_inline", noinline=False
    )
    kernel_src = (
        base._kernel_cu_source_replacement_coalesced4(sbox_cuda)
        + "\n"
        + _converter_kernels_source()
    )
    host_src = _host_bench_cu_source()

    with tempfile.TemporaryDirectory(prefix="aes10_bp128_byteio_") as td:
        tdp = Path(td)
        kernel_cu = tdp / "kernel.cu"
        kernel_cubin = tdp / "kernel.cubin"
        host_cu = tdp / "host.cu"
        host_bin = tdp / "bench_host"

        out_prefix = Path(args.out)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        out_ptx = out_prefix.with_suffix(".kernel.ptx")
        out_cubin = out_prefix.with_suffix(".cubin")

        kernel_cu.write_text(kernel_src, encoding="utf-8")
        host_cu.write_text(host_src, encoding="utf-8")

        res = subprocess.run(
            [
                "nvcc",
                "-ptx",
                "-O3",
                f"-arch={args.sm}",
                "-o",
                str(out_ptx),
                str(kernel_cu),
            ],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            print(res.stderr.strip()[:4000])
            return res.returncode

        res = subprocess.run(
            [
                "nvcc",
                "-cubin",
                "-O3",
                f"-arch={args.sm}",
                "-o",
                str(kernel_cubin),
                str(kernel_cu),
            ],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            print(res.stderr.strip()[:4000])
            return res.returncode
        out_cubin.write_bytes(kernel_cubin.read_bytes())

        res = subprocess.run(
            ["nvcc", "-O3", "-o", str(host_bin), str(host_cu), "-lcuda"],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            print(res.stderr.strip()[:4000])
            return res.returncode

        blocks = [args.block]
        if args.autotune_blocks.strip():
            blocks = [
                int(x.strip()) for x in args.autotune_blocks.split(",") if x.strip()
            ]

        best_eval = -1.0
        best_block = -1
        for idx, block in enumerate(blocks):
            do_check = "1" if (args.check and idx == 0) else "0"
            run = subprocess.run(
                [
                    str(host_bin),
                    str(kernel_cubin),
                    str(args.threads),
                    str(block),
                    str(args.reps),
                    do_check,
                ],
                capture_output=True,
                text=True,
                timeout=900,
            )
            print(run.stdout.strip())
            if run.returncode != 0:
                if run.stderr.strip():
                    print(run.stderr.strip()[:2000])
                print(f"skipping block={block} due to failure")
                continue
            m = re.search(r"full_pipeline:\s+[0-9.]+ ms,\s+([0-9.]+)B evals/sec", run.stdout)
            if m:
                ev = float(m.group(1))
                if ev > best_eval:
                    best_eval = ev
                    best_block = block

        if best_eval < 0.0:
            print("no successful runs")
            return 3
        if len(blocks) > 1:
            print(f"autotune best block={best_block} ({best_eval:.3f}B evals/sec)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
