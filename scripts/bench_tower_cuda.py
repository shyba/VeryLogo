#!/usr/bin/env python3
"""
Benchmark tower field S-box compiled to CUDA PTX with lop3.b32.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def wrap_ptx_as_kernel(
    ptx_func: str, func_name: str = "circuit", sm: str = "sm_75"
) -> str:
    """Wrap PTX function as CUDA kernel."""
    header = "\n".join(
        [
            ".version 7.0",
            f".target {sm}",
            ".address_size 64",
            "",
        ]
    )

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
    return header + ptx_func + "\n" + kernel


def generate_cuda_driver(cubin_path: str, iterations: int) -> str:
    """Generate CUDA driver C++ code."""
    return f"""
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#define CHECK_CUDA(call) {{ \\
    CUresult err = call; \\
    if (err != CUDA_SUCCESS) {{ \\
        const char* errStr; \\
        cuGetErrorString(err, &errStr); \\
        fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, errStr); \\
        exit(1); \\
    }} \\
}}

static uint64_t get_time_ns() {{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}}

int main() {{
    CHECK_CUDA(cuInit(0));

    CUdevice device;
    CUcontext context;
    CHECK_CUDA(cuDeviceGet(&device, 0));
    CHECK_CUDA(cuCtxCreate(&context, 0, device));

    // Load cubin
    CUmodule module;
    CHECK_CUDA(cuModuleLoad(&module, "{cubin_path}"));

    CUfunction kernel;
    CHECK_CUDA(cuModuleGetFunction(&kernel, module, "circuit_kernel"));

    // Allocate device memory
    const int threads = 65536;
    const int iterations = {iterations};
    const size_t bytes = threads * 8 * sizeof(uint32_t);

    CUdeviceptr d_in, d_out;
    CHECK_CUDA(cuMemAlloc(&d_in, bytes));
    CHECK_CUDA(cuMemAlloc(&d_out, bytes));

    // Initialize input
    uint32_t* h_in = (uint32_t*)malloc(bytes);
    for (int i = 0; i < threads * 8; i++)
        h_in[i] = 0xAAAAAAAA;
    CHECK_CUDA(cuMemcpyHtoD(d_in, h_in, bytes));

    // Warmup
    uint32_t threads_u32 = threads;
    void* args[] = {{(void*)&d_in, (void*)&d_out, (void*)&threads_u32}};
    for (int i = 0; i < 10; i++) {{
        CHECK_CUDA(cuLaunchKernel(kernel, (threads + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args, 0));
    }}
    CHECK_CUDA(cuCtxSynchronize());

    // Benchmark
    uint64_t start = get_time_ns();
    for (int i = 0; i < iterations; i++) {{
        CHECK_CUDA(cuLaunchKernel(kernel, (threads + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args, 0));
    }}
    CHECK_CUDA(cuCtxSynchronize());
    uint64_t end = get_time_ns();

    double elapsed_s = (end - start) / 1e9;
    uint64_t total_evals = (uint64_t)threads * 32 * iterations;
    double ns_per_eval = (end - start) / (double)total_evals;
    double evals_per_sec = total_evals / elapsed_s;

    printf("Tower Field S-box (CUDA PTX):\\n");
    printf("  Gates: 132 -> 98 (ternary synthesis)\\n");
    printf("  lop3.b32 instructions: 34\\n");
    printf("  Threads: %d\\n", threads);
    printf("  Iterations: %d\\n", iterations);
    printf("  Total evaluations: %lu (threads * 32 * iterations)\\n", total_evals);
    printf("  Total time: %.3f ms\\n", elapsed_s * 1000);
    printf("  Time per eval: %.3f ns\\n", ns_per_eval);
    printf("  Throughput: %.2f B evals/sec\\n", evals_per_sec / 1e9);

    cuMemFree(d_in);
    cuMemFree(d_out);
    free(h_in);
    cuCtxDestroy(context);

    return 0;
}}
"""


def main():
    print("=" * 60)
    print("Tower Field S-box - CUDA PTX Benchmark")
    print("=" * 60)
    print()

    # Read PTX
    ptx_path = "out/tower_sbox_ptx/circuit.ptx"
    if not os.path.exists(ptx_path):
        print(f"ERROR: {ptx_path} not found. Run compile_tower_ptx.py first.")
        return 1

    print("Reading PTX code...")
    with open(ptx_path, "r") as f:
        ptx_func = f.read()

    lop3_count = ptx_func.count("lop3.b32")
    print(f"  lop3.b32 instructions: {lop3_count}")
    print()

    # Check for CUDA
    try:
        subprocess.run(["nvcc", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: nvcc not found. CUDA toolkit required.")
        return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Wrap PTX as kernel
        print("Generating CUDA kernel...")
        full_ptx = wrap_ptx_as_kernel(ptx_func, "circuit", "sm_75")
        ptx_file = tmpdir / "circuit.ptx"
        ptx_file.write_text(full_ptx)

        # Assemble to cubin
        print("Assembling PTX to cubin...")
        cubin_file = tmpdir / "circuit.cubin"
        result = subprocess.run(
            ["ptxas", "-arch=sm_75", "-o", str(cubin_file), str(ptx_file)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("PTX assembly failed:")
            print(result.stderr)
            return 1

        # Generate CUDA driver
        print("Generating CUDA driver...")
        driver_code = generate_cuda_driver(str(cubin_file), iterations=100)
        driver_file = tmpdir / "driver.cu"
        driver_file.write_text(driver_code)

        # Compile driver
        print("Compiling CUDA driver...")
        driver_bin = tmpdir / "bench"
        result = subprocess.run(
            ["nvcc", "-O3", "-lcuda", "-o", str(driver_bin), str(driver_file)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("CUDA compilation failed:")
            print(result.stderr)
            return 1

        # Run benchmark
        print("Running benchmark...")
        print()
        result = subprocess.run([str(driver_bin)], capture_output=True, text=True)

        if result.returncode != 0:
            print("Benchmark failed:")
            print(result.stderr)
            return 1

        print(result.stdout)

    return 0


if __name__ == "__main__":
    sys.exit(main())
