from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from tempfile import TemporaryDirectory

from stc.backend_sched import generate_scheduled_code
from stc.circuit_synth import CircuitState


@dataclass(frozen=True)
class PtxKernel:
    ptx: str
    kernel_name: str
    lop3_count: int


def _wrap_func_as_kernel(ptx_func: str, func_name: str, sm: str) -> PtxKernel:
    header = "\n".join(
        [
            ".version 7.0",
            f".target {sm}",
            ".address_size 64",
            "",
        ]
    )

    kernel_name = f"{func_name}_kernel"
    kernel = f"""
.visible .entry {kernel_name}(
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

    lop3_count = len(re.findall(r"\blop3\.b32\b", ptx_func))
    return PtxKernel(
        ptx=header + ptx_func + "\n" + kernel,
        kernel_name=kernel_name,
        lop3_count=lop3_count,
    )


def emit_lop3_kernel(circuit: CircuitState, *, sm: str, func_name: str) -> PtxKernel:
    ptx_func = generate_scheduled_code(
        circuit, target="ptx_legacy", function_name=func_name
    )
    return _wrap_func_as_kernel(ptx_func, func_name, sm)


def emit_inline_lop3_kernel(
    circuit: CircuitState, *, sm: str, kernel_name: str = "sbox_lop3_inline_kernel"
) -> PtxKernel:
    """Emit a PTX `.entry` kernel that inlines the circuit as register ops.

    This avoids global loads/stores inside the circuit and is meant for
    benchmarking the boolean network + `lop3.b32` throughput, not end-to-end IO.

    Kernel signature:
      (out_ptr: u64, n_threads: u32, iters: u32)

    Each thread computes `iters` iterations on 8x32-bit bit-planes and stores one
    32-bit checksum to out_ptr[tid].
    """
    if circuit.input_bits != 8 or circuit.output_bits != 8:
        raise ValueError(
            "inline lop3 kernel currently supports 8->8 S-box circuits only"
        )

    header = "\n".join(
        [
            ".version 7.0",
            f".target {sm}",
            ".address_size 64",
            "",
        ]
    )

    gates = list(circuit.gates)
    outputs = list(circuit.outputs)

    # Map node index -> PTX register name.
    # Inputs 0..7 -> %r0..%r7
    # Gates -> %r8..%r{8+len(gates)-1}
    def r(node: int) -> str:
        if node < 8:
            return f"%r{node}"
        return f"%r{node}"

    # We'll use extra regs after nodes for control/addr.
    first_tmp = 8 + len(gates) + 8
    r_tid = f"%r{first_tmp}"
    r_n = f"%r{first_tmp+1}"
    r_iters = f"%r{first_tmp+2}"
    r_blk = f"%r{first_tmp+3}"
    r_cta = f"%r{first_tmp+4}"
    r_bid = f"%r{first_tmp+5}"
    r_off = f"%r{first_tmp+6}"
    r_acc = f"%r{first_tmp+7}"
    r_i = f"%r{first_tmp+8}"
    rd_out = "%rd0"
    rd_addr = "%rd1"
    rd_off = "%rd2"

    num_r = first_tmp + 16

    lines: list[str] = []
    lines.append(header)
    lines.append(f".visible .entry {kernel_name}(")
    lines.append("    .param .u64 out_ptr,")
    lines.append("    .param .u32 n_threads,")
    lines.append("    .param .u32 iters")
    lines.append(") {")
    lines.append(f"    .reg .b32 %r<{num_r}>;")
    lines.append("    .reg .b64 %rd<3>;")
    lines.append("    .reg .pred %p<1>;")
    lines.append("")
    lines.append(f"    ld.param.u64 {rd_out}, [out_ptr];")
    lines.append(f"    ld.param.u32 {r_n}, [n_threads];")
    lines.append(f"    ld.param.u32 {r_iters}, [iters];")
    lines.append("")
    lines.append(f"    mov.u32 {r_bid}, %tid.x;")
    lines.append(f"    mov.u32 {r_blk}, %ntid.x;")
    lines.append(f"    mov.u32 {r_cta}, %ctaid.x;")
    lines.append(f"    mad.lo.u32 {r_tid}, {r_cta}, {r_blk}, {r_bid};")
    lines.append(f"    setp.ge.u32 %p0, {r_tid}, {r_n};")
    lines.append("    @%p0 bra DONE;")
    lines.append("")

    # Seed inputs from tid. Use a cheap LCG/xorshift-ish mix.
    # Each input plane is a 32-bit word, representing 32 parallel evals.
    lines.append(f"    mov.u32 {r_acc}, 0;")
    for i in range(8):
        lines.append(f"    mov.u32 %r{i}, {r_tid};")
        lines.append(f"    mad.lo.u32 %r{i}, %r{i}, 1664525, {12345 + i * 101};")
        lines.append(
            f"    xor.b32 %r{i}, %r{i}, 0x{(0x9E3779B9 ^ (i * 0x11111111)) & 0xFFFFFFFF:08x};"
        )

    lines.append("")
    lines.append(f"    mov.u32 {r_i}, 0;")
    lines.append("LOOP:")
    lines.append(f"    setp.ge.u32 %p0, {r_i}, {r_iters};")
    lines.append("    @%p0 bra LOOP_DONE;")

    lop3_count = 0
    # Emit gates sequentially; write each into its node register.
    for g_idx, gate in enumerate(gates):
        dst = f"%r{8 + g_idx}"
        if len(gate) == 5:
            _, a, b, c, imm8 = gate
            lines.append(f"    lop3.b32 {dst}, {r(a)}, {r(b)}, {r(c)}, {imm8};")
            lop3_count += 1
            continue
        op, a, b = gate
        if op == "xor":
            lines.append(f"    xor.b32 {dst}, {r(a)}, {r(b)};")
        elif op == "and":
            lines.append(f"    and.b32 {dst}, {r(a)}, {r(b)};")
        elif op == "or":
            lines.append(f"    or.b32 {dst}, {r(a)}, {r(b)};")
        elif op == "not":
            lines.append(f"    not.b32 {dst}, {r(a)};")
        elif op == "const":
            lines.append(f"    mov.b32 {dst}, {0xFFFFFFFF if a else 0};")
        else:
            lines.append(f"    mov.b32 {dst}, 0;")

    # Fold outputs into accumulator and feed back into inputs to keep dependency chain.
    # Use out bitplanes (8 words).
    for out_i, (node, inv) in enumerate(outputs):
        src = r(node)
        if inv:
            tmp = f"%r{first_tmp+9+out_i}"
            lines.append(f"    not.b32 {tmp}, {src};")
            src = tmp
        lines.append(f"    xor.b32 {r_acc}, {r_acc}, {src};")
        # Feed some outputs back to inputs to keep the loop stateful.
        if out_i < 8:
            lines.append(f"    xor.b32 %r{out_i}, %r{out_i}, {src};")

    lines.append(f"    add.u32 {r_i}, {r_i}, 1;")
    lines.append("    bra LOOP;")
    lines.append("LOOP_DONE:")
    lines.append("")

    # Store checksum: out_ptr[tid] = acc
    lines.append(f"    cvt.u64.u32 {rd_off}, {r_tid};")
    lines.append(f"    mul.wide.u32 {rd_off}, {r_tid}, 4;")
    lines.append(f"    add.u64 {rd_addr}, {rd_out}, {rd_off};")
    lines.append(f"    st.global.u32 [{rd_addr}], {r_acc};")
    lines.append("")
    lines.append("DONE:")
    lines.append("    ret;")
    lines.append("}")
    lines.append("")

    return PtxKernel(
        ptx="\n".join(lines), kernel_name=kernel_name, lop3_count=lop3_count
    )


def assemble_ptx_to_cubin(ptx: str, *, sm: str) -> bytes:
    # `ptxas` doesn't reliably support writing cubin bytes to stdout across versions.
    # Use a temp file for robustness.
    with TemporaryDirectory() as td:
        ptx_path = os.path.join(td, "in.ptx")
        cubin_path = os.path.join(td, "out.cubin")
        with open(ptx_path, "w", encoding="utf-8") as f:
            f.write(ptx)

        res = subprocess.run(
            ["ptxas", f"-arch={sm}", ptx_path, "-o", cubin_path],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            raise RuntimeError((res.stderr or res.stdout).strip()[:4000])
        with open(cubin_path, "rb") as f:
            return f.read()


def ptxas_available() -> bool:
    return bool(shutil.which("ptxas"))
