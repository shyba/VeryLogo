"""Hand-scheduled AVX-512 GEMM micro-kernels, generated from the Agner machine model.

The micro-kernel is emitted as GAS (AT&T syntax) with a *fixed register
assignment and instruction order*, so no compiler pass can reorder, spill,
or otherwise reshape the intended sequence (gcc demonstrably does all three
to the equivalent intrinsics at >= 26 live zmm; see bench_gemm_avx512.py).

The generated structure is derived from stc/aggen (Agner Zen 5 table +
local measurements):

- The FMA-class op (VPDPBUSD / VDPBF16PS) sustains 2/cyc on P01, so the
  tile must expose >= 2 * latency independent accumulator chains; the
  kernel allocates MR*(NR/16) accumulator registers.
- A-operand broadcasts and B-vector loads run on the 2 load pipes in
  parallel with P01; the loop interleaves one broadcast + its two MACs per
  row so load latency hides under the MAC stream.
- Register budget (32 zmm): MR*(NR/16) accumulators + NR/16 B vectors +
  1 broadcast temp + 2 spare <= 30.
- Loop-carried chains: every accumulator is updated once per K-chunk, so
  the per-chunk issue rate is 2 MACs per accumulator-vector per P01 pipe.

Usage:
  python -m stc.gemm_asm --family vnni8 --tile 8x32   # print GAS to stdout
  python -m stc.gemm_asm --family bf16 --tile 8x32 -o gemm.S
"""

from __future__ import annotations

import argparse
import sys

from stc.aggen import get_machine

# K elements consumed per 32-bit lane per instruction (i.e. per K-chunk).
K_PER_CHUNK = {"vnni8": 4, "bf16": 2}
# MACs per 512-bit instruction (16 lanes * k_per_chunk for int, 2*16 for bf16).
_INSTR_MAC = {"vnni8": "vpdpbusd", "bf16": "vdpbf16ps"}
_FP_ACC = {"vnni8": False, "bf16": True}


def _acc_reg(i: int, j: int, mr: int, nb: int) -> str:
    """Register for accumulator (row i, col-block j): zmm0..7 then zmm16..23."""
    idx = i * nb + j
    if idx < 8:
        return "zmm%d" % idx
    return "zmm%d" % (16 + (idx - 8))


def emit_gemm_kernel(
    family: str,
    mr: int,
    nr: int,
    name: str | None = None,
    unroll: int = 2,
) -> str:
    """Emit a hand-scheduled GAS micro-kernel for an MR x NR tile.

    C signature:
      void NAME(const uint8_t* A, const int32_t* Bp, <int32_t|float>* C,
                int K, int N)
    A:   MR rows x K (K multiple of K_PER_CHUNK)
    Bp:  K/K_PER_CHUNK chunks x NR dwords; dword j of chunk c holds the
         K-chunk's bytes/pair for output column j (see pack_b_* in
         scripts/bench_gemm_avx512.py)
    C:   MR x N (tile rows at stride N)

    unroll: K-loop unroll factor (amortizes the pointer-advance and loop
    control over U chunks; the intrinsic version needs the same trick).
    """
    m = get_machine()
    spec = m.spec(family)
    kpc = K_PER_CHUNK[family]
    nb = nr // 16
    nacc = mr * nb
    if nacc + nb + 2 > 30:
        raise ValueError(
            f"{family} tile {mr}x{nr}: {nacc} accs + {nb} B-vecs + temp "
            f"exceeds the 32-zmm budget"
        )
    if mr > 8:
        raise ValueError("MR > 8 not supported (8 GPR row pointers)")

    name = name or f"gemm_{family}_{mr}x{nr}_asm"
    instr = _INSTR_MAC[family]
    fp_acc = _FP_ACC[family]
    acc_type = "float" if fp_acc else "int32_t"
    chk = "K/%d" % kpc
    b_adv = nr * 4  # bytes per chunk (NR dwords)
    # A operand addressing: bf16 elements are 2 bytes, int8 are 1 byte.
    a_bytes = 2 if family == "bf16" else 1
    a_adv = kpc * a_bytes  # row-pointer advance per chunk, in bytes
    end_scale = a_bytes  # end = A + K*a_bytes (lea scale)

    # accumulator zeroing
    zero = []
    for i in range(mr):
        for j in range(nb):
            r = _acc_reg(i, j, mr, nb)
            zero.append(f"    vpxord %{r}, %{r}, %{r}")

    # row pointers: r9..r15, rbx (all but r9-11 are callee-saved: push them)
    saved = ["%rbx", "%r12", "%r13", "%r14", "%r15"]
    row_regs = ["%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%rbx"]
    ptr = []
    ptr.append("    mov %rdi, %r9")
    prev = "%r9"
    for i in range(1, mr):
        ptr.append(f"    lea ({prev},%rcx,{end_scale}), {row_regs[i]}")
        prev = row_regs[i]

    # loop body, unrolled: per sub-chunk u, b loads at +u*b_adv, row
    # broadcasts at +u*a_adv; temps renamed per sub-chunk to avoid aliasing.
    bodies = []
    for u in range(unroll):
        body = []
        for j in range(nb):
            body.append(f"    vmovdqu64 {j*64 + u*b_adv}(%rsi), %zmm{24 + j + u*nb}")
        off = "" if u == 0 else "+%d" % (u * a_adv)
        for i in range(mr):
            body.append(f"    vpbroadcastd {off}({row_regs[i]}), %zmm{28 + u}")
            for j in range(nb):
                dst = _acc_reg(i, j, mr, nb)
                body.append(f"    {instr} %zmm{24 + j + u*nb}, %zmm{28 + u}, %{dst}")
        bodies.extend(body)
        if u + 1 < unroll:
            bodies.append("")

    # pointer advance (once per U chunks)
    adv = []
    for i in range(mr):
        adv.append(f"    add ${a_adv * unroll}, {row_regs[i]}")
    adv.append(f"    add ${b_adv * unroll}, %rsi")

    # stores: row i at C + i*N, col-block j at +j*16 dwords
    store = []
    for i in range(mr):
        if i > 0:
            store.append("    lea (%rdx,%r8,4), %rdx")
        for j in range(nb):
            dst = _acc_reg(i, j, mr, nb)
            if fp_acc:
                store.append(f"    vmovups %{dst}, {j*64}(%rdx)")
            else:
                store.append(f"    vmovdqu32 %{dst}, {j*64}(%rdx)")

    push = "\n".join(f"    push {r}" for r in saved)
    pop = "\n".join(f"    pop {r}" for r in reversed(saved))
    lines = [
        f"# {name}: {family} {mr}x{nr} GEMM micro-kernel, hand-scheduled, generated",
        f"# by stc.gemm_asm from the Agner Zen 5 machine model "
        f"(lat {spec.latency}, rt {spec.rt}, pipes {spec.pipes}).",
        f"# void {name}(const uint8_t* A, const int32_t* Bp, {acc_type}* C, int K, int N)",
        "    .text",
        f"    .globl {name}",
        f"    .type {name}, @function",
        f"{name}:",
        push,
        "",
        *zero,
        "",
        f"    lea (%rdi,%rcx,{end_scale}), %rax",
        *ptr,
        "",
        ".Lloop_%s:" % name,
        *bodies,
        "",
        *adv,
        "    cmp %rax, %r9",
        "    jne .Lloop_%s" % name,
        "",
        *store,
        pop,
        "    ret",
        f"    .size {name}, .-{name}",
        '    .section .note.GNU-stack,"",@progbits',
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=sorted(K_PER_CHUNK), default="vnni8")
    ap.add_argument("--tile", default="8x32", help="MRxNR")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()
    mr, nr = (int(x) for x in args.tile.split("x"))
    src = emit_gemm_kernel(args.family, mr, nr)
    if args.output:
        with open(args.output, "w") as f:
            f.write(src)
    else:
        sys.stdout.write(src)


if __name__ == "__main__":
    main()
