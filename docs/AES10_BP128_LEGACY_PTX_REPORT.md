# AES10 BP128 Legacy PTX Line-Walk Report

Artifact reviewed:
- `out/aes10_bp128_legacy_linewalk.kernel.ptx`
- PTX: `.version 8.4`, target `sm_61`
- Size/shape: 6,442 lines, ~225 KiB

## 1) Kernel structure

The PTX has a single entrypoint:
- `.entry aes10_bp128_kernel` (`line 16`)

Control flow is minimal:
- One bounds guard branch to return (`line 32` -> `line 6438`)
- One counted loop branch for rounds (`line 3498` -> `line 229`)
- One final `ret` (`line 6439`)

There are no function calls in PTX (`call` count = 0).

## 2) Resource behavior

From `ptxas -v` on this artifact:
- Registers: `255`
- Stack frame: `0`
- Spills: `0` store / `0` load
- Const memory: `cmem[3]=5632 bytes`, `cmem[0]=332 bytes`

PTX declaration still shows virtual registers `.reg .b32 %r<6142>` (`line 22`), but physical allocation is 255 registers/thread.

## 3) Dataflow and memory model

Constant key/bitplane table:
- `RK_BITS[5632]` is embedded as `.const` (`line 14`)
- Layout matches `11 round keys * 16 bytes * 8 bitplanes * 4 bytes`

Memory traffic in kernel:
- `ld.const.u32`: 384
- `st.global.u32`: 128
- `ld.global.u32`: 0
- `ld.local/st.local`: 0 / 0

Interpretation:
- No global input loads (plaintext is hardcoded in kernel source path)
- State lives entirely in registers
- Only output writeback touches global memory (128 x `u32` bitplanes per thread)

## 4) Round mechanics in PTX

Initialization:
- Thread id math and bounds check (`lines 26-32`)
- Initial round-key preloads/bit inversions into state rails (`lines 34-228`)

Main AES rounds (`1..9`):
- Loop counter in `%r6141` (`lines 226`, `3496-3498`)
- Body starts at `$L__BB0_2` (`line 229`)
- Repeated S-box motifs use `lop3.b32` from inline asm (`e.g., lines 246, 250, 259, ...`)
- Mid-loop rekeying uses round-stride const pointer math (`lines 3238-3240`) and bulk `ld.const` + `xor` rail updates (`lines 3240+`)

Final round:
- Separate unrolled block starts after loop (`line 3500`)
- Final output address and store sequence (`lines 6306-6436`)

## 5) Instruction mix (whole kernel)

- `xor.b32`: 2,544
- `lop3.b32`: 864
- `and.b32`: 544
- `not.b32`: 192
- `ld.const.u32`: 384
- `st.global.u32`: 128
- Branch ops: 2

Conclusion: this legacy kernel is compute-dense, register-resident, and intentionally unrolled in PTX/SASS style; its performance depends mainly on register-limited occupancy and launch geometry, not memory bandwidth.
