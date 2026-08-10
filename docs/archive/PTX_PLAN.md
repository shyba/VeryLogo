# NVIDIA PTX Backend Plan (sm_61 / GTX 1070)

Target GPU:

- NVIDIA Pascal, GeForce GTX 1070
- Compute capability: `sm_61`
- Warp size: 32
- No Tensor Cores (`mma.*` out of scope)

Backend goal:

- Compile Tick-IR modules into a CUDA kernel that evaluates exactly one tick per invocation for many independent instances in parallel (one thread = one instance).
- Preserve Tick-IR determinism: no cross-thread communication, no atomics, no shared state.

Execution model:

- Inputs `I`, state `S` and outputs `O`, next-state `S'` are stored in global memory as packed little-endian bitvectors split into 32-bit words.
- Kernel signature conceptually:
  - `in_words`: packed `I` for `N` instances
  - `state_in_words`: packed `S` for `N` instances
  - `out_words`: packed `O` for `N` instances
  - `state_out_words`: packed `S'` for `N` instances
  - `n_instances`: number of instances
- Each thread `tid` loads its own `I[tid]` and `S[tid]`, computes `O[tid]` and `S'[tid]`, and stores results.

Initial scope (MVP for PTX):

- Types: `bool`, `bitvec[N]` for `N <= 32` first; then `N <= 64`; then arbitrary `N` as multiword.
- Combinational expressions only (no Tick-IR state) first; then add state load/store.
- No SIMD types in the PTX backend initially (SIMT is used for instance-parallelism, not lane-parallelism).

Current implementation status:

- Implemented: combinational evaluation for `bool` and `bitvec[N]` for arbitrary `N`, including state load/store when Tick-IR has state.
- Implemented: PTX assembly checks via `ptxas` for `sm_61`.
- Implemented when available: runtime execution via the CUDA driver API on an `sm_61` device (`tests/test_backend_ptx_run_sm61_optional.py`).

Runtime notes:

- If the host has a visible GTX 1070 (or other `sm_61`) and `nvcc` on `PATH`, the run tests execute and validate results against the Python interpreter.
- If Docker is available but `--gpus` is not configured, `scripts/run_ptx_sm61_docker.sh` runs the same GPU tests by passing through `/dev/nvidia*` and bind-mounting the required driver libraries.

## Test Strategy (TDD)

All new tests are optional and skipped unless the toolchain/runtime is present.

- Codegen unit tests (always run): verify the emitted PTX contains the expected instruction mnemonics and uses only the allowed feature set.
- Compile tests (optional): use `nvcc`/`ptxas` to assemble PTX targeting `sm_61`.
- Run tests (optional): use the CUDA driver API via Python `ctypes` to JIT-load PTX and execute kernels on the GTX 1070, comparing outputs to the Python interpreter bit patterns.

Runtime gating recommendation:

- Require `nvcc` and `libcuda.so` available.
- Require a device with compute capability `(6,1)` (or allow `(6,0..6,2)` but pin tests to `sm_61` codegen).

## Instruction + Test Matrix (sm_61)

This table lists the PTX instruction set surface that the backend should use for `sm_61`, with a concrete test per row. “Test” describes the smallest kernel/program fragment that exercises the instruction and compares against the Tick-IR interpreter.

Conventions:

- `b1` values are held in `.pred` registers during evaluation; stored as `u8` in memory (0/1).
- `bitvec[N]` values are represented as little-endian `u32` word arrays for all `N` (with per-word masking for non-multiple-of-32 widths).
- For masked widths (e.g. `bitvec[5]`), results are masked via `and.b32` with a compile-time mask constant.

| Area | PTX instruction(s) (typical form) | Used for Tick-IR | Semantics notes | Test (minimal) |
|---|---|---|---|---|
| Module | `.version`, `.target sm_61`, `.address_size 64` | Backend header | Pin PTX version compatible with `sm_61` | Codegen unit test: header contains target + address size |
| Kernel | `.entry`, `.param`, `ld.param` | Kernel ABI | Parameters passed via `.param` space | Compile test assembles a no-op kernel |
| Thread id | `mov.u32 %r, %tid.x`, `mov.u32 %r, %ctaid.x`, `mov.u32 %r, %ntid.x`, `mad.lo.u32` | Instance indexing | `idx = ctaid.x * ntid.x + tid.x` | Run test with `N=64`, verify mapping to instance slots |
| Bounds | `setp.ge.u32`, `@%p bra` | `if (idx >= n)` guard | Avoid OOB global memory access | Run test with `N` not multiple of block size |
| Global load | `ld.global.u32` | Load input/state words | `bitvec[N]` packed as `u32` words | Run test: load constant pattern and pass-through |
| Global store | `st.global.u32` | Store outputs/next-state words | Coalesced word stores | Run test: store computed word equals interpreter |
| Move | `mov.b32`, `mov.b64`, `cvt` | Copy/casts | Pure register moves | Covered implicitly by most kernels |
| Bool not | `not.pred` | `Not` on `bool` | Predicate logical NOT | Run test: `o = !a` for `a ∈ {0,1}` |
| Bool and/or/xor | `and.pred`, `or.pred`, `xor.pred` | `And/Or/Xor` on `bool` | Predicate ops | Run test: full truth table for `(a,b)` |
| Bitwise not | `not.b32` / `not.b64` | `Not` on `bitvec[N]` | Mask after op when `N` not power-of-2 | Run test: `~x` for widths `{1,5,8,31,32}` |
| Bitwise and/or/xor | `and.b32`, `or.b32`, `xor.b32` | `And/Or/Xor` on `bitvec[N]` | Mask after op | Run test: random vectors + edge masks |
| Add/sub | `add.u32`, `sub.u32` | `Add/Sub` on `bitvec[N]` | Modular wrap; mask after op for `N<32` | Run test: random + overflow cases |
| Shifts (var) | `shl.b32`, `shr.u32`, `shr.s32` | `Shl/LShr/AShr` | Shift amount is masked by PTX rules; define explicit masking for Tick-IR | Run test: `sh` across `{0..40}` for `N=32` |
| Compare eq | `setp.eq.u32` / `setp.eq.b32` | `Eq` | Produces `.pred` | Run test: `eq` on random + equal pairs |
| Compare unsigned | `setp.lt.u32`, `setp.le.u32`, `setp.gt.u32`, `setp.ge.u32` | `Ult/Ule/Ugt/Uge` | Unsigned comparisons | Run test: random + boundary values |
| Select (mux) | `selp.b32` / `selp.b64` | `Mux` | Branchless conditional select | Run test: `sel ? a : b` |
| Branch | `bra`, predicated `@%p bra` | Internal lowering (if needed) | Prefer `selp` for expression muxes | Compile test: ensure no unintended divergent CFG for pure muxes |
| 64-bit ops (optional) | `add.u64`, `sub.u64`, `and.b64`, `or.b64`, `xor.b64`, `shl.b64`, `shr.u64`, `setp.*.u64`, `selp.b64` | Future optimization | Can be used as a fast path for `N <= 64` | Compile test for a 64-bit kernel |
| Mul (lo) | `mul.lo.u32` | Future: `Mul` if added to Verilog subset | `sm_61` supports integer mul | Run test: random multiply, compare masked result |
| Wide mul | `mul.wide.u32` | Multiword arithmetic building block | Produces `u64` product | Run test: compare against Python product |
| Mul hi | `mul.hi.u32` / `mul.hi.s32` | SIMD-like ops / future IR ops | High-half extraction | Run test: handpicked values to cover carry |
| Add with carry | `add.cc.u32`, `addc.u32` | Multiword add | Needed for `bitvec[N>32]` | Run test: 96-bit add via three words |
| Sub with borrow | `sub.cc.u32`, `subc.u32` | Multiword sub | Needed for `bitvec[N>32]` | Run test: 96-bit sub with borrow propagation |
| Predicated move | `@%p mov.b32` | Lowering helper | Used when `selp` not available for a type | Covered by mux tests |
| Constants | `mov.u32` with literal, `.const` (optional) | Masks, widths | Keep masks immediate when possible | Codegen unit test: mask constants emitted |

## Phased Implementation Plan

Phase 1 — Tooling + Harness

- Add `stc/backend_ptx.py` that emits syntactically valid PTX for a single kernel stub.
- Add optional assembler test that compiles PTX to cubin for `sm_61`.
- Add optional driver-runtime test harness in Python `ctypes`:
  - load `libcuda.so`
  - initialize context
  - load PTX module
  - launch kernel
  - copy outputs back

Phase 2 — Minimal combinational bitvec (<= 32)

- Support Tick-IR nodes used by current Verilog subset:
  - `Not/And/Or/Xor/Add/Sub/Shl/LShr/AShr/Eq/Ult/Ule/Ugt/Uge/Mux`
- Restrict `bitvec[N]` to `N <= 32`.
- Add run tests per instruction row in the matrix.

Phase 3 — Add state (sequential tick)

- Load state words, compute `S'`, store `S'`.
- Provide a stable memory layout for state variables.
- Add run tests for small FSM and counter fixtures (instance-parallel).

Phase 4 — Optional `u64` fast path (<= 64)

- Add an optional 64-bit path using `u64` instructions and masking for `N <= 64`.
- Keep the default path as multiword `u32` for simplicity.

Phase 5 — Arbitrary widths (multiword `u32[]`)

- Implement multiword add/sub/bitwise/shifts using `add.cc/addc`, `sub.cc/subc`, `shf`-style sequences where available, or explicit word loops unrolled at compile time.
- Add run tests for representative sizes `{96, 128, 192, 256}`.

Phase 6 — Performance passes (optional)

- Coalesce loads/stores by grouping words and using vectorized patterns where profitable.
- Prefer `selp` to avoid divergence for `Mux`.
- Add a microbenchmark test mode (optional) that measures effective ticks/s for large `N`.
