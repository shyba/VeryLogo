# TODO

## MVP (Completed)

- [x] Add `pyproject.toml` with Black configuration
- [x] Create `stc/` package skeleton
- [x] Create `tests/` using `unittest`
- [x] Ensure `python3 -m unittest discover -s tests` is green

## Tick-IR (Source of Truth)

- [x] Define Tick-IR types: `bool`, `bitvec[N]`
- [x] Define Tick-IR expressions (minimal operator set)
- [x] Define Tick-IR module container (`S`, `I`, `O`, `S'`)
- [x] Implement Tick-IR JSON schema and round-trip
- [x] Add fixtures for Tick-IR samples (MVP cases)

## Yosys JSON Ingestion

- [x] Implement `normalized.json` loader
- [x] Add fixtures for small Yosys JSON netlists
- [x] Validate module/port/cell integrity

## Subset Enforcement

- [x] Implement subset checker for the MVP frontend boundary
- [x] Add negative tests for rejected constructs

## Tick-IR Extraction

- [x] Map a minimal set of Yosys cells into Tick-IR
- [x] Extract state elements and next-state equations (minimal `$dff`)
- [x] Extract output equations
- [x] Emit `tick_ir.bin` (via CLI)
- [x] Support sync-reset FF extraction (`$sdff`)

## Reduction + Solvers (MVP)

- [x] Implement combinational simplification (constant folding + boolean identities)
- [x] Implement bounded dead-state removal (COI + bounded constant-state elimination for small designs)
- [x] Emit `reduced_tick_ir.bin` (via CLI)
- [x] Emit `metrics.bin` (before/after)

## ATtiny85 Backend

- [x] Generate C types and masks for `bool` and `bitvec[N]`
- [x] Generate compute/commit phases
- [x] Generate GPIO read/write (`PINB`/`PORTB`)
- [x] Emit `avr.c` deterministically

## Validation

- [x] Implement Tick-IR interpreter in Python
- [x] Generate tick traces for MVP cases (IR-level)
- [x] Integrate Verilator golden tracing
- [x] Integrate bounded equivalence checks when available (optional, requires `z3`)

## Tooling and DX

- [x] Add `.venv/` virtual environment and Black installation via `requirements-dev.txt`
- [x] Add `scripts/fmt.sh` formatter entrypoint
- [x] Add env var tool overrides (`STC_YOSYS`, `STC_VERILATOR`, etc.)

## Pending (Next)

### Bounded Equivalence (Hardening)

- [x] Generalize miter generation to arbitrary Tick-IR input/output ports (names + widths)
- [x] Make reset constraints non-vacuous (reset asserted at time 0, deasserted afterwards)
- [x] Add `dffunmap` before `write_smt2` for `$sdff` compatibility
- [x] Add bounded-equivalence optional tests for sequential cases (FSM toggle, counter)
- [x] Add bounded-equivalence optional tests for multi-bit ports

### Verilog Frontend (Coverage)

- [x] Expand supported Yosys cell set for the MVP subset boundary (shifts, subtract, compares)
- [x] Support `$dffe` (clock enable) for sequential extraction
- [x] Reject async resets explicitly (require synchronous reset cells)
- [x] Support multi-module inputs by explicit `--top` selection, and reject submodule instantiation explicitly
- [x] Add a deterministic Yosys script file output (`normalized.ys`) alongside `normalized.json`

### Tick-IR (Expressiveness)

- [x] Explicitly forbid signed Yosys cell params in the subset checker
- [x] Add a `delay(x, k)` representation and extraction lowering rules (k > 1)
- [x] Add explicit undefined/poison handling rules (or reject unknown initial state constructs)

### Reduction (Scalability)

- [x] Replace enumerative reachability with SMT-based bounded reachability for larger designs
- [x] Add equivalence-preserving rewrite proofs for reductions (per-pass, bounded)
- [x] Add a structural hashing pass to deduplicate identical subexpressions

### ATtiny85 Backend (Executable Firmware)

- [x] Emit full ATtiny85 project scaffold (`main.c`, `Makefile`, `avr-gcc` flags)
- [x] Add a host-side GPIO simulation shim to test generated C without AVR toolchain
- [x] Define a stable I/O mapping file format and validate it (pin directions + masks)
- [x] Add an end-to-end example: `.v` → `avr.c` → host simulation trace equals Tick-IR trace

### DX and CI

- [x] Add a single `scripts/test.sh` that runs format + unit tests
- [x] Add pre-commit hooks (Black + unit tests)

### SIMD + Superoptimization (Z3)

- [x] Define a stable SIMD cost model (ops, immediates, shuffle cost)
- [x] Add more SIMD ops: lane-wise subtract, shifts, compares (mask vectors)
- [x] Add lane ops: splat, extract, insert, shuffle/permutation
- [x] Add SMT-guided pruning (counterexample-driven, memoized equivalence classes)
- [x] Integrate superopt as an optional optimization pass for Tick-IR expressions

## SIMD Support (Detailed, TDD)

### 1) Semantics Spec (Tests First)

- [x] Add `tests/test_simd_semantics.py` covering lane packing (LSB lane 0) and wrap semantics
- [x] Add `tests/test_simd_masks.py` covering compare result layout as `SimdType(lane_width=1, lanes=N)`
- [x] Define (in `design.md`) the canonical lane layout and arithmetic semantics for all SIMD ops

### 2) SIMD Op Set (Add One Op At A Time, With Tests)

- [x] `SimdSub`: add `tests/test_simd_sub.py` then implement in Tick-IR + interpreter + Z3 + Verilog
- [x] `SimdNot/SimdAnd/SimdOr/SimdXor`: add `tests/test_simd_bitwise.py` then implement end-to-end
- [x] `SimdShl/SimdLShr/SimdAShr` (scalar shift amount first): add `tests/test_simd_shifts.py` then implement end-to-end
- [x] `SimdUlt/SimdUle/SimdUgt/SimdUge`: add `tests/test_simd_cmp_unsigned.py` then implement end-to-end
- [x] Signed compares only after an explicit signedness rule is defined: add `tests/test_simd_cmp_signed.py`

### 3) Lane Ops (Needed For Real Vectorization)

- [x] `Splat`: add `tests/test_simd_lane_ops.py` then implement
- [x] `ExtractLane`: add `tests/test_simd_lane_ops.py` then implement
- [x] `InsertLane`: add `tests/test_simd_lane_ops.py` then implement
- [x] `Shuffle`: add `tests/test_simd_shuffle.py` then implement (define permutation encoding)

### 4) Z3 Proof Harness (Fast, Deterministic)

- [x] Add `tests/test_z3_simd_identities.py` proving identities per op (UNSAT of `spec != cand`)
- [x] Add `tests/test_z3_simd_nonidentities.py` ensuring wrong rewrites are SAT (guard against vacuous proofs)
- [x] Add a shared proof helper with strict per-proof timeouts and clear failures

### 5) Autovectorization Pass (Prove Every Rewrite)

- [x] Add `tests/test_autovec_matrix.py` (table-driven) across `(lane_width, lanes, op)` positive + negative cases
- [x] Expand pattern coverage: commutativity, lane count > 2, lane_width != 4, nested `bitcast/concat/slice`
- [x] Add lane-wise subtraction autovectorization (`Sub` → `SimdSub`)
- [x] Add negative patterns: packed add (cross-lane carry), misaligned slices, mixed operands, partial lanes
- [x] Gate every rewrite behind a Z3 equivalence proof with a small timeout

### 6) Superoptimizer (CEGIS + Cost Model)

- [x] Add `tests/test_superopt_simd_canonical.py` ensuring SIMD forms are found from scalar-lane specs
- [x] Add a cost model and tests (`tests/test_superopt_cost_model.py`) so “vector wins” deterministically
- [x] Add CEGIS pruning + memoization with tests (`tests/test_superopt_pruning.py`) and timeouts

### 7) Exhaustive Test Matrix (Bounded Runtime)

- [x] Add `tests/test_simd_matrix.py` parameterized over `lane_width ∈ {1,2,4,8}` and `lanes ∈ {2,4,8}` for core ops
- [x] Add randomized spot-checks (fixed seeds) comparing interpreter vs Z3 evaluation (`tests/test_simd_interp_vs_z3.py`)
- [x] Add global caps so tests never hang (timeouts, max candidates), with a dedicated timeout test

## x86 SIMD Backend (SSE/AVX)

### CPU Feature Snapshot

- [x] Add `scripts/cpu_simd_features.py` to parse `/proc/cpuinfo` flags into Markdown
- [x] Generate `CPU_SIMD_FEATURES.md` for the current machine

### SSE2 Backend (Integer, 128-bit)

- [x] Add `stc/backend_x86_sse2.py` that emits C intrinsics (`<emmintrin.h>`)
- [x] Add optional compile+run tests that compare backend outputs to the interpreter
- [x] Add compare mask packing lowering via `_mm_movemask_epi8`
- [x] Add unsigned saturating u8/u16 ops (`SimdAddSatU`, `SimdSubSatU`)
- [x] Add signed saturating s8/s16 ops (`SimdAddSatS`, `SimdSubSatS`)
- [x] Add `SSE2_TESTCASES.md` mapping Tick-IR ops to intrinsics and tests

### Next Instruction Coverage (Blocked on Tick-IR Ops)

- [x] Multiply/madd (`pmullw/pmulhw/pmulhuw/pmaddwd`)
- [x] Pack/unpack and interleave (`pack*`, `punpck*`)
- [x] Canonical mask pack/unpack ops (`SimdMaskExpand`, `SimdMaskPack`)

### AVX / AVX2 / AVX-512 (Later)

- [x] Define a backend ABI for `SimdType.total_width` 256 and 512 (`X86_SIMD_ABI.md`)
- [x] Add AVX2 integer backend (`__m256i`) for wrap ops, shifts, compares (no pack/unpack/shuffle yet)
- [x] Add AVX-512 backend (`__m512i`) and masking support (initially epi32/epi64)
- [x] Add runtime dispatch based on detected CPU features (`stc/backend_x86_auto.py`)

## Pending / Blocked

### AES (GPU Tick/PTX) Superoptimization

- [x] Implement PTX `steps=k` fused execution (single launch for multi-tick designs)
- [x] Add correctness tests comparing `k` single-step launches vs `steps=k`
- [ ] Add control specialization for mux-heavy FSMs (general round peeling beyond reset specialization)
- [ ] Extend superopt search/templates to cover `Lut8` + XOR-linear transforms (MixColumns)
- [ ] Add SIMD lane-lift for SubBytes (`SimdMapLut8`) and wire it end-to-end
- [ ] Add S-box lowering options (const LUT vs boolean network) with Z3 proofs and a PTX cost profile
- [ ] Track this plan in `AES_SUPEROPT_PLAN.md`

### AVX-512 Predication (Requires Tick-IR Mask-on-Op)

- [x] Add first-class masked arithmetic in Tick-IR (mask as an input to SIMD ops, with merge semantics via passthrough).
- [x] Lower masked SIMD ops in the AVX-512 backend using AVX-512 opmasks (optional compile/run tests per op).

### AVX512VL (Optional)

- [x] Add an AVX-512 VL backend variant (128/256-bit vectors using AVX-512 encodings) when it becomes necessary for mask-based predication coverage on 128/256-bit vectors.

## Superopt (Z3) Next (Typed Search)

- [x] Add typed candidate enumeration (multi-type terms) so superopt can build SIMD masks as intermediates
- [x] Add `SimdEq/SimdUlt/...` as mask-producing candidate ops
- [x] Add `SimdBlend(mask, a, b)` as a ternary candidate op
- [x] Extend `expr_cost` with stable costs for compares and blends
- [x] Add always-on unit tests for mask and blend synthesis
- [x] Add optional Verilog end-to-end tests for select/min patterns
