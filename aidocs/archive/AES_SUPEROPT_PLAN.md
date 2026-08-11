# AES Superoptimizer Plan (GPU Tick/PTX)

## Goal

Make the GPU Tick/PTX path competitive on AES-128 by removing per-tick launch overhead and enabling Z3-guided rewrites that match the structure of AES (byte-lane parallelism + lookup-heavy S-box).

## Baseline (Current)

- Verilog: `fixtures/verilog/aes128_fixedkey_seq_lut.v`
- Execution model: 11 ticks = 11 kernel launches + global state ping-pong.
- Bottlenecks:
  - Kernel-launch overhead dominates at small/medium `n`.
  - SubBytes is implemented as 16 `Lut8` constant-memory loads per round.
  - Control (`round`) is lowered into mux-heavy logic.

## Success Criteria

1. Functional correctness:
   - AES-128 known vector passes after all rewrites.
   - Random plaintext spot-checks pass (host interpreter + Z3 proof where feasible).
2. Performance:
   - Eliminate multi-launch overhead by executing all rounds in a single kernel launch.
   - Demonstrate a material throughput increase vs current baseline for large `n`.

## Non-Goals (For This Phase)

- Full, general “compile any FSM into one kernel” with arbitrary control flow.
- Proving/optimizing across unbounded time (only fixed-step fusion for known tick counts).
- Matching RustCrypto AES-NI on CPU for all machines (GPU path is different hardware).

## Work Plan (TDD)

### Phase 1 — Tick Fusion as a Backend Feature (Required for Throughput)

**Idea:** keep Tick-IR unchanged, but generate PTX that executes `k` ticks per thread inside one kernel launch (register-resident state), then stores state/output once.

**Features (Implemented)**
- Add `stc.backend_ptx.emit_ptx_steps(ir, sm=..., steps=k)` that:
  - Loads inputs/state.
  - Executes the tick body `k` times with local temporaries.
  - Keeps inputs constant across steps.
  - Stores outputs and final state once.

**Tests (Implemented, optional CUDA)**
- `tests/test_backend_ptx_steps_run_sm61_optional.py`:
  - Build a small stateful IR (counter) and compare:
    - `k` single-step launches
    - 1 fused `steps=k` launch
  - Compare outputs and next-state buffers.

### Phase 2 — Control Specialization (Remove Round Muxes)

**Idea:** specialize the AES FSM for a fixed number of rounds so the per-round `Mux` network disappears.

**Features**
- Add a “control state peeling” pass that:
  - Detects a small control register that only feeds mux selects.
  - Partially evaluates the circuit for each step (0..k-1).
  - Produces either:
    - a fused-step IR (k-step transition), or
    - a per-step schedule for the PTX backend (preferred to avoid IR blow-up).

**Tests**
- `tests/test_control_specialize.py`:
  - Construct a small toy FSM with mux-based next-state selection and verify:
    - specialized schedule matches baseline for `k` steps (interpreter).
- `tests/test_aes_control_specialize_optional.py` (requires yosys):
  - Extract AES IR and confirm mux count decreases after specialization.

### Phase 3 — Superopt Support for Lookup-Centric Graphs

**Idea:** make the expression superoptimizer aware of AES-relevant operators so it can simplify round logic and byte-lane wiring.

**Features**
- Extend `superopt_expr` search space to include:
  - `Slice`, `Concat`, `Mux` (guarded: only when sizes are small).
  - `Lut8` (either as an allowed leaf, or as a rewrite target).
- Add rewrite templates (syntax-guided search) rather than full enumerative search for wide expressions:
  - Example: search only among equivalent reassociations for XOR trees.
  - Example: optimize `xtime()` and MixColumns linear transforms.

**Tests**
- `tests/test_superopt_lut8_templates.py`:
  - Prove a few `Lut8`-adjacent rewrite rules (e.g., constant-fold, mux hoisting).
- `tests/test_superopt_mixcolumns_linear.py`:
  - Encode MixColumns as a linear GF(2) transform and verify superopt finds a cheaper XOR network.

### Phase 4 — SIMD Lane-Lift for Byte-Parallel AES

**Idea:** represent the AES state as `simd[8,16]` so the optimizer and backend can treat it as 16 independent byte lanes.

**Features**
- Add `SimdLut8` (or `SimdMapLut8`) that applies the same 256×8 table to each lane.
- Teach:
  - interpreter
  - reducer
  - Z3 encoder
  - PTX backend
  about the new op.
- Update autovectorization/inference so `{slice/concat over 16 bytes}` can become `simd[8,16]`.

**Tests**
- `tests/test_simd_lut8.py`:
  - Interpreter vs Z3 equivalence for random vectors.
- `tests/test_autovec_aes_subbytes.py`:
  - A reduced SubBytes-only IR is autovectorized into `SimdMapLut8`.

### Phase 5 — S-box Lowering Options (When Const Loads Dominate)

**Idea:** provide multiple equivalent S-box implementations and let the optimizer choose by cost model:

- `Lut8` (const memory lookup)
- boolean circuit (bitsliced)
- (optional) T-tables (4×256×32 lookup)

**Features**
- Add a lowering pass `lower_lut8(mode=...)` with modes:
  - `const_lut` (status quo)
  - `boolean_network` (predefined circuit + Z3 proof once)
  - `t_tables` (optional, larger const footprint)
- Extend the cost model with a PTX-specific profile:
  - penalize `ld.const` count and register pressure
  - account for `lop3` opportunities

**Tests**
- `tests/test_aes_sbox_boolean_network.py`:
  - Prove boolean S-box matches table for all 256 inputs (Z3).
- `tests/test_cost_model_prefers_boolean_when_lut_penalized.py`:
  - Ensure selection is deterministic under configured costs.

## Benchmarks

Add a reproducible benchmark harness:

- GPU: `scripts/bench_aes_gpu_tick.py` gains `--steps` and `--mode` (`baseline`, `fused`, `fused+specialized`, etc.)
- CPU: `bench/aes_cpu` stays as-is.
- Print a single line summary per run so results can be pasted into docs.

## Expected Order of Wins

1. PTX `steps=11` fusion: removes 11 launches → largest win.
2. Control specialization: removes muxes + dead logic.
3. SIMD lane-lift: improves internal representation for future rewrites.
4. S-box strategy selection: hardware-dependent win; may help GPUs where const loads bottleneck.
