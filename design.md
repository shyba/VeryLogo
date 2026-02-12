# 1. MVP Objective

The MVP proves the technical hypothesis in `info.md`: a restricted HDL module is converted into an explicit sequential machine (Tick-IR), reduced via formal simplifications, and compiled into deterministic ATtiny85 firmware, while preserving tick-level functional equivalence.

The MVP does not include: multi-lane execution, GPU/TPU, aggressive performance tuning, a custom Verilog frontend, or advanced schedule visualization.


# 2. Compilation Pipeline (Overview)

Fixed flow:

input.v
 → normalized.json
 → tick_ir.bin
 → reduced_tick_ir.bin
 → avr.c

Role of each stage:

- `input.v`: HDL source of truth, within the MVP subset.
- `normalized.json`: Yosys-produced normalized representation (elaboration + lowering), stable for downstream tooling.
- `tick_ir.bin`: project IR with explicit tick semantics (`S' = f(S,I)`, `O = g(S,I)`), with unambiguous simultaneous state updates.
- `reduced_tick_ir.bin`: Tick-IR after combinational reduction and dead-state removal, preserving functional equivalence.
- `avr.c`: deterministic C code for ATtiny85 that executes one tick per loop iteration with simultaneous state commit.


# 3. Verilog Frontend (Yosys)

Yosys is responsible for parsing, elaboration, and normalization, and does not perform aggressive area/timing optimization for the final target.

Allowed passes in the MVP:

- Read and elaborate: `read_verilog -sv`, `hierarchy -check`.
- Structural lowering required to remove high-level processes: `proc`.
- Conservative local normalization/cleanup: `opt`, `opt_clean`.

Output format:

- `write_json` to produce `normalized.json`.

Accepted Verilog subset:

- Single top module.
- Single clock domain.
- State defined only by registers updated in `always_ff @(posedge clk)` with active-high synchronous reset in the form `if (rst) reg <= <constant>; else reg <= <expr>;`.
- Combinational logic defined by `assign` and/or `always_comb`.
- Types limited to bit vectors (bitvec) and booleans (1 bit).
- No `initial`, no delays (`#`), no `fork/join`, no tasks/functions, no `generate`, no multiple clocks, no complex memories, no tri-states.

Any construct outside the subset fails the frontend with an explicit error.


# 4. Tick-IR (Source of Truth)

Semantic model:

S' = f(S, I)
O  = g(S, I)

Where:

- `I` is the set of inputs sampled at the start of the tick.
- `S` is the state at the start of the tick.
- `S'` is the state to be committed at the end of the tick.
- `O` is the set of observable outputs, defined as a function of tick-local `S` and `I`.

Supported types:

- `bool`: 1 bit.
- `bitvec[N]`: `N`-bit vector, with `N >= 1`.
- `float[N]`: IEEE-754 binary floating-point, with `N ∈ {32,64}`, represented as a raw `N`-bit bit pattern in Tick-IR.
- `simd[lane_width, lanes]`: packed vector with `lane_width >= 1` and `lanes >= 1`, representing `lanes` independent lanes of `lane_width` bits each, with lane 0 stored in the least-significant `lane_width` bits.

State definition:

- State is any variable committed at tick end and read in later ticks (registers).
- Each state element has a stable name, fixed width, and a reset-defined initial value.
- If a register has no explicit reset in the input HDL, the MVP assigns a reset value of zero for that register in Tick-IR.

Combinational definition:

- Combinational expressions depend only on tick-local `S` and `I` and produce temporaries, outputs `O`, and candidate next-state values `S'`.

Delay/pipeline/register representation:

- A Tick-IR `reg` represents a state element committed simultaneously at tick end.
- `delay(x, k)` represents a value delayed by `k` ticks. In the MVP, `k` is a positive integer and is implemented by explicit register chains in Tick-IR.

Tick-IR guarantees:

- Determinism: for a fixed `S` and `I`, `S'` and `O` are unique.
- Simultaneous updates: all state is committed atomically at tick end; reads within a tick observe only the input `S`.
- No undefined behavior: operations are total over `bool`/`bitvec`, with modular wrap semantics for arithmetic.
- SIMD arithmetic is lane-wise with modular wrap within each lane and no cross-lane carry.
- SIMD ops have full-vector semantics over `simd[lane_width, lanes]`; backends preserve semantics even when the target ISA has lane-group boundaries (for example, some AVX2 shuffle instructions operate independently on 128-bit halves).

Float semantics:

- Float ops interpret `float[N]` bit patterns as IEEE-754 values using Z3 FP theory and a fixed rounding mode policy (`RNE`).
- Compare ops use ordered predicates (no unordered/NaN-equals behavior).
- NaN payloads are canonicalized for scalar and SIMD float ops (inputs and results); non-float bitvector ops preserve payload bits (for example, `SimdBlend` acts as a raw bit blend).


# 5. Scheduling and Time Track

Tick-IR is converted into a 1-lane time track in the MVP to produce a deterministic sequence of micro-operations within a tick.

MVP rules:

- The track is an ordered list of pure operations (micro-ops) that compute temporaries, `O`, and `S'`.
- The order is a topological sort of the data-dependency DAG derived from `f` and `g`.
- The track is 1-lane: one micro-op starts at a time.

Latency in the MVP:

- Each micro-op has unit latency on the track (one slot).
- `bitvec[N]` ops remain bitvector-semantic micro-ops; backend cost is estimated and recorded as a metric without changing Tick-IR semantics.

Not implemented in the MVP:

- Multi-lane scheduling.
- Initiation interval smaller than latency.
- Resource-typed scheduling (ALU vs MUL) and heterogeneous resource modeling.


# 6. Reduction and Solvers

The solver phase runs between `tick_ir.bin` and `reduced_tick_ir.bin`.

Reductions allowed in the MVP:

- Combinational reduction:
  - constant folding for `bool`/`bitvec`.
  - boolean simplification (identities, absorption, De Morgan, double-not elimination).
  - elimination of equivalent temporaries via structural hash-consing.
- Dead-state removal:
  - bounded reachability from reset.
  - removal of registers that are not observable and not influential on `O` within the validation bound.

Required preservation:

- Functional equivalence: for any input sequence within bound `N`, `O` and `S` of the reduced design match the original design under the same reset and tick clock.

Bounded validation:

- A bounded checker builds `N` steps of the transition relation for both designs and proves equivalence of `O` and `S` using SMT bitvectors, producing a concrete counterexample when the proof fails.


# 7. ATtiny85 Backend

Execution model:

- `tick` executes in the main loop. Timer-ISR execution is out of scope for the MVP.

State mapping (Tick-IR → C):

- Each `reg` maps to a `static` variable with minimal width (`uint8_t`, `uint16_t`, `uint32_t`), with explicit masking for non-multiple-of-8 widths.
- Each tick computes `next_<reg>` temporaries and commits all state simultaneously at tick end.

I/O mapping (GPIO):

- Inputs `I` are read from `PINB` (and other registers as needed) at tick start and normalized to `bool`/`bitvec`.
- Outputs `O` are written via a shadow `PORTB` value and committed in a fixed order at tick end.

Code generation rules:

- `bool`/`bitvec` operations are emitted as C bitwise/arithmetic ops with explicit masks.
- Branchless when possible: mux/select is emitted via masking (`(a & m) | (b & ~m)`) for `bool` and `bitvec`.
- No state reads observe partial updates; code has an explicit compute phase and an explicit commit phase.


# 8. MVP Validation

Functional simulation:

- A Python Tick-IR interpreter simulates ticks from `tick_ir.bin` and produces `S`/`O` traces.
- The original Verilog module is simulated with Verilator to produce reference traces under the same input sequences.

Golden model:

- The golden model is the tick-aligned simulation of the original Verilog module under the same reset and tick clock, with inputs sampled and outputs observed per cycle.

Bounded equivalence:

- Optional in the MVP: bounded equivalence checking via Yosys/SymbiYosys between the original Verilog and a Verilog form derived from Tick-IR, with explicit alignment for any `delay`.

Objective criteria for “MVP validated”:

- The MVP subset compiles to `avr.c` without manual intervention.
- The main test cases (combinational, simple FSM, temporal pipeline, debounce) pass for `N` ticks under pseudo-random and directed inputs.
- Reduction produces a measurable improvement in at least one structural metric (register count or operation count) without breaking bounded equivalence.
- The generated firmware executes deterministic ticks and matches expected traces in a test harness.


# 9. Artifacts and Debug

Generated, versionable artifacts:

- `normalized.json`: Yosys output.
- `tick_ir.bin`: extracted Tick-IR.
- `reduced_tick_ir.bin`: reduced Tick-IR.
- `avr.c`: C backend output.
- `compile.log`: pipeline log with hashes and counts.
- `metrics.bin`: minimal metrics (regs, state bits, ops per tick, estimated combinational depth, C size).

Required manual inspectability:

- Tick-IR is JSON and contains stable names for state, inputs, and outputs.
- The diff between `tick_ir.bin` and `reduced_tick_ir.bin` is human-readable (explicit removals and simplifications).

Minimal logs and metrics:

- Register and state-bit counts before/after.
- Combinational-node counts before/after.
- Validation bound `N` and result status (ok/counterexample).


# 10. Out of Scope (Explicitly)

- GPU / PTX
- Multi-lane
- Performance tuning
- Custom Verilog frontend
- UI / advanced visualization
