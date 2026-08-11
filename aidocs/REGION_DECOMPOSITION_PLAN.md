# Generic Region Decomposition + Region Compilation (Plan)

This document proposes a **generic** “decompose → compile regions → fuse at emit-time” pipeline for VeryLogo. It is **not Keccak-specific**; it targets any large sequential Verilog design lowered to `TickIR`/`CircuitState`.

## Why decomposition helps (and when it doesn’t)

### What made S-box tower decomposition work
The big wins were not “smaller is easier”, but **structure exposure**:
- **Nonlinear isolation**: AND-like operations (or LUT/ternary ops) are concentrated; optimizing nonlinear count/depth becomes tractable.
- **Linear saturation**: after nonlinear boundaries are fixed, large linear (XOR) networks can be aggressively minimized and CSE’d.
- **Local search feasibility**: bounded SAT/SMT/superopt windows become practical only when cut sizes are small.

### Translation to generic sequential designs
Decomposition helps when it:
- Bounds the size of optimization, mapping, scheduling/RA problems.
- Reduces peak liveness and spills by running smaller “blocks” in sequence.
- Enables **emit-time fusion across regions**: run region A then B while carrying boundary values in registers or compact packed buffers, avoiding “materialize everything” patterns.

Decomposition **does not** help if region boundaries are huge and unstructured; it just moves state materialization to the boundary. In that case, first change representation (packing) or choose different boundaries.

## Current pipeline (repo-anchored)

Typical flow (high-level):
1) Verilog → `TickIR`: `stc/extract.py::extract_tick_ir()` (via `stc/yosys_frontend.py`, `stc/yosys_json.py`)
2) TickIR reduction: `stc/reduce.py::{optimize_tick_ir,reduce_tick_ir}`
3) Lower to `CircuitState` + IO layout (used by scheduled backends)
4) Scheduling/emit: `stc/backend_sched.py`, `stc/sched/*`, `stc/sched/emit/*`
5) Emit-time stepping + state pointer swap (already implemented for AVX512 scheduled backend; see `docs/EMIT_TIME_STEPS.md`)

**Insertion point** for regioning:
- Preferred: **after lowering to `CircuitState`**, before scheduling/emit.
  - Reason: region boundaries are naturally “wire-level”, and region compilation can reuse existing scheduler/emit logic on subgraphs.

## Design overview

### Core idea
Compile a single tick as a **DAG of regions**:
- Each region is a subgraph with a small packed interface.
- Region compilation produces a compact scheduled “region program”.
- The final emitter produces a wrapper that executes regions in order, passing boundaries **without bulk materialization** (register-carry if possible; otherwise spill to a packed scratch array).

This composes cleanly with emit-time stepping:
- A tick = run region DAG once to compute outputs + next_state.
- A multi-step loop = repeat the same region DAG while pointer-swapping state buffers (no per-step memcpy).

## Architectural decision (confirmed)

VeryLogo’s current `CircuitState` is fully bit-level: `input_bits` counts bits, gate operands are bit indices, and the scheduled AVX emitters operate on **one `__m512i` per bit**. Because of that, the “pack region boundaries into words and carry them in registers” assumption is not realistically implementable as a small incremental change on top of today’s bit-level `CircuitState`.

We therefore commit to a representation shift analogous to the S-box tower/composite-field work:

- **Path chosen: introduce a packed, word-level lowering and IR as the primary path.**
- **Semantic packing width: 64-bit words** (`BitVecType(64)`-like “lanes”).
  - “512” is a SIMD width (backend concern), not an IR semantic width.
  - Backends (SSE/AVX2/AVX-512/PTX) vectorize these 64-bit words across instances.
- The existing bit-level `CircuitState` remains supported as fallback/debug/compatibility.

This makes region boundaries measurable in **words**, enabling the same class of wins as tower decomposition: narrow/structured interfaces, tractable local optimization, and meaningful emit-time fusion without boundary materialization.

## Region formation (PR1)

### Graph model (PackedCircuitState-level)
Define dependency graph `G`:
- Nodes: PackedCircuitState value nodes (word ops / wires) used to compute next-state and outputs.
- Edges: producer → consumer data dependency.
- Sources: primary inputs + current-state values.
- Sinks: primary outputs + next-state values.

### Partition strategy (default)
**SCC-first partitioning** + caps:
1) Compute SCCs of `G` (strongly-coupled groups).
2) Build condensation DAG of SCCs.
3) Greedily merge SCCs along DAG edges until caps are hit.
4) If a merged SCC-region exceeds caps, split by a cut/window heuristic (topological windowing).

Caps (first cut):
- `max_nodes_per_region`
- `max_boundary_words` (after packing; see below)
- Optional `max_depth_estimate`

### Region interface computation
For a region `R`:
- `inputs(R)`: values used by nodes in `R` but produced outside `R` (including primaries and prior regions’ outputs).
- `outputs(R)`: values produced in `R` that are used outside `R` (including next_state and primary outputs).

### Determinism guarantees
Regioning must be deterministic:
- Stable topological ordering for nodes/regions.
- Stable tie-breaking (e.g., by node index / name).

### Proposed module/API
Add `stc/regions.py` (already started for bit-level; will be extended/ported to packed IR):
 - `build_dep_graph(packed_circuit_state) -> DepGraph`
 - `partition_into_regions(graph, caps) -> list[Region]`
 - `compute_region_interface(region, graph) -> (inputs, outputs)`

Data structures:
- `@dataclass RegionCaps(max_nodes: int, max_boundary_words: int, max_depth: int | None = None)`
- `@dataclass Region(id: int, nodes: list[int], inputs: list[SignalRef], outputs: list[SignalRef])`

## Interface management (make-or-break)

### Packing policy
Boundaries are defined in terms of **64-bit semantic words**:
- Word-level lowering should expose natural word signals (e.g., 64-bit lanes in crypto cores, counters, address/state words).
- When inputs/state are still bit-like, lowering must pack them into a stable word layout (word-major), rather than creating one node per bit.
- Maintain a stable `PackedBoundaryLayout` so region compilation/emit agree on order.

Proposed structures:
- `@dataclass PackedBoundaryLayout(words: list[PackedWord])`
- `@dataclass PackingMap(signal_to_slot: dict[SignalRef, (word_idx, bit_off)])`

### Spill policy (register-carry first)
Emitter should:
- Carry up to `K` packed boundary words in registers between regions.
- Spill overflow to a compact `packed_scratch[]` array (of `__m512i` / words), not per-bit arrays.

### Region ordering to reduce peak live boundary
Default: topological order of region DAG.
Optional: heuristic selection among ready regions to minimize live boundary size (simple “min live-out” heuristic is fine initially).

## PackedCircuitState (PR2)

Introduce a new lowered IR for word-level computation:
- `PackedCircuitState` where values are 64-bit semantic words and gates are wordwise ops (`xor/and/not/ternary`, plus minimal word-level helpers like rotates/slices if needed by lowering).
- Make `TickIR -> PackedCircuitState` the **primary lowering** for sequential designs.
- Keep bit-level `CircuitState` as fallback/debug/compat.

This is the representation shift that makes decomposition effective in the “tower” sense: interfaces become meaningful algebraic words, nonlinear work isolates better, and linear work is compressible.

## Region compilation (PR3)

### Compilation contract
Compile each region as a “mini circuit”:
Input:
- region subgraph nodes
- region interface layout (packed)
Output:
- scheduled instruction list (or other backend-ready representation)

Pipeline per region (reuse existing components):
1) Extract region subgraph → mini `CircuitState` (or the structure the scheduler consumes).
2) Local CSE/simplification/DCE (reuse existing reduction passes where available).
3) Optional tech mapping (ternary/LUT/etc.) inside region.
4) Schedule + RA inside region with bounded budgets.
5) Emit a region kernel with ABI based on packed boundary layout.

Proposed module/API: `stc/region_compile.py`
- `compile_regions(circuit_state, regions, *, backend: str, caps: CompileCaps) -> list[RegionProgram]`

## Emit-time region fusion (PR4)

### Emitted shape
Emit:
- Region kernels: `region_i(in_boundary_ptr, out_boundary_ptr, state_ptrs...)`
- A fused wrapper `tick_core(...)` that runs regions in order, passing boundaries:
  - in registers where possible
  - else through `packed_scratch[]`

Then integrate with emit-time stepping:
- `tick_steps_shared(..., steps)` repeats `tick_core` with state pointer swap (as in `docs/EMIT_TIME_STEPS.md`).

### Minimal change strategy
Avoid broad rewrites:
- Keep existing scheduled emitter for a “whole tick” intact.
- Add a parallel emission path that accepts multiple region programs and emits a region runner.

## Optional local resynthesis (PR5)
Enable bounded local resynthesis inside regions:
- Extract a window (bounded by node count and cut size).
- Use existing Z3/superopt infrastructure to minimize backend cost (e.g., ternary op count).
- Only run when the window cut size is small enough to keep it tractable.

## CLI integration + benchmarks (PR6)
Add flags (names tentative):
- `--use-regions`
- `--region-max-nodes N`
- `--region-max-boundary-words W`
- `--region-pack {auto,off}`
- `--region-order {topo,heuristic}`
- `--region-resynth-window N` (optional)

Benchmarks should report:
- region count, boundary sizes (packed words)
- compile time per region
- (native) run time and spill stats (if available)

## Acceptance criteria per PR

### PR1 (region formation on current bit-level CircuitState)
- Deterministic region partitioning.
- Correct interface sets on known toy graphs.
- No backend changes.

### PR2 (PackedCircuitState + lowering)
- `TickIR -> PackedCircuitState` works on small sequential designs.
- Packed evaluator matches reference simulation on small tests.
- Deterministic packed IO/state layout.

### PR3 (region compilation driver on packed IR)
- Region-compiled tick is equivalent to non-region packed tick on small designs.
- Per-region compile stays within cap budgets.

### PR4 (emit-time fusion with packed boundaries)
- No bulk boundary materialization per region; boundaries are packed words and only spilled compactly.
- Correctness matches baseline.
- Meaningful reduction in memory traffic on large designs with “state materialization” failure mode.

### PR5 (optional resynthesis)
- Improves mapped cost on region microbenchmarks without breaking equivalence.

### PR6 (CLI + benchmarks)
- Stable CLI UX and reproducible benchmark scripts.
- Regioning is optional and off by default (but packed lowering is the default for sequential designs once mature).
