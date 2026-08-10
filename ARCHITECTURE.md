# Architecture

This document describes how the Space-Time Compiler (STC) is organized and
how a design flows through the pipeline.

## Pipeline overview

```
input.v
  │  yosys: read_verilog -sv, hierarchy -check, proc, opt, opt_clean, write_json
  ▼
normalized.json
  │  stc.extract: cells → Tick-IR equations
  ▼
tick_ir.bin
  │  stc.reduce: constant folding, boolean simplification, hash-consing,
  │  bounded dead-state removal, (optional) autovec / superopt /
  │  ternary mapping / tick fusion
  ▼
reduced_tick_ir.bin
  │  lowering: bit-level CircuitState or packed 64-bit word-level
  │  (automatic selection with fallback, see lowering_choice.bin)
  ▼
CircuitState / PackedCircuitState
  │  scheduling + backend emission
  ▼
avr.c · circuit_*.c · kernel.ptx · circuit.fut
```

## Modules

### Frontend and IR

- `stc/yosys_frontend.py`, `stc/yosys_json.py` — Yosys invocation and JSON
  netlist loading.
- `stc/subset.py` — enforces the accepted Verilog subset at the JSON boundary
  with explicit errors.
- `stc/extract.py` — maps Yosys cells to Tick-IR equations
  (`S' = f(S, I)`, `O = g(S, I)`).
- `stc/tick_ir.py` — the IR data model: types (`bool`, `bitvec[N]`,
  `float[32/64]`, `simd[lane_width, lanes]`), expressions, and the TickIR
  container.
- `stc/tick_ir_bin2.py` — compact binary serialization.
- `stc/tick_ir_validate.py` — IR integrity checks (types, widths, acyclicity).

### Optimization

- `stc/reduce.py` — combinational reduction, constant folding, structural
  hash-consing, and the pass pipeline.
- `stc/hashcons.py` — structural deduplication of identical subexpressions.
- `stc/interp.py` — Tick-IR interpreter (all expression types).
- `stc/z3_encode.py`, `stc/z3_prove.py`, `stc/bounded_equiv.py` — Z3
  bitvector/float encoding and bounded equivalence checking.
- `stc/infer_simd.py`, `stc/autovec.py`, `stc/autovec_pass.py` — SIMD type
  inference and Z3-validated autovectorization.
- `stc/superopt.py` — Z3-guided expression superoptimization with a cost model.
- `stc/fuse_ticks.py` — temporal fusion/unrolling of sequential ticks.
- `stc/gate_ternary_synth.py`, `stc/mapping/` — ternary logic mapping
  (`lop3` / `vpternlog`), including a precomputed 4×4 S-box database.
- `stc/circuit_synth.py`, `stc/window_synth.py`, `stc/linear_opt.py` —
  boolean circuit synthesis and SAT/Z3 window resynthesis (used for S-box
  and circuit optimization research).
- `stc/passmgr.py`, `stc/tech.py`, `stc/anytime.py` — extensible pass
  manager, backend technology/cost/depth models, anytime optimization.

### Lowering

- `stc/tick_ir_to_circuit_state.py` — bit-level lowering (one node per bit).
- `stc/tick_ir_to_packed_circuit_state.py` — packed 64-bit word-level lowering
  for wide designs (Keccak-class).
- `stc/lowering_coordinator.py` — tries packed first, falls back to bit-level,
  writes `lowering_choice.bin` with the reason.
- `stc/regions.py`, `stc/packed_regions.py`, `stc/packed_region_emit.py` —
  region decomposition and region-fused emission.
- `stc/autotune.py` — reproducible configuration search.

### Backends

- `stc/backend_avr.py` — ATtiny85 C (compute/commit phases, GPIO mapping).
- `stc/backend_x86_auto.py` — CPU-feature-driven selection among:
  `backend_x86_sse2.py`, `backend_x86_avx.py`, `backend_x86_avx2.py`,
  `backend_x86_avx512.py`, `backend_x86_avx512_float.py`,
  `backend_x86_avx512vl.py`.
- `stc/backend_ptx.py`, `stc/ptx_lop3.py`, `stc/cuda_driver.py` — PTX
  emission (sm_61), lop3 mapping, and CUDA driver-API runtime.
- `stc/backend_futhark.py` — Futhark emission with batch/step entry points.
- `stc/backend_vulkan.py` — experimental SPIR-V path.
- `stc/sched/` — list/pipelined scheduling, register pressure analysis,
  target models, and per-target emitters.
- `stc/backend_sched.py` — scheduled backend orchestration (AVX-512/PTX).

## Execution models

- **Tick-per-invocation**: each call executes one tick; state is committed
  atomically (AVR, most SIMD paths).
- **Fused steps**: `circuit_steps_shared` runs `N` ticks in one call using
  double-buffered state pointers (AVX-512 scheduled backend).
- **Instance-parallel**: one GPU thread per design instance; `steps=k`
  executes `k` ticks in a single kernel launch (PTX).

## Conventions

- State is committed simultaneously at tick end; reads within a tick observe
  only the incoming state.
- SIMD arithmetic is lane-wise with modular wrap and no cross-lane carry;
  backends preserve full-vector semantics even across ISA lane-group
  boundaries.
- Float ops use IEEE-754 semantics with RNE rounding and canonicalized NaN
  payloads.
- All optimizations are Z3-verified for equivalence (bounded) before being
  applied.
