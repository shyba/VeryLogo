# Plan: Better Compilation for Large Stateful Designs (Generic, Keccak as Stress-Test)

This plan targets “Keccak-like” large stateful Verilog cores but is **generic**: it improves the compilation pipeline for *any* design lowered to `TickIR` and then compiled to fast native code (x86 AVX-512 first, but extensible to AVX2/PTX).

**Primary goal:** produce fast code while staying debuggable and deterministic.  
**Secondary goal:** make optimization choices auto-tuned by default, yet easy to configure and troubleshoot.

---

## 0) Current Known Failure Modes (to address)

1) **Packed (64-bit semantic word) lowering rejects real designs**
   - Current packed lowering is conservative (`stc/tick_ir_to_packed_circuit_state.py`) and will throw `PackedLoweringError` for patterns common in real cores.
2) **Bit-level fallback is functional but memory-heavy**
   - Bit-level lowering (`lower_tick_ir_to_circuit_state`) creates one node per bit; even with emit-time stepping it tends to be bandwidth-bound for large state.
3) **“All-or-nothing” optimization**
   - When a pass can’t handle an op/formation, we fall back globally. We need partial lowering and structured fallback.

Keccak is a good stress-test because it combines wide state, mux/control cones, rotates/shifts, and word-level structure.

---

## 1) Deliverables and Success Criteria

### Correctness
- Cross-check against a reference implementation:
  - `openssl dgst -keccak-512` for Keccak-512.
  - For generic designs: Python reference evaluators (TickIR, PackedCircuitState, CircuitState).
- Deterministic outputs: same input + flags → identical emitted code (modulo timestamps).

### Performance
- For Keccak-512 “digest-per-call” benchmark, target a consistent improvement vs bit-level baseline.
- For generic designs, target reduced memory traffic (fewer loads/stores/spills) and improved throughput.

### Usability
- A pipeline that can be run in a “single command” mode.
- Artifact outputs that make it obvious which path was used and why.
- Auto-tuning that is reproducible and configurable.

---

## 2) High-Level Architecture (multi-phase, generic)

### Phase A — Normalize `TickIR` into “word-friendly” form (generic)
Goal: transform the extracted `TickIR` into a subset that the packed lowering can handle.

Key additions (new module recommended: `stc/tick_ir_normalize.py`):
1) **Canonicalize bitvector ops**
   - Normalize `Concat`/`Slice` patterns.
   - Pull masks into canonical `And` with constants.
2) **Recognize and normalize rotates**
   - Detect rotation idioms, lower into a dedicated `Rotl/Rotr` node (or canonical shift/or pair).
   - Keccak relies heavily on rotates; this should be generic for any design.
3) **Normalize shifts**
   - Ensure shift amounts are constant where possible (propagate constants).
   - Split “big shifts” into word-shift + intra-word shift.
4) **Mux normalization**
   - If `$pmux`/wide mux lowered into nested muxes, canonicalize mux trees into a uniform form.

Artifacts:
- Write `out/normalized_tick_ir.json` plus a `out/normalize_report.json` explaining which rewrites triggered.

### Phase B — Packed lowering becomes the default for x86-avx512 (generic)
Goal: lower to **64-bit semantic word** IR (`PackedCircuitState`) whenever possible.

Improvements to packed lowering (`stc/tick_ir_to_packed_circuit_state.py`):
1) **Support `Concat`/`Slice` beyond the current conservative subset**
   - Implement unaligned `Slice` > 64 bits using a word-by-word gather (still pure word ops).
   - Support mixed-width `Concat` by emitting packing shifts/or across words.
2) **Support rotate nodes (from Phase A)**
   - Implement rotate lowering as `(x << k) | (x >> (64-k))` with word-shift carry.
3) **Partial lowering + fallback**
   - If a sub-expression is unsupported, fall back **locally**:
     - Option 1: keep a small “bit island” lowered with bit-level CircuitState and bridge to words via pack/unpack.
     - Option 2 (simpler first): fall back the entire design but emit a detailed reason report.

Artifacts:
- Always write `out/lowering_choice.json`:
  - `{"path":"packed"|"bit", "reason":"...", "unsupported":[...], "stats":{...}}`
- If packed succeeds: `out/packed_circuit_state.json` + `out/packed_word_layout.json`.

### Phase C — Region decomposition on packed IR (generic)
Goal: reduce scheduling/register pressure and enable local optimization by compiling regions.

Use `stc/packed_regions.py` + `stc/packed_region_emit.py`:
- Default region caps:
  - `--region-max-gates` (node cap)
  - `--region-max-boundary` (boundary node cap)
- Add region diagnostics:
  - region count, boundary distribution, critical-path estimates
  - optional “region graph” dump (DOT/JSON) for debugging

Artifacts:
- `out/regions.json` (list of regions, interfaces, sizes)
- `out/regions_stats.json` (histograms and summary)

### Phase D — Backend codegen improvements (packed AVX-512 u64 first)
Goal: make emitted code fast and predictable.

1) **Tighter emitted ABI**
   - Keep current split ABI:
     - `circuit__core(in_io, st_in, out_io, st_out)`
     - `circuit_steps_shared(in_io, out_io, state_in, state_out, steps)`
   - Add `*_steps_replicate` in the future (inputs vary per step) without changing the core ABI.
2) **Avoid pathological temporaries**
   - In region emission, keep region temps local and bounded.
   - Optional: add a small temp allocator / reuse pool inside each region emitter.
3) **Autotunable region caps**
   - Compile multiple configurations (caps, ordering) and select best by a fast proxy metric:
     - compile-time cost (node count) + predicted live set size
     - optional microbenchmark if `--autotune` enabled

Artifacts:
- `out/codegen_report.json` with estimated register pressure, region metrics, and chosen config.

### Phase E — Autotuning (fast, reproducible, easy to configure)
Goal: “auto by default” but deterministic and controllable.

Introduce `--autotune` mode with:
- `--autotune-seed`
- `--autotune-budget-ms`
- `--autotune-candidates` (small list of region caps / ordering strategies)

Autotune outputs:
- `out/autotune_results.json` (candidate configs + measured/estimated scores)
- `out/autotune_choice.json` (selected config)

Default behavior when autotune is off:
- Use a safe preset (`--use-regions` on for packed AVX-512 if packed lowering succeeds).

---

## 3) Troubleshooting & Debuggability (must be first-class)

### “Explain why we fell back”
If packed lowering fails, provide:
- exact unsupported `TickIR` node kinds and types
- minimal path to reproduce (var name + subexpr hash)
- suggestion for which normalization/lowering feature is missing

### “Reduce to a smaller repro”
Add a helper script (recommended):
- `scripts/minimize_lowering_failure.py`
  - takes `reduced_tick_ir.json`
  - isolates the smallest output/next-state cone that still triggers `PackedLoweringError`
  - optionally emits a small `.v` or `.json` fixture for regression tests

### Regression tests
Add fixtures for:
- rotate patterns, wide concat/slice, mux cones, and mixed-width interfaces
- a small keccak-derived subcone (not the full core) once minimized

---

## 4) Concrete Next PRs (multi-phase execution)

### PR1: Normalization + better packed lowering coverage
- Add `stc/tick_ir_normalize.py` and run it before lowering.
- Extend packed lowering to handle:
  - unaligned slices > 64 bits
  - mixed concat packing
  - rotate (either explicit op or recognized pattern)
- Add targeted unit tests in `tests/` for each new construct.

### PR2: “Lowering choice report” + minimizer tooling
- Implement `out/lowering_choice.json` and detailed `PackedLoweringError` reporting.
- Add `scripts/minimize_lowering_failure.py` producing reproducible fixtures.

### PR3: Region pipeline hardening + diagnostics
- Emit `out/regions.json`, `out/regions_stats.json`.
- Add region ordering heuristic options and deterministic tie-breaking.
- Add an optional native correctness test that compiles the emitted region code and compares to `eval_packed_circuit_words`.

### PR4: Autotune (proxy metric first, optional microbench)
- Add `--autotune` and implement:
  - candidate enumeration
  - fast scoring (boundary size + rough liveness estimate)
  - optional microbench using `ctypes` stepping if enabled

### PR5: Keccak benchmark harness integration (but generic)
- Provide a generic benchmark runner that:
  - runs the pipeline
  - compiles emitted C
  - runs `*_steps_shared` for a fixed workload
  - prints throughput and stores results JSON in `out/bench.json`
- Keep Keccak as one benchmark case, but make the harness accept arbitrary `TickIR`/Verilog.

---

## 5) How to Run and Benchmark

### Generate Keccak flattened JSON (requires `yosys`)
Use the same pattern as `scripts/bench_keccak_steps_avx512.py`:
```bash
rtl=external-sha3-verilog/low_throughput_core/rtl
yosys -q -p "read_verilog ${rtl}/*.v; hierarchy -top keccak; proc; flatten; opt; opt_clean; write_json out/keccak_flat.json"
```

### Compile with VeryLogo (packed-first, fallback to bit-level)
```bash
PYTHONPATH=. .venv/bin/python -m stc out/keccak_flat.json --top keccak --out out/keccak_out --backend x86-avx512 --bound 8 --use-regions
```

Expected artifacts:
- Always: `out/keccak_out/reduced_tick_ir.json`
- If packed succeeded: `out/keccak_out/packed_circuit_state.json`, `out/keccak_out/packed_word_layout.json`, and either:
  - `out/keccak_out/circuit_avx512_u64.c` (scheduled) or
  - `out/keccak_out/circuit_avx512_u64_regions.c` (region-fused) when `--use-regions`
- If packed failed: bit-level artifacts `out/keccak_out/circuit_avx512.c`, `out/keccak_out/io_layout.json`

### Benchmark Keccak (current bit-level baseline)
```bash
PYTHONPATH=. .venv/bin/python scripts/bench_keccak_steps_avx512.py --msg abc --reps 50 --idle-chunk 128
```

### Benchmark packed region emission (generic runner)
```bash
PYTHONPATH=. .venv/bin/python scripts/bench_packed_avx512_u64.py --tick-ir out/keccak_out/reduced_tick_ir.json --steps 1024 --iters 200 --check
```

---

## 6) Public API expectations for new features

### Stable user-facing flags (proposed)
- `--backend x86-avx512`: uses packed lowering by default, falls back to bit-level with an explanation report.
- `--use-regions`: enable region decomposition for the packed path.
- `--region-max-gates`, `--region-max-boundary`: caps (defaults chosen to be safe).
- `--autotune`: run a reproducible config search.
- `--autotune-budget-ms`, `--autotune-seed`, `--autotune-candidates`: tuning control.
- `--dump-regions`, `--dump-normalize`, `--dump-lowering-report`: troubleshooting outputs.

### Programmatic entrypoints (proposed)
Add a high-level “compile” API that returns a structured result:
```python
CompileResult = {
  "path": "packed"|"bit",
  "tick_ir": TickIR,
  "artifacts": {...paths...},
  "layout": ...,
  "code": "...",
}
```
so scripts can consume it without re-implementing pipeline logic.

---

## 7) Immediate action items for Keccak specifically (but implemented generically)

1) Make packed lowering accept the constructs Keccak uses most:
   - rotates and wide slices/concats
2) Produce a minimizable repro when packed lowering fails:
   - smallest cone that still triggers failure
3) Once packed lowering succeeds, compare:
   - packed scheduled (`circuit_avx512_u64.c`) vs packed regions (`*_regions.c`)
   - region cap autotuning vs fixed caps

When these are in place, Keccak becomes a continuously useful benchmark rather than a one-off integration.  

