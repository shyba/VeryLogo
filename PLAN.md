# Implementation Plan (MVP)

This plan follows TDD and keeps the repository in a continuously working state. Every step is implemented behind tests, and `python3 -m unittest discover -s tests` remains green at all times.

## Global Rules

- All new documentation is written in English.
- Development is test-driven: write a failing test, implement the smallest change to pass, refactor with tests staying green.
- No code comments are added.
- Code is formatted in Black style. The repository includes Black configuration even if Black is not installed in the environment.

## Phase 0 — Repo and Tooling Baseline

Goal: establish a minimal, stable Python project layout with a deterministic test runner.

Deliverables:

- `pyproject.toml` with Black configuration.
- `stc/` Python package skeleton.
- `tests/` using `unittest` from the standard library.
- A single command to run tests: `python3 -m unittest discover -s tests`.
- A project virtual environment at `.venv/`.
- A formatter entrypoint at `scripts/fmt.sh`.

Done criteria:

- Tests run successfully in a clean environment.

## Phase 1 — Tick-IR Data Model + JSON Schema

Goal: implement the Tick-IR as the source of truth, with a stable JSON encoding.

Deliverables:

- Tick-IR types: `bool`, `bitvec[N]`.
- Tick-IR expressions: constants, variables, and a minimal operator set required by MVP tests.
- Tick-IR containers: inputs, outputs, state regs, next-state equations, output equations.
- Round-trip JSON serialization: Tick-IR → JSON → Tick-IR.

Done criteria:

- Round-trip tests pass.
- Invalid IR inputs fail fast with explicit exceptions.

## Phase 2 — Normalized Netlist Ingestion (Yosys JSON)

Goal: ingest `normalized.json` as produced by Yosys and build an internal netlist model.

Deliverables:

- Yosys JSON loader for modules, ports, bits, and cells.
- Netlist integrity checks (unique names, supported structures, required top module).

Done criteria:

- Fixture-driven tests pass on representative Yosys JSON samples.

## Phase 3 — Frontend Subset Checking

Goal: enforce the MVP Verilog subset at the `normalized.json` boundary.

Deliverables:

- A checker that rejects unsupported constructs early.
- Clear, structured errors that identify the first unsupported construct encountered.

Done criteria:

- Tests cover both accepted and rejected cases.

## Phase 4 — Tick-IR Extraction

Goal: convert normalized netlist into Tick-IR equations `S' = f(S,I)` and `O = g(S,I)`.

Deliverables:

- Mapping from a minimal supported set of Yosys cells into Tick-IR expressions.
- Explicit identification of state elements (registers) and combinational logic.

Done criteria:

- End-to-end test: `normalized.json` fixture → `tick_ir.bin` equals expected IR.

## Phase 5 — Reduction + Metrics

Goal: implement conservative reductions and measurable metrics.

Deliverables:

- Combinational constant folding and basic boolean simplifications.
- Dead-state removal within a bounded reachability framework.
- `metrics.bin` generation before/after reduction.

Done criteria:

- Reduction produces expected simplifications on fixtures.
- Metrics are stable and test-verified.

## Phase 6 — ATtiny85 C Backend

Goal: generate deterministic C for a tick-based execution model.

Deliverables:

- C emitter that produces:
  - a pure compute phase (temporaries + next-state)
  - a commit phase (state updates)
  - GPIO read/write shims for ATtiny85
- Masking rules for non-byte-aligned bit widths.
- Branchless selection lowering when representable as masks.

Done criteria:

- Golden-file tests validate generated C output for small fixtures.

## Phase 7 — Validation Harness

Goal: validate semantics via simulation traces and bounded checks.

Deliverables:

- Python Tick-IR interpreter that produces tick traces (`S`, `O`).
- Integration hooks for Verilator-based golden traces when tooling is available.
- Optional bounded equivalence checker integration via Yosys/SymbiYosys when tooling is available.

Done criteria:

- Tick-IR interpreter tests pass on the main MVP cases (combinational, simple FSM, temporal pipeline, debounce) using IR-level fixtures.

## Tool Integration Notes

- `yosys` is required to start from `.v` input. If it is installed but not on `PATH`, set `STC_YOSYS` to the full path of the `yosys` executable.
- `verilator` is used for golden simulation and is expected on `PATH` (or via `STC_VERILATOR`).
