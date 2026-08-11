# Repository Guidelines

## Project Structure & Module Organization

- `stc/`: core Python package (CLI entrypoints in `stc/__main__.py` and `stc/cli.py`).
- `tests/`: `unittest`-based test suite. Many tests are marked “optional” by filename and may depend on external tooling or specific hardware.
- `scripts/`: developer utilities (formatting, test runner, benchmarks, helpers).
- `external-*/`: vendored/checked-out reference projects used for experiments (e.g., bitsliced AES, SHA3 cores). Treat as inputs; avoid editing unless explicitly needed.
- `fixtures/`: example inputs (e.g., Verilog used by tests/benchmarks).
- `docs/` and `*.md`: design notes, plans, and research writeups.
- `bench/`: performance comparisons (includes Rust CPU AES benchmark).
- `out/`: generated artifacts (treat as build output; don’t hand-edit).

## Build, Test, and Development Commands

Set up the dev environment:

- `python3 -m venv .venv`
- `.venv/bin/pip install -r requirements-dev.txt`

Common tasks:

- `./scripts/fmt.sh`: format Python code (Black).
- `./scripts/test.sh`: format, then run the full Python test suite.
- `.venv/bin/python -m unittest discover -s tests`: run tests only.
- `.venv/bin/python -m stc path/to/input.v --out out`: run the Verilog→Tick-IR→backend pipeline (requires `yosys` on `PATH` or `STC_YOSYS`).

Optional tooling used by some tests/flows: `verilator`, `yosys-smtbmc`, and `z3` (or `.venv/bin/z3` via `z3-solver`).

### CUDA / PTX (optional)

- `scripts/bench_lop3_cuda.py`: generates PTX `lop3.b32` kernels from S-box circuits and benchmarks via CUDA Driver API. Use `--load jit --sm sm_61` for “sm_61+” portability and `--save-ptx out/foo.ptx` to export PTX.
- `tests/test_cuda_lop3_optional.py`: optional CUDA correctness smoke test (skips if `nvcc`/`ptxas`/GPU are missing).

## Coding Style & Naming Conventions

- Python: format with Black (`line-length = 88`, target `py312`).
- Indentation: 4 spaces; avoid manual alignment that fights Black.
- Naming: modules/functions use `snake_case`; tests use `test_*.py` and `test_*` methods.

## Testing Guidelines

- Framework: `unittest` (`python -m unittest discover -s tests`).
- Prefer small, deterministic tests; if a test needs external dependencies, follow the existing pattern of marking it “optional” in the filename (e.g., `*_optional.py`) and skipping at runtime when unavailable.
- Regression note: ternary cone mapping has strict `imm8`/leaf semantics; use the small regression in `tests/test_ternary_cone_mapping_regression.py` when touching ternary mapping / vpternlog/lop3 codegen.

## Commit & Pull Request Guidelines

- Commits: short, imperative subject lines (e.g., “Implement Phase 2: …”, “Add …”, “Fix …”); keep messages focused and avoid mixing refactors with behavior changes.
- PRs: include a concise summary, how to reproduce/validate (`./scripts/test.sh`), and note any optional dependencies required; attach artifacts or snippets (e.g., generated `out/` diffs) only when they help reviewers.

## HDL Import Notes

- Yosys JSON extraction supports a restricted cell subset. `$pmux` is supported by lowering to cascaded muxes during extraction; `$scopeinfo` may appear in flattened netlists.
- Some SystemVerilog features (e.g., packages/`import pkg::...`) may not be directly ingestible by the current Yosys frontend without preprocessing (e.g., `sv2v`).
