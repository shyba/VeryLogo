# Contributing

Thanks for your interest in STC. This project is test-driven: every change is
backed by a failing test first, and the full suite must stay green.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

Optional tools used by parts of the test suite (tests skip cleanly when
missing): `yosys`, `verilator`, a C compiler (`cc`), `ptxas`/CUDA, `futhark`.

## Development loop

1. Write a failing test (`tests/test_*.py`).
2. Implement the smallest change to make it pass.
3. Refactor with tests staying green.

```bash
./scripts/fmt.sh                        # Black, line-length 88, target py312
./scripts/test.sh                       # format + full test suite
.venv/bin/python -m unittest discover -s tests   # tests only
.venv/bin/python -m unittest tests.test_extract   # single module
```

## Guidelines

- Python is formatted with Black (line-length 88, target py312). No manual
  alignment that fights Black.
- Code is self-documenting; do not add comments.
- Prefer small, deterministic tests. Tests that need external tooling or
  specific hardware follow the existing pattern: filename suffix
  `_optional.py`, runtime skip when the dependency is unavailable.
- Only apply reductions that are provable via Z3 bounded equivalence checks.
- If you touch ternary mapping / vpternlog / lop3 codegen, run
  `tests/test_ternary_cone_mapping_regression.py`.

## Commit and PR conventions

- Commits: short, imperative subject lines ("Add …", "Fix …", "Implement
  Phase 2: …"); keep changes focused and avoid mixing refactors with behavior
  changes.
- PRs: concise summary, reproduction/validation steps
  (`./scripts/test.sh`), and any optional dependencies required.

## License

The project is licensed under the GNU Affero General Public License v3 or
later (see `LICENSE`). By contributing, you agree that your contributions
are licensed under the same terms.

## Repository layout

- `stc/` — compiler core. Keep it free of imports from `scripts/`.
- `tests/` — `unittest` suite.
- `scripts/` — thin CLI runners and benchmarks; reusable logic belongs in
  `stc/` or `integrations/`.
- `docs/` — design notes and backend contracts; `docs/archive/` holds
  historical plans.
- `external-*/` — vendored reference projects treated as inputs; avoid
  editing them.
