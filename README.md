# Space-Time Compiler (STC)

A compiler that turns a restricted subset of Verilog into an explicit
**space-time machine** — a sequential IR with first-class state, time, and
resources — then applies solver-guided optimization and emits deterministic
code for microcontrollers, CPUs with SIMD, and GPUs.

```
input.v ──▶ yosys ──▶ normalized.json ──▶ Tick-IR ──▶ reduced Tick-IR ──▶
                backends: ATtiny85 C · x86 SIMD C · PTX (CUDA) · Futhark
```

The central idea (see `info.md` and `design.md`): every design is modeled as a
discrete state transition

```
S' = f(S, I)    next state
O  = g(S, I)    outputs
```

where `S` is explicit state (registers), `I` are inputs, `O` are outputs, and
the tick (clock) is a real semantic unit. Because the IR is equivalent to a
sequential circuit, classic EDA techniques apply: boolean simplification,
state reduction, bounded equivalence checking, and constraint-guided
synthesis (Z3).

## What's inside

- **Frontend** — restricted Verilog subset elaborated with Yosys
  (`read_verilog -sv`, `proc`, `opt`, `write_json`). Single top module, single
  clock, synchronous reset.
- **Tick-IR** — the source-of-truth IR (`stc/tick_ir.py`), with types
  `bool`, `bitvec[N]`, `float[32/64]` (IEEE-754, RNE), and
  `simd[lane_width, lanes]` (packed, lane-wise, no cross-lane carry).
  Serialized as JSON or a compact binary format.
- **Optimization** — combinational reduction, bounded dead-state removal,
  structural hash-consing, Z3-backed autovectorization, Z3-guided
  superoptimization, ternary-logic mapping (`lop3.b32` / `vpternlogd`),
  constant-time tick fusion, and packed word-level lowering with automatic
  path selection.
- **Backends**
  - `avr` — branchless C for ATtiny85 (GPIO mapping, tick loop).
  - x86 SIMD — SSE2/SSE4.1/AVX/AVX2/AVX-512/AVX-512VL C with intrinsics;
    auto-selection by CPU features, or explicit `--backend x86-avx2` /
    `x86-avx512`.
  - `ptx` — NVIDIA PTX kernels (sm_61), one thread per instance; fused
    multi-tick execution for sequential designs.
  - `futhark` — Futhark source with batch/step entry points.
  - Vulkan SPIR-V codegen exists as an experimental path.
- **Validation** — a Tick-IR interpreter, Verilator golden traces, and
  bounded Z3 equivalence checks.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

Dependencies: `z3-solver`, `msgpack` (runtime); `black` (dev). External
tools used when available: `yosys` (required for Verilog input), `verilator`
(golden simulation), a C compiler (native SIMD tests), `ptxas`/CUDA (PTX
tests), `futhark` (Futhark checks).

## Usage

Compile a Verilog design to AVR C:

```bash
.venv/bin/python -m stc fixtures/verilog/combinational_not.v --out out
```

Compile to AVX-512 C with SIMD inference and autovectorization:

```bash
.venv/bin/python -m stc fixtures/verilog/simd_lane_add_slices.v \
  --out out --infer-simd --autovec --backend x86-avx512
```

Compile a sequential design to PTX and execute 11 fused ticks on a GPU:

```bash
.venv/bin/python scripts/bench_aes_gpu_tick.py \
  --mode fused --n 262144 --ticks 11 --reps 20
```

Emit a full ATtiny85 project:

```bash
.venv/bin/python -m stc input.v --out out --avr-project
```

Key flags: `--backend {generic,avr,ptx,x86-avx2,x86-avx512,futhark}`,
`--infer-simd`, `--autovec`, `--superopt`, `--fuse-ticks N`,
`--use-regions`, `--autotune`, `--ternary-mapping`, `--bound N`.

Run `python -m stc --help` for the full list.

## Development

```bash
./scripts/fmt.sh                          # Black formatting
./scripts/test.sh                         # format + full test suite
.venv/bin/python -m unittest discover -s tests   # tests only
```

The test suite is ~1000 tests and runs in about two minutes with the toolchain
installed. Tests that need external tools are marked `*_optional.py` and skip
cleanly when the tool is missing.

## Repository layout

- `stc/` — compiler core (IR, extraction, optimization, lowering, backends)
- `scripts/` — developer utilities, benchmarks, and experiment runners
- `tests/` — `unittest` suite
- `fixtures/` — Verilog and Tick-IR inputs
- `docs/` — design notes, backend contracts, and archived plans
- `bench/` — performance comparisons (e.g., Rust CPU AES)
- `external-*/` — vendored reference projects used as inputs for experiments

## Documentation

- `ARCHITECTURE.md` — pipeline and module overview
- `design.md` — Tick-IR semantics and the MVP specification
- `docs/X86_SIMD_ABI.md` — x86 SIMD C ABI (word layout, masks)
- `CONTRIBUTING.md` — how to build, test, and contribute

## License

GNU Affero General Public License v3 or later — see `LICENSE`.
