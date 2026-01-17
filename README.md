# Space-Time Compiler (MVP)

This repo contains a Python MVP for a space-time compiler that converts a restricted HDL design into Tick-IR, applies conservative reductions, and emits deterministic C for an ATtiny85-style tick loop.

## Setup

Create a virtual environment and install dev tools:

`python3 -m venv .venv`

`.venv/bin/pip install -r requirements-dev.txt`

## Tooling

- Verilog input (`.v`) requires `yosys` to be installed and available on `PATH`.
- If `yosys` is installed but not on `PATH`, set `STC_YOSYS` to the full path to the binary.
- Verilator-based golden simulation requires `verilator` to be installed and available on `PATH`.
- Optional bounded equivalence requires `yosys-smtbmc` and `z3`. Installing `z3-solver` in `.venv` provides `.venv/bin/z3`.

## Formatting

`./scripts/fmt.sh`

## Tests

`\.venv/bin/python -m unittest discover -s tests`

## Run the pipeline

From an existing `normalized.json`:

`\.venv/bin/python -m stc path/to/normalized.json --out out`

From a Verilog file:

`\.venv/bin/python -m stc path/to/input.v --out out`

The Verilog path requires `yosys` to be installed and on `PATH` (or `STC_YOSYS` set).

### Optional SIMD inference + autovectorization

- `--infer-simd` upgrades eligible packed bit-vectors into `simd[lane_width, lanes]` types.
- `--autovec` runs a solver-validated autovectorization pass during reduction.
- `--superopt` runs a bounded Z3-guided expression superoptimizer during reduction.
- `--no-backend` skips emitting `avr.c` (required when SIMD types are present).

### AVR artifacts

- `io_map.json` is emitted for non-SIMD designs and defines the PORTB bit layout used by the AVR backend.
- `--io-map path/to/io_map.json` overrides the default packing.
- `--avr-project` also emits `main.c` and `Makefile` (expects `avr.c` in the same folder).

### Host simulation (no AVR toolchain)

Compile the generated `avr.c` for the host using `STC_HOST`:

`cc -std=c99 -O2 -DSTC_HOST -o host_runner host_main.c out/avr.c`

## AES throughput (GPU Tick/PTX vs CPU Rust)

**GPU (host)**: runs AES-128 (fixed key) compiled from `fixtures/verilog/aes128_fixedkey_seq_lut.v` to Tick-IR, then to PTX, then executes 11 ticks on the GPU (single kernel launch via `steps=11`).

`\.venv/bin/python scripts/bench_aes_gpu_tick.py --mode fused --n 262144 --ticks 11 --reps 20`

**GPU (Docker)**: runs the same benchmark inside `nvidia/cuda` with manual `/dev/nvidia*` + driver library mounts.

`N=262144 TICKS=11 ./scripts/bench_aes_gpu_tick_docker.sh`

**CPU (Rust)**: uses RustCrypto `aes` (AES-NI when available).

`cd bench/aes_cpu && cargo run --release -- --blocks 262144 --min-seconds 0.2`
